import { app, BrowserWindow, ipcMain } from 'electron'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { freemem } from 'node:os'
import type { DatabaseSync } from 'node:sqlite'
import { join } from 'node:path'
import { scanSystem } from '../core/system/scanner'
import { detectRuntimes } from '../core/runtimes'
import { LlamaCppBackend, killStaleServer } from '../core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../core/runtimes/llamacpp/parse'
import { openDb } from '../core/storage/db'
import { getSession, getSessionRequest, latestRecommendation, listSessions, makeSessionStorage, seedDemoSession, type PlanFor } from '../core/storage/sessions'
import { WORKLOADS } from '../core/scoring/workloads'
import { val } from '../core/scoring/cliff'
import { runSession, type SessionStorage } from '../core/benchmark/session'
import { DEFAULT_CANDIDATE_RULES, generateCandidates, machineFromProfile } from '../core/benchmark/candidates'
import { findGgufModels, toModelMeta } from '../core/models/gguf'
import { startSampler } from '../core/telemetry/sampler'
import { evaluateAsync } from '../core/quality'
import type { SessionEvent, SessionRequest } from '../shared/bench-events'
import type { ModelMeta, WorkloadId } from '../shared/bench-types'
import type { AppSettings, SmokeResult, StartResult, SystemProfile } from '../shared/types'

// ponytail: app.getAppPath() is the project root in dev/preview; revisit for packaged builds.
const llamaDir = () => join(app.getAppPath(), 'vendor', 'llama.cpp')

// Large models live on D: (user decision); used when settings.json has no modelDirs key.
const DEFAULT_MODEL_DIRS = ['D:\\llm-models']

const settingsFile = () => join(app.getPath('userData'), 'settings.json')
function readSettings(): AppSettings {
  return existsSync(settingsFile()) ? (JSON.parse(readFileSync(settingsFile(), 'utf8')) as AppSettings) : {}
}
function writeSettings(patch: Partial<AppSettings>): AppSettings {
  const next = { ...readSettings(), ...patch }
  writeFileSync(settingsFile(), JSON.stringify(next, null, 2))
  return next
}

/** <project>/models plus settings.modelDirs (default D:\llm-models). Missing dirs are skipped. */
function modelDirs(): string[] {
  const s = readSettings()
  const extra = Array.isArray(s.modelDirs) ? s.modelDirs.filter((d): d is string => typeof d === 'string') : DEFAULT_MODEL_DIRS
  return [join(app.getAppPath(), 'models'), ...extra]
}

const llama = new LlamaCppBackend(llamaDir(), { pidFile: join(app.getPath('userData'), 'llama-server.pid') })
let smokeBusy = false
let db: DatabaseSync | null = null
const DEMO = process.env.LAO_SEED_DEMO === '1'
const needDb = () => { if (!db) throw new Error('database not open yet'); return db }

/** Forward a runner event to every window (channel bench:event). */
export function sendBenchEvent(e: SessionEvent): void {
  for (const w of BrowserWindow.getAllWindows()) w.webContents.send('bench:event', e)
}

ipcMain.handle('runtimes:detect', () => detectRuntimes(llamaDir()))
ipcMain.handle('system:scan', async () => {
  const [profile, runtimes] = await Promise.all([scanSystem(), detectRuntimes(llamaDir())])
  return { ...profile, runtimes }
})
ipcMain.handle('models:list', () => llama.enumerateModels(modelDirs()))
ipcMain.handle('settings:get', () => readSettings())
ipcMain.handle('settings:setWorkload', (_e, w: WorkloadId) => {
  if (!(w in WORKLOADS)) throw new Error(`unknown workload ${String(w)}`)
  return writeSettings({ workload: w })
})
ipcMain.handle('workloads:list', () => Object.values(WORKLOADS))
ipcMain.handle('sessions:list', () => listSessions(needDb()))
ipcMain.handle('sessions:get', (_e, id: number) => getSession(needDb(), Number(id)))
ipcMain.handle('recommendation:latest', (_e, w: WorkloadId) => latestRecommendation(needDb(), w))
ipcMain.handle('bench:start', (_e, req: SessionRequest) => startSession(req))
ipcMain.handle('bench:resume', (_e, id: number) => {
  const req = getSessionRequest(needDb(), Number(id))
  return req ? startSession({ ...req, resumeSessionId: String(id) }) : { ok: false, error: `session ${id} has no stored request` }
})
ipcMain.handle('bench:cancel', () => {
  if (!active) return { ok: false, error: 'no benchmark running' }
  active.abort()
  return { ok: true }
})
ipcMain.handle('bench:smoke', async (_e, modelPath: string): Promise<SmokeResult> => {
  if (typeof modelPath !== 'string' || !modelDirs().some((d) => modelPath.startsWith(d))) throw new Error('model path not in a configured model dir')
  if (smokeBusy || active) throw new Error('a smoke run or benchmark is already in progress')
  smokeBusy = true
  try {
    // Deliberately tiny: ctx 2048, 32 tokens, one warmup + one measured request.
    const dev = pickDiscreteDevice(await llama.listDevices())
    if (!dev) throw new Error('no discrete Vulkan device reported by llama-server --list-devices')
    const load = await llama.loadModel({ modelPath, contextSize: 2048, gpuLayers: 99, device: dev.id })
    const text = 'Explain in one sentence what a GPU does.'
    await llama.warmup(text)
    const prompt = await llama.runPrompt({ prompt: text, maxTokens: 32, timeoutMs: 60_000 })
    return { load, prompt }
  } finally {
    await llama.unloadModel()
    smokeBusy = false
  }
})

// ---- Benchmark session (one at a time, in the main process) ----
let active: AbortController | null = null
let profileCache: SystemProfile | null = null

async function startSession(req: SessionRequest): Promise<StartResult> {
  if (active || smokeBusy) return { ok: false, error: 'a benchmark or smoke run is already in progress' }
  if (!req || !(req.workload in WORKLOADS) || !Array.isArray(req.modelIds) || !req.modelIds.length) return { ok: false, error: 'pick a workload and at least one model' }
  const dirs = modelDirs()
  if (!req.modelIds.every((id) => typeof id === 'string' && dirs.some((d) => id.startsWith(d)))) return { ok: false, error: 'model path not in a configured model dir' }
  const ctl = new AbortController()
  active = ctl // claim before the first await so a double click can't start two sessions
  try {
    profileCache ??= await scanSystem() // ponytail: static facts cached per app run; RAM is re-read live via freemem()
    const [infos, devices, runtime] = await Promise.all([findGgufModels(dirs), llama.listDevices(), llama.detect()])
    const device = pickDiscreteDevice(devices)?.id ?? null
    const models: ModelMeta[] = []
    for (const id of req.modelIds) {
      const info = infos.find((m) => m.path === id)
      const mm = info ? toModelMeta(info) : { meta: null, reason: 'file not found' }
      if (mm.meta) models.push(mm.meta)
      else sendBenchEvent({ sessionId: '', type: 'log', level: 'warn', msg: `${id}: ${mm.reason}; skipped` })
    }
    if (!models.length) throw new Error('none of the selected models can be benchmarked')
    const backendKind = device ? 'vulkan' as const : 'cpu' as const
    const profile = profileCache
    // Same deterministic candidate generation the runner does, so stored sessions carry full configs.
    const planFor: PlanFor = (r) => {
      const machine = machineFromProfile(profile, device)
      const rules = { ...DEFAULT_CANDIDATE_RULES, ...r.candidateRules }
      return {
        vramBytes: val(machine.vramBytes, true),
        candidates: models.flatMap((model) => generateCandidates(machine, model, { backend: backendKind }, WORKLOADS[r.workload], rules).candidates.map((config) => ({ config, model })))
      }
    }
    let gotId: (id: string) => void = () => {}
    const idReady = new Promise<string>((r) => { gotId = r })
    const base = makeSessionStorage(needDb(), planFor)
    const storage: SessionStorage = { ...base, createSession: async (s) => { const id = await base.createSession(s); gotId(id); return id } }
    if (req.resumeSessionId) gotId(req.resumeSessionId)
    const run = runSession(req, {
      backend: () => llama, startSampler: (pid) => startSampler({ pid }), storage, machine: profile, gpuDevice: device, backendKind,
      models, clock: Date, readRamAvailableBytes: () => freemem(), evaluate: evaluateAsync, signal: ctl.signal,
      runtimeVersion: runtime.version ?? null
    }, sendBenchEvent).finally(() => { if (active === ctl) active = null })
    const id = await Promise.race([idReady, run.then(() => null)])
    return id ? { ok: true, sessionId: id } : { ok: false, error: 'session ended before it started (see event log)' }
  } catch (e) {
    if (active === ctl) active = null
    return { ok: false, error: (e as Error).message }
  }
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: '#0f1115',
    title: 'Local AI Optimizer',
    webPreferences: { preload: join(__dirname, '../preload/index.js'), sandbox: true, contextIsolation: true }
  })
  if (process.env.ELECTRON_RENDERER_URL) win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else win.loadFile(join(__dirname, '../renderer/index.html'))
}

app.whenReady().then(async () => {
  // LAO_SEED_DEMO=1 uses a separate DB file so fixture data can never reach the real one.
  db = openDb(join(app.getPath('userData'), DEMO ? 'optimizer-demo.db' : 'optimizer.db'))
  if (DEMO) seedDemoSession(db, join(app.getAppPath(), 'tests', 'fixtures', 'scoring'))
  const stale = await killStaleServer(join(app.getPath('userData'), 'llama-server.pid')).catch((e: Error) => `stale-server check failed: ${e.message}`)
  if (stale) console.warn(stale)
  createWindow()
})
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => { active?.abort(); llama.killSync() })
process.on('exit', () => llama.killSync())
