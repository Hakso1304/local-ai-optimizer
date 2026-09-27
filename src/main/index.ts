import { app, BrowserWindow, ipcMain } from 'electron'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import type { DatabaseSync } from 'node:sqlite'
import { join } from 'node:path'
import { scanSystem } from '../core/system/scanner'
import { detectRuntimes } from '../core/runtimes'
import { LlamaCppBackend, killStaleServer } from '../core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../core/runtimes/llamacpp/parse'
import { openDb } from '../core/storage/db'
import { getSession, latestRecommendation, listSessions, seedDemoSession } from '../core/storage/sessions'
import { WORKLOADS } from '../core/scoring/workloads'
import type { SessionEvent, SessionRequest } from '../shared/bench-events'
import type { WorkloadId } from '../shared/bench-types'
import type { AppSettings, SmokeResult } from '../shared/types'

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
// TODO(next task): wire #1's session runner; it reports progress through sendBenchEvent().
ipcMain.handle('bench:start', (_e, _req: SessionRequest) => ({ ok: false as const, error: 'runner not wired yet' }))
ipcMain.handle('bench:cancel', () => ({ ok: false as const, error: 'runner not wired yet' }))
ipcMain.handle('bench:smoke', async (_e, modelPath: string): Promise<SmokeResult> => {
  if (typeof modelPath !== 'string' || !modelDirs().some((d) => modelPath.startsWith(d))) throw new Error('model path not in a configured model dir')
  if (smokeBusy) throw new Error('a smoke run is already in progress')
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
app.on('before-quit', () => llama.killSync())
process.on('exit', () => llama.killSync())
