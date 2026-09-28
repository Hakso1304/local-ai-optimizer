import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { freemem } from 'node:os'
import type { DatabaseSync } from 'node:sqlite'
import { join } from 'node:path'
import { scanSystem } from '../core/system/scanner'
import { detectRuntimes } from '../core/runtimes'
import { LlamaCppBackend, killStaleServer } from '../core/runtimes/llamacpp'
import { pickBenchmarkDevice, pickDiscreteDevice } from '../core/runtimes/llamacpp/parse'
import { openDb } from '../core/storage/db'
import { sessionInputs, getSession, getSessionResume, latestRecommendation, listSessions, listVramBudget, makeSessionStorage, markInterrupted, seedDemoSession, telemetryForRun, type PlanFor } from '../core/storage/sessions'
import { REQUIRED_CTX, insideSomeRoot, isWorkloadId, rowId, sanitizeRequest, sanitizeServeConfig } from './validate'
import { modelAlias, toLoadConfig } from '../core/export/config'
import { registerHubIpc } from './hub'
import { modelSuggestions } from './suggestions'
import { WORKLOADS } from '../core/scoring/workloads'
import { val } from '../core/scoring/cliff'
import { runSession, type InstalledBackend, type SessionStorage } from '../core/benchmark/session'
import { applicableObservations, generateCandidates, machineFromProfile, planCandidates, rulesForRequest, vramBudgetKey, type PlannedBackend } from '../core/benchmark/candidates'
import { findGgufModels, toModelMeta } from '../core/models/gguf'
import { fetchGenerationConfig, readSidecar, writeSidecar } from '../core/hub/modelcard'
import { refreshSiblings } from '../core/hub/quants'
import { defaultLmStudioDirs, defaultOllamaRoot, listOllamaModels, toModelInfo as toOllamaModelInfo } from '../core/runtimes/ollama/models'
import { readVramInUse, startSampler, stopAllSamplers, withNvidia } from '../core/telemetry/sampler'
import { probeNvidiaSmi, startNvidiaSampler, type NvidiaProbe } from '../core/telemetry/nvidia'
import { evaluateAsync } from '../core/quality'
import type { SessionEvent, SessionRequest } from '../shared/bench-events'
import type { CandidateConfig, GpuBackendKind, ModelMeta, WorkloadId } from '../shared/bench-types'
import { recommendForWorkload } from '../core/scoring/recommend'
import type { AppSettings, InstalledRuntime, ComputedRecommendation, ModelFit, ModelInfo, ServeStatus, SmokeResult, StartResult, SystemProfile } from '../shared/types'

// Dev/test runs get their own userData so they never write the installed app's database or settings.
// Must run before anything calls app.getPath('userData') (the llama pid file below does).
if (!app.isPackaged) app.setPath('userData', `${app.getPath('userData')}-dev`)

// One instance per userData (portable + installed share it). A second copy would mark the first one's live session
// 'interrupted' and kill its llama-server via the pid file, so it exits before touching anything (W4b F7).
if (!app.requestSingleInstanceLock()) app.exit(0)
app.on('second-instance', () => {
  const w = BrowserWindow.getAllWindows()[0]
  if (w) { if (w.isMinimized()) w.restore(); w.focus() }
})

// Dev: runtime + sample models live in the project. Packaged: app dir is read-only (asar), so the runtime is
// downloaded on first run into userData and models come from userData/models + settings.modelDirs.
const llamaDir = () => (app.isPackaged ? join(app.getPath('userData'), 'runtime', 'llama.cpp') : join(app.getAppPath(), 'vendor', 'llama.cpp'))
/** Opt-in ROCm/HIP build, side by side with the Vulkan one (docs/HIP-BACKEND.md §2); dev: vendor/llama.cpp-hip. */
const hipDir = () => (app.isPackaged ? join(app.getPath('userData'), 'runtime', 'llama.cpp-hip') : join(app.getAppPath(), 'vendor', 'llama.cpp-hip'))
const bundledModelsDir = () => (app.isPackaged ? join(app.getPath('userData'), 'models') : join(app.getAppPath(), 'models'))

const settingsFile = () => join(app.getPath('userData'), 'settings.json')
function readSettings(): AppSettings {
  return existsSync(settingsFile()) ? (JSON.parse(readFileSync(settingsFile(), 'utf8')) as AppSettings) : {}
}
function writeSettings(patch: Partial<AppSettings>): AppSettings {
  const next = { ...readSettings(), ...patch }
  writeFileSync(settingsFile(), JSON.stringify(next, null, 2))
  return next
}

/** Writable default models dir plus settings.modelDirs (user-added; none by default). */
function modelDirs(): string[] {
  const s = readSettings()
  const extra = Array.isArray(s.modelDirs) ? s.modelDirs.filter((d): d is string => typeof d === 'string') : []
  return [bundledModelsDir(), ...extra]
}

/** Folder GGUFs (configured dirs) + LM Studio dirs + Ollama blobs, deduplicated by path (first source wins). */
async function listAllModels(): Promise<ModelInfo[]> {
  const [folder, lms, ollama] = await Promise.all([
    findGgufModels(modelDirs()),
    findGgufModels(defaultLmStudioDirs()).then((ms) => ms.map((m) => ({ ...m, runtime: 'lmstudio' as const }))),
    listOllamaModels().then((ms) => Promise.all(ms.map((m) => toOllamaModelInfo(m)))).catch(() => [])
  ])
  const seen = new Set<string>()
  return [...folder, ...lms, ...ollama].filter((m) => (seen.has(m.path.toLowerCase()) ? false : (seen.add(m.path.toLowerCase()), true)))
}
/** Fetch + cache one model card (5 s timeout inside). Records the error instead of retrying forever. */
async function fetchCard(path: string, repoId: string): Promise<{ generation?: unknown; error?: string }> {
  // sibling quantizations of the linked repo (I-9.2 suggestions); a failure here never blocks the model card
  if (!readSidecar(path)?.siblingsFetchedAt) await refreshSiblings(path, repoId).catch(() => writeSidecar(path, { siblingsFetchedAt: new Date().toISOString() }))
  try {
    const generation = await fetchGenerationConfig(repoId)
    writeSidecar(path, { generation: generation ?? undefined, fetchedAt: new Date().toISOString(), fetchError: generation ? undefined : 'repository has no generation_config.json' })
    return { generation }
  } catch (e) {
    writeSidecar(path, { fetchedAt: new Date().toISOString(), fetchError: (e as Error).message })
    return { error: (e as Error).message }
  }
}
let cardRefresh: Promise<void> | null = null
/** Folder models with a known repo and no fetch attempt yet: fetch their model cards one by one in the background. */
function refreshModelCards(ms: ModelInfo[]): void {
  if (cardRefresh) return
  const dirs = modelDirs()
  const todo = ms.filter((m) => m.runtime === 'llamacpp' && dirs.some((d) => insideSomeRoot(m.path, [d])) && (() => { const s = readSidecar(m.path); return !!s?.repoId && (!s.fetchedAt || !s.siblingsFetchedAt) })())
  if (!todo.length) return
  cardRefresh = (async () => { for (const m of todo) await fetchCard(m.path, readSidecar(m.path)!.repoId!) })().finally(() => { cardRefresh = null })
}

/** Roots a benchmark may load a model from (validation). */
const modelRoots = () => [...modelDirs(), ...defaultLmStudioDirs(), join(defaultOllamaRoot(), 'blobs')]

const llama = new LlamaCppBackend(llamaDir(), { pidFile: join(app.getPath('userData'), 'llama-server.pid') })
const llamaHip = new LlamaCppBackend(hipDir(), { pidFile: join(app.getPath('userData'), 'llama-server-hip.pid') })
const markerOf = (dir: string): string | null => { try { return readFileSync(join(dir, 'release-tag.txt'), 'utf8').trim() } catch { return null } }
const primaryBackendKind = (): 'cuda' | 'vulkan' => /\bcuda-/.test(markerOf(llamaDir()) ?? '') ? 'cuda' : 'vulkan'

/** Each installed llama.cpp backend, detected on its own (a HIP build that fails --version never hides Vulkan). */
async function installedBackends(): Promise<InstalledRuntime[]> {
  const one = async (b: LlamaCppBackend, dir: string, kind: GpuBackendKind): Promise<InstalledRuntime> => {
    const d = await b.detect()
    return { kind, vendorDir: dir, exePath: b.exePath, build: d.status === 'available' ? d.version ?? null : null, status: d.status, ...(d.error ? { error: d.error } : {}) }
  }
  // the primary dir holds a CUDA build when the installer chose one (marker "<tag> cuda-X.Y")
  const primaryKind = primaryBackendKind()
  return Promise.all([one(llama, llamaDir(), primaryKind), one(llamaHip, hipDir(), 'hip')])
}
let smokeBusy = false
let db: DatabaseSync | null = null
const DEMO = process.env.LAO_SEED_DEMO === '1'
const needDb = () => { if (!db) throw new Error('database not open yet'); return db }

/** Forward a runner event to every window (channel bench:event). */
export function sendBenchEvent(e: SessionEvent): void {
  for (const w of BrowserWindow.getAllWindows()) w.webContents.send('bench:event', e)
}

ipcMain.handle('runtimes:detect', () => detectRuntimes(llamaDir()))
// nvidia-smi is probed once per app run (it takes ~1 s and the answer doesn't change).
let nvProbe: Promise<NvidiaProbe> | null = null
const nvidia = () => (nvProbe ??= probeNvidiaSmi().catch((e: Error) => ({ available: false as const, reason: e.message, source: 'nvidia-smi' })))
ipcMain.handle('system:scan', async () => {
  const [profile, runtimes, nv] = await Promise.all([scanSystem(), detectRuntimes(llamaDir()), nvidia()])
  profileCache = profile
  const cuda = nv.available && nv.cudaVersion ? `${nv.cudaVersion.major}.${nv.cudaVersion.minor}` : null
  return { ...profile, runtimes, nvidiaSmi: { available: nv.available, reason: nv.available ? null : nv.reason, cudaVersion: cuda } }
})
ipcMain.handle('models:list', async () => {
  const ms = await listAllModels()
  refreshModelCards(ms) // background; results show up on the next listing
  return ms
})
/** Attach a Hugging Face repo to a local model file (folder models only) and fetch its generation_config.json now. */
ipcMain.handle('models:linkRepo', async (_e, path: unknown, repoId: unknown): Promise<{ ok: boolean; generation?: unknown; error?: string }> => {
  if (typeof path !== 'string' || !existsSync(path) || !modelDirs().some((d) => insideSomeRoot(path, [d]))) return { ok: false, error: 'model file not in a configured model folder' }
  if (typeof repoId !== 'string' || !/^[\w.-]+\/[\w.-]+$/.test(repoId.trim())) return { ok: false, error: 'repository id must look like owner/name' }
  writeSidecar(path, { repoId: repoId.trim(), revision: 'main', generation: undefined, fetchedAt: undefined, fetchError: undefined })
  const r = await fetchCard(path, repoId.trim())
  return r.error ? { ok: false, error: r.error } : { ok: true, generation: r.generation ?? null }
})
/** Per model: null if normal mode yields candidates for this workload, else the planner's reason (e.g. "full GPU
 *  offload does not fit — enable heavy-model mode"). Same generateCandidates call as a real session. */
ipcMain.handle('models:fit', async (_e, w: WorkloadId): Promise<ModelFit> => {
  if (!isWorkloadId(w)) throw new Error(`unknown workload ${String(w)}`)
  profileCache ??= await scanSystem()
  const infos = await listAllModels()
  const devices = await llama.listDevices().catch(() => null)
  if (!devices) return { reasons: Object.fromEntries(infos.map((m) => [m.id, 'llama.cpp runtime not installed (System page)'])), vramInUseBytes: null, vramTotalBytes: null }
  const device = pickBenchmarkDevice(devices, !!profileCache.gpus.value?.some((g) => g.isIntegrated))?.id ?? null
  const kind = device ? primaryBackendKind() : 'cpu'
  const bk = vramBudgetKey(profileCache, kind, (await llama.detect()).version)
  const machine = machineFromProfile(await withVramInUse(profileCache), device, undefined, bk ? applicableObservations(bk, listVramBudget(needDb(), bk.key)) : [])
  const vramInUseBytes = machine.vramInUseBytes.kind === 'measured' ? machine.vramInUseBytes.value : null
  const out: Record<string, string | null> = {}
  for (const info of infos) {
    const mm = toModelMeta(info)
    if (!mm.meta) { out[info.id] = mm.reason; continue }
    const set = generateCandidates(machine, mm.meta, { backend: kind }, WORKLOADS[w], rulesForRequest({ heavyMode: false }))
    out[info.id] = set.candidates.length ? null : set.rejected.map((r) => r.reason).join('; ') || 'no candidate configuration'
  }
  return { reasons: out, vramInUseBytes, vramTotalBytes: machine.vramBytes.value, suggestions: modelSuggestions(machine, infos, WORKLOADS[w]) }
})
let installing: Promise<unknown> | null = null
ipcMain.handle('runtime:install', async () => {
  if (installing) throw new Error('runtime install already running')
  if (active || smokeBusy) throw new Error('a benchmark is running; install the runtime after it finishes')
  const progress = (msg: string) => { for (const w of BrowserWindow.getAllWindows()) w.webContents.send('runtime:progress', msg) }
  // Claimed synchronously, before the first await: a benchmark/smoke started meanwhile sees it (W4b F18).
  installing = (async () => {
    profileCache ??= await scanSystem()
    const gpu = (profileCache.gpus.value ?? []).filter((g) => !g.isIntegrated).sort((a, b) => (b.dedicatedVramBytes.value ?? 0) - (a.dedicatedVramBytes.value ?? 0))[0]
    const vendor = gpu?.vendor ?? 'other'
    const nv = vendor === 'nvidia' ? await nvidia() : null
    return llama.ensureRuntime(progress, { vendor, cudaMajor: nv?.available ? nv.cudaVersion?.major : undefined })
  })()
  try { return await installing } finally { installing = null }
})
ipcMain.handle('runtime:backends', () => installedBackends())
ipcMain.handle('runtime:installHip', async () => {
  if (installing) throw new Error('runtime install already running')
  if (active || smokeBusy) throw new Error('a benchmark is running; install the runtime after it finishes')
  const progress = (msg: string) => { for (const w of BrowserWindow.getAllWindows()) w.webContents.send('runtime:progress', msg) }
  installing = (async () => {
    profileCache ??= await scanSystem()
    if (!(profileCache.gpus.value ?? []).some((g) => g.vendor === 'amd' && !g.isIntegrated)) throw new Error('the ROCm/HIP build is for AMD GPUs; no discrete AMD GPU was found')
    // same llama.cpp build as the installed Vulkan runtime, so a backend A/B isolates the backend
    const tag = markerOf(llamaDir())?.split(/\s+/)[0]
    return llamaHip.ensureRuntime(progress, { vendor: 'amd', hip: true, ...(tag ? { tag } : {}) })
  })()
  try { return await installing } finally { installing = null }
})
registerHubIpc(ipcMain, () => BrowserWindow.getAllWindows()[0] ?? null, { userDataDir: app.getPath('userData'), modelDirs })
ipcMain.handle('settings:get', () => readSettings())
ipcMain.handle('settings:setWorkload', (_e, w: WorkloadId) => {
  if (!isWorkloadId(w)) throw new Error(`unknown workload ${String(w)}`)
  return writeSettings({ workload: w })
})
ipcMain.handle('settings:setRequiredContext', (_e, ctx: unknown) => {
  if (ctx !== null && !REQUIRED_CTX.includes(ctx as number)) throw new Error('required context must be 32K, 64K, 128K or Auto')
  return writeSettings({ requiredContext: ctx as number | null })
})
ipcMain.handle('workloads:list', () => Object.values(WORKLOADS))
ipcMain.handle('sessions:list', () => listSessions(needDb()))
ipcMain.handle('sessions:get', (_e, id: unknown) => getSession(needDb(), rowId(id)))
/** Save exported text via the OS save dialog. Only the name and content come from the renderer; the user picks the path. */
ipcMain.handle('file:save', async (e, name: string, content: string) => {
  if (typeof name !== 'string' || typeof content !== 'string' || content.length > 1024 * 1024) throw new Error('bad export')
  const win = BrowserWindow.fromWebContents(e.sender)
  const opts = { defaultPath: name.replace(/[\\/:*?"<>|]/g, '_'), filters: [{ name: 'All files', extensions: ['*'] }] }
  const r = win ? await dialog.showSaveDialog(win, opts) : await dialog.showSaveDialog(opts)
  if (r.canceled || !r.filePath) return { saved: null }
  writeFileSync(r.filePath, content)
  return { saved: r.filePath }
})
ipcMain.handle('telemetry:run', (_e, runId: unknown) => telemetryForRun(needDb(), rowId(runId)))
/** Same measurements, other workload: never written back as the session's own recommendation. */
ipcMain.handle('recommendation:compute', async (_e, rawId: unknown, w: WorkloadId): Promise<ComputedRecommendation | null> => {
  const id = rowId(rawId)
  if (!isWorkloadId(w)) throw new Error(`unknown workload ${String(w)}`)
  const s = sessionInputs(needDb(), Number(id))
  if (!s) return null
  const device = s.inputs.find((i) => i.config.device)?.config.device ?? null
  const machine = machineFromProfile(s.machine ?? (profileCache ??= await scanSystem()), device)
  // The session's own requiredContext / minDecodeTps still apply when viewing it as another workload.
  const recommendation = recommendForWorkload(
    { candidates: s.inputs, machine, allRuns: s.allRuns, planningSnapshot: s.planningSnapshot, ...(s.stopReason ? { stopReason: s.stopReason } : {}), ...(s.sessionVersions ? { sessionVersions: s.sessionVersions } : {}) },
    w, { requiredContext: s.request?.requiredContext, minDecodeTps: s.request?.minDecodeTps })
  return { sessionId: Number(id), workload: w, recommendation, label: `computed from session #${Number(id)}` }
})
ipcMain.handle('recommendation:latest', (_e, w: unknown) => {
  if (!isWorkloadId(w)) throw new Error(`unknown workload ${String(w)}`)
  return latestRecommendation(needDb(), w)
})
ipcMain.handle('bench:start', (_e, raw: unknown) => {
  const v = sanitizeRequest(raw, modelRoots())
  return v.ok ? startSession(v.req) : { ok: false, error: v.error }
})
/** Continue a paused/cancelled/interrupted/failed session. opts: retryFailed / rerunConfigIds (validated). */
ipcMain.handle('bench:resume', (_e, rawId: unknown, opts?: { retryFailed?: unknown; rerunConfigIds?: unknown }) => {
  const id = rowId(rawId)
  const stored = getSessionResume(needDb(), Number(id))
  if (!stored) return { ok: false, error: `session ${id} has no stored request` }
  const v = sanitizeRequest({ ...stored.request, retryFailed: opts?.retryFailed, rerunConfigIds: opts?.rerunConfigIds }, modelRoots())
  return v.ok ? startSession({ ...v.req, resumeSessionId: String(Number(id)) }, stored.machine, stored.plan) : { ok: false, error: v.error }
})
ipcMain.handle('bench:cancel', () => {
  if (!active) return { ok: false, error: 'no benchmark running' }
  active.cancel.abort()
  return { ok: true }
})
/** Stops between steps; the session ends as 'paused' and can be resumed. */
ipcMain.handle('bench:pause', () => {
  if (!active) return { ok: false, error: 'no benchmark running' }
  active.pause.abort()
  return { ok: true }
})
ipcMain.handle('bench:smoke', async (_e, modelPath: string): Promise<SmokeResult> => {
  if (typeof modelPath !== 'string' || !insideSomeRoot(modelPath, modelRoots())) throw new Error('model path not in a configured model dir')
  if (smokeBusy || active || installing || serving) throw new Error('a smoke run, benchmark, served model or runtime install is already in progress')
  smokeBusy = true
  try {
    // Deliberately tiny: ctx 2048, 32 tokens, one warmup + one measured request.
    const profile = profileCache ??= await scanSystem()
    const dev = pickBenchmarkDevice(await llama.listDevices(), !!profile.gpus.value?.some((g) => g.isIntegrated))
    if (!dev) throw new Error('no physical Vulkan device reported by llama-server --list-devices')
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

// ---- Serve the recommended config from the app (llama-server + its built-in web UI in the default browser) ----
let serving: { backend: LlamaCppBackend; configId: string; alias: string; ctx: number } | null = null
ipcMain.handle('serve:start', async (_e, raw: unknown): Promise<{ ok: boolean; url?: string; error?: string }> => {
  // Models page "Run…" markers: device 'auto' → the device a benchmark would use; threads 0 → physical cores.
  if (raw && typeof raw === 'object') {
    const r = raw as Record<string, unknown>
    if (r.device === 'auto' || r.threads === 0) {
      const profile = profileCache ??= await scanSystem()
      if (r.threads === 0) r.threads = profile.cpu.value?.physicalCores ?? 0
      if (r.device === 'auto') {
        r.device = (r.backend === 'hip'
          ? pickDiscreteDevice(await llamaHip.listDevices())
          : pickBenchmarkDevice(await llama.listDevices(), !!profile.gpus.value?.some((g) => g.isIntegrated)))?.id ?? null
      }
    }
  }
  const v = sanitizeServeConfig(raw, modelRoots())
  if (!v.ok) return { ok: false, error: v.error }
  if (smokeBusy || active || installing || serving) return { ok: false, error: 'a benchmark, smoke run, served model or runtime install is already in progress' }
  const backend = v.cfg.backend === 'hip' ? llamaHip : llama
  serving = { backend, configId: v.cfg.configId, alias: modelAlias(v.cfg.modelName), ctx: v.cfg.ctx } // claim before the first await
  try {
    await backend.loadModel(toLoadConfig(v.cfg))
    const url = backend.url!
    await shell.openExternal(url)
    return { ok: true, url }
  } catch (e) {
    await backend.unloadModel().catch(() => {})
    serving = null
    return { ok: false, error: (e as Error).message }
  }
})
ipcMain.handle('serve:stop', async () => {
  if (!serving) return { ok: false, error: 'nothing is being served' }
  await serving.backend.unloadModel()
  serving = null
  return { ok: true }
})
ipcMain.handle('serve:status', (): ServeStatus => (serving?.backend.url ? { url: serving.backend.url, configId: serving.configId, alias: serving.alias, ctx: serving.ctx } : { url: null, configId: null, alias: null, ctx: null }))

/** The scan plus a fresh reading of VRAM already in use (other apps), so planning budgets what is actually free. */
async function withVramInUse(p: SystemProfile, signal?: AbortSignal): Promise<SystemProfile> {
  if (!(p.gpus.value ?? []).some((g) => !g.isIntegrated)) return { ...p, vramInUse: { value: null, status: 'unavailable', source: 'integrated GPU', error: 'no separate dedicated-VRAM budget' } }
  const r = await readVramInUse(20_000, signal)
  return {
    ...p,
    vramInUse: r
      ? { value: r.bytes, status: 'available', source: `typeperf GPU Adapter Memory(luid_${r.luid}_phys_0)\\Dedicated Usage` }
      : { value: null, status: 'unavailable', source: 'typeperf GPU Adapter Memory', error: 'reading failed' }
  }
}

// ---- Benchmark session (one at a time, in the main process) ----
let active: { cancel: AbortController; pause: AbortController } | null = null
let profileCache: SystemProfile | null = null

/** req is already sanitized. storedMachine / storedPlan: the scan and the candidate configs a resumed session was
 *  planned with — the runner re-uses the plan as-is, so config ids survive estimator changes. */
async function startSession(req: SessionRequest, storedMachine?: SystemProfile, storedPlan?: CandidateConfig[]): Promise<StartResult> {
  if (active || smokeBusy || installing || serving) return { ok: false, error: 'a benchmark, smoke run, served model or runtime install is already in progress' }
  const me = { cancel: new AbortController(), pause: new AbortController() }
  active = me // claim before the first await so a double click can't start two sessions
  try {
    // ponytail: static facts cached per app run; RAM is re-read live via freemem(). A resume re-uses the stored scan
    // so generateCandidates yields the same configIds/ctxSteps as the original plan (review item d).
    const profile = storedMachine ?? (await withVramInUse((profileCache ??= await scanSystem()), me.cancel.signal))
    if (me.cancel.signal.aborted) throw new Error('cancelled')
    const runtime = await llama.detect()
    if (runtime.status !== 'available') throw new Error('llama.cpp runtime is not installed: install it on the System page first')
    const [infos, devices] = await Promise.all([listAllModels(), llama.listDevices()])
    const device = pickBenchmarkDevice(devices, !!profile.gpus.value?.some((g) => g.isIntegrated))?.id ?? null
    const models: ModelMeta[] = []
    for (const id of req.modelIds) {
      const info = infos.find((m) => m.path === id)
      const mm = info ? toModelMeta(info) : { meta: null, reason: 'file not found' }
      if (mm.meta) models.push(mm.meta)
      else sendBenchEvent({ sessionId: '', type: 'log', level: 'warn', msg: `${id}: ${mm.reason}; skipped` })
    }
    if (!models.length) throw new Error('none of the selected models can be benchmarked')
    const primaryGpuKind = primaryBackendKind()
    const backendKind = device ? primaryGpuKind : 'cpu' as const
    // Installed backends, each with its OWN device id (Vulkan0 / ROCm0) from its own --list-devices. The opt-in HIP
    // build joins only when it runs and enumerates the GPU; otherwise the session is Vulkan-only and says why.
    const backends: InstalledBackend[] = [{ kind: primaryGpuKind, backend: () => llama, runtimeVersion: runtime.version ?? null, exePath: llama.exePath, device }]
    const hipRt = await llamaHip.detect()
    if (hipRt.status === 'available' && req.compareBackends !== false) {
      const hipDev = pickDiscreteDevice(await llamaHip.listDevices().catch(() => []))?.id ?? null
      if (hipDev) backends.push({ kind: 'hip', backend: () => llamaHip, runtimeVersion: hipRt.version ?? null, exePath: llamaHip.exePath, device: hipDev })
      else sendBenchEvent({ sessionId: '', type: 'log', level: 'warn', msg: 'ROCm (HIP) build installed but it lists no discrete GPU (llama-server --list-devices): comparing on Vulkan only' })
    }
    const nvOk = (await nvidia()).available
    // Same deterministic candidate generation the runner does (planCandidates over the same backends, rulesForRequest
    // keeps heavyMode), so stored sessions carry full configs with identical ids.
    const planFor: PlanFor = (r) => {
      // Per backend: the per-process budget observations the runner reads (key: GPU + driver + backend build).
      const planned: PlannedBackend[] = backends.map((b) => {
        const kind = b.device ? b.kind : 'cpu' as const
        const bk = vramBudgetKey(profile, kind, b.runtimeVersion)
        return { kind, device: b.device, runtimeVersion: b.runtimeVersion, observations: bk ? applicableObservations(bk, listVramBudget(needDb(), bk.key)) : [] }
      })
      const machine = machineFromProfile(profile, device, undefined, planned[0].observations ?? [])
      return {
        machine: profile,
        vramBytes: val(machine.vramBytes, true),
        candidates: models.flatMap((model) => planCandidates(profile, model, planned, WORKLOADS[r.workload], rulesForRequest(r)).candidates.map((config) => ({ config, model })))
      }
    }
    let gotId: (id: string) => void = () => {}
    const idReady = new Promise<string>((r) => { gotId = r })
    const base = makeSessionStorage(needDb(), planFor)
    const storage: SessionStorage = { ...base, createSession: async (s) => { const id = await base.createSession(s); gotId(id); return id } }
    if (req.resumeSessionId) gotId(req.resumeSessionId)
    const run = runSession(req, {
      backend: () => llama, backends, storage, machine: profile, gpuDevice: device, backendKind,
      // PDH (all vendors) + nvidia-smi temp/power when it works (NVIDIA only; AMD has no non-admin source).
      startSampler: (pid) => withNvidia(startSampler({ pid }), nvOk ? startNvidiaSampler() : null),
      models, clock: Date, readRamAvailableBytes: () => freemem(), evaluate: evaluateAsync,
      signal: me.cancel.signal, pauseSignal: me.pause.signal, runtimeVersion: runtime.version ?? null,
      ...(storedPlan?.length ? { plan: storedPlan } : {})
    }, sendBenchEvent).finally(() => { if (active === me) active = null })
    const id = await Promise.race([idReady, run.then(() => null)])
    return id ? { ok: true, sessionId: id } : { ok: false, error: 'session ended before it started (see event log)' }
  } catch (e) {
    if (active === me) active = null
    return { ok: false, error: (e as Error).message }
  }
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1200,
    minWidth: 960,
    minHeight: 640,
    height: 800,
    backgroundColor: '#0f1115',
    title: 'Local AI Optimizer',
    webPreferences: { preload: join(__dirname, '../preload/index.js'), sandbox: true, contextIsolation: true }
  })
  if (process.env.ELECTRON_RENDERER_URL) win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else win.loadFile(join(__dirname, '../renderer/index.html'))
}

app.whenReady().then(async () => {
  // The packaged app's default download destination is userData/models, not the read-only install/asar dir.
  // Also covers portable builds and upgrades that did not run the NSIS install hook.
  mkdirSync(bundledModelsDir(), { recursive: true })
  // LAO_SEED_DEMO=1 uses a separate DB file so fixture data can never reach the real one.
  db = openDb(join(app.getPath('userData'), DEMO ? 'optimizer-demo.db' : 'optimizer.db'))
  const n = markInterrupted(db) // a previous app run quit mid-session (before-quit can't await the runner)
  if (n) console.warn(`${n} session(s) marked interrupted`)
  const fixtures = join(app.getAppPath(), 'tests', 'fixtures', 'scoring')
  if (DEMO && existsSync(fixtures)) seedDemoSession(db, fixtures) // dev-only: fixtures are not packaged
  for (const pid of ['llama-server.pid', 'llama-server-hip.pid']) {
    const stale = await killStaleServer(join(app.getPath('userData'), pid)).catch((e: Error) => `stale-server check failed: ${e.message}`)
    if (stale) console.warn(stale)
  }
  createWindow()
})
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => { active?.cancel.abort(); stopAllSamplers(); llama.killSync(); llamaHip.killSync() })
process.on('exit', () => { llama.killSync(); llamaHip.killSync() })
