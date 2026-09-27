// Core session runner E2E on real hardware, independent of the Electron/IPC wiring.
// Usage: npx tsx scripts/run-session.ts <A|B|C|H> [--heavy] [--workload coding] [--models a,b] [--ladder 2048,8192] [--no-quality] [--ram-abort-gib 5] [--required-ctx 131072]
//        [--gen-configs '[{...GenConfig}]'] [--gen-search] [--quality-mode quick|thorough] [--max-per-model 1] [--db <optimizer.db>]
//   A  coding workload, qwen2.5-1.5b + llama-3.1-8b, quality on, default ladder/reps
//   B  same session, abort ~20 s into the 8B ladder (cancel path)
//   C  RAM floor 64 GiB (guard path: every step skipped_memory, no load)
//   H  custom: flags choose workload/models/ladder/heavy mode (quality on unless --no-quality)
// Dumps everything (events, runs, quality, recommendation, raw quality prompts/replies) to docs/session-run-<scenario>-<ts>.json.
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { freemem, tmpdir } from 'node:os'
import { join } from 'node:path'
import type { SessionEvent, SessionRequest } from '../src/shared/bench-events'
import type { BenchmarkRunResult, ModelMeta, QualityResult, Recommendation } from '../src/shared/bench-types'
import type { ModelInfo } from '../src/shared/types'
import { runSession, type RunDetail, type SessionBackend, type SessionStorage } from '../src/core/benchmark/session'
import { findGgufModels, toModelMeta } from '../src/core/models/gguf'
import { LlamaCppBackend } from '../src/core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../src/core/runtimes/llamacpp/parse'
import { scanSystem } from '../src/core/system/scanner'
import { readVramInUse, startSampler } from '../src/core/telemetry/sampler'
import { openDb } from '../src/core/storage/db'
import { getSessionResume, listVramBudget, makeSessionStorage, type PlanFor } from '../src/core/storage/sessions'
import { applicableObservations, machineFromProfile, planCandidates, rulesForRequest, vramBudgetKey } from '../src/core/benchmark/candidates'
import type { CandidateConfig, KvType } from '../src/shared/bench-types'
import { WORKLOADS } from '../src/core/scoring/workloads'
import { val } from '../src/core/scoring/cliff'
import type { SystemProfile } from '../src/shared/types'

const scenario = (process.argv[2] ?? 'A').toUpperCase()
const MODELS_DIR = 'D:\\llm-models'
const flag = (k: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined }
const WANT = flag('--models')?.split(',') ?? ['qwen2.5-1.5b-instruct-q4_k_m', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M']
/** --request-cap-ms N: upper bound on every prompt request's timeout (overnight runs: 300000). */
const REQUEST_CAP_MS = flag('--request-cap-ms') ? Number(flag('--request-cap-ms')) : null
/** --pin '[{"model":"Q4_K_M","ngl":54,"kv":"f16"}]': run exactly these configs instead of the planner's pick. Each is
 *  the planner's first candidate of a matching model (substring of name/id) with gpuLayers/kvType/id replaced and the
 *  steps = --ladder. ponytail: estimates/notes stay the planner's (said in notes); fine for experiments, not the app. */
const PINS: { model: string; ngl: number; kv?: KvType }[] | null = flag('--pin') ? JSON.parse(flag('--pin')!) : null
function pin(model: ModelMeta, cands: CandidateConfig[]): CandidateConfig[] {
  if (!PINS) return cands
  const base = cands[0]
  const pins = PINS.filter((p) => model.name.includes(p.model) || model.id.includes(p.model))
  if (pins.length && !base) throw new Error(`--pin: planner produced no candidate for ${model.name} to clone`)
  return pins.map((p) => {
    const all = p.ngl >= model.layers, kv = p.kv ?? base.kvType
    return {
      ...base, id: `${model.id}|ngl=${all ? 'all' : p.ngl}|kv=${kv}|t=${base.threads}`, gpuLayers: Math.min(p.ngl, model.layers), gpuLayersAll: all, kvType: kv,
      kvOffload: undefined, mmap: all ? base.mmap : false, ...(all ? {} : { expectDegraded: true, degradedReason: 'pinned partial offload' }),
      ctxSteps: flag('--ladder') ? flag('--ladder')!.split(',').map(Number) : base.ctxSteps, skippedSteps: [],
      notes: [...base.notes, `pinned by run-session --pin; estimates are the planner's for ngl ${base.gpuLayers} kv ${base.kvType}`]
    }
  })
}
const GiB = 1024 ** 3
const t0 = Date.now()
const el = () => `${((Date.now() - t0) / 1000).toFixed(1).padStart(7)}s`

// GGUF → ModelMeta via the app's own mapping (carries supportsThinking etc.); missing fields → model skipped.
function toMeta(info: ModelInfo): ModelMeta | string {
  const r = toModelMeta(info)
  return r.meta ?? r.reason
}

// In-memory storage implementing #1's SessionStorage.
const db = {
  sessions: [] as { id: string; workload: string; request: SessionRequest; startedAt: number; status: string; error?: string }[],
  runs: [] as { sessionId: string; run: BenchmarkRunResult; detail: Omit<RunDetail, 'samples'> & { sampleCount: number; samples: RunDetail['samples'] } }[],
  quality: [] as { sessionId: string; modelId: string; configId: string; ctx: number; results: QualityResult[] }[],
  recommendations: [] as { sessionId: string; rec: Recommendation }[]
}
const storage: SessionStorage = {
  createSession: (s) => { const id = `s${db.sessions.length + 1}`; db.sessions.push({ id, ...s, status: 'created' }); return id },
  setSessionStatus: (id, status, error) => { Object.assign(db.sessions.find((s) => s.id === id)!, { status, error }) },
  listRuns: (id) => db.runs.filter((r) => r.sessionId === id).map((r) => r.run),
  saveRun: (id, run, detail) => { db.runs.push({ sessionId: id, run, detail: { ...detail, sampleCount: detail.samples.length } }) },
  listQuality: (id, modelId) => db.quality.filter((q) => q.sessionId === id && q.modelId === modelId).flatMap((q) => q.results),
  saveQuality: (id, modelId, configId, ctx, results) => { db.quality.push({ sessionId: id, modelId, configId, ctx, results }) },
  saveRecommendation: (id, rec) => { db.recommendations.push({ sessionId: id, rec }) }
}

// Backend wrapper: records templated prompts + raw replies so the chat template can be checked by eye.
const transcripts: { configId: string; templated: string; reply: string; stopType: string | null }[] = []
let currentConfig = ''
// Host-RAM checks for the quality phase (the default 8 GiB prompt cache once held ~13.6 GiB there; b3e671f passes
// --cache-ram 0): lowest system RAM available per phase, and whether the server log (rolling tail) ever reports the cache.
let phase: 'ladder' | 'quality' = 'ladder'
const minRamAvail = { ladder: Infinity, quality: Infinity }
let promptCacheSeen = false
// Patch the REAL backend instance (only the two methods we observe) instead of re-listing methods in a new object:
// a hand-written wrapper silently drops whatever the backend adds later (it once dropped applyTemplate's opts, then
// tokenize/templateHash — so the runner fell back to untokenized prompt sizing). The app passes the backend directly.
function wrap(b: LlamaCppBackend): SessionBackend {
  let lastTemplated: string | null = null
  const applyTemplate = b.applyTemplate.bind(b)
  const runPrompt = b.runPrompt.bind(b)
  b.applyTemplate = async (...a: Parameters<LlamaCppBackend['applyTemplate']>) => (lastTemplated = await applyTemplate(...a))
  // --no-warmup (E3 cold vs warm): warmup is skipped but the row still says warm — persist such runs to a separate --db.
  if (process.argv.includes('--no-warmup')) b.warmup = async () => {}
  b.runPrompt = async (...a: Parameters<LlamaCppBackend['runPrompt']>) => {
    if (REQUEST_CAP_MS) a[0] = { ...a[0], timeoutMs: Math.min(a[0].timeoutMs ?? REQUEST_CAP_MS, REQUEST_CAP_MS) }
    const r = await runPrompt(...a)
    const req = a[0]
    if (b.log.some((l) => /prompt cache is enabled/i.test(l))) promptCacheSeen = true // read-only: b.log feeds exit classification
    if (lastTemplated !== null && req.prompt === lastTemplated) transcripts.push({ configId: currentConfig, templated: req.prompt.slice(-400), reply: r.text, stopType: r.stopType })
    return r
  }
  return b
}

const servers = () =>
  execFileSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true }).split('\n').filter((l) => /^"llama-server\.exe"/i.test(l)).length

async function main(): Promise<void> {
  for (let i = 0; servers() > 0; i++) {
    if (i >= 20) throw new Error('llama-server still running after 10 min')
    console.log(`${el()} llama-server already running (another worker) — waiting 30 s`)
    await new Promise((r) => setTimeout(r, 30_000))
  }
  const vendor = join('vendor', 'llama.cpp')
  const probe = new LlamaCppBackend(vendor)
  const [scanned, det, devs, infos] = await Promise.all([scanSystem(), probe.detect(), probe.listDevices(), findGgufModels([MODELS_DIR])])
  // Same as main.ts withVramInUse: other-process VRAM measured at planning time (effective budget).
  const vr = await readVramInUse()
  const machine: SystemProfile = { ...scanned, vramInUse: vr
    ? { value: vr.bytes, status: 'available', source: `typeperf GPU Adapter Memory(luid_${vr.luid}_phys_0)\Dedicated Usage` }
    : { value: null, status: 'unavailable', source: 'typeperf GPU Adapter Memory', error: 'reading failed' } }
  const dev = pickDiscreteDevice(devs)
  const dbPath = flag('--db')
  const appDb = dbPath ? openDb(dbPath) : null
  // --resume <id> (needs --db): continue a stored session exactly like the app's Resume — stored request, scan and
  // candidate plan; already measured (configId, ctx) steps are reused, cancelled/skipped ones re-run.
  const resumeId = flag('--resume')
  const stored = resumeId && appDb ? getSessionResume(appDb, Number(resumeId)) : null
  if (resumeId && !stored) throw new Error(`session ${resumeId} not found in ${dbPath ?? '(no --db)'}`)
  // Resume selects the stored session's models by id (= absolute path), never the --models default: a missing model
  // must fail here, not run an empty session that re-saves an empty recommendation over the stored one.
  const models: ModelMeta[] = []
  for (const name of stored ? stored.request.modelIds : WANT) {
    const info = infos.find((i) => (stored ? i.path === name : i.name === name))
    if (!info) throw new Error(`${name} not found in ${MODELS_DIR}`)
    const m = toMeta(info)
    if (typeof m === 'string') throw new Error(`${name}: ${m}`)
    models.push(m)
  }
  console.log(`${el()} runtime ${det.version}; device ${dev?.id} ${dev?.name}; models ${models.map((m) => `${m.name} (${m.layers}L, ctx ${m.ctxTrain})`).join(', ')}`)

  const ctl = new AbortController()
  const events: SessionEvent[] = []
  let abortAt: number | null = null
  let abortScheduled = false
  let cancelledAt: number | null = null
  const req: SessionRequest = {
    workload: (flag('--workload') ?? 'coding') as SessionRequest['workload'], modelIds: models.map((m) => m.id),
    runQuality: scenario === 'A' || (scenario === 'H' && !process.argv.includes('--no-quality')),
    heavyMode: process.argv.includes('--heavy'),
    ...(flag('--required-ctx') ? { requiredContext: Number(flag('--required-ctx')) } : {}),
    ...(flag('--gen-configs') ? { genConfigs: JSON.parse(flag('--gen-configs')!) } : {}),
    ...(process.argv.includes('--gen-search') ? { genSearch: true } : {}),
    ...(flag('--quality-mode') ? { qualityMode: flag('--quality-mode') as 'quick' | 'thorough' } : {}),
    ...(flag('--max-per-model') ? { candidateRules: { maxPerModel: Number(flag('--max-per-model')) } } : {}),
    ...(flag('--ladder') ? { ladder: flag('--ladder')!.split(',').map(Number) } : {}),
    ...(flag('--reps') ? { reps: Number(flag('--reps')) } : {}),
    ...(flag('--quality-seed') ? { qualitySeed: Number(flag('--quality-seed')) } : {})
  }
  const pidFile = join(tmpdir(), `lao-session-${scenario}.pid`)
  // --db <optimizer.db>: persist through the app's own storage (makeSessionStorage + planFor, as main.ts does), so the
  // session shows up in the app (Results / Dashboard) and resume/read-time reinterpretation work on it.
  const git = (args: string[]) => { try { return execFileSync('git', args, { encoding: 'utf8', windowsHide: true }).trim() } catch { return null } }
  // Snapshots (git archive) have no .git: HEAD.txt records the commit.
  const gitState = { head: git(['rev-parse', 'HEAD']) ?? (existsSync('HEAD.txt') ? readFileSync('HEAD.txt', 'utf8').trim() : null), dirty: (git(['status', '--porcelain']) ?? '').split('\n').filter(Boolean) }
  // Same planning as the runner / main.ts: planCandidates over the (Vulkan-only) backend with the learned per-process
  // budget observations, so the stored plan carries the configs the runner actually runs.
  const bk = vramBudgetKey(machine, 'vulkan', det.version ?? null)
  const observations = bk && appDb ? applicableObservations(bk, listVramBudget(appDb, bk.key)) : []
  const planFor: PlanFor = (r) => {
    const mach = machineFromProfile(machine, dev?.id ?? null, undefined, observations)
    return {
      machine, vramBytes: val(mach.vramBytes, true),
      candidates: models.flatMap((model) => pin(model, planCandidates(machine, model, [{ kind: 'vulkan', device: dev?.id ?? null, runtimeVersion: det.version ?? null, observations }], WORKLOADS[r.workload], rulesForRequest(r)).candidates).map((config) => ({ config, model })))
    }
  }
  const appStorage = appDb ? makeSessionStorage(appDb, planFor) : null
  if (stored) {
    Object.assign(req, stored.request, { resumeSessionId: resumeId })
    console.log(`${el()} resuming session ${resumeId}: ${stored.request.workload}, ${stored.plan.length} planned configs`)
  }
  if (dbPath) console.log(`${el()} persisting to ${dbPath}; vramInUse ${vr ? (vr.bytes / GiB).toFixed(2) + ' GiB' : 'unavailable'}`)
  const file = join('docs', `session-run-${scenario}${scenario === 'H' ? `-${req.workload}${req.heavyMode ? '-heavy' : ''}` : ''}-${new Date(t0).toISOString().replace(/[:.]/g, '-')}.json`)
  let rec: Recommendation | null = null
  let ramAbort: string | null = null
  // File-backed: rewritten after every step/candidate/session event, so a killed job keeps what it measured.
  const save = () => {
    const unloadMs = abortAt !== null && cancelledAt !== null ? cancelledAt - abortAt : null
    writeFileSync(file, JSON.stringify({
      scenario, startedAt: new Date(t0).toISOString(), wallMs: Date.now() - t0, abortToCancelledMs: unloadMs, ramAbort,
      git: gitState, dbPath: dbPath ?? null, vramInUseAtPlanning: vr ? { bytes: vr.bytes, luid: vr.luid } : null,
      runtime: det.version, device: dev, models, request: req, recommendation: rec, db, transcripts, minRamAvailGiB: { ladder: +(minRamAvail.ladder / GiB).toFixed(2), quality: +(minRamAvail.quality / GiB).toFixed(2) }, promptCacheSeen,
      events: events.filter((e) => e.type !== 'telemetry'), telemetryEvents: events.filter((e) => e.type === 'telemetry').length
    }, null, 1))
  }
  // Self-abort when system RAM runs low (heavy runs on a 31 GB box).
  const ramAbortBytes = Number(flag('--ram-abort-gib') ?? 5) * GiB // 5 GiB: shared box (several agents + builds)
  const watchdog = setInterval(() => {
    minRamAvail[phase] = Math.min(minRamAvail[phase], freemem())
    if (ctl.signal.aborted || freemem() >= ramAbortBytes) return
    ramAbort = `system RAM available ${(freemem() / GiB).toFixed(1)} GiB < ${(ramAbortBytes / GiB).toFixed(1)} GiB at ${el().trim()}`
    console.log(`${el()} >>> RAM WATCHDOG ABORT: ${ramAbort}`)
    ctl.abort()
  }, 500)
  rec = await runSession(req, {
    backend: () => wrap(new LlamaCppBackend(vendor, { pidFile })),
    startSampler: (pid) => startSampler({ pid }),
    storage: appStorage ?? storage, machine, gpuDevice: dev?.id ?? null, backendKind: 'vulkan', runtimeVersion: det.version ?? null, models,
    clock: { now: () => Date.now() },
    readRamAvailableBytes: () => freemem(), // Windows: GlobalMemoryStatusEx ullAvailPhys = "Available"
    signal: ctl.signal,
    ...(stored ? { plan: stored.plan, machine: stored.machine ?? machine } : PINS ? { plan: planFor(req).candidates.map((c) => c.config) } : {}),
    config: scenario === 'C' ? { ramFloorMinBytes: 64 * GiB } : {}
  }, (e) => {
    events.push(e)
    if (e.type === 'telemetry') return
    if (e.type === 'phase' && e.phase === 'quality') phase = 'quality'
    if (e.type === 'candidate:started' || (e.type === 'phase' && e.phase === 'quality')) currentConfig = e.configId // quality runs after all ladders
    if (scenario === 'B' && e.type === 'step:started' && /Llama-3\.1-8B/i.test(e.configId) && !abortScheduled) {
      abortScheduled = true
      setTimeout(() => { abortAt = Date.now(); console.log(`${el()} >>> ABORT`); ctl.abort() }, 20_000)
    }
    if (e.type === 'session:cancelled') cancelledAt = Date.now()
    const short = e.type === 'step:done'
      ? `step:done ${e.configId.split('\\').pop()} @${e.ctx} ${e.verdict} ${e.result.status}${e.result.failureKind ? `/${e.result.failureKind}` : ''} pp=${e.result.prefillTps.value?.toFixed(0) ?? 'n/a'} tg=${e.result.decodeTps.value?.toFixed(1) ?? 'n/a'} ttft=${e.result.ttftMs.value?.toFixed(0) ?? 'n/a'} vram=${e.result.peakVramBytes.value != null ? (e.result.peakVramBytes.value / GiB).toFixed(2) : 'n/a'} shr=${e.result.peakSharedGpuBytes.value != null ? (e.result.peakSharedGpuBytes.value / GiB).toFixed(2) : 'n/a'}`
      : e.type === 'session:done' ? `session:done best=${e.recommendation.best?.configId ?? 'none'}`
        : JSON.stringify(e).replace(/D:\\\\llm-models\\\\/g, '').slice(0, 300)
    console.log(`${el()} ${short}`)
    if (e.type === 'step:done' || e.type === 'candidate:done' || e.type.startsWith('session:')) save()
  })
  clearInterval(watchdog)
  save()
  const unloadMs = abortAt !== null && cancelledAt !== null ? cancelledAt - abortAt : null
  console.log(`${el()} wall ${((Date.now() - t0) / 1000).toFixed(0)} s; abort→cancelled ${unloadMs ?? 'n/a'} ms; ram abort ${ramAbort ?? 'no'}; leftover llama-server ${servers()}; wrote ${file}`)
}

main().catch((e) => { console.error(e); process.exit(1) })
