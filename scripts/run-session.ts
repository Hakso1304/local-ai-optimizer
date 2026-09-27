// Core session runner E2E on real hardware, independent of the Electron/IPC wiring.
// Usage: npx tsx scripts/run-session.ts <A|B|C|H> [--heavy] [--workload coding] [--models a,b] [--ladder 2048,8192] [--no-quality] [--ram-abort-gib 3]
//   A  coding workload, qwen2.5-1.5b + llama-3.1-8b, quality on, default ladder/reps
//   B  same session, abort ~20 s into the 8B ladder (cancel path)
//   C  RAM floor 64 GiB (guard path: every step skipped_memory, no load)
//   H  custom: flags choose workload/models/ladder/heavy mode (quality on unless --no-quality)
// Dumps everything (events, runs, quality, recommendation, raw quality prompts/replies) to docs/session-run-<scenario>-<ts>.json.
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { freemem, tmpdir } from 'node:os'
import { join } from 'node:path'
import type { SessionEvent, SessionRequest } from '../src/shared/bench-events'
import type { BenchmarkRunResult, ModelMeta, QualityResult, Recommendation } from '../src/shared/bench-types'
import type { ModelInfo } from '../src/shared/types'
import { runSession, type RunDetail, type SessionBackend, type SessionStorage } from '../src/core/benchmark/session'
import { findGgufModels } from '../src/core/models/gguf'
import { LlamaCppBackend } from '../src/core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../src/core/runtimes/llamacpp/parse'
import { scanSystem } from '../src/core/system/scanner'
import { startSampler } from '../src/core/telemetry/sampler'

const scenario = (process.argv[2] ?? 'A').toUpperCase()
const MODELS_DIR = 'D:\\llm-models'
const flag = (k: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined }
const WANT = flag('--models')?.split(',') ?? ['qwen2.5-1.5b-instruct-q4_k_m', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M']
const GiB = 1024 ** 3
const t0 = Date.now()
const el = () => `${((Date.now() - t0) / 1000).toFixed(1).padStart(7)}s`

// Local GGUF → ModelMeta (#2's toModelMeta is not committed yet). No field is invented: missing → model skipped.
function toMeta(info: ModelInfo): ModelMeta | string {
  const g = info.meta
  if (!g) return info.metaError ?? 'no GGUF metadata'
  if (!g.arch || !g.blockCount || !g.embeddingLength || !g.headCount || !g.nVocab) return 'GGUF lacks arch/blockCount/embeddingLength/headCount/nVocab'
  return {
    id: info.path, name: g.name ?? info.name, fileBytes: g.fileSizeBytes, paramCount: g.parameterCount.value, quant: g.quantName,
    arch: g.arch, ctxTrain: g.contextLength, layers: g.blockCount, nEmbd: g.embeddingLength, heads: g.headCount,
    headsKv: g.headCountKv ?? g.headCount, keyLength: g.keyLength, valueLength: g.valueLength, nVocab: g.nVocab, slidingWindow: g.slidingWindow
  }
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
function wrap(b: LlamaCppBackend): SessionBackend {
  let lastTemplated: string | null = null
  return {
    get pid() { return b.pid },
    get lastExit() { return b.lastExit },
    loadModel: (c) => b.loadModel(c),
    unloadModel: () => b.unloadModel(),
    warmup: (p) => b.warmup(p),
    cancel: () => b.cancel(),
    applyTemplate: async (m) => (lastTemplated = await b.applyTemplate(m)),
    runPrompt: async (req) => {
      const r = await b.runPrompt(req)
      if (lastTemplated !== null && req.prompt === lastTemplated) transcripts.push({ configId: currentConfig, templated: req.prompt.slice(-400), reply: r.text, stopType: r.stopType })
      return r
    }
  }
}

const servers = () =>
  execFileSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], { encoding: 'utf8' }).split('\n').filter((l) => /^"llama-server\.exe"/i.test(l)).length

async function main(): Promise<void> {
  for (let i = 0; servers() > 0; i++) {
    if (i >= 20) throw new Error('llama-server still running after 10 min')
    console.log(`${el()} llama-server already running (another worker) — waiting 30 s`)
    await new Promise((r) => setTimeout(r, 30_000))
  }
  const vendor = join('vendor', 'llama.cpp')
  const probe = new LlamaCppBackend(vendor)
  const [machine, det, devs, infos] = await Promise.all([scanSystem(), probe.detect(), probe.listDevices(), findGgufModels([MODELS_DIR])])
  const dev = pickDiscreteDevice(devs)
  const models: ModelMeta[] = []
  for (const name of WANT) {
    const info = infos.find((i) => i.name === name)
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
    ...(flag('--ladder') ? { ladder: flag('--ladder')!.split(',').map(Number) } : {})
  }
  const pidFile = join(tmpdir(), `lao-session-${scenario}.pid`)
  const file = join('docs', `session-run-${scenario}${scenario === 'H' ? `-${req.workload}${req.heavyMode ? '-heavy' : ''}` : ''}-${new Date(t0).toISOString().replace(/[:.]/g, '-')}.json`)
  let rec: Recommendation | null = null
  let ramAbort: string | null = null
  // File-backed: rewritten after every step/candidate/session event, so a killed job keeps what it measured.
  const save = () => {
    const unloadMs = abortAt !== null && cancelledAt !== null ? cancelledAt - abortAt : null
    writeFileSync(file, JSON.stringify({
      scenario, startedAt: new Date(t0).toISOString(), wallMs: Date.now() - t0, abortToCancelledMs: unloadMs, ramAbort,
      runtime: det.version, device: dev, models, request: req, recommendation: rec, db, transcripts,
      events: events.filter((e) => e.type !== 'telemetry'), telemetryEvents: events.filter((e) => e.type === 'telemetry').length
    }, null, 1))
  }
  // Self-abort when system RAM runs low (heavy runs on a 31 GB box).
  const ramAbortBytes = Number(flag('--ram-abort-gib') ?? 3) * GiB
  const watchdog = setInterval(() => {
    if (ctl.signal.aborted || freemem() >= ramAbortBytes) return
    ramAbort = `system RAM available ${(freemem() / GiB).toFixed(1)} GiB < ${(ramAbortBytes / GiB).toFixed(1)} GiB at ${el().trim()}`
    console.log(`${el()} >>> RAM WATCHDOG ABORT: ${ramAbort}`)
    ctl.abort()
  }, 500)
  rec = await runSession(req, {
    backend: () => wrap(new LlamaCppBackend(vendor, { pidFile })),
    startSampler: (pid) => startSampler({ pid }),
    storage, machine, gpuDevice: dev?.id ?? null, backendKind: 'vulkan', runtimeVersion: det.version ?? null, models,
    clock: { now: () => Date.now() },
    readRamAvailableBytes: () => freemem(), // Windows: GlobalMemoryStatusEx ullAvailPhys = "Available"
    signal: ctl.signal,
    config: scenario === 'C' ? { ramFloorMinBytes: 64 * GiB } : {}
  }, (e) => {
    events.push(e)
    if (e.type === 'telemetry') return
    if (e.type === 'candidate:started') currentConfig = e.configId
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
