// Benchmark session runner (DESIGN §3): candidates → per context step (restart with -c) warmup + reps with
// telemetry → quality once per model → recommend. One plain async function; every failure becomes data.
import type { SessionEvent, SessionEventBody, SessionRequest } from '../../shared/bench-events'
import type {
  BenchmarkRunResult, CandidateConfig, CandidateInput, FailureKind, Metric, ModelMeta, QualityResult, Recommendation, RunStatus
} from '../../shared/bench-types'
import type { SystemProfile } from '../../shared/types'
import type { ExitInfo } from '../runtimes/llamacpp'
import type { LoadConfig, LoadResult, PromptRequest, PromptResult } from '../runtimes/types'
import type { Field, TelemetrySample } from '../telemetry/sampler'
import { peaks } from '../telemetry/sampler'
import { buildQualityPrompts, defaultTestSet, evaluateAsync } from '../quality'
import { detectCliffs, isUsable, val } from '../scoring/cliff'
import { recommend } from '../scoring/recommend'
import { DEFAULT_SCORING_CONFIG } from '../scoring/workloads'
import { estimateMemory, generateCandidates, machineFromProfile, rulesForRequest } from './candidates'
import { LADDER_PREDICT, PROMPT_VERSION, ladderPrompt } from './prompts'

/** Bump when the runner's measurement procedure changes (warmup, reps, reduction, timeouts). */
export const BENCHMARK_VERSION = 'bench-1.0.0'

const GiB = 1024 ** 3
type Awaitable<T> = T | Promise<T>

/** What the runner needs from a runtime. LlamaCppBackend satisfies it. */
export interface SessionBackend {
  readonly pid: number | undefined
  readonly lastExit: ExitInfo | null
  loadModel(cfg: LoadConfig): Promise<LoadResult>
  unloadModel(): Promise<void>
  warmup(prompt: string): Promise<void>
  runPrompt(req: PromptRequest): Promise<PromptResult>
  /** Model chat template → raw prompt (llama-server POST /apply-template). */
  applyTemplate(messages: { role: string; content: string }[], opts?: { templateKwargs?: Record<string, unknown> }): Promise<string>
  cancel(): Promise<void>
}

export interface SessionSampler {
  readonly samples: TelemetrySample[]
  readonly unavailable: Partial<Record<Field, string>>
  stop(): TelemetrySample[]
}

export interface RunDetail {
  samples: TelemetrySample[]
  /** Why it failed / was flagged; null on a clean pass. */
  reason: string | null
  stderrTail: string[]
  load: LoadResult | null
  startedAt: number
  endedAt: number
}

/** #2 implements this against db.ts. Resume reads back what save* wrote. */
export interface SessionStorage {
  createSession(s: { workload: SessionRequest['workload']; request: SessionRequest; startedAt: number }): Awaitable<string>
  setSessionStatus(sessionId: string, status: 'running' | 'done' | 'cancelled' | 'paused' | 'failed', error?: string): Awaitable<void>
  listRuns(sessionId: string): Awaitable<BenchmarkRunResult[]>
  saveRun(sessionId: string, run: BenchmarkRunResult, detail: RunDetail): Awaitable<void>
  listQuality(sessionId: string, modelId: string): Awaitable<QualityResult[]>
  saveQuality(sessionId: string, modelId: string, configId: string, ctx: number, results: QualityResult[]): Awaitable<void>
  saveRecommendation(sessionId: string, rec: Recommendation): Awaitable<void>
}

export const DEFAULT_SESSION_CONFIG = {
  reps: 2,
  predictTokens: LADDER_PREDICT,
  promptTimeoutBaseMs: 60_000,
  promptTimeoutPerCtxMs: 10, // + 10 ms per context token: 64K → ~12 min cap for a slow partial offload
  qualityTimeoutMs: 180_000,
  ramFloorMinBytes: 2 * GiB, // DESIGN §3.3: floor = max(2 GiB, 8% of RAM)
  ramFloorFraction: 0.08,
  sharedSpillAbortBytes: 2 * GiB, // per-PID shared GPU memory: kill the run (DESIGN §3.3)
  maxConsecutiveDegraded: 2,
  guardPollMs: 1000,
  /** typeperf needs ~2 s for its first row; a 0.5B step can finish in ~1.5 s. After the reps, wait (real time) up to
   *  this long from sampler start for one real row so memory peaks aren't lost. Never fabricated: still 0 → unavailable. */
  firstSampleWaitMs: 3000,
  qualityFillerMax: 3000
}
export type SessionConfig = typeof DEFAULT_SESSION_CONFIG

export interface SessionDeps {
  backend: () => SessionBackend
  startSampler: (pid: number) => SessionSampler
  storage: SessionStorage
  machine: SystemProfile
  /** Runtime device of the benchmark GPU (from listDevices), e.g. 'Vulkan0'; null = CPU only. */
  gpuDevice: string | null
  backendKind?: 'vulkan' | 'cuda' | 'cpu'
  /** From the runtime's detect() (e.g. llama.cpp build); stored on every run (X17). */
  runtimeVersion?: string | null
  /** ModelMeta.id is the absolute GGUF path (ModelInfo convention); it is passed to loadModel. */
  models: ModelMeta[]
  clock: { now(): number }
  /** Live system RAM available, re-read before every step (F10). Falls back to the scan value. */
  readRamAvailableBytes?: () => number | null
  evaluate?: typeof evaluateAsync
  signal?: AbortSignal
  /** Resume: the stored candidate plan, used verbatim instead of generateCandidates, so configIds (and heavy-mode
   *  ngl values, which depend on the KV estimate) can't drift across code or GGUF-parser changes. Per-step live RAM
   *  pre-checks still apply. */
  plan?: CandidateConfig[]
  /** Pause: checked between steps only — the current step finishes (never mid-request), then the session unloads,
   *  persists and ends with status 'paused'. Resume it like any other session (resumeSessionId). */
  pauseSignal?: AbortSignal
  config?: Partial<SessionConfig>
}

const measured = (v: number | null | undefined, source: string, reason = 'not observed'): Metric =>
  v == null || !Number.isFinite(v) ? { value: null, kind: 'unavailable', reason } : { value: v, kind: 'measured', source }

function median(xs: (number | null)[]): number | null {
  const v = xs.filter((x): x is number => x != null && Number.isFinite(x)).sort((a, b) => a - b)
  if (!v.length) return null
  const m = v.length >> 1
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2
}

const failKind = (exit: ExitInfo | null): FailureKind | null =>
  exit ? (exit.reason === 'oom' ? 'oom' : exit.reason === 'device_lost' ? 'device_lost' : 'crash') : null

export async function runSession(req: SessionRequest, deps: SessionDeps, emit: (e: SessionEvent) => void): Promise<Recommendation | null> {
  const cfg = { ...DEFAULT_SESSION_CONFIG, ...deps.config }
  const rules = rulesForRequest(req)
  const profile = DEFAULT_SCORING_CONFIG.profiles[req.workload]
  const { storage, clock, signal } = deps
  const machine = machineFromProfile(deps.machine, deps.gpuDevice)
  const vramTotal = val(machine.vramBytes, true)
  const ramTotal = val(machine.ramTotalBytes, true)
  // Heavy mode runs close to the RAM limit on purpose: floor = the 2 GiB minimum, never lower.
  const ramFloor = req.heavyMode ? cfg.ramFloorMinBytes : Math.max(cfg.ramFloorMinBytes, (ramTotal ?? 0) * cfg.ramFloorFraction)
  const evaluate = deps.evaluate ?? evaluateAsync

  let sessionId = req.resumeSessionId ?? ''
  const send = (e: SessionEventBody) => emit({ sessionId, ...e })
  const log = (level: 'info' | 'warn' | 'error', msg: string) => send({ type: 'log', level, msg })
  const backend = deps.backend()
  const onAbort = () => { void backend.cancel() }
  signal?.addEventListener('abort', onAbort)

  try {
    if (!sessionId) sessionId = await storage.createSession({ workload: req.workload, request: req, startedAt: clock.now() })
    await storage.setSessionStatus(sessionId, 'running')
    // Resume re-runs steps that never really ran: cancelled ones and RAM-guard skips (memory may be free now).
    // retryFailed also re-runs fail/timeout steps (not config_drift: same config → same drift; a changed config has a
    // new configId anyway). rerunConfigIds re-runs every step of those configs. Later rows win in storage.
    const rerunIds = new Set(req.rerunConfigIds ?? [])
    const rerun = (r: BenchmarkRunResult) =>
      r.status === 'cancelled' || r.failureKind === 'skipped_memory' || rerunIds.has(r.configId) ||
      (!!req.retryFailed && (r.status === 'fail' || r.status === 'timeout') && r.failureKind !== 'config_drift')
    const done = new Map((req.resumeSessionId ? await storage.listRuns(sessionId) : []).filter((r) => !rerun(r)).map((r) => [`${r.configId}@${r.ctx}`, r] as const))

    // All candidates of all requested models, smallest estimated footprint first.
    const plan: { cand: CandidateConfig; model: ModelMeta }[] = []
    for (const id of req.modelIds) {
      const model = deps.models.find((m) => m.id === id)
      if (!model) { log('warn', `model ${id} not found; skipped`); continue }
      if (deps.plan) {
        for (const cand of deps.plan.filter((c) => c.modelId === model.id)) plan.push({ cand, model })
        continue
      }
      const set = generateCandidates(machine, model, { backend: deps.backendKind ?? 'vulkan' }, profile, rules)
      for (const r of set.rejected) log('info', `rejected ${r.id}: ${r.reason}`)
      for (const cand of set.candidates) plan.push({ cand, model })
    }
    const est = (c: CandidateConfig) => (val(c.estVramBytes) ?? 0) + (val(c.estRamBytes) ?? 0)
    plan.sort((a, b) => est(a.cand) - est(b.cand) || (a.cand.id < b.cand.id ? -1 : 1))
    send({ type: 'session:started', workload: req.workload, modelIds: req.modelIds, resumed: !!req.resumeSessionId, candidates: plan.length })

    const inputs: CandidateInput[] = []
    const quality = new Map<string, QualityResult[]>() // modelId → results
    let gpuLost = false
    const paused = () => !signal?.aborted && !!deps.pauseSignal?.aborted

    for (const { cand, model } of plan) {
      if (signal?.aborted || paused()) break
      send({ type: 'candidate:started', configId: cand.id, model: model.id, gpuLayers: cand.gpuLayers, ctxSteps: cand.ctxSteps })
      if (gpuLost && cand.gpuLayers > 0) {
        send({ type: 'candidate:done', configId: cand.id, status: 'skipped', reason: 'GPU device was lost earlier in this session' })
        continue
      }
      for (const s of cand.skippedSteps) log('info', `${cand.id} @${s.ctx}: skipped (${s.reason})`)
      const steps = req.ladder ? cand.ctxSteps.filter((c) => req.ladder!.includes(c)) : cand.ctxSteps
      const runs: BenchmarkRunResult[] = []
      let degradedRun = 0
      let stopReason: string | null = null

      send({ type: 'phase', configId: cand.id, ctx: steps[0] ?? 0, phase: 'ladder' })
      for (const ctx of steps) {
        if (signal?.aborted || paused()) break
        let run = done.get(`${cand.id}@${ctx}`)
        if (run) {
          log('info', `${cand.id} @${ctx}: already measured in this session; reused`)
        } else {
          const out = await runStep(cand, model, ctx)
          run = out.run
          await storage.saveRun(sessionId, run, out.detail)
          if (run.failureKind === 'device_lost') gpuLost = true
          if (out.detail.reason) log(run.status === 'pass' ? 'warn' : 'error', `${cand.id} @${ctx}: ${out.detail.reason}`)
        }
        runs.push(run)
        const verdict = detectCliffs(runs, vramTotal).steps.find((s) => s.ctx === ctx)!.verdict
        send({ type: 'step:done', configId: cand.id, ctx, result: run, verdict })
        if (verdict === 'fail') { stopReason = `stopped after ${run.status}${run.failureKind ? ` (${run.failureKind})` : ''} at ${ctx}`; break }
        degradedRun = verdict === 'degraded' ? degradedRun + 1 : 0
        if (degradedRun >= cfg.maxConsecutiveDegraded) { stopReason = `stopped after ${degradedRun} consecutive degraded steps`; break }
      }

      const anyUsable = runs.some(isUsable)
      await backend.unloadModel().catch((e) => log('error', `unload failed: ${(e as Error).message}`))
      inputs.push({ config: cand, model, runs, quality: [] })
      const status = signal?.aborted ? 'cancelled' : paused() ? 'paused' : anyUsable ? 'done' : 'failed'
      send({ type: 'candidate:done', configId: cand.id, status, reason: stopReason })
    }

    // Quality once per model, after all ladders, on its best-offload usable candidate (most GPU layers, then fastest
    // decode) — never on a CPU baseline or -nkvo probe just because the plan sorted it first.
    if (req.runQuality !== false) {
      for (const modelId of [...new Set(inputs.map((i) => i.model.id))]) {
        if (signal?.aborted || paused()) break
        const decode = (i: CandidateInput) => Math.max(0, ...i.runs.filter(isUsable).map((r) => val(r.decodeTps, true) ?? 0))
        const best = inputs.filter((i) => i.model.id === modelId && i.runs.some(isUsable))
          .sort((a, b) => b.config.gpuLayers - a.config.gpuLayers || decode(b) - decode(a) || (a.config.id < b.config.id ? -1 : 1))[0]
        if (!best) continue
        const stored = req.resumeSessionId ? await storage.listQuality(sessionId, modelId) : []
        if (stored.length) { quality.set(modelId, stored); continue }
        const cliff = detectCliffs(best.runs, vramTotal)
        const usable = best.runs.filter(isUsable).map((r) => r.ctx)
        const qctx = Math.min(profile.targetContext, val(cliff.practicalContextCeiling) ?? Math.max(...usable))
        const results = await runQuality(best.config, best.model, qctx)
        await backend.unloadModel().catch((e) => log('error', `unload failed: ${(e as Error).message}`))
        if (results.length) {
          quality.set(modelId, results)
          await storage.saveQuality(sessionId, modelId, best.config.id, qctx, results)
        }
      }
    }

    if (signal?.aborted) {
      await storage.setSessionStatus(sessionId, 'cancelled')
      send({ type: 'session:cancelled' })
      return null
    }
    if (paused()) {
      await storage.setSessionStatus(sessionId, 'paused')
      send({ type: 'session:paused' })
      return null
    }
    for (const i of inputs) i.quality = quality.get(i.model.id) ?? []
    const rec = recommend(inputs, machine, req.workload)
    await storage.saveRecommendation(sessionId, rec)
    await storage.setSessionStatus(sessionId, 'done')
    send({ type: 'session:done', recommendation: rec })
    return rec
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e)
    if (sessionId) await Promise.resolve(storage.setSessionStatus(sessionId, 'failed', msg)).catch(() => {})
    send({ type: 'session:failed', error: msg })
    return null
  } finally {
    signal?.removeEventListener('abort', onAbort)
    await backend.unloadModel().catch(() => {})
  }

  // --- helpers (closures over deps/cfg) ---

  function loadCfg(cand: CandidateConfig, model: ModelMeta, ctx: number): LoadConfig {
    return {
      modelPath: model.id, contextSize: ctx, gpuLayers: cand.gpuLayersAll ? 999 : cand.gpuLayers, device: cand.device ?? 'none',
      threads: cand.threads, batchSize: 2048,
      extraArgs: ['-ub', String(rules.ubatch), '-fa', cand.flashAttn ? 'on' : 'off', ...(cand.kvType === 'f16' ? [] : ['-ctk', cand.kvType, '-ctv', cand.kvType]),
        ...(cand.kvOffload === false ? ['-nkvo'] : [])] // -nkvo / --no-kv-offload: KV cache in RAM (b11208 --help)
    }
  }

  async function runStep(cand: CandidateConfig, model: ModelMeta, ctx: number): Promise<{ run: BenchmarkRunResult; detail: RunDetail }> {
    const startedAt = clock.now()
    send({ type: 'step:started', configId: cand.id, ctx })
    const na = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
    const result = (status: RunStatus, failureKind: FailureKind | null, reason: string | null, extra: Partial<BenchmarkRunResult> = {}, detail: Partial<RunDetail> = {}) => ({
      run: {
        configId: cand.id, ctx, promptTokens: null, status, failureKind, reason,
        loadTimeMs: na('not loaded'), ttftMs: na('no request'), prefillTps: na('no request'), decodeTps: na('no request'), totalMs: na('no request'),
        peakVramBytes: na('no telemetry'), peakSharedGpuBytes: na('no telemetry'), peakRamBytes: na('no telemetry'),
        avgGpuUtil: na('no telemetry'), avgCpuUtil: na('no telemetry'), warm: false,
        versions: { benchmark: BENCHMARK_VERSION, prompts: PROMPT_VERSION, quality: defaultTestSet.suite, runtime: deps.runtimeVersion ?? null }, ...extra
      },
      detail: { samples: [], reason, stderrTail: backend.lastExit?.tail.slice(-50) ?? [], load: null, startedAt, endedAt: clock.now(), ...detail }
    })

    // A25 pre-check against live RAM (other apps come and go, F10). Unload the previous step's server FIRST: its
    // mmap'd weights still count against available RAM (14B 4K was skipped with 5.3 GiB "available" otherwise).
    // unloadModel waits for the process to exit (kill → taskkill /T /F) before returning.
    await backend.unloadModel().catch((e: Error) => log('warn', `unload before ${ctx}: ${e.message}`))
    const avail = deps.readRamAvailableBytes?.() ?? val(machine.ramAvailableBytes)
    const e = estimateMemory(model, cand.gpuLayersAll ? model.layers : cand.gpuLayers, ctx, cand.kvType, rules.ubatch, cand.kvOffload !== false)
    const need = req.heavyMode ? e.ramResidentBytes : e.ramBytes
    // mmap'd weights already uploaded to the GPU are clean file pages the OS can drop: they lower 'available RAM'
    // (calibration: ≈ file size) without being memory pressure. Credit them in the in-step floor check.
    const mmapCredit = (model.fileBytes * Math.min(cand.gpuLayersAll ? model.layers : cand.gpuLayers, model.layers)) / model.layers
    if (avail !== null && need > avail - ramFloor) {
      return result('fail', 'skipped_memory', `est. RAM ${(need / GiB).toFixed(1)} GiB > available ${(avail / GiB).toFixed(1)} GiB − floor ${(ramFloor / GiB).toFixed(1)} GiB`)
    }

    send({ type: 'phase', configId: cand.id, ctx, phase: 'load' })
    // Start the sampler as soon as the NEW server's pid exists (during load), not after /health: typeperf's ~2 s
    // start-up then overlaps the load. Per-step (not per-candidate) because each step restarts the server and
    // typeperf's per-PID counter set is fixed at start.
    const oldPid = backend.pid
    let sampler: SessionSampler | null = null
    let samplerAt = 0
    const startIfNew = () => {
      if (!sampler && backend.pid !== undefined && backend.pid !== oldPid) { sampler = deps.startSampler(backend.pid); samplerAt = Date.now() }
    }
    const pidPoll = setInterval(startIfNew, 50)
    let load: LoadResult
    try {
      load = await backend.loadModel(loadCfg(cand, model, ctx))
    } catch (e) {
      clearInterval(pidPoll)
      ;(sampler as SessionSampler | null)?.stop()
      const msg = (e as Error).message
      const drift = (e as { failureKind?: string }).failureKind === 'config_drift' // ConfigDriftError (runtimes/llamacpp)
      const kind = signal?.aborted ? null : drift ? 'config_drift' : failKind(backend.lastExit) ?? (/healthy within/.test(msg) ? 'load_timeout' : 'load_fail')
      return result(signal?.aborted ? 'cancelled' : kind === 'load_timeout' ? 'timeout' : 'fail', kind, msg)
    }
    clearInterval(pidPoll)
    if (!sampler && backend.pid !== undefined) { sampler = deps.startSampler(backend.pid); samplerAt = Date.now() }
    const loadDoneAt = Date.now()
    const smp = sampler as SessionSampler | null

    let emitted = 0
    let guard: string | null = null
    const flush = () => {
      const xs = smp?.samples ?? []
      for (; emitted < xs.length; emitted++) {
        const s = xs[emitted]
        send({ type: 'telemetry', configId: cand.id, ctx, sample: s })
        if (!guard && s.ramAvailBytes != null && s.ramAvailBytes + mmapCredit < ramFloor) guard = `RAM available ${(s.ramAvailBytes / GiB).toFixed(1)} GiB fell below the floor`
        if (!guard && s.procVramSharedBytes != null && s.procVramSharedBytes > cfg.sharedSpillAbortBytes) guard = `shared GPU memory spill ${(s.procVramSharedBytes / GiB).toFixed(1)} GiB exceeded the abort limit`
        if (guard) void backend.cancel()
      }
    }
    // A guard's cancel() makes the in-flight request fail with "cancelled": the guard reason must win over that.
    const fail = (error: string, timedOut: boolean) => {
      flush()
      return guard ? { status: 'fail' as const, kind: 'guard_abort' as const, reason: guard } : classify(error, timedOut)
    }
    const timer = setInterval(flush, cfg.guardPollMs)
    const prompt = ladderPrompt(ctx)
    const timeoutMs = cfg.promptTimeoutBaseMs + ctx * cfg.promptTimeoutPerCtxMs
    const reps: PromptResult[] = []
    const windows: [number, number][] = []
    let warm = false
    let failure: { status: RunStatus; kind: FailureKind | null; reason: string } | null = null
    try {
      send({ type: 'phase', configId: cand.id, ctx, phase: 'warmup' })
      // Size-matched warmup: compiles the pipelines for this batch shape (F7).
      const w0 = Date.now()
      await backend.warmup(prompt).then(() => { warm = true }, (e: Error) => { failure = fail(e.message, false) })
      windows.push([w0, Date.now()])
      send({ type: 'phase', configId: cand.id, ctx, phase: 'measure' })
      for (let i = 0; i < cfg.reps && !failure && !signal?.aborted; i++) {
        const r0 = Date.now()
        const r = await backend.runPrompt({ prompt, maxTokens: cfg.predictTokens, temperature: 0, seed: 1, timeoutMs })
        windows.push([r0, Date.now()])
        flush()
        if (r.error) { failure = fail(r.error, r.timedOut); break }
        reps.push(r)
        send({ type: 'token-rate', configId: cand.id, ctx, prefillTps: r.prefillTps, decodeTps: r.decodeTps, ttftMs: r.ttftMs })
      }
    } finally {
      clearInterval(timer)
    }
    const measureEndAt = Date.now()
    if (smp && !failure && !smp.samples.length) {
      while (!smp.samples.length && Date.now() - samplerAt < cfg.firstSampleWaitMs) await new Promise((r) => setTimeout(r, 100))
    }
    const samples = smp?.stop() ?? []
    flush()
    if (!failure && guard) failure = { status: 'fail', kind: 'guard_abort', reason: guard }
    if (!failure && signal?.aborted) failure = { status: 'cancelled', kind: null, reason: 'cancelled by user' }
    const f = failure as { status: RunStatus; kind: FailureKind | null; reason: string } | null

    // Peaks over every sample (load included: memory stays allocated); means only over warmup+measure (X10).
    const pk = peaks(samples)
    // Means only over rows that cover a request: a typeperf row at ts averages (ts − interval, ts], so keep rows
    // arriving during a request or within one interval after it. Idle gaps between requests are excluded (D9).
    const inWindow = (t: number) => windows.some(([a, b]) => t > a && t <= b + cfg.guardPollMs)
    const win = peaks(samples.filter((x) => x.ts >= loadDoneAt && x.ts <= measureEndAt + cfg.guardPollMs && inWindow(x.ts)))
    const tele = (v: number | null, field: Field, n = pk.n): Metric =>
      n === 0 ? { value: null, kind: 'unavailable', reason: 'no telemetry samples in the window (run too short or sampler failed)' }
        : measured(v, 'typeperf', smp?.unavailable[field] ?? 'counter reported nothing')
    const prefill = median(reps.map((r) => r.prefillTps))
    const decode = median(reps.map((r) => r.decodeTps))
    const ttft = median(reps.map((r) => r.ttftMs))
    // A12: without runtime timings, derive from wall clock and token counts, labelled estimated.
    const estDecode = median(reps.map((r) => (r.decodeTokens && r.ttftMs != null && r.totalMs > r.ttftMs ? (r.decodeTokens * 1000) / (r.totalMs - r.ttftMs) : null)))
    const estPrefill = median(reps.map((r) => (r.promptTokens && r.ttftMs ? (r.promptTokens * 1000) / r.ttftMs : null)))
    const tps = (v: number | null, e: number | null, what: string): Metric =>
      v != null ? { value: v, kind: 'measured', source: 'llama-server timings' }
        : e != null ? { value: e, kind: 'estimated', source: `${what} tokens / wall clock` } : { value: null, kind: 'unavailable', reason: 'no timings' }

    return result(f?.status ?? 'pass', f?.kind ?? null, f?.reason ?? null, {
      warm,
      promptTokens: median(reps.map((r) => r.promptTokens)),
      loadTimeMs: measured(load.loadTimeMs, 'spawn → /health ok'),
      ttftMs: measured(ttft, 'client wall clock, request → first token', 'no successful request'),
      prefillTps: tps(prefill, estPrefill, 'prompt'),
      decodeTps: tps(decode, estDecode, 'predicted'),
      totalMs: measured(median(reps.map((r) => r.totalMs)), 'client wall clock', 'no successful request'),
      peakVramBytes: tele(pk.max.procVramDedicatedBytes, 'procVramDedicatedBytes'),
      peakSharedGpuBytes: tele(pk.max.procVramSharedBytes, 'procVramSharedBytes'),
      peakRamBytes: tele(pk.max.procRamPrivateBytes, 'procRamPrivateBytes'),
      avgGpuUtil: tele(win.meanGpuUtilPct, 'gpuUtilPct', win.n),
      avgCpuUtil: tele(win.meanCpuPct, 'cpuPct', win.n)
    }, { samples, load })
  }

  function classify(error: string, timedOut: boolean): { status: RunStatus; kind: FailureKind | null; reason: string } {
    const exit = backend.lastExit
    if (exit) return { status: 'fail', kind: failKind(exit), reason: `${error} (exit code ${exit.code})` }
    if (timedOut) return { status: 'timeout', kind: 'req_timeout', reason: error }
    if (signal?.aborted) return { status: 'cancelled', kind: null, reason: 'cancelled by user' }
    return { status: 'fail', kind: 'request_error', reason: error }
  }

  async function runQuality(cand: CandidateConfig, model: ModelMeta, ctx: number): Promise<QualityResult[]> {
    send({ type: 'phase', configId: cand.id, ctx, phase: 'quality' })
    try {
      await backend.loadModel(loadCfg(cand, model, ctx))
    } catch (e) {
      log('error', `${cand.id}: quality load at ${ctx} failed: ${(e as Error).message}`)
      return []
    }
    // Thinking models (chat template has enable_thinking): quality runs with thinking OFF — deterministic, fast, and the
    // suite's max_tokens fit. The ×4 token boost (buildQualityPrompts thinking:true) is kept for a future "thinking on".
    const templateKwargs = model.supportsThinking ? { enable_thinking: false } : undefined
    if (templateKwargs) log('info', `${model.name}: quality suite runs with thinking disabled (enable_thinking=false)`)
    const prompts = buildQualityPrompts(defaultTestSet, { fillerTokens: Math.min(cfg.qualityFillerMax, Math.floor(ctx * 0.6)), thinking: false })
    const out: QualityResult[] = []
    for (const p of prompts) {
      if (signal?.aborted || backend.lastExit) break
      const test = defaultTestSet.tests.find((t) => t.id === p.testId)!
      try {
        const prompt = await backend.applyTemplate(p.messages, templateKwargs ? { templateKwargs } : undefined)
        const r = await backend.runPrompt({ prompt, maxTokens: p.maxTokens, temperature: p.temperature, seed: p.seed, timeoutMs: cfg.qualityTimeoutMs })
        out.push(r.error
          ? { testId: test.id, category: test.category, weight: test.weight, pass: false, score: 0, detail: `request failed: ${r.error}` }
          : await evaluate(test, r.text))
      } catch (e) {
        out.push({ testId: test.id, category: test.category, weight: test.weight, pass: false, score: 0, detail: `request failed: ${(e as Error).message}` })
      }
    }
    // Partial suites (cancel/crash) are not persisted: a missing category must not look like a measured 0.
    if (out.length < prompts.length) { log('warn', `${cand.id}: quality suite incomplete (${out.length}/${prompts.length}); discarded`); return [] }
    return out
  }
}
