// Benchmark session runner (DESIGN §3): candidates → per context step (restart with -c) warmup + reps with
// telemetry → quality once per model → recommend. One plain async function; every failure becomes data.
import type { SessionEvent, SessionEventBody, SessionRequest } from '../../shared/bench-events'
import type {
  BenchmarkRunResult, CandidateConfig, CandidateInput, FailureKind, GenConfig, GenQuality, Metric, ModelMeta, QualityResult, Recommendation, RunStatus
} from '../../shared/bench-types'
import type { SystemProfile } from '../../shared/types'
import type { ExitInfo } from '../runtimes/llamacpp'
import type { LoadConfig, LoadResult, PromptRequest, PromptResult } from '../runtimes/types'
import type { Field, TelemetrySample } from '../telemetry/sampler'
import { peaks } from '../telemetry/sampler'
import { buildQualityPrompts, defaultTestSet, evaluateAsync, needlePrompt, type QualityTest } from '../quality'
import { detectCliffs, fmtCtx, isUsable, val } from '../scoring/cliff'
import { recommend } from '../scoring/recommend'
import { DEFAULT_SCORING_CONFIG, effectiveProfile, withProfile } from '../scoring/workloads'
import { DEFAULT_CANDIDATE_RULES, estimateMemory, generateCandidates, machineFromProfile, rulesForRequest, type CandidateRules } from './candidates'
import { LADDER_FILL, LADDER_PREDICT, PROMPT_VERSION, ladderPrompt } from './prompts'
import { RULES_VERSION } from '../interpret'
import { BASELINE_GEN, genConfigsFor, genLabel, samplingFor, splitReasoning, summarizeGen, templateKwargsFor, type GenRow } from './gen'

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
  /** null until typeperf printed its header; false = the GPU Process Memory(pid_*) columns are missing because
   *  typeperf fixed its instance set before llama-server created them → restart() once the model is loaded. */
  readonly hasPidColumns?: boolean | null
  /** Respawn with the same counters; keeps samples and errors. */
  restart?(): void
  readonly errors?: string[]
}

export interface RunDetail {
  samples: TelemetrySample[]
  /** Why it failed / was flagged; null on a clean pass. */
  reason: string | null
  stderrTail: string[]
  load: LoadResult | null
  startedAt: number
  endedAt: number
  /** typeperf failures / restarts during this step (e.g. "typeperf restarted after N samples"). */
  samplerErrors?: string[]
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
  /** Thinking gen configs: maxTokens ×4 (at least this), timeout ×2 per test. */
  thinkingMinTokens: 1024,
  /** qualityMode 'thorough': seeded samples per test for stochastic (T > 0) gen configs. */
  thoroughSamples: 3,
  // Floor = max(4 GiB, 8 % of RAM), heavy mode included: a 27B CPU baseline drove available RAM to 1.0 GiB with a
  // 2 GiB floor before the 1 s guard fired, and the host went into memory pressure (2026-09-27 heavy run).
  ramFloorMinBytes: 4 * GiB,
  ramFloorFraction: 0.08,
  sharedSpillAbortBytes: 2 * GiB, // per-PID shared GPU memory: kill the run (DESIGN §3.3)
  maxConsecutiveDegraded: 2,
  guardPollMs: 1000,
  /** Consecutive guard polls with no RAM reading at all (OS nor typeperf) before the step is aborted as unsafe. */
  guardBlindPollsMax: 3,
  /** Heavy (expectDegraded) configs: poll the guard every 250 ms from the start of load — mmap RAM ramps fast. */
  heavyGuardPollMs: 250,
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

const GPU_DEVICE = /^(Vulkan|CUDA|ROCm|SYCL|Metal|MTL)\d+$/
/** The exact LoadConfig the runner launches a step with (pure; export-equivalence tests compare it with
 *  core/export toLlamaServerArgs). LlamaCppBackend.loadModel adds the fixed -fit off, --parallel 1, host/port and log
 *  flags itself. */
export function loadConfigFor(cand: CandidateConfig, model: ModelMeta, ctx: number, rules: CandidateRules = DEFAULT_CANDIDATE_RULES): LoadConfig {
  return {
    modelPath: model.id, contextSize: ctx, gpuLayers: cand.gpuLayersAll ? 999 : cand.gpuLayers, device: cand.device ?? 'none',
    threads: cand.threads, batchSize: 2048,
    extraArgs: ['-ub', String(rules.ubatch), '-fa', cand.flashAttn ? 'on' : 'off', ...(cand.kvType === 'f16' ? [] : ['-ctk', cand.kvType, '-ctv', cand.kvType]),
      ...(cand.kvOffload === false ? ['-nkvo'] : []), // -nkvo / --no-kv-offload: KV cache in RAM (b11208 --help)
      ...(cand.mmap === false ? ['-lm', 'none'] : [])] // --load-mode none: no mmap (b11208 --help)
  }
}

/** Host-side buffers from the load log. WDDM reports pinned host memory as the process's "shared GPU memory":
 *  Qwen3.8 ngl 50 under -lm none showed 3.82 GiB shared with 4 GiB of VRAM free (≈ 15/65 of the weights).
 *  CPU_Mapped is a file mapping under mmap (not shared: 8B showed 282 MiB CPU_Mapped, 0.02 GiB shared). */
export function hostPinnedBytes(d: LoadResult['declared'], mmap: boolean): number {
  let mib = 0
  for (const map of [d.modelBufferMiB, d.kvBufferMiB, d.computeBufferMiB]) {
    for (const [dev, v] of Object.entries(map ?? {})) if (!GPU_DEVICE.test(dev) && !(mmap && dev === 'CPU_Mapped')) mib += v
  }
  return mib * 1024 ** 2
}

const failKind = (exit: ExitInfo | null): FailureKind | null =>
  exit ? (exit.reason === 'oom' ? 'oom' : exit.reason === 'device_lost' ? 'device_lost' : 'crash') : null

export async function runSession(req: SessionRequest, deps: SessionDeps, emit: (e: SessionEvent) => void): Promise<Recommendation | null> {
  const cfg = { ...DEFAULT_SESSION_CONFIG, ...deps.config }
  const rules = rulesForRequest(req)
  // requiredContext / minDecodeTps (explicit user choices) reshape the profile; the same object drives planning,
  // quality ctx and recommend().
  const profile = effectiveProfile(DEFAULT_SCORING_CONFIG.profiles[req.workload], req)
  const scoringCfg = withProfile(DEFAULT_SCORING_CONFIG, profile)
  const required = req.requiredContext ?? null
  const { storage, clock, signal } = deps
  const machine = machineFromProfile(deps.machine, deps.gpuDevice)
  const vramTotal = val(machine.vramBytes, true)
  // L1: the saturation share for spill is taken against what this process could get — VRAM total minus what other
  // processes held at planning (measured), not the adapter total.
  const vramEffective = vramTotal === null ? null : vramTotal - (val(machine.vramInUseBytes) ?? 0)
  const ramTotal = val(machine.ramTotalBytes, true)
  // Heavy mode runs close to the RAM limit on purpose: floor = the 2 GiB minimum, never lower.
  const ramFloor = Math.max(cfg.ramFloorMinBytes, (ramTotal ?? 0) * cfg.ramFloorFraction)
  const evaluate = deps.evaluate ?? evaluateAsync
  /** Live OS free RAM; null when not provided or unreadable. */
  const osRam = (): number | null => {
    try { const v = deps.readRamAvailableBytes?.(); return typeof v === 'number' && Number.isFinite(v) ? v : null } catch { return null }
  }

  let sessionId = req.resumeSessionId ?? ''
  const send = (e: SessionEventBody) => emit({ sessionId, ...e })
  const log = (level: 'info' | 'warn' | 'error', msg: string) => send({ type: 'log', level, msg })
  const backend = deps.backend()
  // ServerStuckError (runtimes/llamacpp, `.fatal`): unload couldn't confirm the server exited. That is a hard stop —
  // no more candidates, and no later cleanup that would clear its pid file.
  let stuck: Error | null = null
  const unload = (onErr?: (e: Error) => void) => backend.unloadModel().catch((e: Error & { fatal?: boolean }) => {
    if (e?.fatal) stuck ??= e
    else onErr?.(e)
  })
  const checkStuck = () => { if (stuck) throw stuck }
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
    const unplanned: { model: string; reason: string }[] = []
    for (const id of req.modelIds) {
      const model = deps.models.find((m) => m.id === id)
      if (!model) { log('warn', `model ${id} not found; skipped`); continue }
      if (deps.plan) {
        for (const cand of deps.plan.filter((c) => c.modelId === model.id)) plan.push({ cand, model })
        continue
      }
      const set = generateCandidates(machine, model, { backend: deps.backendKind ?? 'vulkan' }, profile, rules)
      for (const r of set.rejected) log('info', `rejected ${r.id}: ${r.reason}`)
      if (!set.candidates.length) unplanned.push({ model: model.name, reason: [...new Set(set.rejected.map((r) => r.reason))].join('; ') || 'no candidate configuration' })
      for (const cand of set.candidates) plan.push({ cand, model })
    }
    const est = (c: CandidateConfig) => (val(c.estVramBytes) ?? 0) + (val(c.estRamBytes) ?? 0)
    // Full-offload configs first (cheapest estimate first). Heavy configs after them, most-offloaded first and CPU
    // baselines last, so quality/recommendation exist even if a later, riskier config aborts.
    const share = (c: CandidateConfig, m: ModelMeta) => (c.gpuLayersAll ? 1 : c.gpuLayers / m.layers)
    // Within heavy configs: KV-on-GPU rungs by layer share, then -nkvo (dominated at short ctx), then the CPU baseline.
    const heavyRank = (c: CandidateConfig) => (c.gpuLayers === 0 ? 2 : c.kvOffload === false ? 1 : 0)
    plan.sort((a, b) => Number(!!a.cand.expectDegraded) - Number(!!b.cand.expectDegraded) ||
      (a.cand.expectDegraded ? heavyRank(a.cand) - heavyRank(b.cand) || share(b.cand, b.model) - share(a.cand, a.model) : 0) ||
      est(a.cand) - est(b.cand) || (a.cand.id < b.cand.id ? -1 : 1))
    send({ type: 'session:started', workload: req.workload, modelIds: req.modelIds, resumed: !!req.resumeSessionId, candidates: plan.length })

    const inputs: CandidateInput[] = []
    const quality = new Map<string, QualityResult[]>() // modelId → baseline results
    const genQ = new Map<string, GenQuality[]>() // modelId → every gen config that completed
    const longNotes: string[] = []
    let gpuLost = false
    const paused = () => !signal?.aborted && !!deps.pauseSignal?.aborted

    const passRungs = (i: CandidateInput): number[] => {
      const c = detectCliffs(i.runs, vramTotal)
      const first = c.steps.findIndex((x) => x.verdict !== 'pass')
      return (first < 0 ? c.steps : c.steps.slice(0, first)).filter((x) => i.runs.some((r) => r.ctx === x.ctx && isUsable(r))).map((x) => x.ctx)
    }
    /** Quality once per model, as soon as its last planned config finished its ladder, on its best-offload passing config. */
    const qualityFor = async (modelId: string): Promise<void> => {
      const decode = (i: CandidateInput) => Math.max(0, ...i.runs.filter(isUsable).map((r) => val(r.decodeTps, true) ?? 0))
      // D09: only a config with a measured PASS rung, and never a CPU baseline or -nkvo probe unless it is the only one.
      const passing = inputs.filter((i) => i.model.id === modelId && passRungs(i).length)
      const gpuKv = passing.filter((i) => i.config.gpuLayers > 0 && i.config.kvOffload !== false)
      const pool = gpuKv.length ? gpuKv : passing
      const best = [...pool].sort((a, b) => b.config.gpuLayers - a.config.gpuLayers || decode(b) - decode(a) || (a.config.id < b.config.id ? -1 : 1))[0]
      if (!best) { log('warn', `${modelId}: no configuration passed a context rung; quality suite not run`); return }
      if (!gpuKv.length && machine.gpuDevice !== null) longNotes.push(`[I-3.5] Quality for ${best.model.name} ran on ${best.config.id} (CPU / KV-in-RAM): the only configuration that passed a context rung`)
      const gens = genConfigsFor(best.model, req)
      const samplesOf = (g: GenConfig) => (g.temperature > 0 && req.qualityMode !== 'quick' ? cfg.thoroughSamples : 1)
      // Reuse stored quality only when it is the COMPLETE current suite (every test id × sample for every gen config,
      // same suite version). Rows without genId are the baseline (stored before the gen-config search).
      const stored = (req.resumeSessionId ? await storage.listQuality(sessionId, modelId) : []) as GenRow[]
      const rowsOf = (g: GenConfig) => stored.filter((r) => (r.genId ?? BASELINE_GEN.id) === g.id)
      const complete = stored.length > 0 && stored.every((r) => (r as { suite?: string }).suite === defaultTestSet.suite) &&
        gens.every((g) => defaultTestSet.tests.every((t) => rowsOf(g).filter((r) => r.testId === t.id).length >= samplesOf(g)))
      if (complete) {
        const list = gens.map((g) => summarizeGen(g, rowsOf(g), samplesOf(g)))
        quality.set(modelId, list[0].results)
        if (gens.length > 1) genQ.set(modelId, list)
        return
      }
      if (stored.length) log('warn', `${modelId}: stored quality results are incomplete or from another suite version; re-running the suite`)
      const cliff = detectCliffs(best.runs, vramTotal)
      // D09: min(target, ceiling) snapped down to a rung that actually passed.
      const rungs = passRungs(best)
      const qctx = rungs.filter((c) => c <= profile.targetContext).at(-1) ?? rungs[0]
      const perGen = await runQuality(best.config, best.model, qctx, gens.map((g) => ({ gen: g, samples: samplesOf(g) })))
      const results = perGen[0]?.gen.id === BASELINE_GEN.id ? perGen[0].results : []
      await unload((e) => log('error', `unload failed: ${(e as Error).message}`))
      // One long-context retrieval test at the required ctx (a full prefill of it), on a config that reached it.
      if (required && required >= 32768 && results.length && !signal?.aborted && !paused()) {
        const reach = inputs.filter((i) => i.model.id === modelId && (val(detectCliffs(i.runs, vramTotal).practicalContextCeiling) ?? 0) >= required)
          .sort((a, b) => b.config.gpuLayers - a.config.gpuLayers || decode(b) - decode(a) || (a.config.id < b.config.id ? -1 : 1))[0]
        if (reach) {
          const r = await runLongNeedle(reach.config, reach.model, required)
          await unload()
          if (r) for (const g of perGen) g.results.push({ ...r, genId: g.gen.id, sample: 1 } as GenRow)
        } else {
          longNotes.push(`[I-2.5] Long-context needle CR-04-long at ${fmtCtx(required)} skipped for ${best.model.name}: practical context ${fmtCtx(val(cliff.practicalContextCeiling) ?? 0)} < ${fmtCtx(required)}`)
        }
      }
      if (results.length) {
        quality.set(modelId, results)
        if (gens.length > 1) genQ.set(modelId, perGen)
        // One transaction for every gen config of the model (resume needs all of them or re-runs).
        await storage.saveQuality(sessionId, modelId, best.config.id, qctx, perGen.flatMap((g) => g.results).map((r) => ({ ...r, suite: defaultTestSet.suite })))
      }
    }
    const withQuality = () => {
      for (const i of inputs) {
        i.quality = quality.get(i.model.id) ?? []
        const g = genQ.get(i.model.id)
        if (g) i.genQuality = g
      }
    }

    for (const [planIdx, { cand, model }] of plan.entries()) {
      checkStuck()
      if (signal?.aborted || paused()) break
      send({ type: 'candidate:started', configId: cand.id, model: model.id, gpuLayers: cand.gpuLayers, ctxSteps: cand.ctxSteps })
      if (gpuLost && cand.gpuLayers > 0) {
        send({ type: 'candidate:done', configId: cand.id, status: 'skipped', reason: 'GPU device was lost earlier in this session' })
        continue
      }
      for (const s of cand.skippedSteps) log('info', `${cand.id} @${s.ctx}: skipped (${s.reason})`)
      // An explicit required context wins over the UI ladder cap: run every rung up to it, nothing above it.
      // L2: an explicit ladder that reaches the required rung is a rung selection and is honoured; a lower ladder is only
      // the UI's size cap, which the required context overrides.
      const explicit = !!req.ladder && (!required || Math.max(...req.ladder) >= required)
      const steps = cand.ctxSteps.filter((c) => (!required || c <= required) && (!explicit || req.ladder!.includes(c)))
      const runs: BenchmarkRunResult[] = []
      let spillBase = 0 // shared level of the config's first step when VRAM was not saturated (not spill)
      let degradedRun = 0
      let stopReason: string | null = null

      send({ type: 'phase', configId: cand.id, ctx: steps[0] ?? 0, phase: 'ladder' })
      for (const ctx of steps) {
        checkStuck()
        if (signal?.aborted || paused()) break
        let run = done.get(`${cand.id}@${ctx}`)
        if (run) {
          log('info', `${cand.id} @${ctx}: already measured in this session; reused`)
        } else {
          const out = await runStep(cand, model, ctx, spillBase)
          checkStuck()
          run = out.run
          await storage.saveRun(sessionId, run, out.detail)
          if (run.failureKind === 'device_lost') gpuLost = true
          if (out.detail.reason) log(run.status === 'pass' ? 'warn' : 'error', `${cand.id} @${ctx}: ${out.detail.reason}`)
        }
        runs.push(run)
        if (runs.length === 1) {
          const raw = val(run.peakSharedGpuRawBytes) ?? val(run.peakSharedGpuBytes), pin = val(run.hostPinnedBytes) ?? 0, ded = val(run.peakVramBytes)
          if (raw !== null && (vramEffective === null || ded === null || ded < vramEffective * DEFAULT_SCORING_CONFIG.cliff.vramSaturation)) spillBase = Math.max(0, raw - pin)
        }
        const verdict = detectCliffs(runs, vramTotal).steps.find((s) => s.ctx === ctx)!.verdict
        send({ type: 'step:done', configId: cand.id, ctx, result: run, verdict })
        if (verdict === 'fail') { stopReason = `stopped after ${run.status}${run.failureKind ? ` (${run.failureKind})` : ''} at ${ctx}`; break }
        const sh = val(run.peakSharedGpuBytes)
        if (cand.expectDegraded && sh !== null && sh > cfg.sharedSpillAbortBytes) {
          stopReason = `spilled ${(sh / GiB).toFixed(2)} GiB into shared GPU memory at ${ctx}; next configuration`
          break
        }
        degradedRun = verdict === 'degraded' ? degradedRun + 1 : 0
        if (degradedRun >= cfg.maxConsecutiveDegraded && !(required && ctx < required)) { stopReason = `stopped after ${degradedRun} consecutive degraded steps`; break }
      }

      const anyUsable = runs.some(isUsable)
      await unload((e) => log('error', `unload failed: ${(e as Error).message}`))
      inputs.push({ config: cand, model, runs, quality: [] })
      const status = signal?.aborted ? 'cancelled' : paused() ? 'paused' : anyUsable ? 'done' : 'failed'
      send({ type: 'candidate:done', configId: cand.id, status, reason: stopReason })
      const rest = plan.slice(planIdx + 1)
      if (req.runQuality !== false && !rest.some((p) => p.model.id === model.id) && !signal?.aborted && !paused()) {
        checkStuck()
        await qualityFor(model.id)
        // D05: persist a provisional recommendation after each model, so a later cancel/abort still leaves one.
        if (rest.length && !signal?.aborted && !paused()) {
          withQuality()
          const pro = recommend(inputs, machine, req.workload, scoringCfg, unplanned, req)
          pro.provisional = true
          pro.reasons.unshift(`[I-1.1] Provisional: saved after ${model.name}; ${rest.length} configuration${rest.length > 1 ? 's' : ''} still to run`)
          await storage.saveRecommendation(sessionId, pro)
        }
      }
    }

    checkStuck()
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
    withQuality()
    const rec = recommend(inputs, machine, req.workload, scoringCfg, unplanned, req)
    rec.reasons.push(...longNotes)
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
    if (!stuck) await unload()
  }

  // --- helpers (closures over deps/cfg) ---

  // Cancel during load kills the server immediately (not after the 120 s /health deadline).
  function loadCfg(cand: CandidateConfig, model: ModelMeta, ctx: number): LoadConfig { return { ...loadConfigFor(cand, model, ctx, rules), signal } }

  async function runStep(cand: CandidateConfig, model: ModelMeta, ctx: number, spillBase = 0): Promise<{ run: BenchmarkRunResult; detail: RunDetail }> {
    let pinned = 0 // host-pinned bytes, known once the load log is parsed
    const startedAt = clock.now()
    send({ type: 'step:started', configId: cand.id, ctx })
    const na = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
    let sampler: SessionSampler | null = null
    const result = (status: RunStatus, failureKind: FailureKind | null, reason: string | null, extra: Partial<BenchmarkRunResult> = {}, detail: Partial<RunDetail> = {}) => ({
      run: {
        configId: cand.id, ctx, promptTokens: null, status, failureKind, reason,
        loadTimeMs: na('not loaded'), ttftMs: na('no request'), prefillTps: na('no request'), decodeTps: na('no request'), totalMs: na('no request'),
        peakVramBytes: na('no telemetry'), peakSharedGpuBytes: na('no telemetry'), peakRamBytes: na('no telemetry'),
        avgGpuUtil: na('no telemetry'), avgCpuUtil: na('no telemetry'), warm: false,
        versions: { benchmark: BENCHMARK_VERSION, prompts: PROMPT_VERSION, quality: defaultTestSet.suite, runtime: deps.runtimeVersion ?? null, rules: RULES_VERSION },
        ramFloorBytes: ramFloor, ...extra
      },
      detail: {
        samples: [], reason, stderrTail: backend.lastExit?.tail.slice(-50) ?? [], load: null, startedAt, endedAt: clock.now(),
        samplerErrors: [...((sampler as SessionSampler | null)?.errors ?? [])], ...detail
      }
    })

    // A25 pre-check against live RAM (other apps come and go, F10). Unload the previous step's server FIRST: its
    // mmap'd weights still count against available RAM (14B 4K was skipped with 5.3 GiB "available" otherwise).
    // unloadModel waits for the process to exit (kill → taskkill /T /F) before returning.
    await unload((e: Error) => log('warn', `unload before ${ctx}: ${e.message}`))
    checkStuck() // the previous server could not be confirmed dead: never start another one
    const avail = osRam() ?? val(machine.ramAvailableBytes)
    const e = estimateMemory(model, cand.gpuLayersAll ? model.layers : cand.gpuLayers, ctx, cand.kvType, rules.ubatch, cand.kvOffload !== false)
    const need = req.heavyMode ? e.ramResidentBytes : e.ramBytes
    // mmap'd weights already uploaded to the GPU are clean file pages the OS can drop: they lower 'available RAM'
    // (calibration: ≈ file size) without being memory pressure. Credit them in the in-step floor check.
    // Without mmap (heavy configs) there are no reclaimable file pages to credit.
    const mmapCredit = cand.mmap === false ? 0 : (model.fileBytes * Math.min(cand.gpuLayersAll ? model.layers : cand.gpuLayers, model.layers)) / model.layers
    const beforeLoad: Metric = avail === null ? { value: null, kind: 'unavailable', reason: 'no RAM reading before load' }
      : { value: avail, kind: osRam() !== null ? 'measured' : 'declared', source: 'RAM available before load (after the previous unload)' }
    if (avail !== null && need > avail - ramFloor) {
      return result('fail', 'skipped_memory', `est. RAM ${(need / GiB).toFixed(1)} GiB > available ${(avail / GiB).toFixed(1)} GiB − floor ${(ramFloor / GiB).toFixed(1)} GiB`,
        { ramAvailBeforeLoadBytes: beforeLoad, skip: { resource: 'ram', estimateBytes: need, budgetBytes: avail - ramFloor, ruleId: 'I-4.3' } })
    }

    send({ type: 'phase', configId: cand.id, ctx, phase: 'load' })
    // Start the sampler as soon as the NEW server's pid exists (during load), not after /health: typeperf's ~2 s
    // start-up then overlaps the load. Per-step (not per-candidate) because each step restarts the server and
    // typeperf's per-PID counter set is fixed at start.
    const oldPid = backend.pid
    let samplerAt = 0
    // typeperf fixes its instance set at start; if it started before llama-server created its GPU allocations the
    // pid-scoped columns are missing → restart it once, as soon as that is known (after load, or on a guard poll).
    let restarted = false
    const restartIfBlind = () => {
      const sm = sampler as SessionSampler | null
      if (!restarted && !loading && sm?.hasPidColumns === false && sm.restart) { restarted = true; sm.restart() }
    }
    const startIfNew = () => {
      if (!sampler && backend.pid !== undefined && backend.pid !== oldPid) { sampler = deps.startSampler(backend.pid); samplerAt = Date.now() }
    }
    const pidPoll = setInterval(startIfNew, 50)

    // The guard runs from the start of LOAD (mmap ramps RAM there), faster for heavy configs. During load a trip
    // kills the server (unloadModel); afterwards it cancels the in-flight request.
    let emitted = 0
    let guard: string | null = null
    let loading = true
    let blindPolls = 0
    let minRam: number | null = null // lowest RAM available seen by the guard (OS reading, else telemetry rows)
    const flush = () => {
      restartIfBlind()
      // Independent of typeperf: OS free RAM every poll (typeperf may be slow, localized, or not running at all).
      const os = osRam()
      if (os !== null) minRam = Math.min(minRam ?? os, os)
      if (!guard && os !== null && os + mmapCredit < ramFloor) {
        guard = `RAM available ${(os / GiB).toFixed(1)} GiB (OS) fell below the floor`
        void (loading ? unload() : backend.cancel())
      }
      // Fail safe: if neither the OS reading nor any typeperf RAM row is available, don't continue blind.
      const ramRows = ((sampler as SessionSampler | null)?.samples ?? []).some((x) => x.ramAvailBytes != null)
      blindPolls = deps.readRamAvailableBytes && os === null && !ramRows ? blindPolls + 1 : 0
      if (!guard && blindPolls >= cfg.guardBlindPollsMax) {
        guard = 'RAM guard inputs unreadable (no OS reading and no telemetry rows); stopped rather than continue blind'
        void (loading ? unload() : backend.cancel())
      }
      const xs = (sampler as SessionSampler | null)?.samples ?? []
      for (; emitted < xs.length; emitted++) {
        const s = xs[emitted]
        send({ type: 'telemetry', configId: cand.id, ctx, sample: s })
        if (s.ramAvailBytes != null) minRam = Math.min(minRam ?? s.ramAvailBytes, s.ramAvailBytes)
        // Heavy (partial-offload) configs: a shared-memory spill is a measurement (degraded + spill reason, then the
        // ladder moves on), not an abort. The RAM floor always aborts.
        const spillAbort = !cand.expectDegraded && s.procVramSharedBytes != null && s.procVramSharedBytes - pinned - spillBase > cfg.sharedSpillAbortBytes
        const trip = !guard && ((s.ramAvailBytes != null && s.ramAvailBytes + mmapCredit < ramFloor) || spillAbort)
        if (!trip) continue
        guard = s.ramAvailBytes != null && s.ramAvailBytes + mmapCredit < ramFloor
          ? `RAM available ${(s.ramAvailBytes / GiB).toFixed(1)} GiB fell below the floor`
          : `shared GPU memory spill ${((s.procVramSharedBytes! - pinned - spillBase) / GiB).toFixed(1)} GiB exceeded the abort limit`
        void (loading ? unload() : backend.cancel())
      }
    }
    const timer = setInterval(flush, cand.expectDegraded ? cfg.heavyGuardPollMs : cfg.guardPollMs)
    const guardAbort = () => {
      clearInterval(timer)
      ;(sampler as SessionSampler | null)?.stop()
      return result('fail', 'guard_abort', guard, {}, { samples: (sampler as SessionSampler | null)?.samples ?? [] })
    }

    let load: LoadResult
    try {
      load = await backend.loadModel(loadCfg(cand, model, ctx))
    } catch (e) {
      clearInterval(pidPoll)
      flush()
      if (guard) return guardAbort()
      clearInterval(timer)
      ;(sampler as SessionSampler | null)?.stop()
      const msg = (e as Error).message
      const drift = (e as { failureKind?: string }).failureKind === 'config_drift' // ConfigDriftError (runtimes/llamacpp)
      const kind = signal?.aborted ? null : drift ? 'config_drift' : failKind(backend.lastExit) ?? (/healthy within/.test(msg) ? 'load_timeout' : 'load_fail')
      return result(signal?.aborted ? 'cancelled' : kind === 'load_timeout' ? 'timeout' : 'fail', kind, msg)
    }
    clearInterval(pidPoll)
    if (!sampler && backend.pid !== undefined) { sampler = deps.startSampler(backend.pid); samplerAt = Date.now() }
    pinned = hostPinnedBytes(load.declared, cand.mmap !== false)
    loading = false
    restartIfBlind()
    flush()
    if (guard) { await unload(); return guardAbort() }
    const loadDoneAt = Date.now()
    const smp = sampler as SessionSampler | null
    // A guard's cancel() makes the in-flight request fail with "cancelled": the guard reason must win over that.
    const fail = (error: string, timedOut: boolean) => {
      flush()
      return guard ? { status: 'fail' as const, kind: 'guard_abort' as const, reason: guard } : classify(error, timedOut)
    }
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
    // Spill = shared − host-pinned − the config's unsaturated baseline, and only while dedicated VRAM is near full:
    // WDDM moves allocations to shared memory only when dedicated is exhausted (14B spilled at 83 % dedicated).
    const spillMetric = (): Metric => {
      const raw = pk.max.procVramSharedBytes, ded = pk.max.procVramDedicatedBytes
      if (pk.n === 0 || raw == null) return tele(raw, 'procVramSharedBytes')
      const saturated = vramEffective === null || ded == null || ded >= vramEffective * DEFAULT_SCORING_CONFIG.cliff.vramSaturation
      const v = saturated ? Math.max(0, raw - pinned - spillBase) : 0
      return { value: v, kind: 'measured', source: `per-PID shared ${(raw / GiB).toFixed(2)} GiB − host-pinned ${(pinned / GiB).toFixed(2)} GiB − baseline ${(spillBase / GiB).toFixed(2)} GiB${saturated ? '' : ` (dedicated below ${Math.round(DEFAULT_SCORING_CONFIG.cliff.vramSaturation * 100)} % of the effective budget ${vramEffective === null ? '?' : (vramEffective / GiB).toFixed(2)} GiB: not spill)`}` }
    }
    const prefill = median(reps.map((r) => r.prefillTps))
    const decode = median(reps.map((r) => r.decodeTps))
    const ttft = median(reps.map((r) => r.ttftMs))
    // A12: without runtime timings, derive from wall clock and token counts, labelled estimated.
    // Token counts: the server's timings when present, else the client's streamed-token count (#2's stream counter,
    // `streamedTokens`); prompt tokens fall back to the ladder prompt's size (≈ LADDER_FILL × ctx), labelled estimated.
    const decTok = (r: PromptResult) => r.decodeTokens ?? (r as PromptResult & { streamedTokens?: number | null }).streamedTokens ?? null
    const estDecode = median(reps.map((r) => { const n = decTok(r); return n && r.ttftMs != null && r.totalMs > r.ttftMs ? (n * 1000) / (r.totalMs - r.ttftMs) : null }))
    const estPrefill = median(reps.map((r) => { const n = r.promptTokens ?? Math.floor(ctx * LADDER_FILL); return n && r.ttftMs ? (n * 1000) / r.ttftMs : null }))
    const tps = (v: number | null, e: number | null, what: string): Metric =>
      v != null ? { value: v, kind: 'measured', source: 'llama-server timings' }
        : e != null ? { value: e, kind: 'estimated', source: `${what} tokens / wall clock` } : { value: null, kind: 'unavailable', reason: 'no timings' }

    return result(f?.status ?? 'pass', f?.kind ?? null, f?.reason ?? null, {
      warm,
      ramAvailBeforeLoadBytes: beforeLoad, mmapCreditBytes: mmapCredit,
      peakVramPlateauSamples: pk.max.procVramDedicatedBytes == null ? 0 : samples.filter((x) => x.procVramDedicatedBytes != null && x.procVramDedicatedBytes >= pk.max.procVramDedicatedBytes! * 0.99).length,
      repDecodeTps: reps.map((r) => r.decodeTps).filter((x): x is number => typeof x === 'number'),
      minRamAvailBytes: minRam === null ? { value: null, kind: 'unavailable', reason: 'no RAM reading during the step' } : { value: minRam, kind: 'measured', source: 'RAM guard: OS free RAM / typeperf' },
      promptTokens: median(reps.map((r) => r.promptTokens)),
      loadTimeMs: measured(load.loadTimeMs, 'spawn → /health ok'),
      ttftMs: measured(ttft, 'client wall clock, request → first token', 'no successful request'),
      prefillTps: tps(prefill, estPrefill, 'prompt'),
      decodeTps: tps(decode, estDecode, 'predicted'),
      totalMs: measured(median(reps.map((r) => r.totalMs)), 'client wall clock', 'no successful request'),
      peakVramBytes: tele(pk.max.procVramDedicatedBytes, 'procVramDedicatedBytes'),
      peakSharedGpuBytes: spillMetric(),
      peakSharedGpuRawBytes: tele(pk.max.procVramSharedBytes, 'procVramSharedBytes'),
      hostPinnedBytes: { value: pinned, kind: 'declared', source: 'llama-server load log: host-side model/KV/compute buffers' },
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

  /** Quality-phase loads get the same safety as ladder steps: previous server unloaded, live RAM pre-check,
   *  cancellable load, and a RAM-floor guard (OS reading + typeperf rows, fail-safe when both are missing) that
   *  kills the load or cancels the request. No spill guard here: quality runs at ≤ the practical ceiling, which had
   *  no spill by definition. Returns tripped() for the caller to stop between requests. */
  async function guardedLoad(cand: CandidateConfig, model: ModelMeta, ctx: number, what: string):
    Promise<{ ok: true; tripped: () => string | null; stop: () => void } | { ok: false; reason: string }> {
    await unload((e: Error) => log('warn', `unload before ${what}: ${e.message}`))
    checkStuck()
    const ngl = cand.gpuLayersAll ? model.layers : cand.gpuLayers
    const e = estimateMemory(model, ngl, ctx, cand.kvType, rules.ubatch, cand.kvOffload !== false)
    const need = req.heavyMode ? e.ramResidentBytes : e.ramBytes
    const avail = osRam() ?? val(machine.ramAvailableBytes)
    if (avail !== null && need > avail - ramFloor) {
      return { ok: false, reason: `${what} skipped: est. RAM ${(need / GiB).toFixed(1)} GiB > available ${(avail / GiB).toFixed(1)} GiB − floor ${(ramFloor / GiB).toFixed(1)} GiB` }
    }
    const mmapCredit = cand.mmap === false ? 0 : (model.fileBytes * Math.min(ngl, model.layers)) / model.layers
    const oldPid = backend.pid
    let sampler: SessionSampler | null = null
    let tripped: string | null = null
    let loading = true
    let blind = 0
    const poll = () => {
      if (!sampler && backend.pid !== undefined && backend.pid !== oldPid) sampler = deps.startSampler(backend.pid)
      const rows = (sampler as SessionSampler | null)?.samples ?? []
      const row = [...rows].reverse().find((x) => x.ramAvailBytes != null)?.ramAvailBytes ?? null
      const ram = osRam() ?? row
      blind = deps.readRamAvailableBytes && ram === null ? blind + 1 : 0
      if (tripped) return
      if (ram !== null && ram + mmapCredit < ramFloor) tripped = `RAM available ${(ram / GiB).toFixed(1)} GiB fell below the floor during ${what}`
      else if (blind >= cfg.guardBlindPollsMax) tripped = `RAM guard inputs unreadable during ${what}; stopped rather than continue blind`
      if (tripped) void (loading ? unload() : backend.cancel())
    }
    const timer = setInterval(poll, cand.expectDegraded ? cfg.heavyGuardPollMs : cfg.guardPollMs)
    const stop = () => { clearInterval(timer); (sampler as SessionSampler | null)?.stop() }
    try {
      await backend.loadModel(loadCfg(cand, model, ctx))
    } catch (err) {
      poll()
      stop()
      return { ok: false, reason: tripped ?? `${what} load at ${ctx} failed: ${(err as Error).message}` }
    }
    loading = false
    poll()
    if (tripped) { stop(); await unload(); return { ok: false, reason: tripped } }
    return { ok: true, tripped: () => { poll(); return tripped }, stop }
  }

  /** CR-04-long: needle at 50 % depth of a prompt filling ~0.75 × ctx. null when the load itself failed. */
  async function runLongNeedle(cand: CandidateConfig, model: ModelMeta, ctx: number): Promise<QualityResult | null> {
    send({ type: 'phase', configId: cand.id, ctx, phase: 'quality' })
    const needle = 'OBSIDIAN-42'
    const test: QualityTest = {
      id: 'CR-04-long', category: 'context', weight: 1, maxTokens: 24, template: 'needle',
      params: { depth: 0.5, seed: 4242, needle }, checker: { type: 'needle', needle }
    } as QualityTest
    const g = await guardedLoad(cand, model, ctx, 'long-context needle')
    if (!g.ok) { log('error', `${cand.id}: ${g.reason}`); return null }
    const templateKwargs = templateKwargsFor(model, BASELINE_GEN)
    try {
      const prompt = await backend.applyTemplate([{ role: 'user', content: needlePrompt(test.params!, Math.floor(ctx * 0.75)) }], templateKwargs ? { templateKwargs } : undefined)
      const r = await backend.runPrompt({ prompt, maxTokens: test.maxTokens, temperature: 0, seed: 1, timeoutMs: cfg.promptTimeoutBaseMs + ctx * cfg.promptTimeoutPerCtxMs })
      const why = g.tripped()
      if (why) { log('error', `${cand.id}: ${why}`); return null }
      return r.error ? { testId: test.id, category: 'context', weight: 1, pass: false, score: 0, detail: `request failed: ${r.error}` } : await evaluate(test, r.text)
    } catch (e) {
      return { testId: test.id, category: 'context', weight: 1, pass: false, score: 0, detail: `request failed: ${(e as Error).message}` }
    } finally {
      g.stop()
    }
  }

  /** The suite once per gen config (× samples for stochastic ones) on ONE guarded load. Returns the gen configs whose
   *  suite completed, baseline first; an interrupted config and everything after it is dropped (never partial). */
  async function runQuality(cand: CandidateConfig, model: ModelMeta, ctx: number, gens: { gen: GenConfig; samples: number }[]): Promise<GenQuality[]> {
    send({ type: 'phase', configId: cand.id, ctx, phase: 'quality' })
    const g = await guardedLoad(cand, model, ctx, 'quality suite')
    if (!g.ok) { log('error', `${cand.id}: ${g.reason}`); return [] }
    const prompts = buildQualityPrompts(defaultTestSet, { fillerTokens: Math.min(cfg.qualityFillerMax, Math.floor(ctx * 0.6)), thinking: false })
    const done: GenQuality[] = []
    const renders: (string | null)[] = []
    try {
      for (const { gen, samples } of gens) {
        const templateKwargs = templateKwargsFor(model, gen)
        let firstRender: string | null = null
        if (templateKwargs) log('info', `${model.name}: quality suite with ${genLabel(gen)}${samples > 1 ? `, ${samples} samples` : ''}`)
        const rows: GenRow[] = []
        let interrupted = false
        for (let sample = 1; sample <= samples && !interrupted; sample++) {
          for (const p of prompts) {
            const why = g.tripped()
            if (why) { log('error', `${cand.id}: ${why}`); interrupted = true; break }
            if (signal?.aborted || backend.lastExit) { interrupted = true; break }
            const test = defaultTestSet.tests.find((t) => t.id === p.testId)!
            const tag = { genId: gen.id, sample }
            try {
              const prompt = await backend.applyTemplate(p.messages, templateKwargs ? { templateKwargs } : undefined)
              firstRender ??= prompt
              const s = samplingFor(gen)
              const preq: PromptRequest & { topP?: number; topK?: number; minP?: number } = {
                prompt, seed: sample, temperature: s.temperature, topP: s.top_p, topK: s.top_k, minP: s.min_p,
                maxTokens: gen.thinking ? Math.max(p.maxTokens * 4, cfg.thinkingMinTokens) : p.maxTokens,
                timeoutMs: gen.thinking ? cfg.qualityTimeoutMs * 2 : cfg.qualityTimeoutMs
              }
              const r = await backend.runPrompt(preq)
              const split = splitReasoning(r.text ?? '')
              const toks = r.decodeTokens ?? r.streamedTokens ?? null
              const chars = split.reasoningChars + split.answerChars
              // Prefer the runtime's own reasoning count (reasoning_content / thought channel); else split by text length.
              const reasoningTokens = r.reasoningTokens !== undefined ? r.reasoningTokens ?? 0 : toks === null ? null : chars > 0 ? Math.round((toks * split.reasoningChars) / chars) : 0
              const truncated = r.timedOut || r.stopType === 'limit'
              const counts = {
                answerTokens: toks === null || reasoningTokens === null ? null : toks - reasoningTokens, reasoningTokens, totalMs: r.totalMs ?? null,
                tokenSource: (r.reasoningTokens !== undefined ? 'runtime' : 'estimated') as GenRow['tokenSource'], promptTokens: r.promptTokens, ctx,
                maxTokens: preq.maxTokens, checkerVersion: defaultTestSet.suite, outputTruncated: truncated,
                // A failed request is an infrastructure failure, never a wrong answer (rule I-5.7); a budget stop is truncation (I-5.8).
                evaluationStatus: (r.error && !r.timedOut ? 'infra_error' : truncated ? 'truncated' : 'valid') as GenRow['evaluationStatus'],
                ...(templateKwargs ? { requestedTemplateKwargs: templateKwargs } : {})
              }
              rows.push(r.error
                ? { testId: test.id, category: test.category, weight: test.weight, pass: false, score: 0, detail: `request failed: ${r.error}`, ...tag, ...counts }
                : { ...(await evaluate(test, r.text)), ...tag, ...counts })
            } catch (e) {
              rows.push({ testId: test.id, category: test.category, weight: test.weight, pass: false, score: 0, detail: `request failed: ${(e as Error).message}`, ...tag, evaluationStatus: 'infra_error', checkerVersion: defaultTestSet.suite, ctx })
            }
          }
        }
        // Partial suites (cancel/crash/guard) are not kept: a missing category must not look like a measured 0.
        if (interrupted || rows.length < prompts.length * samples) {
          log('warn', `${cand.id}: quality suite (${gen.id}) incomplete (${rows.length}/${prompts.length * samples}); discarded`)
          break
        }
        renders.push(firstRender)
        done.push(summarizeGen(gen, rows, samples))
      }
      // I-8.0: kwargs count as applied only when they changed the rendered prompt vs another gen config (a template
      // that ignores enable_thinking renders identically — then the comparison is not evaluable).
      for (let k = 0; k < done.length; k++) {
        const kw = templateKwargsFor(model, done[k].gen)
        if (!kw || renders[k] === null) continue
        if (renders.some((x, j) => j !== k && x !== null && x !== renders[k])) for (const r of done[k].results as GenRow[]) r.appliedTemplateKwargs = kw
      }
    } finally {
      g.stop()
    }
    return done
  }
}

