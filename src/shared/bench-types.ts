// Benchmark / scoring / recommendation types. Pure data, no runtime code. See docs/DESIGN.md §5.
import type { QualityCategory, QualityResult } from '../core/quality'
import type { DecisionTrace, Insight } from '../core/interpret'
export type { QualityCategory, QualityResult }

/** Where a value came from (ACCEPTANCE R1).
 *  measured = observed on this machine this session; declared = read from GGUF/driver/OS/runtime log;
 *  estimated = computed by our formulas (formula named in `source`); unavailable = could not get it (`reason`). */
export type ProvenanceKind = 'measured' | 'estimated' | 'declared' | 'unavailable'

/** value is null iff kind === 'unavailable'. Never encode "unknown" as 0. */
export interface Metric<T = number> {
  value: T | null
  kind: ProvenanceKind
  source?: string
  reason?: string
}

export type WorkloadId =
  | 'general_chat' | 'coding' | 'long_context_coding' | 'reasoning'
  | 'document_analysis' | 'fast_assistant' | 'max_quality' | 'large_coding'

export type ComponentId = 'quality' | 'genSpeed' | 'prefillSpeed' | 'latency' | 'memory' | 'stability' | 'context'

export interface WorkloadProfile {
  id: WorkloadId
  label: string
  /** Non-negative, sum to 1. */
  weights: Record<ComponentId, number>
  targetContext: number
  /** Largest context worth recommending for this workload (default targetContext). The recommended ctx is the
   *  largest PASS step ≤ this whose full-prompt TTFT is within latencyToleranceMs. */
  maxContext?: number
  /** TTFT (ms) for a prompt filling the reference step; latency score is 0 at/above this. */
  latencyToleranceMs: number
  genTargetTps: number
  prefillTargetTps: number
  /** Quality gate (0–100). */
  minQuality: number
  /** Decode gate (t/s at the scoring step): below this the workload is unusable, whatever else scores. */
  minDecodeTps?: number
  /** Set from SessionRequest.requiredContext: eligibility needs practical context ≥ this. */
  requiredContext?: number
  /** TTFT is reported and scored but never gates (large-scale work, or an explicit required context). */
  latencyAdvisory?: boolean
  /** Quality-suite categories that count for this profile. */
  promptSetIds: QualityCategory[]
  /** Quality decides the winner when two candidates' quality confidence bands don't overlap; speed only within the band. */
  qualityFirst?: boolean
}

/** Machine facts the scorer/candidate generator needs. Built from SystemProfile (machineFromProfile). */
export interface MachineLimits {
  /** Total dedicated VRAM of the benchmark GPU (declared). */
  vramBytes: Metric
  /** Dedicated VRAM used by other processes before launch (measured). */
  vramInUseBytes: Metric
  ramTotalBytes: Metric
  ramAvailableBytes: Metric
  physicalCores: number
  /** Runtime device id, e.g. 'Vulkan0'. null = CPU only. */
  gpuDevice: string | null
  /** Per-process dedicated VRAM ceilings observed on this GPU + driver + backend build (spill began there). */
  vramBudgetObservations?: VramBudgetObservation[]
  /** The most conservative per-process budget (min of observations), or the 80 % fallback labelled estimated. */
  vramEffectiveBudgetBytes?: Metric
}

/** One per-process VRAM observation (advisory until qualified). 'capacity': shared residency reproduced on a fresh
 *  server after the I-2.8 retry, with a successful measured request — the only kind that can prune, and only when
 *  `qualified` and comparable. 'clean': a measured clean run at this dedicated peak — protects that allocation. */
export interface VramBudgetObservation {
  kind: 'capacity' | 'clean'
  /** adapter identity (PNP id) + driver + backend build all verified; false → advisory, never prunes */
  qualified: boolean
  ceilingBytes: number
  modelId: string
  ctx: number
  kvType: KvType
  gpuLayers: number
  /** KV buffer on the GPU as declared by the runtime at load; null when not logged. */
  kvBytes: number | null
  /** Largest individual device buffer from the load log (model / KV / compute); null when not logged. */
  largestBufferBytes: number | null
  /** Where the observation came from: session, config, the attempts behind it and the first attempt's peak. */
  origin: { sessionId: string; configId: string; status: RunStatus; attempts: number; firstPeakVramBytes: number | null; firstResidentSharedBytes: number | null }
  observedAt: number
}

/** GGUF header facts; every field is DECLARED (paramCount derived from tensor dims). null = key absent. */
export interface ModelMeta {
  id: string
  name: string
  fileBytes: number
  paramCount: number | null
  quant: string | null
  arch: string
  ctxTrain: number | null
  layers: number
  nEmbd: number
  heads: number
  headsKv: number
  keyLength: number | null
  valueLength: number | null
  nVocab: number
  slidingWindow: number | null
  /** Optional KV layout keys (GGUF), so hybrid / SWA models aren't estimated as all-full-attention:
   *  `<arch>.attention.head_count_kv` as a per-layer array (gemma4) — headsKv is then its max. */
  headsKvPerLayer?: number[] | null
  /** `<arch>.full_attention_interval` (qwen35/qwen3next hybrids): only layers with (i+1) % n == 0 keep a KV cache;
   *  the others are recurrent (fixed-size state, covered by the VRAM margin). */
  fullAttentionInterval?: number | null
  /** `<arch>.attention.sliding_window_pattern` (true = sliding-window layer) + `<arch>.attention.{key,value}_length_swa`. */
  slidingWindowPattern?: boolean[] | null
  keyLengthSwa?: number | null
  valueLengthSwa?: number | null
  /** Chat template supports a reasoning toggle ("enable_thinking"); quality then runs with thinking disabled. */
  supportsThinking?: boolean
  /** MoE: `<arch>.expert_count` / `<arch>.expert_used_count` (gemma4: 128 / 8). Absent or 0 = dense. */
  expertCount?: number | null
  expertUsedCount?: number | null
  /** Generation knobs (filled by #2 from the chat template + optional HF generation_config.json). */
  genKnobs?: GenKnobs
}

/** What a model's chat template / model card lets us vary at generation time. */
export interface GenKnobs {
  supportsThinking: boolean
  /** Allowed reasoning-effort values in ascending order (e.g. ['low','medium','high','xhigh']). */
  effortValues?: string[]
  /** Template variable that takes the effort (default 'reasoning_effort'). */
  effortKw?: string
  /** Template variable for a thinking-token budget, if the template has one (recorded, not searched). */
  thinkingBudgetKw?: string
  /** Model-card sampling (generation_config.json). */
  recommended?: { temperature?: number; topP?: number; topK?: number; minP?: number }
  /** Hugging Face repo `recommended` was read from (the model's .meta.json sidecar). */
  repoId?: string
}

/** One generation setting the quality suite is run with. `id` is deterministic (e.g. 'off', 'think-low-t1'). */
export interface GenConfig {
  id: string
  thinking: boolean
  effort?: string
  temperature: number
  topP?: number
  topK?: number
  minP?: number
  source: 'model-card' | 'default' | 'template'
}

/** Quality suite results for one GenConfig on the model's best-offload config. Token counts are medians per request. */
export interface GenQuality {
  gen: GenConfig
  /** Every sample of every test (stochastic configs: `samples` rows per test). */
  results: QualityResult[]
  samples: number
  /** T > 0: seeded but not deterministic across builds/hardware. */
  stochastic: boolean
  answerTokens: Metric
  reasoningTokens: Metric
  /** Median request wall time = effective answer latency for a suite-sized prompt. */
  effectiveAnswerLatencyMs: Metric
  /** Answer tokens per total second (reasoning time counts against it). */
  effectiveTps: Metric
  /** Median time spent reasoning per request (total × reasoning/(answer + reasoning)), same requests. */
  reasoningMs?: Metric
  /** Raw decode of the same requests: (answer + reasoning) tokens per total second. */
  rawTps?: Metric
}

export type KvType = 'f16' | 'q8_0'

export interface CandidateConfig {
  /** Deterministic: `${modelId}|ngl=<all|n>|kv=<type>|t=<threads>`. Tie-breaks sort on it. */
  id: string
  modelId: string
  device: string | null
  gpuLayers: number
  gpuLayersAll: boolean
  kvType: KvType
  flashAttn: boolean
  threads: number
  /** Context steps to run, ascending. */
  ctxSteps: number[]
  skippedSteps: { ctx: number; reason: string; /** structured (rule I-2.3) */ skip?: PlannedSkip }[]
  /** At the smallest step. kind 'estimated'. Used for pruning only. */
  estVramBytes: Metric
  estRamBytes: Metric
  notes: string[]
  /** false = KV cache in system RAM (llama-server -nkvo). Absent = true. */
  kvOffload?: boolean
  /** false = load without mmap (llama-server `-lm none`). Heavy/partial configs: with mmap the whole GGUF stays
   *  resident (a 16.4 GB 27B cost ≈17 GiB of available RAM at 55/65 layers); without it host RAM ≈ CPU layers + KV. */
  mmap?: boolean
  /** Heavy-model mode: partial offload chosen on purpose; slow decode is expected, with the reason. */
  expectDegraded?: boolean
  degradedReason?: string
  /** What the planner assumed (data contract §12; rules I-4.1 / I-4.2). Persisted with the plan. */
  planning?: PlanningSnapshot
}

/** Why the planner did not attempt a rung. resource 'declared' = above the model's declared context. */
export interface PlannedSkip {
  resource: 'vram' | 'ram' | 'declared'
  estimateBytes?: number
  budgetBytes?: number
  ruleId: string
  /** VRAM skips: the estimate's breakdown (weights incl. output, KV on GPU, compute + logits). */
  weightsBytes?: number
  kvBytes?: number
  overheadBytes?: number
}

/** The planner's machine snapshot and budgets for this config. */
export interface PlanningSnapshot {
  vramTotalBytes: number | null
  /** measured, or the assumed default (estimated) when the reading was unavailable */
  vramInUse: Metric
  /** VRAM total − in use (before the reserve). */
  planningVramBudgetBytes: number | null
  /** Safety margin kept free (candidate rules vramMarginBytes). */
  planningReserveBytes: number
  /** Per-process budget used for this candidate (at its largest planned buffer); the planning budget is
   *  min(total − in use − reserve, this). Absent on plans made before cand-1.5. */
  effectiveBudget?: Metric
  ramAvailableBytes: number | null
  ramReserveBytes: number
  candidateRulesVersion: string
}

export interface RejectedCandidate {
  id: string
  modelId: string
  reason: string
}

export interface CandidateSet {
  candidates: CandidateConfig[]
  rejected: RejectedCandidate[]
}

/** Runner writes pass|fail|timeout|cancelled; 'degraded' = completed but runner-flagged (unstable/noisy).
 *  ACCEPTANCE A11 mapping: ok→pass, failed/oom/device_lost/crashed→fail + failureKind. */
export type RunStatus = 'pass' | 'degraded' | 'fail' | 'timeout' | 'cancelled'

export type FailureKind =
  | 'oom' | 'load_fail' | 'device_lost' | 'crash' | 'exit_1' | 'load_timeout' | 'req_timeout'
  | 'guard_abort' | 'config_drift' | 'skipped_memory' | 'request_error'

/** One config at one context step (reps already reduced to median by the runner). */
export interface BenchmarkRunResult {
  configId: string
  ctx: number
  promptTokens: number | null
  status: RunStatus
  failureKind: FailureKind | null
  loadTimeMs: Metric
  /** Client wall clock, request → first predicted token. */
  ttftMs: Metric
  prefillTps: Metric
  decodeTps: Metric
  totalMs: Metric
  /** Per-PID dedicated VRAM peak. */
  peakVramBytes: Metric
  /** Per-PID shared GPU memory peak minus pre-load baseline (spill signal; never adapter totals). */
  /** SPILL: per-PID shared GPU memory − host-pinned buffers (load log) − the config's unsaturated first-step
   *  level, counted only while dedicated VRAM is ≥ the saturation share (WDDM spills only near full). */
  peakSharedGpuBytes: Metric
  /** Raw per-PID shared GPU memory peak (includes pinned host buffers, e.g. CPU layers under -lm none). */
  peakSharedGpuRawBytes?: Metric
  /** Host-side model/KV/compute buffers from the load log (Vulkan_Host, CPU, CPU_Mapped without mmap). */
  hostPinnedBytes?: Metric
  /** Per-PID private working set peak (excludes mmap file cache). */
  peakRamBytes: Metric
  /** Decode-window averages (load phase excluded). */
  avgGpuUtil: Metric
  avgCpuUtil: Metric
  /** Why the step failed / was skipped, as the runner recorded it (e.g. the RAM-guard arithmetic). */
  reason?: string | null
  /** true = measured after a successful size-matched warmup (X13). Absent in pre-1.0 data. */
  warm?: boolean
  /** Decode t/s of each measured rep (server timings), for the rep-variance rule. */
  repDecodeTps?: number[]
  /** Lowest RAM available the guard saw during the step (load + requests). */
  minRamAvailBytes?: Metric
  /** Lowest RAM available during the load phase only (I-4.4 lifecycle: before load / during load / after unload). */
  minRamAvailDuringLoadBytes?: Metric
  /** RAM available just before this step's load (i.e. after the previous server was unloaded). */
  ramAvailBeforeLoadBytes?: Metric
  /** The RAM floor the guard enforced and the mmap credit it granted (rule I-4.3: signed distance, credit separately). */
  ramFloorBytes?: number
  mmapCreditBytes?: number
  /** Telemetry samples within 1 % of the dedicated-VRAM peak BEFORE spill onset (rule I-4.5 needs ≥ 3); 0 = no spill. */
  peakVramPlateauSamples?: number
  /** Algorithm of peakVramPlateauSamples; only 'pre-spill-1' rows support I-4.5 (older rows counted any sample). */
  peakVramPlateauVersion?: string
  /** Adapter-wide free dedicated VRAM (total − adapter dedicated) in the sample where per-PID shared peaked (I-2.8). */
  adapterFreeAtSharedPeakBytes?: Metric
  /** I-2.8: this row is the re-measurement after a fresh server restart; `placementFirst` is the first observation. */
  placementRetry?: boolean
  placementFirst?: { peakVramBytes: Metric; peakSharedGpuBytes: Metric; peakSharedGpuRawBytes?: Metric; adapterFreeAtSharedPeakBytes?: Metric; decodeTps: Metric }
  /** Structured reason when the runner skipped the step before loading (live RAM pre-check). */
  skip?: PlannedSkip
  /** What produced this row (X17), all declared. runtime null = not reported by detect(). */
  versions?: { benchmark: string; prompts: string; quality: string; runtime: string | null; rules?: string }
}

export type StepVerdict = 'pass' | 'degraded' | 'fail'

export type CliffCode =
  | 'run_failed' | 'invalid_metrics' | 'decode_drop' | 'prefill_drop' | 'shared_spill' | 'vram_spill' | 'beyond_limit'

export interface CliffReason {
  code: CliffCode
  metric: string
  fromCtx: number | null
  toCtx: number
  from: number | null
  to: number | null
  ratio: number | null
  threshold: number | null
  message: string
}

export interface ContextStepResult {
  ctx: number
  verdict: StepVerdict
  reasons: CliffReason[]
}

export interface CliffReport {
  steps: ContextStepResult[]
  /** Largest ctx such that it and every lower step PASS (measured). */
  practicalContextCeiling: Metric
  /** Largest non-FAIL ctx below the first FAIL (measured). */
  degradedContextCeiling: Metric
  /** What ended the PASS prefix: 'none' = every step passed; 'untested' = no step passed. */
  limitedBy: 'none' | 'cliff' | 'failure' | 'untested'
  /** Largest usable ctx with no spill reason at or below it; null if none. */
  spillFreeUpTo: number | null
}

export interface ComponentScore {
  /** 0–100, always finite. Unavailable input scores norm.unknownScore (neutral) and says so in note. */
  score: number
  input: Metric
  note?: string
  /** Quality only (core/scoring/uncertainty, unc-1 heuristic): independent units, band, half-width, method, coverage. */
  n?: number
  ci95?: number
  lower?: number
  upper?: number
  method?: string
  unit?: string
  algorithm?: string
  coverage?: { uniqueItems: number; uniqueSkills: number; seeds: number; completions: number; validItems: number; categoriesCovered: string[]; infraErrors: number; truncated: number }
  /** Quality rows contained an infrastructure error (I-5.7): excluded, shown as invalid. */
  quarantined?: boolean
}

export interface ComponentScores {
  components: Record<ComponentId, ComponentScore>
  cliff: CliffReport
  /** ctx of the step whose metrics feed speed/latency/memory (≈ targetContext); null when no usable step. */
  referenceCtx: number | null
  /** Context to configure (rule I-3.10): always a measured PASS rung — the largest ≤ maxContext whose measured
   *  full-prompt TTFT is within tolerance (advisory latency: the largest PASS rung ≤ maxContext); if none is within
   *  tolerance, the smallest PASS rung with recommendedFits=false; null without a PASS rung. */
  recommendedCtx: number | null
  /** false = the recommended rung is outside the latency tolerance (or its TTFT is unknown) and not advisory. */
  recommendedFits?: boolean
  /** false = the candidate does not reach the common scoring rung (I-7.1): speed read lower, latency scored 0. */
  reachesScoringRung?: boolean
  /** Why these rungs were chosen (rules I-2.8 / I-3.10). */
  referenceWhy?: string
  recommendedWhy?: string
  usable: boolean
}

export interface BreakdownRow {
  component: ComponentId
  input: Metric
  score: number
  weight: number
  /** weight × score; rows sum to WorkloadScore.total. */
  contribution: number
  /** Quality row: graded items and the 95 % half-width ("Q ± ci (n)"). */
  n?: number
  ci95?: number
}

export interface WorkloadScore {
  configId: string
  workload: WorkloadId
  total: number
  eligible: boolean
  gateFailures: string[]
  breakdown: BreakdownRow[]
  referenceCtx: number | null
  recommendedCtx: number | null
  /** The generation config this score used (absent = the baseline quality). */
  gen?: GenConfig
}

export interface CandidateInput {
  config: CandidateConfig
  model: ModelMeta
  runs: BenchmarkRunResult[]
  /** Quality-suite results for this candidate's model (quality is per model+quant, not per ngl). With genQuality this
   *  is the deterministic baseline (thinking off, T=0). */
  quality: QualityResult[]
  /** One entry per generation config that was run (baseline included); the scorer picks the best per workload. */
  genQuality?: GenQuality[]
}

export interface Recommendation {
  workload: WorkloadId
  scoringVersion: string
  best: {
    configId: string
    /** Set when no candidate passed every gate but this one reaches the required context (fastest such). */
    fallback?: 'meets required context; below preferred speed'
    /** "<model> <quant> @ <recommendedCtx> — <decode> t/s, quality <Q>, no spill up to <ctx>" (always set by recommend()). */
    headline?: string
    /** Chosen generation config, with e.g. "thinking on (effort low, T=1.0): quality 88 ± 6 vs 71 ± 7 with thinking off; answers 2.3× slower". */
    gen?: { config: GenConfig; reason: string }
    score: WorkloadScore
    practicalContext: Metric
    declaredContext: Metric
    cliff: CliffReport
  } | null
  /** configIds among eligible candidates; null when none eligible. */
  alternatives: { fastest: string | null; bestQuality: string | null; bestLongContext: string | null; lowestMemory: string | null }
  /** Scored candidates (eligible first, then total desc, tie-break chain). */
  ranked: WorkloadScore[]
  /** Candidates with no usable step (A17). */
  excluded: { configId: string; reasons: string[] }[]
  reasons: string[]
  /** Why the runners-up lost: the top 2 non-winners by rank, plus any with higher measured quality than the winner.
   *  One sentence each, built only from the numbers the score used. Empty when there is no winner. */
  whyNot?: { configId: string; model: string; summary: string; /** set for the winner's other generation configs */ genId?: string }[]
  /** Some candidate's quality is an estimated prior (no quality run), or the winner was decided on an estimated
   *  component (rules I-1.1 / I-5.5): the ranking may change once it is measured. */
  provisional?: boolean
  /** Interpretation rules version (core/interpret rules.v2.json) and the insight panel they produced. */
  rulesVersion?: string
  insights?: Insight[]
  /** I-1.2: best candidate that depends on an estimated/unavailable decisive term — never the confirmed best. */
  provisionalBest?: { configId: string; headline: string; estimatedTerms: string[]; reason: string }
  /** I-2.5: configs that reach the required context but miss a user constraint (not recommendations). */
  unmetAlternatives?: { configId: string; unmet: string[] }[]
  /** I-7.2: every constraint, eligibility, neutralization and tie-break that produced this result. */
  decisionTrace?: DecisionTrace
}
