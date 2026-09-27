// Benchmark / scoring / recommendation types. Pure data, no runtime code. See docs/DESIGN.md §5.
import type { QualityCategory, QualityResult } from '../core/quality'
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
  skippedSteps: { ctx: number; reason: string }[]
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
  /** What produced this row (X17), all declared. runtime null = not reported by detect(). */
  versions?: { benchmark: string; prompts: string; quality: string; runtime: string | null }
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
  /** Quality only: effective number of graded items (items × samples) and the 95 % half-width in points. */
  n?: number
  ci95?: number
}

export interface ComponentScores {
  components: Record<ComponentId, ComponentScore>
  cliff: CliffReport
  /** ctx of the step whose metrics feed speed/latency/memory (≈ targetContext); null when no usable step. */
  referenceCtx: number | null
  /** Context to configure for this workload: largest PASS step ≤ maxContext whose full-prompt TTFT is within tolerance. */
  recommendedCtx: number | null
  usable: boolean
}

export interface BreakdownRow {
  component: ComponentId
  input: Metric
  score: number
  weight: number
  /** weight × score; rows sum to WorkloadScore.total. */
  contribution: number
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
  /** Some candidate's quality is an estimated prior (no quality run): the ranking may change once it is measured. */
  provisional?: boolean
}
