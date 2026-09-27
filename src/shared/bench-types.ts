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
  | 'document_analysis' | 'fast_assistant' | 'max_quality'

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
  /** Quality-suite categories that count for this profile. */
  promptSetIds: QualityCategory[]
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
  peakSharedGpuBytes: Metric
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
}

export interface CandidateInput {
  config: CandidateConfig
  model: ModelMeta
  runs: BenchmarkRunResult[]
  /** Quality-suite results for this candidate's model (quality is per model+quant, not per ngl). */
  quality: QualityResult[]
}

export interface Recommendation {
  workload: WorkloadId
  scoringVersion: string
  best: {
    configId: string
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
}
