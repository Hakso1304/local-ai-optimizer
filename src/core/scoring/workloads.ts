// Workload profiles + every scoring threshold, as one plain config object (DESIGN §5).
import type { ComponentId, QualityCategory, WorkloadId, WorkloadProfile } from '../../shared/bench-types'

const MiB = 1024 ** 2
const GiB = 1024 ** 3

type W = [q: number, g: number, p: number, l: number, m: number, s: number, c: number]
const w = ([quality, genSpeed, prefillSpeed, latency, memory, stability, context]: W): Record<ComponentId, number> =>
  ({ quality, genSpeed, prefillSpeed, latency, memory, stability, context })

const ALL: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']

// latencyToleranceMs is TTFT for a prompt that fills the reference step (≈ ctx − 192 tokens), not a short chat turn.
export const WORKLOADS: Record<WorkloadId, WorkloadProfile> = {
  general_chat: { id: 'general_chat', label: 'General Chat', weights: w([0.30, 0.25, 0.05, 0.15, 0.10, 0.10, 0.05]), targetContext: 8192, latencyToleranceMs: 8000, genTargetTps: 30, prefillTargetTps: 1000, minQuality: 40, promptSetIds: ['instruction', 'reasoning', 'structured', 'extraction'] },
  coding: { id: 'coding', label: 'Coding', weights: w([0.40, 0.20, 0.10, 0.10, 0.05, 0.10, 0.05]), targetContext: 16384, latencyToleranceMs: 15000, genTargetTps: 30, prefillTargetTps: 1500, minQuality: 50, promptSetIds: ['coding', 'instruction', 'structured'] },
  long_context_coding: { id: 'long_context_coding', label: 'Long-context Coding', weights: w([0.30, 0.10, 0.20, 0.05, 0.05, 0.10, 0.20]), targetContext: 32768, latencyToleranceMs: 40000, genTargetTps: 20, prefillTargetTps: 2000, minQuality: 50, promptSetIds: ['coding', 'context'] },
  reasoning: { id: 'reasoning', label: 'Reasoning', weights: w([0.45, 0.20, 0.00, 0.05, 0.05, 0.15, 0.10]), targetContext: 8192, latencyToleranceMs: 10000, genTargetTps: 25, prefillTargetTps: 1000, minQuality: 55, promptSetIds: ['reasoning', 'instruction'] },
  document_analysis: { id: 'document_analysis', label: 'Document Analysis', weights: w([0.30, 0.05, 0.25, 0.05, 0.05, 0.10, 0.20]), targetContext: 32768, latencyToleranceMs: 60000, genTargetTps: 15, prefillTargetTps: 2000, minQuality: 45, promptSetIds: ['extraction', 'context', 'structured'] },
  fast_assistant: { id: 'fast_assistant', label: 'Fast Assistant', weights: w([0.15, 0.35, 0.10, 0.25, 0.05, 0.10, 0.00]), targetContext: 4096, latencyToleranceMs: 2000, genTargetTps: 60, prefillTargetTps: 1000, minQuality: 30, promptSetIds: ['instruction', 'extraction'] },
  max_quality: { id: 'max_quality', label: 'Maximum Quality', weights: w([0.70, 0.05, 0.00, 0.00, 0.05, 0.15, 0.05]), targetContext: 8192, latencyToleranceMs: 20000, genTargetTps: 8, prefillTargetTps: 500, minQuality: 0, promptSetIds: ALL }
}

export const DEFAULT_SCORING_CONFIG = {
  version: 'scoring-1.0.0',
  profiles: WORKLOADS,
  qualityCategoryWeights: { instruction: 0.2, reasoning: 0.25, coding: 0.25, structured: 0.1, extraction: 0.1, context: 0.1 } as Record<QualityCategory, number>,
  /** Used only when no quality results exist; labelled ESTIMATED. [A] */
  qualityPrior: { base: 35, perDoubling: 15, max: 90 },
  cliff: {
    decodeDropRatio: 0.60, // A15: cliff when decode(b)/decode(a) ≤ 0.60, i.e. a ≥40% drop vs the previous step
    minDecodeDropTps: 2, // ...and the absolute drop is ≥ 2 t/s (no cliffs from tiny bases, X8)
    prefillDropPerDoubling: 0.5, // prefill(b)/prefill(a) < 0.5^log2(ctx_b/ctx_a): worse than attention cost explains
    sharedSpillBytes: 256 * MiB, // per-PID shared GPU memory above baseline → spill (F9)
    vramSaturation: 0.95, // per-PID dedicated ≥ 95% of VRAM ...
    ramGrowthBytes: 1 * GiB // ... while private RAM grows ≥ 1 GiB vs previous step → spill (never RAM alone, X11)
  },
  norm: {
    decodeFloorTps: 2, // ≤ this scores 0
    prefillFloorTps: 20,
    latencyGoodFraction: 0.1, // TTFT ≤ tol×0.1 scores 100, log-linear to 0 at tol
    memFullUntil: 0.80, // utilization ≤ 80% scores 100
    memKnee: 0.97, // linear 100→40 between 80% and 97%, then →0 at 100%
    memKneeScore: 40,
    memSpillCap: 30, // shared spill caps memory score
    memPartialOffloadCap: 60,
    minCtx: 2048 // context score 0 at this ceiling, 100 at targetContext (log2)
  },
  gates: { minCtxFraction: 0.5, minStability: 50 },
  stability: { crashPenalty: 20 },
  /** Totals are compared rounded to this many decimals, so float noise can't break ties. */
  tieDecimals: 6
}

export type ScoringConfig = typeof DEFAULT_SCORING_CONFIG
