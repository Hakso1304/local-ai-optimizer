// Workload profiles + every scoring threshold, as one plain config object (DESIGN §5).
import type { ComponentId, QualityCategory, WorkloadId, WorkloadProfile } from '../../shared/bench-types'

const MiB = 1024 ** 2
const GiB = 1024 ** 3

type W = [q: number, g: number, p: number, l: number, m: number, s: number, c: number]
const w = ([quality, genSpeed, prefillSpeed, latency, memory, stability, context]: W): Record<ComponentId, number> =>
  ({ quality, genSpeed, prefillSpeed, latency, memory, stability, context })

const ALL: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']

// minDecodeTps: usability gate (heavy mode: a 27B partial offload at ~8 t/s may win Maximum Quality, never Fast
// Assistant or Coding). genTargetTps: full score only near full-offload speed on this class of GPU (8B Q4: 52–109 t/s), so a partial offload
// (17.5 t/s) is clearly below — calibration showed 25–30 t/s targets saturated and hid a 5.7× slowdown.
// latencyToleranceMs is TTFT for a prompt that fills the step (≈0.75·ctx tokens), not a short chat turn. It picks the
// recommended ctx: the largest PASS step ≤ maxContext within tolerance. Calibrated on RX 9070 XT, 8B Q4_K_M full offload
// (TTFT 1.1 s@4K, 2.2 s@8K, 4.9 s@16K, 12 s@32K, 32 s@64K): fast 4K, chat/reasoning 16K, coding 32K, doc/long-ctx 64K.
export const WORKLOADS: Record<WorkloadId, WorkloadProfile> = {
  general_chat: { id: 'general_chat', label: 'General Chat', weights: w([0.30, 0.25, 0.05, 0.15, 0.10, 0.10, 0.05]), targetContext: 8192, maxContext: 16384, latencyToleranceMs: 8000, genTargetTps: 60, prefillTargetTps: 1000, minQuality: 40, minDecodeTps: 10, promptSetIds: ['instruction', 'reasoning', 'structured', 'extraction'] },
  coding: { id: 'coding', label: 'Coding', weights: w([0.40, 0.20, 0.10, 0.10, 0.05, 0.10, 0.05]), targetContext: 16384, maxContext: 32768, latencyToleranceMs: 15000, genTargetTps: 60, prefillTargetTps: 1500, minQuality: 50, minDecodeTps: 10, promptSetIds: ['coding', 'instruction', 'structured'] },
  long_context_coding: { id: 'long_context_coding', label: 'Long-context Coding', weights: w([0.30, 0.10, 0.20, 0.05, 0.05, 0.10, 0.20]), targetContext: 32768, maxContext: 131072, latencyToleranceMs: 40000, genTargetTps: 40, prefillTargetTps: 2000, minQuality: 50, minDecodeTps: 5, promptSetIds: ['coding', 'context'] },
  reasoning: { id: 'reasoning', label: 'Reasoning', weights: w([0.45, 0.20, 0.00, 0.05, 0.05, 0.15, 0.10]), targetContext: 8192, maxContext: 16384, latencyToleranceMs: 10000, genTargetTps: 50, prefillTargetTps: 1000, minQuality: 55, minDecodeTps: 5, promptSetIds: ['reasoning', 'instruction'] },
  document_analysis: { id: 'document_analysis', label: 'Document Analysis', weights: w([0.30, 0.05, 0.25, 0.05, 0.05, 0.10, 0.20]), targetContext: 32768, maxContext: 131072, latencyToleranceMs: 60000, genTargetTps: 30, prefillTargetTps: 2000, minQuality: 45, minDecodeTps: 3, promptSetIds: ['extraction', 'context', 'structured'] },
  fast_assistant: { id: 'fast_assistant', label: 'Fast Assistant', weights: w([0.15, 0.35, 0.10, 0.25, 0.05, 0.10, 0.00]), targetContext: 4096, maxContext: 8192, latencyToleranceMs: 2000, genTargetTps: 100, prefillTargetTps: 1000, minQuality: 30, minDecodeTps: 30, promptSetIds: ['instruction', 'extraction'] },
  max_quality: { id: 'max_quality', label: 'Maximum Quality', weights: w([0.70, 0.05, 0.00, 0.00, 0.05, 0.15, 0.05]), targetContext: 8192, maxContext: 16384, latencyToleranceMs: 20000, genTargetTps: 15, prefillTargetTps: 500, minQuality: 0, minDecodeTps: 2, promptSetIds: ALL }
}

export const DEFAULT_SCORING_CONFIG = {
  version: 'scoring-1.0.0',
  profiles: WORKLOADS,
  qualityCategoryWeights: { instruction: 0.2, reasoning: 0.25, coding: 0.25, structured: 0.1, extraction: 0.1, context: 0.1 } as Record<QualityCategory, number>,
  /** Used only when no quality results exist; labelled ESTIMATED. [A] */
  qualityPrior: { base: 35, perDoubling: 15, max: 90 },
  cliff: {
    // Calibrated 2026-09-27 (8B full offload 2K→64K): healthy per-step decode decay −4/−5/−12/−18/−27%, prefill worst
    // 0.74/doubling, per-PID shared ≤ 0.12 GiB, dedicated ≤ 92% — none fire. Do not go below ~0.65 (−35%): 64K would flag.
    decodeDropRatio: 0.60, // A15: cliff when decode(b)/decode(a) ≤ 0.60, i.e. a ≥40% drop vs the previous step
    minDecodeDropTps: 2, // ...and the absolute drop is ≥ 2 t/s (no cliffs from tiny bases, X8)
    prefillDropPerDoubling: 0.5, // prefill(b)/prefill(a) < 0.5^log2(ctx_b/ctx_a): worse than attention cost explains
    sharedSpillBytes: 256 * MiB, // per-PID shared GPU memory above baseline → spill (F9)
    // WDDM starts spilling well before "full": Qwen2.5-14B per-PID dedicated stalled at 13.25 GiB = 83% of 15.92 GiB
    // while 1.05 GiB went to shared memory (0.95 could never fire). Corroborates shared_spill; 8B peaked at 81% with
    // flat RAM, so the RAM-growth condition keeps it quiet there.
    vramSaturation: 0.80, // per-PID dedicated ≥ 80% of VRAM ...
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
    minCtx: 2048, // context score 0 at this ceiling, 100 at targetContext (log2)
    unknownScore: 50 // unavailable input (e.g. no telemetry samples for a <1 s request): neutral, flagged — never 0, never 100
  },
  gates: { minCtxFraction: 0.5, minStability: 50 },
  stability: { crashPenalty: 20 },
  /** Totals are compared rounded to this many decimals, so float noise can't break ties. */
  tieDecimals: 6
}

export type ScoringConfig = typeof DEFAULT_SCORING_CONFIG
