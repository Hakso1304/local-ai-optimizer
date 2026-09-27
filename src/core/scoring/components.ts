// Component scores 0–100 with ABSOLUTE normalization (fixed floors/targets, never min-max or rank across
// candidates), so one candidate still scores and adding a bad candidate can't reorder others (X1, X5).
import type {
  BenchmarkRunResult, CandidateInput, CliffReport, ComponentId, ComponentScore, ComponentScores,
  MachineLimits, Metric, QualityCategory, WorkloadProfile
} from '../../shared/bench-types'
import { detectCliffs, isUsable, val } from './cliff'
import { DEFAULT_SCORING_CONFIG, type ScoringConfig } from './workloads'

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
const NA = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
const zero = (input: Metric, note?: string): ComponentScore => ({ score: 0, input, note: note ?? input.reason })

/** 0 at/below floor, 100 at/above target, log-linear between. Non-positive x → 0. */
export function logScore(x: number, floor: number, target: number): number {
  if (!(x > 0) || !(target > floor) || !(floor > 0)) return 0
  return 100 * clamp01(Math.log(x / floor) / Math.log(target / floor))
}

/** Step whose speed/latency/memory represent the candidate: largest PASS step ≤ target, else the smallest
 *  PASS step, else the smallest usable step. Never a step beyond the practical ceiling when one passed (X9). */
export function referenceStep(runs: BenchmarkRunResult[], cliff: CliffReport, target: number): BenchmarkRunResult | null {
  const byCtx = new Map(runs.filter(isUsable).map((r) => [r.ctx, r] as const))
  const pass = cliff.steps.filter((s) => s.verdict === 'pass' && byCtx.has(s.ctx)).map((s) => s.ctx)
  const ctx = pass.filter((c) => c <= target).at(-1) ?? pass[0] ?? cliff.steps.find((s) => byCtx.has(s.ctx))?.ctx
  return ctx === undefined ? null : byCtx.get(ctx)!
}

function quality(input: CandidateInput, profile: WorkloadProfile, cfg: ScoringConfig): ComponentScore {
  // Same formula as core/quality qualityScore (not imported: it pulls node:vm), restricted to the profile's categories.
  const mine = input.quality.filter((q) => profile.promptSetIds.includes(q.category))
  let num = 0, den = 0
  for (const cat of Object.keys(cfg.qualityCategoryWeights) as QualityCategory[]) {
    const rows = mine.filter((q) => q.category === cat)
    const w = rows.reduce((s, r) => s + r.weight, 0)
    if (!(w > 0)) continue
    num += cfg.qualityCategoryWeights[cat] * (rows.reduce((s, r) => s + (r.pass ? r.weight : 0), 0) / w)
    den += cfg.qualityCategoryWeights[cat]
  }
  if (den > 0) {
    const q = (100 * num) / den
    return { score: q, input: { value: q, kind: 'measured', source: `quality suite, ${mine.length} tests` } }
  }
  const { paramCount, fileBytes } = input.model
  if (!paramCount || !(paramCount > 0)) return zero(NA('no quality results and parameter count unknown'))
  // ponytail: size/quant prior, only when the suite didn't run; replace with measured Q whenever available.
  const bpw = (fileBytes * 8) / paramCount
  const p = cfg.qualityPrior
  const base = Math.min(p.max, Math.max(0, p.base + p.perDoubling * Math.log2(paramCount / 1e9)))
  const q = base * (bpw >= 6 ? 1 : bpw >= 4.5 ? 0.97 : bpw >= 3.5 ? 0.9 : 0.75)
  return {
    score: q,
    input: { value: q, kind: 'estimated', source: `prior from ${(paramCount / 1e9).toFixed(1)}B params at ${bpw.toFixed(1)} bits/weight` },
    note: 'quality is ESTIMATED from parameter count and quantization, not measured'
  }
}

function memory(ref: BenchmarkRunResult, input: CandidateInput, machine: MachineLimits, cfg: ScoringConfig): ComponentScore {
  const n = cfg.norm
  const cpu = input.config.gpuLayers === 0
  const used = val(cpu ? ref.peakRamBytes : ref.peakVramBytes)
  const total = val(cpu ? machine.ramTotalBytes : machine.vramBytes, true)
  if (used === null || total === null) return zero(NA(`${cpu ? 'RAM' : 'VRAM'} ${used === null ? 'peak' : 'total'} unavailable`))
  const u = used / total
  let m = u <= n.memFullUntil ? 100
    : u <= n.memKnee ? 100 - ((u - n.memFullUntil) / (n.memKnee - n.memFullUntil)) * (100 - n.memKneeScore)
      : n.memKneeScore * clamp01((1 - u) / (1 - n.memKnee))
  const notes: string[] = []
  if ((val(ref.peakSharedGpuBytes) ?? 0) > cfg.cliff.sharedSpillBytes) { m = Math.min(m, n.memSpillCap); notes.push('spills to shared GPU memory') }
  if (!cpu && !input.config.gpuLayersAll) { m = Math.min(m, n.memPartialOffloadCap); notes.push('partial GPU offload') }
  return { score: m, input: { value: u, kind: 'measured', source: `peak ${cpu ? 'RAM' : 'VRAM'} / total` }, note: notes.join('; ') || undefined }
}

export function componentScores(
  input: CandidateInput,
  machine: MachineLimits,
  profile: WorkloadProfile,
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG
): ComponentScores {
  const n = cfg.norm
  const cliff = detectCliffs(input.runs, val(machine.vramBytes, true), cfg.cliff)
  const ref = referenceStep(input.runs, cliff, profile.targetContext)
  const none = zero(NA('no usable context step'))

  const fromRef = (m: Metric | undefined, f: (v: number) => number, what: string): ComponentScore => {
    const v = val(m, true)
    return v === null ? zero(m?.kind === 'unavailable' ? m : NA(`${what} unavailable`)) : { score: f(v), input: m! }
  }

  // Stability over steps the profile needs (failures above the target are the ladder working, not instability).
  const steps = input.runs.filter((r) => r.ctx <= profile.targetContext)
  const scope = steps.length ? steps : input.runs
  const ok = scope.filter(isUsable).length
  const crashed = input.runs.some((r) => r.failureKind === 'crash' || r.failureKind === 'device_lost')
  const stability: ComponentScore = scope.length
    ? {
        score: Math.max(0, (100 * ok) / scope.length - (crashed ? cfg.stability.crashPenalty : 0)),
        input: { value: ok / scope.length, kind: 'measured', source: `${ok}/${scope.length} steps ≤ ${profile.targetContext} usable` },
        note: crashed ? 'crash or device loss observed' : undefined
      }
    : zero(NA('no runs'))

  const ceil = cliff.practicalContextCeiling
  const c = val(ceil, true)
  const context: ComponentScore = c === null ? zero(ceil)
    : {
        score: profile.targetContext <= n.minCtx ? (c >= profile.targetContext ? 100 : 0)
          : 100 * clamp01(Math.log2(c / n.minCtx) / Math.log2(profile.targetContext / n.minCtx)),
        input: ceil
      }

  const tol = profile.latencyToleranceMs
  const good = tol * n.latencyGoodFraction
  const components: Record<ComponentId, ComponentScore> = {
    quality: quality(input, profile, cfg),
    genSpeed: ref ? fromRef(ref.decodeTps, (v) => logScore(v, n.decodeFloorTps, profile.genTargetTps), 'decode TPS') : none,
    prefillSpeed: ref ? fromRef(ref.prefillTps, (v) => logScore(v, n.prefillFloorTps, profile.prefillTargetTps), 'prefill TPS') : none,
    latency: ref ? fromRef(ref.ttftMs, (v) => 100 * clamp01(1 - Math.log(Math.max(v, good) / good) / Math.log(tol / good)), 'TTFT') : none,
    memory: ref ? memory(ref, input, machine, cfg) : none,
    stability,
    context
  }
  return { components, cliff, referenceCtx: ref?.ctx ?? null, usable: ref !== null }
}
