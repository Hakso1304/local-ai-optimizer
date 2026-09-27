// Component scores 0–100 with ABSOLUTE normalization (fixed floors/targets, never min-max or rank across
// candidates), so one candidate still scores and adding a bad candidate can't reorder others (X1, X5).
import type {
  BenchmarkRunResult, CandidateInput, CliffReport, ComponentId, ComponentScore, ComponentScores,
  GenQuality, MachineLimits, Metric, QualityCategory, QualityResult, WorkloadProfile
} from '../../shared/bench-types'
import { detectCliffs, isUsable, val } from './cliff'
import { DEFAULT_SCORING_CONFIG, type ScoringConfig } from './workloads'

const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
const NA = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
const zero = (input: Metric, note?: string): ComponentScore => ({ score: 0, input, note: note ?? input.reason })
/** Unknown ≠ bad and ≠ good: neutral score, flagged in the note (calibration: <1 s requests get 0 telemetry samples). */
const unknown = (input: Metric, cfg: ScoringConfig): ComponentScore =>
  ({ score: cfg.norm.unknownScore, input, note: `unknown (${input.reason ?? 'unavailable'}); scored neutral ${cfg.norm.unknownScore}` })

/** 0 at/below floor, 100 at/above target, log-linear between. Non-positive x → 0. */
export function logScore(x: number, floor: number, target: number): number {
  if (!(x > 0) || !(target > floor) || !(floor > 0)) return 0
  return 100 * clamp01(Math.log(x / floor) / Math.log(target / floor))
}

/** 0 at/below floor, 100 at/above target, linear between. */
export function linScore(x: number, floor: number, target: number): number {
  if (!Number.isFinite(x) || !(target > floor)) return 0
  return 100 * clamp01((x - floor) / (target - floor))
}

/** Without `limits`: the scoring step. With `limits`: the recommended context.
 *  With `limits`: the largest PASS step ≤ maxContext whose full-prompt TTFT ≤ latencyToleranceMs (steps with unknown
 *  TTFT qualify only up to target). Otherwise / if none qualifies: largest PASS step ≤ target, else the smallest PASS
 *  step, else the smallest usable step. Never beyond the practical ceiling when a step passed (X9). */
export function referenceStep(
  runs: BenchmarkRunResult[],
  cliff: CliffReport,
  target: number,
  limits?: { maxContext?: number; latencyToleranceMs?: number }
): BenchmarkRunResult | null {
  const byCtx = new Map(runs.filter(isUsable).map((r) => [r.ctx, r] as const))
  const firstNonPass = cliff.steps.findIndex((s) => s.verdict !== 'pass')
  const pass = (firstNonPass < 0 ? cliff.steps : cliff.steps.slice(0, firstNonPass)).filter((s) => byCtx.has(s.ctx)).map((s) => s.ctx)
  const fits = (c: number) => {
    const t = val(byCtx.get(c)!.ttftMs, true)
    return c <= (limits?.maxContext ?? target) && (t === null || limits?.latencyToleranceMs === undefined ? c <= target : t <= limits.latencyToleranceMs)
  }
  const ctx = (limits ? pass.filter(fits).at(-1) : undefined) ?? pass.filter((c) => c <= target).at(-1) ?? pass[0] ?? cliff.steps.find((s) => byCtx.has(s.ctx))?.ctx
  return ctx === undefined ? null : byCtx.get(ctx)!
}

/** Category-weighted pass rate (0–100) with a 95 % half-width, for any suite size: per category an Agresti–Coull
 *  interval on the weighted pass rate (Kish effective n = (Σw)²/Σw², so repeated samples count as more items), combined
 *  with the category weights. n = graded rows (items × samples). null when no category has results. */
export function qualityStats(results: QualityResult[], categories: QualityCategory[], cfg: ScoringConfig = DEFAULT_SCORING_CONFIG):
  { value: number; n: number; ci95: number } | null {
  const z = cfg.qualityCiZ
  const mine = results.filter((q) => categories.includes(q.category))
  let num = 0, den = 0, v = 0
  const parts: { W: number; var: number }[] = []
  for (const cat of Object.keys(cfg.qualityCategoryWeights) as QualityCategory[]) {
    const rows = mine.filter((q) => q.category === cat)
    const w = rows.reduce((s, r) => s + r.weight, 0)
    if (!(w > 0)) continue
    const p = rows.reduce((s, r) => s + (r.pass ? r.weight : 0), 0) / w
    const nEff = w ** 2 / rows.reduce((s, r) => s + r.weight ** 2, 0)
    const pt = (p * nEff + z * z / 2) / (nEff + z * z)
    const W = cfg.qualityCategoryWeights[cat]
    num += W * p
    den += W
    parts.push({ W, var: (pt * (1 - pt)) / (nEff + z * z) })
  }
  if (!(den > 0)) return null
  for (const x of parts) v += (x.W / den) ** 2 * x.var
  return { value: (100 * num) / den, n: mine.length, ci95: 100 * z * Math.sqrt(v) }
}

function quality(input: CandidateInput, profile: WorkloadProfile, cfg: ScoringConfig, results = input.quality): ComponentScore {
  // Same formula as core/quality qualityScore (not imported: it pulls node:vm), restricted to the profile's categories.
  const st = qualityStats(results, profile.promptSetIds, cfg)
  if (st) {
    return { score: st.value, n: st.n, ci95: st.ci95, input: { value: st.value, kind: 'measured', source: `quality suite, ${st.n} graded items, ±${st.ci95.toFixed(0)} (95 %)` } }
  }
  const { paramCount, fileBytes } = input.model
  if (!paramCount || !(paramCount > 0)) return unknown(NA('no quality results and parameter count unknown'), cfg)
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
  // ngl 0 on a GPU machine still uses the GPU for prefill (calibration) → scored as partial offload on VRAM.
  const cpu = input.config.gpuLayers === 0 && machine.gpuDevice === null
  const used = val(cpu ? ref.peakRamBytes : ref.peakVramBytes)
  const total = val(cpu ? machine.ramTotalBytes : machine.vramBytes, true)
  if (used === null || total === null) return unknown(NA(`${cpu ? 'RAM' : 'VRAM'} ${used === null ? 'peak' : 'total'} unavailable`), cfg)
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
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
  /** Score with this generation config's quality; thinking configs also pay for their reasoning tokens in speed/latency. */
  gen?: GenQuality
): ComponentScores {
  const n = cfg.norm
  const cliff = detectCliffs(input.runs, val(machine.vramBytes, true), cfg.cliff)
  // Score at the workload's target (comparable across candidates); recommend the largest ctx the latency budget allows.
  const ref = referenceStep(input.runs, cliff, profile.targetContext)
  const rec = referenceStep(input.runs, cliff, profile.targetContext, { maxContext: profile.maxContext, latencyToleranceMs: profile.latencyAdvisory ? undefined : profile.latencyToleranceMs })
  const none = zero(NA('no usable context step'))

  const fromRef = (m: Metric | undefined, f: (v: number) => number, what: string): ComponentScore => {
    const v = val(m, true)
    return v === null ? unknown(m?.kind === 'unavailable' ? m : NA(`${what} unavailable`), cfg) : { score: f(v), input: m! }
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
  // Thinking: the reader waits for reasoning tokens too. Effective decode = decode × answer/(answer + reasoning);
  // time to answer = TTFT + reasoning tokens / decode. Token counts are the suite's medians (estimated for the ladder).
  let decodeM = ref?.decodeTps, ttftM = ref?.ttftMs
  if (ref && gen?.gen.thinking) {
    const d = val(ref.decodeTps, true), t = val(ref.ttftMs, true), a = val(gen.answerTokens, true), r = val(gen.reasoningTokens)
    const src = `${gen.gen.id}: ${r ?? '?'} reasoning + ${a ?? '?'} answer tokens (suite median)`
    decodeM = d !== null && a !== null && r !== null ? { value: (d * a) / (a + r), kind: 'estimated', source: `effective decode, ${src}` } : NA(`reasoning token count unavailable for ${gen.gen.id}`)
    ttftM = d !== null && t !== null && r !== null ? { value: t + (r / d) * 1000, kind: 'estimated', source: `time to answer, ${src}` } : NA(`reasoning token count unavailable for ${gen.gen.id}`)
  }
  const components: Record<ComponentId, ComponentScore> = {
    quality: quality(input, profile, cfg, gen ? gen.results : input.quality),
    // Linear, not log: calibration showed log scaling left a 5.7× slower partial offload (17.5 vs 99.9 t/s) only ~6–12
    // points behind; reading speed is felt linearly in t/s. Prefill spans orders of magnitude, so it stays log.
    genSpeed: ref ? fromRef(decodeM, (v) => linScore(v, n.decodeFloorTps, profile.genTargetTps), 'decode TPS') : none,
    prefillSpeed: ref ? fromRef(ref.prefillTps, (v) => logScore(v, n.prefillFloorTps, profile.prefillTargetTps), 'prefill TPS') : none,
    latency: ref ? fromRef(ttftM, (v) => 100 * clamp01(1 - Math.log(Math.max(v, good) / good) / Math.log(tol / good)), 'TTFT') : none,
    memory: ref ? memory(ref, input, machine, cfg) : none,
    stability,
    context
  }
  return { components, cliff, referenceCtx: ref?.ctx ?? null, recommendedCtx: rec?.ctx ?? null, usable: ref !== null }
}
