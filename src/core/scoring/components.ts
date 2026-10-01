// Component scores 0–100 with ABSOLUTE normalization (fixed floors/targets, never min-max or rank across
// candidates), so one candidate still scores and adding a bad candidate can't reorder others (X1, X5).
import type {
  BenchmarkRunResult, CandidateInput, CliffReport, ComponentId, ComponentScore, ComponentScores,
  GenQuality, MachineLimits, Metric, QualityCategory, QualityResult, WorkloadProfile
} from '../../shared/bench-types'
import { detectCliffs, fmtCtx, isUsable, val } from './cliff'
import { DEFAULT_SCORING_CONFIG, type ScoringConfig } from './workloads'
import { coverage, qualityUncertainty, type QualityInterval, type UncertaintyRow } from './uncertainty'

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

/** The PASS prefix (practical ceiling and below) that has a usable row, ascending. */
function passRungs(runs: BenchmarkRunResult[], cliff: CliffReport): { byCtx: Map<number, BenchmarkRunResult>; pass: number[] } {
  const byCtx = new Map(runs.filter(isUsable).map((r) => [r.ctx, r] as const))
  const firstNonPass = cliff.steps.findIndex((s) => s.verdict !== 'pass')
  const pass = (firstNonPass < 0 ? cliff.steps : cliff.steps.slice(0, firstNonPass)).filter((s) => byCtx.has(s.ctx)).map((s) => s.ctx)
  return { byCtx, pass }
}

/** Scoring step (rule I-7.1 scoring rung): the largest PASS rung ≤ target, else the smallest PASS rung, else the smallest usable
 *  rung — so candidates are compared at the workload's need, never beyond the practical ceiling (X9). */
export function referenceStep(runs: BenchmarkRunResult[], cliff: CliffReport, target: number, common = false): { run: BenchmarkRunResult; why: string; reaches: boolean } | null {
  const { byCtx, pass } = passRungs(runs, cliff)
  const le = pass.filter((c) => c <= target).at(-1)
  if (common && le === target) return { run: byCtx.get(le)!, why: `common scoring rung ${fmtCtx(target)}`, reaches: true }
  // Passed a larger rung but the common one was not in its ladder: read at the nearest passing rung below, no penalty.
  if (common && le !== undefined && pass.some((c) => c > target)) return { run: byCtx.get(le)!, why: `common scoring rung ${fmtCtx(target)} not in this ladder; read at ${fmtCtx(le)} (clean up to ${fmtCtx(pass.at(-1)!)})`, reaches: true }
  if (common && le !== undefined) return { run: byCtx.get(le)!, why: `does not reach the common scoring rung ${fmtCtx(target)}; speed read at ${fmtCtx(le)}, latency scored 0`, reaches: false }
  if (le !== undefined) return { run: byCtx.get(le)!, why: `largest passing rung ≤ target ${fmtCtx(target)}`, reaches: true }
  if (pass.length) return { run: byCtx.get(pass[0])!, why: `no passing rung ≤ ${fmtCtx(target)}; smallest passing rung`, reaches: !common }
  const u = cliff.steps.find((s) => byCtx.has(s.ctx))
  return u ? { run: byCtx.get(u.ctx)!, why: 'no passing rung; smallest usable (degraded) rung', reaches: !common } : null
}

/** Recommended -c (rule I-3.7, audit D06): always a measured PASS rung. Latency advisory → the largest PASS rung ≤
 *  maxContext. Otherwise the largest PASS rung ≤ maxContext whose measured TTFT ≤ tolerance (unknown TTFT never
 *  qualifies, D07); if none, the smallest PASS rung with fits=false (the TTFT gate then fails). null without a PASS rung. */
export function recommendedStep(runs: BenchmarkRunResult[], cliff: CliffReport, profile: WorkloadProfile):
  { run: BenchmarkRunResult; fits: boolean; why: string } | null {
  const { byCtx, pass } = passRungs(runs, cliff)
  const max = profile.maxContext ?? profile.targetContext
  const tol = profile.latencyToleranceMs
  if (!pass.length) return null
  if (profile.latencyAdvisory) {
    const c = pass.filter((x) => x <= max).at(-1) ?? pass[0]
    return { run: byCtx.get(c)!, fits: true, why: `largest passing rung ≤ ${fmtCtx(max)} (latency advisory)` }
  }
  const fit = pass.filter((c) => { const t = val(byCtx.get(c)!.ttftMs, true); return c <= max && t !== null && t <= tol }).at(-1)
  if (fit !== undefined) return { run: byCtx.get(fit)!, fits: true, why: `largest passing rung ≤ ${fmtCtx(max)} with measured TTFT within the ${(tol / 1000).toFixed(0)} s tolerance` }
  return { run: byCtx.get(pass[0])!, fits: false, why: `no passing rung has a measured TTFT within the ${(tol / 1000).toFixed(0)} s tolerance; smallest passing rung` }
}

/** Quality over the profile's categories via core/scoring/uncertainty (unc-1, a HEURISTIC band: unique items or
 *  skill clusters are the unit, repeats collapse to an item mean, truncated = fail). Rows with an infrastructure
 *  error quarantine the result (rule I-5.7): quality is then unavailable, never a measured 0. */
export function measuredQuality(results: QualityResult[], categories: QualityCategory[], cfg: ScoringConfig = DEFAULT_SCORING_CONFIG):
  ({ ok: true; u: QualityInterval; coverage: ReturnType<typeof coverage> } | { ok: false; quarantined: boolean; reason: string }) {
  // G10: any infrastructure error in the result quarantines it, whether or not its category is weighted here.
  if ((results as UncertaintyRow[]).some((r) => r.evaluationStatus === 'infra_error')) return { ok: false, quarantined: true, reason: 'Quality quarantined: infra_error in the result' }
  const rows = (results as UncertaintyRow[]).filter((q) => categories.includes(q.category)).map(({ genId: _g, ...r }) => r as UncertaintyRow)
  if (!rows.length) return { ok: false, quarantined: false, reason: 'no quality results for these categories' }
  const weights = Object.fromEntries(categories.map((c) => [c, cfg.qualityCategoryWeights[c]]))
  try {
    return { ok: true, u: qualityUncertainty(rows, weights), coverage: coverage(rows) }
  } catch (e) {
    const msg = (e as Error).message
    return { ok: false, quarantined: /quarantined/.test(msg), reason: msg }
  }
}

function quality(input: CandidateInput, profile: WorkloadProfile, cfg: ScoringConfig, results = input.quality): ComponentScore {
  const m = measuredQuality(results, profile.promptSetIds, cfg)
  if (m.ok) {
    const { u, coverage: cov } = m
    return {
      score: u.q, n: u.n, ci95: (u.upper - u.lower) / 2, lower: u.lower, upper: u.upper, method: u.method, unit: u.unit, algorithm: u.version, coverage: cov,
      input: { value: u.q, kind: 'measured', source: `quality suite, ${cov.uniqueItems} items / ${cov.uniqueSkills} skills (${u.method} ${u.version}, heuristic band)` }
    }
  }
  if (m.quarantined) return { ...unknown(NA('quality quarantined: infrastructure error in the harness (I-5.7)'), cfg), quarantined: true }
  const { paramCount, fileBytes } = input.model
  if (!paramCount || !(paramCount > 0)) return unknown(NA('no quality results and parameter count unknown'), cfg)
  // ponytail: size/quant prior, only when the suite didn't run; immutable (never adjusted from other candidates, I-1.2).
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
  // ngl 0 on a GPU machine still uses the GPU for prefill (calibration).
  const cpu = input.config.gpuLayers === 0 && machine.gpuDevice === null
  const sharedRam = machine.gpuSharedRam === true
  const total = val(cpu || sharedRam ? machine.ramTotalBytes : machine.vramBytes, true)
  // On an iGPU device buffers are system RAM. Use the lowest measured available RAM (all processes),
  // not per-PID dedicated VRAM or private RAM, which omit shared device allocations.
  const peak = sharedRam ? ref.minRamAvailBytes : cpu ? ref.peakRamBytes : ref.peakVramBytes
  const available = sharedRam ? val(peak, true) : null
  const used = sharedRam ? total !== null && available !== null ? Math.max(0, total - available) : null : val(peak)
  if (used === null || total === null) return unknown(NA(`${sharedRam ? 'shared RAM' : cpu ? 'RAM' : 'VRAM'} ${used === null ? 'peak' : 'total'} unavailable`), cfg)
  const u = used / total
  let m = u <= n.memFullUntil ? 100
    : u <= n.memKnee ? 100 - ((u - n.memFullUntil) / (n.memKnee - n.memFullUntil)) * (100 - n.memKneeScore)
      : n.memKneeScore * clamp01((1 - u) / (1 - n.memKnee))
  const notes: string[] = []
  const shared = ref.peakSharedGpuBytes
  // Shared GPU usage is normal on an iGPU; only the RAM floor can establish memory safety.
  if (!cpu && !sharedRam && shared.kind !== 'measured') return unknown(NA(`shared-GPU usage not measured at ${ref.ctx} (${shared.reason ?? shared.kind}); spill not verified`), cfg)
  if (!sharedRam && (val(shared) ?? 0) > cfg.cliff.sharedSpillBytes) { m = Math.min(m, n.memSpillCap); notes.push('spills to shared GPU memory') }
  if (!cpu && !input.config.gpuLayersAll) { m = Math.min(m, n.memPartialOffloadCap); notes.push('partial GPU offload') }
  // Provenance follows the peak; an unmeasured RAM minimum cannot become a measured memory term.
  return { score: m, input: { value: u, kind: peak?.kind === 'measured' ? 'measured' : peak?.kind ?? 'unavailable', source: `peak ${sharedRam ? 'system RAM in use' : cpu ? 'RAM' : 'VRAM'} / total${peak?.kind === 'measured' ? '' : ` (peak ${peak?.kind ?? 'unavailable'})`}` }, note: notes.join('; ') || undefined }
}

export function componentScores(
  input: CandidateInput,
  machine: MachineLimits,
  profile: WorkloadProfile,
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
  /** Score with this generation config's quality; thinking configs also pay for their reasoning time in speed/latency. */
  gen?: GenQuality,
  /** I-7.1: the rung every candidate of the workload is scored at (min(target, largest clean context of any candidate)). */
  opts: { scoringRung?: number | null } = {}
): ComponentScores {
  const n = cfg.norm
  const cliff = detectCliffs(input.runs, val(machine.vramBytes, true), cfg.cliff)
  // Score at a rung common to all candidates (else the workload's target); recommend the largest ctx the latency budget allows.
  const refStep = referenceStep(input.runs, cliff, opts.scoringRung ?? profile.targetContext, opts.scoringRung != null)
  const ref = refStep?.run ?? null
  const rec = recommendedStep(input.runs, cliff, profile)
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
  // Thinking (I-3.2): speed is the same-request effective answer rate of the suite (answer tokens / total s), and the
  // wait adds the suite's measured reasoning time to the rung's TTFT. Provenance follows the token-count source.
  let decodeM = ref?.decodeTps, ttftM = ref?.ttftMs
  if (ref && gen?.gen.thinking) {
    const t = val(ref.ttftMs, true), rm = val(gen.reasoningMs)
    decodeM = gen.effectiveTps
    ttftM = t !== null && rm !== null
      // G08: a projection across contexts (ladder TTFT + suite reasoning time) stays ESTIMATED.
      ? { value: t + rm, kind: 'estimated', source: `TTFT at ${fmtCtx(ref.ctx)} + ${gen.gen.id} reasoning time (quality-suite median, other context)` }
      : NA(`reasoning time unavailable for ${gen.gen.id}`)
  }
  const components: Record<ComponentId, ComponentScore> = {
    quality: quality(input, profile, cfg, gen ? gen.results : input.quality),
    // Linear, not log: calibration showed log scaling left a 5.7× slower partial offload (17.5 vs 99.9 t/s) only ~6–12
    // points behind; reading speed is felt linearly in t/s. Prefill spans orders of magnitude, so it stays log.
    genSpeed: ref ? fromRef(decodeM, (v) => linScore(v, n.decodeFloorTps, profile.genTargetTps), 'decode TPS') : none,
    prefillSpeed: ref ? fromRef(ref.prefillTps, (v) => logScore(v, n.prefillFloorTps, profile.prefillTargetTps), 'prefill TPS') : none,
    latency: !ref ? none : refStep!.reaches ? fromRef(ttftM, (v) => 100 * clamp01(1 - Math.log(Math.max(v, good) / good) / Math.log(tol / good)), 'TTFT')
      : { score: 0, input: ttftM ?? NA('TTFT unavailable'), note: refStep!.why },
    memory: ref ? memory(ref, input, machine, cfg) : none,
    stability,
    context
  }
  return {
    components, cliff, referenceCtx: ref?.ctx ?? null, recommendedCtx: rec?.run.ctx ?? null, usable: ref !== null,
    ...(rec ? { recommendedFits: rec.fits, recommendedWhy: rec.why } : {}), ...(refStep ? { referenceWhy: refStep.why, reachesScoringRung: refStep.reaches } : {})
  }
}
