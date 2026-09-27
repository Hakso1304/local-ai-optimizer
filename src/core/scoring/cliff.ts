// Context-step verdicts and practical context ceiling (DESIGN §5.3). Pure, deterministic.
import type { BenchmarkRunResult, CliffReason, CliffReport, ContextStepResult, Metric } from '../../shared/bench-types'
import { DEFAULT_SCORING_CONFIG } from './workloads'

/** Finite value of a metric, else null. `positive` also rejects ≤ 0 (a 0 TPS/TTFT is not a measurement). */
export const val = (m: Metric | undefined, positive = false): number | null => {
  const v = m?.value
  return typeof v === 'number' && Number.isFinite(v) && (!positive || v > 0) ? v : null
}

export const fmtCtx = (n: number): string => (n % 1024 === 0 ? `${n / 1024}K` : String(n))
export const fmtGiB = (b: number): string => `${(b / 1024 ** 3).toFixed(1)} GiB`

/** A step feeds scoring only if it completed and has a real decode rate (X4, X18). */
export const isUsable = (r: BenchmarkRunResult): boolean =>
  (r.status === 'pass' || r.status === 'degraded') && val(r.decodeTps, true) !== null

const reason = (r: Omit<CliffReason, 'from' | 'to' | 'ratio' | 'threshold' | 'fromCtx'> & Partial<CliffReason>): CliffReason =>
  ({ fromCtx: null, from: null, to: null, ratio: null, threshold: null, ...r })

const isSpill = (r: CliffReason) => r.code === 'shared_spill' || r.code === 'vram_spill'

/** steps: one config's results, one per ctx. vramTotalBytes: null if unknown (saturation rule then off). */
export function detectCliffs(
  steps: BenchmarkRunResult[],
  vramTotalBytes: number | null,
  cfg = DEFAULT_SCORING_CONFIG.cliff
): CliffReport {
  const sorted = [...steps].sort((a, b) => a.ctx - b.ctx)
  const out: ContextStepResult[] = []
  let limit: 'cliff' | 'failure' | null = null
  let limitCtx: number | null = null

  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i]
    const prev = i > 0 ? sorted[i - 1] : null
    const reasons: CliffReason[] = []
    const to = s.ctx
    const fromCtx = prev?.ctx ?? null
    const span = prev ? `between ${fmtCtx(prev.ctx)} and ${fmtCtx(to)}` : `at ${fmtCtx(to)}`

    if (!isUsable(s)) {
      const completed = s.status === 'pass' || s.status === 'degraded'
      reasons.push(reason({
        code: completed ? 'invalid_metrics' : 'run_failed', metric: completed ? 'decodeTps' : 'status', toCtx: to,
        message: completed
          ? `${fmtCtx(to)}: run completed but reported no valid decode TPS`
          : `${fmtCtx(to)}: run ${s.status}${s.failureKind ? ` (${s.failureKind})` : ''}${s.reason ? `: ${s.reason}` : ''}`
      }))
      out.push({ ctx: to, verdict: 'fail', reasons })
      if (!limit) { limit = 'failure'; limitCtx = to }
      continue
    }

    // Relative rules compare with the previous step only (X8), and only if it was usable.
    if (prev && isUsable(prev)) {
      const d0 = val(prev.decodeTps, true)!, d1 = val(s.decodeTps, true)!
      const ratio = d1 / d0
      // Recovery-aware: a drop the NEXT rung recovers from (back above the drop threshold vs the pre-drop rate) is a
      // transient dip (e.g. a competing app for a moment), not a sticky cliff. The last rung can't be confirmed → counts.
      const next = sorted[i + 1]
      const dn = next && isUsable(next) ? val(next.decodeTps, true) : null
      const transient = dn !== null && dn / d0 > cfg.decodeDropRatio
      if (ratio <= cfg.decodeDropRatio && d0 - d1 >= cfg.minDecodeDropTps && !transient) {
        reasons.push(reason({
          code: 'decode_drop', metric: 'decodeTps', fromCtx, toCtx: to, from: d0, to: d1, ratio, threshold: cfg.decodeDropRatio,
          message: `decode TPS fell ${Math.round((1 - ratio) * 100)}% ${span} (${d0.toFixed(1)} → ${d1.toFixed(1)} t/s)`
        }))
      }
      const p0 = val(prev.prefillTps, true), p1 = val(s.prefillTps, true)
      if (p0 !== null && p1 !== null) {
        const pr = p1 / p0
        const thr = cfg.prefillDropPerDoubling ** Math.log2(to / prev.ctx)
        if (pr < thr) {
          reasons.push(reason({
            code: 'prefill_drop', metric: 'prefillTps', fromCtx, toCtx: to, from: p0, to: p1, ratio: pr, threshold: thr,
            message: `prefill TPS fell ${Math.round((1 - pr) * 100)}% ${span} (${p0.toFixed(0)} → ${p1.toFixed(0)} t/s), more than context growth explains`
          }))
        }
      }
      const v1 = val(s.peakVramBytes), r0 = val(prev.peakRamBytes), r1 = val(s.peakRamBytes)
      if (vramTotalBytes && v1 !== null && r0 !== null && r1 !== null &&
          v1 / vramTotalBytes >= cfg.vramSaturation && r1 - r0 >= cfg.ramGrowthBytes) {
        reasons.push(reason({
          code: 'vram_spill', metric: 'peakRamBytes', fromCtx, toCtx: to, from: r0, to: r1, ratio: v1 / vramTotalBytes, threshold: cfg.vramSaturation,
          message: `VRAM at ${Math.round((v1 / vramTotalBytes) * 100)}% while RAM grew +${fmtGiB(r1 - r0)} ${span}`
        }))
      }
    }
    // Absolute rule, every step including the first.
    const sh = val(s.peakSharedGpuBytes)
    if (sh !== null && sh > cfg.sharedSpillBytes) {
      reasons.push(reason({
        code: 'shared_spill', metric: 'peakSharedGpuBytes', toCtx: to, to: sh, threshold: cfg.sharedSpillBytes,
        message: `spilled ${fmtGiB(sh)} into shared GPU memory at ${fmtCtx(to)}`
      }))
    }
    // ponytail: sticky — once past a cliff/failure, larger steps are never "practical". A real ≥40% dip that
    // recovers would still end the ceiling; upstream medians-of-reps are the noise defence.
    if (reasons.length === 0 && limit) {
      reasons.push(reason({ code: 'beyond_limit', metric: 'ctx', toCtx: to, message: `${fmtCtx(to)} is beyond the ${limit} at ${fmtCtx(limitCtx!)}` }))
    }
    if (reasons.length && !limit) { limit = 'cliff'; limitCtx = to }
    out.push({ ctx: to, verdict: reasons.length ? 'degraded' : 'pass', reasons })
  }

  const firstFail = out.findIndex((s) => s.verdict === 'fail')
  const beforeFail = firstFail < 0 ? out : out.slice(0, firstFail)
  const firstNonPass = out.findIndex((s) => s.verdict !== 'pass')
  const passPrefix = firstNonPass < 0 ? out : out.slice(0, firstNonPass)
  const practical = passPrefix.at(-1)?.ctx ?? null
  const degraded = beforeFail.at(-1)?.ctx ?? null
  const spillAt = out.findIndex((s) => s.verdict === 'fail' || s.reasons.some(isSpill))
  const spillFree = (spillAt < 0 ? out : out.slice(0, spillAt)).at(-1)?.ctx ?? null
  const m = (v: number | null, what: string): Metric =>
    v === null ? { value: null, kind: 'unavailable', reason: `no step ${what}` } : { value: v, kind: 'measured', source: 'context sweep' }

  return {
    steps: out,
    practicalContextCeiling: m(practical, 'passed'),
    degradedContextCeiling: m(degraded, 'completed'),
    limitedBy: practical === null ? 'untested' : (limit ?? 'none'),
    spillFreeUpTo: spillFree
  }
}
