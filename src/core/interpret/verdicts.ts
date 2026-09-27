// Every ranking decision (rules interp-2), recorded in a decision trace (I-7.2). Pure and deterministic.
// Order: I-6.0 speed eligibility → common scoring rung (I-7.1) → generation config (I-8.0/8.1) → hard constraints
// first (I-1.1: required context, decode floor, latency; unknown = not met) → soft gates → partial-offload veto
// (I-7.6) → confirmed vs provisional (I-1.2) → quality difference with neutralization (I-5.2) → tie-break chain.
import type {
  BenchmarkRunResult, BreakdownRow, CandidateInput, ComponentId, ComponentScores, GenQuality, MachineLimits, WorkloadId, WorkloadProfile
} from '../../shared/bench-types'
import { componentScores } from '../scoring/components'
import { fmtCtx, val } from '../scoring/cliff'
import { includesZero, pairedDifference, type UncertaintyRow } from '../scoring/uncertainty'
import { DEFAULT_SCORING_CONFIG, type ScoringConfig } from '../scoring/workloads'
import { genLabel } from '../benchmark/gen'
import { cite, P, rule, RULES, RULES_VERSION, tag } from './catalog'

export type StoredRun = BenchmarkRunResult & { runId?: string; supersededBy?: string; startedAt?: number; endedAt?: number }
export interface InterpretData {
  candidates: CandidateInput[]
  machine: MachineLimits
  /** Every persisted run of the session incl. superseded retries (I-6.3); default = the candidates' latest rows. */
  allRuns?: StoredRun[]
  /** Session-level planning values when runs don't carry them (§12). */
  planningSnapshot?: { ramFloorBytes?: number; mmapCreditBytes?: number }
  stopReason?: 'done' | 'cancelled' | 'paused' | 'interrupted' | 'user-cap'
}
export interface Request { requiredContext?: number | null; minDecodeTps?: number | null }

export interface GateFailure { ruleId: string; text: string; /** a user-set hard constraint (never overridden) */ user?: boolean; hard?: boolean; /** evidence missing → counted as not met (I-1.1) */ notVerified?: string }
export interface Difference { diff: number; lower: number; upper: number; sharedItems: number }
export interface GenOption { gq: GenQuality; cs: ComponentScores; total: number; withinTolerance: boolean; comparable: boolean; why?: string }
export type LimitKind = 'failure' | 'spill' | 'cliff' | 'planned-skip:memory' | 'planned-skip:ram' | 'user-cap' | 'cancelled' | 'largest-tested' | 'unknown'
export interface Coverage {
  largestCleanTested: number | null
  firstObservedFailure: { ctx: number; reason: string } | null
  firstPlannedSkip: { ctx: number; reason: string; resource?: string; estimateBytes?: number; budgetBytes?: number } | null
  limitKind: LimitKind
}
export interface CandidateVerdict {
  input: CandidateInput
  /** input with only speed-eligible completed rows (I-6.0); failed rows kept */
  scored: CandidateInput
  dropped: { ctx: number; reason: string }[]
  cs: ComponentScores
  gen: GenOption | null
  genOptions: GenOption[]
  breakdown: BreakdownRow[]
  total: number
  eligible: boolean
  failures: GateFailure[]
  /** decisive components that are ESTIMATED or UNAVAILABLE (I-1.2) */
  undecided: { component: ComponentId; kind: string; reason?: string }[]
  confirmed: boolean
  decode: number | null
  vram: number | null
  ram: number | null
  qualityMeasured: boolean
  qualityRows: UncertaintyRow[]
  coverage: Coverage
}
export interface TraceStep { ruleId: string; kind: string; winner: string; over?: string; detail: string; difference?: Difference | null }
export interface DecisionTrace {
  rulesVersion: string
  scoringVersion: string
  workload: WorkloadId
  scoringRung: number | null
  scoringRungWhy: string
  hardConstraints: { requiredContext: number | null; minDecodeTps: { value: number; source: 'user' | 'workload' } | null; latencyToleranceMs: number; latencyAdvisory: boolean }
  eligibleSet: { configId: string; failingRuleIds: string[] }[]
  candidates: { configId: string; confirmed: boolean; undecided: string[]; total: number; qualityContribution: number; gen: string | null; referenceCtx: number | null; failures: string[] }[]
  steps: TraceStep[]
  neutralizations: { a: string; b: string; difference: Difference | null; reason?: string; totalsWithoutQuality: Record<string, number>; winner: string }[]
  tieBreakChain: { step: string; a: string; b: string; a_value: number | string | null; b_value: number | string | null; decided: boolean }[]
  alternatives: Record<'fastest' | 'bestQuality' | 'bestLongContext' | 'lowestMemory', { configId: string | null; ruleId: string }>
  winner: string | null
  provisionalWinner: string | null
  unmetAlternatives: { configId: string; unmet: string[] }[]
  genChoices: { configId: string; chosen: string | null; steps: string[] }[]
  thresholdsUsed: Record<string, number | boolean | null>
}
export interface Verdicts {
  workload: WorkloadId
  profile: WorkloadProfile
  cfg: ScoringConfig
  data: InterpretData
  request: Request
  /** confirmed eligible (decided order), provisional eligible, ineligible */
  ranked: CandidateVerdict[]
  provisional: CandidateVerdict[]
  excluded: { configId: string; reasons: string[] }[]
  winner: CandidateVerdict | null
  provisionalWinner: CandidateVerdict | null
  /** I-2.5: winner reaches the required context and fails only the workload's own speed gates */
  fallback: boolean
  unmet: CandidateVerdict[]
  trace: DecisionTrace
  quality: { decisive: boolean; a: CandidateVerdict; b: CandidateVerdict; difference: Difference | null; reason?: string } | null
  scoringRung: number | null
  sessionVersion: string | null
}

const byId = (a: CandidateVerdict, b: CandidateVerdict) => (a.input.config.id < b.input.config.id ? -1 : a.input.config.id > b.input.config.id ? 1 : 0)
const cmp = (a: number | null, b: number | null, desc: boolean) =>
  a === b ? 0 : a === null ? 1 : b === null ? -1 : desc ? b - a : a - b
const t1 = (x: number) => x.toFixed(1)
const sec = (ms: number) => `${(ms / 1000).toFixed(1)} s`
const Q = (v: CandidateVerdict) => v.cs.components.quality
const qContribution = (v: CandidateVerdict) => v.breakdown.find((r) => r.component === 'quality')?.contribution ?? 0

/** I-6.0: may this completed row enter speed scoring? null = yes (or not a completed row). */
export function speedIneligible(r: BenchmarkRunResult, sessionVersion: string | null): string | null {
  if (r.status !== 'pass' && r.status !== 'degraded') return null
  if (r.warm !== true) return r.warm === false ? 'the warmup failed' : 'warmup not recorded'
  if (r.decodeTps.kind !== 'measured') return `decode is ${r.decodeTps.kind}${r.decodeTps.source ? ` (${r.decodeTps.source})` : ''}`
  if (val(r.decodeTps, true) === null) return 'no valid decode TPS (zero or missing)'
  const v = r.versions ? `${r.versions.benchmark}/${r.versions.prompts}` : null
  if (sessionVersion && v && v !== sessionVersion) return `benchmark/prompt version ${v} differs from the session's ${sessionVersion}`
  return null
}

function sessionVersionOf(runs: BenchmarkRunResult[]): string | null {
  const count = new Map<string, number>()
  for (const r of runs) if (r.versions) { const k = `${r.versions.benchmark}/${r.versions.prompts}`; count.set(k, (count.get(k) ?? 0) + 1) }
  return [...count].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0]?.[0] ?? null
}

/** I-2.1: coverage, not proof. */
export function coverageOf(input: CandidateInput, cs: ComponentScores, stopReason?: InterpretData['stopReason']): Coverage {
  const clean = val(cs.cliff.practicalContextCeiling)
  const bad = cs.cliff.steps.find((s) => s.verdict !== 'pass')
  const firstObservedFailure = bad ? { ctx: bad.ctx, reason: bad.reasons.find((r) => r.code !== 'beyond_limit')?.message ?? `${bad.verdict}` } : null
  const skip = input.config.skippedSteps.filter((s) => s.ctx > (clean ?? 0)).sort((a, b) => a.ctx - b.ctx)[0]
  const firstPlannedSkip = skip ? { ctx: skip.ctx, reason: skip.reason, ...(skip.skip ? { resource: skip.skip.resource, estimateBytes: skip.skip.estimateBytes, budgetBytes: skip.skip.budgetBytes } : {}) } : null
  let limitKind: LimitKind = 'unknown'
  const codes = bad?.reasons.map((r) => r.code) ?? []
  if (bad && input.runs.some((r) => r.ctx === bad.ctx && r.status === 'cancelled')) limitKind = 'cancelled'
  else if (bad && bad.verdict === 'fail') limitKind = 'failure'
  else if (codes.some((c) => c === 'shared_spill' || c === 'vram_spill')) limitKind = 'spill'
  else if (bad) limitKind = 'cliff'
  else if (skip?.skip?.resource === 'vram' || (!skip?.skip && skip && /VRAM/.test(skip.reason))) limitKind = 'planned-skip:memory'
  else if (skip?.skip?.resource === 'ram' || (!skip?.skip && skip && /RAM/.test(skip.reason))) limitKind = 'planned-skip:ram'
  else if (stopReason === 'user-cap') limitKind = 'user-cap'
  else if (stopReason === 'cancelled' || stopReason === 'paused' || stopReason === 'interrupted') limitKind = 'cancelled'
  else if (clean !== null) limitKind = 'largest-tested'
  return { largestCleanTested: clean, firstObservedFailure, firstPlannedSkip, limitKind }
}

function scoreOf(input: CandidateInput, machine: MachineLimits, profile: WorkloadProfile, cfg: ScoringConfig, gq: GenQuality | undefined, scoringRung: number | null) {
  const cs = componentScores(input, machine, profile, cfg, gq, { scoringRung })
  const breakdown: BreakdownRow[] = (Object.keys(profile.weights) as ComponentId[]).map((k) => {
    const c = cs.components[k]
    return { component: k, input: c.input, score: c.score, weight: profile.weights[k], contribution: profile.weights[k] * c.score, ...(c.n !== undefined ? { n: c.n, ci95: c.ci95 } : {}) }
  })
  return { cs, breakdown, total: breakdown.reduce((s, r) => s + r.contribution, 0) }
}

const stripGen = (rows: readonly unknown[]): UncertaintyRow[] => (rows as UncertaintyRow[]).map(({ genId: _g, ...r }) => r as UncertaintyRow)

/** I-5.2: paired difference A − B on shared (testId, seed) items over the profile's categories. */
export function difference(a: CandidateVerdict | UncertaintyRow[], b: CandidateVerdict | UncertaintyRow[], profile: WorkloadProfile, cfg: ScoringConfig):
  { d: Difference | null; reason?: string } {
  const rows = (x: CandidateVerdict | UncertaintyRow[]) => (Array.isArray(x) ? stripGen(x) : x.qualityRows).filter((r) => profile.promptSetIds.includes(r.category))
  const weights = Object.fromEntries(profile.promptSetIds.map((c) => [c, cfg.qualityCategoryWeights[c]]))
  try {
    const p = pairedDifference(rows(a), rows(b), weights)
    return 'interval' in p ? { d: null, reason: p.reason } : { d: { diff: p.diff, lower: p.lower, upper: p.upper, sharedItems: p.sharedItems } }
  } catch (e) {
    return { d: null, reason: (e as Error).message }
  }
}
export const fmtDiff = (d: Difference) => {
  const s = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(Math.round(x))}`
  return `${s(d.diff)} [${s(d.lower)}, ${s(d.upper)}]`
}

export function verdicts(data: InterpretData, workload: WorkloadId, request: Request = {}, cfg: ScoringConfig = DEFAULT_SCORING_CONFIG): Verdicts {
  const profile = cfg.profiles[workload]
  const { machine } = data
  const round = (x: number) => Number(x.toFixed(cfg.tieDecimals))
  const userMin = request.minDecodeTps != null
  const sessionVersion = sessionVersionOf([...data.candidates.flatMap((c) => c.runs), ...(data.allRuns ?? [])])
  const excluded: Verdicts['excluded'] = []

  // I-6.0 + usable set
  const prepared = data.candidates.map((input) => {
    const dropped: { ctx: number; reason: string }[] = []
    const runs = input.runs.filter((r) => { const why = speedIneligible(r, sessionVersion); if (why) dropped.push({ ctx: r.ctx, reason: why }); return !why })
    const scored = { ...input, runs }
    return { input, scored, dropped, base: scoreOf(scored, machine, profile, cfg, undefined, null) }
  })
  const usable = prepared.filter((p) => {
    if (p.base.cs.usable) return true
    const why = p.base.cs.cliff.steps.flatMap((s) => s.reasons.map((r) => tag('stab.failures', r.message)))
    const drop = p.dropped.map((d) => tag('stab.cold-rows', `${fmtCtx(d.ctx)}: excluded from speed scoring — ${d.reason}`))
    excluded.push({ configId: p.input.config.id, reasons: [...why, ...drop].length ? [...why, ...drop] : [tag('stab.failures', 'no runs recorded')] })
    return false
  })

  // I-7.1: one scoring rung for the workload = min(target, largest clean context of any candidate).
  const ceilings = usable.map((p) => val(p.base.cs.cliff.practicalContextCeiling)).filter((x): x is number => x !== null)
  const scoringRung = ceilings.length ? Math.min(profile.targetContext, Math.max(...ceilings)) : null
  const scoringRungWhy = scoringRung === null ? 'no candidate has a clean rung; each is scored at its own smallest usable rung'
    : scoringRung === profile.targetContext ? `the ${profile.label} target ${fmtCtx(profile.targetContext)}`
      : `min(target ${fmtCtx(profile.targetContext)}, largest clean context of any candidate ${fmtCtx(Math.max(...ceilings))})`

  const tol = profile.latencyToleranceMs
  const genChoices: DecisionTrace['genChoices'] = []
  const all: CandidateVerdict[] = usable.map(({ input, scored, dropped }) => {
    const base = scoreOf(scored, machine, profile, cfg, undefined, scoringRung)
    // I-8.0 / I-8.1: generation configs — comparable only with verified applied template kwargs; a later config is
    // chosen only when its paired quality difference vs the current choice excludes 0 (and it answers within tolerance).
    const thinkingModel = !!(input.model.genKnobs?.supportsThinking || input.model.supportsThinking)
    const genOptions: GenOption[] = (input.genQuality ?? []).map((gq) => {
      const s = scoreOf(scored, machine, profile, cfg, gq, scoringRung)
      const lat = val(s.cs.components.latency.input, true)
      const applied = (gq.results as { appliedTemplateKwargs?: unknown }[]).some((r) => r.appliedTemplateKwargs)
      const comparable = !gq.gen.thinking || applied
      const quarantined = !!s.cs.components.quality.quarantined
      return {
        gq, cs: s.cs, total: s.total, comparable: comparable && !quarantined,
        withinTolerance: !gq.gen.thinking || !!profile.latencyAdvisory || (lat !== null && lat <= tol),
        ...(quarantined ? { why: 'quality quarantined (infrastructure error)' } : !comparable ? { why: 'template kwargs not verified as applied (identical renders or not recorded)' } : {})
      }
    })
    let gen: GenOption | null = null
    const gsteps: string[] = []
    for (const g of genOptions) {
      if (!g.comparable || !g.withinTolerance) { gsteps.push(`${g.gq.gen.id}: skipped (${g.why ?? 'time to answer above tolerance'})`); continue }
      if (!gen) { gen = g; gsteps.push(`${g.gq.gen.id}: first comparable config`); continue }
      const { d, reason } = difference(g.gq.results as UncertaintyRow[], gen.gq.results as UncertaintyRow[], profile, cfg)
      if (d && !includesZero(d) && d.diff > 0) { gsteps.push(`${g.gq.gen.id} over ${gen.gq.gen.id}: ${fmtDiff(d)}`); gen = g }
      else gsteps.push(`${g.gq.gen.id} not over ${gen.gq.gen.id}: ${d ? `${fmtDiff(d)} includes 0` : reason}`)
    }
    if (thinkingModel && genOptions.length) genChoices.push({ configId: input.config.id, chosen: gen?.gq.gen.id ?? null, steps: gsteps })
    const chosen = gen ? { cs: gen.cs, breakdown: scoreOf(scored, machine, profile, cfg, gen.gq, scoringRung).breakdown, total: gen.total } : base
    const ref = scored.runs.find((r) => r.ctx === chosen.cs.referenceCtx)
    const q = chosen.cs.components.quality
    const undecided = (Object.keys(profile.weights) as ComponentId[])
      .filter((k) => profile.weights[k] > 0 && chosen.cs.components[k].input.kind !== 'measured' && chosen.cs.components[k].input.kind !== 'declared')
      .map((k) => ({ component: k, kind: chosen.cs.components[k].quarantined ? 'quarantined' : chosen.cs.components[k].input.kind, reason: chosen.cs.components[k].input.reason ?? chosen.cs.components[k].input.source }))
    return {
      input, scored, dropped, cs: chosen.cs, gen, genOptions, breakdown: chosen.breakdown, total: chosen.total, eligible: true, failures: [],
      undecided, confirmed: undecided.length === 0,
      decode: val(ref?.decodeTps, true), vram: input.config.gpuLayers === 0 ? 0 : val(ref?.peakVramBytes), ram: val(ref?.peakRamBytes),
      qualityMeasured: q.input.kind === 'measured', qualityRows: q.input.kind === 'measured' ? stripGen(gen ? gen.gq.results : input.quality) : [],
      coverage: coverageOf(scored, chosen.cs, data.stopReason)
    }
  })

  // Hard constraints first (I-1.1), then soft gates.
  const fail = (v: CandidateVerdict, f: GateFailure) => { v.eligible = false; v.failures.push(f) }
  for (const v of all) {
    const { cs, coverage } = v
    const clean = coverage.largestCleanTested
    const pc = clean === null ? 'none' : fmtCtx(clean)
    if (profile.requiredContext) {
      if (clean === null || clean < profile.requiredContext) fail(v, { ruleId: rule('ctx.required').id, hard: true, user: true, text: tag('ctx.required', `largest clean context ${pc} < required ${fmtCtx(profile.requiredContext)} (${coverage.limitKind})`) })
    } else if (clean === null || clean < profile.targetContext * P('gate.context-floor', 'minCtxFraction')) {
      fail(v, { ruleId: rule('gate.context-floor').id, text: cite('gate.context-floor', { clean: pc, floor: fmtCtx(profile.targetContext * P('gate.context-floor', 'minCtxFraction')), workload: profile.label }) })
    }
    const ref = v.scored.runs.find((r) => r.ctx === cs.referenceCtx)
    if (profile.minDecodeTps !== undefined) {
      const floor = userMin ? `your floor ${profile.minDecodeTps} t/s` : `the ${profile.label} gate ${profile.minDecodeTps} t/s`
      if (v.decode === null) fail(v, { ruleId: rule('speed.decode-band').id, hard: true, user: userMin, notVerified: `decode floor ${profile.minDecodeTps} t/s`, text: tag('speed.decode-band', `decode at ${ref ? fmtCtx(ref.ctx) : '?'} not verified — ${floor} counts as not met`) })
      else if (v.decode < profile.minDecodeTps) fail(v, { ruleId: rule('speed.decode-band').id, hard: true, user: userMin, text: tag('speed.decode-band', `decode ${t1(v.decode)} t/s at ${fmtCtx(ref!.ctx)} is below ${floor}`) })
    }
    if (!profile.latencyAdvisory) {
      const tl = `${(tol / 1000).toFixed(0)} s`
      const t = val(ref?.ttftMs, true)
      if (t === null) fail(v, { ruleId: rule('speed.ttft-band').id, hard: true, notVerified: `latency tolerance ${tl}`, text: tag('speed.ttft-band', `TTFT at ${ref ? fmtCtx(ref.ctx) : '?'} not verified — the ${tl} tolerance counts as not met`) })
      else if (t > tol) fail(v, { ruleId: rule('speed.ttft-band').id, hard: true, text: tag('speed.ttft-band', `TTFT ${sec(t)} at ${fmtCtx(ref!.ctx)} exceeds the ${tl} tolerance`) })
      else if (cs.recommendedFits === false) fail(v, { ruleId: rule('speed.ttft-band').id, hard: true, text: tag('speed.ttft-band', `no passing rung ≤ ${fmtCtx(profile.maxContext ?? profile.targetContext)} has a measured TTFT within the ${tl} tolerance`) })
    }
    if (cs.components.stability.score < P('gate.stability', 'min')) fail(v, { ruleId: rule('gate.stability').id, text: cite('gate.stability', { s: cs.components.stability.score.toFixed(0), min: P('gate.stability', 'min') }) })
    const q = cs.components.quality
    if (v.qualityMeasured && q.score < profile.minQuality) fail(v, { ruleId: rule('gate.quality-min').id, text: cite('gate.quality-min', { q: q.score.toFixed(0), min: profile.minQuality, workload: profile.label }) })
  }
  // I-7.6: a partial config is dominated only by a same-model full offload that meets the same hard constraints.
  const hardOk = (v: CandidateVerdict) => !v.failures.some((f) => f.hard)
  for (const v of all) {
    if (v.input.config.gpuLayersAll || machine.gpuDevice === null) continue
    const full = all.filter((x) => x !== v && x.input.model.id === v.input.model.id && x.input.config.gpuLayersAll && hardOk(x)).sort(byId)[0]
    if (full) fail(v, { ruleId: rule('cmp.partial-veto').id, text: cite('cmp.partial-veto', { full: full.input.config.id }) })
  }

  // Ranking: tie-break chain (S2) total → decode → VRAM → configId.
  const chainCmp = (a: CandidateVerdict, b: CandidateVerdict, ta = a.total, tb = b.total) =>
    round(tb) - round(ta) || cmp(a.decode, b.decode, true) || cmp(a.vram, b.vram, false) || byId(a, b)
  const confirmed = all.filter((v) => v.eligible && v.confirmed).sort((a, b) => chainCmp(a, b))
  const provisional = all.filter((v) => v.eligible && !v.confirmed).sort((a, b) => chainCmp(a, b))
  const ineligible = all.filter((v) => !v.eligible).sort((a, b) => chainCmp(a, b))

  // I-5.2: paired quality difference; decisive (interval excludes 0) → quality decides for quality-weighted workloads;
  // otherwise the quality delta is neutralized (its contribution removed from both totals) and the rest decides.
  const steps: TraceStep[] = []
  const neutralizations: DecisionTrace['neutralizations'] = []
  let top: CandidateVerdict | null = confirmed[0] ?? null
  let quality: Verdicts['quality'] = null
  if (top) {
    steps.push({ ruleId: rule('cmp.decision-trace').id, kind: 'total', winner: top.input.config.id, detail: `highest total ${top.total.toFixed(1)} among confirmed eligible candidates` })
    const seen = new Set<string>()
    for (let iter = 0; iter < confirmed.length * confirmed.length; iter++) {
      let moved = false
      for (const x of confirmed) {
        if (x === top) continue
        const both = x.qualityMeasured && top.qualityMeasured
        const { d, reason } = both ? difference(x, top, profile, cfg) : { d: null, reason: 'quality not measured for both' }
        const decisive = !!d && !includesZero(d)
        if (decisive && d!.diff > 0 && profile.qualityFirst) {
          steps.push({ ruleId: rule('quality.difference').id, kind: 'quality-decides', winner: x.input.config.id, over: top.input.config.id, detail: `paired quality difference ${fmtDiff(d!)} excludes 0`, difference: d })
          top = x; moved = true; break
        }
        if (!decisive) {
          const nx = x.total - qContribution(x), nt = top.total - qContribution(top)
          const key = [x.input.config.id, top.input.config.id].sort().join('|')
          const better = chainCmp(x, top, nx, nt) < 0
          if (!seen.has(key)) {
            seen.add(key)
            neutralizations.push({ a: x.input.config.id, b: top.input.config.id, difference: d, ...(reason ? { reason } : {}), totalsWithoutQuality: { [x.input.config.id]: Number(nx.toFixed(3)), [top.input.config.id]: Number(nt.toFixed(3)) }, winner: better ? x.input.config.id : top.input.config.id })
          }
          if (better) {
            steps.push({ ruleId: rule('quality.difference').id, kind: 'quality-neutralized', winner: x.input.config.id, over: top.input.config.id, detail: `quality indistinguishable (${d ? `${fmtDiff(d)} includes 0` : reason}); without its contribution ${nx.toFixed(1)} vs ${nt.toFixed(1)}`, difference: d })
            top = x; moved = true; break
          }
        } else if (!profile.qualityFirst && chainCmp(x, top) < 0) { // quality-weighted: a decisive quality gap is never outvoted
          steps.push({ ruleId: rule('cmp.decision-trace').id, kind: 'total', winner: x.input.config.id, over: top.input.config.id, detail: `decisive quality difference ${fmtDiff(d!)} does not override the total for ${profile.label}`, difference: d })
          top = x; moved = true; break
        }
      }
      if (!moved) break
    }
    // The comparison the reasons explain: winner vs the best other confirmed candidate.
    const rival = confirmed.find((v) => v !== top)
    if (rival && rival.qualityMeasured && top.qualityMeasured) {
      const { d, reason } = difference(top, rival, profile, cfg)
      quality = { decisive: !!d && !includesZero(d), a: top, b: rival, difference: d, ...(reason ? { reason } : {}) }
    }
    confirmed.splice(confirmed.indexOf(top), 1)
    confirmed.unshift(top)
  }

  // Tie-break chain actually walked between the winner and the runner-up.
  const tieBreakChain: DecisionTrace['tieBreakChain'] = []
  if (confirmed.length > 1) {
    const [a, b] = confirmed
    const links: [string, number | string | null, number | string | null, number][] = [
      ['total', round(a.total), round(b.total), round(b.total) - round(a.total)],
      ['decode', a.decode, b.decode, cmp(a.decode, b.decode, true)],
      ['vram', a.vram, b.vram, cmp(a.vram, b.vram, false)],
      ['configId', a.input.config.id, b.input.config.id, byId(a, b)]
    ]
    for (const [step, av, bv, c] of links) { tieBreakChain.push({ step, a: a.input.config.id, b: b.input.config.id, a_value: av, b_value: bv, decided: c !== 0 }); if (c !== 0) break }
  }

  // I-2.5 fallback: only over the workload's own speed gates — never over a user floor (unmet alternative instead).
  let fallback = false
  const unmet: CandidateVerdict[] = []
  if (!top && profile.requiredContext) {
    const reach = all.filter((v) => v.confirmed && (v.coverage.largestCleanTested ?? 0) >= profile.requiredContext!)
    const speedOnly = (v: CandidateVerdict) => v.failures.every((f) => f.ruleId === rule('speed.decode-band').id || f.ruleId === rule('speed.ttft-band').id)
    const byDecode = (a: CandidateVerdict, b: CandidateVerdict) => cmp(a.decode, b.decode, true) || byId(a, b)
    top = reach.filter((v) => speedOnly(v) && !v.failures.some((f) => f.user)).sort(byDecode)[0] ?? null
    fallback = !!top
    if (top) steps.push({ ruleId: rule('ctx.required').id, kind: 'fallback', winner: top.input.config.id, detail: 'meets the required context; below the workload speed gate' })
    unmet.push(...reach.filter((v) => speedOnly(v) && v.failures.some((f) => f.user)).sort(byDecode))
  }

  const pick = (f: (a: CandidateVerdict, b: CandidateVerdict) => number) => [...all.filter((v) => v.eligible && v.confirmed)].sort((a, b) => f(a, b) || byId(a, b))[0]?.input.config.id ?? null
  const alternatives: DecisionTrace['alternatives'] = {
    fastest: { configId: pick((a, b) => cmp(a.decode, b.decode, true)), ruleId: rule('speed.decode-band').id },
    bestQuality: { configId: pick((a, b) => Q(b).score - Q(a).score), ruleId: rule('quality.report').id },
    bestLongContext: { configId: pick((a, b) => cmp(a.coverage.largestCleanTested, b.coverage.largestCleanTested, true) || cmp(a.decode, b.decode, true)), ruleId: rule('ctx.coverage').id },
    lowestMemory: { configId: pick((a, b) => cmp(a.vram, b.vram, false) || cmp(a.ram, b.ram, false)), ruleId: rule('mem.budget-basis').id }
  }

  const ranked = [...confirmed, ...provisional, ...ineligible]
  const thresholdsUsed: DecisionTrace['thresholdsUsed'] = {
    minDecodeTps: profile.minDecodeTps ?? null, latencyToleranceMs: tol, latencyAdvisory: !!profile.latencyAdvisory, minQuality: profile.minQuality,
    requiredContext: profile.requiredContext ?? null, qualityFirst: !!profile.qualityFirst, targetContext: profile.targetContext
  }
  for (const r of RULES) for (const [k, x] of Object.entries(r.params)) thresholdsUsed[`${r.id}.${k}`] = x
  const trace: DecisionTrace = {
    rulesVersion: RULES_VERSION, scoringVersion: cfg.version, workload, scoringRung, scoringRungWhy,
    hardConstraints: {
      requiredContext: profile.requiredContext ?? null,
      minDecodeTps: profile.minDecodeTps === undefined ? null : { value: profile.minDecodeTps, source: userMin ? 'user' : 'workload' },
      latencyToleranceMs: tol, latencyAdvisory: !!profile.latencyAdvisory
    },
    eligibleSet: ranked.map((v) => ({ configId: v.input.config.id, failingRuleIds: [...new Set(v.failures.map((f) => f.ruleId))] })),
    candidates: ranked.map((v) => ({
      configId: v.input.config.id, confirmed: v.confirmed, undecided: v.undecided.map((u) => `${u.component} (${u.kind})`), total: Number(v.total.toFixed(3)),
      qualityContribution: Number(qContribution(v).toFixed(3)), gen: v.gen?.gq.gen.id ?? null, referenceCtx: v.cs.referenceCtx, failures: v.failures.map((f) => f.text)
    })),
    steps, neutralizations, tieBreakChain, alternatives,
    winner: top?.input.config.id ?? null, provisionalWinner: provisional[0]?.input.config.id ?? null,
    unmetAlternatives: unmet.map((v) => ({ configId: v.input.config.id, unmet: v.failures.filter((f) => f.user).map((f) => f.text) })),
    genChoices, thresholdsUsed
  }
  return {
    workload, profile, cfg, data, request, ranked, provisional, excluded, winner: top, provisionalWinner: provisional[0] ?? null, fallback, unmet, trace, quality, scoringRung, sessionVersion
  }
}

export { genLabel }
