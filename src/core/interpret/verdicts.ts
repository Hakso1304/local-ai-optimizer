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
import { DEFAULT_SCORING_CONFIG, effectiveProfile, withProfile, type ScoringConfig } from '../scoring/workloads'
import { BASELINE_GEN, genLabel, templateKwargsFor } from '../benchmark/gen'
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
  /** The session's own benchmark/prompt versions (I-6.0); default = those of the candidates' latest rows. */
  sessionVersions?: { benchmark: string; prompts: string }
}
export interface Request { requiredContext?: number | null; minDecodeTps?: number | null }

export interface GateFailure { ruleId: string; text: string; /** a user-set hard constraint (never overridden) */ user?: boolean; hard?: boolean; /** evidence missing → counted as not met (I-1.1) */ notVerified?: string }
export interface Difference { diff: number; lower: number; upper: number; sharedItems: number }
export interface GenOption { gq: GenQuality; cs: ComponentScores; total: number; withinTolerance: boolean; comparable: boolean; why?: string }
export type LimitKind = 'failure' | 'spill' | 'cliff' | 'planned-skip:memory' | 'planned-skip:ram' | 'user-cap' | 'cancelled' | 'largest-tested' | 'unknown'
export interface Coverage {
  largestCleanTested: number | null
  firstObservedFailure: { ctx: number; reason: string } | null
  firstPlannedSkip: { ctx: number; reason: string; resource?: string; estimateBytes?: number; budgetBytes?: number; weightsBytes?: number; kvBytes?: number; overheadBytes?: number } | null
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
  candidates: {
    configId: string; confirmed: boolean; undecided: string[]; total: number; qualityContribution: number; gen: string | null
    referenceCtx: number | null; referenceWhy: string | null; recommendedCtx: number | null; recommendedWhy: string | null; failures: string[]
    basis: { component: string; rung: number | null; kind: string; scope?: string }[]
    safety: { ramFloor: 'ok' | 'violated' | 'not verified'; spill: 'measured' | 'not verified' }
    qualityVsWinner: { difference: Difference | null; reason?: string } | null
  }[]
  steps: TraceStep[]
  neutralizations: { a: string; b: string; difference: Difference | null; reason?: string; totalsWithoutQuality: Record<string, number>; winner: string }[]
  tieBreakChain: { step: string; a: string; b: string; a_value: number | string | null; b_value: number | string | null; decided: boolean }[]
  alternatives: Record<'fastest' | 'bestQuality' | 'bestLongContext' | 'lowestMemory', { configId: string | null; ruleId: string }>
  winner: string | null
  provisionalWinner: string | null
  unmetAlternatives: { configId: string; unmet: string[] }[]
  genChoices: { configId: string; chosen: string | null; steps: string[] }[]
  thresholdsUsed: Record<string, number | boolean | null>
  /** Every pairwise comparison as it happened, with the totals actually compared (G05). */
  comparisons: { a: string; b: string; basis: 'total' | 'without-quality' | 'quality'; aValue: number; bValue: number; difference: Difference | null; reason?: string; winner: string; ruleId: string }[]
  /** Set when the pairwise walk revisited a leader: the order is then not a proven total order. */
  cycle?: string[]
  excluded: { configId: string; reasons: string[] }[]
  qualityVsSpeed: { winner: string; fastest: string; difference: Difference | null; reason?: string; decodeWinner: number; decodeFastest: number } | null
  /** F3: provisional candidates evaluated without their undecided terms against the confirmed winner. */
  counterfactuals: { configId: string; without: string[]; total: number; vs: string | null; vsTotal: number | null; result: string }[]
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
  /** Coverage of EVERY prepared candidate, excluded ones included (F10). */
  coverageAll: { input: CandidateInput; dropped: { ctx: number; reason: string }[]; coverage: Coverage; excluded: boolean }[]
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
  const ok = (x: unknown) => typeof x === 'string' && x.trim().length > 0
  const v = r.versions && ok(r.versions.benchmark) && ok(r.versions.prompts) ? `${r.versions.benchmark}/${r.versions.prompts}` : null
  if (!v) return `benchmark/prompt versions not recorded${sessionVersion ? ` (session ${sessionVersion})` : ''} — no version proof`
  if (sessionVersion && v !== sessionVersion) return `benchmark/prompt version ${v} differs from the session's ${sessionVersion}`
  return null
}

function sessionVersionOf(runs: BenchmarkRunResult[]): string | null {
  const count = new Map<string, number>()
  const ok = (x: unknown) => typeof x === 'string' && x.trim().length > 0
  for (const r of runs) if (r.versions && ok(r.versions.benchmark) && ok(r.versions.prompts)) { const k = `${r.versions.benchmark}/${r.versions.prompts}`; count.set(k, (count.get(k) ?? 0) + 1) }
  return [...count].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0]?.[0] ?? null
}

type ContractRow = UncertaintyRow & { appliedTemplateKwargs?: Record<string, unknown>; templateHash?: string | null; runtimeVersion?: string | null; modelFingerprint?: string | null; acceptedSampling?: Record<string, unknown> | null }
/** F5: what is missing for a generation config to be comparable with its baseline (empty = contract met). */
/** I-8.0 contract check for one generation config's rows: contradictions (errors — the rows say something other
 *  than the config) are separated from absent proof (missing — not evaluable). kwargs are compared with what the
 *  template should have received for this config (templateKwargsFor); a template without thinking = not applicable. */
function contractCheck(rows: ContractRow[], base: ContractRow[], gen: GenQuality['gen'], expected: Record<string, unknown> | undefined): { errors: string[]; missing: string[] } {
  const errors: string[] = [], missing: string[] = []
  if (!rows.length) return { errors, missing: ['no rows'] }
  if (expected) {
    const applied = rows.filter((r) => r.appliedTemplateKwargs && Object.keys(r.appliedTemplateKwargs).length > 0)
    const bad = applied.find((r) => Object.entries(expected).some(([k, x]) => r.appliedTemplateKwargs![k] !== x) || Object.keys(r.appliedTemplateKwargs!).some((k) => !(k in expected)))
    if (bad) errors.push(`applied template kwargs ${JSON.stringify(bad.appliedTemplateKwargs)} differ from the config's ${JSON.stringify(expected)}`)
    if (applied.length < rows.length) missing.push('applied template kwargs not verified on every row')
  }
  for (const k of ['templateHash', 'runtimeVersion', 'modelFingerprint'] as const) {
    const present = [...rows, ...base].filter((r) => !!r[k]).map((r) => r[k])
    if (new Set(present).size > 1) errors.push(`${k} not uniform across this config and its baseline (${[...new Set(present)].join(', ')})`)
    if (!rows.every((r) => !!r[k])) missing.push(`${k} not recorded`)
  }
  // Required fields: temperature always, plus every sampler the config sets — present and finite on EVERY row, and
  // matching. Absent (e.g. only {seed}) → not evaluable; a different value → contradiction.
  const want: [string, number][] = [['temperature', gen.temperature ?? 0], ...([['top_p', gen.topP], ['top_k', gen.topK], ['min_p', gen.minP]] as [string, number | undefined][]).filter((x): x is [string, number] => x[1] !== undefined)]
  for (const [k, x] of want) {
    const got = rows.map((r) => r.acceptedSampling?.[k])
    if (got.some((g) => typeof g === 'number' && Number.isFinite(g) && Math.abs(g - x) > 1e-6)) errors.push(`accepted ${k} differs from the config's ${x}`)
    if (got.some((g) => !(typeof g === 'number' && Number.isFinite(g)))) missing.push(`runtime-accepted ${k} not recorded on every row`)
  }
  return { errors, missing }
}

/** R3: the quality observation scope: one ctx, mixed ctxs, or unknown. */
function qualityScope(v: CandidateVerdict): string {
  const all = (v.qualityRows as (UncertaintyRow & { ctx?: number })[]).map((r) => r.ctx)
  const cs = [...new Set(all.filter((x): x is number => typeof x === 'number' && x > 0))].sort((a, b) => a - b)
  const without = all.filter((x) => !(typeof x === 'number' && x > 0)).length
  if (cs.length === 0) return 'unknown (quality ctx not recorded)'
  if (cs.length > 1) return `mixed (${cs.join(', ')})${without ? `; ${without} of ${all.length} rows without ctx` : ''}`
  return without ? `partial (${all.length - without} of ${all.length} rows carry ctx ${cs[0]})` : `ctx ${cs[0]}`
}

/** F3: the rung each component was actually observed at. */
function basisRung(v: CandidateVerdict, k: ComponentId): number | null {
  if (k === 'quality') {
    const ctxs = (v.qualityRows as (UncertaintyRow & { ctx?: number })[]).map((r) => r.ctx)
    return ctxs.length > 0 && ctxs.every((x) => typeof x === 'number' && x > 0 && x === ctxs[0]) ? ctxs[0]! : null
  }
  if (k === 'context') return v.coverage.largestCleanTested
  if (k === 'stability') return null
  return v.cs.referenceCtx
}

/** I-2.1: coverage, not proof. */
export function coverageOf(input: CandidateInput, cs: ComponentScores, stopReason?: InterpretData['stopReason']): Coverage {
  const clean = val(cs.cliff.practicalContextCeiling)
  const bad = cs.cliff.steps.find((s) => s.verdict !== 'pass')
  const firstObservedFailure = bad ? { ctx: bad.ctx, reason: bad.reasons.find((r) => r.code !== 'beyond_limit')?.message ?? `${bad.verdict}` } : null
  const skip = input.config.skippedSteps.filter((s) => s.ctx > (clean ?? 0)).sort((a, b) => a.ctx - b.ctx)[0]
  const firstPlannedSkip = skip ? { ctx: skip.ctx, reason: skip.reason, ...(skip.skip ? { resource: skip.skip.resource, estimateBytes: skip.skip.estimateBytes, budgetBytes: skip.skip.budgetBytes, weightsBytes: skip.skip.weightsBytes, kvBytes: skip.skip.kvBytes, overheadBytes: skip.skip.overheadBytes } : {}) } : null
  let limitKind: LimitKind = 'unknown'
  const codes = bad?.reasons.map((r) => r.code) ?? []
  if (bad && input.runs.some((r) => r.ctx === bad.ctx && r.status === 'cancelled')) limitKind = 'cancelled'
  else if (bad && bad.verdict === 'fail') limitKind = 'failure'
  else if (codes.some((c) => c === 'shared_spill' || c === 'vram_spill')) limitKind = 'spill'
  else if (bad) limitKind = 'cliff'
  // A cap below the first planned skip is the binding stop (the planner's skip is then not what ended the ladder).
  else if (stopReason === 'user-cap' && skip && clean !== null && skip.ctx > clean * 2) limitKind = 'user-cap'
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
type ScopedRow = UncertaintyRow & { checkerVersion?: string; maxTokens?: number }

/** I-7.1 scope of a paired comparison: same generation config (between candidates), same checker/suite version and
 *  token budget on every shared item. Returns the mismatch, or null. */
function scopeMismatch(A: ScopedRow[], B: ScopedRow[], scope: 'candidates' | 'gen'): string | null {
  const gid = (rs: ScopedRow[]) => [...new Set(rs.map((r) => r.genId ?? 'off'))].sort().join(',')
  if (scope === 'candidates' && gid(A) !== gid(B)) return `generation configs differ (${gid(A)} vs ${gid(B)})`
  const key = (r: ScopedRow) => `${r.testId}|${r.generatorSeed ?? ''}`
  const other = new Map(B.map((r) => [key(r), r]))
  for (const r of A) {
    const o = other.get(key(r))
    if (!o) continue
    if (r.checkerVersion && o.checkerVersion && r.checkerVersion !== o.checkerVersion) return `checker/suite versions differ (${r.checkerVersion} vs ${o.checkerVersion})`
    if (scope === 'candidates' && r.maxTokens && o.maxTokens && r.maxTokens !== o.maxTokens) return `token budgets differ on ${r.testId} (${r.maxTokens} vs ${o.maxTokens})`
    const x = r as ScopedRow & { ctx?: number; templateHash?: string | null; kvType?: string }, y = o as typeof x
    if (x.ctx && y.ctx && x.ctx !== y.ctx) return `quality contexts differ (${x.ctx} vs ${y.ctx})`
    if (x.templateHash && y.templateHash && x.templateHash !== y.templateHash) return 'chat templates differ (template hash)'
    if (x.kvType && y.kvType && x.kvType !== y.kvType) return `KV types differ (${x.kvType} vs ${y.kvType})`
  }
  return null
}

/** I-5.2: paired difference A − B on shared (testId, seed) items over the profile's categories, after the scope check. */
export function difference(a: CandidateVerdict | UncertaintyRow[], b: CandidateVerdict | UncertaintyRow[], profile: WorkloadProfile, cfg: ScoringConfig, scope: 'candidates' | 'gen' = 'candidates'):
  { d: Difference | null; reason?: string } {
  const raw = (x: CandidateVerdict | UncertaintyRow[]) => ((Array.isArray(x) ? x : x.qualityRows) as ScopedRow[]).filter((r) => profile.promptSetIds.includes(r.category))
  // R1: the persisted configs' KV types are part of the scope, whether or not the rows repeat the field.
  const kvOf = (x: CandidateVerdict | UncertaintyRow[]) => (Array.isArray(x) ? null : x.input.config.kvType)
  const ka = kvOf(a), kb = kvOf(b)
  const mismatch = scope === 'candidates' && ka && kb && ka !== kb ? `KV types differ (${ka} vs ${kb})` : scopeMismatch(raw(a), raw(b), scope)
  if (mismatch) return { d: null, reason: `not comparable: ${mismatch}` }
  const rows = (x: CandidateVerdict | UncertaintyRow[]) => stripGen(raw(x))
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

export function verdicts(data: InterpretData, workload: WorkloadId, request: Request = {}, cfgIn: ScoringConfig = DEFAULT_SCORING_CONFIG): Verdicts {
  // G04: request overrides (required context, decode floor) are applied here, for every caller (idempotent).
  const cfg = withProfile(cfgIn, effectiveProfile(cfgIn.profiles[workload], request))
  const profile = cfg.profiles[workload]
  const { machine } = data
  const round = (x: number) => Number(x.toFixed(cfg.tieDecimals))
  const userMin = request.minDecodeTps != null
  // G09: the session's identity, not a majority over history (superseded attempts must not re-define it).
  const sessionVersion = data.sessionVersions ? `${data.sessionVersions.benchmark}/${data.sessionVersions.prompts}` : sessionVersionOf(data.candidates.flatMap((c) => c.runs))
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
    const efforts = input.model.genKnobs?.effortValues ?? []
    const order = (g: GenQuality) => (g.gen.thinking ? 1 + (g.gen.effort && efforts.includes(g.gen.effort) ? efforts.indexOf(g.gen.effort) : efforts.length) : 0)
    const sortedGens = [...(input.genQuality ?? [])].sort((a, b) => order(a) - order(b) || (a.gen.id < b.gen.id ? -1 : a.gen.id > b.gen.id ? 1 : 0))
    const genOptions: GenOption[] = sortedGens.map((gq) => {
      const s = scoreOf(scored, machine, profile, cfg, gq, scoringRung)
      const lat = val(s.cs.components.latency.input, true)
      // F5 (I-8.0): the full application contract on every row — applied kwargs, template/runtime/model identity and
      // the sampling the runtime accepted — and the same identities as the baseline it is compared with.
      // A (w4l): every config, the baseline included — a contradiction always excludes it; absent proof excludes
      // only thinking configs (the baseline then stands, comparisons with it are not evaluable).
      const off = sortedGens.find((x) => !x.gen.thinking)
      const base = (gq.gen.thinking ? (off?.results ?? input.quality) : []) as ContractRow[]
      const chk = contractCheck(gq.results as ContractRow[], base, gq.gen, templateKwargsFor(input.model, gq.gen))
      // A thinking result is comparable only to a baseline whose own requested
      // sampling and template kwargs were verified on every row. The baseline
      // may still stand alone with missing proof, but it cannot prove a delta.
      const offCheck = gq.gen.thinking ? contractCheck(base, [], off?.gen ?? BASELINE_GEN, templateKwargsFor(input.model, off?.gen ?? BASELINE_GEN)) : null
      const missing = [...chk.errors, ...(gq.gen.thinking ? chk.missing : [])]
      if (offCheck) missing.push(...offCheck.errors.map((x) => `baseline: ${x}`), ...offCheck.missing.map((x) => `baseline: ${x}`))
      const comparable = missing.length === 0
      const quarantined = !!s.cs.components.quality.quarantined
      return {
        gq, cs: s.cs, total: s.total, comparable: comparable && !quarantined,
        withinTolerance: !gq.gen.thinking || !!profile.latencyAdvisory || (lat !== null && lat <= tol),
        ...(quarantined ? { why: 'quality quarantined (infrastructure error)' } : !comparable ? { why: `application contract not met: ${missing.join('; ')}` } : {})
      }
    })
    let gen: GenOption | null = null
    const gsteps: string[] = []
    for (const g of genOptions) {
      if (!g.comparable || !g.withinTolerance) { gsteps.push(`${g.gq.gen.id}: skipped (${g.why ?? 'time to answer above tolerance'})`); continue }
      if (!gen) { gen = g; gsteps.push(`${g.gq.gen.id}: first comparable config`); continue }
      const { d, reason } = difference(g.gq.results as UncertaintyRow[], gen.gq.results as UncertaintyRow[], profile, cfg, 'gen')
      if (d && !includesZero(d) && d.diff > 0) { gsteps.push(`${g.gq.gen.id} over ${gen.gq.gen.id}: ${fmtDiff(d)} excludes 0`); gen = g }
      else gsteps.push(`${g.gq.gen.id} not over ${gen.gq.gen.id}: ${d ? `${fmtDiff(d)} ${includesZero(d) ? 'includes 0' : 'excludes 0 (lower)'}; the lower effort is kept` : reason}`)
    }
    if (thinkingModel && genOptions.length) genChoices.push({ configId: input.config.id, chosen: gen?.gq.gen.id ?? null, steps: gsteps })
    const selected = gen ? { cs: gen.cs, breakdown: scoreOf(scored, machine, profile, cfg, gen.gq, scoringRung).breakdown, total: gen.total } : base
    // A (w4l): the rows the quality term comes from must not contradict the config they claim (sampling / kwargs).
    // If every option was rejected, the baseline rows still carry the request's
    // explicit off sampling contract (which may be T=1), not the default T=0.
    const qConfig = gen?.gq.gen ?? sortedGens.find((x) => !x.gen.thinking)?.gen ?? BASELINE_GEN
    const qRows = (gen ? gen.gq.results : input.quality) as ContractRow[]
    const qErrors = contractCheck(qRows, [], qConfig, templateKwargsFor(input.model, qConfig)).errors
    // A contradictory baseline cannot remain a measured quality contribution merely
    // because no alternative generation config was selected. Preserve the rows for
    // audit, but score this candidate with quality unavailable.
    const chosen = qErrors.length ? scoreOf({ ...scored, quality: [], genQuality: [] }, machine, profile, cfg, undefined, scoringRung) : selected
    const ref = scored.runs.find((r) => r.ctx === chosen.cs.referenceCtx)
    const q = chosen.cs.components.quality
    // G01: a decisive term confirms only when MEASURED (declared / estimated / unavailable / quarantined never do).
    const undecided = (Object.keys(profile.weights) as ComponentId[])
      .filter((k) => profile.weights[k] > 0 && chosen.cs.components[k].input.kind !== 'measured' && !(k === 'quality' && qErrors.length))
      .map((k) => ({ component: k, kind: chosen.cs.components[k].quarantined ? 'quarantined' : chosen.cs.components[k].input.kind, reason: chosen.cs.components[k].input.reason ?? chosen.cs.components[k].input.source }))
    if (qErrors.length && profile.weights.quality > 0) undecided.push({ component: 'quality', kind: 'contract-error', reason: qErrors.join('; ') })
    // G03: read at another rung than the common one (its ladder skipped it) → the speed comparison is not matched.
    // F2: any substitution (ladder skipped the rung, or a shorter ladder) leaves the speed/latency basis unmatched.
    if (scoringRung !== null && chosen.cs.referenceCtx !== scoringRung) {
      for (const k of ['genSpeed', 'latency', 'prefillSpeed'] as ComponentId[]) if (profile.weights[k] > 0) undecided.push({ component: k, kind: 'unmatched-rung', reason: `measured at ${chosen.cs.referenceCtx === null ? 'no rung' : fmtCtx(chosen.cs.referenceCtx)}, not at the common rung ${fmtCtx(scoringRung)}` })
    }
    return {
      input, scored, dropped, cs: chosen.cs, gen, genOptions, breakdown: chosen.breakdown, total: chosen.total, eligible: true, failures: [],
      undecided, confirmed: undecided.length === 0,
      decode: val(ref?.decodeTps, true), vram: input.config.gpuLayers === 0 ? 0 : val(ref?.peakVramBytes), ram: val(ref?.peakRamBytes),
      qualityMeasured: !qErrors.length && q.input.kind === 'measured', qualityRows: !qErrors.length && q.input.kind === 'measured' ? ((gen ? gen.gq.results : input.quality) as UncertaintyRow[]) : [],
      coverage: coverageOf(scored, chosen.cs, data.stopReason)
    }
  })

  // F2: at the common rung the prompts must be comparable in size (±5 % of the median); unknown size = unmatched.
  if (scoringRung !== null) {
    const at = all.filter((v) => v.cs.referenceCtx === scoringRung)
    const sizes = at.map((v) => v.scored.runs.find((r) => r.ctx === scoringRung)?.promptTokens ?? null)
    const known = sizes.filter((x): x is number => typeof x === 'number')
    // Reference size: the largest cluster of mutually comparable prompts (ties → the larger, i.e. the intended fill).
    const near = (x: number, y: number) => Math.abs(x - y) / Math.max(x, y) <= 0.05
    const med = known.length ? [...known].sort((x, y) => known.filter((k) => near(k, y)).length - known.filter((k) => near(k, x)).length || y - x)[0] : null
    at.forEach((v, i) => {
      const n = sizes[i]
      const bad = n === null ? 'prompt size not recorded' : med !== null && !near(n, med) ? `prompt ${n} tokens vs ${med} for the others` : null
      if (bad) for (const k of ['genSpeed', 'latency', 'prefillSpeed'] as ComponentId[]) if (profile.weights[k] > 0 && !v.undecided.some((u) => u.component === k)) v.undecided.push({ component: k, kind: 'unmatched-prompt', reason: bad })
      v.confirmed = v.undecided.length === 0
    })
  }

  // Hard constraints first (I-1.1), then soft gates.
  const fail = (v: CandidateVerdict, f: GateFailure) => { v.eligible = false; v.failures.push(f) }
  const safety = new Map<CandidateVerdict, DecisionTrace['candidates'][number]['safety']>()
  for (const v of all) {
    // G01 safety verdict: a measured RAM minimum below the recorded floor on a rung this pick relies on is unsafe;
    // absent evidence is "not verified" (the memory term is then unavailable → provisional).
    const upTo = Math.max(v.cs.referenceCtx ?? 0, v.cs.recommendedCtx ?? 0)
    let ram: 'ok' | 'violated' | 'not verified' = 'ok'
    for (const r of v.scored.runs.filter((x) => x.ctx <= upTo)) {
      const floor = r.ramFloorBytes ?? data.planningSnapshot?.ramFloorBytes
      const min = r.minRamAvailBytes
      if (floor === undefined || !min || min.kind !== 'measured' || min.value === null) { if (ram === 'ok') ram = 'not verified'; continue }
      if (min.value < floor) {
        ram = 'violated'
        fail(v, { ruleId: rule('mem.ram-floor').id, hard: true, text: tag('mem.ram-floor', `RAM available fell to ${(min.value / 1024 ** 3).toFixed(2)} GiB at ${fmtCtx(r.ctx)}, below the ${(floor / 1024 ** 3).toFixed(2)} GiB floor — unsafe`) })
        break
      }
    }
    // F1: unknown RAM safety is not met (I-1.1) — independent of score weights; it also cannot veto (hard).
    if (ram === 'not verified') fail(v, { ruleId: rule('mem.ram-floor').id, hard: true, notVerified: 'RAM safety floor', text: tag('mem.ram-floor', `RAM safety not verified up to ${fmtCtx(upTo)} (no recorded floor or measured minimum) — counts as not met`) })
    const recRun = v.scored.runs.find((r) => r.ctx === (v.cs.recommendedCtx ?? v.cs.referenceCtx))
    safety.set(v, { ramFloor: ram, spill: recRun?.peakSharedGpuBytes.kind === 'measured' ? 'measured' : 'not verified' })
  }
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
  const comparisons: DecisionTrace['comparisons'] = []
  let cycle: string[] | undefined
  let top: CandidateVerdict | null = confirmed[0] ?? null
  let quality: Verdicts['quality'] = null
  // The basis each pair was actually decided on (G05): totals, totals without quality, or quality.
  const basisOf = new Map<string, { basis: 'total' | 'without-quality' | 'quality'; ta: number; tb: number }>()
  const pairKey = (a: CandidateVerdict, b: CandidateVerdict) => [a.input.config.id, b.input.config.id].sort().join('|')
  if (top) {
    steps.push({ ruleId: rule('cmp.decision-trace').id, kind: 'total', winner: top.input.config.id, detail: `highest total ${top.total.toFixed(1)} among confirmed eligible candidates` })
    const leaders = [top.input.config.id]
    const seen = new Set<string>()
    for (let iter = 0; iter < confirmed.length * confirmed.length && !cycle; iter++) {
      let moved = false
      for (const x of confirmed) {
        if (x === top) continue
        const key = pairKey(x, top)
        const both = x.qualityMeasured && top.qualityMeasured
        const { d, reason } = both ? difference(x, top, profile, cfg) : { d: null, reason: 'quality not measured for both' }
        const decisive = !!d && !includesZero(d)
        let winner: CandidateVerdict = top
        if (decisive && profile.qualityFirst) {
          if (d!.diff > 0) winner = x
          if (!seen.has(key)) comparisons.push({ a: x.input.config.id, b: top.input.config.id, basis: 'quality', aValue: d!.diff, bValue: 0, difference: d, winner: winner.input.config.id, ruleId: rule('quality.difference').id })
          basisOf.set(key, { basis: 'quality', ta: x.total, tb: top.total })
          if (winner === x) steps.push({ ruleId: rule('quality.difference').id, kind: 'quality-decides', winner: x.input.config.id, over: top.input.config.id, detail: `paired quality difference ${fmtDiff(d!)} excludes 0`, difference: d })
        } else if (!decisive) {
          const nx = x.total - qContribution(x), nt = top.total - qContribution(top)
          if (chainCmp(x, top, nx, nt) < 0) winner = x
          if (!seen.has(key)) {
            neutralizations.push({ a: x.input.config.id, b: top.input.config.id, difference: d, ...(reason ? { reason } : {}), totalsWithoutQuality: { [x.input.config.id]: Number(nx.toFixed(3)), [top.input.config.id]: Number(nt.toFixed(3)) }, winner: winner.input.config.id })
            comparisons.push({ a: x.input.config.id, b: top.input.config.id, basis: 'without-quality', aValue: Number(nx.toFixed(3)), bValue: Number(nt.toFixed(3)), difference: d, ...(reason ? { reason } : {}), winner: winner.input.config.id, ruleId: rule('quality.difference').id })
          }
          basisOf.set(key, { basis: 'without-quality', ta: nx, tb: nt })
          if (winner === x) steps.push({ ruleId: rule('quality.difference').id, kind: 'quality-neutralized', winner: x.input.config.id, over: top.input.config.id, detail: `quality indistinguishable (${d ? `${fmtDiff(d)} includes 0` : reason}); without its contribution ${nx.toFixed(1)} vs ${nt.toFixed(1)}`, difference: d })
        } else {
          if (chainCmp(x, top) < 0) winner = x
          if (!seen.has(key)) comparisons.push({ a: x.input.config.id, b: top.input.config.id, basis: 'total', aValue: Number(x.total.toFixed(3)), bValue: Number(top.total.toFixed(3)), difference: d, winner: winner.input.config.id, ruleId: rule('cmp.decision-trace').id })
          basisOf.set(key, { basis: 'total', ta: x.total, tb: top.total })
          if (winner === x) steps.push({ ruleId: rule('cmp.decision-trace').id, kind: 'total', winner: x.input.config.id, over: top.input.config.id, detail: `decisive quality difference ${fmtDiff(d!)} does not override the total for ${profile.label}`, difference: d })
        }
        seen.add(key)
        if (winner === x) {
          if (leaders.includes(x.input.config.id)) { cycle = [...leaders, x.input.config.id]; break } // not a total order
          leaders.push(x.input.config.id)
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

  // Tie-break chain actually walked for the deciding comparison (the last switch, else winner vs runner-up), on the
  // totals that comparison used (G05).
  const tieBreakChain: DecisionTrace['tieBreakChain'] = []
  if (confirmed.length > 1) {
    const lastSwitch = [...steps].reverse().find((s) => s.over)
    const a = confirmed[0]
    const b = lastSwitch ? confirmed.find((x) => x.input.config.id === lastSwitch.over) ?? confirmed[1] : confirmed[1]
    const bb = basisOf.get(pairKey(a, b))
    const [ta, tb, label] = bb?.basis === 'without-quality'
      ? [a.total - qContribution(a), b.total - qContribution(b), 'total without quality']
      : bb?.basis === 'quality' ? [a.total, b.total, 'quality difference (decided) / total'] : [a.total, b.total, 'total']
    const links: [string, number | string | null, number | string | null, number][] = [
      [label, round(ta), round(tb), bb?.basis === 'quality' ? -1 : round(tb) - round(ta)],
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
  // F3: for each provisional candidate, the evaluated counterfactual without its undecided terms (both sides).
  const counterfactuals: DecisionTrace['counterfactuals'] = provisional.map((p) => {
    const terms = p.undecided.map((u) => u.component)
    const without = (x: CandidateVerdict) => x.breakdown.filter((r) => !terms.includes(r.component)).reduce((s, r) => s + r.contribution, 0)
    const w = top && top !== p ? top : null
    return { configId: p.input.config.id, without: [...new Set(terms)], total: Number(without(p).toFixed(3)), vs: w?.input.config.id ?? null, vsTotal: w ? Number(without(w).toFixed(3)) : null,
      result: !w ? 'no confirmed winner to compare with' : ((a, b) => (a > b ? `ahead of ${w.input.config.id}` : a < b ? `behind ${w.input.config.id}` : `tied with ${w.input.config.id}`))(Number(without(p).toFixed(1)), Number(without(w).toFixed(1))) + ' without those terms' }
  })
  // I-7.4 basis: the winner against the fastest confirmed candidate (stored, so reasons never recompute it).
  const fast = ranked.find((v) => v.input.config.id === alternatives.fastest.configId)
  let qvs: DecisionTrace['qualityVsSpeed'] = null
  if (top && fast && fast !== top && fast.decode && top.decode && fast.decode > top.decode) {
    const { d, reason } = top.qualityMeasured && fast.qualityMeasured ? difference(top, fast, profile, cfg) : { d: null, reason: 'quality not measured for both' }
    qvs = { winner: top.input.config.id, fastest: fast.input.config.id, difference: d, ...(reason ? { reason } : {}), decodeWinner: top.decode, decodeFastest: fast.decode }
  }
  const thresholdsUsed: DecisionTrace['thresholdsUsed'] = {
    minDecodeTps: profile.minDecodeTps ?? null, latencyToleranceMs: tol, latencyAdvisory: !!profile.latencyAdvisory, minQuality: profile.minQuality,
    requiredContext: profile.requiredContext ?? null, qualityFirst: !!profile.qualityFirst, targetContext: profile.targetContext
  }
  for (const r of RULES) for (const [k, x] of Object.entries(r.params)) thresholdsUsed[`${r.id}.${k}`] = x
  // G15: the policy values that actually executed (custom configs included), not only catalog defaults.
  for (const [k, x] of Object.entries(cfg.cliff)) thresholdsUsed[`cliff.${k}`] = x as number
  for (const [k, x] of Object.entries(cfg.norm)) thresholdsUsed[`norm.${k}`] = x as number
  thresholdsUsed['gate.minCtxFraction'] = P('gate.context-floor', 'minCtxFraction')
  thresholdsUsed['gate.minStability'] = P('gate.stability', 'min')
  const trace: DecisionTrace = {
    rulesVersion: RULES_VERSION, scoringVersion: cfg.version, workload, scoringRung, scoringRungWhy,
    hardConstraints: {
      requiredContext: profile.requiredContext ?? null,
      minDecodeTps: profile.minDecodeTps === undefined ? null : { value: profile.minDecodeTps, source: userMin ? 'user' : 'workload' },
      latencyToleranceMs: tol, latencyAdvisory: !!profile.latencyAdvisory
    },
    eligibleSet: [
      ...ranked.map((v) => ({ configId: v.input.config.id, failingRuleIds: [...new Set(v.failures.map((f) => f.ruleId))] })),
      ...excluded.map((e) => ({ configId: e.configId, failingRuleIds: [...new Set(e.reasons.map((r) => /^\[(I-[\d.]+)\]/.exec(r)?.[1] ?? rule('stab.failures').id))] }))
    ],
    candidates: ranked.map((v) => ({
      configId: v.input.config.id, confirmed: v.confirmed, undecided: v.undecided.map((u) => `${u.component} (${u.kind})`), total: Number(v.total.toFixed(3)),
      qualityContribution: Number(qContribution(v).toFixed(3)), gen: v.gen?.gq.gen.id ?? null,
      referenceCtx: v.cs.referenceCtx, referenceWhy: v.cs.referenceWhy ?? null, recommendedCtx: v.cs.recommendedCtx, recommendedWhy: v.cs.recommendedWhy ?? null,
      failures: v.failures.map((f) => f.text),
      basis: (Object.keys(profile.weights) as ComponentId[]).filter((k) => profile.weights[k] > 0).map((k) => ({ component: k, rung: basisRung(v, k), kind: v.cs.components[k].input.kind, ...(k === 'quality' ? { scope: qualityScope(v) } : {}) })),
      safety: safety.get(v)!,
      qualityVsWinner: top && v !== top && v.qualityMeasured && top.qualityMeasured ? (({ d, reason }) => ({ difference: d, ...(reason ? { reason } : {}) }))(difference(v, top, profile, cfg)) : null
    })),
    steps, neutralizations, tieBreakChain, alternatives, comparisons, ...(cycle ? { cycle } : {}), excluded, counterfactuals,
    qualityVsSpeed: qvs,
    winner: top?.input.config.id ?? null, provisionalWinner: provisional[0]?.input.config.id ?? null,
    unmetAlternatives: unmet.map((v) => ({ configId: v.input.config.id, unmet: v.failures.filter((f) => f.user).map((f) => f.text) })),
    genChoices, thresholdsUsed
  }
  const coverageAll = prepared.map((p) => {
    const inRanked = ranked.find((v) => v.input === p.input)
    return { input: p.input, dropped: p.dropped, coverage: inRanked ? inRanked.coverage : coverageOf(p.scored, p.base.cs, data.stopReason), excluded: !inRanked }
  }).sort((a, b) => {
    const ra = ranked.findIndex((x) => x.input === a.input), rb = ranked.findIndex((x) => x.input === b.input)
    return (ra < 0 ? Infinity : ra) - (rb < 0 ? Infinity : rb) || (a.input.config.id < b.input.config.id ? -1 : a.input.config.id > b.input.config.id ? 1 : 0)
  })
  return {
    coverageAll,
    workload, profile, cfg, data, request, ranked, provisional, excluded, winner: top, provisionalWinner: provisional[0] ?? null, fallback, unmet, trace, quality, scoringRung, sessionVersion
  }
}

export { genLabel }
