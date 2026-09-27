// Interpretation rules v1 (docs/INTERPRETATION.md), evaluated as data. Pure and deterministic.
// - verdicts(): every decision recommend() makes — gates, generation-config choice, quality-first, required-context
//   fallback, provisional status — each tagged with the rule id that made it. No ranking decision exists outside here.
// - interpret(): the insight panel (rule id, severity, text, evidence with provenance, action).
// Thresholds and texts live in rules.v1.json; the conditions are the evaluators below, keyed by rule key.
import data from './rules.v1.json'
import type {
  BreakdownRow, CandidateInput, ComponentId, ComponentScores, GenQuality, MachineLimits, Metric, QualityCategory, WorkloadId, WorkloadProfile
} from '../../shared/bench-types'
import { componentScores } from '../scoring/components'
import { fmtCtx, isUsable, val } from '../scoring/cliff'
import { DEFAULT_SCORING_CONFIG, type ScoringConfig } from '../scoring/workloads'
import { genLabel } from '../benchmark/gen'

export type Severity = 'info' | 'note' | 'warn' | 'critical'
export interface Rule { id: string; key: string; section: number; severity: Severity; metric: string; params: Record<string, number>; text: string; action?: string }
export interface Evidence { metric: string; value: number | string | null; kind: Metric['kind']; ctx?: number; configId?: string }
export interface Insight { ruleId: string; key: string; severity: Severity; metric: string; text: string; evidence: Evidence[]; action?: string; configId?: string }

export const RULES_VERSION: string = data.version
export const RULES: Rule[] = data.rules as Rule[]
const BY_KEY = new Map(RULES.map((r) => [r.key, r]))

export function rule(key: string): Rule {
  const r = BY_KEY.get(key)
  if (!r) throw new Error(`unknown interpretation rule ${key}`)
  return r
}
const fill = (t: string, vars: Record<string, unknown>) => t.replace(/\{(\w+)\}/g, (m, k: string) => (vars[k] === undefined ? m : String(vars[k])))
/** The rule's text with its id: "[I-2.1] Practical context …". */
export const cite = (key: string, vars: Record<string, unknown> = {}): string => `[${rule(key).id}] ${fill(rule(key).text, vars)}`
const P = (key: string, name: string): number => rule(key).params[name]

const GiB = 1024 ** 3
const gib = (b: number) => `${(b / GiB).toFixed(2)} GiB`
const t1 = (x: number) => x.toFixed(1)
const sec = (ms: number) => `${(ms / 1000).toFixed(1)} s`
const ev = (metric: string, m: Metric | undefined, ctx?: number, configId?: string): Evidence =>
  ({ metric, value: m?.value ?? null, kind: m?.kind ?? 'unavailable', ...(ctx !== undefined ? { ctx } : {}), ...(configId ? { configId } : {}) })
const num = (metric: string, value: number | string | null, kind: Metric['kind'], ctx?: number, configId?: string): Evidence =>
  ({ metric, value, kind, ...(ctx !== undefined ? { ctx } : {}), ...(configId ? { configId } : {}) })

/** "Qwen3.8-27B Q4_K_M (54/65 layers)": model, quant (unless the name has it), and what differs from a full offload. */
export function label(i: CandidateInput): string {
  const { model: m, config: c } = i
  const name = m.quant && !m.name.includes(m.quant) ? `${m.name} ${m.quant}` : m.name
  const extra = [
    c.gpuLayersAll ? null : c.gpuLayers === 0 ? 'CPU only' : `${c.gpuLayers}/${m.layers} layers`,
    c.kvType === 'q8_0' ? 'KV q8_0' : null,
    c.kvOffload === false ? 'KV in RAM' : null
  ].filter((x): x is string => x !== null)
  return extra.length ? `${name} (${extra.join(', ')})` : name
}

// ---------------------------------------------------------------------------------------------------------------
// Verdicts

export interface GateFailure { ruleId: string; text: string }
export interface GenOption { gq: GenQuality; cs: ComponentScores; total: number; withinTolerance: boolean }
export interface CandidateVerdict {
  input: CandidateInput
  cs: ComponentScores
  /** Chosen generation config (I-8.1); null = baseline quality only. */
  gen: GenOption | null
  genOptions: GenOption[]
  breakdown: BreakdownRow[]
  total: number
  eligible: boolean
  failures: GateFailure[]
  /** Decode / peak VRAM / peak RAM at the scoring step (ranking tie-breaks). */
  decode: number | null
  vram: number | null
  ram: number | null
  qualityMeasured: boolean
}
export interface Verdicts {
  workload: WorkloadId
  profile: WorkloadProfile
  cfg: ScoringConfig
  machine: MachineLimits
  /** Eligible first, then total, tie-break chain; the quality-first decision (I-5.2) already applied. */
  ranked: CandidateVerdict[]
  excluded: { configId: string; reasons: string[] }[]
  winner: CandidateVerdict | null
  /** I-2.10: nothing eligible, the winner reaches the required context and fails only speed gates. */
  fallback: boolean
  /** I-5.2 outcome between the winner and the best-scoring other eligible candidate, when both are measured. */
  quality: { decisive: boolean; over: CandidateVerdict; text: string } | null
  /** I-1.1 / I-5.5: why the result is provisional; empty = not provisional. */
  provisional: GateFailure[]
  userMinDecode: boolean
}

const byId = (a: CandidateVerdict, b: CandidateVerdict) => (a.input.config.id < b.input.config.id ? -1 : a.input.config.id > b.input.config.id ? 1 : 0)
const cmp = (a: number | null, b: number | null, desc: boolean) =>
  a === b ? 0 : a === null ? 1 : b === null ? -1 : desc ? b - a : a - b
const q = (v: CandidateVerdict) => v.cs.components.quality
const qText = (v: CandidateVerdict) => `${Math.round(q(v).score)}${q(v).ci95 !== undefined ? ` ± ${Math.round(q(v).ci95!)}` : q(v).input.kind === 'estimated' ? ' (estimated)' : ''}`
/** Bands overlap (I-5.2): |qa − qb| ≤ ca + cb. */
const overlap = (a: CandidateVerdict, b: CandidateVerdict) => Math.abs(q(a).score - q(b).score) <= (q(a).ci95 ?? 0) + (q(b).ci95 ?? 0)

function limitWord(input: CandidateInput, cs: ComponentScores, ceiling: number | null): string {
  if (cs.cliff.steps.some((s) => s.reasons.some((r) => r.code === 'shared_spill' || r.code === 'vram_spill'))) return 'spill'
  if (cs.cliff.limitedBy === 'cliff') return 'cliff'
  if (cs.cliff.limitedBy === 'failure') return 'failure'
  const next = input.config.skippedSteps.find((s) => s.ctx > (ceiling ?? 0))
  if (next && /VRAM|RAM/.test(next.reason)) return 'memory'
  if (next) return 'declared context'
  return 'largest step tested'
}

/** What bounded the practical ceiling, in words (calibration: 8B on 16 GB is memory-bound, not cliff-bound). */
function limitText(input: CandidateInput, cs: ComponentScores, ceiling: number): string {
  if (cs.cliff.limitedBy === 'cliff') return 'limited by a performance cliff above it'
  if (cs.cliff.limitedBy === 'failure') return 'limited by a failed run above it'
  const next = input.config.skippedSteps.find((s) => s.ctx > ceiling)
  if (next && /VRAM|RAM/.test(next.reason)) return `memory-bound at ${fmtCtx(ceiling)} (${fmtCtx(next.ctx)}: ${next.reason})`
  if (next) return `${fmtCtx(next.ctx)} not tested (${next.reason})`
  return 'largest step tested'
}

function scoreOf(input: CandidateInput, machine: MachineLimits, profile: WorkloadProfile, cfg: ScoringConfig, gq?: GenQuality) {
  const cs = componentScores(input, machine, profile, cfg, gq)
  const breakdown: BreakdownRow[] = (Object.keys(profile.weights) as ComponentId[]).map((k) => {
    const c = cs.components[k]
    return { component: k, input: c.input, score: c.score, weight: profile.weights[k], contribution: profile.weights[k] * c.score, ...(c.n !== undefined ? { n: c.n, ci95: c.ci95 } : {}) }
  })
  return { cs, breakdown, total: breakdown.reduce((s, r) => s + r.contribution, 0) }
}

export function verdicts(
  data: { candidates: CandidateInput[]; machine: MachineLimits },
  workload: WorkloadId,
  request: { requiredContext?: number | null; minDecodeTps?: number | null } = {},
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG
): Verdicts {
  const profile = cfg.profiles[workload]
  const { machine } = data
  const round = (x: number) => Number(x.toFixed(cfg.tieDecimals))
  const excluded: Verdicts['excluded'] = []
  const all: CandidateVerdict[] = []
  const userMinDecode = request.minDecodeTps != null

  for (const input of data.candidates) {
    const base = scoreOf(input, machine, profile, cfg)
    if (!base.cs.usable) {
      const why = base.cs.cliff.steps.flatMap((s) => s.reasons.map((r) => r.message))
      excluded.push({ configId: input.config.id, reasons: why.length ? why : ['no runs recorded'] })
      continue
    }
    // I-8.1: per model, the best generation config for this workload — highest quality among the configs whose
    // effective time-to-answer is within the tolerance (the baseline always qualifies), then the faster answer.
    const tol = profile.latencyToleranceMs
    const genOptions: GenOption[] = (input.genQuality ?? []).map((gq) => {
      const s = scoreOf(input, machine, profile, cfg, gq)
      const lat = val(s.cs.components.latency.input, true)
      return { gq, cs: s.cs, total: s.total, withinTolerance: !gq.gen.thinking || !!profile.latencyAdvisory || (lat !== null && lat <= tol) }
    })
    const gen = [...genOptions].filter((g) => g.withinTolerance).sort((a, b) =>
      b.cs.components.quality.score - a.cs.components.quality.score ||
      cmp(val(a.cs.components.latency.input, true), val(b.cs.components.latency.input, true), false))[0] ?? null
    const chosen = gen ? { cs: gen.cs, breakdown: scoreOf(input, machine, profile, cfg, gen.gq).breakdown, total: gen.total } : base
    const ref = input.runs.find((r) => r.ctx === chosen.cs.referenceCtx)
    all.push({
      input, cs: chosen.cs, gen, genOptions, breakdown: chosen.breakdown, total: chosen.total, eligible: true, failures: [],
      decode: val(ref?.decodeTps, true), vram: input.config.gpuLayers === 0 ? 0 : val(ref?.peakVramBytes), ram: val(ref?.peakRamBytes),
      qualityMeasured: chosen.cs.components.quality.input.kind === 'measured'
    })
  }

  // I-1.1: an ESTIMATED quality never out-ranks a MEASURED one — cap priors at the lowest measured quality present.
  const measuredQ = all.filter((v) => v.qualityMeasured).map((v) => q(v).score)
  if (measuredQ.length) {
    const cap = Math.min(...measuredQ)
    for (const v of all) {
      if (v.qualityMeasured || q(v).score <= cap) continue
      const row = v.breakdown.find((r) => r.component === 'quality')!
      const before = row.contribution
      q(v).score = cap
      q(v).note = `${q(v).note ?? ''}${q(v).note ? '; ' : ''}capped at the lowest measured quality ${Math.round(cap)} (an estimate never out-ranks a measurement)`
      row.score = cap
      row.contribution = row.weight * cap
      v.total += row.contribution - before
    }
  }

  const fail = (v: CandidateVerdict, key: string, vars: Record<string, unknown>) => {
    v.eligible = false
    v.failures.push({ ruleId: rule(key).id, text: cite(key, vars) })
  }
  for (const v of all) {
    const { cs, input } = v
    const ceil = val(cs.cliff.practicalContextCeiling)
    const pc = ceil === null ? 'none' : fmtCtx(ceil)
    if (profile.requiredContext) {
      if (ceil === null || ceil < profile.requiredContext) fail(v, 'gate.required-context', { practical: pc, required: fmtCtx(profile.requiredContext), limit: limitWord(input, cs, ceil) })
    } else if (ceil === null || ceil < profile.targetContext * P('gate.context-floor', 'minCtxFraction')) {
      fail(v, 'gate.context-floor', { practical: pc, floor: fmtCtx(profile.targetContext * P('gate.context-floor', 'minCtxFraction')) })
    }
    if (cs.components.stability.score < P('gate.stability', 'min')) fail(v, 'gate.stability', { s: cs.components.stability.score.toFixed(0), min: P('gate.stability', 'min') })
    const qc = cs.components.quality
    // D07: unknown/prior quality never satisfies a quality-weighted workload's gate.
    if (!v.qualityMeasured && profile.qualityFirst) fail(v, 'gate.quality-unmeasured', { workload: profile.label })
    else if (qc.input.kind !== 'unavailable' && qc.score < profile.minQuality) fail(v, 'gate.quality-min', { q: `${qc.input.kind === 'estimated' ? 'estimated ' : ''}${qc.score.toFixed(0)}`, min: profile.minQuality })
    const ref = input.runs.find((r) => r.ctx === cs.referenceCtx)
    if (profile.minDecodeTps !== undefined && v.decode !== null && v.decode < profile.minDecodeTps) {
      fail(v, 'gate.decode', { decode: t1(v.decode), ctx: fmtCtx(ref!.ctx), whose: userMinDecode ? 'your preferred' : `the ${profile.label} minimum`, min: profile.minDecodeTps })
    }
    if (!profile.latencyAdvisory) {
      const tol = `${(profile.latencyToleranceMs / 1000).toFixed(0)} s`
      const t = val(ref?.ttftMs, true)
      // D07: an unknown TTFT fails the latency gate (unless latency is advisory).
      if (t === null) fail(v, 'gate.ttft', { what: 'TTFT', ctx: fmtCtx(ref!.ctx), how: 'is unknown, so it cannot be checked against', tol })
      else if (t > profile.latencyToleranceMs) fail(v, 'gate.ttft', { what: `TTFT ${sec(t)}`, ctx: fmtCtx(ref!.ctx), how: 'exceeds', tol })
      else if (cs.recommendedFits === false) fail(v, 'gate.ttft', { what: 'no passing rung has a measured TTFT', ctx: `≤ ${fmtCtx(profile.maxContext ?? profile.targetContext)}`, how: 'within', tol })
    }
  }
  // I-3.9: a partial config is never recommended for a model whose full offload has any usable step.
  const full = new Map(all.filter((v) => v.input.config.gpuLayersAll).map((v) => [v.input.model.id, v.input.config.id]))
  for (const v of all) {
    if (!v.input.config.gpuLayersAll && machine.gpuDevice !== null && full.has(v.input.model.id)) fail(v, 'gate.partial-offload', { full: full.get(v.input.model.id) })
  }

  all.sort((a, b) =>
    Number(b.eligible) - Number(a.eligible) || round(b.total) - round(a.total) ||
    cmp(a.vram, b.vram, false) || cmp(a.ram, b.ram, false) || byId(a, b))
  excluded.sort((a, b) => (a.configId < b.configId ? -1 : a.configId > b.configId ? 1 : 0))

  // I-5.2 / I-7.3: for quality-weighted workloads a candidate whose measured quality band lies entirely above the
  // leader's wins regardless of speed; within the band, the score (speed, memory…) decides.
  const eligible = all.filter((v) => v.eligible)
  let top = eligible[0] ?? null
  let quality: Verdicts['quality'] = null
  if (top && profile.qualityFirst && top.qualityMeasured) {
    const first = top
    for (;;) {
      const better = eligible.find((v) => v !== top && v.qualityMeasured && q(v).score > q(top!).score && !overlap(v, top!))
      if (!better) break
      top = better
    }
    if (top !== first) {
      quality = { decisive: true, over: first, text: cite('quality.ci-overlap', { qa: qText(top), ca: '', qb: qText(first), cb: '', other: first.input.config.id, verdict: 'outside the confidence band — quality decides' }).replace(/ ± (?= vs| \()/g, '') }
      all.splice(all.indexOf(top), 1)
      all.unshift(top)
    }
  }
  if (top && !quality) {
    const rival = eligible.find((v) => v !== top && v.qualityMeasured && top!.qualityMeasured && q(v).score > q(top!).score)
    if (rival) quality = { decisive: false, over: rival, text: '' }
  }
  if (quality && !quality.text) {
    const r = quality.over
    quality.text = `[${rule('quality.ci-overlap').id}] Quality ${qText(top!)} vs ${qText(r)} (${r.input.config.id}): the difference is inside the confidence band (n=${q(top!).n ?? '?'}) — not decisive; speed and memory decided.`
  }

  // I-2.10: an explicit required context never ends in "no recommendation" just because everything is slow.
  let fallback = false
  if (!top && profile.requiredContext) {
    top = all.filter((v) => (val(v.cs.cliff.practicalContextCeiling) ?? 0) >= profile.requiredContext! && v.failures.every((f) => f.ruleId === rule('gate.decode').id || f.ruleId === rule('gate.ttft').id))
      .sort((a, b) => cmp(a.decode, b.decode, true) || byId(a, b))[0] ?? null
    fallback = !!top
  }

  // I-1.1 / I-5.5: provisional when the winner was decided on an ESTIMATED component, or any candidate lacks measured quality.
  const provisional: GateFailure[] = []
  if (top) {
    const est = (['quality', 'genSpeed', 'prefillSpeed', 'latency'] as ComponentId[]).filter((k) => top!.cs.components[k].input.kind === 'estimated' && profile.weights[k] > 0)
    if (est.length) provisional.push({ ruleId: rule('prov.estimated-in-ranking').id, text: cite('prov.estimated-in-ranking', { components: est.join(', '), verb: est.length > 1 ? 'are' : 'is' }) })
  }
  const unmeasured = [...new Set(all.filter((v) => !v.qualityMeasured).map((v) => v.input.model.name))].sort()
  if (unmeasured.length) provisional.push({ ruleId: rule('quality.estimated').id, text: cite('quality.estimated', { models: unmeasured.join(', ') }) })

  return { workload, profile, cfg, machine, ranked: all, excluded, winner: top, fallback, quality, provisional, userMinDecode }
}

// ---------------------------------------------------------------------------------------------------------------
// Insights

const cats: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']
function band(x: number, edges: [number, string][], last: string): string {
  for (const [e, name] of edges) if (x < e) return name
  return last
}

export function interpret(v: Verdicts): Insight[] {
  const out: Insight[] = []
  const add = (key: string, vars: Record<string, unknown>, evidence: Evidence[], o: { severity?: Severity; configId?: string; action?: string | null } = {}) => {
    const r = rule(key)
    const action = o.action === null ? undefined : o.action ?? r.action
    out.push({ ruleId: r.id, key, severity: o.severity ?? r.severity, metric: r.metric, text: cite(key, vars), evidence, ...(action ? { action } : {}), ...(o.configId ? { configId: o.configId } : {}) })
  }
  const { profile, winner: w, machine } = v
  const everyone = v.ranked

  // §2 context
  if (w) {
    const pc = val(w.cs.cliff.practicalContextCeiling)
    const declared = w.input.model.ctxTrain
    if (pc !== null) {
      add('ctx.ceiling', { practical: fmtCtx(pc), declared: declared ? `model declares ${fmtCtx(declared)}` : 'declared context unknown', limit: limitText(w.input, w.cs, pc) },
        [ev('practicalContextCeiling', w.cs.cliff.practicalContextCeiling, undefined, w.input.config.id)], { configId: w.input.config.id })
    }
    if (w.cs.referenceCtx !== null) add('ctx.reference', { ctx: fmtCtx(w.cs.referenceCtx), why: w.cs.referenceWhy ?? '' }, [num('referenceCtx', w.cs.referenceCtx, 'measured', undefined, w.input.config.id)], { configId: w.input.config.id })
    if (w.cs.recommendedCtx !== null) add('ctx.recommended', { ctx: fmtCtx(w.cs.recommendedCtx), why: w.cs.recommendedWhy ?? '' }, [num('recommendedCtx', w.cs.recommendedCtx, 'measured', undefined, w.input.config.id)], { configId: w.input.config.id })
  }
  for (const c of everyone) {
    const id = c.input.config.id
    const reasons = c.cs.cliff.steps.flatMap((s) => s.reasons)
    const spill = reasons.find((r) => r.code === 'shared_spill')
    if (spill) {
      const drop = reasons.find((r) => r.code === 'decode_drop' && r.toCtx === spill.toCtx)
      const clean = c.cs.cliff.spillFreeUpTo
      add('ctx.spill', { clean: clean ? fmtCtx(clean) : 'the smallest rung', config: label(c.input), spill: gib(spill.to!), ctx: fmtCtx(spill.toCtx), decodeDrop: drop ? `; decode fell ${t1(drop.from!)} → ${t1(drop.to!)} t/s` : '' },
        [num('peakSharedGpuBytes', spill.to, 'measured', spill.toCtx, id)], { configId: id })
    }
    const pc = val(c.cs.cliff.practicalContextCeiling)
    const next = c.input.config.skippedSteps.find((s) => s.ctx > (pc ?? 0))
    if (c.cs.cliff.limitedBy === 'none' && next && /VRAM|RAM/.test(next.reason)) {
      add('ctx.memory-bound', { config: label(c.input), next: fmtCtx(next.ctx), reason: next.reason }, [num('estVramBytes', next.reason, 'estimated', next.ctx, id)], { configId: id })
    }
    const declared = c.input.model.ctxTrain
    if (pc !== null && declared && pc < declared * P('ctx.declared-vs-practical', 'fraction') && c === w) {
      add('ctx.declared-vs-practical', { model: c.input.model.name, declared: fmtCtx(declared), practical: fmtCtx(pc) }, [ev('practicalContextCeiling', c.cs.cliff.practicalContextCeiling, undefined, id), num('ctxTrain', declared, 'declared')], { configId: id })
    }
    // I-2.6: a ≥40 % dip the next rung recovers from
    const u = [...c.input.runs].filter(isUsable).sort((a, b) => a.ctx - b.ctx)
    for (let i = 1; i + 1 < u.length; i++) {
      const a = val(u[i - 1].decodeTps, true)!, b = val(u[i].decodeTps, true)!, cc = val(u[i + 1].decodeTps, true)!
      if (b / a <= P('ctx.transient-dip', 'dropRatio') && a - b >= P('ctx.transient-dip', 'minDropTps') && cc > a * P('ctx.transient-dip', 'dropRatio')) {
        add('ctx.transient-dip', { ctx: fmtCtx(u[i].ctx), config: label(c.input), a: t1(a), b: t1(b), c: t1(cc) }, [ev('decodeTps', u[i].decodeTps, u[i].ctx, id)], { configId: id })
      }
    }
  }
  if (profile.requiredContext && !everyone.some((c) => (val(c.cs.cliff.practicalContextCeiling) ?? 0) >= profile.requiredContext!)) {
    const ceilings = everyone.map((c) => { const pc = val(c.cs.cliff.practicalContextCeiling); return `${label(c.input)} ${pc === null ? 'none' : fmtCtx(pc)} (${limitWord(c.input, c.cs, pc)})` }).join('; ') || 'no usable configuration'
    add('ctx.required-not-met', { required: fmtCtx(profile.requiredContext), ceilings }, everyone.map((c) => ev('practicalContextCeiling', c.cs.cliff.practicalContextCeiling, undefined, c.input.config.id)))
  }

  // §5 quality (leads together with §2 — I-0.1)
  if (w) {
    const qc = q(w)
    if (qc.input.kind === 'measured' && qc.ci95 !== undefined) {
      const rows = w.gen ? w.gen.gq.results : w.input.quality
      const per = cats.filter((c) => profile.promptSetIds.includes(c)).map((c) => {
        const r = rows.filter((x) => x.category === c)
        return r.length ? `${c} ${r.filter((x) => x.pass).length}/${r.length}` : null
      }).filter(Boolean).join(', ')
      const b = rule('quality.band').params
      add('quality.band', { q: Math.round(qc.score), ci: Math.round(qc.ci95), n: qc.n, samples: w.gen && w.gen.gq.samples > 1 ? `, ${w.gen.gq.samples} samples` : '',
        band: band(qc.score, [[b.weak, 'weak'], [b.limited, 'limited'], [b.solid, 'solid'], [b.strong, 'strong']], 'saturated (suite too easy to separate models here)'), categories: per },
      [ev('quality', qc.input, undefined, w.input.config.id)], { configId: w.input.config.id })
      if ((qc.n ?? 0) < P('quality.small-suite', 'minItems')) add('quality.small-suite', { n: qc.n, ci: Math.round(qc.ci95) }, [ev('quality', qc.input)])
      // I-5.3: weak categories
      const weak = cats.filter((c) => profile.promptSetIds.includes(c)).map((c) => {
        const r = rows.filter((x) => x.category === c)
        const rate = r.length ? r.filter((x) => x.pass).length / r.length : null
        return rate === null ? null : { c, rate, r }
      }).filter((x): x is { c: QualityCategory; rate: number; r: typeof rows } => x !== null)
      const low = weak.filter((x) => x.rate < P('quality.category-gap', 'weak'))
      const codingWeak = (profile.id === 'coding' || profile.id === 'large_coding' || profile.id === 'long_context_coding') && weak.some((x) => x.c === 'coding' && x.rate < P('quality.category-gap', 'codingWarn'))
      if (low.length || codingWeak) {
        const list = [...new Set([...low, ...(codingWeak ? weak.filter((x) => x.c === 'coding') : [])])]
        add('quality.category-gap', { model: w.input.model.name, categories: list.map((x) => `${x.c} ${x.r.filter((y) => y.pass).length}/${x.r.length}`).join(', ') },
          list.map((x) => num(`quality.${x.c}`, x.rate, 'measured')), { severity: codingWeak ? 'warn' : 'note', action: codingWeak ? 'try-thinking-config' : undefined })
      }
    }
    if (v.quality) {
      out.push({ ruleId: rule('quality.ci-overlap').id, key: 'quality.ci-overlap', severity: v.quality.decisive ? 'info' : 'warn', metric: 'quality', text: v.quality.text,
        evidence: [ev('quality', q(w).input, undefined, w.input.config.id), ev('quality', q(v.quality.over).input, undefined, v.quality.over.input.config.id)],
        ...(v.quality.decisive ? {} : { action: rule('quality.ci-overlap').action }) })
    }
  }
  for (const p of v.provisional) {
    const key = p.ruleId === rule('quality.estimated').id ? 'quality.estimated' : 'prov.estimated-in-ranking'
    out.push({ ruleId: p.ruleId, key, severity: rule(key).severity, metric: rule(key).metric, text: p.text, evidence: [], action: rule(key).action })
  }
  // One per model: its best-ranked config (the winner, when it is that model's).
  const firstOf = (list: CandidateVerdict[]) => list.filter((c, i) => list.findIndex((x) => x.input.model.id === c.input.model.id) === i)
  const thinkModels = firstOf(everyone.filter((c) => c.input.model.supportsThinking || c.input.model.genKnobs?.supportsThinking))
  for (const c of thinkModels) {
    if (c.gen?.gq.gen.thinking) continue
    const alt = c.genOptions.find((g) => g.gq.gen.thinking)
    add('quality.thinking-off', { model: c.input.model.name, also: alt ? `; ${genLabel(alt.gq.gen)} measured quality ${Math.round(alt.cs.components.quality.score)}` : '' }, [], { configId: c.input.config.id })
  }
  if (everyone.some((c) => c.qualityMeasured)) add('quality.not-a-leaderboard', {}, [])

  // §3 speed
  if (w) {
    const rc = w.input.runs.find((r) => r.ctx === w.cs.recommendedCtx) ?? w.input.runs.find((r) => r.ctx === w.cs.referenceCtx)
    const d = val(rc?.decodeTps, true)
    if (rc && d !== null) {
      const b = rule('speed.decode-band').params
      const gate = profile.minDecodeTps !== undefined ? `${profile.label} ${v.userMinDecode ? 'preferred' : 'gate'} ${profile.minDecodeTps} t/s` : 'no decode gate'
      add('speed.decode-band', { decode: t1(d), ctx: fmtCtx(rc.ctx), band: band(d, [[b.unusable, 'unusable'], [b.patient, 'patient'], [b.usable, 'usable'], [b.comfortable, 'comfortable'], [b.snappy, 'snappy']], 'instant'), gate },
        [ev('decodeTps', rc.decodeTps, rc.ctx, w.input.config.id)], { configId: w.input.config.id })
    }
    const t = val(rc?.ttftMs, true)
    if (rc && t !== null) {
      const b = rule('speed.ttft-band').params
      const over = t > profile.latencyToleranceMs
      const sc = w.input.runs.find((r) => r.ctx === w.cs.referenceCtx)
      const scoredAt = sc && sc.ctx !== rc.ctx ? ` (scored at ${fmtCtx(sc.ctx)}: TTFT ${val(sc.ttftMs, true) === null ? 'unknown' : sec(val(sc.ttftMs, true)!)}, decode ${t1(val(sc.decodeTps, true)!)} t/s)` : ''
      add('speed.ttft-band', {
        ctx: fmtCtx(rc.ctx), ttft: sec(t), tokens: rc.promptTokens ?? '?', band: band(t, [[b.immediate, 'immediate'], [b.short, 'short wait'], [b.noticeable, 'noticeable'], [b.long, 'long']], 'impractical'),
        tol: `${(profile.latencyToleranceMs / 1000).toFixed(0)} s`, accepted: over && profile.requiredContext ? `; accepted because you required ${fmtCtx(profile.requiredContext)}` : over && profile.latencyAdvisory ? '; latency is advisory for this workload' : '',
        decode: d === null ? '?' : t1(d), scoredAt
      }, [ev('ttftMs', rc.ttftMs, rc.ctx, w.input.config.id)], { configId: w.input.config.id, severity: over ? 'warn' : 'info', action: over ? 'use-context' : null })
    }
    const g = w.gen?.gq
    if (g?.gen.thinking) {
      const a = val(g.answerTokens, true), r = val(g.reasoningTokens)
      if (a !== null && r !== null && a / (a + r) < P('speed.thinking-effective', 'ratio')) {
        add('speed.thinking-effective', { reasoning: Math.round(r), gen: genLabel(g.gen), effective: val(w.cs.components.genSpeed.input) === null ? '?' : t1(val(w.cs.components.genSpeed.input)!), decode: w.decode === null ? '?' : t1(w.decode) },
          [g.reasoningTokens, g.answerTokens].map((m, i) => ev(i ? 'answerTokens' : 'reasoningTokens', m)), { configId: w.input.config.id })
      }
    }
  }
  for (const c of everyone) {
    const pd = c.cs.cliff.steps.flatMap((s) => s.reasons).find((r) => r.code === 'prefill_drop')
    if (pd) add('speed.prefill-scaling', { message: pd.message }, [num('prefillTps', pd.to, 'measured', pd.toCtx, c.input.config.id)], { configId: c.input.config.id })
    const cand = c.input.config
    if (!cand.gpuLayersAll && machine.gpuDevice !== null) {
      const fullAlt = everyone.find((x) => x.input.model.id === c.input.model.id && x.input.config.gpuLayersAll)
      const nkvo = everyone.find((x) => x.input.model.id === c.input.model.id && x.input.config.kvOffload === false && x !== c)
      add('speed.partial-offload', {
        layers: cand.gpuLayers === 0 ? 'CPU only' : `${cand.gpuLayers}/${c.input.model.layers} layers${cand.kvOffload === false ? ', KV cache in RAM' : ''}`, config: cand.id,
        decode: c.decode === null ? '?' : t1(c.decode), vs: fullAlt?.decode != null ? ` vs ${t1(fullAlt.decode)} t/s with full offload` : '',
        nkvo: nkvo ? '; KV on CPU (-nkvo) is slower than dropping ~5 layers at short context' : ''
      }, [num('decodeTps', c.decode, 'measured', c.cs.referenceCtx ?? undefined, cand.id)], { configId: cand.id })
    }
  }
  for (const m of [...new Map(everyone.map((c) => [c.input.model.id, c.input.model])).values()]) {
    if (m.expertCount && m.expertCount > 0) add('speed.moe-note', { model: m.name, used: m.expertUsedCount ?? '?', experts: m.expertCount }, [num('expertCount', m.expertCount, 'declared')])
  }

  // §4 memory
  if (w) {
    const rc = w.input.runs.find((r) => r.ctx === (w.cs.recommendedCtx ?? w.cs.referenceCtx))
    const peak = val(rc?.peakVramBytes), total = val(machine.vramBytes, true)
    if (rc && peak !== null && total !== null && w.input.config.gpuLayers > 0) {
      const budget = total - (val(machine.vramInUseBytes) ?? 0)
      const head = budget - peak
      const low = head < P('mem.headroom', 'warnBytes')
      add('mem.headroom', { ctx: fmtCtx(rc.ctx), headroom: gib(head), peak: gib(peak), budget: gib(budget), warn: low ? ' — little headroom; other GPU apps will push this into shared memory' : '' },
        [ev('peakVramBytes', rc.peakVramBytes, rc.ctx, w.input.config.id), ev('vramInUseBytes', machine.vramInUseBytes)], { configId: w.input.config.id, severity: low ? 'warn' : 'info', action: low ? 'use-context' : null })
    }
    if (w.input.config.gpuLayersAll && w.input.config.mmap !== false) add('mem.mmap-note', { file: gib(w.input.model.fileBytes), config: w.input.config.id }, [num('fileBytes', w.input.model.fileBytes, 'declared')], { configId: w.input.config.id })
  }
  const inUse = val(machine.vramInUseBytes)
  if (inUse !== null && inUse > P('mem.in-use-at-plan', 'bytes')) add('mem.in-use-at-plan', { inUse: gib(inUse) }, [ev('vramInUseBytes', machine.vramInUseBytes)])
  const floorish = (c: CandidateVerdict) => c.input.runs
  for (const c of everyone) {
    for (const r of floorish(c)) {
      const id = c.input.config.id
      if (r.failureKind === 'guard_abort') add('mem.ram-floor', { config: id, ctx: fmtCtx(r.ctx), what: `was stopped by the safety guard: ${r.reason ?? 'RAM floor'}` }, [ev('minRamAvailBytes', r.minRamAvailBytes, r.ctx, id)], { configId: id, severity: 'critical' })
      const min = val(r.minRamAvailBytes)
      const floor = Math.max(4 * GiB, 0.08 * (val(machine.ramTotalBytes) ?? 0))
      if (r.failureKind !== 'guard_abort' && min !== null && min < floor + P('mem.ram-floor', 'marginBytes')) {
        add('mem.ram-floor', { config: id, ctx: fmtCtx(r.ctx), what: `came within ${gib(Math.max(0, min - floor))} of the RAM safety floor (${gib(floor)})` }, [ev('minRamAvailBytes', r.minRamAvailBytes, r.ctx, id)], { configId: id })
      }
      if (isUsable(r) && (r.peakVramBytes.kind === 'unavailable' || r.peakSharedGpuBytes.kind === 'unavailable')) {
        add('prov.unavailable-memory', { what: r.peakVramBytes.kind === 'unavailable' ? 'Peak VRAM' : 'Shared-GPU spill', config: id, ctx: fmtCtx(r.ctx) }, [ev('peakVramBytes', r.peakVramBytes, r.ctx, id), ev('peakSharedGpuBytes', r.peakSharedGpuBytes, r.ctx, id)], { configId: id })
      }
    }
    const sp = c.cs.cliff.steps.find((s) => s.reasons.some((x) => x.code === 'shared_spill'))
    const total = val(machine.vramBytes, true)
    const at = sp && c.input.runs.find((r) => r.ctx === sp.ctx)
    const ded = at ? val(at.peakVramBytes) : null
    if (sp && total && ded !== null && ded < 0.95 * total) add('mem.wddm-83', { pct: Math.round((ded / total) * 100), config: c.input.config.id }, [ev('peakVramBytes', at!.peakVramBytes, sp.ctx, c.input.config.id)], { configId: c.input.config.id })
  }

  // §6 stability
  for (const c of everyone) {
    for (const r of c.input.runs) {
      const id = c.input.config.id
      const reps = r.repDecodeTps ?? []
      if (reps.length >= 2) {
        const lo = Math.min(...reps), hi = Math.max(...reps)
        if (hi > 0 && (hi - lo) / hi > P('stab.rep-variance', 'spread')) add('stab.rep-variance', { config: id, ctx: fmtCtx(r.ctx), reps: reps.map(t1).join(' / '), spread: Math.round(((hi - lo) / hi) * 100) }, reps.map((x) => num('decodeTps', x, 'measured', r.ctx, id)), { configId: id })
      }
      if (r.failureKind && ['oom', 'device_lost', 'crash', 'config_drift'].includes(r.failureKind)) {
        add('stab.failures', { config: id, ctx: fmtCtx(r.ctx), kind: r.failureKind, detail: r.failureKind === 'device_lost' ? ' — GPU reset; results after it are suspect' : r.reason ? ` (${r.reason})` : '' },
          [num('failureKind', r.failureKind, 'measured', r.ctx, id)], { configId: id, severity: r.failureKind === 'device_lost' ? 'critical' : 'warn' })
      }
      if (r.warm === false) add('stab.cold-run', { config: id, ctx: fmtCtx(r.ctx) }, [num('warm', 'false', 'measured', r.ctx, id)], { configId: id })
    }
  }
  const runs = everyone.flatMap((c) => c.input.runs)
  const verSet = (f: (x: NonNullable<(typeof runs)[number]['versions']>) => string) => [...new Set(runs.map((r) => (r.versions ? f(r.versions) : null)).filter((x): x is string => x !== null))].sort()
  const rt = verSet((x) => `${x.runtime ?? '?'}/${x.benchmark}/${x.prompts}`), qs = verSet((x) => x.quality)
  if (rt.length > 1) add('stab.versions', { what: 'runtime/benchmark', versions: rt.join(', ') }, [])
  if (qs.length > 1) add('stab.versions', { what: 'quality suite', versions: qs.join(', ') }, [])

  // §7 comparison notes
  for (let i = 0; i < everyone.length; i++) {
    for (let j = i + 1; j < everyone.length; j++) {
      const a = everyone[i], b = everyone[j]
      const ma = a.input.model, mb = b.input.model
      if (ma.id === mb.id || ma.arch !== mb.arch || !ma.paramCount || ma.paramCount !== mb.paramCount || ma.quant === mb.quant) continue
      if (!a.qualityMeasured || !b.qualityMeasured || !a.input.config.gpuLayersAll || !b.input.config.gpuLayersAll || !overlap(a, b)) continue
      const smaller = ma.fileBytes <= mb.fileBytes ? a : b
      add('cmp.same-model-quant', { a: label(a.input), b: label(b.input), qa: Math.round(q(a).score), ca: Math.round(q(a).ci95 ?? 0), qb: Math.round(q(b).score), cb: Math.round(q(b).ci95 ?? 0), smaller: label(smaller.input) },
        [ev('quality', q(a).input, undefined, a.input.config.id), ev('quality', q(b).input, undefined, b.input.config.id)])
    }
  }

  // §8 generation configs
  for (const c of firstOf(everyone.filter((x) => x.genOptions.length > 1))) {
    const off = c.genOptions.find((g) => !g.gq.gen.thinking && g.gq.gen.temperature === 0)
    const g = c.gen
    if (g && off && g !== off) {
      const lo = val(off.gq.effectiveAnswerLatencyMs, true), hi = val(g.gq.effectiveAnswerLatencyMs, true)
      add('gen.best-config', { model: c.input.model.name, gen: genLabel(g.gq.gen), q: Math.round(g.cs.components.quality.score), ci: Math.round(g.cs.components.quality.ci95 ?? 0), qo: Math.round(off.cs.components.quality.score), co: Math.round(off.cs.components.quality.ci95 ?? 0), speed: lo && hi ? `${t1(hi / lo)}× ${hi >= lo ? 'slower' : 'faster'}` : 'speed not measured' },
        [ev('quality', g.cs.components.quality.input), ev('quality', off.cs.components.quality.input), ev('effectiveAnswerLatencyMs', g.gq.effectiveAnswerLatencyMs), ev('effectiveAnswerLatencyMs', off.gq.effectiveAnswerLatencyMs)], { configId: c.input.config.id })
    }
    if (g && g.gq.stochastic) add('gen.stochastic', { gen: genLabel(g.gq.gen), samples: g.gq.samples, ci: Math.round(g.cs.components.quality.ci95 ?? 0) }, [ev('quality', g.cs.components.quality.input)], { configId: c.input.config.id })
    // I-8.3: a higher effort that adds reasoning but no quality beyond the band
    const think = c.genOptions.filter((x) => x.gq.gen.thinking && x.gq.gen.effort)
    const efforts = c.input.model.genKnobs?.effortValues ?? []
    think.sort((x, y) => efforts.indexOf(x.gq.gen.effort!) - efforts.indexOf(y.gq.gen.effort!))
    for (let i = 0; i + 1 < think.length; i++) {
      const lo = think[i], hi = think[i + 1]
      const rlo = val(lo.gq.reasoningTokens), rhi = val(hi.gq.reasoningTokens)
      const qlo = lo.cs.components.quality, qhi = hi.cs.components.quality
      if (rlo !== null && rhi !== null && rhi > rlo && qhi.score - qlo.score <= (qlo.ci95 ?? 0) + (qhi.ci95 ?? 0)) {
        add('gen.effort-saturation', { model: c.input.model.name, hi: hi.gq.gen.effort, lo: lo.gq.gen.effort, rhi: Math.round(rhi), rlo: Math.round(rlo), qhi: Math.round(qhi.score), qlo: Math.round(qlo.score) },
          [ev('reasoningTokens', hi.gq.reasoningTokens), ev('reasoningTokens', lo.gq.reasoningTokens)], { configId: c.input.config.id })
      }
    }
  }

  // I-0.1: context and quality lead, then the rest in guide order; stable within a section.
  const lead = (i: Insight) => { const s = rule(i.key).section; return s === 2 ? 0 : s === 5 ? 1 : 2 + s }
  return out.map((x, i) => ({ x, i })).sort((a, b) => lead(a.x) - lead(b.x) || a.i - b.i).map((a) => a.x)
}
