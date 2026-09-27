// Rank candidates for one workload and explain the pick in plain English. Pure and deterministic (A19).
// Every decision comes from the interpretation rules (core/interpret verdicts); every reason cites its rule id.
import type { CandidateInput, ComponentId, MachineLimits, Recommendation, WorkloadId } from '../../shared/bench-types'
import { cite, interpret, label, rule, RULES_VERSION, verdicts, type CandidateVerdict, type Insight } from '../interpret'
import { genLabel } from '../benchmark/gen'
import { fmtCtx, val } from './cliff'
import { DEFAULT_SCORING_CONFIG, effectiveProfile, withProfile, type ScoringConfig } from './workloads'

const LABEL: Record<ComponentId, string> = {
  quality: 'quality', genSpeed: 'generation speed', prefillSpeed: 'prefill speed', latency: 'latency',
  memory: 'memory headroom', stability: 'stability', context: 'context capability'
}

const t1 = (x: number) => x.toFixed(1)
const gib2 = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
const byId = (a: CandidateVerdict, b: CandidateVerdict) => (a.input.config.id < b.input.config.id ? -1 : a.input.config.id > b.input.config.id ? 1 : 0)
const cmp = (a: number | null, b: number | null, desc: boolean) =>
  a === b ? 0 : a === null ? 1 : b === null ? -1 : desc ? b - a : a - b
const Q = (v: CandidateVerdict) => v.cs.components.quality
const qText = (v: CandidateVerdict) => `${Math.round(Q(v).score)}${Q(v).ci95 !== undefined ? ` ± ${Math.round(Q(v).ci95!)}` : Q(v).input.kind === 'estimated' ? ' (estimated)' : ' (unknown)'}`
const id = (key: string) => `[${rule(key).id}]`

function headline(s: CandidateVerdict): string {
  const ctx = s.cs.recommendedCtx ?? s.cs.referenceCtx
  const run = s.input.runs.find((r) => r.ctx === ctx)
  const dec = val(run?.decodeTps, true)
  const cliff = s.cs.cliff
  const firstSpill = cliff.steps.find((st) => st.reasons.some((r) => r.code === 'shared_spill' || r.code === 'vram_spill'))
  const spill = cliff.spillFreeUpTo !== null ? `no spill up to ${fmtCtx(cliff.spillFreeUpTo)}` : firstSpill ? `VRAM spill from ${fmtCtx(firstSpill.ctx)}` : 'spill not measured'
  const gen = s.gen?.gq.gen.thinking ? `, ${genLabel(s.gen.gq.gen)}` : ''
  return `${label(s.input)}${ctx !== null ? ` @ ${fmtCtx(ctx)}` : ''} — ${dec === null ? 'decode unknown' : `${t1(dec)} t/s`}, quality ${qText(s)}${gen}, ${spill}`
}

/** I-7.2: quality delta, speed at the scoring contexts, ceiling (with its cliff), then the failed gate or the totals. */
function whyNotText(s: CandidateVerdict, top: CandidateVerdict, profileLabel: string): string {
  const parts: string[] = []
  const dq = Math.round(Q(s).score) - Math.round(Q(top).score)
  parts.push(dq === 0 ? `same quality (${qText(s)} vs ${qText(top)})` : `quality ${dq > 0 ? '+' : '−'}${Math.abs(dq)} pts (${qText(s)} vs ${qText(top)})`)
  if (s.decode !== null && top.decode !== null) {
    const a = s.cs.referenceCtx, b = top.cs.referenceCtx
    parts.push(a === b ? `decode ${t1(s.decode)} vs ${t1(top.decode)} t/s${a !== null ? ` at ${fmtCtx(a)}` : ''}`
      : `decode ${t1(s.decode)} t/s at ${a !== null ? fmtCtx(a) : '?'} vs ${t1(top.decode)} at ${b !== null ? fmtCtx(b) : '?'}`)
  }
  const reasons = s.cs.cliff.steps.flatMap((st) => st.reasons)
  const drop = reasons.find((r) => r.code === 'decode_drop')
  const spill = reasons.find((r) => r.code === 'shared_spill')
  const why = drop ? ` (decode fell ${t1(drop.from!)} → ${t1(drop.to!)} t/s after ${fmtCtx(drop.fromCtx!)}${spill ? `, shared-VRAM spill ${gib2(spill.to!)} at ${fmtCtx(spill.toCtx)}` : ''})`
    : spill ? ` (shared-VRAM spill ${gib2(spill.to!)} at ${fmtCtx(spill.toCtx)})` : ''
  const pc = val(s.cs.cliff.practicalContextCeiling), pt = val(top.cs.cliff.practicalContextCeiling)
  parts.push(`practical context ${pc === null ? 'none' : fmtCtx(pc)} vs ${pt === null ? 'none' : fmtCtx(pt)}${why}`)
  parts.push(!s.eligible ? `ineligible — ${s.failures.map((f) => f.text).join('; ')}`
    : `${profileLabel} total ${Math.round(s.total)} vs ${Math.round(top.total)}${s.total > top.total ? ` — outranked because quality decides (bands apart, ${id('quality.ci-overlap')})` : ''}`)
  return `${label(s.input)}: ${parts.join('; ')}`
}

const CLIFF_RULE: Record<string, string> = {
  decode_drop: 'ctx.ceiling', prefill_drop: 'speed.prefill-scaling', shared_spill: 'ctx.spill', vram_spill: 'mem.wddm-83', run_failed: 'stab.failures', invalid_metrics: 'stab.failures'
}

export function recommend(
  inputs: CandidateInput[],
  machine: MachineLimits,
  workload: WorkloadId,
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
  /** Models the planner produced no candidates for, with its reason — surfaced in reasons, not just the log. */
  unplanned: { model: string; reason: string }[] = [],
  /** The session request (only to word the user's own speed preference, I-3.7). */
  request: { requiredContext?: number | null; minDecodeTps?: number | null } = {}
): Recommendation {
  const v = verdicts({ candidates: inputs, machine }, workload, request, cfg)
  const { profile } = v
  const insights: Insight[] = interpret(v)
  const said = (key: string, configId?: string) => insights.filter((i) => i.key === key && (configId === undefined || i.configId === configId)).map((i) => i.text)
  const eligible = v.ranked.filter((s) => s.eligible)
  const pick = (f: (a: CandidateVerdict, b: CandidateVerdict) => number) => [...eligible].sort((a, b) => f(a, b) || byId(a, b))[0]?.input.config.id ?? null
  const alternatives = {
    fastest: pick((a, b) => cmp(a.decode, b.decode, true)),
    bestQuality: pick((a, b) => Q(b).score - Q(a).score),
    bestLongContext: pick((a, b) => cmp(val(a.cs.cliff.practicalContextCeiling), val(b.cs.cliff.practicalContextCeiling), true) || cmp(a.decode, b.decode, true)),
    lowestMemory: pick((a, b) => cmp(a.vram, b.vram, false) || cmp(a.ram, b.ram, false))
  }

  const top = v.winner
  const reasons: string[] = []
  const win = (text: string) => reasons.push(cite('cmp.winner', { text }))
  if (!inputs.length) win('No recommendation: no candidates were benchmarked')
  else if (!v.ranked.length) {
    const all = inputs.flatMap((i) => i.runs)
    if (all.length && all.every((r) => r.failureKind === 'skipped_memory')) {
      win('No recommendation: nothing was run — the RAM guard skipped every step before loading')
      for (const why of [...new Set(all.map((r) => r.reason).filter((x): x is string => !!x))]) reasons.push(`${id('mem.ram-floor')} ${why}`)
    } else win('No recommendation: no successful runs')
  } else if (!top || v.fallback) {
    win(`No recommendation: no candidate meets the ${profile.label} requirements`)
    for (const s of v.ranked) reasons.push(`${id('cmp.winner')} ${s.input.config.id}: ${s.failures.map((f) => f.text).join('; ')}`)
  }
  reasons.push(...said('ctx.required-not-met'), ...said('stab.versions'))
  for (const u of unplanned) reasons.push(`${id('ctx.memory-bound')} Not benchmarked: ${u.model} — ${u.reason}`)

  let best: Recommendation['best'] = null
  const whyNot: NonNullable<Recommendation['whyNot']> = []
  if (top) {
    const cid = top.input.config.id
    const { cliff } = top.cs
    const declared = top.input.model.ctxTrain
    const genReason = said('gen.best-config', cid)[0]
    best = {
      configId: cid, headline: headline(top),
      score: { configId: cid, workload, total: top.total, eligible: top.eligible, gateFailures: top.failures.map((f) => f.text), breakdown: top.breakdown, referenceCtx: top.cs.referenceCtx, recommendedCtx: top.cs.recommendedCtx, ...(top.gen ? { gen: top.gen.gq.gen } : {}) },
      practicalContext: cliff.practicalContextCeiling,
      declaredContext: declared ? { value: declared, kind: 'declared', source: 'GGUF context_length' } : { value: null, kind: 'unavailable', reason: 'GGUF has no context_length' },
      cliff,
      ...(v.fallback ? { fallback: 'meets required context; below preferred speed' as const } : {}),
      ...(top.gen?.gq.gen.thinking && genReason ? { gen: { config: top.gen.gq.gen, reason: genReason } } : {})
    }
    if (v.fallback) {
      reasons.push(cite('fallback.required-context', { config: cid, required: fmtCtx(profile.requiredContext!), failures: top.failures.map((f) => f.text).join('; ') }))
    } else win(`Best for ${profile.label}: ${cid} scores ${top.total.toFixed(1)}/100`)
    reasons.push(...v.provisional.map((p) => p.text))
    if (v.quality) reasons.push(v.quality.text)
    // I-7.3: how much speed the quality pick costs.
    const fast = v.ranked.find((s) => s.input.config.id === alternatives.fastest)
    if (fast && fast !== top && fast.decode && top.decode && fast.decode / top.decode >= rule('cmp.quality-vs-speed').params.slowerRatio) {
      reasons.push(cite('cmp.quality-vs-speed', { text: `Chosen for quality over speed: quality ${qText(top)} vs ${qText(fast)}; decode ${t1(top.decode)} vs ${t1(fast.decode)} t/s (${(fast.decode / top.decode).toFixed(1)}× slower than ${fast.input.config.id})` }))
    }
    if (genReason) reasons.push(genReason)
    if (profile.requiredContext) {
      const at = top.input.runs.find((r) => r.ctx === profile.requiredContext && r.status !== 'fail')
      const t = val(at?.ttftMs, true)
      const over = t !== null && t > profile.latencyToleranceMs
      reasons.push(`${id('speed.ttft-band')} Required context ${fmtCtx(profile.requiredContext)}: TTFT ${t === null ? 'not measured' : `${(t / 1000).toFixed(1)} s`} at ${fmtCtx(profile.requiredContext)}` +
        (over ? ` (above the profile's ${(profile.latencyToleranceMs / 1000).toFixed(0)} s tolerance; accepted because you required ${fmtCtx(profile.requiredContext)})` : ''))
    }
    if (v.ranked.length === 1) win('Only one candidate; not compared')
    const leaders = [...top.breakdown].sort((a, b) => b.contribution - a.contribution || (a.component < b.component ? -1 : 1)).slice(0, 2)
    win(`Largest contributions: ${leaders.map((r) => `${LABEL[r.component]} ${r.contribution.toFixed(1)}`).join(', ')}`)
    reasons.push(...said('speed.partial-offload', cid), ...said('speed.ttft-band', cid), ...said('ctx.ceiling', cid), ...said('ctx.recommended', cid))
    if (cliff.spillFreeUpTo !== null) reasons.push(`${id('ctx.spill')} No VRAM spill up to ${fmtCtx(cliff.spillFreeUpTo)}`)
    for (const s of cliff.steps) for (const r of s.reasons) if (r.code !== 'beyond_limit') reasons.push(`${id(CLIFF_RULE[r.code] ?? 'ctx.ceiling')} ${r.message}`)
    const qn = top.cs.components.quality.note
    if (qn) reasons.push(`${id('quality.estimated')} ${qn}`)
    reasons.push(...said('quality.thinking-off', cid))

    // Why not: the top 2 other candidates by rank, plus the best config of each other model with higher measured quality.
    const others = v.ranked.filter((s) => s !== top)
    const picks = others.slice(0, 2)
    for (const s of others) {
      if (s.qualityMeasured && Q(s).score > Q(top).score && !picks.some((p) => p.input.model.id === s.input.model.id)) picks.push(s)
    }
    for (const s of picks) whyNot.push({ configId: s.input.config.id, model: s.input.model.name, summary: whyNotText(s, top, profile.label) })
    // …and the winner's other generation configs.
    for (const g of top.genOptions) {
      if (g === top.gen) continue
      const a = val(g.gq.effectiveAnswerLatencyMs, true), b = top.gen ? val(top.gen.gq.effectiveAnswerLatencyMs, true) : null
      const dq = Math.round(g.cs.components.quality.score) - Math.round(Q(top).score)
      whyNot.push({
        configId: cid, genId: g.gq.gen.id, model: top.input.model.name,
        summary: `${genLabel(g.gq.gen)}: quality ${dq > 0 ? '+' : dq < 0 ? '−' : '±'}${Math.abs(dq)} pts (${Math.round(g.cs.components.quality.score)} ± ${Math.round(g.cs.components.quality.ci95 ?? 0)} vs ${qText(top)})` +
          `${a && b ? `; answers ${t1(a >= b ? a / b : b / a)}× ${a >= b ? 'slower' : 'faster'}` : ''}${g.withinTolerance ? '' : `; time to answer above the ${(profile.latencyToleranceMs / 1000).toFixed(0)} s tolerance`}`
      })
    }
  }

  return {
    workload, scoringVersion: cfg.version, rulesVersion: RULES_VERSION, best, alternatives,
    ranked: v.ranked.map((s) => ({ configId: s.input.config.id, workload, total: s.total, eligible: s.eligible, gateFailures: s.failures.map((f) => f.text), breakdown: s.breakdown, referenceCtx: s.cs.referenceCtx, recommendedCtx: s.cs.recommendedCtx, ...(s.gen ? { gen: s.gen.gq.gen } : {}) })),
    excluded: v.excluded, reasons, whyNot, insights, provisional: v.provisional.length > 0
  }
}

/** Recommendation for any workload from stored session data — no re-run. Same scoring as the runner (effective
 *  profile from requiredContext / minDecodeTps), so a card for another workload can be computed at read time. */
export function recommendForWorkload(
  data: { candidates: CandidateInput[]; machine: MachineLimits },
  workload: WorkloadId,
  request: { requiredContext?: number | null; minDecodeTps?: number | null } = {},
  unplanned: { model: string; reason: string }[] = [],
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG
): Recommendation {
  return recommend(data.candidates, data.machine, workload, withProfile(cfg, effectiveProfile(cfg.profiles[workload], request)), unplanned, request)
}
