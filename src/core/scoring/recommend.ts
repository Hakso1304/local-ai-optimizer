// Recommendation for one workload, rendered from the interpretation verdicts and their decision trace (rules interp-2).
// No decision is made here: every reason, alternative, why-not and generation line comes from a verdict or trace entry
// and cites its rule id. Pure and deterministic (A19).
import type { CandidateInput, ComponentId, MachineLimits, Recommendation, WorkloadId } from '../../shared/bench-types'
import { cite, difference, fmtDiff, interpret, label, rule, RULES_VERSION, tag, verdicts, type CandidateVerdict, type InterpretData } from '../interpret'
import { genLabel } from '../benchmark/gen'
import type { UncertaintyRow } from './uncertainty'
import { fmtCtx, val } from './cliff'
import { DEFAULT_SCORING_CONFIG, effectiveProfile, withProfile, type ScoringConfig } from './workloads'

const LABEL: Record<ComponentId, string> = {
  quality: 'quality', genSpeed: 'generation speed', prefillSpeed: 'prefill speed', latency: 'latency',
  memory: 'memory headroom', stability: 'stability', context: 'context capability'
}
const t1 = (x: number) => x.toFixed(1)
const gib2 = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
const Q = (v: CandidateVerdict) => v.cs.components.quality
const qText = (v: CandidateVerdict) => {
  const q = Q(v)
  return q.lower !== undefined ? `${Math.round(q.score)} [${Math.round(q.lower)}, ${Math.round(q.upper!)}]` : q.quarantined ? 'quarantined' : q.input.kind === 'estimated' ? `${Math.round(q.score)} (estimated)` : `${Math.round(q.score)} (${q.input.kind})`
}

/** Largest rung up to which spill was MEASURED absent (unknown shared usage never counts as "no spill"). */
function spillVerifiedUpTo(v: CandidateVerdict): number | null {
  const upTo = v.cs.cliff.spillFreeUpTo
  if (upTo === null) return null
  const rows = v.scored.runs.filter((r) => r.ctx <= upTo).sort((a, b) => a.ctx - b.ctx)
  let ok: number | null = null
  for (const r of rows) { if (r.peakSharedGpuBytes.kind !== 'measured') break; ok = r.ctx }
  return ok
}

function headline(s: CandidateVerdict): string {
  const ctx = s.cs.recommendedCtx ?? s.cs.referenceCtx
  const dec = val(s.scored.runs.find((r) => r.ctx === ctx)?.decodeTps, true)
  const first = s.cs.cliff.steps.find((st) => st.reasons.some((r) => r.code === 'shared_spill' || r.code === 'vram_spill'))
  const upTo = spillVerifiedUpTo(s)
  const spill = upTo !== null ? `no spill up to ${fmtCtx(upTo)}` : first ? `VRAM spill from ${fmtCtx(first.ctx)}` : 'spill not verified'
  const gen = s.gen?.gq.gen.thinking ? `, ${genLabel(s.gen.gq.gen)}` : ''
  return `${label(s.input)}${ctx !== null ? ` @ ${fmtCtx(ctx)}` : ''} — ${dec === null ? 'decode unknown' : `${t1(dec)} t/s`}, quality ${qText(s)}${gen}, ${spill}`
}

export function recommend(
  inputs: CandidateInput[],
  machine: MachineLimits,
  workload: WorkloadId,
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
  /** Models the planner produced no candidates for, with its reason — surfaced in reasons, not just the log. */
  unplanned: { model: string; reason: string }[] = [],
  /** The session request (user floor vs workload gate, I-1.1). */
  request: { requiredContext?: number | null; minDecodeTps?: number | null } = {},
  /** Extra session inputs for interpretation (all persisted runs, planning snapshot, stop reason). */
  extra: Omit<InterpretData, 'candidates' | 'machine'> = {}
): Recommendation {
  const v = verdicts({ candidates: inputs, machine, ...extra }, workload, request, cfg)
  const { profile, trace } = v
  const insights = interpret(v)
  const said = (key: string, configId?: string) => insights.filter((i) => i.key === key && (configId === undefined || i.configId === configId)).map((i) => i.text)
  const top = v.winner
  const reasons: string[] = []
  const decided = (text: string) => reasons.push(cite('cmp.decision-trace', { text }))

  if (!inputs.length) decided('no recommendation — no candidates were benchmarked')
  else if (!v.ranked.length) {
    const all = inputs.flatMap((i) => i.runs)
    if (all.length && all.every((r) => r.failureKind === 'skipped_memory')) {
      decided('no recommendation — nothing was run: the RAM guard skipped every step before loading')
      for (const why of [...new Set(all.map((r) => r.reason).filter((x): x is string => !!x))]) reasons.push(tag('mem.ram-floor', why))
    } else decided('no recommendation — no successful runs')
  } else if (!top) {
    decided(v.provisionalWinner
      ? `no confirmed recommendation for ${profile.label}; best provisional candidate ${v.provisionalWinner.input.config.id} (see I-1.2)`
      : `no recommendation — no candidate meets the ${profile.label} requirements`)
    for (const s of v.ranked.filter((x) => !x.eligible)) reasons.push(tag('cmp.decision-trace', `${s.input.config.id}: ${s.failures.map((f) => f.text).join('; ')}`))
    for (const u of trace.unmetAlternatives) reasons.push(tag('ctx.required', `${u.configId} reaches the required context but misses your constraint: ${u.unmet.join('; ')} — shown as an unmet-constraint alternative, not a recommendation`))
  }
  reasons.push(...said('ctx.required'), ...said('stab.versions'))
  for (const u of unplanned) reasons.push(tag('ctx.planned-skip', `Not benchmarked: ${u.model} — ${u.reason}`))

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
    const last = trace.steps.at(-1)!
    const chain = trace.tieBreakChain.length ? trace.tieBreakChain.map((l) => `${l.step}${l.decided ? ' (decided)' : ''}`).join(' → ') : 'not needed (one candidate)'
    decided(`${v.fallback ? `${cid} meets the required context ${fmtCtx(profile.requiredContext!)} but is below the ${profile.label} speed gate (${top.failures.map((f) => f.text).join('; ')}); constraints ${top.failures.length ? 'partly unmet' : 'met'}`
      : `best for ${profile.label}: ${cid} scores ${top.total.toFixed(1)}/100; constraints met`}; decided by ${last.kind} (${last.detail})${trace.neutralizations.length ? `; ${trace.neutralizations.length} quality neutralization(s)` : ''}; tie-break chain ${chain}`)
    reasons.push(...said('quality.difference'))
    // I-7.4: rendered from the stored trace entry (never recomputed here).
    const qs = trace.qualityVsSpeed
    if (qs) {
      const fast = v.ranked.find((s) => s.input.config.id === qs.fastest)!
      const d = qs.difference
      const qd = d ? `quality ${fmtDiff(d)} vs ${qs.fastest}` : `quality ${qText(top)} vs ${qText(fast)} (${qs.reason ?? 'not paired'})`
      reasons.push(cite('cmp.quality-vs-speed', { text: d && d.lower > 0 ? `chosen for quality: ${qd}; decode ${t1(qs.decodeWinner)} vs ${t1(qs.decodeFastest)} t/s (${(qs.decodeFastest / qs.decodeWinner).toFixed(1)}× slower)` : `${qd} — not a quality win; decode ${t1(qs.decodeWinner)} vs ${t1(qs.decodeFastest)} t/s, other terms decided` }))
    }
    if (genReason) reasons.push(genReason)
    if (profile.requiredContext) {
      const at = top.scored.runs.find((r) => r.ctx === profile.requiredContext && r.status !== 'fail')
      const t = val(at?.ttftMs, true)
      const over = t !== null && t > profile.latencyToleranceMs
      reasons.push(tag('speed.ttft-band', `Required context ${fmtCtx(profile.requiredContext)}: TTFT ${t === null ? 'not measured' : `${(t / 1000).toFixed(1)} s`} at ${fmtCtx(profile.requiredContext)}` +
        (over ? ` (above the profile's ${(profile.latencyToleranceMs / 1000).toFixed(0)} s tolerance; accepted because you required ${fmtCtx(profile.requiredContext)})` : '')))
    }
    if (v.ranked.length === 1) reasons.push(cite('cmp.scope', { workload: profile.label, text: 'only one candidate; not compared' }))
    const leaders = [...top.breakdown].sort((a, b) => b.contribution - a.contribution || (a.component < b.component ? -1 : 1)).slice(0, 2)
    reasons.push(tag('cmp.decision-trace', `Largest contributions: ${leaders.map((r) => `${LABEL[r.component]} ${r.contribution.toFixed(1)}`).join(', ')}`))
    reasons.push(...said('cmp.scope'), ...said('speed.partial-offload', cid), ...said('speed.ttft-band', cid), ...said('ctx.coverage', cid), ...said('ctx.recommended', cid))
    const upTo = spillVerifiedUpTo(top)
    if (upTo !== null) reasons.push(tag('ctx.spill', `No VRAM spill measured up to ${fmtCtx(upTo)}`))
    else if (cliff.spillFreeUpTo !== null) reasons.push(tag('prov.unavailable-reason', `Spill not verified: shared-GPU usage was not measured on the clean rungs`))
    reasons.push(...said('ctx.spill', cid), ...said('speed.prefill-scaling', cid), ...said('quality.estimated', cid), ...said('quality.thinking', cid))

    // I-7.3: the top 2 other candidates, plus the best config of each other model with higher measured quality.
    const others = v.ranked.filter((s) => s !== top)
    const picks = others.slice(0, 2)
    for (const s of others) if (s.qualityMeasured && Q(s).score > Q(top).score && !picks.some((p) => p.input.model.id === s.input.model.id)) picks.push(s)
    for (const s of picks) whyNot.push({ configId: s.input.config.id, model: s.input.model.name, summary: cite('cmp.why-not', { text: whyNotText(s, top) }) })
    for (const g of top.genOptions) {
      if (g === top.gen) continue
      const { d, reason } = difference(g.gq.results as UncertaintyRow[], (top.gen ? top.gen.gq.results : top.qualityRows) as UncertaintyRow[], profile, cfg, 'gen')
      const a = val(g.gq.effectiveAnswerLatencyMs, true), b = top.gen ? val(top.gen.gq.effectiveAnswerLatencyMs, true) : null
      whyNot.push({
        configId: cid, genId: g.gq.gen.id, model: top.input.model.name,
        summary: cite('cmp.why-not', { text: `${genLabel(g.gq.gen)}: quality ${d ? fmtDiff(d) : `not comparable (${g.why ?? reason})`} vs the chosen config` +
          `${a && b ? `; answers ${t1(a >= b ? a / b : b / a)}× ${a >= b ? 'slower' : 'faster'}` : ''}${g.withinTolerance ? '' : `; time to answer above the ${(profile.latencyToleranceMs / 1000).toFixed(0)} s tolerance`}` })
      })
    }
  }
  const provisional = v.provisionalWinner

  function whyNotText(s: CandidateVerdict, w: CandidateVerdict): string {
    const parts: string[] = []
    const stored = trace.candidates.find((c) => c.configId === s.input.config.id)?.qualityVsWinner
    const { d, reason } = stored ? { d: stored.difference, reason: stored.reason } : { d: null, reason: 'quality not measured for both' }
    parts.push(d ? `quality ${fmtDiff(d)} (${qText(s)} vs ${qText(w)})` : `quality ${qText(s)} vs ${qText(w)} (${reason})`)
    if (s.decode !== null && w.decode !== null) {
      const a = s.cs.referenceCtx, b = w.cs.referenceCtx
      parts.push(a === b ? `decode ${t1(s.decode)} vs ${t1(w.decode)} t/s${a !== null ? ` at ${fmtCtx(a)}` : ''}` : `decode ${t1(s.decode)} t/s at ${a !== null ? fmtCtx(a) : '?'} vs ${t1(w.decode)} at ${b !== null ? fmtCtx(b) : '?'}`)
    }
    const rs = s.cs.cliff.steps.flatMap((st) => st.reasons)
    const drop = rs.find((r) => r.code === 'decode_drop'), spill = rs.find((r) => r.code === 'shared_spill')
    const why = drop ? ` (decode fell ${t1(drop.from!)} → ${t1(drop.to!)} t/s after ${fmtCtx(drop.fromCtx!)}${spill ? `, shared-VRAM spill ${gib2(spill.to!)} at ${fmtCtx(spill.toCtx)}` : ''})` : spill ? ` (shared-VRAM spill ${gib2(spill.to!)} at ${fmtCtx(spill.toCtx)})` : ''
    const pc = s.coverage.largestCleanTested, pt = w.coverage.largestCleanTested
    parts.push(`largest clean context ${pc === null ? 'none' : fmtCtx(pc)} vs ${pt === null ? 'none' : fmtCtx(pt)}${why}`)
    const n = trace.neutralizations.find((x) => (x.a === s.input.config.id && x.b === w.input.config.id) || (x.b === s.input.config.id && x.a === w.input.config.id))
    const q = trace.steps.find((x) => x.kind === 'quality-decides' && x.over === s.input.config.id)
    parts.push(!s.eligible ? `ineligible — ${s.failures.map((f) => f.text).join('; ')}`
      : !s.confirmed ? `provisional — ${s.undecided.map((u) => `${u.component} ${u.kind}`).join(', ')}`
        : q ? `${profile.label} total ${Math.round(s.total)} vs ${Math.round(w.total)} — outranked because quality decides (${q.detail})`
          : n ? `${profile.label} totals without quality ${n.totalsWithoutQuality[s.input.config.id]?.toFixed(1)} vs ${n.totalsWithoutQuality[w.input.config.id]?.toFixed(1)} (quality neutralized)`
            : `${profile.label} total ${Math.round(s.total)} vs ${Math.round(w.total)}`)
    return `${label(s.input)}: ${parts.join('; ')}`
  }

  const alternatives = {
    fastest: trace.alternatives.fastest.configId, bestQuality: trace.alternatives.bestQuality.configId,
    bestLongContext: trace.alternatives.bestLongContext.configId, lowestMemory: trace.alternatives.lowestMemory.configId
  }
  return {
    workload, scoringVersion: cfg.version, rulesVersion: RULES_VERSION, best, alternatives,
    ranked: v.ranked.map((s) => ({ configId: s.input.config.id, workload, total: s.total, eligible: s.eligible, gateFailures: s.failures.map((f) => f.text), breakdown: s.breakdown, referenceCtx: s.cs.referenceCtx, recommendedCtx: s.cs.recommendedCtx, ...(s.gen ? { gen: s.gen.gq.gen } : {}) })),
    excluded: v.excluded, reasons, whyNot, insights,
    provisional: !top && !!provisional,
    ...(provisional ? { provisionalBest: { configId: provisional.input.config.id, headline: headline(provisional), estimatedTerms: provisional.undecided.map((u) => `${u.component} (${u.kind})`), reason: cite('prov.confirmed-vs-provisional', { config: provisional.input.config.id, terms: provisional.undecided.map((u) => `${u.component} (${u.kind})`).join(', ') }) } } : {}),
    ...(trace.unmetAlternatives.length ? { unmetAlternatives: trace.unmetAlternatives } : {}),
    decisionTrace: trace
  }
}

/** Recommendation for any workload from stored session data — no re-run. Same scoring as the runner (effective
 *  profile from requiredContext / minDecodeTps), so a card for another workload can be computed at read time. */
export function recommendForWorkload(
  data: { candidates: CandidateInput[]; machine: MachineLimits } & Omit<InterpretData, 'candidates' | 'machine'>,
  workload: WorkloadId,
  request: { requiredContext?: number | null; minDecodeTps?: number | null } = {},
  unplanned: { model: string; reason: string }[] = [],
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG
): Recommendation {
  const { candidates, machine, ...extra } = data
  return recommend(candidates, machine, workload, withProfile(cfg, effectiveProfile(cfg.profiles[workload], request)), unplanned, request, extra)
}

export { rule }
