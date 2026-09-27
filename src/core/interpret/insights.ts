// The insight panel (rules interp-2): rule id, severity, text, scoped evidence, typed action. Pure.
// Rules whose inputs are absent are omitted or say "not evaluable" — never guessed (§12).
import type { CandidateInput, Metric, QualityCategory } from '../../shared/bench-types'
import { fmtCtx, isUsable, val } from '../scoring/cliff'
import { categoryFlags, type UncertaintyRow } from '../scoring/uncertainty'
import { genLabel } from '../benchmark/gen'
import { KV_Q8_OVER_F16, quantSuggestions } from '../benchmark/candidates'
import { DEFAULT_SCORING_CONFIG } from '../scoring/workloads'
import { action, cite, P, rule, RULES_VERSION, tag, type Severity } from './catalog'
import { difference, fmtDiff, type CandidateVerdict, type StoredRun, type Verdicts } from './verdicts'

export interface Evidence {
  metric: string; value: number | string | null; unit?: string; kind: Metric['kind']; source?: string; reason?: string
  ctx?: number; configId?: string; samples?: number; algorithm?: string; rulesVersion: string
}
export interface Insight { ruleId: string; key: string; severity: Severity; metric: string; text: string; evidence: Evidence[]; action?: string; configId?: string; evaluable?: boolean }

const GiB = 1024 ** 3
const gib = (b: number) => `${(b / GiB).toFixed(2)} GiB`
const sgib = (b: number) => `${b < 0 ? '−' : '+'}${(Math.abs(b) / GiB).toFixed(2)} GiB`
const t1 = (x: number) => x.toFixed(1)
const sec = (ms: number) => `${(ms / 1000).toFixed(1)} s`
const unitOf = (metric: string) => (/Bytes$/.test(metric) ? 'bytes' : /Tps$/.test(metric) ? 't/s' : /Ms$/.test(metric) ? 'ms' : /ctx|Ctx|Context/.test(metric) ? 'tokens' : undefined)
const ev = (metric: string, m: Metric | undefined, ctx?: number, configId?: string, extra: Partial<Evidence> = {}): Evidence => ({
  metric, value: m?.value ?? null, kind: m?.kind ?? 'unavailable', ...(unitOf(metric) ? { unit: unitOf(metric) } : {}),
  ...(m?.source ? { source: m.source } : {}), ...(m?.reason ? { reason: m.reason } : {}),
  ...(ctx !== undefined ? { ctx } : {}), ...(configId ? { configId } : {}), rulesVersion: RULES_VERSION, ...extra
})
const num = (metric: string, value: number | string | null, kind: Metric['kind'], ctx?: number, configId?: string, extra: Partial<Evidence> = {}): Evidence => ev(metric, { value, kind } as Metric, ctx, configId, extra)
function band(x: number, edges: [number, string][], last: string): string {
  for (const [e, name] of edges) if (x < e) return name
  return last
}
const firstOf = (list: CandidateVerdict[]) => list.filter((c, i) => list.findIndex((x) => x.input.model.id === c.input.model.id) === i)
const CATS: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']

/** F9: which remedy can actually fit a VRAM-skipped rung, from the planner's weights / KV / overhead breakdown. */
function vramRemedy(cfg: CandidateInput['config'], s: { estimateBytes?: number; budgetBytes?: number; weightsBytes?: number; kvBytes?: number; overheadBytes?: number }): string {
  if (s.budgetBytes === undefined || s.weightsBytes === undefined || s.kvBytes === undefined || s.overheadBytes === undefined) return action('inspect-diagnostics')
  if (s.weightsBytes + s.overheadBytes > s.budgetBytes) return cfg.gpuLayersAll ? action('enable-heavy-mode') : action('inspect-diagnostics') // weights alone do not fit
  if (cfg.kvType === 'f16' && s.weightsBytes + s.kvBytes * KV_Q8_OVER_F16 + s.overheadBytes <= s.budgetBytes) return action('enable-kv-q8') // the estimator's exact q8_0/f16 ratio; the budget already excludes the reserve
  return cfg.gpuLayersAll ? action('enable-heavy-mode') : action('lower-required-context')
}

/** F4: the adjusted spill as observed, or why it is unavailable. */
function adjustedText(m: Metric | undefined): string {
  if (!m || m.kind !== 'measured' || m.value === null) return `the adjusted spill is unavailable (${m?.reason ?? 'not recorded'})`
  return `the adjusted spill was ${gib(m.value)} (${m.source ?? 'adjusted-spill'})`
}

export function interpret(v: Verdicts): Insight[] {
  const out: Insight[] = []
  const push = (key: string, text: string, evidence: Evidence[], o: { severity?: Severity; configId?: string; action?: string | null; evaluable?: boolean } = {}) => {
    const r = rule(key)
    const act = o.action === null ? undefined : o.action ?? (r.action ? action(r.action as never) : undefined)
    out.push({ ruleId: r.id, key, severity: o.severity ?? r.severity, metric: r.metric, text, evidence, ...(act ? { action: act } : {}), ...(o.configId ? { configId: o.configId } : {}), ...(o.evaluable === false ? { evaluable: false } : {}) })
  }
  const add = (key: string, vars: Record<string, unknown>, evidence: Evidence[], o: Parameters<typeof push>[3] = {}) => push(key, cite(key, vars), evidence, o)
  const { profile, winner: w, data, trace } = v
  const machine = data.machine
  const everyone = v.ranked
  const allInputs: CandidateInput[] = data.candidates
  const id = (c: { input: CandidateInput }) => c.input.config.id

  // §1 provenance
  for (const c of everyone) for (const f of c.failures) {
    if (f.notVerified) add('prov.hard-constraints-first', { config: id(c), constraint: f.notVerified, why: f.text.replace(/^\[I-[\d.]+\] /, '') }, [], { configId: id(c), action: action('retry-telemetry') })
  }
  for (const c of v.provisional) {
    add('prov.confirmed-vs-provisional', { config: id(c), terms: c.undecided.map((u) => `${u.component} (${u.kind})`).join(', ') },
      c.undecided.map((u) => ev(u.component, c.cs.components[u.component].input, undefined, id(c))), { configId: id(c), action: c.undecided.some((u) => u.component === 'quality') ? action('run-thorough-quality') : action('retry-telemetry') })
  }
  if (!w && v.provisionalWinner) {
    const p = v.provisionalWinner
    push('prov.decisive-trace', tag('prov.decisive-trace', `No confirmed winner: the best provisional candidate ${id(p)} depends on ${p.undecided.map((u) => u.component).join(', ')}; without ${p.undecided.length > 1 ? 'those terms' : 'that term'} it cannot be confirmed against measured candidates`), [])
  }
  for (const c of everyone) {
    for (const r of c.scored.runs) {
      if (!isUsable(r)) continue
      for (const [m, label] of [['peakVramBytes', 'Peak VRAM'], ['peakSharedGpuBytes', 'Shared-GPU usage']] as const) {
        const x = r[m]
        if (x.kind === 'unavailable') add('prov.unavailable-reason', { what: label, config: id(c), ctx: fmtCtx(r.ctx), reason: x.reason ?? 'no reason recorded' }, [ev(m, x, r.ctx, id(c))], { configId: id(c), action: action('retry-telemetry') })
      }
    }
  }

  // §2 coverage
  for (const c of v.coverageAll) {
    const cov = c.coverage
    const cleanM = { value: cov.largestCleanTested, kind: cov.largestCleanTested === null ? 'unavailable' : 'measured', ...(cov.largestCleanTested === null ? { reason: 'no clean rung' } : {}) } as Metric
    const declared = c.input.model.ctxTrain
    const stopped = cov.limitKind === 'failure' || cov.limitKind === 'spill' || cov.limitKind === 'cliff'
      ? `Stopped because of a ${cov.limitKind} at ${fmtCtx(cov.firstObservedFailure!.ctx)}: ${cov.firstObservedFailure!.reason}.`
      : cov.limitKind.startsWith('planned-skip') ? `Higher contexts were not attempted: ${fmtCtx(cov.firstPlannedSkip!.ctx)} ${cov.firstPlannedSkip!.reason}.`
        : cov.limitKind === 'user-cap' ? 'Higher contexts were not attempted (the ladder was capped by the request).'
          : cov.limitKind === 'cancelled' ? 'Higher contexts were not attempted (the session stopped early).'
            : 'Higher contexts were not attempted.'
    if (cov.largestCleanTested === null) {
      push('ctx.coverage', tag('ctx.coverage', `${id(c)}: no clean context measured in this run${cov.firstObservedFailure ? ` — first observed at ${fmtCtx(cov.firstObservedFailure.ctx)}: ${cov.firstObservedFailure.reason}` : ''}${c.dropped.length ? `; not verified: ${c.dropped.map((d) => `${fmtCtx(d.ctx)} (${d.reason})`).join(', ')}` : ''}${c.excluded ? ' — excluded from ranking' : ''}.`), [], { configId: id(c) })
    } else {
      add('ctx.coverage', { config: id(c), clean: fmtCtx(cov.largestCleanTested), declared: declared ? fmtCtx(declared) : 'unknown', stopped },
        [ev('largestCleanTested', cleanM, undefined, id(c))], { configId: id(c) })
      if (declared && cov.largestCleanTested < declared * P('ctx.declared-vs-tested', 'fraction')) {
        const observed = stopped.startsWith('Stopped') ? ` (a ${cov.limitKind} was observed at ${fmtCtx(cov.firstObservedFailure!.ctx)})` : ''
        add('ctx.declared-vs-tested', { config: id(c), clean: fmtCtx(cov.largestCleanTested), declared: fmtCtx(declared), observed },
          [ev('largestCleanTested', cleanM, undefined, id(c)), num('ctxTrain', declared, 'declared')],
          { configId: id(c), severity: observed ? 'warn' : 'note', action: observed ? action('use-context', fmtCtx(cov.largestCleanTested)) : null })
      }
    }
    if (cov.firstPlannedSkip && cov.firstPlannedSkip.resource && cov.firstPlannedSkip.resource !== 'declared') {
      const s = cov.firstPlannedSkip
      const what = `estimated ${s.resource === 'vram' ? 'VRAM' : 'RAM'} ${s.estimateBytes !== undefined ? gib(s.estimateBytes) : '?'} > budget ${s.budgetBytes !== undefined ? gib(s.budgetBytes) : '?'} (planning snapshot)`
      add('ctx.planned-skip', { config: id(c), ctx: fmtCtx(s.ctx), what }, [num('estimateBytes', s.estimateBytes ?? null, 'estimated', s.ctx, id(c)), num('budgetBytes', s.budgetBytes ?? null, 'estimated', s.ctx, id(c))],
        { configId: id(c), action: s.resource === 'vram' ? vramRemedy(c.input.config, s) : action('rerun-idle') })
    }
  }
  for (const c of everyone) {
    const reasons = c.cs.cliff.steps.flatMap((s) => s.reasons)
    const spill = reasons.find((r) => r.code === 'shared_spill')
    if (spill) {
      const runs = [...c.scored.runs].sort((a, b) => a.ctx - b.ctx)
      const i = runs.findIndex((r) => r.ctx === spill.toCtx)
      const prev = i > 0 ? runs[i - 1] : null
      const d0 = prev ? val(prev.decodeTps, true) : null, d1 = val(runs[i]?.decodeTps, true)
      const delta = prev && d0 !== null && d1 !== null ? `; decode ${t1(d0)} → ${t1(d1)} t/s vs ${fmtCtx(prev.ctx)}` : ''
      const clean = c.cs.cliff.spillFreeUpTo
      const single = ' — one peak per rung, sample count not recorded: provisional'
      const act = clean ? action('use-context', fmtCtx(clean)) : c.input.config.kvType === 'f16' ? action('enable-kv-q8') : action('inspect-diagnostics')
      if (spill.metric === 'peakSharedGpuRawBytes') {
        push('ctx.spill', tag('ctx.spill', `${id(c)}: raw shared-GPU usage (host-pinned excluded) grew +${gib(spill.to!)} at ${fmtCtx(spill.toCtx)} vs ${spill.fromCtx ? fmtCtx(spill.fromCtx) : 'the previous rung'} — above the ${gib(v.cfg.cliff.rawSharedGrowthBytes)} growth rule (raw-growth-1); ${adjustedText(runs[i]?.peakSharedGpuBytes)}${delta}${single}.`),
          [num('peakSharedGpuRawBytes', spill.to, 'measured', spill.toCtx, id(c), { algorithm: 'raw-growth-1' })], { configId: id(c), action: act })
      } else {
        add('ctx.spill', { config: id(c), threshold: gib(v.cfg.cliff.sharedSpillBytes), ctx: fmtCtx(spill.toCtx), spill: gib(spill.to!), delta, single },
          [num('peakSharedGpuBytes', spill.to, 'measured', spill.toCtx, id(c), { algorithm: 'adjusted-spill' })], { configId: id(c), action: act })
      }
    }
    // I-2.8 placement spill (heuristic): shared residency while ≥ 1 GiB dedicated was free in the SAME window
    // (adapterFreeAtSharedPeakBytes — required; no budget fallback). Cleared after a fresh restart → placement (note);
    // not re-measured → cause unconfirmed (warn, restart-runtime); persisted after the restart → capacity-suspect,
    // carried by the reconciled spill metric / cliff / run reason, never "not a capacity limit" here.
    // Use the same benign first-rung baseline as the runner before judging placement (I-4.0).
    const firstRung = c.input.runs[0]
    const firstRaw = firstRung?.peakSharedGpuRawBytes ? val(firstRung.peakSharedGpuRawBytes) : null
    const firstPin = firstRung?.hostPinnedBytes ? val(firstRung.hostPinnedBytes) ?? 0 : 0
    const baseline = firstRaw !== null && firstRaw - firstPin < v.cfg.cliff.rawSharedGrowthBytes ? Math.max(0, firstRaw - firstPin) : 0
    const resident = (x: { peakSharedGpuRawBytes?: Metric; peakSharedGpuBytes: Metric }, pin: number) => { const raw = x.peakSharedGpuRawBytes ? val(x.peakSharedGpuRawBytes) : null; return raw === null ? val(x.peakSharedGpuBytes) : Math.max(0, raw - pin - baseline) }
    const over = (x: number | null) => x !== null && x > v.cfg.cliff.sharedSpillBytes
    for (const r of c.input.runs) {
      const pin = val(r.hostPinnedBytes) ?? 0
      const first = r.placementFirst, ev0 = first ?? r
      const sh0 = resident(ev0, pin), fr = ev0.adapterFreeAtSharedPeakBytes ? val(ev0.adapterFreeAtSharedPeakBytes) : null
      if (!over(sh0) || fr === null || fr < P('ctx.placement-spill', 'freeBytes')) continue
      const shN = resident(r, pin)
      if (first && (over(shN) || !isUsable(r))) continue // persisted / unmeasured retry: not placement
      const dec = (m: Metric) => (val(m, true) === null ? '?' : t1(val(m, true)!))
      // N3: an unknown re-measurement is unknown — never "cleared".
      const unknown = !!first && shN === null
      const outcome = unknown
        ? ' — re-measured after a fresh restart, but that attempt had no shared-memory reading: cause unknown (placement or capacity); re-measure this rung'
        : first
          ? `; it cleared after a fresh restart (${gib(shN!)} shared, decode ${dec(r.decodeTps)} vs ${dec(first.decodeTps)} t/s before) — driver placement after a previous large load, not a capacity limit`
          : ' — not re-measured: cause unconfirmed (driver placement or capacity); restart the runtime and re-measure this rung'
      add('ctx.placement-spill', { config: id(c), ctx: fmtCtx(r.ctx), shared: gib(sh0!), free: gib(fr), outcome },
        [ev('peakSharedGpuRawBytes', ev0.peakSharedGpuRawBytes ?? ev0.peakSharedGpuBytes, r.ctx, id(c)), ev('adapterFreeAtSharedPeakBytes', ev0.adapterFreeAtSharedPeakBytes!, r.ctx, id(c))],
        { configId: id(c), action: first && !unknown ? null : action('restart-runtime'), severity: first && !unknown ? 'note' : 'warn' })
    }
    // I-2.6 recovered dip; disclose skipped rungs between
    const u = [...c.scored.runs].filter(isUsable).sort((a, b) => a.ctx - b.ctx)
    for (let i = 1; i + 1 < u.length; i++) {
      const a = val(u[i - 1].decodeTps, true)!, b = val(u[i].decodeTps, true)!, cc = val(u[i + 1].decodeTps, true)!
      if (b / a <= v.cfg.cliff.decodeDropRatio && a - b >= v.cfg.cliff.minDecodeDropTps && cc > a * v.cfg.cliff.decodeDropRatio) {
        const gap = u[i + 1].ctx > u[i].ctx * 2 ? `; rungs between ${fmtCtx(u[i].ctx)} and ${fmtCtx(u[i + 1].ctx)} were not tested` : ''
        add('ctx.recovered-dip', { config: id(c), ctx: fmtCtx(u[i].ctx), a: t1(a), b: t1(b), next: fmtCtx(u[i + 1].ctx), c: t1(cc), gap }, [ev('decodeTps', u[i].decodeTps, u[i].ctx, id(c))], { configId: id(c) })
      }
    }
  }
  if (profile.requiredContext) {
    const req = profile.requiredContext
    const models = [...new Map(allInputs.map((i) => [i.model.id, i.model])).values()]
    let reached = 0
    const per = models.map((m) => {
      const mine = v.coverageAll.filter((c) => c.input.model.id === m.id)
      const ok = mine.find((c) => (c.coverage.largestCleanTested ?? 0) >= req)
      if (ok) { reached++; return `${m.name} reached (${id(ok)})` }
      const tried = allInputs.filter((i) => i.model.id === m.id).flatMap((i) => i.runs).filter((r) => r.ctx >= req)
      const unverified = mine.flatMap((c) => c.dropped).find((d) => d.ctx >= req)
      if (unverified) return `${m.name} measured at ${fmtCtx(unverified.ctx)} but not verified (${unverified.reason})`
      const failed = tried.find((r) => r.status !== 'pass' && r.status !== 'degraded')
      if (failed) return `${m.name} tested and failed at ${fmtCtx(failed.ctx)} (${failed.failureKind ?? failed.status})`
      if (tried.length) return `${m.name} tested at ${fmtCtx(tried[0].ctx)} but not clean (${mine.map((c) => c.coverage.limitKind).join(', ') || 'degraded'})`
      const best = mine.map((c) => c.coverage).sort((a, b) => (b.largestCleanTested ?? 0) - (a.largestCleanTested ?? 0))[0]
      return `${m.name} not tested (${best ? `largest clean ${best.largestCleanTested === null ? 'none' : fmtCtx(best.largestCleanTested)}, ${best.limitKind}` : 'no usable configuration'})`
    })
    add('ctx.required', { required: fmtCtx(req), perModel: per.join('; ') || 'no configuration ran' }, everyone.map((c) => ev('largestCleanTested', c.cs.cliff.practicalContextCeiling, undefined, id(c))),
      { severity: reached === models.length && models.length ? 'info' : 'critical', action: reached === models.length && models.length ? null : action('lower-required-context') })
  }

  // §5 quality
  for (const c of firstOf(everyone)) {
    const q = c.cs.components.quality
    const rows = (c.gen ? c.gen.gq.results : c.input.quality) as UncertaintyRow[]
    const mine = rows.filter((r) => profile.promptSetIds.includes(r.category))
    if (q.input.kind === 'measured' && q.lower !== undefined) {
      const cov = q.coverage!
      let flagsU: ReturnType<typeof categoryFlags> = []
      try { flagsU = categoryFlags(mine, { weakAtMost: P('quality.category', 'weakAtMost'), codingAtMost: P('quality.category', 'codingAtMost'), minItems: 1 }) } catch { /* quarantined */ }
      const cats = CATS.filter((k) => profile.promptSetIds.includes(k)).map((k) => {
        const f = flagsU.find((x) => x.category === k)
        if (!f || !f.validItems) return null
        const completions = mine.filter((x) => x.category === k && x.evaluationStatus !== 'infra_error' && x.evaluationStatus !== 'unrun').length
        return `${k} ${Math.round(f.rate! * 100)} % (${f.validItems} item${f.validItems === 1 ? '' : 's'}${completions > f.validItems ? `, ${completions} completions` : ''})`
      }).filter(Boolean).join(', ')
      add('quality.report', { model: c.input.model.name, q: Math.round(q.score), lo: Math.round(q.lower), hi: Math.round(q.upper!), method: q.method, version: q.algorithm, items: cov.uniqueItems, skills: cov.uniqueSkills, samples: cov.uniqueItems ? Math.round(cov.completions / cov.uniqueItems) : 0, categories: cats },
        [ev('quality', q.input, undefined, id(c), { samples: cov.completions, algorithm: `${q.method} ${q.algorithm}` })], { configId: id(c) })
      const missingCats = profile.promptSetIds.filter((k) => (flagsU.find((x) => x.category === k)?.validItems ?? 0) < P('quality.coverage', 'minPerCategory'))
      if (cov.uniqueItems < P('quality.coverage', 'minItems') || missingCats.length) {
        add('quality.coverage', { model: c.input.model.name, why: [cov.uniqueItems < P('quality.coverage', 'minItems') ? `${cov.uniqueItems} unique items < ${P('quality.coverage', 'minItems')}` : null, missingCats.length ? `< ${P('quality.coverage', 'minPerCategory')} items in ${missingCats.join(', ')}` : null].filter(Boolean).join('; ') }, [ev('quality', q.input, undefined, id(c))], { configId: id(c) })
      }
    }
    if (mine.length && !q.quarantined) {
      try {
        const flags = categoryFlags(mine, { weakAtMost: P('quality.category', 'weakAtMost'), codingAtMost: P('quality.category', 'codingAtMost'), minItems: P('quality.category', 'minItems') })
          .filter((f) => profile.promptSetIds.includes(f.category) && f.validItems > 0 && f.flags.length)
        const coding = ['coding', 'large_coding', 'long_context_coding'].includes(profile.id)
        const list = flags.map((f) => f.flags.includes('insufficient coverage') ? `${f.category}: insufficient coverage (${f.validItems} item${f.validItems === 1 ? '' : 's'})`
          : `${f.category} ${Math.round(f.rate! * 100)} %${f.flags.includes('weak') ? ' weak' : ''}${coding && f.flags.includes('coding warning') ? ' (coding warning)' : ''}`)
          .filter((x) => !/^coding \d+ %$/.test(x))
        const warn = coding && flags.some((f) => f.flags.includes('coding warning'))
        if (list.length) add('quality.category', { model: c.input.model.name, list: list.join('; ') }, flags.map((f) => num(`quality.${f.category}`, f.rate, 'measured', undefined, id(c), { samples: f.validItems })), { configId: id(c), severity: warn ? 'warn' : 'note', action: warn ? action('try-thinking-config') : null })
      } catch { /* quarantined rows: reported by I-5.7 */ }
    }
    if (q.input.kind === 'estimated') add('quality.estimated', { config: id(c), basis: q.input.source ?? 'parameters × quantization' }, [ev('quality', q.input, undefined, id(c))], { configId: id(c) })
    // I-5.7 / I-5.8 over every generation config's rows
    const sets = [{ gen: 'thinking off (T=0)', rows: c.input.quality as UncertaintyRow[] }, ...c.genOptions.filter((g) => g.gq.gen.id !== 'off').map((g) => ({ gen: genLabel(g.gq.gen), rows: g.gq.results as UncertaintyRow[] }))]
    for (const s of sets) {
      const infra = s.rows.filter((r) => r.evaluationStatus === 'infra_error')
      if (infra.length) add('quality.harness-invalid', { model: c.input.model.name, gen: s.gen, n: infra.length }, infra.map((r) => num('evaluationStatus', 'infra_error', 'measured', undefined, id(c), { source: r.testId })), { configId: id(c) })
      const trunc = s.rows.filter((r) => r.evaluationStatus === 'truncated' || r.outputTruncated)
      if (trunc.length) {
        const budgets = [...new Set(trunc.map((r) => (r as { maxTokens?: number }).maxTokens).filter((x) => x !== undefined))]
        add('quality.truncated', { model: c.input.model.name, gen: s.gen, n: trunc.length, budget: budgets.length ? `${budgets.join('/')} tokens` : 'budget not recorded', items: [...new Set(trunc.map((r) => r.testId))].join(', ') }, trunc.map((r) => num('outputTruncated', 'true', 'measured', undefined, id(c), { source: r.testId })), { configId: id(c) })
      }
    }
  }
  if (v.quality) {
    const { a, b, difference: d, decisive, reason } = v.quality
    const neutral = trace.steps.find((s) => s.kind === 'quality-neutralized')
    const text = d && decisive
      ? `${id(a)} vs ${id(b)}: quality difference ${fmtDiff(d)} on ${d.sharedItems} shared items — the interval excludes 0${trace.steps.some((s) => s.kind === 'quality-decides') ? '; quality decided' : ''}`
      : `${id(a)} vs ${id(b)}: insufficient evidence to distinguish quality on this suite (${d ? `difference ${fmtDiff(d)} on ${d.sharedItems} shared items includes 0` : reason}); the quality delta is neutralized — ${neutral ? `without it ${neutral.detail.split('; without its contribution ')[1] ?? ''} decided` : 'the remaining score decided'}`
    push('quality.difference', tag('quality.difference', text), [ev('quality', a.cs.components.quality.input, undefined, id(a)), ev('quality', b.cs.components.quality.input, undefined, id(b))],
      { severity: decisive ? 'info' : 'warn', action: decisive ? null : action('run-thorough-quality') })
  }
  for (const c of firstOf(everyone.filter((x) => x.input.model.supportsThinking || x.input.model.genKnobs?.supportsThinking))) {
    const used = c.gen ? genLabel(c.gen.gq.gen) : 'thinking off (T=0)'
    const alt = c.genOptions.filter((g) => g !== c.gen && g.comparable)
    const also = alt.length ? `; also measured: ${alt.map((g) => `${genLabel(g.gq.gen)} Q ${Math.round(g.cs.components.quality.score)}, effective ${val(g.gq.effectiveTps) === null ? '?' : t1(val(g.gq.effectiveTps)!)} t/s`).join('; ')}` : ''
    add('quality.thinking', { model: c.input.model.name, gen: used, also }, [], { configId: id(c) })
  }
  if (everyone.some((c) => c.qualityMeasured)) add('quality.not-a-leaderboard', {}, [])

  // §3 speed (winner, or the best provisional one)
  const lead = w ?? v.provisionalWinner
  if (lead) {
    const rc = lead.scored.runs.find((r) => r.ctx === lead.cs.recommendedCtx)
    if (!rc) push('ctx.recommended', tag('ctx.recommended', `Recommended -c: unavailable — no measured passing rung (${lead.cs.referenceWhy ?? 'no usable rung'})`), [], { configId: id(lead), evaluable: false })
    else add('ctx.recommended', { ctx: fmtCtx(rc.ctx), why: lead.cs.recommendedWhy ?? '' }, [num('recommendedCtx', rc.ctx, 'measured', rc.ctx, id(lead))], { configId: id(lead) })
    const d = rc ? val(rc.decodeTps, true) : null
    if (rc && d !== null) {
      const b = rule('speed.decode-band').params
      const gate = profile.minDecodeTps === undefined ? 'no decode gate' : v.request.minDecodeTps != null ? `${profile.label} gate ${DEFAULT_SCORING_CONFIG.profiles[profile.id].minDecodeTps ?? '—'} t/s; your floor ${profile.minDecodeTps} t/s` : `${profile.label} gate ${profile.minDecodeTps} t/s`
      push('speed.decode-band', tag('speed.decode-band', `${id(lead)}: decode ${t1(d)} t/s at ${fmtCtx(rc.ctx)} (${rc.promptTokens ?? '?'} prompt tokens) — ${band(d, [[b.unusable, 'unusable'], [b.patient, 'patient'], [b.usable, 'usable'], [b.comfortable, 'comfortable'], [b.snappy, 'snappy']], 'very fast streaming')} (${gate})`),
        [ev('decodeTps', rc.decodeTps, rc.ctx, id(lead))], { configId: id(lead), action: null })
    }
    const t = rc ? val(rc.ttftMs, true) : null
    if (rc && t !== null) {
      const b = rule('speed.ttft-band').params
      const over = t > profile.latencyToleranceMs
      const sc = lead.scored.runs.find((r) => r.ctx === lead.cs.referenceCtx)
      const scoredAt = sc && sc.ctx !== rc.ctx ? ` (scored at ${fmtCtx(sc.ctx)}: TTFT ${val(sc.ttftMs, true) === null ? 'unknown' : sec(val(sc.ttftMs, true)!)}, decode ${t1(val(sc.decodeTps, true)!)} t/s)` : ''
      const accepted = over && profile.requiredContext ? `; accepted because you required ${fmtCtx(profile.requiredContext)}` : over && profile.latencyAdvisory ? '; latency is advisory for this workload' : ''
      push('speed.ttft-band', tag('speed.ttft-band', `Recommended context ${fmtCtx(rc.ctx)}: TTFT ${sec(t)} with ${rc.promptTokens ?? '?'} prompt tokens — ${band(t, [[b.immediate, 'immediate'], [b.short, 'short wait'], [b.noticeable, 'noticeable'], [b.long, 'long']], 'very long')} (tolerance ${(profile.latencyToleranceMs / 1000).toFixed(0)} s${accepted}), decode ${d === null ? '?' : t1(d)} t/s${scoredAt}`),
        [ev('ttftMs', rc.ttftMs, rc.ctx, id(lead))], { configId: id(lead), severity: over ? (accepted ? 'note' : 'warn') : 'info', action: over && !accepted && lead.coverage.largestCleanTested ? action('use-context', fmtCtx(Math.min(...lead.scored.runs.filter((r) => isUsable(r) && (val(r.ttftMs, true) ?? Infinity) <= profile.latencyToleranceMs).map((r) => r.ctx), rc.ctx))) : null })
    }
    const g = lead.gen?.gq
    if (g?.gen.thinking) {
      const e = val(g.effectiveTps, true), r = val(g.rawTps ?? { value: null, kind: 'unavailable' }, true), rt = val(g.reasoningTokens)
      if (e !== null && r !== null && e / r < P('speed.thinking-effective', 'ratio')) {
        add('speed.thinking-effective', { model: lead.input.model.name, gen: genLabel(g.gen), reasoning: Math.round(rt ?? 0), effective: t1(e), raw: t1(r), source: g.effectiveTps.kind === 'measured' ? 'runtime-reported (measured)' : 'by text length (estimated)' },
          [ev('effectiveTps', g.effectiveTps), ev('rawTps', g.rawTps)], { configId: id(lead) })
      }
    }
  }
  for (const c of everyone) {
    const pd = c.cs.cliff.steps.flatMap((s) => s.reasons).find((r) => r.code === 'prefill_drop')
    if (pd) {
      const ra = c.scored.runs.find((r) => r.ctx === pd.fromCtx), rb = c.scored.runs.find((r) => r.ctx === pd.toCtx)
      const ta = val(ra?.ttftMs, true), tb = val(rb?.ttftMs, true)
      add('speed.prefill-scaling', { config: id(c), a: t1(pd.from!), b: t1(pd.to!), from: fmtCtx(pd.fromCtx!), to: fmtCtx(pd.toCtx), tokens: ra?.promptTokens && rb?.promptTokens ? `prompt tokens ×${(rb.promptTokens / ra.promptTokens).toFixed(2)}` : 'prompt tokens not recorded', ta: ta === null ? '?' : sec(ta), tb: tb === null ? '?' : sec(tb) },
        [num('prefillTps', pd.to, 'measured', pd.toCtx, id(c))], { configId: id(c) })
    }
    const cand = c.input.config
    if (cand.expectDegraded && !cand.gpuLayersAll && machine.gpuDevice !== null) {
      const ctx = c.cs.referenceCtx
      const same = everyone.filter((x) => x !== c && x.input.model.id === c.input.model.id)
        .map((x) => ({ x, r: x.scored.runs.find((r) => r.ctx === ctx && isUsable(r)) })).filter((p) => p.r)
      const vs = same.length ? ` vs ${same.map((p) => `${t1(val(p.r!.decodeTps, true)!)} t/s (${id(p.x)})`).join(', ')} at the same context` : ''
      add('speed.partial-offload', { config: cand.id, layers: cand.gpuLayers === 0 ? 'CPU only' : `${cand.gpuLayers}/${c.input.model.layers} layers${cand.kvOffload === false ? ', KV cache in RAM' : ''}`, decode: c.decode === null ? '?' : t1(c.decode), ctx: ctx === null ? '?' : fmtCtx(ctx), vs },
        [num('decodeTps', c.decode, 'measured', ctx ?? undefined, cand.id)], { configId: cand.id })
    }
  }
  // I-3.9: the same model/config on two backend builds (HIP id = Vulkan id + '|hip'): decode at the largest rung both
  // measured, and each one's dedicated ceiling (max per-PID dedicated over usable rungs). Backend is a config axis.
  const byId = new Map(everyone.map((c) => [c.input.config.id, c]))
  for (const o of everyone.filter((c) => c.input.config.backend === 'hip')) {
    const b = byId.get(o.input.config.id.replace(/\|hip$/, ''))
    if (!b) continue
    const ok = (c: typeof o) => c.input.runs.filter(isUsable)
    const common = ok(o).map((r) => r.ctx).filter((x) => ok(b).some((r) => r.ctx === x))
    if (!common.length) continue
    const ctx = Math.max(...common)
    const at = (c: typeof o) => ok(c).find((r) => r.ctx === ctx)!
    const ceil = (c: typeof o) => { const xs = ok(c).map((r) => val(r.peakVramBytes)).filter((x): x is number => x !== null); return xs.length ? gib(Math.max(...xs)) : 'unavailable' }
    add('speed.backend', { other: 'HIP', base: 'Vulkan', config: id(b), dOther: t1(val(at(o).decodeTps, true)!), dBase: t1(val(at(b).decodeTps, true)!), ctx: fmtCtx(ctx), cOther: ceil(o), cBase: ceil(b) },
      [ev('decodeTps', at(o).decodeTps, ctx, id(o)), ev('decodeTps', at(b).decodeTps, ctx, id(b))], { configId: id(o) })
  }
  // I-9.2: a smaller sibling quantization of the linked repo would put more layers on the GPU (estimated, not run).
  // Siblings present in this session are benchmarked as their own models and skipped here.
  const localFiles = allInputs.map((c) => c.model.id.split(/[\\/]/).pop()!)
  for (const m of [...new Map(allInputs.map((c) => [c.model.id, c.model])).values()]) {
    for (const s of quantSuggestions(machine, m, profile, undefined, localFiles)) {
      add('plan.sibling-quant', { model: m.name, text: s.text }, [ev('estVramBytes', s.estVramBytes, s.ctx), num('siblingSizeBytes', s.sibling.sizeBytes, 'declared')],
        { action: action('download', `${s.sibling.repoId}/${s.sibling.path}`) })
    }
  }
  for (const m of [...new Map(allInputs.map((c) => [c.model.id, c.model])).values()]) {
    if (m.expertCount && m.expertCount > 0) add('speed.moe-note', { model: m.name, used: m.expertUsedCount ?? '?', experts: m.expertCount }, [num('expertCount', m.expertCount, 'declared')])
  }

  // §4 memory
  if (lead) {
    const rc = lead.scored.runs.find((r) => r.ctx === (lead.cs.recommendedCtx ?? lead.cs.referenceCtx))
    const peak = val(rc?.peakVramBytes)
    const plan = lead.input.config.planning
    if (rc && lead.input.config.gpuLayers > 0) {
      if (peak !== null && plan?.planningVramBudgetBytes != null) {
        // Budget = min(total − in use − reserve, the effective per-process budget) — the tighter one is named.
        const eff = plan.effectiveBudget ? val(plan.effectiveBudget, true) : null
        const adapterLeft = plan.planningVramBudgetBytes - plan.planningReserveBytes
        // Only a measured per-process budget is a planning basis (the estimated fallback is disclosed, not applied).
        const measuredEff = eff !== null && plan.effectiveBudget!.kind === 'measured'
        const perProcess = measuredEff && eff! < adapterLeft
        const remaining = (perProcess ? eff! : adapterLeft) - peak
        const low = remaining < P('mem.budget-basis', 'warnBytes')
        const effText = eff === null ? '' : measuredEff ? `effective per-process budget ${gib(eff)} (measured on this GPU/driver/backend)`
          : `per-process budget ${gib(eff)} estimated — ${plan.effectiveBudget!.source}; not applied to planning`
        push('mem.budget-basis', tag('mem.budget-basis', `${id(lead)} at ${fmtCtx(rc.ctx)}: planning budget remaining ${sgib(remaining)} (${perProcess ? `${effText} − peak ${gib(peak)}; adapter budget ${gib(plan.planningVramBudgetBytes)} − reserve ${gib(plan.planningReserveBytes)} is looser` : `budget ${gib(plan.planningVramBudgetBytes)} − reserve ${gib(plan.planningReserveBytes)} − peak ${gib(peak)}${effText ? `; ${effText}${measuredEff ? ' is looser' : ''}` : ''}`}; in use at planning ${val(plan.vramInUse) === null ? `unavailable (${plan.vramInUse.reason ?? 'not measured'})` : `${gib(val(plan.vramInUse)!)} ${plan.vramInUse.kind}${plan.vramInUse.kind === 'estimated' ? ' — an assumed default, not a reading' : ''}`})${low ? ' — less room for other apps, spill risk increases' : ''}`),
          [ev('peakVramBytes', rc.peakVramBytes, rc.ctx, id(lead)), num('planningVramBudgetBytes', plan.planningVramBudgetBytes, 'measured'), num('planningReserveBytes', plan.planningReserveBytes, 'declared'), ev('vramInUseBytes', plan.vramInUse), ...(plan.effectiveBudget ? [ev('vramEffectiveBudgetBytes', plan.effectiveBudget)] : [])],
          { configId: id(lead), severity: low ? 'warn' : 'info', action: low && lead.coverage.largestCleanTested ? action('use-context', fmtCtx(lead.coverage.largestCleanTested)) : null })
      } else if (peak !== null && val(machine.vramBytes, true) !== null) {
        push('mem.budget-basis', tag('mem.budget-basis', `${id(lead)} at ${fmtCtx(rc.ctx)}: per-PID dedicated ${gib(peak)} of the adapter total ${gib(val(machine.vramBytes, true)!)} (basis: per-PID vs adapter total; planning budget not recorded — headroom not evaluable)`),
          [ev('peakVramBytes', rc.peakVramBytes, rc.ctx, id(lead)), ev('vramBytes', machine.vramBytes)], { configId: id(lead), evaluable: false, action: null })
      } else push('mem.budget-basis', tag('mem.budget-basis', `${id(lead)}: VRAM headroom not evaluable (peak or total unavailable)`), [ev('peakVramBytes', rc.peakVramBytes, rc.ctx, id(lead))], { configId: id(lead), evaluable: false, action: null })
    }
  }
  const inUse = machine.vramInUseBytes
  const plans = everyone.map((c) => c.input.config.planning).filter(Boolean)
  if (val(inUse) !== null && val(inUse)! > P('mem.in-use-at-plan', 'bytes')) push('mem.in-use-at-plan', tag('mem.in-use-at-plan', `Planned while ${gib(val(inUse)!)} of VRAM was already in use by other apps (measured at the scan); rerun on an idle GPU for the full budget`), [ev('vramInUseBytes', inUse)])
  else if (val(inUse) === null) push('mem.in-use-at-plan', tag('mem.in-use-at-plan', `VRAM in use by other apps was not measured${plans[0] ? `; planning assumed ${gib(val(plans[0]!.vramInUse) ?? 0)} (estimated)` : ''}`), [ev('vramInUseBytes', inUse)], { action: action('retry-telemetry') })
  for (const c of everyone) {
    const cid = id(c)
    for (const r of c.input.runs) {
      const floor = r.ramFloorBytes ?? data.planningSnapshot?.ramFloorBytes
      const credit = r.mmapCreditBytes ?? data.planningSnapshot?.mmapCreditBytes
      const min = val(r.minRamAvailBytes)
      if (r.failureKind === 'guard_abort') add('mem.ram-floor', { config: cid, ctx: fmtCtx(r.ctx), what: `stopped by the safety guard: ${r.reason ?? 'RAM floor'}` }, [ev('minRamAvailBytes', r.minRamAvailBytes, r.ctx, cid)], { configId: cid, severity: 'critical' })
      else if (min !== null && floor !== undefined && min - floor < P('mem.ram-floor', 'marginBytes')) {
        const dist = min - floor
        add('mem.ram-floor', { config: cid, ctx: fmtCtx(r.ctx), what: `minimum RAM available ${gib(min)} is ${sgib(dist)} vs the ${gib(floor)} floor${credit !== undefined ? ` (mmap credit ${gib(credit)} applied separately by the guard)` : ''}` },
          [ev('minRamAvailBytes', r.minRamAvailBytes, r.ctx, cid), num('ramFloorBytes', floor, 'declared'), ...(credit !== undefined ? [num('mmapCreditBytes', credit, 'estimated')] : [])], { configId: cid, severity: dist < 0 ? 'critical' : 'warn' })
      }
    }
    // I-4.4: only with the before / during / after lifecycle observed
    const runs = [...c.input.runs].sort((a, b) => a.ctx - b.ctx)
    for (let i = 0; i + 1 < runs.length && c.input.config.mmap !== false; i++) {
      const before = val(runs[i].ramAvailBeforeLoadBytes), min = val(runs[i].minRamAvailDuringLoadBytes), after = val(runs[i + 1].ramAvailBeforeLoadBytes)
      if (before === null || min === null || after === null || before - min < GiB) continue
      if (after >= before - P('mem.mmap-note', 'releasedWithinBytes')) {
        add('mem.mmap-note', { config: cid, ctx: fmtCtx(runs[i].ctx), drop: gib(before - min), before: gib(before), min: gib(min), after: gib(after) },
          [ev('ramAvailBeforeLoadBytes', runs[i].ramAvailBeforeLoadBytes, runs[i].ctx, cid), ev('minRamAvailDuringLoadBytes', runs[i].minRamAvailDuringLoadBytes, runs[i].ctx, cid), ev('ramAvailBeforeLoadBytes', runs[i + 1].ramAvailBeforeLoadBytes, runs[i + 1].ctx, cid)], { configId: cid })
        break
      }
    }
    const sp = c.cs.cliff.steps.find((s) => s.reasons.some((x) => x.code === 'shared_spill'))
    const at = sp && c.input.runs.find((r) => r.ctx === sp.ctx)
    const total = val(machine.vramBytes, true), ded = at ? val(at.peakVramBytes) : null
    if (at && total && ded !== null && at.peakVramPlateauVersion === 'pre-spill-1' && (at.peakVramPlateauSamples ?? 0) >= P('mem.saturation-observed', 'minSamples')) {
      add('mem.saturation-observed', { config: cid, ded: gib(ded), pct: Math.round((ded / total) * 100), samples: at.peakVramPlateauSamples }, [ev('peakVramBytes', at.peakVramBytes, sp!.ctx, cid, { samples: at.peakVramPlateauSamples })], { configId: cid })
    }
  }

  // G12: the gate verdicts themselves, as insights with their catalog action.
  for (const c of everyone) for (const f of c.failures) {
    const key = f.ruleId === rule('gate.context-floor').id ? 'gate.context-floor' : f.ruleId === rule('gate.quality-min').id ? 'gate.quality-min' : f.ruleId === rule('gate.stability').id ? 'gate.stability' : null
    if (key) push(key, `${id(c)}: ${f.text}`, [], { configId: id(c) })
  }

  // §6 eligibility and stability — failures from ALL persisted runs (I-6.3)
  for (const c of everyone) for (const d of c.dropped) {
    add('stab.cold-rows', { config: id(c), ctx: fmtCtx(d.ctx), reason: d.reason }, [], { configId: id(c), action: /version/.test(d.reason) ? action('rerun-comparable') : action('rerun-idle') })
  }
  for (const c of everyone) {
    for (const r of c.scored.runs) {
      const reps = (r.repDecodeTps ?? []).filter((x) => Number.isFinite(x) && x > 0)
      if (reps.length < P('stab.rep-spread', 'minReps')) continue
      const s = [...reps].sort((a, b) => a - b), med = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2
      const spread = (s[s.length - 1] - s[0]) / med
      if (spread > P('stab.rep-spread', 'spreadOfMedian')) add('stab.rep-spread', { config: id(c), ctx: fmtCtx(r.ctx), reps: reps.map(t1).join(' / '), spread: (spread * 100).toFixed(1) }, reps.map((x) => num('decodeTps', x, 'measured', r.ctx, id(c))), { configId: id(c) })
    }
  }
  const persisted: StoredRun[] = data.allRuns ?? allInputs.flatMap((i) => i.runs)
  for (const r of persisted) {
    if (!r.failureKind || !['oom', 'device_lost', 'crash', 'config_drift', 'guard_abort'].includes(r.failureKind)) continue
    if (r.failureKind === 'guard_abort' && !data.allRuns) continue // already critical under I-4.3
    add('stab.failures', { config: r.configId, ctx: fmtCtx(r.ctx), kind: r.failureKind, detail: r.failureKind === 'device_lost' ? ` — GPU reset; results after ${r.endedAt !== undefined ? `t=${r.endedAt}` : 'it'} are suspect` : r.reason ? ` (${r.reason})` : '', superseded: r.supersededBy ? ` (superseded by retry ${r.supersededBy})` : '' },
      [num('failureKind', r.failureKind, 'measured', r.ctx, r.configId)], { configId: r.configId, severity: r.failureKind === 'device_lost' || r.failureKind === 'guard_abort' ? 'critical' : 'warn' })
  }
  const runs = [...allInputs.flatMap((c) => c.runs), ...(data.allRuns ?? [])]
  const verSet = (f: (x: NonNullable<(typeof runs)[number]['versions']>) => string | undefined) => [...new Set(runs.map((r) => (r.versions ? f(r.versions) : undefined)).filter((x): x is string => !!x))].sort()
  // Pre-backend-axis rows stored bare build tags. They denote the primary Vulkan
  // build; compare them as Vulkan without rewriting the historical row (I-6.2).
  const runtimeLabel = (value: string | null | undefined) => !value ? '?' : /^(vulkan|cuda|hip|cpu):/.test(value) ? value : `vulkan:${value}`
  for (const [what, set] of [['runtime/benchmark/prompt', verSet((x) => `${runtimeLabel(x.runtime)}/${x.benchmark}/${x.prompts}`)], ['quality suite', verSet((x) => x.quality)], ['rules', verSet((x) => x.rules)]] as const) {
    if (set.length > 1) add('stab.versions', { what, versions: set.join(', ') }, [])
  }

  // §7 comparisons
  if (v.scoringRung !== null && everyone.length > 1) {
    const short = everyone.filter((c) => c.cs.reachesScoringRung === false)
    add('cmp.scope', { workload: profile.label, text: `candidates are compared at the common scoring rung ${fmtCtx(v.scoringRung)} (${trace.scoringRungWhy})${short.length ? `; not reaching it (speed read lower, latency 0): ${short.map(id).join(', ')}` : ''}` }, [num('scoringRung', v.scoringRung, 'measured')])
  }
  for (let i = 0; i < everyone.length; i++) {
    for (let j = i + 1; j < everyone.length; j++) {
      const a = everyone[i], b = everyone[j], ma = a.input.model, mb = b.input.model
      if (ma.id === mb.id || ma.quant === mb.quant || ma.arch !== mb.arch || !ma.paramCount || ma.paramCount !== mb.paramCount) continue
      const idA = ma as typeof ma & { baseModelId?: string; fineTuneId?: string }, idB = mb as typeof mb & { baseModelId?: string; fineTuneId?: string }
      const same = !!idA.baseModelId && idA.baseModelId === idB.baseModelId && (idA.fineTuneId ?? null) === (idB.fineTuneId ?? null)
      if (!same && (idA.fineTuneId || idB.fineTuneId) && idA.fineTuneId !== idB.fineTuneId) continue // different fine-tunes: not comparable as quantizations
      const { d } = a.qualityMeasured && b.qualityMeasured ? difference(a, b, profile, v.cfg) : { d: null }
      const diffText = d ? `quality difference ${fmtDiff(d)}` : 'quality difference not measured'
      push('cmp.identity', tag('cmp.identity', same
        ? `${ma.name} (${ma.quant}) vs ${mb.name} (${mb.quant}): same base model, two quantizations — ${diffText}; no non-inferiority margin is declared, so neither is preferred on quality`
        : `${ma.name} vs ${mb.name}: related models (same architecture and size; identity not verified) — ${diffText}`), [], {})
    }
  }

  // §8 generation configs
  for (const c of firstOf(everyone.filter((x) => x.genOptions.length))) {
    for (const g of c.genOptions.filter((x) => !x.comparable)) add('gen.comparable', { model: c.input.model.name, gen: genLabel(g.gq.gen), why: g.why ?? 'not comparable' }, [], { configId: id(c), evaluable: false })
    const off = c.genOptions.find((g) => !g.gq.gen.thinking)
    const g = c.gen
    if (g && off && g !== off) {
      const { d } = difference(g.gq.results as UncertaintyRow[], off.gq.results as UncertaintyRow[], profile, v.cfg, 'gen')
      const lo = val(off.gq.effectiveAnswerLatencyMs, true), hi = val(g.gq.effectiveAnswerLatencyMs, true)
      const ratio = lo && hi ? (hi >= lo ? `answers ${t1(hi / lo)}× slower` : `answers ${t1(lo / hi)}× faster`) : 'answer time not measured'
      push('gen.best-config', tag('gen.best-config', `${c.input.model.name}: ${genLabel(g.gq.gen)}: Q ${d ? fmtDiff(d) : '?'} vs thinking off; ${ratio} (effective ${val(g.gq.effectiveTps) === null ? '?' : t1(val(g.gq.effectiveTps)!)} vs ${val(off.gq.effectiveTps) === null ? '?' : t1(val(off.gq.effectiveTps)!)} t/s)`),
        [ev('quality', g.cs.components.quality.input), ev('quality', off.cs.components.quality.input), ev('effectiveTps', g.gq.effectiveTps), ev('effectiveTps', off.gq.effectiveTps)], { configId: id(c) })
    }
    for (const x of c.genOptions) if (x.gq.stochastic) add('gen.stochastic', { model: c.input.model.name, gen: genLabel(x.gq.gen), samples: x.gq.samples }, [], { configId: id(c) })
    const efforts = c.input.model.genKnobs?.effortValues ?? []
    const think = c.genOptions.filter((x) => x.gq.gen.thinking && x.gq.gen.effort && x.comparable).sort((a, b) => efforts.indexOf(a.gq.gen.effort!) - efforts.indexOf(b.gq.gen.effort!))
    for (let i = 0; i + 1 < think.length; i++) {
      const lo = think[i], hi = think[i + 1]
      const rlo = val(lo.gq.reasoningTokens), rhi = val(hi.gq.reasoningTokens)
      const { d } = difference(hi.gq.results as UncertaintyRow[], lo.gq.results as UncertaintyRow[], profile, v.cfg, 'gen')
      if (rlo !== null && rhi !== null && rhi > rlo && d && d.lower <= 0 && d.upper >= 0) {
        add('gen.effort-saturation', { model: c.input.model.name, hi: hi.gq.gen.effort, lo: lo.gq.gen.effort, rhi: Math.round(rhi), rlo: Math.round(rlo), diff: fmtDiff(d) }, [ev('reasoningTokens', hi.gq.reasoningTokens), ev('reasoningTokens', lo.gq.reasoningTokens)], { configId: id(c) })
      }
    }
  }

  // I-0.1: critical first, then coverage/quality (+ provenance), speed, memory, comparisons, the rest.
  const rank = (i: Insight) => {
    if (i.severity === 'critical') return 0
    const s = rule(i.key).section
    return s === 2 || s === 5 || s === 1 ? 1 : s === 3 ? 2 : s === 4 ? 3 : s === 7 || s === 8 ? 4 : 5
  }
  return out.map((x, i) => ({ x, i })).sort((a, b) => rank(a.x) - rank(b.x) || a.i - b.i).map((a) => a.x)
}
