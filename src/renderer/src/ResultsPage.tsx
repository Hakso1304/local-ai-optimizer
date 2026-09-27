import { useCallback, useEffect, useState, type ReactNode } from 'react'
import type { Metric } from '../../shared/bench-types'
import type { WorkloadId } from '../../shared/bench-types'
import type { ComputedRecommendation, SessionCandidate, SessionDetail, SessionSummary } from '../../shared/types'
import type { TelemetrySample } from '../../shared/bench-events'
import { WORKLOADS } from '../../core/scoring/workloads'
import { genLabel, type GenRow } from '../../core/benchmark/gen'
import { ExportMenu } from './ExportMenu'
import { DecisionTrace } from './DecisionTrace'
import { ENGINE_RULES, InterpretPanel, Reason, insightsOf, rulesOf, type InsightActions } from './InterpretPanel'
import type { BenchPreset } from './LargeCodingCard'
import { LineChart, type Band } from './LineChart'
import { ParetoChart } from './ParetoChart'
import { SloFilter, type SloCheck } from './SloFilter'
import { TelemetryChart } from './TelemetryChart'
import { CtxPick, DemoBanner, M, Prov, fmtCtx, gib, num } from './ui'

const RESUMABLE = new Set(['cancelled', 'failed', 'paused', 'interrupted'])
const COLORS = ['#58a6ff', '#f0883e', '#a371f7', '#3fb950', '#db61a2']
const NA = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })

/** Largest measured value of a run metric across completed steps. */
function peak(c: SessionCandidate, k: 'peakVramBytes' | 'peakRamBytes'): Metric {
  const ms = c.runs.filter((r) => r.status === 'pass' || r.status === 'degraded').map((r) => r[k]).filter((m) => m.value != null)
  return ms.length ? ms.reduce((a, b) => (b.value! > a.value! ? b : a)) : NA('no completed step reported it')
}

const refRun = (c: SessionCandidate) => c.runs.find((r) => r.ctx === c.score?.referenceCtx)
const comp = (c: SessionCandidate, id: string) => c.score?.breakdown.find((b) => b.component === id)

/** Planner snapshot (§12) of the first config that has one: what the budgets were when the plan was made. */
function PlanningLine({ d }: { d: SessionDetail }) {
  const p = d.candidates.find((c) => c.config.planning)?.config.planning
  if (!p) return null
  const g = (b: number | null) => (b == null ? '—' : gib(b))
  return (
    <p className="muted">
      Planned with: VRAM {g(p.vramTotalBytes)} total, <M m={p.vramInUse} fmt={gib} /> in use by other apps, budget {g(p.planningVramBudgetBytes)}
      {' '}(reserve {gib(p.planningReserveBytes)}); RAM {g(p.ramAvailableBytes)} available (reserve {gib(p.ramReserveBytes)}); rules {p.candidateRulesVersion}
    </p>
  )
}

/** Structured skip (§12, rule I-2.3) when present, else the planner's text. */
function skipText(s: SessionCandidate['config']['skippedSteps'][number]): ReactNode {
  const k = s.skip
  if (!k) return s.reason
  const what = k.resource === 'declared' ? 'above the declared context' : `${k.resource.toUpperCase()} estimate ${k.estimateBytes != null ? gib(k.estimateBytes) : '?'} > budget ${k.budgetBytes != null ? gib(k.budgetBytes) : '?'}`
  return <><Reason text={`[${k.ruleId}] ${what}`} /> <span className="muted">— {s.reason}</span></>
}

/** Quality rows that are not valid grades (§12): infra errors, truncated outputs, unrun items. */
function EvalPills({ rows }: { rows: GenRow[] }) {
  const n = (st: string) => rows.filter((r) => r.evaluationStatus === st).length
  const trunc = rows.filter((r) => r.evaluationStatus === 'truncated' || r.outputTruncated).length
  const pills = [['infra_error', n('infra_error')], ['truncated', trunc], ['unrun', n('unrun')]] as const
  return <>{pills.filter(([, c]) => c > 0).map(([k, c]) => <span key={k} className="pill warn-pill" title={`${c} of ${rows.length} items ${k.replace('_', ' ')} — not a valid grade`}>{c} {k.replace('_', ' ')}</span>)}</>
}

/** Per model: every generation config the quality suite ran with (one row each), measured from the stored rows. */
function GenTable({ d, chosen, tag }: { d: SessionDetail; chosen: string | null; tag: string }) {
  const models = [...new Map(d.candidates.filter((c) => (c.genQuality?.length ?? 0) > 1).map((c) => [c.model.id, c])).values()]
  if (!models.length) return null
  return (
    <>
      <h2>Generation settings{tag}</h2>
      {models.map((c) => (
        <table key={c.model.id}>
          <thead><tr><th>{c.model.name}</th><th>Quality</th><th>Answer t/s (effective)</th><th>Answer latency</th><th>Reasoning tokens</th><th>Samples</th></tr></thead>
          <tbody>
            {c.genQuality!.map((g) => (
              <tr key={g.gen.id}>
                <td>{genLabel(g.gen)}{g.gen.id === chosen && <span className="pill">chosen</span>}{g.gen.source === 'model-card' && <span className="pill" title="sampling from the model's generation_config.json">model card</span>}</td>
                <td>{g.qualityScore != null ? g.qualityScore.toFixed(0) : '—'} <Prov kind="measured" /></td>
                <td><M m={g.effectiveTps} /></td>
                <td><M m={g.effectiveAnswerLatencyMs} fmt={(v) => `${(v / 1000).toFixed(1)} s`} /></td>
                <td><M m={g.reasoningTokens} fmt={(v) => num(v, 0)} /></td>
                <td>{g.samples}{g.stochastic && <span className="muted" title="T > 0: seeded, not bit-identical across builds/hardware"> (sampled)</span>}
                  <EvalPills rows={g.results as GenRow[]} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </>
  )
}

function Detail({ d, onRerun, go }: { d: SessionDetail; onRerun: (configId: string) => void; go: Go }) {
  const [exportCtx, setExportCtx] = useState<number | null>(null)
  const preset = (extra: Partial<BenchPreset>): BenchPreset => ({ workload: d.session.workload, heavyMode: d.session.heavyMode, requiredContext: d.session.requiredContext, ...extra })
  const actions: InsightActions = {
    enableHeavyMode: () => go('Benchmark', undefined, preset({ heavyMode: true })),
    runThoroughQuality: () => go('Benchmark', undefined, preset({ qualityMode: 'thorough' })),
    tryThinkingConfig: () => go('Benchmark', undefined, preset({ genSearch: true })),
    download: () => go('Download'),
    useContext: (ctx) => setExportCtx(ctx)
  }
  const rec = d.recommendation
  const xs = [...new Set(d.candidates.flatMap((c) => c.runs.map((r) => r.ctx)))].sort((a, b) => a - b)
  const at = (c: SessionCandidate, f: (r: SessionCandidate['runs'][number]) => number | null) => xs.map((x) => { const r = c.runs.find((q) => q.ctx === x); return r ? f(r) : null })
  const bands: Band[] = d.candidates.flatMap((c, i) => {
    const s = c.cliff.steps.find((st) => st.verdict !== 'pass')
    const from = s?.reasons.find((r) => r.fromCtx != null)?.fromCtx
    return s && from != null ? [{ from, to: s.ctx, color: COLORS[i % COLORS.length], label: `${s.verdict} ${fmtCtx(s.ctx)}` }] : []
  })
  // Same model can appear with several configs, so always show the configId too.
  const name = (id: string | null) => (id ? `${d.candidates.find((c) => c.config.id === id)?.model.name ?? '?'} — ${id}` : '—')
  const tag = d.session.demo ? ' (DEMO DATA)' : ''
  const req = d.session.requiredContext
  const meets = (c: SessionCandidate) => req != null && (c.cliff.practicalContextCeiling.value ?? 0) >= req
  const reqLine = req == null ? null : d.candidates.some(meets)
    ? `Required context: ${fmtCtx(req)} — met by ${d.candidates.filter(meets).map((c) => name(c.config.id)).join(', ')}`
    : `No configuration reached ${fmtCtx(req)}: ${d.candidates.map((c) => `${c.config.id} practical ${c.cliff.practicalContextCeiling.value != null ? fmtCtx(c.cliff.practicalContextCeiling.value) : 'none'} (limited by ${c.cliff.limitedBy})`).join('; ')}`
  const [slo, setSlo] = useState<SloCheck>(() => () => ({ ok: true, failed: [] }))
  const onSlo = useCallback((p: SloCheck) => setSlo(() => p), [])
  const [tele, setTele] = useState<{ key: string; samples: TelemetrySample[] } | null>(null)

  return (
    <>
      {d.session.demo && <DemoBanner />}
      {insightsOf(rec).length > 0 && <InterpretPanel insights={insightsOf(rec)} actions={actions} tag={tag} />}
      <h2>Comparison — {d.session.workload}{tag}</h2>
      <PlanningLine d={d} />
      <SloFilter session={d.session} candidates={d.candidates} onChange={onSlo} />
      <table>
        <thead><tr><th /><th>Model</th><th>Quant</th><th>Config</th><th>Recommended ctx</th>{req != null && <th>Required {fmtCtx(req)}</th>}<th>Practical ctx</th><th>Quality</th><th>Decode t/s</th><th>Prefill t/s</th><th>Peak VRAM</th><th>Peak RAM</th><th>Stability</th><th /></tr></thead>
        <tbody>
          {d.candidates.map((c, i) => {
            const q = comp(c, 'quality'), st = comp(c, 'stability'), r = refRun(c), fit = slo(c)
            return (
              <tr key={c.config.id} id={`cand-row-${i}`} style={fit.ok ? undefined : { opacity: 0.45 }} title={fit.ok ? undefined : `Outside constraints: ${fit.failed.join(', ')}`}>
                <td><span className="swatch" style={{ background: COLORS[i % COLORS.length] }} /></td>
                <td>{c.model.name}{rec?.best?.configId === c.config.id && <span className="pill">best</span>}
                  {!fit.ok && <span className="pill warn-pill">{rec?.best?.configId === c.config.id ? 'outside your constraints' : 'filtered'}: {fit.failed.join(', ')}</span>}</td>
                <td>{c.model.quant ?? '—'} <Prov kind="declared" /></td>
                <td className="muted">ngl {c.config.gpuLayersAll ? 'all' : c.config.gpuLayers}, kv {c.config.kvType}, t {c.config.threads}
                  {c.config.expectDegraded && <span className="pill warn-pill" title={c.config.degradedReason ?? ''}>partial offload — degraded</span>}</td>
                <td><CtxPick recommended={c.score?.recommendedCtx} scored={c.score?.referenceCtx} /></td>
                {req != null && <td>{meets(c) ? <span className="pill">met</span> : <span className="pill warn-pill" title={`practical ceiling ${c.cliff.practicalContextCeiling.value != null ? fmtCtx(c.cliff.practicalContextCeiling.value) : 'none'}`}>not met</span>}</td>}
                <td><M m={c.cliff.practicalContextCeiling} fmt={fmtCtx} /></td>
                <td>{q ? <M m={{ ...q.input, value: q.score }} fmt={(v) => { const ci = (q as { ci95?: number; n?: number }); return ci.ci95 != null ? `${v.toFixed(0)} ± ${ci.ci95.toFixed(0)}${ci.n != null ? ` (n ${ci.n})` : ''}` : v.toFixed(0) }} /> : '—'}</td>
                <td><M m={r?.decodeTps} /> {r && <span className="muted">@{fmtCtx(r.ctx)}</span>}</td>
                <td><M m={r?.prefillTps} fmt={(v) => num(v, 0)} /></td>
                <td><M m={peak(c, 'peakVramBytes')} fmt={gib} /></td>
                <td><M m={peak(c, 'peakRamBytes')} fmt={gib} /></td>
                <td>{st ? <>{st.score.toFixed(0)} <Prov kind={st.input.kind} /></> : '—'}</td>
                <td className="bar">
                  {!d.session.demo && <button onClick={() => onRerun(c.config.id)} title="Re-run every step of this configuration">Rerun</button>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <h2>Quality vs speed (Pareto){tag}</h2>
      <ParetoChart candidates={d.candidates.filter((c) => slo(c).ok)} />

      <GenTable d={d} chosen={rec?.best?.gen?.config.id ?? null} tag={tag} />
      <h2>Context scaling{tag}</h2>
      <div className="charts">
        <div><h3>Prefill t/s</h3><LineChart xs={xs} bands={bands} yLabel="prefill t/s" fmtY={(v) => num(v, 0)}
          series={d.candidates.map((c, i) => ({ label: c.config.id, color: COLORS[i % COLORS.length], ys: at(c, (r) => r.prefillTps.value) }))} /></div>
        <div><h3>Decode t/s</h3><LineChart xs={xs} bands={bands} yLabel="decode t/s" fmtY={(v) => num(v, 0)}
          series={d.candidates.map((c, i) => ({ label: c.config.id, color: COLORS[i % COLORS.length], ys: at(c, (r) => r.decodeTps.value) }))} /></div>
      </div>
      <h2>Memory scaling{tag} <span className="muted">(solid = VRAM, dashed = RAM, per process)</span></h2>
      <LineChart xs={xs} bands={bands} yLabel="GiB" fmtY={(v) => v.toFixed(1)}
        series={d.candidates.flatMap((c, i) => [
          { label: `${c.config.id} VRAM`, color: COLORS[i % COLORS.length], ys: at(c, (r) => (r.peakVramBytes.value == null ? null : r.peakVramBytes.value / 1024 ** 3)) },
          { label: `${c.config.id} RAM`, color: COLORS[i % COLORS.length], dashed: true, ys: at(c, (r) => (r.peakRamBytes.value == null ? null : r.peakRamBytes.value / 1024 ** 3)) }
        ])} />

      <h2>Cliff analysis{tag}</h2>
      {d.candidates.map((c) => (
        <div key={c.config.id} className="cliff">
          <b>{c.config.id}</b> — limited by <b>{c.cliff.limitedBy}</b>; practical <M m={c.cliff.practicalContextCeiling} fmt={fmtCtx} />, degraded <M m={c.cliff.degradedContextCeiling} fmt={fmtCtx} />
          <ul>
            {c.cliff.steps.flatMap((s) => s.reasons.map((r, k) => (
              <li key={`${s.ctx}-${k}`}><span className={`verdict ${s.verdict}`}>{s.verdict}</span> <code>{r.code}</code> {r.message}</li>
            )))}
            {c.cliff.steps.every((s) => !s.reasons.length) && <li className="muted">No cliff reasons: every step passed.</li>}
            {c.config.skippedSteps.map((s) => (
              <li key={`skip-${s.ctx}`}><span className="verdict">skipped</span> {fmtCtx(s.ctx)}: {skipText(s)}</li>
            ))}
          </ul>
          <div className="bar">{c.runs.map((r, k) => (
            <button key={r.ctx} className="mini" title="Show this step's telemetry"
              onClick={() => void window.api.telemetryForRun(c.runIds[k]).then((samples) => setTele({ key: `${c.config.id}@${r.ctx}`, samples }))}>{fmtCtx(r.ctx)} telemetry</button>
          ))}</div>
          {tele?.key.startsWith(`${c.config.id}@`) && <><p className="muted">Telemetry {tele.key}</p><TelemetryChart samples={tele.samples} /></>}
        </div>
      ))}

      <h2>Recommendation{tag}</h2>
      {rec ? (
        <div className="card">
          <p><b>{rec.best ? name(rec.best.configId) : 'No recommendation'}</b>{rec.best?.fallback && <span className="pill warn-pill">{rec.best.fallback}</span>}{rec.provisional && <span className="pill warn-pill" title="Some candidate's quality is an estimated prior (no quality run): the ranking may change once it is measured">provisional</span>}{rec.best && <> — {rec.best.score.total.toFixed(1)}/100, context <CtxPick recommended={rec.best.score.recommendedCtx} scored={rec.best.score.referenceCtx} /></>}</p>
          {!rec.best && rec.provisionalBest && (
            <div className="provisional">
              <p><span className="pill warn-pill">provisional pick — not a recommendation</span> <b>{rec.provisionalBest.headline}</b></p>
              <p className="muted"><Reason text={rec.provisionalBest.reason} />{rec.provisionalBest.estimatedTerms.length ? ` · estimated: ${rec.provisionalBest.estimatedTerms.join(', ')}` : ''}</p>
            </div>
          )}
          {!!rec.unmetAlternatives?.length && (
            <><h3>Not eligible</h3><ul>{rec.unmetAlternatives.map((u) => <li key={u.configId}>{name(u.configId)}: {u.unmet.map((x, i) => <span key={i}><Reason text={x} />{i < u.unmet.length - 1 ? '; ' : ''}</span>)}</li>)}</ul></>
          )}
          {rec.decisionTrace && <DecisionTrace trace={rec.decisionTrace} />}
          {rec.best?.gen && <p>Generation: <b>{genLabel(rec.best.gen.config)}</b> <span className="muted">— {rec.best.gen.reason}</span></p>}
          {!!rec.whyNot?.length && (
            <>
              <h3>Why not the others</h3>
              <ul>
                {rec.whyNot.map((w) => {
                  const i = d.candidates.findIndex((c) => c.config.id === w.configId)
                  return (
                    <li key={w.configId}>
                      {i >= 0
                        ? <button className="linkish" title={w.configId} onClick={() => { const el = document.getElementById(`cand-row-${i}`); el?.scrollIntoView({ behavior: 'smooth', block: 'center' }); el?.classList.add('flash'); setTimeout(() => el?.classList.remove('flash'), 1500) }}>{w.model}</button>
                        : <b>{w.model}</b>}: {w.summary}
                    </li>
                  )
                })}
              </ul>
            </>
          )}
          {reqLine && <p className={d.candidates.some(meets) ? '' : 'err'}>{reqLine}</p>}
          {(() => {
            const own = d.session.minDecodeTps
            const def = WORKLOADS[d.session.workload].minDecodeTps
            return own != null ? <p>Decode gate: {num(own)} t/s (yours)</p> : def != null ? <p className="muted">Decode gate: {num(def)} t/s (workload default)</p> : null
          })()}
          <ul>{rec.reasons.map((r) => <li key={r}><Reason text={r} /></li>)}</ul>
          <table>
            <tbody>
              <tr><td>Fastest</td><td>{name(rec.alternatives.fastest)}</td></tr>
              <tr><td>Best quality</td><td>{name(rec.alternatives.bestQuality)}</td></tr>
              <tr><td>Best long context</td><td>{name(rec.alternatives.bestLongContext)}</td></tr>
              <tr><td>Lowest memory</td><td>{name(rec.alternatives.lowestMemory)}</td></tr>
            </tbody>
          </table>
          {rec.excluded.map((e) => <p key={e.configId} className="err">Excluded {e.configId}: {e.reasons.join('; ')}</p>)}
          {!d.session.demo && <><h3>Export</h3><ExportMenu rec={rec} cand={d.candidates.find((c) => c.config.id === rec.best?.configId)} sessionId={d.session.id} ctx={exportCtx} />{exportCtx && <p className="muted">Export uses -c {exportCtx} (from an interpretation action). <button className="mini" onClick={() => setExportCtx(null)}>reset</button></p>}</>}
        </div>
      ) : <p className="muted">No recommendation stored for this session.</p>}
    </>
  )
}

type Go = (section: 'Benchmark' | 'Download', sessionId?: number, preset?: BenchPreset) => void

export function ResultsPage({ sessionId, go }: { sessionId?: number; go: Go }) {
  const [list, setList] = useState<SessionSummary[] | null>(null)
  const [sel, setSel] = useState<number | undefined>(sessionId)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  // "View as": the same measurements re-scored for another workload (computed, never saved).
  const [viewAs, setViewAs] = useState<WorkloadId | null>(null)
  const [computed, setComputed] = useState<ComputedRecommendation | null>(null)
  useEffect(() => { setViewAs(null); setComputed(null) }, [sel])
  useEffect(() => {
    setComputed(null) // never show the previous session/workload's result while the new one is pending
    if (!detail) return
    // Same workload as benchmarked: the stored recommendation is shown — unless it predates the rule engine, then it
    // is reinterpreted with the current rules for display (the stored one is kept as is).
    const own = !viewAs || viewAs === detail.session.workload
    if (own && (rulesOf(detail.recommendation) === ENGINE_RULES || ENGINE_RULES === null) && rulesOf(detail.recommendation) !== null) return
    if (own && !detail.recommendation) return
    const w = viewAs ?? detail.session.workload
    let live = true // stale-reply guard: a slower answer for an earlier session+workload is dropped (W4b F11)
    window.api.computeRecommendation(detail.session.id, w).then((r) => { if (live) setComputed(r) }, (e: Error) => { if (live) setErr(e.message) })
    return () => { live = false }
  }, [detail, viewAs])
  // Own workload, reinterpreted: the stored recommendation (ranking, reasons, scores) stays as recorded; only the
  // insights come from the current rules. Another workload: the recomputed recommendation replaces it.
  const reinterpreting = !!(detail && computed && computed.workload === detail.session.workload)
  const shown: SessionDetail | null = detail && computed && reinterpreting && detail.recommendation
    ? { ...detail, recommendation: { ...detail.recommendation, insights: insightsOf(computed.recommendation) } as typeof detail.recommendation }
    : detail && computed ? {
    ...detail,
    session: { ...detail.session, workload: computed.workload },
    recommendation: computed.recommendation,
    candidates: detail.candidates.map((c) => ({ ...c, score: computed.recommendation.ranked.find((r) => r.configId === c.config.id) ?? null }))
  } : detail
  const [err, setErr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => { window.api.listSessions().then(setList, (e: Error) => setErr(e.message)) }, [tick])
  useEffect(() => {
    if (sel == null) return setDetail(null)
    let live = true // session A's late reply must not replace session B's detail
    window.api.getSession(sel).then((d) => { if (live) setDetail(d) }, (e: Error) => { if (live) setErr(e.message) })
    return () => { live = false }
  }, [sel, tick])
  // Refresh when a session ends or a step lands while this page is open (list status + detail charts).
  useEffect(() => window.api.onBenchEvent((e) => {
    if (e.type === 'step:done' || e.type.startsWith('session:')) setTick((t) => t + 1)
  }), [])
  const act = (p: Promise<{ ok: boolean; error?: string }>) => void p.then((r) => { if (!r.ok) setErr(r.error ?? 'failed'); else setTick((t) => t + 1) })

  return (
    <section>
      <header className="bar"><h1>Results</h1></header>
      {err && <p className="err">{err}</p>}
      <table>
        <thead><tr><th>#</th><th>Created</th><th>Workload</th><th>Status</th><th>Candidates</th><th>Best</th></tr></thead>
        <tbody>
          {list?.map((s) => (
            <tr key={s.id} className={`click${s.id === sel ? ' selected' : ''}`} onClick={() => setSel(s.id)}>
              <td>{s.id}</td><td>{s.createdAt}</td><td>{s.workload}</td>
              <td>
                {s.demo ? <span className="pill demo-pill">DEMO DATA</span> : s.status}
                {s.error && <span className="err" title={s.error}> ⚠</span>}
                {!s.demo && RESUMABLE.has(s.status) && (
                  <button className="mini" onClick={(e) => { e.stopPropagation(); act(window.api.resumeBench(s.id)) }}>Resume</button>
                )}
              </td>
              <td>{s.candidateCount}</td><td className="muted">{s.bestConfigId ?? '—'}</td>
            </tr>
          ))}
          {list?.length === 0 && <tr><td colSpan={6} className="muted">No benchmark sessions yet.</td></tr>}
        </tbody>
      </table>
      {detail && !detail.session.demo && (
        <div className="bar actions">
          <button onClick={() => act(window.api.resumeBench(detail.session.id, { retryFailed: true }))} title="Re-run steps that failed or timed out">Retry failed tests</button>
        </div>
      )}
      {detail && (
        <div className="bar actions">
          <label>View as{' '}
            <select value={viewAs ?? detail.session.workload} onChange={(e) => setViewAs(e.target.value as WorkloadId)}>
              {Object.values(WORKLOADS).map((w) => <option key={w.id} value={w.id}>{w.label}{w.id === detail.session.workload ? ' (benchmarked)' : ''}</option>)}
            </select>
          </label>
          {computed && (computed.workload === detail.session.workload
            ? <span className="pill warn-pill" title="The stored recommendation was made before the interpretation rules; the numbers are the same measurements">recommended with rules {rulesOf(detail.recommendation) ?? 'pre-interp'}; reinterpreted with rules {rulesOf(computed.recommendation) ?? '?'}</span>
            : <span className="pill warn-pill">{computed.label} — not the session's own recommendation</span>)}
        </div>
      )}
      {shown && <Detail d={shown} go={go} onRerun={(configId) => act(window.api.resumeBench(shown.session.id, { rerunConfigIds: [configId] }))} />}
    </section>
  )
}
