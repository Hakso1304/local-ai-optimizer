// Results → "How this was decided": the engine's decision trace (interp-2), rendered as recorded. Read-only.
import type { Recommendation } from '../../shared/bench-types'
import { Reason, RulesVersion } from './InterpretPanel'
import { fmtCtx } from './ui'

type Trace = NonNullable<Recommendation['decisionTrace']>
const v = (x: unknown) => (x == null ? '—' : typeof x === 'number' ? String(Math.round(x * 100) / 100) : String(x))
type Diff = { diff: number; lower: number; upper: number; sharedItems: number } | null
const dq = (d: Diff, reason?: string) => (d ? `Δ ${v(d.diff)} [${v(d.lower)}, ${v(d.upper)}] over ${d.sharedItems} shared items` : reason ?? 'not comparable')

export function DecisionTrace({ trace }: { trace: Trace }) {
  const t = trace
  return (
    <RulesVersion.Provider value={t.rulesVersion}>
    <details className="trace">
      <summary><b>How this was decided</b> <span className="muted">(rules {t.rulesVersion}, scoring {t.scoringVersion})</span></summary>
      <p>Scoring rung: <b>{t.scoringRung != null ? fmtCtx(t.scoringRung) : '—'}</b> <span className="muted">— {t.scoringRungWhy}</span></p>
      <p className="muted">
        Hard constraints: required context {t.hardConstraints.requiredContext != null ? fmtCtx(t.hardConstraints.requiredContext) : 'none'};
        {' '}min decode {t.hardConstraints.minDecodeTps ? `${t.hardConstraints.minDecodeTps.value} t/s (${t.hardConstraints.minDecodeTps.source})` : 'none'};
        {' '}latency tolerance {(t.hardConstraints.latencyToleranceMs / 1000).toFixed(0)} s{t.hardConstraints.latencyAdvisory ? ' (advisory)' : ''}
      </p>
      <h3>Candidates</h3>
      <table>
        <thead><tr><th>Config</th><th>Eligible</th><th>Confirmed</th><th>Total</th><th>Quality part</th><th>Gen</th><th>Scored at</th><th>Recommended -c</th><th>Basis</th><th>Safety</th><th>Quality vs winner</th><th>Undecided / failures</th></tr></thead>
        <tbody>
          {t.candidates.map((c) => {
            const e = t.eligibleSet.find((x) => x.configId === c.configId)
            return (
              <tr key={c.configId} className={c.configId === t.winner ? 'selected' : ''}>
                <td>{c.configId}{c.configId === t.winner && <span className="pill">winner</span>}{c.configId === t.provisionalWinner && <span className="pill warn-pill">provisional</span>}</td>
                <td>{e ? (e.failingRuleIds.length ? <span className="err">{e.failingRuleIds.join(', ')}</span> : 'yes') : '—'}</td>
                <td>{c.confirmed ? 'yes' : 'no'}</td>
                <td>{v(c.total)}</td>
                <td>{v(c.qualityContribution)}</td>
                <td className="muted">{c.gen ?? '—'}</td>
                <td title={c.referenceWhy ?? ''}>{c.referenceCtx != null ? fmtCtx(c.referenceCtx) : '—'}</td>
                <td>{c.recommendedCtx != null ? fmtCtx(c.recommendedCtx) : '—'}{c.recommendedWhy && <div className="muted">{c.recommendedWhy}</div>}</td>
                <td className="muted">{(c.basis ?? []).map((b) => `${b.component}${b.rung != null ? `@${fmtCtx(b.rung)}` : ''} (${b.kind})`).join(', ') || '—'}</td>
                <td>{c.safety ? <>RAM floor <span className={c.safety.ramFloor === 'violated' ? 'err' : c.safety.ramFloor === 'ok' ? '' : 'muted'}>{c.safety.ramFloor}</span>; spill <span className={c.safety.spill === 'measured' ? '' : 'muted'}>{c.safety.spill}</span></> : '—'}</td>
                <td className="muted">{c.configId === t.winner ? '—' : c.qualityVsWinner ? dq(c.qualityVsWinner.difference, c.qualityVsWinner.reason) : '—'}</td>
                <td className="muted">{[...c.undecided, ...c.failures].join('; ') || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {!!t.cycle?.length && <p className="err">The pairwise comparisons revisited a leader ({t.cycle.join(' → ')}): the order is not a proven total order.</p>}
      {!!t.comparisons?.length && (
        <><h3>Comparisons</h3>
          <table>
            <thead><tr><th>A</th><th>B</th><th>Compared on</th><th>A value</th><th>B value</th><th>Quality difference</th><th>Winner</th><th>Rule</th></tr></thead>
            <tbody>{t.comparisons.map((c, i) => (
              <tr key={i}><td>{c.a}</td><td>{c.b}</td><td>{c.basis}</td><td>{v(c.aValue)}</td><td>{v(c.bValue)}</td><td className="muted">{c.basis === 'total' && !c.difference && !c.reason ? '—' : dq(c.difference, c.reason)}</td><td><b>{c.winner}</b></td><td><span className="rule-tag">{c.ruleId}</span></td></tr>
            ))}</tbody>
          </table></>
      )}
      {t.qualityVsSpeed && <p>Quality vs speed: winner {t.qualityVsSpeed.winner} ({v(t.qualityVsSpeed.decodeWinner)} t/s) vs fastest {t.qualityVsSpeed.fastest} ({v(t.qualityVsSpeed.decodeFastest)} t/s) — {dq(t.qualityVsSpeed.difference, t.qualityVsSpeed.reason)}</p>}
      {!!t.excluded?.length && <><h3>Excluded before ranking</h3><ul>{t.excluded.map((e) => <li key={e.configId}>{e.configId}: {e.reasons.map((x, i) => <span key={i}><Reason text={x} />{i < e.reasons.length - 1 ? '; ' : ''}</span>)}</li>)}</ul></>}
      {t.steps.length > 0 && <><h3>Decision steps</h3><ol>{t.steps.map((s, i) => <li key={i}><Reason text={`[${s.ruleId}] ${s.kind}: ${s.winner}${s.over ? ` over ${s.over}` : ''} — ${s.detail}`} /></li>)}</ol></>}
      {t.neutralizations.length > 0 && (
        <><h3>Quality differences treated as ties</h3>
          <ul>{t.neutralizations.map((n, i) => (
            <li key={i}>{n.a} vs {n.b}: {dq(n.difference, n.reason)} → decided without quality: <b>{n.winner}</b> <span className="muted">({Object.entries(n.totalsWithoutQuality).map(([k, x]) => `${k} ${v(x)}`).join(', ')})</span></li>
          ))}</ul></>
      )}
      {t.tieBreakChain.length > 0 && (
        <><h3>Tie-break chain</h3>
          <ol>{t.tieBreakChain.map((s, i) => <li key={i} className={s.decided ? '' : 'muted'}>{s.step}: {s.a} {v(s.a_value)} vs {s.b} {v(s.b_value)}{s.decided ? ' — decided' : ''}</li>)}</ol></>
      )}
      <h3>Alternatives</h3>
      <ul>{Object.entries(t.alternatives).map(([k, a]) => <li key={k}>{k}: {a.configId ?? '—'} <span className="rule-tag" title={`rule ${a.ruleId}`}>{a.ruleId}</span></li>)}</ul>
      {t.unmetAlternatives.length > 0 && <><h3>Not eligible</h3><ul>{t.unmetAlternatives.map((u) => <li key={u.configId}>{u.configId}: {u.unmet.map((x, i) => <span key={i}><Reason text={x} />{i < u.unmet.length - 1 ? '; ' : ''}</span>)}</li>)}</ul></>}
      {t.genChoices.length > 0 && <><h3>Generation choices</h3><ul>{t.genChoices.map((g) => <li key={g.configId}>{g.configId}: {g.chosen ?? 'baseline'}{g.steps.length ? <span className="muted"> — {g.steps.join('; ')}</span> : null}</li>)}</ul></>}
      <h3>Thresholds used</h3>
      <p className="muted">{Object.entries(t.thresholdsUsed).map(([k, x]) => `${k} ${v(x)}`).join(' · ')}</p>
    </details>
    </RulesVersion.Provider>
  )
}
