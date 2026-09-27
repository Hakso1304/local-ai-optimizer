// Results → "How this was decided": the engine's decision trace (interp-2), rendered as recorded. Read-only.
import type { Recommendation } from '../../shared/bench-types'
import { Reason } from './InterpretPanel'
import { fmtCtx } from './ui'

type Trace = NonNullable<Recommendation['decisionTrace']>
const v = (x: unknown) => (x == null ? '—' : typeof x === 'number' ? String(Math.round(x * 100) / 100) : String(x))

export function DecisionTrace({ trace }: { trace: Trace }) {
  const t = trace
  return (
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
        <thead><tr><th>Config</th><th>Eligible</th><th>Confirmed</th><th>Total</th><th>Quality part</th><th>Gen</th><th>Scored at</th><th>Undecided / failures</th></tr></thead>
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
                <td>{c.referenceCtx != null ? fmtCtx(c.referenceCtx) : '—'}</td>
                <td className="muted">{[...c.undecided, ...c.failures].join('; ') || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {t.steps.length > 0 && <><h3>Decision steps</h3><ol>{t.steps.map((s, i) => <li key={i}><Reason text={`[${s.ruleId}] ${s.kind}: ${s.winner}${s.over ? ` over ${s.over}` : ''} — ${s.detail}`} /></li>)}</ol></>}
      {t.neutralizations.length > 0 && (
        <><h3>Quality differences treated as ties</h3>
          <ul>{t.neutralizations.map((n, i) => (
            <li key={i}>{n.a} vs {n.b}: {n.difference ? `Δ ${v(n.difference.diff)} [${v(n.difference.lower)}, ${v(n.difference.upper)}] over ${n.difference.sharedItems} shared items` : n.reason ?? 'not comparable'} → decided without quality: <b>{n.winner}</b> <span className="muted">({Object.entries(n.totalsWithoutQuality).map(([k, x]) => `${k} ${v(x)}`).join(', ')})</span></li>
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
  )
}
