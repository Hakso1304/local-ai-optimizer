import { useEffect, useState } from 'react'
import type { WorkloadId } from '../../shared/bench-types'
import type { ComputedRecommendation, SessionDetail } from '../../shared/types'
import { COMPONENT_LABEL, CtxPick, ScoreBar, num } from './ui'

export const LARGE: WorkloadId = 'large_coding'
export interface BenchPreset { workload: WorkloadId; heavyMode: boolean; requiredContext: number | null }
export const LARGE_PRESET: BenchPreset = { workload: LARGE, heavyMode: true, requiredContext: 65536 }

const decodeOf = (d: SessionDetail, configId: string | undefined, ctx: number | null | undefined) =>
  d.candidates.find((c) => c.config.id === configId)?.runs.find((r) => r.ctx === ctx)?.decodeTps.value ?? null

/** Second Dashboard card: the large-scale coding pick, re-scored from the latest real session that reached ≥32K
 *  (heavy-mode sessions first). fastDecode = decode t/s of the selected workload's pick, for the "N× slower" line. */
export function LargeCodingCard({ fastDecode, onBenchmark, onDetails }: { fastDecode: number | null; onBenchmark: () => void; onDetails: (id: number) => void }) {
  const [state, setState] = useState<{ d: SessionDetail; r: ComputedRecommendation } | null | undefined>(undefined)
  useEffect(() => {
    void (async () => {
      const list = (await window.api.listSessions()).filter((s) => !s.demo)
      for (const s of [...list.filter((x) => x.heavyMode), ...list.filter((x) => !x.heavyMode)]) {
        const d = await window.api.getSession(s.id)
        if (!d?.candidates.some((c) => (c.cliff.practicalContextCeiling.value ?? 0) >= 32768)) continue
        const r = await window.api.computeRecommendation(s.id, LARGE)
        if (r) return setState({ d, r })
      }
      setState(null)
    })()
  }, [])

  const best = state?.r.recommendation.best
  const c = state && best ? state.d.candidates.find((x) => x.config.id === best.configId) : undefined
  const dec = state && best ? decodeOf(state.d, best.configId, best.score.referenceCtx) : null
  return (
    <div className="card">
      <h2>Large-scale coding <span className="muted">(bigger model, slower)</span></h2>
      {state === undefined ? <p className="muted">Looking for a long-context session…</p>
        : !state ? <p>No benchmark session reached 32K context yet.</p>
        : !best || !c ? (
          <><p>No configuration in session #{state.r.sessionId} meets the large-scale coding requirements.</p>
            <ul>{state.r.recommendation.reasons.slice(0, 4).map((x) => <li key={x}>{x}</li>)}</ul></>
        ) : (
          <>
            <p className="muted">{state.r.label}</p>
            <table className="kv">
              <tbody>
                <tr><td>Model</td><td>{c.model.name}{best.headline && <div className="muted">{best.headline}</div>}</td></tr>
                <tr><td>Quant</td><td>{c.model.quant ?? '—'}</td></tr>
                <tr><td>Context</td><td><CtxPick recommended={best.score.recommendedCtx} scored={best.score.referenceCtx} /></td></tr>
                <tr><td>Decode</td><td>{dec != null ? `${num(dec)} t/s` : '—'}
                  {dec != null && fastDecode != null && dec > 0 && fastDecode > dec && <span className="muted"> — {num(fastDecode / dec)}× slower than the fast pick</span>}</td></tr>
                <tr><td>Score</td><td>{best.score.total.toFixed(1)} / 100{best.fallback && <span className="pill warn-pill">{best.fallback}</span>}</td></tr>
              </tbody>
            </table>
            {best.score.breakdown.map((b) => <ScoreBar key={b.component} label={COMPONENT_LABEL[b.component]} score={b.score} kind={b.input.kind} weight={b.weight} />)}
            {!!state.r.recommendation.whyNot?.length && (
              <><h3>Why not the others</h3><ul>{state.r.recommendation.whyNot.map((w) => <li key={w.configId}><b>{w.model}</b>: {w.summary}</li>)}</ul></>
            )}
          </>
        )}
      <div className="bar actions">
        <button onClick={onBenchmark}>Benchmark for large-scale coding</button>
        {state?.r && <button onClick={() => onDetails(state.r.sessionId)}>View Details</button>}
      </div>
    </div>
  )
}
