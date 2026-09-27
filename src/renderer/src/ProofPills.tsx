// Per generation config: how far the stored quality rows prove that the requested setting was applied (row-bound
// template render proof, requested vs runtime-accepted sampling). Read-only, from the rows as stored. Old rows without
// these fields say so ("pre-P1"); nothing is inferred. The comparability verdict itself is read from the engine's
// I-8.0 insight, never recomputed here.
import type { GenRow } from '../../core/benchmark/gen'
import type { Insight } from '../../shared/interpret-types'

/** The provenance fields of a stored quality row (committed GenRow types; all optional — absent on older rows). */
type ProofRow = Pick<GenRow, 'renderProof' | 'proofProvenance' | 'requestedSampling' | 'acceptedSampling'>

export interface ProofSummary {
  rows: number
  /** rows carrying none of the proof fields */
  preP1: number
  proved: number; unproved: number; contradicted: number
  /** proof rebuilt afterwards (no original prompt hash): not bound to the row that was graded */
  reconstructed: number
  /** requested seed not confirmed by the runtime-accepted seed */
  seedUnverified: number
  /** a requested sampler differs from the accepted value (or the runtime reported none for it) */
  samplingMismatch: number
}

const SAMPLERS: [keyof NonNullable<ProofRow['requestedSampling']>, string][] = [['temperature', 'temperature'], ['topP', 'top_p'], ['topK', 'top_k'], ['minP', 'min_p']]
const same = (a: unknown, b: unknown) => typeof a === 'number' && typeof b === 'number' && Math.abs(a - b) < 1e-6

export function proofSummary(rows: unknown[]): ProofSummary {
  const s: ProofSummary = { rows: rows.length, preP1: 0, proved: 0, unproved: 0, contradicted: 0, reconstructed: 0, seedUnverified: 0, samplingMismatch: 0 }
  for (const x of rows as ProofRow[]) {
    if (!x.renderProof && !x.proofProvenance && !x.requestedSampling) { s.preP1++; continue }
    const st = x.renderProof?.status
    const rebuilt = st === 'reconstructed' || x.proofProvenance?.status === 'reconstructed' || x.proofProvenance?.originalPromptHashPresent === false
    if (rebuilt) s.reconstructed++
    else if (st === 'proved') s.proved++
    else if (st === 'contradicted') s.contradicted++
    else s.unproved++
    const req = x.requestedSampling, acc = x.acceptedSampling ?? null
    if (req) {
      if (req.seed == null || !acc || !same(acc.seed, req.seed)) s.seedUnverified++
      if (SAMPLERS.some(([k, a]) => req[k] != null && !same(acc?.[a], req[k]))) s.samplingMismatch++
    }
  }
  return s
}

/** The engine's I-8.0 "not comparable" reason for this config's generation setting, if it gave one. */
export function notEvaluableReason(insights: Insight[], configId: string | undefined, genLabelText: string): string | null {
  const i = insights.find((x) => x.key === 'gen.comparable' && x.evaluable === false && (!configId || x.configId === configId) && x.text.includes(`: ${genLabelText} not comparable — `))
  return i ? (/not comparable — (.*?); it is not considered\.?$/.exec(i.text)?.[1] ?? i.text) : null
}

export function ProofPills({ rows }: { rows: unknown[] }) {
  const s = proofSummary(rows)
  if (!s.rows) return null
  if (s.preP1 === s.rows) return <span className="pill" title="rows written before per-row render proof existed">no proof recorded (pre-P1)</span>
  const n = s.rows - s.preP1
  return (
    <>
      <span className="pill" title="row-bound render proof: the requested kwargs re-render this row's exact prompt">proved {s.proved}/{n} rows</span>
      {s.reconstructed > 0 && <span className="pill warn-pill" title="proof rebuilt afterwards without the original prompt hash">reconstructed (not row-bound) {s.reconstructed}</span>}
      {s.unproved > 0 && <span className="pill warn-pill">unproved {s.unproved}</span>}
      {s.contradicted > 0 && <span className="pill err-pill">contradicted {s.contradicted}</span>}
      {s.seedUnverified > 0 && <span className="pill warn-pill" title="the runtime did not report the requested seed back">seed unverified {s.seedUnverified}</span>}
      {s.samplingMismatch > 0 && <span className="pill warn-pill" title="accepted sampling differs from the requested values">sampling mismatch {s.samplingMismatch}</span>}
      {s.preP1 > 0 && <span className="pill muted">pre-P1 {s.preP1}</span>}
    </>
  )
}
