import { useEffect, useMemo, useState } from 'react'
import type { ProvenanceKind, WorkloadProfile } from '../../shared/bench-types'
import type { SessionCandidate, SessionSummary } from '../../shared/types'
import { WORKLOADS, effectiveProfile } from '../../core/scoring/workloads'
import { fmtCtx } from './ui'

/** A constraint left null is off. */
export interface SloValues { maxTtftS: number | null; minDecodeTps: number | null; minPracticalCtx: number | null; maxVramGiB: number | null }
/** Measured facts per candidate at its scored step; null = not measured. */
export interface SloFacts {
  ttftMs: number | null; decodeTps: number | null; practicalCtx: number | null; peakVramBytes: number | null; ctx: number | null
  /** Provenance of decodeTps: 'estimated' = wall-clock derived (no runtime timings). */
  decodeKind?: ProvenanceKind
}
export type SloCheck = (c: SessionCandidate) => { ok: boolean; failed: string[] }

/** Defaults from the session's EFFECTIVE profile (see sessionProfile): its latency tolerance (none when latency is
 *  advisory, e.g. a requiredContext session), its decode gate, and the context floor (required ctx, else half the target). */
export function sloDefaults(p: WorkloadProfile, requiredContext?: number | null): SloValues {
  return {
    maxTtftS: p.latencyAdvisory ? null : p.latencyToleranceMs / 1000,
    minDecodeTps: p.minDecodeTps ?? null,
    minPracticalCtx: requiredContext ?? p.requiredContext ?? p.targetContext / 2,
    maxVramGiB: null
  }
}

/** The profile the session was scored with: workload + its request's requiredContext / minDecodeTps. */
export const sessionProfile = (s: Pick<SessionSummary, 'workload' | 'requiredContext' | 'minDecodeTps'>): WorkloadProfile =>
  effectiveProfile(WORKLOADS[s.workload], { requiredContext: s.requiredContext, minDecodeTps: s.minDecodeTps })

const ok = (c: SessionCandidate['runs'][number]) => c.status === 'pass' || c.status === 'degraded'

/** Facts at the scoring step (score.referenceCtx): every config is compared at the rung its score and the eligibility
 *  gate used, so the default minDecodeTps means the same thing here as in the recommendation. */
export function factsOf(c: SessionCandidate): SloFacts {
  const ctx = c.score?.referenceCtx ?? null
  const r = c.runs.find((x) => x.ctx === ctx && ok(x))
  return {
    ctx: r ? ctx : null,
    ttftMs: r?.ttftMs?.value ?? null,
    decodeTps: r?.decodeTps?.value ?? null,
    decodeKind: r?.decodeTps?.kind,
    practicalCtx: c.cliff.practicalContextCeiling.value,
    peakVramBytes: r?.peakVramBytes?.value ?? null
  }
}

/** An active constraint on an unmeasured value fails: "unknown" never counts as meeting an SLO. */
export function meetsSlo(f: SloFacts, s: SloValues): { ok: boolean; failed: string[] } {
  const failed: string[] = []
  const chk = (on: number | null, v: number | null, pass: (v: number) => boolean, what: string) => {
    if (on == null) return
    if (v == null) failed.push(`${what} not measured`)
    else if (!pass(v)) failed.push(what)
  }
  chk(s.maxTtftS, f.ttftMs, (v) => v <= s.maxTtftS! * 1000, 'TTFT')
  chk(s.minDecodeTps, f.decodeTps, (v) => v >= s.minDecodeTps!, f.decodeKind === 'estimated' ? 'decode (ESTIMATED)' : 'decode')
  chk(s.minPracticalCtx, f.practicalCtx, (v) => v >= s.minPracticalCtx!, 'practical ctx')
  chk(s.maxVramGiB, f.peakVramBytes, (v) => v <= s.maxVramGiB! * 1024 ** 3, 'peak VRAM')
  return { ok: failed.length === 0, failed }
}

/** Input text → constraint: blank, non-numeric or negative = off. */
export const parseLimit = (text: string): number | null => {
  const v = Number(text)
  return text.trim() === '' || !Number.isFinite(v) || v < 0 ? null : v
}

const CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

export function SloFilter({ session, candidates, onChange }: {
  /** Defaults follow this session's effective profile (requiredContext, custom minDecodeTps, advisory latency). */
  session: Pick<SessionSummary, 'workload' | 'requiredContext' | 'minDecodeTps'>
  candidates: SessionCandidate[]
  onChange: (check: SloCheck) => void
}) {
  const { workload, requiredContext, minDecodeTps } = session
  const defaults = useMemo(() => sloDefaults(sessionProfile({ workload, requiredContext, minDecodeTps }), requiredContext), [workload, requiredContext, minDecodeTps])
  const [s, setS] = useState<SloValues>(defaults)
  useEffect(() => setS(defaults), [defaults])
  const check = useMemo<SloCheck>(() => (c) => meetsSlo(factsOf(c), s), [s])
  useEffect(() => onChange(check), [check, onChange])
  const n = candidates.filter((c) => check(c).ok).length
  const numIn = (k: keyof SloValues, label: string, step: string) => (
    <label>{label}{' '}
      <input type="number" min="0" step={step} value={s[k] ?? ''} placeholder="off" style={{ width: '6em' }}
        onChange={(e) => setS({ ...s, [k]: parseLimit(e.target.value) })} />
    </label>
  )
  return (
    <div className="bar slo">
      {numIn('maxTtftS', 'Max TTFT s', '0.5')}
      {numIn('minDecodeTps', 'Min decode t/s', '1')}
      <label>Min practical ctx{' '}
        <select value={s.minPracticalCtx ?? 0} onChange={(e) => setS({ ...s, minPracticalCtx: Number(e.target.value) || null })}>
          <option value={0}>off</option>
          {CTX.map((c) => <option key={c} value={c}>{fmtCtx(c)}</option>)}
        </select>
      </label>
      {numIn('maxVramGiB', 'Max peak VRAM GiB', '0.5')}
      <button className="mini" onClick={() => setS(defaults)}>Workload defaults</button>
      <span className="muted">{n} of {candidates.length} configurations meet the constraints (measured at each config's scored context)</span>
    </div>
  )
}
