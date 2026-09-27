import { useEffect, useMemo, useState } from 'react'
import type { WorkloadId, WorkloadProfile } from '../../shared/bench-types'
import type { SessionCandidate } from '../../shared/types'
import { WORKLOADS } from '../../core/scoring/workloads'
import { fmtCtx } from './ui'

/** A constraint left null is off. */
export interface SloValues { maxTtftS: number | null; minDecodeTps: number | null; minPracticalCtx: number | null; maxVramGiB: number | null }
/** Measured facts per candidate; null = not measured. */
export interface SloFacts { ttftMs: number | null; decodeTps: number | null; practicalCtx: number | null; peakVramBytes: number | null; ctx: number | null }

/** Defaults from the workload: its latency tolerance, its decode gate, and the eligibility floor (half the target ctx). */
export function sloDefaults(p: WorkloadProfile): SloValues {
  return { maxTtftS: p.latencyToleranceMs / 1000, minDecodeTps: p.minDecodeTps ?? null, minPracticalCtx: p.targetContext / 2, maxVramGiB: null }
}

const ok = (c: SessionCandidate['runs'][number]) => c.status === 'pass' || c.status === 'degraded'

/** The step the recommendation speaks for: recommendedCtx, else the scored (reference) ctx. */
export function factsOf(c: SessionCandidate): SloFacts {
  const ctx = c.score?.recommendedCtx ?? c.score?.referenceCtx ?? null
  const r = c.runs.find((x) => x.ctx === ctx && ok(x))
  const vram = c.runs.filter(ok).map((x) => x.peakVramBytes.value).filter((v): v is number => v != null)
  return {
    ctx: r ? ctx : null,
    ttftMs: r?.ttftMs.value ?? null,
    decodeTps: r?.decodeTps.value ?? null,
    practicalCtx: c.cliff.practicalContextCeiling.value,
    peakVramBytes: vram.length ? Math.max(...vram) : null
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
  chk(s.minDecodeTps, f.decodeTps, (v) => v >= s.minDecodeTps!, 'decode')
  chk(s.minPracticalCtx, f.practicalCtx, (v) => v >= s.minPracticalCtx!, 'practical ctx')
  chk(s.maxVramGiB, f.peakVramBytes, (v) => v <= s.maxVramGiB! * 1024 ** 3, 'peak VRAM')
  return { ok: failed.length === 0, failed }
}

const CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

export function SloFilter({ workload, candidates, onChange }: {
  workload: WorkloadId
  candidates: SessionCandidate[]
  onChange: (pred: (c: SessionCandidate) => boolean) => void
}) {
  const [s, setS] = useState<SloValues>(() => sloDefaults(WORKLOADS[workload]))
  useEffect(() => setS(sloDefaults(WORKLOADS[workload])), [workload])
  const pred = useMemo(() => (c: SessionCandidate) => meetsSlo(factsOf(c), s).ok, [s])
  useEffect(() => onChange(pred), [pred, onChange])
  const n = candidates.filter(pred).length
  const numIn = (k: keyof SloValues, label: string, step: string) => (
    <label>{label}{' '}
      <input type="number" min="0" step={step} value={s[k] ?? ''} placeholder="off" style={{ width: '6em' }}
        onChange={(e) => setS({ ...s, [k]: e.target.value === '' ? null : Number(e.target.value) })} />
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
      <button className="mini" onClick={() => setS(sloDefaults(WORKLOADS[workload]))}>Workload defaults</button>
      <span className="muted">{n} of {candidates.length} configurations meet the constraints</span>
    </div>
  )
}
