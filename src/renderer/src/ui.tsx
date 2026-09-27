// Small shared view helpers. No logic beyond formatting.
import type { Metric, ProvenanceKind } from '../../shared/bench-types'

export const fmtCtx = (n: number) => (n % 1024 === 0 ? `${n / 1024}K` : String(n))
export const gib = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
export const num = (v: number, d = 1) => v.toLocaleString(undefined, { maximumFractionDigits: d })

export function Prov({ kind, title }: { kind: ProvenanceKind; title?: string }) {
  return <span className={`prov ${kind}`} title={title}>{kind.toUpperCase()}</span>
}

/** Value + provenance badge; unavailable shows "—" with the reason as tooltip, never 0. */
export function M({ m, fmt = (v: number) => num(v) }: { m: Metric | null | undefined; fmt?: (v: number) => string }) {
  if (!m) return <span className="muted">—</span>
  const tip = m.kind === 'unavailable' ? m.reason : m.source
  return (
    <span className="metric">
      {m.value == null ? <span className="muted" title={tip}>—</span> : fmt(m.value)} <Prov kind={m.kind} title={tip} />
    </span>
  )
}

export function DemoBanner({ what = 'DEMO DATA' }: { what?: string }) {
  return <div className="demo">{what}: generated from test fixtures, not measured on this machine</div>
}

export function ScoreBar({ label, score, kind, weight }: { label: string; score: number; kind: ProvenanceKind; weight?: number }) {
  return (
    <div className="scorebar">
      <span className="lbl">{label}</span>
      <span className="track"><span className="fill" style={{ width: `${Math.max(0, Math.min(100, score))}%` }} /></span>
      <span className="val">{score.toFixed(0)}</span>
      <Prov kind={kind} />
      {weight != null && <span className="muted">w {(weight * 100).toFixed(0)}%</span>}
    </div>
  )
}

export const COMPONENT_LABEL: Record<string, string> = {
  quality: 'Quality', genSpeed: 'Speed', prefillSpeed: 'Prefill', latency: 'Latency', memory: 'Memory', stability: 'Stability', context: 'Context'
}

/** "32K (scored at 16K)" — recommendedCtx is what to configure; referenceCtx is where speed was scored. */
export function CtxPick({ recommended, scored }: { recommended: number | null | undefined; scored: number | null | undefined }) {
  if (recommended == null) return <span className="muted" title="no passing step fits the workload latency tolerance">—</span>
  return <span>{fmtCtx(recommended)} <Prov kind="measured" />{scored != null && scored !== recommended && <span className="muted"> (scored at {fmtCtx(scored)})</span>}</span>
}

/** Backend a config runs on, for labels: device null = CPU; absent backend = Vulkan (pre-HIP sessions). */
export function backendLabel(c: { backend?: string; device: string | null }): string {
  if (!c.device) return 'llama.cpp CPU'
  const kind = c.backend === 'hip' ? 'ROCm (HIP)' : c.backend === 'cuda' ? 'CUDA' : 'Vulkan'
  return `llama.cpp ${kind} (${c.device})`
}
