// Per-run telemetry over time (ACCEPTANCE A14), same style as LineChart: hand-rolled SVG, null = gap (never
// interpolated). Left axis 0–100 % (GPU, CPU), right axis GiB (per-PID VRAM dedicated/shared, private WS).
// ponytail: no warmup/measure shading — samples carry only ts, RunDetail only startedAt/endedAt; add it when the
// runner stores request windows.
import type { TelemetrySample } from '../../shared/bench-events'

export interface Pt { t: number; v: number | null }

/** Min/max-preserving downsample to ≤ maxPoints: each bucket keeps its min and max point in time order, and a
 *  bucket containing a null keeps one null so gaps survive. */
export function downsample(pts: Pt[], maxPoints: number): Pt[] {
  if (pts.length <= maxPoints || maxPoints < 2) return pts
  // Up to 3 points per bucket: min, max (of the measured values) and one gap marker if any sample was unavailable.
  // ponytail: maxPoints < 3 still yields one full bucket (≤ 3 points); nothing charts at that size.
  const buckets = Math.max(1, Math.floor(maxPoints / 3))
  const size = pts.length / buckets
  const out: Pt[] = []
  for (let b = 0; b < buckets; b++) {
    const chunk = pts.slice(Math.floor(b * size), Math.floor((b + 1) * size))
    const vals = chunk.filter((p) => p.v != null)
    const keep: Pt[] = []
    if (vals.length) {
      const lo = vals.reduce((a, p) => (p.v! < a.v! ? p : a)), hi = vals.reduce((a, p) => (p.v! > a.v! ? p : a))
      keep.push(...(lo === hi ? [lo] : [lo, hi]))
    }
    const gap = chunk.find((p) => p.v == null)
    if (gap) keep.push({ t: gap.t, v: null })
    out.push(...keep.sort((x, y) => x.t - y.t))
  }
  return out
}

const W = 620, H = 240, L = 44, R = 52, T = 12, B = 36
const GiB = 1024 ** 3
const SERIES: { key: keyof TelemetrySample; label: string; color: string; axis: 'pct' | 'gib'; dashed?: boolean }[] = [
  { key: 'gpuUtilPct', label: 'GPU %', color: '#58a6ff', axis: 'pct' },
  { key: 'cpuPct', label: 'CPU %', color: '#f0883e', axis: 'pct' },
  { key: 'procVramDedicatedBytes', label: 'VRAM dedicated (GiB)', color: '#3fb950', axis: 'gib' },
  { key: 'procVramSharedBytes', label: 'VRAM shared (GiB)', color: '#f85149', axis: 'gib', dashed: true },
  { key: 'procRamPrivateBytes', label: 'RAM private (GiB)', color: '#d2a8ff', axis: 'gib', dashed: true }
]

export function TelemetryChart({ samples, maxPoints = 300 }: { samples: TelemetrySample[]; maxPoints?: number }) {
  if (!samples.length) return <p className="muted">No telemetry samples for this run (too short for typeperf's 1 s rows, or the sampler failed).</p>
  const t0 = samples[0].ts
  const tMax = Math.max(1, (samples[samples.length - 1].ts - t0) / 1000)
  const lines = SERIES.map((s) => ({
    ...s,
    pts: downsample(samples.map((x) => {
      const v = x[s.key] as number | null | undefined
      return { t: (x.ts - t0) / 1000, v: v == null ? null : s.axis === 'gib' ? v / GiB : v }
    }), maxPoints)
  })).filter((s) => s.pts.some((p) => p.v != null))
  const gibMax = Math.max(1, ...lines.filter((s) => s.axis === 'gib').flatMap((s) => s.pts.map((p) => p.v ?? 0))) * 1.1
  const x = (t: number) => L + ((W - L - R) * t) / tMax
  const y = (v: number, axis: 'pct' | 'gib') => T + (H - T - B) * (1 - v / (axis === 'pct' ? 100 : gibMax))
  const path = (pts: Pt[], axis: 'pct' | 'gib') => {
    const out: string[] = []
    let cur = ''
    for (const p of pts) {
      if (p.v == null) { if (cur) out.push(cur); cur = ''; continue }
      cur += `${cur ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.v, axis).toFixed(1)}`
    }
    if (cur) out.push(cur)
    return out
  }
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="telemetry over time">
        {ticks.map((f) => (
          <g key={f}>
            <line x1={L} x2={W - R} y1={y(f * 100, 'pct')} y2={y(f * 100, 'pct')} className="grid" />
            <text x={L - 6} y={y(f * 100, 'pct') + 4} textAnchor="end" className="tick">{Math.round(f * 100)}</text>
            <text x={W - R + 6} y={y(f * 100, 'pct') + 4} className="tick">{(f * gibMax).toFixed(1)}</text>
          </g>
        ))}
        {ticks.map((f) => <text key={`x${f}`} x={x(f * tMax)} y={H - B + 16} textAnchor="middle" className="tick">{(f * tMax).toFixed(0)}s</text>)}
        <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="axis" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>%</text>
        <text x={W - 8} y={(T + H - B) / 2} textAnchor="middle" className="axis" transform={`rotate(90 ${W - 8} ${(T + H - B) / 2})`}>GiB</text>
        {lines.map((s) => path(s.pts, s.axis).map((d) => (
          <path key={`${s.key}${d.slice(0, 24)}`} d={d} fill="none" stroke={s.color} strokeWidth={1.5} strokeDasharray={s.dashed ? '5 4' : undefined} />
        )))}
      </svg>
      <div className="bar muted" style={{ flexWrap: 'wrap', fontSize: 12 }}>
        {lines.map((s) => <span key={s.key}><span className="swatch" style={{ background: s.color }} /> {s.label}</span>)}
        <span>{samples.length} samples · gaps = no reading</span>
      </div>
    </div>
  )
}
