// Hand-rolled SVG line chart: x = ctx steps (evenly spaced, the ladder doubles), y linear from 0.
// null points break the line (unavailable is never drawn as 0). `bands` shade cliff steps.
import { fmtCtx } from './ui'

export interface Series { label: string; color: string; ys: (number | null)[]; dashed?: boolean }
export interface Band { from: number; to: number; color: string; label: string }

const W = 520, H = 220, L = 56, R = 12, T = 12, B = 40

export function LineChart({ xs, series, bands = [], yLabel, fmtY = String }: {
  xs: number[]; series: Series[]; bands?: Band[]; yLabel: string; fmtY?: (v: number) => string
}) {
  const vals = series.flatMap((s) => s.ys).filter((v): v is number => v != null)
  if (!xs.length || !vals.length) return <p className="muted">No measured points to plot.</p>
  const yMax = Math.max(...vals) * 1.1 || 1
  const x = (i: number) => L + (xs.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (xs.length - 1))
  const y = (v: number) => T + (H - T - B) * (1 - v / yMax)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax)
  const segs = (ys: (number | null)[]) => {
    const out: string[] = []
    let cur = ''
    ys.forEach((v, i) => {
      if (v == null) { if (cur) out.push(cur); cur = ''; return }
      cur += `${cur ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`
    })
    if (cur) out.push(cur)
    return out
  }
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={yLabel}>
      {bands.map((b) => {
        const i0 = xs.indexOf(b.from), i1 = xs.indexOf(b.to)
        if (i1 < 0) return null
        const x0 = i0 >= 0 ? x(i0) : x(i1) - 10
        return (
          <g key={`${b.from}-${b.to}-${b.color}`}>
            <rect x={x0} y={T} width={x(i1) - x0 || 10} height={H - T - B} fill={b.color} opacity={0.15} />
            <text x={x(i1) - 4} y={T + 12} textAnchor="end" className="band-label" fill={b.color}>{b.label}</text>
          </g>
        )
      })}
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
          <text x={L - 6} y={y(t) + 4} textAnchor="end" className="tick">{fmtY(t)}</text>
        </g>
      ))}
      {xs.map((c, i) => <text key={c} x={x(i)} y={H - B + 16} textAnchor="middle" className="tick">{fmtCtx(c)}</text>)}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="axis">context (tokens)</text>
      <text x={12} y={(T + H - B) / 2} textAnchor="middle" className="axis" transform={`rotate(-90 12 ${(T + H - B) / 2})`}>{yLabel}</text>
      {series.map((s) => (
        <g key={s.label}>
          {segs(s.ys).map((d) => <path key={d} d={d} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? '5 4' : undefined} />)}
          {s.ys.map((v, i) => v != null && <circle key={i} cx={x(i)} cy={y(v)} r={3} fill={s.color}><title>{`${s.label} @ ${fmtCtx(xs[i])}: ${fmtY(v)}`}</title></circle>)}
        </g>
      ))}
    </svg>
  )
}
