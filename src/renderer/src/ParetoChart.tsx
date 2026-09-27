import type { SessionCandidate } from '../../shared/types'
import { factsOf } from './SloFilter'
import { fmtCtx, num } from './ui'

export interface ParetoPoint { id: string; x: number; y: number; label: string }

/** Points no other point beats on both axes (maximize x and y), sorted by x ascending. Identical points are all
 *  kept (neither dominates). O(n²); n is a handful of configs. */
export function paretoFrontier<P extends { id: string; x: number; y: number }>(pts: P[]): P[] {
  const dominated = (p: P) => pts.some((q) => q !== p && q.x >= p.x && q.y >= p.y && (q.x > p.x || q.y > p.y))
  return pts.filter((p) => !dominated(p)).sort((a, b) => a.x - b.x || b.y - a.y || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
}

/** x = decode t/s at the recommended ctx (else the scored ctx); y = MEASURED quality. Anything else is listed, not plotted. */
export function paretoPoints(cs: SessionCandidate[]): { points: ParetoPoint[]; skipped: { id: string; why: string }[] } {
  const points: ParetoPoint[] = []
  const skipped: { id: string; why: string }[] = []
  for (const c of cs) {
    const f = factsOf(c)
    const q = c.score?.breakdown.find((b) => b.component === 'quality')
    if (f.decodeTps == null) skipped.push({ id: c.config.id, why: 'decode t/s not measured at the recommended context' })
    else if (!q || q.input.kind !== 'measured') skipped.push({ id: c.config.id, why: q ? `quality is ${q.input.kind.toUpperCase()}, not measured` : 'no quality score' })
    else points.push({ id: c.config.id, x: f.decodeTps, y: q.score, label: `${c.model.name} ${c.model.quant ?? ''} @${f.ctx ? fmtCtx(f.ctx) : '?'}${c.config.gpuLayersAll ? '' : ` ngl ${c.config.gpuLayers}`}` })
  }
  return { points, skipped }
}

const W = 640, H = 300, PAD = { l: 48, r: 16, t: 16, b: 36 }

export function ParetoChart({ candidates }: { candidates: SessionCandidate[] }) {
  const { points, skipped } = paretoPoints(candidates)
  const front = paretoFrontier(points)
  const onFront = new Set(front.map((p) => p.id))
  const xMax = Math.max(1, ...points.map((p) => p.x)) * 1.1
  const sx = (x: number) => PAD.l + (x / xMax) * (W - PAD.l - PAD.r)
  const sy = (y: number) => H - PAD.b - (y / 100) * (H - PAD.t - PAD.b)
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <div>
      {points.length ? (
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Pareto chart: decode speed vs measured quality">
          {ticks.map((t) => (
            <g key={t} className="grid">
              <line x1={PAD.l} x2={W - PAD.r} y1={sy(t * 100)} y2={sy(t * 100)} stroke="currentColor" opacity={0.12} />
              <text x={PAD.l - 6} y={sy(t * 100) + 4} textAnchor="end" fontSize={11} fill="currentColor">{t * 100}</text>
              <text x={sx(t * xMax)} y={H - PAD.b + 16} textAnchor="middle" fontSize={11} fill="currentColor">{num(t * xMax, 0)}</text>
            </g>
          ))}
          <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="currentColor">decode t/s at recommended ctx</text>
          <text x={12} y={(PAD.t + H - PAD.b) / 2} textAnchor="middle" fontSize={11} fill="currentColor" transform={`rotate(-90 12 ${(PAD.t + H - PAD.b) / 2})`}>quality (measured)</text>
          {front.length > 1 && <polyline points={front.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} fill="none" stroke="#3fb950" strokeWidth={2} />}
          {points.map((p) => (
            <g key={p.id}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={onFront.has(p.id) ? 6 : 4} fill={onFront.has(p.id) ? '#3fb950' : '#8b949e'}>
                <title>{`${p.id}\ndecode ${num(p.x)} t/s, quality ${p.y.toFixed(0)}${onFront.has(p.id) ? ' — Pareto-optimal' : ''}`}</title>
              </circle>
              <text x={sx(p.x) + 8} y={sy(p.y) - 8} fontSize={11} fill="currentColor">{p.label}</text>
            </g>
          ))}
        </svg>
      ) : <p className="muted">No configuration has both a measured decode rate and a measured quality score.</p>}
      {front.length > 0 && <p className="muted">Pareto-optimal (green): {front.map((p) => p.label).join(', ')}</p>}
      {skipped.length > 0 && <ul className="muted">{skipped.map((s) => <li key={s.id}>Not plotted: {s.id} — {s.why}</li>)}</ul>}
    </div>
  )
}
