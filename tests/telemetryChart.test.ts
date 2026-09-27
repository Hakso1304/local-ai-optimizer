import { describe, expect, it } from 'vitest'
import { downsample, type Pt } from '../src/renderer/src/TelemetryChart'

describe('downsample', () => {
  it('returns short series unchanged', () => {
    const pts: Pt[] = [{ t: 0, v: 1 }, { t: 1, v: 2 }]
    expect(downsample(pts, 10)).toBe(pts)
  })
  it('keeps each bucket min and max (spikes survive) in time order, within maxPoints', () => {
    const pts: Pt[] = Array.from({ length: 1000 }, (_, i) => ({ t: i, v: i === 500 ? 99 : i === 250 ? -5 : 10 }))
    const out = downsample(pts, 100)
    expect(out.length).toBeLessThanOrEqual(100)
    expect(out.some((p) => p.v === 99)).toBe(true)
    expect(out.some((p) => p.v === -5)).toBe(true)
    expect(out.every((p, i) => i === 0 || p.t >= out[i - 1].t)).toBe(true)
  })
  it('a bucket with a null keeps a null (gap), never interpolates', () => {
    const pts: Pt[] = Array.from({ length: 100 }, (_, i) => ({ t: i, v: i >= 40 && i < 45 ? null : 1 }))
    const out = downsample(pts, 20)
    expect(out.filter((p) => p.v === null).length).toBeGreaterThanOrEqual(1)
    expect(out.some((p) => p.v === 0)).toBe(false)
  })
})
