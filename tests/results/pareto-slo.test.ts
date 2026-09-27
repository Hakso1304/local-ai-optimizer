import { describe, expect, it } from 'vitest'
import type { Metric } from '../../src/shared/bench-types'
import type { SessionCandidate } from '../../src/shared/types'
import { paretoFrontier, paretoPoints } from '../../src/renderer/src/ParetoChart'
import { factsOf, meetsSlo, parseLimit, sessionProfile, sloDefaults } from '../../src/renderer/src/SloFilter'
import { WORKLOADS } from '../../src/core/scoring/workloads'

const p = (id: string, x: number, y: number) => ({ id, x, y })
const ids = (xs: { id: string }[]) => xs.map((x) => x.id)

describe('paretoFrontier (maximize decode and quality)', () => {
  it('keeps only non-dominated points, sorted by x', () => {
    // real 2026-09-27 shape: 1.5B fast/low quality, 8B mid, 27B partial slow/high quality, 14B@2K dominated by 8B
    const pts = [p('qwen1.5', 309, 61), p('llama8', 93, 100), p('q27-partial', 8, 100), p('q14', 62, 85), p('cpu', 3, 40)]
    expect(ids(paretoFrontier(pts))).toEqual(['llama8', 'qwen1.5'])
  })
  it('a point equal on one axis and worse on the other is dominated', () => {
    expect(ids(paretoFrontier([p('a', 10, 50), p('b', 10, 60), p('c', 5, 60)]))).toEqual(['b'])
  })
  it('identical points both stay on the frontier (neither dominates)', () => {
    expect(ids(paretoFrontier([p('a', 10, 50), p('b', 10, 50)]))).toEqual(['a', 'b'])
  })
  it('single point and empty input', () => {
    expect(ids(paretoFrontier([p('only', 1, 1)]))).toEqual(['only'])
    expect(paretoFrontier([])).toEqual([])
  })
  it('is independent of input order', () => {
    const pts = [p('a', 1, 90), p('b', 50, 70), p('c', 100, 20), p('d', 40, 60), p('e', 100, 10)]
    const want = ids(paretoFrontier(pts))
    expect(want).toEqual(['a', 'b', 'c'])
    expect(ids(paretoFrontier([...pts].reverse()))).toEqual(want)
  })
})

// Minimal SessionCandidate builder (only the fields factsOf/paretoPoints read).
const m = (v: number | null): Metric => (v == null ? { value: null, kind: 'unavailable', reason: 'x' } : { value: v, kind: 'measured' })
function cand(o: { id: string; rec?: number | null; ref?: number | null; practical?: number | null; runs: { ctx: number; status?: string; ttft?: number | null; decode?: number | null; vram?: number | null }[]; quality?: { score: number; kind: Metric['kind'] } }): SessionCandidate {
  return {
    config: { id: o.id, gpuLayersAll: true, gpuLayers: 99 },
    model: { name: o.id, quant: 'Q4_K_M' },
    runs: o.runs.map((r) => ({ ctx: r.ctx, status: r.status ?? 'pass', ttftMs: m(r.ttft ?? null), decodeTps: m(r.decode ?? null), peakVramBytes: m(r.vram ?? null) })),
    runIds: [],
    cliff: { practicalContextCeiling: m(o.practical ?? null) },
    score: o.rec === undefined && o.ref === undefined ? null : {
      recommendedCtx: o.rec ?? null, referenceCtx: o.ref ?? null,
      breakdown: o.quality ? [{ component: 'quality', score: o.quality.score, input: { value: o.quality.score, kind: o.quality.kind } }] : []
    },
    quality: []
  } as unknown as SessionCandidate
}

const GiB = 1024 ** 3
const llama = cand({ id: 'llama8', rec: 32768, ref: 16384, practical: 65536, quality: { score: 100, kind: 'measured' },
  runs: [{ ctx: 16384, ttft: 3506, decode: 93, vram: 6.55 * GiB }, { ctx: 32768, ttft: 8130, decode: 79, vram: 8.56 * GiB }, { ctx: 65536, ttft: 20643, decode: 60.7, vram: 12.6 * GiB }] })

describe('factsOf', () => {
  it('uses the scored (reference) step for TTFT/decode/VRAM — not recommendedCtx, not the max over rungs', () => {
    expect(factsOf(llama)).toEqual({ ctx: 16384, ttftMs: 3506, decodeTps: 93, decodeKind: 'measured', practicalCtx: 65536, peakVramBytes: 6.55 * GiB })
  })
  it('a failed step never supplies numbers', () => {
    const c = cand({ id: 'x', rec: null, ref: 8192, practical: 8192, runs: [{ ctx: 8192, ttft: 900, decode: 40 }, { ctx: 16384, status: 'fail', decode: 999, vram: 99 * GiB }] })
    expect(factsOf(c)).toMatchObject({ ctx: 8192, decodeTps: 40, peakVramBytes: null })
  })
})

describe('meetsSlo', () => {
  it('coding defaults: TTFT ≤ 15 s, decode ≥ 10, practical ≥ 8K, no VRAM cap', () => {
    const s = sloDefaults(WORKLOADS.coding)
    expect(s).toEqual({ maxTtftS: 15, minDecodeTps: 10, minPracticalCtx: 8192, maxVramGiB: null })
    expect(meetsSlo(factsOf(llama), s)).toEqual({ ok: true, failed: [] })
  })
  it('each constraint can fail on its own', () => {
    const f = factsOf(llama)
    expect(meetsSlo(f, { maxTtftS: 3, minDecodeTps: null, minPracticalCtx: null, maxVramGiB: null }).failed).toEqual(['TTFT'])
    expect(meetsSlo(f, { maxTtftS: null, minDecodeTps: 95, minPracticalCtx: null, maxVramGiB: null }).failed).toEqual(['decode'])
    expect(meetsSlo(f, { maxTtftS: null, minDecodeTps: null, minPracticalCtx: 131072, maxVramGiB: null }).failed).toEqual(['practical ctx'])
    expect(meetsSlo(f, { maxTtftS: null, minDecodeTps: null, minPracticalCtx: null, maxVramGiB: 6 }).failed).toEqual(['peak VRAM'])
    // VRAM of the scored step (6.55 GiB), not of the 64K rung (12.6 GiB): a 7 GiB cap passes
    expect(meetsSlo(f, { maxTtftS: null, minDecodeTps: null, minPracticalCtx: null, maxVramGiB: 7 }).ok).toBe(true)
  })
  it('labels an estimated decode rate in the failure reason', () => {
    const f = { ...factsOf(llama), decodeKind: 'estimated' as const }
    expect(meetsSlo(f, { maxTtftS: null, minDecodeTps: 200, minPracticalCtx: null, maxVramGiB: null }).failed).toEqual(['decode (ESTIMATED)'])
  })
  it('defaults follow the effective profile: advisory latency → no TTFT limit, required ctx → ctx floor', () => {
    const p = sessionProfile({ workload: 'coding', requiredContext: 65536, minDecodeTps: 20 })
    expect(sloDefaults(p, 65536)).toEqual({ maxTtftS: null, minDecodeTps: 20, minPracticalCtx: 65536, maxVramGiB: null })
  })
  it('an unmeasured value fails an active constraint (unknown ≠ ok) but passes when the constraint is off', () => {
    const f = { ctx: null, ttftMs: null, decodeTps: null, practicalCtx: null, peakVramBytes: null }
    expect(meetsSlo(f, { maxTtftS: 2, minDecodeTps: null, minPracticalCtx: null, maxVramGiB: null })).toEqual({ ok: false, failed: ['TTFT not measured'] })
    expect(meetsSlo(f, { maxTtftS: null, minDecodeTps: null, minPracticalCtx: null, maxVramGiB: null }).ok).toBe(true)
  })
})

describe('parseLimit (input clamp)', () => {
  it('blank, non-numeric and negative mean off; 0 and positives pass through', () => {
    expect(parseLimit('')).toBeNull()
    expect(parseLimit('  ')).toBeNull()
    expect(parseLimit('abc')).toBeNull()
    expect(parseLimit('-3')).toBeNull()
    expect(parseLimit('0')).toBe(0)
    expect(parseLimit('12.5')).toBe(12.5)
  })
})

describe('paretoPoints', () => {
  it('plots measured quality × decode; lists estimated quality and missing decode instead of plotting them', () => {
    const est = cand({ id: 'est', rec: 8192, ref: 8192, practical: 8192, quality: { score: 80, kind: 'estimated' }, runs: [{ ctx: 8192, decode: 20 }] })
    const nodec = cand({ id: 'nodec', rec: null, ref: null, practical: null, quality: { score: 90, kind: 'measured' }, runs: [] })
    const r = paretoPoints([llama, est, nodec])
    expect(r.points).toEqual([{ id: 'llama8', x: 93, y: 100, label: 'llama8 Q4_K_M @16K' }])
    expect(r.skipped.map((s) => [s.id, s.why])).toEqual([
      ['est', 'quality is ESTIMATED, not measured'],
      ['nodec', 'decode t/s not measured at the scored context']
    ])
  })
})
