import { describe, expect, it } from 'vitest'
import type { ComponentId, Recommendation } from '../../src/shared/bench-types'
import { detectCliffs } from '../../src/core/scoring/cliff'
import { componentScores } from '../../src/core/scoring/components'
import { recommend } from '../../src/core/scoring/recommend'
import { DEFAULT_SCORING_CONFIG, WORKLOADS } from '../../src/core/scoring/workloads'
import { inputs, load, machine, toRun } from './helpers'

const sweep = (name: string) => {
  const f = load(name)
  return detectCliffs(f.runs.map(toRun), f.vramBytes)
}
const finiteEverywhere = (r: Recommendation) =>
  r.ranked.every((s) => Number.isFinite(s.total) && s.breakdown.every((b) => Number.isFinite(b.score) && Number.isFinite(b.contribution)))

describe('workload profiles', () => {
  it('has 8 profiles whose weights are non-negative and sum to 1 (X6)', () => {
    expect(Object.keys(WORKLOADS)).toHaveLength(8)
    for (const p of Object.values(WORKLOADS)) {
      const ws = Object.values(p.weights)
      expect(ws.every((w) => w >= 0)).toBe(true)
      expect(Math.abs(ws.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9)
      expect(p.targetContext).toBeGreaterThan(DEFAULT_SCORING_CONFIG.norm.minCtx)
    }
  })
})

describe('detectCliffs', () => {
  it('flags the 16K→32K collapse; practical ceiling 16K; 64K degraded (A15)', () => {
    const r = sweep('sweep-cliff-16k-32k.json')
    expect(r.steps.map((s) => s.verdict)).toEqual(['pass', 'pass', 'pass', 'pass', 'degraded', 'degraded'])
    const drop = r.steps[4].reasons.find((x) => x.code === 'decode_drop')!
    expect(drop).toMatchObject({ metric: 'decodeTps', fromCtx: 16384, toCtx: 32768, from: 82, to: 31, threshold: 0.6 })
    expect(drop.message).toBe('decode TPS fell 62% between 16K and 32K (82.0 → 31.0 t/s)')
    expect(r.steps[4].reasons.map((x) => x.code)).toEqual(expect.arrayContaining(['vram_spill', 'shared_spill']))
    expect(r.practicalContextCeiling).toMatchObject({ value: 16384, kind: 'measured' })
    expect(r.limitedBy).toBe('cliff')
    expect(r.spillFreeUpTo).toBe(16384)
  })

  it('no false cliffs on smooth decline or a single −35% dip (X8)', () => {
    for (const f of ['sweep-smooth.json', 'sweep-noisy.json']) {
      const r = sweep(f)
      expect(r.steps.every((s) => s.verdict === 'pass'), f).toBe(true)
      expect(r.practicalContextCeiling.value).toBe(65536)
      expect(r.limitedBy).toBe('none')
    }
  })

  it('a failed 32K step limits by failure, not cliff (X8)', () => {
    const r = sweep('sweep-fail-32k.json')
    expect(r.steps.at(-1)).toMatchObject({ ctx: 32768, verdict: 'fail' })
    expect(r.practicalContextCeiling.value).toBe(16384)
    expect(r.limitedBy).toBe('failure')
  })

  it('RAM growth with VRAM far from its limit is not spill (X11)', () => {
    const r = sweep('sweep-mmap-ram.json')
    expect(r.steps.every((s) => s.verdict === 'pass')).toBe(true)
  })

  it('F13: a single transient dip that the next rung recovers from is not a sticky cliff; a real collapse still is', () => {
    const at = (ctx: number, d: number) => toRun({ configId: 'x', model: 'm', ctx, gpuLayers: 32, threads: 8, status: 'ok', promptTokens: 1000, loadMs: 1, ttftMs: 100, prefillTps: 2000, decodeTps: d, totalMs: 1, peakRamBytes: 1, peakVramBytes: 1, peakSharedGpuBytes: 0, cpuAvgPct: 1, gpuAvgPct: 1 })
    const dip = detectCliffs([at(2048, 100), at(4096, 60), at(8192, 100)], null)
    expect(dip.steps.map((x) => x.verdict)).toEqual(['pass', 'pass', 'pass'])
    expect(dip.practicalContextCeiling.value).toBe(8192)
    const real = sweep('sweep-cliff-16k-32k.json') // 82 → 31 → 18: no recovery
    expect(real.practicalContextCeiling.value).toBe(16384)
    const lastRung = detectCliffs([at(2048, 100), at(4096, 55)], null) // nothing to confirm with → counts
    expect(lastRung.steps[1].verdict).toBe('degraded')
  })

  it('empty input is untested, not NaN', () => {
    const r = detectCliffs([], null)
    expect(r).toMatchObject({ steps: [], limitedBy: 'untested', practicalContextCeiling: { value: null, kind: 'unavailable' } })
  })
})

describe('componentScores', () => {
  it('scores one candidate absolutely; all components finite and in 0–100 (X1)', () => {
    const [c] = inputs(load('session-single.json'))
    const s = componentScores(c, machine(), WORKLOADS.coding)
    for (const k of Object.keys(s.components) as ComponentId[]) {
      expect(s.components[k].score).toBeGreaterThanOrEqual(0)
      expect(s.components[k].score).toBeLessThanOrEqual(100)
    }
    expect(s.referenceCtx).toBe(16384)
    expect(s.components.quality.input.kind).toBe('estimated') // no suite results → prior, labelled (X20)
  })

  it('lower TTFT scores higher latency, all else equal (X7)', () => {
    const [a] = inputs(load('session-single.json'))
    const slow = { ...a, runs: a.runs.map((r) => ({ ...r, ttftMs: { value: r.ttftMs.value! * 3, kind: 'measured' as const } })) }
    const fa = componentScores(a, machine(), WORKLOADS.coding).components.latency.score
    const sl = componentScores(slow, machine(), WORKLOADS.coding).components.latency.score
    expect(fa).toBeGreaterThan(sl)
  })

  it('uses the practical ceiling, not the declared context (X9)', () => {
    const [c] = inputs(load('sweep-cliff-16k-32k.json'))
    const r = recommend([c], machine(), 'coding') // coding targets 16K; long-context coding now needs ≥ 32K
    expect(r.best?.practicalContext).toMatchObject({ value: 16384, kind: 'measured' })
    expect(r.best?.declaredContext).toMatchObject({ value: 131072, kind: 'declared' })
    expect(r.best?.score.referenceCtx).toBe(16384)
    expect(r.reasons).toContain('decode TPS fell 62% between 16K and 32K (82.0 → 31.0 t/s)')
    expect(r.reasons).toContain('No VRAM spill up to 16K')
  })
})

describe('recommend', () => {
  it('single candidate: finite score, picked, noted as not compared (X1)', () => {
    const r = recommend(inputs(load('session-single.json')), machine(), 'general_chat')
    expect(r.best?.configId).toBe('single')
    expect(finiteEverywhere(r)).toBe(true)
    expect(r.reasons).toContain('Only one candidate; not compared')
    const b = r.best!.score
    expect(Math.abs(b.breakdown.reduce((s, x) => s + x.contribution, 0) - b.total)).toBeLessThan(1e-9) // A16
  })

  it('all failed: no winner, every config excluded with a reason (X2, A18)', () => {
    const r = recommend(inputs(load('session-all-failed.json')), machine(), 'coding')
    expect(r.best).toBeNull()
    expect(r.reasons[0]).toBe('No recommendation: no successful runs')
    expect(r.excluded.map((e) => e.configId)).toEqual(['a-oom', 'b-timeout', 'c-crashed'])
    expect(r.excluded[0].reasons[0]).toMatch(/oom/)
    expect(r.alternatives).toEqual({ fastest: null, bestQuality: null, bestLongContext: null, lowestMemory: null })
  })

  it('tie: winner independent of input order, broken by configId (X3)', () => {
    const cs = inputs(load('session-tie.json'))
    const a = recommend(cs, machine(), 'coding')
    const b = recommend([...cs].reverse(), machine(), 'coding')
    expect(a.best?.configId).toBe('a-config')
    expect(b).toEqual(a) // A19: deterministic, deep-equal
  })

  it('zero TPS / zero TTFT run is excluded, no NaN (X4)', () => {
    const r = recommend(inputs(load('session-zero-tps.json')), machine(), 'fast_assistant')
    expect(r.best).toBeNull()
    expect(r.excluded[0].reasons[0]).toMatch(/no valid decode TPS/)
    expect(finiteEverywhere(r)).toBe(true)
  })

  it('adding a much worse candidate does not reorder the others (X5)', () => {
    const [good] = inputs(load('session-single.json'))
    const worse = { ...good, config: { ...good.config, id: 'worse' }, runs: good.runs.map((r) => ({ ...r, configId: 'worse', decodeTps: { value: r.decodeTps.value! / 2, kind: 'measured' as const } })) }
    const awful = { ...good, config: { ...good.config, id: 'awful' }, runs: good.runs.map((r) => ({ ...r, configId: 'awful', decodeTps: { value: 2.5, kind: 'measured' as const } })) }
    const two = recommend([good, worse], machine(), 'general_chat').ranked.map((s) => [s.configId, s.total])
    const three = recommend([good, worse, awful], machine(), 'general_chat').ranked.map((s) => [s.configId, s.total])
    expect(three.slice(0, 2)).toEqual(two)
  })
})
