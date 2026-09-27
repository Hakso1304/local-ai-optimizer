import { expect, it } from 'vitest'
import { applyEvent, initialLive } from '../../src/renderer/src/benchState'
import { downsample } from '../../src/renderer/src/TelemetryChart'
import { paretoPoints } from '../../src/renderer/src/ParetoChart'
import { sloDefaults } from '../../src/renderer/src/SloFilter'
import { WORKLOADS, effectiveProfile } from '../../src/core/scoring/workloads'

it.fails('ignores a late event from a previous session', () => {
  const state = { ...initialLive, sessionId: 'new', status: 'running' as const }
  expect(applyEvent(state, { sessionId: 'old', type: 'session:cancelled' })).toEqual(state)
})
it.fails('clears the previous model rates and telemetry at candidate start', () => {
  const state = { ...initialLive, sessionId: '1', rate: { prefillTps: 1000, decodeTps: 100, ttftMs: 20 } }
  const next = applyEvent(state, { sessionId: '1', type: 'candidate:started', configId: 'new', model: 'new', gpuLayers: 1, ctxSteps: [2048] })
  expect(next.rate).toBeNull()
})
it.fails('preserves measured peaks in a bucket that also contains an unavailable sample', () => {
  const points = [{ t: 0, v: 1 }, { t: 1, v: 99 }, { t: 2, v: null }, { t: 3, v: 2 }]
  expect(downsample(points, 2).some((p) => p.v === 99)).toBe(true)
})
it.fails('does not present estimated decode as a measured Pareto point', () => {
  const c = { config: { id: 'c', gpuLayersAll: true }, model: { name: 'm' },
    score: { referenceCtx: 2048, breakdown: [{ component: 'quality', score: 80, input: { value: 80, kind: 'measured' } }] },
    cliff: { practicalContextCeiling: { value: 2048 } }, runs: [{ ctx: 2048, status: 'pass', decodeTps: { value: 100, kind: 'estimated' } }] }
  expect(paretoPoints([c as any]).points).toHaveLength(0)
})
it.fails('does not enable a hard SLO latency limit for an advisory-latency profile', () => {
  const p = effectiveProfile(WORKLOADS.large_coding, { requiredContext: 65536 })
  expect(sloDefaults(p, 65536).maxTtftS).toBeNull()
})
