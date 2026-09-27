import { describe, expect, it } from 'vitest'
import type { SessionEvent } from '../src/shared/bench-events'
import type { BenchmarkRunResult } from '../src/shared/bench-types'
import { LOG_MAX, applyEvent, initialLive } from '../src/renderer/src/benchState'

const s = 'S1'
const result = { status: 'pass', decodeTps: { value: 88.5, kind: 'measured' } } as unknown as BenchmarkRunResult

describe('live benchmark reducer', () => {
  it('tracks candidate, phase, progress, telemetry and final status', () => {
    const evs: SessionEvent[] = [
      { sessionId: s, type: 'session:started', workload: 'coding', modelIds: ['m'], resumed: false, candidates: 2 },
      { sessionId: s, type: 'candidate:started', configId: 'c1', model: 'm', gpuLayers: 99, ctxSteps: [2048, 4096] },
      { sessionId: s, type: 'phase', configId: 'c1', ctx: 2048, phase: 'measure' },
      { sessionId: s, type: 'token-rate', configId: 'c1', ctx: 2048, prefillTps: 3000, decodeTps: 88.5, ttftMs: 600 },
      { sessionId: s, type: 'step:done', configId: 'c1', ctx: 2048, result, verdict: 'pass' },
      { sessionId: s, type: 'candidate:done', configId: 'c1', status: 'done', reason: null },
      { sessionId: s, type: 'session:paused' }
    ]
    const st = evs.reduce(applyEvent, initialLive)
    expect(st).toMatchObject({ status: 'paused', candidatesTotal: 2, configId: 'c1', phase: 'measure', ctx: 2048, stepsDone: 1, candidatesStarted: 1, candidatesDone: 1, rate: { decodeTps: 88.5 } })
    expect(st.log.some((l) => l.includes('decode 88.5 t/s'))).toBe(true)
  })

  it('a pre-session log event (sessionId "") does not blank the watched session', () => {
    const st = applyEvent({ ...initialLive, sessionId: '7', status: 'running' }, { sessionId: '', type: 'log', level: 'warn', msg: 'skipped' })
    expect(st.sessionId).toBe('7')
  })

  it('caps the log', () => {
    let st = initialLive
    for (let i = 0; i < LOG_MAX + 50; i++) st = applyEvent(st, { sessionId: s, type: 'log', level: 'info', msg: `m${i}` })
    expect(st.log).toHaveLength(LOG_MAX)
    expect(st.log.at(-1)).toMatch(/m249$/)
  })
})
