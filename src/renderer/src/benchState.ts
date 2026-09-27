// Live benchmark view state, folded from bench:event. Pure so it can be unit-tested.
import type { SessionEvent, SessionPhase, TelemetrySample } from '../../shared/bench-events'
import type { Recommendation } from '../../shared/bench-types'

export interface LiveState {
  sessionId: string | null
  status: 'idle' | 'running' | 'done' | 'cancelled' | 'failed'
  configId: string | null
  model: string | null
  phase: SessionPhase | null
  ctx: number | null
  ctxSteps: number[]
  stepsDone: number
  candidatesStarted: number
  candidatesDone: number
  telemetry: TelemetrySample | null
  rate: { prefillTps: number | null; decodeTps: number | null; ttftMs: number | null } | null
  log: string[]
  recommendation: Recommendation | null
  error: string | null
}

export const LOG_MAX = 200

export const initialLive: LiveState = {
  sessionId: null, status: 'idle', configId: null, model: null, phase: null, ctx: null, ctxSteps: [], stepsDone: 0,
  candidatesStarted: 0, candidatesDone: 0, telemetry: null, rate: null, log: [], recommendation: null, error: null
}

const line = (e: SessionEvent): string | null => {
  switch (e.type) {
    case 'session:started': return `session ${e.sessionId} started (${e.workload}, ${e.modelIds.length} models${e.resumed ? ', resumed' : ''})`
    case 'candidate:started': return `candidate ${e.configId}: ctx ${e.ctxSteps.join(', ')}`
    case 'step:done': return `${e.configId} @${e.ctx}: ${e.result.status}/${e.verdict}${e.result.decodeTps.value != null ? `, decode ${e.result.decodeTps.value.toFixed(1)} t/s` : ''}`
    case 'log': return `[${e.level}] ${e.msg}`
    case 'candidate:done': return `candidate ${e.configId} ${e.status}${e.reason ? `: ${e.reason}` : ''}`
    case 'session:done': return 'session done'
    case 'session:cancelled': return 'session cancelled'
    case 'session:failed': return `session failed: ${e.error}`
    default: return null // phase / step:started / telemetry / token-rate are shown in the panel, not the log
  }
}

export function applyEvent(s: LiveState, e: SessionEvent): LiveState {
  const l = line(e)
  const log = l ? [...s.log, `${new Date().toLocaleTimeString()} ${l}`].slice(-LOG_MAX) : s.log
  const n = { ...s, log, sessionId: e.sessionId }
  switch (e.type) {
    case 'session:started': return { ...initialLive, log, sessionId: e.sessionId, status: 'running' }
    case 'candidate:started': return { ...n, configId: e.configId, model: e.model, ctxSteps: e.ctxSteps, stepsDone: 0, phase: null, ctx: null, candidatesStarted: s.candidatesStarted + 1 }
    case 'phase': return { ...n, configId: e.configId, ctx: e.ctx, phase: e.phase }
    case 'step:started': return { ...n, ctx: e.ctx }
    case 'step:done': return { ...n, stepsDone: s.stepsDone + 1 }
    case 'telemetry': return { ...n, telemetry: e.sample }
    case 'token-rate': return { ...n, rate: { prefillTps: e.prefillTps, decodeTps: e.decodeTps, ttftMs: e.ttftMs } }
    case 'candidate:done': return { ...n, candidatesDone: s.candidatesDone + 1 }
    case 'session:done': return { ...n, status: 'done', recommendation: e.recommendation }
    case 'session:cancelled': return { ...n, status: 'cancelled' }
    case 'session:failed': return { ...n, status: 'failed', error: e.error }
    default: return n
  }
}
