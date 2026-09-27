// IPC contract between the benchmark session runner (main) and the live view (renderer). Types only.
import type { TelemetrySample } from '../core/telemetry/sampler'
import type { CandidateRules } from '../core/benchmark/candidates'
import type { BenchmarkRunResult, Recommendation, StepVerdict, WorkloadId } from './bench-types'

export type { TelemetrySample }

export interface SessionRequest {
  workload: WorkloadId
  /** ModelMeta ids (absolute GGUF paths). */
  modelIds: string[]
  candidateRules?: Partial<CandidateRules>
  /** Context steps to try (intersected with each candidate's steps). Default: candidate rules ladder. */
  ladder?: number[]
  /** Measured reps per step, median reported. Default 2. */
  reps?: number
  /** Default true. */
  runQuality?: boolean
  /** Heavy-model mode: models whose full GPU offload does not fit get a partial-offload ladder instead of being
   *  rejected (slow, flagged expectDegraded). Default false. */
  heavyMode?: boolean
  /** With resumeSessionId: also re-run steps that failed or timed out (not config_drift). */
  retryFailed?: boolean
  /** With resumeSessionId: re-run every step of these configs ("rerun selected configuration"); new rows supersede. */
  rerunConfigIds?: string[]
  /** Continue this session: (configId, ctx) steps and quality already persisted are skipped. */
  resumeSessionId?: string
}

export type SessionPhase = 'load' | 'warmup' | 'measure' | 'ladder' | 'quality'
export type CandidateStatus = 'done' | 'failed' | 'cancelled' | 'paused' | 'skipped'

/** Event payload without the session id (what the runner builds). */
export type SessionEventBody =
  | { type: 'session:started'; workload: WorkloadId; modelIds: string[]; resumed: boolean; /** planned candidate configs (progress total) */ candidates?: number }
  | { type: 'candidate:started'; configId: string; model: string; gpuLayers: number; ctxSteps: number[] }
  | { type: 'phase'; configId: string; ctx: number; phase: SessionPhase }
  | { type: 'step:started'; configId: string; ctx: number }
  | { type: 'step:done'; configId: string; ctx: number; result: BenchmarkRunResult; verdict: StepVerdict }
  | { type: 'telemetry'; configId: string; ctx: number; sample: TelemetrySample }
  | { type: 'token-rate'; configId: string; ctx: number; prefillTps: number | null; decodeTps: number | null; ttftMs: number | null }
  | { type: 'log'; level: 'info' | 'warn' | 'error'; msg: string }
  | { type: 'candidate:done'; configId: string; status: CandidateStatus; reason: string | null }
  | { type: 'session:done'; recommendation: Recommendation }
  | { type: 'session:cancelled' }
  | { type: 'session:paused' }
  | { type: 'session:failed'; error: string }

export type SessionEvent = { sessionId: string } & SessionEventBody
