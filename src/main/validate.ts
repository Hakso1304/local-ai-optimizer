// Trust boundary: everything the renderer sends to bench:* is re-validated here. Pure (no electron import) for tests.
import { isAbsolute, relative, resolve } from 'node:path'
import { DEFAULT_CANDIDATE_RULES, type CandidateRules } from '../core/benchmark/candidates'
import { WORKLOADS } from '../core/scoring/workloads'
import type { SessionRequest } from '../shared/bench-events'
import type { WorkloadId } from '../shared/bench-types'

/** True if p is dir itself's descendant (no `..` escape, same drive). */
export function isInside(dir: string, p: string): boolean {
  const rel = relative(resolve(dir), resolve(p))
  return rel !== '' && !rel.startsWith('..') && !isAbsolute(rel)
}

const D = DEFAULT_CANDIDATE_RULES
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** Only these rules may come from the renderer, and safety margins can only get stricter than the defaults. */
function sanitizeRules(raw: unknown): Partial<CandidateRules> | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const r = raw as Record<string, unknown>
  const out: Partial<CandidateRules> = {}
  const mp = num(r.maxPerModel)
  if (mp !== null) out.maxPerModel = Math.min(D.maxPerModel, Math.max(1, Math.round(mp)))
  for (const k of ['ramReserveBytes', 'heavyRamReserveBytes', 'vramMarginBytes'] as const) {
    const v = num(r[k])
    if (v !== null) out[k] = Math.max(D[k], v)
  }
  const ko = num(r.keepOverVramMaxRatio)
  if (ko !== null) out.keepOverVramMaxRatio = Math.min(D.keepOverVramMaxRatio, Math.max(1, ko))
  return Object.keys(out).length ? out : undefined
}

const bool = (v: unknown) => (typeof v === 'boolean' ? v : undefined)
const strings = (v: unknown) => (Array.isArray(v) && v.every((x) => typeof x === 'string') ? (v as string[]) : null)

/** Renderer input → SessionRequest. resumeSessionId is never taken from the renderer (main sets it). */
export function sanitizeRequest(raw: unknown, modelRoots: string[]): { ok: true; req: SessionRequest } | { ok: false; error: string } {
  if (!raw || typeof raw !== 'object') return { ok: false, error: 'bad request' }
  const r = raw as Record<string, unknown>
  if (typeof r.workload !== 'string' || !(r.workload in WORKLOADS)) return { ok: false, error: 'unknown workload' }
  const modelIds = strings(r.modelIds)
  if (!modelIds?.length || modelIds.length > 20) return { ok: false, error: 'pick 1–20 models' }
  const outside = modelIds.find((p) => !modelRoots.some((d) => isInside(d, p)))
  if (outside) return { ok: false, error: `model path not in a configured model dir: ${outside}` }
  const ladder = Array.isArray(r.ladder)
    ? [...new Set((r.ladder as unknown[]).filter((c): c is number => typeof c === 'number' && D.ctxLadder.includes(c)))].sort((a, b) => a - b)
    : []
  const reps = num(r.reps)
  const rerun = strings(r.rerunConfigIds)
  return {
    ok: true,
    req: {
      workload: r.workload as WorkloadId,
      modelIds,
      ...(ladder.length ? { ladder } : {}),
      ...(reps !== null ? { reps: Math.min(5, Math.max(1, Math.round(reps))) } : {}),
      ...(bool(r.runQuality) !== undefined ? { runQuality: bool(r.runQuality) } : {}),
      ...(bool(r.heavyMode) !== undefined ? { heavyMode: bool(r.heavyMode) } : {}),
      ...(bool(r.retryFailed) !== undefined ? { retryFailed: bool(r.retryFailed) } : {}),
      ...(rerun?.length ? { rerunConfigIds: rerun } : {}),
      ...(sanitizeRules(r.candidateRules) ? { candidateRules: sanitizeRules(r.candidateRules) } : {})
    }
  }
}
