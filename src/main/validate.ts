// Trust boundary: everything the renderer sends to bench:* is re-validated here. Pure (no electron import) for tests.
import { existsSync, realpathSync } from 'node:fs'
import { isAbsolute, relative, resolve } from 'node:path'
import { DEFAULT_CANDIDATE_RULES, type CandidateRules } from '../core/benchmark/candidates'
import { WORKLOADS } from '../core/scoring/workloads'
import type { ExportConfig } from '../core/export/config'
import type { SessionRequest } from '../shared/bench-events'
import type { GenConfig, WorkloadId } from '../shared/bench-types'

/** True if p is dir itself's descendant (no `..` escape, same drive). */
export function isInside(dir: string, p: string): boolean {
  const rel = relative(resolve(dir), resolve(p))
  return rel !== '' && !rel.startsWith('..') && !isAbsolute(rel)
}

/** Own keys only: `in` would accept inherited names like "__proto__" or "toString" (W4 F8). */
export const isWorkloadId = (w: unknown): w is WorkloadId => typeof w === 'string' && Object.hasOwn(WORKLOADS, w)

const real = (p: string): string | null => { try { return realpathSync.native(p) } catch { return null } }

/** Lexical containment, and for an existing file also by real path: a symlink/junction inside a model root that
 *  points elsewhere must not pass (W4 F11). A missing file keeps the lexical check (it can't be loaded anyway). */
export function insideSomeRoot(p: string, roots: string[]): boolean {
  if (!roots.some((d) => isInside(d, p))) return false
  const rp = existsSync(p) ? real(p) : null
  if (!rp) return true
  return roots.some((d) => { const rd = real(d); return rd !== null && isInside(rd, rp) })
}

/** DB row ids from the renderer: positive safe integers only (throws otherwise). */
export function rowId(v: unknown): number {
  if (typeof v !== 'number' || !Number.isSafeInteger(v) || v <= 0) throw new Error(`bad id ${String(v)}`)
  return v
}

const D = DEFAULT_CANDIDATE_RULES
export const REQUIRED_CTX = [32768, 65536, 131072]
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
  if (!isWorkloadId(r.workload)) return { ok: false, error: 'unknown workload' }
  const modelIds = strings(r.modelIds)
  if (!modelIds?.length || modelIds.length > 20) return { ok: false, error: 'pick 1–20 models' }
  const outside = modelIds.find((p) => !insideSomeRoot(p, modelRoots))
  if (outside) return { ok: false, error: `model path not in a configured model dir: ${outside}` }
  const ladder = Array.isArray(r.ladder)
    ? [...new Set((r.ladder as unknown[]).filter((c): c is number => typeof c === 'number' && D.ctxLadder.includes(c)))].sort((a, b) => a - b)
    : []
  const reps = num(r.reps)
  // Auto (workload default) = omitted; otherwise one of the offered long-context targets.
  if (r.requiredContext !== undefined && r.requiredContext !== null && !REQUIRED_CTX.includes(r.requiredContext as number)) return { ok: false, error: 'required context must be 32K, 64K or 128K' }
  const minDec = num(r.minDecodeTps)
  if (r.minDecodeTps !== undefined && r.minDecodeTps !== null && (minDec === null || minDec < 0 || minDec > 1000)) return { ok: false, error: 'minimum decode speed must be 0–1000 t/s' }
  const rerun = strings(r.rerunConfigIds)
  return {
    ok: true,
    req: {
      workload: r.workload as WorkloadId,
      modelIds,
      ...(ladder.length ? { ladder } : {}),
      ...(typeof r.requiredContext === 'number' ? { requiredContext: r.requiredContext } : {}),
      ...(minDec !== null ? { minDecodeTps: minDec } : {}),
      ...(reps !== null ? { reps: Math.min(5, Math.max(1, Math.round(reps))) } : {}),
      ...(bool(r.runQuality) !== undefined ? { runQuality: bool(r.runQuality) } : {}),
      ...(bool(r.heavyMode) !== undefined ? { heavyMode: bool(r.heavyMode) } : {}),
      ...(bool(r.compareBackends) !== undefined ? { compareBackends: bool(r.compareBackends) } : {}),
      ...(bool(r.retryFailed) !== undefined ? { retryFailed: bool(r.retryFailed) } : {}),
      ...(rerun?.length ? { rerunConfigIds: rerun } : {}),
      ...(sanitizeRules(r.candidateRules) ? { candidateRules: sanitizeRules(r.candidateRules) } : {})
    }
  }
}

/** Renderer input → ExportConfig for serve:start. Every field becomes a llama-server argument, so only the value
 *  shapes toLoadConfig expects get through; the model path must be inside a configured root like a bench request. */
export function sanitizeServeConfig(raw: unknown, modelRoots: string[]): { ok: true; cfg: ExportConfig } | { ok: false; error: string } {
  if (!raw || typeof raw !== 'object') return { ok: false, error: 'bad config' }
  const r = raw as Record<string, unknown>
  if (typeof r.modelPath !== 'string' || !insideSomeRoot(r.modelPath, modelRoots)) return { ok: false, error: 'model path not in a configured model dir' }
  const backend = r.backend === 'hip' ? 'hip' : r.backend === 'prism' ? 'prism' : r.backend === 'vulkan' ? 'vulkan' : null
  if (!backend) return { ok: false, error: `unsupported backend ${String(r.backend)}` }
  const kvType = r.kvType === 'q8_0' ? 'q8_0' : r.kvType === 'f16' ? 'f16' : null
  if (!kvType) return { ok: false, error: `bad kvType ${String(r.kvType)}` }
  const device = r.device === null ? null : typeof r.device === 'string' && /^[A-Za-z]+\d+$/.test(r.device) ? r.device : undefined
  if (device === undefined) return { ok: false, error: `bad device ${String(r.device)}` }
  const ints: Record<string, number> = {}
  for (const k of ['ctx', 'gpuLayers', 'layers', 'threads', 'batch', 'ubatch']) {
    const v = num(r[k])
    if (v === null || v < 0 || !Number.isInteger(v)) return { ok: false, error: `bad ${k} ${String(r[k])}` }
    ints[k] = v
  }
  let gen: ExportConfig['gen']
  if (r.gen && typeof r.gen === 'object') {
    const g = r.gen as Record<string, unknown>
    const sm = (g.sampling && typeof g.sampling === 'object' ? g.sampling : {}) as Record<string, unknown>
    const t = num(sm.temperature)
    if (t === null) return { ok: false, error: 'bad gen.sampling.temperature' }
    const opt = (k: 'top_p' | 'top_k' | 'min_p') => { const v = num(sm[k]); return v === null ? {} : { [k]: v } }
    const kw = g.templateKwargs && typeof g.templateKwargs === 'object' && !Array.isArray(g.templateKwargs) ? (g.templateKwargs as Record<string, unknown>) : undefined
    // gen.config is display-only downstream (nothing in toLoadConfig reads it); sampling and kwargs are what get launched.
    gen = { config: g.config as GenConfig, sampling: { temperature: t, ...opt('top_p'), ...opt('top_k'), ...opt('min_p') }, ...(kw ? { templateKwargs: kw } : {}) }
  }
  return { ok: true, cfg: {
    sessionId: String(r.sessionId ?? ''), configId: String(r.configId ?? ''), modelPath: r.modelPath, modelName: String(r.modelName ?? ''),
    ctx: ints.ctx, gpuLayers: ints.gpuLayers, gpuLayersAll: r.gpuLayersAll === true, layers: ints.layers, threads: ints.threads, batch: ints.batch, ubatch: ints.ubatch,
    flashAttn: r.flashAttn === true, kvType, device, kvOffload: r.kvOffload !== false, mmap: r.mmap !== false, backend,
    workload: isWorkloadId(r.workload) ? r.workload : 'fast_assistant', ...(gen ? { gen } : {})
  } }
}
