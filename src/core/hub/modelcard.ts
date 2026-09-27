// Model-card sampling defaults (generation_config.json) and the per-model sidecar cache <file>.gguf.meta.json.
// Never blocks model enumeration: callers fetch lazily, 5 s timeout, and read the cache synchronously.
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import type { GenKnobs } from '../../shared/bench-types'

const HF = 'https://huggingface.co'

/** <model>.meta.json — written after a Hub download (or when the user links a repo). */
export interface ModelSidecar {
  repoId?: string
  revision?: string
  /** generation_config.json values (source 'model-card'); absent = not fetched yet or the repo has none. */
  generation?: GenKnobs['recommended']
  fetchedAt?: string
  fetchError?: string
}

export const sidecarPath = (modelPath: string) => `${modelPath}.meta.json`

export function readSidecar(modelPath: string): ModelSidecar | null {
  const p = sidecarPath(modelPath)
  if (!existsSync(p)) return null
  try { return JSON.parse(readFileSync(p, 'utf8')) as ModelSidecar } catch { return null }
}

export function writeSidecar(modelPath: string, patch: ModelSidecar): ModelSidecar {
  const next = { ...readSidecar(modelPath), ...patch }
  writeFileSync(sidecarPath(modelPath), JSON.stringify(next, null, 2))
  return next
}

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)

/** Parse generation_config.json → our sampling fields (only the ones present and numeric). */
export function parseGenerationConfig(j: unknown): NonNullable<GenKnobs['recommended']> | null {
  if (!j || typeof j !== 'object') return null
  const g = j as Record<string, unknown>
  const out = { temperature: num(g.temperature), topP: num(g.top_p), topK: num(g.top_k), minP: num(g.min_p) }
  const kept = Object.fromEntries(Object.entries(out).filter(([, v]) => v !== undefined))
  return Object.keys(kept).length ? kept : null
}

/** GET <repo>/resolve/<rev>/generation_config.json. Public repos need no token; the token is sent to huggingface.co only.
 *  null = the repo has no such file (404); throws on network/timeout/auth so the caller can record why. */
export async function fetchGenerationConfig(repoId: string, o: { token?: string; revision?: string; timeoutMs?: number } = {}): Promise<NonNullable<GenKnobs['recommended']> | null> {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repoId)) throw new Error(`bad repository id ${repoId}`)
  const url = `${HF}/${repoId}/resolve/${encodeURIComponent(o.revision ?? 'main')}/generation_config.json`
  const res = await fetch(url, { headers: o.token ? { authorization: `Bearer ${o.token}` } : {}, signal: AbortSignal.timeout(o.timeoutMs ?? 5_000) })
  if (res.status === 404) return null
  if (res.status === 401 || res.status === 403) throw new Error(`HTTP ${res.status}: gated repository — sign in and accept the license`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return parseGenerationConfig(await res.json())
}
