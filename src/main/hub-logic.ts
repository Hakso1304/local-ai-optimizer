// Pure parts of the hub IPC (no Electron): progress/ETA, user-facing error text, disk and destination checks.
import { existsSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve } from 'node:path'
import { HubError } from '../core/hub/hf'
import type { HubFailKind, HubProgress } from '../shared/hub-types'

export function progressInfo(repoId: string, path: string, bytes: number, total: number | null, bytesPerSec: number): HubProgress {
  const known = total !== null && total > 0
  return {
    repoId, path, bytes, total, bytesPerSec,
    pct: known ? Math.min(100, (bytes / total) * 100) : null,
    etaSec: known && bytesPerSec > 0 ? Math.max(0, (total - bytes) / bytesPerSec) : null
  }
}

/** HubError (or anything) → a message the user can act on. */
export function userMessage(e: unknown): { kind: HubFailKind; error: string } {
  if (!(e instanceof HubError)) return { kind: 'network', error: (e as Error)?.message ?? String(e) }
  const msg: Record<string, string> = {
    auth_required: 'Sign in with a Hugging Face access token (read access) to download this file.',
    gated_accept_license: e.message, // already names the model page URL
    not_found: 'That repository or file does not exist on Hugging Face (check the name or revision).',
    network: `Network problem: ${e.message}. Resume continues from where it stopped.`,
    cancelled: 'Paused. Resume continues from the partial file.',
    size_mismatch: `${e.message}. Resume to continue.`,
    sha256_mismatch: `${e.message}. Download it again.`,
    bad_path: e.message,
    http_error: `Hugging Face returned HTTP ${e.status ?? '?'}. Try again later.`
  }
  return { kind: e.kind, error: msg[e.kind] ?? e.message }
}

/** destDir must be one of the configured model dirs (or inside one) — never an arbitrary renderer-chosen path. */
export function isAllowedDest(destDir: string, modelDirs: string[]): boolean {
  const d = resolve(destDir)
  return modelDirs.some((m) => { const r = relative(resolve(m), d); return r === '' || (!r.startsWith('..') && !isAbsolute(r)) })
}

/** Nearest existing directory at or above dir (for the free-space check before destDir is created); null when
 *  not even the drive root exists (e.g. D:\ on a PC without a D: drive). */
export function nearestExisting(dir: string, exists: (p: string) => boolean = existsSync): string | null {
  for (let d = resolve(dir); ; d = dirname(d)) {
    if (exists(d)) return d
    if (dirname(d) === d) return null
  }
}

/** Bytes still to fetch (resume-aware) plus a 1 GiB margin must fit in the free space. */
export function diskCheck(freeBytes: number, fileBytes: number, partBytes: number): { ok: true } | { ok: false; error: string } {
  const need = Math.max(0, fileBytes - partBytes) + 1024 ** 3
  if (freeBytes >= need) return { ok: true }
  const g = (b: number) => `${(b / 1024 ** 3).toFixed(1)} GiB`
  return { ok: false, error: `Not enough disk space: need ${g(need)} (file ${g(fileBytes - partBytes)} + 1 GiB margin), ${g(freeBytes)} free` }
}
