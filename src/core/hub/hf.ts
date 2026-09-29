// Hugging Face Hub: search GGUF repos, list their GGUF files, resumable verified download. Pure Node (fetch + fs),
// no Electron; the token is passed in per call and only ever sent to huggingface.co (or the configured base host).
// Endpoints [S: huggingface.co/docs/hub/api]: /api/models (search), /api/models/{repo}/tree/{rev} (paginated via
// Link rel="next"), /api/whoami-v2, /{repo}/resolve/{rev}/{path} (302 → CDN).
import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream, existsSync, lstatSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { dirname, resolve, sep } from 'node:path'
import { pipeline } from 'node:stream/promises'

export const HF_BASE = 'https://huggingface.co'

export type HubErrorKind =
  | 'auth_required' | 'gated_accept_license' | 'not_found' | 'http_error' | 'network' | 'size_mismatch' | 'sha256_mismatch' | 'cancelled' | 'bad_path' | 'range_mismatch' | 'disk_error'

export class HubError extends Error {
  constructor(readonly kind: HubErrorKind, message: string, readonly status?: number, readonly url?: string) {
    super(message)
    this.name = 'HubError'
  }
}

interface Common { token?: string; baseUrl?: string }

const auth = (token?: string): Record<string, string> => (token ? { authorization: `Bearer ${token}` } : {})

/** Exact origin allowlist, checked before EVERY request that could carry the token (each page, each redirect hop):
 *  https huggingface.co / *.huggingface.co on the default port, or the exact configured base origin (tests: local http). */
export function tokenAllowed(url: string | URL, base: string = HF_BASE): boolean {
  const u = new URL(url)
  if (u.origin === new URL(base).origin) return true
  return u.protocol === 'https:' && !u.port && (u.hostname === 'huggingface.co' || u.hostname.endsWith('.huggingface.co'))
}

const RESERVED = /^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³]|conin\$|conout\$)(\..*)?$/i
/** A repo path segment that is safe as a Windows file name: no ADS `:`, wildcard/redirect chars, controls, trailing dot/space, device names. */
export function safeSegment(s: string): boolean {
  // eslint-disable-next-line no-control-regex
  return s !== '' && s !== '.' && s !== '..' && !/[<>:"|?*\\\x00-\x1f]/.test(s) && !/[. ]$/.test(s) && !RESERVED.test(s)
}

const isLink = (p: string): boolean => {
  try { return lstatSync(p).isSymbolicLink() } catch (e) { if ((e as NodeJS.ErrnoException).code === 'ENOENT') return false; throw e }
}

/** 401/403/404 → user-actionable kinds; repoUrl lets 403 point at the license page. */
function statusError(status: number, url: string, repoUrl?: string): HubError {
  if (status === 401) return new HubError('auth_required', 'Hugging Face sign-in required: add an access token (Settings → Access Tokens on huggingface.co)', status, url)
  if (status === 403) return new HubError('gated_accept_license', `Access denied: this model is gated — open ${repoUrl ?? url}, accept its license with the account that owns the token, then retry`, status, url)
  if (status === 404) return new HubError('not_found', `Not found on Hugging Face: ${url}`, status, url)
  return new HubError('http_error', `HTTP ${status} from ${url}`, status, url)
}

async function getJson<T>(url: string, token?: string, signal?: AbortSignal, base = HF_BASE): Promise<{ body: T; res: Response }> {
  let res: Response
  try {
    res = await fetch(url, { headers: { accept: 'application/json', ...(tokenAllowed(url, base) ? auth(token) : {}) }, signal })
  } catch (e) {
    throw new HubError(signal?.aborted ? 'cancelled' : 'network', `request failed: ${(e as Error).message}`, undefined, url)
  }
  if (!res.ok) throw statusError(res.status, url)
  return { body: (await res.json()) as T, res }
}

export interface HfModel { id: string; downloads: number; likes: number; gated: boolean | 'auto' | 'manual'; lastModified: string | null }

/** Empty query = most-downloaded GGUF repos overall. */
export async function searchModels(query: string, opts: Common & { limit?: number; pipelineTag?: string } = {}): Promise<HfModel[]> {
  const u = new URL('/api/models', opts.baseUrl ?? HF_BASE)
  if (query) u.searchParams.set('search', query)
  u.searchParams.set('filter', 'gguf')
  if (opts.pipelineTag) u.searchParams.set('pipeline_tag', opts.pipelineTag)
  u.searchParams.set('sort', 'downloads')
  u.searchParams.set('direction', '-1')
  u.searchParams.set('limit', String(Math.min(100, Math.max(1, opts.limit ?? 20))))
  for (const f of ['downloads', 'likes', 'gated', 'lastModified']) u.searchParams.append('expand[]', f)
  const { body } = await getJson<{ id: string; downloads?: number; likes?: number; gated?: boolean | 'auto' | 'manual'; lastModified?: string }[]>(u.href, opts.token, undefined, opts.baseUrl)
  return body.map((m) => ({ id: m.id, downloads: m.downloads ?? 0, likes: m.likes ?? 0, gated: m.gated ?? false, lastModified: m.lastModified ?? null }))
}

export interface HfGgufFile {
  path: string
  sizeBytes: number
  /** LFS sha256 (hex) when the file is stored in LFS; null otherwise. */
  sha256: string | null
  quant: string | null
  /** "model-00001-of-00003.gguf" → {index: 1, count: 3}. */
  shard: { index: number; count: number } | null
}

export const quantFromName = (name: string): string | null =>
  /(TQ\d_\d|I?Q\d_[A-Z0-9]+(?:_[A-Z0-9]+)?|BF16|F16|F32|MXFP4)/i.exec(name.split('/').pop()!.replace(/-\d{5}-of-\d{5}\.gguf$/i, ''))?.[1]?.toUpperCase() ?? null

export async function listGgufFiles(repoId: string, opts: Common & { revision?: string } = {}): Promise<HfGgufFile[]> {
  const base = opts.baseUrl ?? HF_BASE
  let url: string | null = `${base}/api/models/${repoId}/tree/${encodeURIComponent(opts.revision ?? 'main')}?recursive=true`
  const out: HfGgufFile[] = []
  for (let page = 0; url && page < 100; page++) {
    const got: { body: { type: string; path: string; size: number; lfs?: { oid: string; size: number } }[]; res: Response } = await getJson(url, opts.token, undefined, base)
    const { body, res } = got
    for (const e of body) {
      if (e.type !== 'file' || !e.path.toLowerCase().endsWith('.gguf')) continue
      const s = /-(\d{5})-of-(\d{5})\.gguf$/i.exec(e.path)
      out.push({ path: e.path, sizeBytes: e.lfs?.size ?? e.size, sha256: e.lfs?.oid ?? null, quant: quantFromName(e.path), shard: s ? { index: Number(s[1]), count: Number(s[2]) } : null })
    }
    const next: string | null = /<([^>]+)>;\s*rel="next"/.exec(res.headers.get('link') ?? '')?.[1] ?? null
    url = next ? new URL(next, base).href : null
  }
  return out.sort((a, b) => (a.path < b.path ? -1 : 1))
}

export async function whoami(token: string, opts: { baseUrl?: string } = {}): Promise<{ name: string; fullname?: string; type?: string } | { error: string }> {
  try {
    const { body } = await getJson<{ name: string; fullname?: string; type?: string }>(`${opts.baseUrl ?? HF_BASE}/api/whoami-v2`, token, undefined, opts.baseUrl)
    return { name: body.name, fullname: body.fullname, type: body.type }
  } catch (e) {
    return { error: e instanceof HubError && e.kind === 'auth_required' ? 'invalid or expired token' : (e as Error).message }
  }
}

export interface DownloadOpts extends Common {
  repoId: string
  path: string
  destDir: string
  revision?: string
  /** From listGgufFiles; enables verification. */
  sha256?: string | null
  sizeBytes?: number | null
  signal?: AbortSignal
  onProgress?: (bytes: number, total: number | null, bytesPerSec: number) => void
}

/** Resumable (Range on `<dest>.part`), redirect-following download with streaming sha256 and an atomic rename.
 *  Cancel leaves the .part for a later resume; a checksum mismatch deletes it. */
export async function downloadFile(o: DownloadOpts): Promise<{ filePath: string; sizeBytes: number; sha256Verified: boolean | null }> {
  const base = o.baseUrl ?? HF_BASE
  const segs = o.path.split('/')
  if (!o.path || !segs.every(safeSegment) || /^[a-z]:|^[\\/]/i.test(o.path)) throw new HubError('bad_path', `refusing path ${o.path}`)
  const root = resolve(o.destDir)
  const dest = resolve(root, ...segs)
  if (!dest.startsWith(root + sep)) throw new HubError('bad_path', `refusing path ${o.path}`)
  const part = `${dest}.part`
  const meta = `${part}.json` // identity of the partial: resume only the same repo/revision/path/size/hash
  mkdirSync(dirname(dest), { recursive: true })
  // Refuse links/junctions from the .part's parent up to destDir, and on the .part/sidecar themselves.
  // ponytail: check-then-open, not O_NOFOLLOW (unavailable on Windows); 'wx' closes the fresh-file case.
  const linkCheck = () => {
    for (let d = dirname(dest); d.startsWith(root); d = dirname(d)) {
      if (isLink(d)) throw new HubError('bad_path', `refusing linked directory ${d}`)
      if (d === root) break
    }
    for (const f of [part, meta]) if (isLink(f)) throw new HubError('bad_path', `refusing linked partial ${f}`)
  }
  linkCheck()
  const repoUrl = `${base}/${o.repoId}`
  const identity = JSON.stringify({ repoId: o.repoId, revision: o.revision ?? 'main', path: o.path, sizeBytes: o.sizeBytes ?? null, sha256: o.sha256?.toLowerCase() ?? null })
  const restart = () => { rmSync(part, { force: true }); rmSync(meta, { force: true }) }

  let offset = existsSync(part) ? statSync(part).size : 0
  if (offset > 0) {
    let same = false
    try { same = readFileSync(meta, 'utf8') === identity } catch { /* no sidecar → unknown origin, don't resume */ }
    if (!same || (o.sizeBytes && offset > o.sizeBytes)) { restart(); offset = 0 }
  }

  // Follow redirects by hand so the token never leaves huggingface.co (CDN URLs are pre-signed).
  let url = new URL(`${repoUrl}/resolve/${encodeURIComponent(o.revision ?? 'main')}/${segs.map(encodeURIComponent).join('/')}`)
  let res: Response | null = null
  for (let hop = 0; hop < 10; hop++) {
    const headers: Record<string, string> = { ...(tokenAllowed(url, base) ? auth(o.token) : {}), ...(offset > 0 ? { range: `bytes=${offset}-` } : {}) }
    try {
      res = await fetch(url, { headers, redirect: 'manual', signal: o.signal })
    } catch (e) {
      throw new HubError(o.signal?.aborted ? 'cancelled' : 'network', `download failed: ${(e as Error).message}`, undefined, url.href)
    }
    const loc = res.headers.get('location')
    if (res.status >= 300 && res.status < 400 && loc) { await res.body?.cancel(); url = new URL(loc, url); continue }
    break
  }
  if (!res) throw new HubError('network', 'no response')
  if (res.status === 416 && o.sizeBytes && offset === o.sizeBytes) {
    await res.body?.cancel() // .part already complete
  } else if (res.status === 416) {
    await res.body?.cancel(); restart()
    return downloadFile(o) // stale .part the server can't continue: restart once from 0
  } else if (!res.ok) {
    await res.body?.cancel()
    throw statusError(res.status, url.href, repoUrl)
  }
  if (res.status === 200 && offset > 0) { restart(); offset = 0 } // server ignored Range → full body

  const cr = /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(res.headers.get('content-range') ?? '')
  const crTotal = cr && cr[3] !== '*' ? Number(cr[3]) : null
  if (res.status === 206 && (!cr || Number(cr[1]) !== offset || (o.sizeBytes && crTotal !== null && crTotal !== o.sizeBytes))) {
    await res.body?.cancel(); restart()
    throw new HubError('range_mismatch', `server answered ${res.headers.get('content-range') ?? 'no Content-Range'} for offset ${offset}; partial discarded, retry starts over`, 206, url.href)
  }
  const len = Number(res.headers.get('content-length'))
  const total = o.sizeBytes ?? crTotal ?? (res.status === 200 && len > 0 ? len : res.status === 206 && len > 0 ? offset + len : null)

  const hash = o.sha256 ? createHash('sha256') : null
  if (hash && offset > 0) for await (const chunk of createReadStream(part)) hash.update(chunk as Buffer) // prefix of a resumed file

  let bytes = offset
  if (res.status !== 416 && res.body) {
    if (offset === 0) { restart(); writeFileSync(meta, identity, { flag: 'wx' }) }
    linkCheck()
    // 'wx' fails on anything (file or link) placed at .part since restart(); 'r+' + start never creates one.
    const out = createWriteStream(part, offset > 0 ? { flags: 'r+', start: offset } : { flags: 'wx' })
    const t0 = Date.now()
    let lastTick = 0
    let over = false
    async function* tap(src: AsyncIterable<Uint8Array>): AsyncGenerator<Uint8Array> {
      for await (const b of src) {
        if (total !== null && bytes + b.length > total) { over = true; throw new Error(`server sent more than the expected ${total} bytes`) }
        hash?.update(b)
        bytes += b.length
        yield b
        const now = Date.now()
        if (now - lastTick >= 250) { lastTick = now; o.onProgress?.(bytes, total, ((bytes - offset) * 1000) / Math.max(1, now - t0)) }
      }
    }
    try {
      // pipeline: write errors (ENOSPC/EACCES) reject here instead of crashing main; backpressure and abort handled.
      await pipeline(res.body as unknown as AsyncIterable<Uint8Array>, tap, out, o.signal ? { signal: o.signal } : {})
    } catch (e) {
      const err = e as NodeJS.ErrnoException
      if (over) { restart(); throw new HubError('size_mismatch', `${err.message}; partial discarded`, undefined, url.href) }
      if (o.signal?.aborted) throw new HubError('cancelled', `download paused at ${bytes} bytes (partial kept for resume)`, undefined, url.href)
      if (err.syscall) throw new HubError('disk_error', `cannot write ${part}: ${err.message} (partial kept)`, undefined, url.href)
      throw new HubError('network', `download interrupted at ${bytes} bytes (partial kept for resume): ${err.message}`, undefined, url.href)
    }
    o.onProgress?.(bytes, total, ((bytes - offset) * 1000) / Math.max(1, Date.now() - t0))
  }

  if (total !== null && bytes !== total) throw new HubError('size_mismatch', `expected ${total} bytes, got ${bytes} (partial kept; retry resumes)`, undefined, url.href)
  let sha256Verified: boolean | null = null
  if (hash) {
    const got = hash.digest('hex')
    if (got !== o.sha256!.toLowerCase()) { restart(); throw new HubError('sha256_mismatch', `sha256 mismatch for ${o.path}: expected ${o.sha256}, got ${got} (download deleted)`, undefined, url.href) }
    sha256Verified = true
  }
  renameSync(part, dest)
  rmSync(meta, { force: true })
  return { filePath: dest, sizeBytes: bytes, sha256Verified }
}
