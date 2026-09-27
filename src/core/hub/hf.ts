// Hugging Face Hub: search GGUF repos, list their GGUF files, resumable verified download. Pure Node (fetch + fs),
// no Electron; the token is passed in per call and only ever sent to huggingface.co (or the configured base host).
// Endpoints [S: huggingface.co/docs/hub/api]: /api/models (search), /api/models/{repo}/tree/{rev} (paginated via
// Link rel="next"), /api/whoami-v2, /{repo}/resolve/{rev}/{path} (302 → CDN).
import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs'
import { dirname, resolve, sep } from 'node:path'

export const HF_BASE = 'https://huggingface.co'

export type HubErrorKind =
  | 'auth_required' | 'gated_accept_license' | 'not_found' | 'http_error' | 'network' | 'size_mismatch' | 'sha256_mismatch' | 'cancelled' | 'bad_path'

export class HubError extends Error {
  constructor(readonly kind: HubErrorKind, message: string, readonly status?: number, readonly url?: string) {
    super(message)
    this.name = 'HubError'
  }
}

interface Common { token?: string; baseUrl?: string }

const auth = (token?: string): Record<string, string> => (token ? { authorization: `Bearer ${token}` } : {})

/** 401/403/404 → user-actionable kinds; repoUrl lets 403 point at the license page. */
function statusError(status: number, url: string, repoUrl?: string): HubError {
  if (status === 401) return new HubError('auth_required', 'Hugging Face sign-in required: add an access token (Settings → Access Tokens on huggingface.co)', status, url)
  if (status === 403) return new HubError('gated_accept_license', `Access denied: this model is gated — open ${repoUrl ?? url}, accept its license with the account that owns the token, then retry`, status, url)
  if (status === 404) return new HubError('not_found', `Not found on Hugging Face: ${url}`, status, url)
  return new HubError('http_error', `HTTP ${status} from ${url}`, status, url)
}

async function getJson<T>(url: string, token?: string, signal?: AbortSignal): Promise<{ body: T; res: Response }> {
  let res: Response
  try {
    res = await fetch(url, { headers: { accept: 'application/json', ...auth(token) }, signal })
  } catch (e) {
    throw new HubError(signal?.aborted ? 'cancelled' : 'network', `request failed: ${(e as Error).message}`, undefined, url)
  }
  if (!res.ok) throw statusError(res.status, url)
  return { body: (await res.json()) as T, res }
}

export interface HfModel { id: string; downloads: number; likes: number; gated: boolean | 'auto' | 'manual'; lastModified: string | null }

export async function searchModels(query: string, opts: Common & { limit?: number } = {}): Promise<HfModel[]> {
  const u = new URL('/api/models', opts.baseUrl ?? HF_BASE)
  u.searchParams.set('search', query)
  u.searchParams.set('filter', 'gguf')
  u.searchParams.set('sort', 'downloads')
  u.searchParams.set('direction', '-1')
  u.searchParams.set('limit', String(Math.min(100, Math.max(1, opts.limit ?? 20))))
  for (const f of ['downloads', 'likes', 'gated', 'lastModified']) u.searchParams.append('expand[]', f)
  const { body } = await getJson<{ id: string; downloads?: number; likes?: number; gated?: boolean | 'auto' | 'manual'; lastModified?: string }[]>(u.href, opts.token)
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
  /(I?Q\d_[A-Z0-9]+(?:_[A-Z0-9]+)?|BF16|F16|F32|MXFP4)/i.exec(name.split('/').pop()!.replace(/-\d{5}-of-\d{5}\.gguf$/i, ''))?.[1]?.toUpperCase() ?? null

export async function listGgufFiles(repoId: string, opts: Common & { revision?: string } = {}): Promise<HfGgufFile[]> {
  const base = opts.baseUrl ?? HF_BASE
  let url: string | null = `${base}/api/models/${repoId}/tree/${encodeURIComponent(opts.revision ?? 'main')}?recursive=true`
  const out: HfGgufFile[] = []
  for (let page = 0; url && page < 100; page++) {
    const got: { body: { type: string; path: string; size: number; lfs?: { oid: string; size: number } }[]; res: Response } = await getJson(url, opts.token)
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
    const { body } = await getJson<{ name: string; fullname?: string; type?: string }>(`${opts.baseUrl ?? HF_BASE}/api/whoami-v2`, token)
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
  if (!o.path || segs.some((s) => s === '' || s === '.' || s === '..') || /^[a-z]:|^[\\/]/i.test(o.path)) throw new HubError('bad_path', `refusing path ${o.path}`)
  const dest = resolve(o.destDir, ...segs)
  if (!dest.startsWith(resolve(o.destDir) + sep)) throw new HubError('bad_path', `refusing path ${o.path}`)
  const part = `${dest}.part`
  mkdirSync(dirname(dest), { recursive: true })
  const repoUrl = `${base}/${o.repoId}`
  const trusted = (u: URL) => u.host === new URL(base).host || u.hostname === 'huggingface.co' || u.hostname.endsWith('.huggingface.co')

  let offset = existsSync(part) ? statSync(part).size : 0
  if (o.sizeBytes && offset > o.sizeBytes) { rmSync(part); offset = 0 }

  // Follow redirects by hand so the token never leaves huggingface.co (CDN URLs are pre-signed).
  let url = new URL(`${repoUrl}/resolve/${encodeURIComponent(o.revision ?? 'main')}/${segs.map(encodeURIComponent).join('/')}`)
  let res: Response | null = null
  for (let hop = 0; hop < 10; hop++) {
    const headers: Record<string, string> = { ...(trusted(url) ? auth(o.token) : {}), ...(offset > 0 ? { range: `bytes=${offset}-` } : {}) }
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
    await res.body?.cancel(); rmSync(part, { force: true })
    return downloadFile(o) // stale .part the server can't continue: restart once from 0
  } else if (!res.ok) {
    await res.body?.cancel()
    throw statusError(res.status, url.href, repoUrl)
  }
  if (res.status === 200 && offset > 0) { rmSync(part); offset = 0 } // server ignored Range → full body

  const range = /\/(\d+)$/.exec(res.headers.get('content-range') ?? '')
  const len = Number(res.headers.get('content-length'))
  const total = o.sizeBytes ?? (range ? Number(range[1]) : res.status === 200 && len > 0 ? len : res.status === 206 && len > 0 ? offset + len : null)

  const hash = o.sha256 ? createHash('sha256') : null
  if (hash && offset > 0) for await (const chunk of createReadStream(part)) hash.update(chunk as Buffer) // prefix of a resumed file

  let bytes = offset
  if (res.status !== 416 && res.body) {
    const out = createWriteStream(part, { flags: offset > 0 ? 'a' : 'w' })
    const t0 = Date.now()
    let lastTick = 0
    try {
      for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
        const b = chunk
        if (!out.write(b)) await new Promise<void>((r) => out.once('drain', () => r()))
        hash?.update(b)
        bytes += b.length
        const now = Date.now()
        if (now - lastTick >= 250) { lastTick = now; o.onProgress?.(bytes, total, ((bytes - offset) * 1000) / Math.max(1, now - t0)) }
      }
    } catch (e) {
      throw new HubError(o.signal?.aborted ? 'cancelled' : 'network', `download interrupted at ${bytes} bytes (partial kept for resume): ${(e as Error).message}`, undefined, url.href)
    } finally {
      await new Promise<void>((r) => out.end(r))
    }
    o.onProgress?.(bytes, total, ((bytes - offset) * 1000) / Math.max(1, Date.now() - t0))
  }

  if (total !== null && bytes !== total) throw new HubError('size_mismatch', `expected ${total} bytes, got ${bytes} (partial kept; retry resumes)`, undefined, url.href)
  let sha256Verified: boolean | null = null
  if (hash) {
    const got = hash.digest('hex')
    if (got !== o.sha256!.toLowerCase()) { rmSync(part, { force: true }); throw new HubError('sha256_mismatch', `sha256 mismatch for ${o.path}: expected ${o.sha256}, got ${got} (download deleted)`, undefined, url.href) }
    sha256Verified = true
  }
  renameSync(part, dest)
  return { filePath: dest, sizeBytes: bytes, sha256Verified }
}
