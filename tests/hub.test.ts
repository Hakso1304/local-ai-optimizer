import { createHash } from 'node:crypto'
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { downloadFile, HubError, listGgufFiles, quantFromName, searchModels, whoami } from '../src/core/hub/hf'

// Two local servers: "hub" (API + resolve, redirects) and "cdn" (a different host:port that must never see the token).
const content = Buffer.from(Array.from({ length: 300_000 }, (_, i) => (i * 31 + 7) % 251))
const sha = createHash('sha256').update(content).digest('hex')
const seen: { cdnAuth: (string | undefined)[]; ranges: (string | undefined)[]; hubAuth: (string | undefined)[] } = { cdnAuth: [], ranges: [], hubAuth: [] }
let hub: Server, cdn: Server, base = '', cdnBase = ''
const dir = mkdtempSync(join(tmpdir(), 'lao-hub-'))

function serveRange(req: IncomingMessage, res: ServerResponse, body: Buffer, slow = false) {
  const m = /bytes=(\d+)-/.exec(req.headers.range ?? '')
  const start = m ? Number(m[1]) : 0
  if (start >= body.length) { res.writeHead(416, { 'content-range': `bytes */${body.length}` }); return res.end() }
  const chunk = body.subarray(start)
  res.writeHead(m ? 206 : 200, { 'content-length': chunk.length, ...(m ? { 'content-range': `bytes ${start}-${body.length - 1}/${body.length}` } : {}) })
  if (!slow) return res.end(chunk)
  res.write(chunk.subarray(0, 65536)) // then stall: the client cancels mid-stream
}

beforeAll(async () => {
  cdn = createServer((req, res) => {
    seen.cdnAuth.push(req.headers.authorization); seen.ranges.push(req.headers.range)
    if (req.url === '/blob') return serveRange(req, res, content)
    if (req.url === '/slow') return serveRange(req, res, content, true)
    res.writeHead(404).end()
  })
  await new Promise<void>((r) => cdn.listen(0, '127.0.0.1', r))
  cdnBase = `http://127.0.0.1:${(cdn.address() as AddressInfo).port}`
  hub = createServer((req, res) => {
    seen.hubAuth.push(req.headers.authorization)
    const u = new URL(req.url!, 'http://x')
    const json = (code: number, body: unknown, headers: Record<string, string> = {}) => { res.writeHead(code, { 'content-type': 'application/json', ...headers }); res.end(JSON.stringify(body)) }
    if (u.pathname === '/api/models') {
      expect(u.searchParams.get('filter')).toBe('gguf')
      return json(200, [{ id: 'org/repo-GGUF', downloads: 10, likes: 2, gated: false, lastModified: '2026-09-01T00:00:00.000Z' }, { id: 'org/gated-GGUF', gated: 'manual' }])
    }
    if (u.pathname === '/api/models/org/repo-GGUF/tree/main') {
      if (u.searchParams.get('cursor') !== '2') return json(200, [
        { type: 'file', path: 'README.md', size: 10 },
        { type: 'file', path: 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', size: 134, lfs: { oid: sha, size: content.length } }
      ], { link: `<${base}/api/models/org/repo-GGUF/tree/main?recursive=true&cursor=2>; rel="next"` })
      return json(200, [{ type: 'directory', path: 'big' }, { type: 'file', path: 'big/model-IQ3_XXS-00001-of-00002.gguf', size: 5, lfs: { oid: 'ab', size: 5 } }])
    }
    if (u.pathname === '/api/whoami-v2') return req.headers.authorization === 'Bearer good' ? json(200, { name: 'alice', type: 'user' }) : json(401, { error: 'Invalid' })
    if (u.pathname === '/org/repo-GGUF/resolve/main/model.gguf') { res.writeHead(302, { location: `${cdnBase}/blob` }); return res.end() }
    if (u.pathname === '/org/repo-GGUF/resolve/main/slow.gguf') { res.writeHead(302, { location: `${cdnBase}/slow` }); return res.end() }
    if (u.pathname === '/org/gated-GGUF/resolve/main/x.gguf') return json(403, { error: 'gated' })
    if (u.pathname === '/org/private/resolve/main/x.gguf') return json(401, { error: 'auth' })
    json(404, { error: 'nope' })
  })
  await new Promise<void>((r) => hub.listen(0, '127.0.0.1', r))
  base = `http://127.0.0.1:${(hub.address() as AddressInfo).port}`
})
afterAll(() => { hub.close(); cdn.close(); rmSync(dir, { recursive: true, force: true }) })

describe('hub API', () => {
  it('searchModels maps fields and defaults missing ones', async () => {
    expect(await searchModels('llama', { baseUrl: base, limit: 5 })).toEqual([
      { id: 'org/repo-GGUF', downloads: 10, likes: 2, gated: false, lastModified: '2026-09-01T00:00:00.000Z', createdAt: null },
      { id: 'org/gated-GGUF', downloads: 0, likes: 0, gated: 'manual', lastModified: null, createdAt: null }
    ])
  })
  it('listGgufFiles follows Link pagination, keeps .gguf only, uses lfs oid/size, parses quant + shards', async () => {
    expect(await listGgufFiles('org/repo-GGUF', { baseUrl: base })).toEqual([
      { path: 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', sizeBytes: content.length, sha256: sha, quant: 'Q4_K_M', shard: null },
      { path: 'big/model-IQ3_XXS-00001-of-00002.gguf', sizeBytes: 5, sha256: 'ab', quant: 'IQ3_XXS', shard: { index: 1, count: 2 } }
    ])
  })
  it('quantFromName', () => {
    expect(['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', 'x-Q8_0.gguf', 'model-f16.gguf', 'plain.gguf'].map(quantFromName)).toEqual(['Q4_K_M', 'Q8_0', 'F16', null])
  })
  it('whoami: name for a good token, error otherwise', async () => {
    expect(await whoami('good', { baseUrl: base })).toEqual({ name: 'alice', fullname: undefined, type: 'user' })
    expect(await whoami('bad', { baseUrl: base })).toEqual({ error: 'invalid or expired token' })
  })
})

describe('downloadFile', () => {
  const dl = (o: Partial<Parameters<typeof downloadFile>[0]> = {}) =>
    downloadFile({ baseUrl: base, repoId: 'org/repo-GGUF', path: 'model.gguf', destDir: dir, token: 'secret', sha256: sha, sizeBytes: content.length, ...o })

  it('follows the redirect; token goes to the hub only, never to the CDN host; sha256 verified; atomic rename', async () => {
    seen.cdnAuth.length = 0; seen.hubAuth.length = 0
    const progress: number[] = []
    const r = await dl({ onProgress: (b) => progress.push(b) })
    expect(r).toEqual({ filePath: join(dir, 'model.gguf'), sizeBytes: content.length, sha256Verified: true })
    expect(readFileSync(r.filePath).equals(content)).toBe(true)
    expect(existsSync(`${r.filePath}.part`)).toBe(false)
    expect(seen.hubAuth.at(-1)).toBe('Bearer secret')
    expect(seen.cdnAuth).toEqual([undefined])
    expect(progress.at(-1)).toBe(content.length)
    rmSync(r.filePath)
  })

  it('resumes from an existing .part with a Range request', async () => {
    const id = { repoId: 'org/repo-GGUF', revision: 'main', path: 'model.gguf', sizeBytes: content.length, sha256: sha }
    writeFileSync(join(dir, 'model.gguf.part'), content.subarray(0, 100_000))
    writeFileSync(join(dir, 'model.gguf.part.json'), JSON.stringify(id))
    seen.ranges.length = 0
    const r = await dl()
    expect(seen.ranges).toEqual(['bytes=100000-'])
    expect(readFileSync(r.filePath).equals(content)).toBe(true)
    expect(r.sha256Verified).toBe(true) // hash covers the resumed prefix too
    expect(existsSync(join(dir, 'model.gguf.part.json'))).toBe(false)
    rmSync(r.filePath)
    // F9: a partial from another repo (or with no sidecar) is never resumed
    writeFileSync(join(dir, 'model.gguf.part'), content.subarray(0, 100_000))
    writeFileSync(join(dir, 'model.gguf.part.json'), JSON.stringify({ ...id, repoId: 'other/repo' }))
    seen.ranges.length = 0
    rmSync((await dl()).filePath)
    expect(seen.ranges).toEqual([undefined])
  })

  it('F10: rejects ADS, reserved device names and trailing dot/space in any segment', async () => {
    for (const path of ['a.gguf:x', 'CON', 'sub/nul.gguf', 'lpt1.txt', 'dir./m.gguf', 'm.gguf ', 'a|b.gguf'])
      await expect(dl({ path })).rejects.toMatchObject({ kind: 'bad_path' })
  })

  it('F3: token only to the exact https hub origins (or the configured base)', async () => {
    const { tokenAllowed } = await import('../src/core/hub/hf')
    expect(tokenAllowed('https://huggingface.co/x')).toBe(true)
    expect(tokenAllowed('https://cdn-lfs.huggingface.co/x')).toBe(true)
    expect(tokenAllowed('http://huggingface.co/x')).toBe(false)
    expect(tokenAllowed('https://huggingface.co:8443/x')).toBe(false)
    expect(tokenAllowed('https://evilhuggingface.co/x')).toBe(false)
    expect(tokenAllowed('http://127.0.0.1:9/x', 'http://127.0.0.1:9')).toBe(true)
  })

  it('sha256 mismatch → error, corrupt .part deleted; unknown sha → sha256Verified null', async () => {
    await expect(dl({ sha256: 'f'.repeat(64) })).rejects.toMatchObject({ kind: 'sha256_mismatch' })
    expect(existsSync(join(dir, 'model.gguf.part'))).toBe(false)
    const r = await dl({ sha256: null })
    expect(r.sha256Verified).toBeNull()
    rmSync(r.filePath)
  })

  it('403 → gated_accept_license pointing at the model page; 401 → auth_required; 404 → not_found', async () => {
    const e = await dl({ repoId: 'org/gated-GGUF', path: 'x.gguf' }).catch((x) => x) as HubError
    expect(e).toMatchObject({ kind: 'gated_accept_license', status: 403 })
    expect(e.message).toContain(`${base}/org/gated-GGUF`)
    await expect(dl({ repoId: 'org/private', path: 'x.gguf' })).rejects.toMatchObject({ kind: 'auth_required' })
    await expect(dl({ path: 'missing.gguf' })).rejects.toMatchObject({ kind: 'not_found' })
  })

  it('cancel mid-stream → cancelled, .part kept for resume', async () => {
    const ac = new AbortController()
    const err = await dl({ path: 'slow.gguf', signal: ac.signal, sha256: null, onProgress: () => ac.abort() }).catch((x) => x)
    expect(err).toMatchObject({ kind: 'cancelled' })
    const part = join(dir, 'slow.gguf.part')
    expect(existsSync(part)).toBe(true)
    expect(statSync(part).size).toBeGreaterThan(0)
  })

  it('refuses paths that escape destDir', async () => {
    await expect(dl({ path: '../evil.gguf' })).rejects.toMatchObject({ kind: 'bad_path' })
    await expect(dl({ path: 'C:/x.gguf' })).rejects.toMatchObject({ kind: 'bad_path' })
  })
})
