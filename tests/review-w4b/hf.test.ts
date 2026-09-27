import { afterEach, describe, expect, it, vi } from 'vitest'
import { Writable, Readable } from 'node:stream'

vi.mock('node:fs', () => ({
  existsSync: vi.fn(() => false), statSync: vi.fn(() => ({ size: 0 })),
  mkdirSync: vi.fn(), renameSync: vi.fn(), rmSync: vi.fn(),
  createReadStream: vi.fn(() => Readable.from([])),
  createWriteStream: vi.fn(() => new Writable({ write(_chunk, _enc, cb) { cb() } }))
}))
import { existsSync, statSync } from 'node:fs'
import { downloadFile, listGgufFiles } from '../../src/core/hub/hf'

afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); vi.mocked(existsSync).mockReturnValue(false) })
const opts = { repoId: 'owner/model', path: 'model.gguf', destDir: 'C:\\models', token: 'hf_TEST_ONLY', sizeBytes: 3 }

describe('HF boundary with mocked network and disk', () => {
  it.fails('never forwards a token to an external pagination URL', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(new Response('[]', { headers: { link: '<https://outside.invalid/page>; rel="next"' } }))
      .mockResolvedValueOnce(new Response('[]'))
    vi.stubGlobal('fetch', fetcher)
    await listGgufFiles('owner/model', { token: opts.token })
    const external = fetcher.mock.calls.find(([url]) => String(url).includes('outside.invalid'))
    expect(external?.[1]?.headers?.authorization).toBeUndefined()
  })

  it.fails('never sends authorization after an HTTPS to HTTP downgrade', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(new Response(null, { status: 302, headers: { location: 'http://huggingface.co/plain' } }))
      .mockResolvedValueOnce(new Response('abc'))
    vi.stubGlobal('fetch', fetcher)
    await downloadFile(opts)
    expect(fetcher.mock.calls[1][1].headers.authorization).toBeUndefined()
  })

  it.fails('rejects a resumed response whose range starts at the wrong offset', async () => {
    vi.mocked(existsSync).mockReturnValue(true)
    vi.mocked(statSync).mockReturnValue({ size: 3 } as any)
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('abc', { status: 206, headers: { 'content-range': 'bytes 0-2/6' } })))
    await expect(downloadFile({ ...opts, sizeBytes: 6 })).rejects.toThrow()
  })

  it.fails('rejects Windows alternate data stream file names', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('abc')))
    await expect(downloadFile({ ...opts, path: 'existing.gguf:hidden.gguf' })).rejects.toThrow()
  })

  it('removes auth on a redirect to a CDN host', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(new Response(null, { status: 302, headers: { location: 'https://cdn.invalid/signed' } }))
      .mockResolvedValueOnce(new Response('abc'))
    vi.stubGlobal('fetch', fetcher)
    await downloadFile(opts)
    expect(fetcher.mock.calls[1][1].headers.authorization).toBeUndefined()
  })
})
