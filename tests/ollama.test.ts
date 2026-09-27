import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { defaultLmStudioDirs, defaultOllamaRoot, listOllamaModels, toModelInfo } from '../src/core/runtimes/ollama/models'

const root = mkdtempSync(join(tmpdir(), 'lao-ollama-'))
afterAll(() => rmSync(root, { recursive: true, force: true }))

// Smallest valid GGUF v3: 0 tensors, 1 kv (general.architecture = "llama").
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b }
const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b }
const gstr = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)])
const tinyGguf = Buffer.concat([u32(0x46554747), u32(3), u64(0), u64(1), gstr('general.architecture'), u32(8), gstr('llama')])

const write = (rel: string, data: string | Buffer) => { const p = join(root, rel); mkdirSync(dirname(p), { recursive: true }); writeFileSync(p, data); return p }
const manifest = (layers: object[]) => JSON.stringify({ schemaVersion: 2, mediaType: 'application/vnd.docker.distribution.manifest.v2+json', layers })
const layer = (mediaType: string, hex: string, size: number) => ({ mediaType, digest: `sha256:${hex}`, size })

write('blobs/sha256-aaa', tinyGguf)
write('blobs/sha256-ppp', '{"temperature":0.7}')
write('blobs/sha256-bad', 'not a gguf')
write('manifests/registry.ollama.ai/library/llama3.1/8b', manifest([layer('application/vnd.ollama.image.model', 'aaa', 999), layer('application/vnd.ollama.image.params', 'ppp', 19)]))
write('manifests/registry.ollama.ai/someone/coder/q4', manifest([layer('application/vnd.ollama.image.model', 'bad', 10)]))
write('manifests/hf.co/org/repo/latest', manifest([layer('application/vnd.ollama.image.model', 'missing', 12345)]))
write('manifests/registry.ollama.ai/library/broken/x', '{ not json')
write('manifests/registry.ollama.ai/library/adapter-only/x', manifest([layer('application/vnd.ollama.image.adapter', 'aaa', 1)]))

describe('listOllamaModels', () => {
  it('enumerates manifests → GGUF blobs, names like `ollama list`, skips broken/non-model manifests', async () => {
    const ms = await listOllamaModels(root)
    expect(ms.map((m) => [m.name, m.exists])).toEqual([
      ['hf.co/org/repo:latest', false],
      ['llama3.1:8b', true],
      ['someone/coder:q4', true]
    ])
    const llama = ms.find((m) => m.name === 'llama3.1:8b')!
    expect(llama).toMatchObject({ blobPath: join(root, 'blobs', 'sha256-aaa'), sizeBytes: tinyGguf.length, paramsBlobPath: join(root, 'blobs', 'sha256-ppp') })
    expect(ms.find((m) => !m.exists)!.sizeBytes).toBe(12345) // from the manifest when the blob is gone
  })

  it('missing root → []', async () => {
    expect(await listOllamaModels(join(root, 'nope'))).toEqual([])
  })

  it('toModelInfo reads the blob as GGUF; bad or missing blobs get metaError, never throw', async () => {
    const [missing, llama, bad] = await Promise.all((await listOllamaModels(root)).map((m) => toModelInfo(m)))
    expect(llama).toMatchObject({ id: llama.path, runtime: 'ollama', ollamaName: 'llama3.1:8b', meta: { arch: 'llama' } })
    expect(bad.meta).toBeNull()
    expect(bad.metaError).toBeTruthy()
    expect(missing.metaError).toMatch(/^blob missing/)
  })
})

describe('default dirs', () => {
  it('OLLAMA_MODELS wins over the home default', () => {
    const prev = process.env.OLLAMA_MODELS
    try {
      process.env.OLLAMA_MODELS = 'D:\\ollama'
      expect(defaultOllamaRoot()).toBe('D:\\ollama')
      delete process.env.OLLAMA_MODELS
      expect(defaultOllamaRoot()).toBe(join(homedir(), '.ollama', 'models'))
    } finally {
      if (prev === undefined) delete process.env.OLLAMA_MODELS
      else process.env.OLLAMA_MODELS = prev
    }
  })
  it('LM Studio current + legacy dirs', () => {
    expect(defaultLmStudioDirs()).toEqual([join(homedir(), '.lmstudio', 'models'), join(homedir(), '.cache', 'lm-studio', 'models')])
  })
})
