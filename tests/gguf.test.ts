import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { findGgufModels, readGgufMetadata, toModelMeta } from '../src/core/models/gguf'

// Minimal GGUF writer: enough of the spec to exercise every code path in the reader.
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b }
const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b }
const gstr = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)])
type Kv = [string, 'u32' | 'str' | 'f32' | 'bool' | 'strarr' | 'i32arr', unknown]
function kvBytes([k, t, v]: Kv): Buffer {
  const val = {
    u32: () => Buffer.concat([u32(4), u32(v as number)]),
    f32: () => { const b = Buffer.alloc(4); b.writeFloatLE(v as number); return Buffer.concat([u32(6), b]) },
    bool: () => Buffer.concat([u32(7), Buffer.from([v ? 1 : 0])]),
    str: () => Buffer.concat([u32(8), gstr(v as string)]),
    strarr: () => Buffer.concat([u32(9), u32(8), u64((v as string[]).length), ...(v as string[]).map(gstr)]),
    i32arr: () => Buffer.concat([u32(9), u32(5), u64((v as number[]).length), ...(v as number[]).map((n) => u32(n))])
  }[t]()
  return Buffer.concat([gstr(k), val])
}
function gguf(version: number, kvs: Kv[], tensors: number[][]): Buffer {
  return Buffer.concat([
    u32(0x46554747), u32(version), u64(tensors.length), u64(kvs.length),
    ...kvs.map(kvBytes),
    ...tensors.map((dims, i) => Buffer.concat([gstr(`t${i}`), u32(dims.length), ...dims.map(u64), u32(0), u64(0)]))
  ])
}

const dir = mkdtempSync(join(tmpdir(), 'lao-gguf-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

const KVS: Kv[] = [
  ['general.architecture', 'str', 'llama'],
  ['general.name', 'str', 'Tiny'],
  ['tokenizer.ggml.tokens', 'strarr', Array.from({ length: 5000 }, (_, i) => `tok${i}`)], // must be skipped, not kept
  ['tokenizer.ggml.token_type', 'i32arr', Array.from({ length: 5000 }, () => 1)],
  ['llama.context_length', 'u32', 4096],
  ['llama.block_count', 'u32', 2],
  ['llama.embedding_length', 'u32', 64],
  ['llama.attention.head_count', 'u32', 4],
  ['llama.attention.head_count_kv', 'u32', 2],
  ['llama.rope.freq_base', 'f32', 10000],
  ['general.file_type', 'u32', 15],
  ['some.flag', 'bool', true]
]

describe('readGgufMetadata', () => {
  it('parses a synthetic v3 file and derives params from tensor dims', async () => {
    const p = join(dir, 'tiny.gguf')
    writeFileSync(p, gguf(3, KVS, [[64, 100], [64, 64], [64]]))
    const m = await readGgufMetadata(p)
    expect(m).toMatchObject({
      ggufVersion: 3, arch: 'llama', name: 'Tiny', contextLength: 4096, blockCount: 2, headCount: 4, headCountKv: 2,
      embeddingLength: 64, fileType: 15, quantName: 'Q4_K_M',
      parameterCount: { value: 6400 + 4096 + 64, kind: 'declared', source: 'sum of tensor dims' },
      headDim: { value: 16, kind: 'estimated' },
      nVocab: 5000, // from the skipped tokenizer.ggml.tokens array length
      keyLength: null,
      slidingWindow: null,
      estimated: { kvCacheBytesPerToken: 2 * 2 * (16 + 16) * 2 }
    })
    const mm = toModelMeta({ id: p, name: 'tiny', path: p, sizeBytes: 1, runtime: 'llamacpp', meta: m })
    expect(mm.meta).toMatchObject({ id: p, arch: 'llama', layers: 2, heads: 4, headsKv: 2, nEmbd: 64, nVocab: 5000, quant: 'Q4_K_M', ctxTrain: 4096 })
  })

  it('toModelMeta refuses models missing planner-critical fields', () => {
    expect(toModelMeta({ id: 'x', name: 'x', path: 'x', sizeBytes: 1, runtime: 'llamacpp', meta: null, metaError: 'bad magic' })).toEqual({ meta: null, reason: 'bad magic' })
  })

  it('accepts v2, prefers general.parameter_count, and falls back to filename quant', async () => {
    const p = join(dir, 'Other-IQ4_XS.gguf')
    writeFileSync(p, gguf(2, [['general.architecture', 'str', 'qwen2'], ['general.parameter_count', 'u32', 123]], []))
    const m = await readGgufMetadata(p)
    expect(m.parameterCount).toEqual({ value: 123, kind: 'declared', source: 'general.parameter_count' })
    expect(m.quantName).toBe('IQ4_XS')
    expect(m.estimated.kvCacheBytesPerToken).toBeNull() // no dims -> no estimate, not a guess
  })

  it('rejects non-GGUF and truncated files', async () => {
    writeFileSync(join(dir, 'bad.gguf'), 'hello world, not gguf')
    await expect(readGgufMetadata(join(dir, 'bad.gguf'))).rejects.toThrow(/bad magic/)
    writeFileSync(join(dir, 'trunc.gguf'), gguf(3, KVS, [[64]]).subarray(0, 200))
    await expect(readGgufMetadata(join(dir, 'trunc.gguf'))).rejects.toThrow(/unexpected end/)
  })

  const real = join(__dirname, '..', 'models', 'qwen2.5-0.5b-instruct-q8_0.gguf')
  it.skipIf(!existsSync(real))('reads the real qwen2.5-0.5B header (DESIGN §2.6 values)', async () => {
    const m = await readGgufMetadata(real)
    expect(m).toMatchObject({
      arch: 'qwen2', sizeLabel: '630M', contextLength: 32768, blockCount: 24, embeddingLength: 896, headCount: 14, headCountKv: 2,
      fileType: 7, quantName: 'Q8_0', estimated: { kvCacheBytesPerToken: 12288 }
    })
    expect(m.nVocab).toBeGreaterThan(150_000)
    expect(m.parameterCount.value).toBeGreaterThan(400e6)
  })
})

describe('findGgufModels', () => {
  it('lists .gguf recursively with metadata, keeps unparseable files with metaError, skips missing dirs', async () => {
    const root = join(dir, 'models')
    mkdirSync(join(root, 'sub'), { recursive: true })
    writeFileSync(join(root, 'sub', 'A.GGUF'), gguf(3, KVS, [[8]]))
    writeFileSync(join(root, 'broken.gguf'), 'nope')
    writeFileSync(join(root, 'notes.txt'), 'x')
    const found = (await findGgufModels([root, join(root, 'missing')])).sort((x, y) => x.name.localeCompare(y.name))
    expect(found.map((m) => m.name)).toEqual(['A.GGUF', 'broken'])
    expect(found[0].meta?.arch).toBe('llama')
    expect(found[1]).toMatchObject({ meta: null, metaError: expect.stringMatching(/magic/) })
  })
})
