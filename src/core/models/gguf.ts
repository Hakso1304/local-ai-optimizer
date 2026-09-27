import { existsSync, readdirSync, statSync } from 'node:fs'
import { open, type FileHandle } from 'node:fs/promises'
import { basename, join } from 'node:path'
import type { GgufMetadata, ModelInfo } from '../../shared/types'

// GGUF v2/v3 header reader (docs/DESIGN.md §2.6). Streams through the file with a 1 MiB window:
// tokenizer arrays (~6 MB on qwen2.5) are skipped element-by-element, never materialized.

const CHUNK = 1 << 20
const MAX_TENSORS = 1_000_000 // sanity bound against corrupt headers
const T = { U8: 0, I8: 1, U16: 2, I16: 3, U32: 4, I32: 5, F32: 6, BOOL: 7, STRING: 8, ARRAY: 9, U64: 10, I64: 11, F64: 12 } as const
const FIXED: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 }

// llama.h LLAMA_FTYPE_MOSTLY_*
const FTYPE: Record<number, string> = {
  0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 7: 'Q8_0', 8: 'Q5_0', 9: 'Q5_1', 10: 'Q2_K', 11: 'Q3_K_S', 12: 'Q3_K_M', 13: 'Q3_K_L',
  14: 'Q4_K_S', 15: 'Q4_K_M', 16: 'Q5_K_S', 17: 'Q5_K_M', 18: 'Q6_K', 19: 'IQ2_XXS', 20: 'IQ2_XS', 21: 'Q2_K_S', 22: 'IQ3_XS',
  23: 'IQ3_XXS', 24: 'IQ1_S', 25: 'IQ4_NL', 26: 'IQ3_S', 27: 'IQ3_M', 28: 'IQ2_S', 29: 'IQ2_M', 30: 'IQ4_XS', 31: 'IQ1_M',
  32: 'BF16', 36: 'TQ1_0', 37: 'TQ2_0', 38: 'MXFP4_MOE', 39: 'NVFP4', 40: 'Q1_0', 41: 'Q2_0'
}

class Cursor {
  pos = 0
  private buf = Buffer.alloc(0)
  private start = 0 // file offset of buf[0]
  constructor(private fh: FileHandle) {}

  private async need(n: number): Promise<number> {
    if (this.pos < this.start || this.pos + n > this.start + this.buf.length) {
      const b = Buffer.alloc(Math.max(n, CHUNK))
      const { bytesRead } = await this.fh.read(b, 0, b.length, this.pos)
      if (bytesRead < n) throw new Error(`unexpected end of file at byte ${this.pos}`)
      this.buf = b.subarray(0, bytesRead)
      this.start = this.pos
    }
    const at = this.pos - this.start
    this.pos += n
    return at
  }

  skip(n: number): void { this.pos += n }
  // Note: always await need() BEFORE touching this.buf — need() may swap the window.
  async u32(): Promise<number> {
    const at = await this.need(4)
    return this.buf.readUInt32LE(at)
  }
  async u64(): Promise<number> {
    const at = await this.need(8)
    const v = this.buf.readBigUInt64LE(at)
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`u64 too large at byte ${this.pos - 8}`)
    return Number(v)
  }
  async str(): Promise<string> {
    const len = await this.u64()
    const at = await this.need(len)
    return this.buf.toString('utf8', at, at + len)
  }

  /** Read a scalar/string value; arrays are skipped and yield undefined. */
  async value(type: number): Promise<unknown> {
    const at = FIXED[type] !== undefined ? await this.need(FIXED[type]) : -1
    const b = this.buf
    switch (type) {
      case T.U8: case T.BOOL: return type === T.BOOL ? b.readUInt8(at) !== 0 : b.readUInt8(at)
      case T.I8: return b.readInt8(at)
      case T.U16: return b.readUInt16LE(at)
      case T.I16: return b.readInt16LE(at)
      case T.U32: return b.readUInt32LE(at)
      case T.I32: return b.readInt32LE(at)
      case T.F32: return b.readFloatLE(at)
      case T.U64: return Number(b.readBigUInt64LE(at))
      case T.I64: return Number(b.readBigInt64LE(at))
      case T.F64: return b.readDoubleLE(at)
      case T.STRING: return this.str()
      case T.ARRAY: {
        const elem = await this.u32()
        const n = await this.u64()
        if (FIXED[elem] !== undefined) this.skip(n * FIXED[elem])
        else for (let i = 0; i < n; i++) await this.value(elem) // strings / nested arrays: walk lengths only
        return undefined
      }
      default: throw new Error(`unknown GGUF value type ${type} at byte ${this.pos}`)
    }
  }
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)

/** Parse GGUF header + tensor infos. Throws on non-GGUF / corrupt files. */
export async function readGgufMetadata(path: string): Promise<GgufMetadata> {
  const fh = await open(path, 'r')
  try {
    const c = new Cursor(fh)
    if ((await c.u32()) !== 0x46554747) throw new Error('not a GGUF file (bad magic)')
    const version = await c.u32()
    if (version !== 2 && version !== 3) throw new Error(`unsupported GGUF version ${version}`)
    const nTensors = await c.u64()
    const nKv = await c.u64()
    const kv = new Map<string, unknown>()
    for (let i = 0; i < nKv; i++) {
      const key = await c.str()
      kv.set(key, await c.value(await c.u32()))
    }
    if (nTensors > MAX_TENSORS) throw new Error(`implausible tensor count ${nTensors}`)
    let params = 0
    for (let i = 0; i < nTensors; i++) {
      c.skip(await c.u64()) // name
      const nDims = await c.u32()
      let p = 1
      for (let d = 0; d < nDims; d++) p *= await c.u64()
      params += p
      c.skip(4 + 8) // ggml_type, offset
    }

    const arch = str(kv.get('general.architecture'))
    const a = (k: string) => (arch ? kv.get(`${arch}.${k}`) : undefined)
    const blockCount = num(a('block_count'))
    const headCount = num(a('attention.head_count'))
    const headCountKv = num(a('attention.head_count_kv')) ?? headCount
    const embeddingLength = num(a('embedding_length'))
    const fileType = num(kv.get('general.file_type'))
    const declaredParams = num(kv.get('general.parameter_count'))
    const splitCount = num(kv.get('split.count')) ?? 1
    const keyLen = num(a('attention.key_length'))
    const valLen = num(a('attention.value_length'))
    const headDim = keyLen ?? (embeddingLength && headCount ? embeddingLength / headCount : null)
    const dv = valLen ?? headDim
    const { size } = await fh.stat()
    return {
      ggufVersion: version,
      arch,
      name: str(kv.get('general.name')),
      sizeLabel: str(kv.get('general.size_label')),
      // ponytail: split models only sum this shard's tensors; sum across shards when split GGUFs are grouped.
      parameterCount: declaredParams != null
        ? { value: declaredParams, kind: 'declared', source: 'general.parameter_count' }
        : splitCount > 1
          ? { value: null, kind: 'unavailable', source: `split file (1 of ${splitCount}); tensor sum would be partial` }
          : { value: params, kind: 'declared', source: 'sum of tensor dims' },
      contextLength: num(a('context_length')),
      blockCount,
      headCount,
      headCountKv,
      embeddingLength,
      fileType,
      quantName: fileType != null ? (FTYPE[fileType] ?? `ftype_${fileType}`) : quantFromFilename(path),
      fileSizeBytes: size,
      headDim: { value: headDim, kind: keyLen != null ? 'declared' : 'estimated' },
      estimated: {
        // f16 K+V: blockCount * headCountKv * (dk + dv) * 2 bytes
        kvCacheBytesPerToken: blockCount && headCountKv && headDim && dv ? blockCount * headCountKv * (headDim + dv) * 2 : null
      }
    }
  } finally {
    await fh.close()
  }
}

function quantFromFilename(path: string): string | null {
  return /(I?Q\d_[A-Z0-9_]+|BF16|F16|F32)/i.exec(basename(path))?.[1]?.toUpperCase() ?? null
}

/** Recursively list *.gguf files under each existing dir (missing dirs skipped) with header metadata.
 *  A file whose header fails to parse is still listed, with meta null and metaError set. */
export async function findGgufModels(dirs: string[]): Promise<ModelInfo[]> {
  const paths = new Set<string>()
  for (const dir of dirs) {
    if (!existsSync(dir)) continue
    for (const rel of readdirSync(dir, { recursive: true, encoding: 'utf8' })) {
      // ponytail: split shards (-0000N-of-0000M) and mmproj files are listed as-is; group/filter when it matters.
      if (rel.toLowerCase().endsWith('.gguf') && statSync(join(dir, rel)).isFile()) paths.add(join(dir, rel))
    }
  }
  return Promise.all(
    [...paths].map(async (path): Promise<ModelInfo> => {
      const base = { id: path, name: basename(path, '.gguf'), path, sizeBytes: statSync(path).size, runtime: 'llamacpp' as const }
      try {
        return { ...base, meta: await readGgufMetadata(path) }
      } catch (e) {
        return { ...base, meta: null, metaError: (e as Error).message }
      }
    })
  )
}
