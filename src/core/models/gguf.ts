import { existsSync, readdirSync, statSync } from 'node:fs'
import { open, type FileHandle } from 'node:fs/promises'
import { basename, join } from 'node:path'
import type { GenKnobs, ModelMeta } from '../../shared/bench-types'
import { readSidecar } from '../hub/modelcard'
import type { GgufMetadata, ModelInfo } from '../../shared/types'

// GGUF v2/v3 header reader (docs/DESIGN.md §2.6). Streams through the file with a 1 MiB window:
// tokenizer arrays (~6 MB on qwen2.5) are skipped element-by-element, never materialized.

const CHUNK = 1 << 20
const MAX_TENSORS = 1_000_000 // sanity bound against corrupt headers
const SMALL_ARRAY = 4096
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

  /** Read a scalar/string value; array contents are skipped and yield only { arrayLength }. */
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
        // Small numeric/bool arrays are layout data (per-layer head_count_kv, sliding_window_pattern): keep them.
        // Big ones (tokenizer token_type, 150k entries) and string arrays are walked/skipped, never materialized.
        if (FIXED[elem] !== undefined && elem !== T.ARRAY && n <= SMALL_ARRAY) {
          const values: unknown[] = []
          for (let i = 0; i < n; i++) values.push(await this.value(elem))
          return { arrayLength: n, values }
        }
        if (FIXED[elem] !== undefined) this.skip(n * FIXED[elem])
        else for (let i = 0; i < n; i++) await this.value(elem) // strings / nested arrays: walk lengths only
        return { arrayLength: n }
      }
      default: throw new Error(`unknown GGUF value type ${type} at byte ${this.pos}`)
    }
  }
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null)
const arrLen = (v: unknown): number | null => (v && typeof v === 'object' && 'arrayLength' in v ? (v as { arrayLength: number }).arrayLength : null)
const arrVals = (v: unknown): unknown[] | null => (v && typeof v === 'object' && 'values' in v ? (v as { values: unknown[] }).values : null)
const nums = (v: unknown): number[] | null => { const a = arrVals(v); return a && a.every((x) => typeof x === 'number') ? (a as number[]) : null }
const bools = (v: unknown): boolean[] | null => { const a = arrVals(v); return a && a.every((x) => typeof x === 'boolean') ? (a as boolean[]) : null }

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
    let last = { offset: -1, bytes: 0 } // tensor with the highest data offset
    for (let i = 0; i < nTensors; i++) {
      c.skip(await c.u64()) // name
      const nDims = await c.u32()
      let p = 1
      for (let d = 0; d < nDims; d++) p *= await c.u64()
      params += p
      const type = await c.u32()
      const offset = await c.u64()
      if (offset > last.offset) last = { offset, bytes: tensorBytes(type, p) }
    }
    // Tensor data starts at the next alignment boundary; a file shorter than data start + last tensor end is a
    // partial download (seen: a 1.22 GiB piece of a 16 GiB model parsed as a valid 27B).
    const align = num(kv.get('general.alignment')) ?? 32
    const dataStart = Math.ceil(c.pos / align) * align
    const expectedMinBytes = nTensors > 0 ? dataStart + last.offset + last.bytes : c.pos

    const arch = str(kv.get('general.architecture'))
    const a = (k: string) => (arch ? kv.get(`${arch}.${k}`) : undefined)
    const blockCount = num(a('block_count'))
    const headCount = num(a('attention.head_count'))
    // head_count_kv is an ARRAY on per-layer-GQA archs (gemma4: [8,8,8,8,8,2,…]); headsKv = max, per-layer kept.
    const kvArr = nums(a('attention.head_count_kv'))
    const headCountKv = num(a('attention.head_count_kv')) ?? (kvArr?.length ? Math.max(...kvArr) : null) ?? headCount
    const fullAttentionInterval = num(a('full_attention_interval')) // qwen35 hybrid: only every Nth layer has KV
    const swaPattern = bools(a('attention.sliding_window_pattern'))
    const slidingWindow = num(a('attention.sliding_window'))
    const keyLenSwa = num(a('attention.key_length_swa'))
    const valLenSwa = num(a('attention.value_length_swa'))
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
          // File truth: tied-embedding models whose GGUF stores output.weight and token_embd.weight as separate
          // tensors count both (qwen2.5-1.5B → 1.78B, matching the converter's size_label "1.8B"), deliberately.
          : { value: params, kind: 'declared', source: 'sum of tensor dims' },
      contextLength: num(a('context_length')),
      blockCount,
      headCount,
      headCountKv,
      embeddingLength,
      fileType,
      quantName: fileType != null ? (FTYPE[fileType] ?? `ftype_${fileType}`) : quantFromFilename(path),
      fileSizeBytes: size,
      incomplete: size < expectedMinBytes, // split shards describe only their own tensors, so this holds per shard
      expectedMinBytes,
      keyLength: keyLen,
      valueLength: valLen,
      nVocab: num(a('vocab_size')) ?? arrLen(kv.get('tokenizer.ggml.tokens')),
      slidingWindow,
      headCountKvPerLayer: kvArr,
      fullAttentionInterval,
      slidingWindowPattern: swaPattern,
      keyLengthSwa: keyLenSwa,
      valueLengthSwa: valLenSwa,
      // Qwen3-style templates switch reasoning with an enable_thinking kwarg; the template text itself isn't kept.
      expertCount: num(a('expert_count')),
      expertUsedCount: num(a('expert_used_count')),
      supportsThinking: /enable_thinking/.test(str(kv.get('tokenizer.chat_template')) ?? ''),
      ...templateKnobs(str(kv.get('tokenizer.chat_template')) ?? ''),
      headDim: { value: headDim, kind: keyLen != null ? 'declared' : 'estimated' },
      estimated: {
        kvCacheBytesPerToken: blockCount && headCountKv && headDim && dv
          ? kvBytesPerToken({ blockCount, headCountKv, kvArr, fullAttentionInterval, swaPattern, dk: headDim, dv, dkSwa: keyLenSwa, dvSwa: valLenSwa })
          : null
      }
    }
  } finally {
    await fh.close()
  }
}

// ggml_type → [block size in elements, bytes per block] (ggml.c type_traits). Unknown types count 0 (lower bound).
const GGML_BLOCK: Record<number, [number, number]> = {
  0: [1, 4], 1: [1, 2], 2: [32, 18], 3: [32, 20], 6: [32, 22], 7: [32, 24], 8: [32, 34], 9: [32, 36], 10: [256, 84],
  11: [256, 110], 12: [256, 144], 13: [256, 176], 14: [256, 210], 15: [256, 292], 16: [256, 66], 17: [256, 74],
  18: [256, 98], 19: [256, 50], 20: [32, 18], 21: [256, 110], 22: [256, 82], 23: [256, 136], 24: [1, 1], 25: [1, 2],
  26: [1, 4], 27: [1, 8], 28: [1, 8], 29: [256, 56], 30: [1, 2]
}
function tensorBytes(type: number, elements: number): number {
  const b = GGML_BLOCK[type]
  return b ? Math.ceil(elements / b[0]) * b[1] : 0
}

const EFFORT_ORDER = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']

/** What a chat template lets the caller switch (never stores the template). kwNames = template variables the caller
 *  may pass (used as `x is defined` / `x | default(…)`) that concern thinking, effort or budgets. effortValues = the
 *  string literals the effort variable (or a variable derived from it) is compared with, ascending. */
export function templateKnobs(tmpl: string): { genKnobs: GenKnobs; templateKwNames: string[] } {
  const names = new Set<string>()
  for (const m of tmpl.matchAll(/\b([a-z_][a-z0-9_]*)\s+is\s+(?:not\s+)?(?:un)?defined\b|\b([a-z_][a-z0-9_]*)\s*\|\s*default\s*\(/gi)) {
    const n = m[1] ?? m[2]
    if (/think|effort|budget|reason/i.test(n) && !/^(resolved_|ns\.)/.test(n)) names.add(n)
  }
  const kwNames = [...names].sort()
  const effortKw = kwNames.find((n) => /effort/i.test(n))
  const values = new Set<string>()
  if (effortKw) {
    // Literals compared with any variable whose name contains "effort" (e.g. resolved_reasoning_effort == 'low',
    // … not in ('xhigh', 'medium', 'low')).
    for (const m of tmpl.matchAll(/\w*effort\w*\s*(?:==|!=|(?:not\s+)?in)\s*(\([^)]*\)|'[^']*'|"[^"]*")/gi)) {
      for (const lit of m[1].matchAll(/'([^']+)'|"([^"]+)"/g)) values.add((lit[1] ?? lit[2]).toLowerCase())
    }
  }
  const rank = (v: string) => { const i = EFFORT_ORDER.indexOf(v); return i < 0 ? EFFORT_ORDER.length : i }
  const effortValues = [...values].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
  const thinkingBudgetKw = kwNames.find((n) => /budget/i.test(n))
  return {
    templateKwNames: kwNames,
    genKnobs: {
      supportsThinking: /enable_thinking/.test(tmpl),
      ...(effortKw && effortValues.length ? { effortKw, effortValues } : {}),
      ...(thinkingBudgetKw ? { thinkingBudgetKw } : {})
    }
  }
}

/** f16 K+V bytes for one token (below any sliding window), layer by layer: recurrent layers (hybrid archs) hold no
 *  KV, SWA layers may use their own head dims. Same layout rules as candidates.kvLayout. */
function kvBytesPerToken(o: {
  blockCount: number; headCountKv: number; kvArr: number[] | null; fullAttentionInterval: number | null
  swaPattern: boolean[] | null; dk: number; dv: number; dkSwa: number | null; dvSwa: number | null
}): number {
  let bytes = 0
  for (let i = 0; i < o.blockCount; i++) {
    if (o.fullAttentionInterval && (i + 1) % o.fullAttentionInterval !== 0) continue
    const swa = !!o.swaPattern?.[i]
    const k = swa ? (o.dkSwa ?? o.dk) : o.dk
    const v = swa ? (o.dvSwa ?? o.dv) : o.dv
    bytes += (o.kvArr?.[i] ?? o.headCountKv) * (k + v) * 2
  }
  return bytes
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
        const meta = await readGgufMetadata(path)
        // cached generation_config.json (fetched lazily elsewhere) and its repo; every caller (app, harness) sees them
        const sc = readSidecar(path)
        const knobs = { ...meta.genKnobs, ...(sc?.generation ? { recommended: sc.generation } : {}), ...(sc?.repoId ? { repoId: sc.repoId } : {}) }
        return { ...base, meta: sc?.generation || sc?.repoId ? { ...meta, genKnobs: knobs } : meta }
      } catch (e) {
        return { ...base, meta: null, metaError: (e as Error).message }
      }
    })
  )
}

/** GGUF facts → the scorer's ModelMeta. null (with reason) when a field the planner needs is missing. */
export function toModelMeta(info: ModelInfo): { meta: ModelMeta } | { meta: null; reason: string } {
  const g = info.meta
  if (!g) return { meta: null, reason: info.metaError ?? 'no GGUF metadata' }
  if (g.incomplete) return { meta: null, reason: `file incomplete (${(g.fileSizeBytes / 1024 ** 3).toFixed(2)} of ≥${(g.expectedMinBytes / 1024 ** 3).toFixed(2)} GiB) — still downloading?` }
  const missing = (['arch', 'blockCount', 'embeddingLength', 'headCount', 'nVocab'] as const).filter((k) => g[k] == null)
  if (missing.length) return { meta: null, reason: `GGUF lacks ${missing.join(', ')}` }
  return {
    meta: {
      id: info.path, name: g.name ?? info.name, fileBytes: g.fileSizeBytes, paramCount: g.parameterCount.value, quant: g.quantName,
      arch: g.arch!, ctxTrain: g.contextLength, layers: g.blockCount!, nEmbd: g.embeddingLength!, heads: g.headCount!,
      headsKv: g.headCountKv ?? g.headCount!, keyLength: g.keyLength, valueLength: g.valueLength, nVocab: g.nVocab!, slidingWindow: g.slidingWindow,
      headsKvPerLayer: g.headCountKvPerLayer, fullAttentionInterval: g.fullAttentionInterval, slidingWindowPattern: g.slidingWindowPattern,
      keyLengthSwa: g.keyLengthSwa, valueLengthSwa: g.valueLengthSwa, supportsThinking: g.supportsThinking,
      expertCount: g.expertCount, expertUsedCount: g.expertUsedCount,
      genKnobs: g.genKnobs
    }
  }
}
