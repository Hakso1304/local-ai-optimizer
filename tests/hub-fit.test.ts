import { describe, expect, it } from 'vitest'
import { activeB, budget, estimateTps, familyOf, fileOptions, kvBytesPerToken, paramsB, place, recommend, trustOf } from '../src/core/hub/fit'
import type { HfGgufFile } from '../src/core/hub/hf'
import { WORKLOADS } from '../src/core/scoring/workloads'
import type { SystemProfile } from '../src/shared/types'

const GiB = 1024 ** 3
const GB = 1e9
/** vram null = no GPU; 'igpu' = integrated (shared RAM). */
const prof = (vram: number | null | 'igpu', ramGiB: number, inUseGiB?: number): SystemProfile => ({
  ram: { value: { totalBytes: ramGiB * GiB, availableBytes: 0 }, status: 'available', source: 't' },
  gpus: { value: vram === null ? [] : vram === 'igpu'
    ? [{ name: 'Intel(R) Arc(TM) Graphics', vendor: 'intel', pnpDeviceId: 'x', driverVersion: '1', isIntegrated: true, dedicatedVramBytes: { value: null, status: 'unavailable', source: 't' } }]
    : [{ name: 'g', vendor: 'nvidia', pnpDeviceId: 'x', driverVersion: '1', isIntegrated: false, dedicatedVramBytes: { value: vram * GiB, status: 'available', source: 't' } }], status: 'available', source: 't' },
  ...(inUseGiB !== undefined ? { vramInUse: { value: inUseGiB * GiB, status: 'available', source: 't' } } : {})
}) as unknown as SystemProfile
const m = (id: string, downloads = 0) => ({ id, downloads, likes: 0, gated: false, lastModified: null })
const f = (path: string, gb: number, shard?: { index: number; count: number }): HfGgufFile =>
  ({ path, sizeBytes: gb * GB, sha256: null, quant: /(I?Q\d_[A-Z0-9]+(?:_[A-Z0-9]+)?|BF16|F16|F32|MXFP4)/i.exec(path)?.[1]?.toUpperCase() ?? null, shard: shard ?? null })

// The three reference machines the ranking is checked on.
const DGPU12 = prof(12, 32)
const DGPU24 = prof(24, 64)
const IGPU32 = prof('igpu', 32)
const CHAT = WORKLOADS.general_chat // ctx 8192, minDecodeTps 10
const CODING = WORKLOADS.coding // ctx 16384

describe('hub fit: parsing', () => {
  it('total and active params from repo names', () => {
    expect(paramsB('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF')).toBe(30)
    expect(activeB('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', 30)).toBe(3)
    expect(paramsB('x/Mixtral-8x7B-v0.1-GGUF')).toBe(56)
    expect(activeB('x/Mixtral-8x7B-v0.1-GGUF', 56)).toBe(14)
    expect(paramsB('LiquidAI/LFM2.5-230M-GGUF')).toBe(0.23)
    expect(activeB('Qwen/Qwen3-8B-GGUF', 8)).toBe(8)
    expect(paramsB('TheBloke/phi-GGUF')).toBeNull()
  })
  it('families fold quantizer repos of the same model; trust from org and name', () => {
    expect(familyOf('Qwen/Qwen3-8B-GGUF')).toBe(familyOf('unsloth/Qwen3-8B-GGUF'))
    expect(familyOf('bartowski/Qwen3-8B-GGUF')).toBe('qwen3-8b')
    expect(familyOf('x/Ornith-1.5-9B-OBLITERATED')).not.toBe(familyOf('ornith-ai/Ornith-1.5-9B-GGUF'))
    expect(trustOf('Qwen/Qwen3-8B-GGUF')).toBe('official')
    expect(trustOf('someone/Qwen3-8B-GGUF')).toBe('community')
    expect(trustOf('OBLITERATUS/Ornith-1.5-9B-OBLITERATED')).toBe('risky')
    expect(trustOf('x/Model-abliterated-GGUF')).toBe('risky')
  })
  it('file options: shards summed under the first shard, mmproj and F16 dropped, ternary flagged, best quant first', () => {
    const o = fileOptions([
      f('m-Q8_0-00002-of-00002.gguf', 3, { index: 2, count: 2 }), f('m-Q8_0-00001-of-00002.gguf', 5, { index: 1, count: 2 }),
      f('m-Q4_K_M.gguf', 4.5), f('m-F16.gguf', 16), f('mmproj-F16.gguf', 0.9), f('Ternary-Bonsai-2-27B-PQ2_0.gguf', 7.2)
    ])
    expect(o.map((x) => [x.quant, Math.round(x.sizeBytes / GB * 10) / 10, x.shards, x.prism, x.path])).toEqual([
      ['Q8_0', 8, 2, false, 'm-Q8_0-00001-of-00002.gguf'], ['Q4_K_M', 4.5, 1, false, 'm-Q4_K_M.gguf'], ['PQ2_0', 7.2, 1, true, 'Ternary-Bonsai-2-27B-PQ2_0.gguf']
    ])
  })
})

describe('hub fit: budget and placement (planner rules)', () => {
  it('discrete: VRAM − in use (measured or default 1.5 GiB) − 1 GiB margin; RAM − 4 GiB reserve', () => {
    expect(budget(DGPU12, 8192)).toMatchObject({ gpu: 9.5 * GiB, shared: false, ram: 28 * GiB })
    expect(budget(prof(12, 32, 3), 8192).gpu).toBe(8 * GiB)
  })
  it('integrated GPU: one shared pool = RAM − reserve; no GPU: gpu 0', () => {
    expect(budget(IGPU32, 8192)).toMatchObject({ gpu: 28 * GiB, shared: true, ram: 28 * GiB })
    expect(budget(prof(null, 16), 8192)).toMatchObject({ gpu: 0, shared: false, ram: 12 * GiB })
  })
  it('placement tiers and the offload share', () => {
    const b = budget(DGPU12, 8192)
    expect(place(6 * GB, 1 * GB, b)).toEqual({ fit: 'gpu', gpuShare: 1 })
    expect(place(15 * GB, 5 * GB, b)?.fit).toBe('offload')
    expect(place(15 * GB, 5 * GB, b)!.gpuShare).toBeCloseTo(9.5 * GiB / 20e9, 3)
    expect(place(40 * GB, 5 * GB, b)).toBeNull()
    expect(place(20 * GB, 2 * GB, budget(IGPU32, 8192))?.fit).toBe('shared')
    expect(place(6 * GB, 1 * GB, budget(prof(null, 16), 8192))?.fit).toBe('cpu')
  })
  it('KV grows with context; half offload is nearly CPU speed', () => {
    expect(kvBytesPerToken(8) * 32768 / GB).toBeGreaterThan(3) // 8B at 32K ≈ 3.7 GB (real llama 3.1: 4.3 GB)
    const full = estimateTps(5 * GB, 0.5 * GB, 1, false), half = estimateTps(5 * GB, 0.5 * GB, 0.5, false), cpu = estimateTps(5 * GB, 0.5 * GB, 0, false)
    expect(full).toBeGreaterThan(40)
    expect(half).toBeLessThan(cpu * 2)
    expect(estimateTps(5 * GB, 0.5 * GB, 1, true)).toBeCloseTo(cpu, 5) // shared RAM: GPU reads the same DDR
  })
})

const POPULAR = [
  m('Qwen/Qwen3-8B-GGUF', 500_000), m('unsloth/Qwen3-8B-GGUF', 450_000), m('Qwen/Qwen3-14B-GGUF', 300_000), m('Qwen/Qwen3-32B-GGUF', 200_000),
  m('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', 900_000), m('unsloth/Llama-3.3-70B-Instruct-GGUF', 100_000),
  m('prism-ml/Ternary-Bonsai-2-27B-gguf', 800_000), m('OBLITERATUS/Qwen3-8B-OBLITERATED', 700_000), m('Qwen/Qwen3-0.6B-GGUF', 350_000)
]
const FILES = new Map([
  ['Qwen/Qwen3-8B-GGUF', [f('Qwen3-8B-Q8_0.gguf', 8.7), f('Qwen3-8B-Q6_K.gguf', 6.7), f('Qwen3-8B-Q4_K_M.gguf', 5.0)]],
  ['Qwen/Qwen3-14B-GGUF', [f('Qwen3-14B-Q8_0.gguf', 15.7), f('Qwen3-14B-Q4_K_M.gguf', 9.0)]],
  ['Qwen/Qwen3-32B-GGUF', [f('Qwen3-32B-Q8_0.gguf', 34.8), f('Qwen3-32B-Q4_K_M.gguf', 19.8)]],
  ['unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', [f('Q8_0/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf', 32.5), f('Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf', 18.6)]],
  ['unsloth/Llama-3.3-70B-Instruct-GGUF', [f('Llama-3.3-70B-Instruct-Q4_K_M.gguf', 42.5)]],
  ['prism-ml/Ternary-Bonsai-2-27B-gguf', [f('Ternary-Bonsai-2-27B-PQ2_0.gguf', 7.2), f('Ternary-Bonsai-2-27B-F16.gguf', 53.8)]]
])
const row = (r: ReturnType<typeof recommend>, id: string) => r.models.find((x) => x.id === id)

describe('hub fit: recommendation on three machines', () => {
  it('12 GB discrete + 32 GB RAM, chat: 8B picks the best quant that fits the GPU, 14B goes partial, 70B is out, dupes fold', () => {
    const r = recommend(POPULAR, FILES, DGPU12, CHAT)
    expect(row(r, 'Qwen/Qwen3-8B-GGUF')).toMatchObject({ fit: 'gpu', file: { quant: 'Q8_0' }, alsoIn: ['unsloth/Qwen3-8B-GGUF'] }) // 8.7 + 0.9 KV ≤ 10.2 GB
    expect(row(recommend(POPULAR, FILES, DGPU12, CODING), 'Qwen/Qwen3-8B-GGUF')).toMatchObject({ fit: 'gpu', file: { quant: 'Q6_K' } }) // 16K KV pushes Q8_0 over
    expect(row(r, 'Qwen/Qwen3-14B-GGUF')?.fit).toBe('offload')
    expect(row(r, 'unsloth/Llama-3.3-70B-Instruct-GGUF')).toBeUndefined()
    expect(row(r, 'Qwen/Qwen3-32B-GGUF')?.fit).toBe('offload') // 19.8 + KV fits VRAM+RAM
    expect(r.models[0].fit).toBe('gpu')
    expect(r.models.at(-1)?.id).toBe('OBLITERATUS/Qwen3-8B-OBLITERATED') // risky sinks
  })
  it('24 GB discrete + 64 GB RAM, coding: the 30B-A3B coder outranks dense 14B on the GPU (larger, coder, fast)', () => {
    const r = recommend(POPULAR, FILES, DGPU24, CODING)
    const coder = row(r, 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF')!
    expect(coder.fit).toBe('gpu')
    expect(coder.activeB).toBe(3)
    expect(coder.estTps).toBeGreaterThan(row(r, 'Qwen/Qwen3-14B-GGUF')!.estTps) // active 3B reads far less per token
    expect(r.models.findIndex((x) => x.id === coder.id)).toBeLessThan(r.models.findIndex((x) => x.id === 'Qwen/Qwen3-14B-GGUF'))
  })
  it('integrated GPU + 32 GB RAM: shared pool; the ternary 27B fits as a 7.2 GB file although its Q4 estimate would not', () => {
    const r = recommend(POPULAR, FILES, IGPU32, CHAT)
    const bonsai = row(r, 'prism-ml/Ternary-Bonsai-2-27B-gguf')!
    expect(bonsai).toMatchObject({ fit: 'shared', file: { quant: 'PQ2_0', prism: true } })
    expect(row(r, 'Qwen/Qwen3-32B-GGUF')?.fit).toBe('shared')
    expect(row(r, 'unsloth/Llama-3.3-70B-Instruct-GGUF')).toBeUndefined()
    expect(r.models.every((x) => x.fit === 'shared')).toBe(true)
    expect(recommend([m('a/Big-27B')], new Map(), IGPU32, CHAT).models[0]?.file).toBeNull() // name estimate until files arrive
  })
  it('usable = estimated decode ≥ the workload gate; only picks under half the gate sink below smaller models', () => {
    const r = recommend(POPULAR, FILES, IGPU32, CHAT)
    expect(row(r, 'Qwen/Qwen3-32B-GGUF')?.usable).toBe(false)
    expect(row(r, 'Qwen/Qwen3-0.6B-GGUF')?.usable).toBe(true)
    const ids = r.models.map((x) => x.id)
    expect(ids.indexOf('Qwen/Qwen3-32B-GGUF')).toBeGreaterThan(ids.indexOf('Qwen/Qwen3-0.6B-GGUF')) // ≈3 t/s < gate/2 = 5: sinks
    expect(ids.indexOf('Qwen/Qwen3-14B-GGUF')).toBeGreaterThan(ids.indexOf('Qwen/Qwen3-0.6B-GGUF')) // Q4 9 GB ≈ 4 t/s: sinks too
    // 8B: no quant clears the 10 t/s gate on shared DDR, so the smallest file (Q4_K_M ≈ 7 t/s) is picked over Q8_0 (≈ 4 t/s);
    // marked slow but above gate/2, it keeps its size rank ahead of the tiny models.
    expect(row(r, 'Qwen/Qwen3-8B-GGUF')).toMatchObject({ file: { quant: 'Q4_K_M' }, usable: false, alsoIn: ['unsloth/Qwen3-8B-GGUF'] })
    expect(ids.indexOf('Qwen/Qwen3-8B-GGUF')).toBeLessThan(ids.indexOf('Qwen/Qwen3-0.6B-GGUF'))
  })
  it('a repo whose only files are far too small for the model (draft/partial uploads) is not recommended', () => {
    const files = new Map([['z-lab/Qwen3.8-27B-DFlash2-GGUF', [f('Qwen3.8-27B-DFlash2-Q8_0.gguf', 2.1)]], ['prism-ml/Ternary-Bonsai-2-27B-gguf', FILES.get('prism-ml/Ternary-Bonsai-2-27B-gguf')!]])
    const r = recommend([m('z-lab/Qwen3.8-27B-DFlash2-GGUF'), m('prism-ml/Ternary-Bonsai-2-27B-gguf')], files, IGPU32, CHAT)
    expect(r.models.map((x) => x.id)).toEqual(['prism-ml/Ternary-Bonsai-2-27B-gguf']) // 7.2 GB ternary is plausible for 27B, 2.1 GB is not
  })
})
