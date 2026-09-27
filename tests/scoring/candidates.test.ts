import { describe, expect, it } from 'vitest'
import type { ModelMeta } from '../../src/shared/bench-types'
import { DEFAULT_CANDIDATE_RULES, estimateMemory, generateCandidates } from '../../src/core/benchmark/candidates'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { load, machine } from './helpers'

const llama8b = load('sweep-smooth.json').models[0]
const model = (o: Partial<ModelMeta>): ModelMeta => ({ ...llama8b, ...o })
// ~14B Q4_K_M (Qwen2.5-14B-like): 48 layers, 8 KV heads, 128 head dim → 192 KiB/token f16 KV.
const q14b = model({ id: 'q14b', fileBytes: 8.99e9, paramCount: 14.8e9, layers: 48, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064 })
// ~70B Q4_K_M: 42 GB file, 80 layers.
const l70b = model({ id: 'l70b', fileBytes: 42.5e9, paramCount: 70.6e9, layers: 80, nEmbd: 8192, heads: 64, headsKv: 8 })
const vulkan = { backend: 'vulkan' as const }

describe('estimateMemory', () => {
  it('KV for Llama-3.1-8B f16 = 128 KiB/token', () => {
    expect(estimateMemory(llama8b, 32, 1, 'f16').kvBytes).toBe(131072)
  })
})

describe('generateCandidates (16 GB VRAM / 31 GB RAM)', () => {
  it('8B fits fully: full offload, all steps ≤ 64K, plus q8 KV for long-context', () => {
    const { candidates } = generateCandidates(machine(), llama8b, vulkan, WORKLOADS.long_context_coding)
    expect(candidates.map((c) => c.id)).toEqual(['llama31-8b-q4km|ngl=all|kv=f16|t=8', 'llama31-8b-q4km|ngl=all|kv=q8_0|t=8'])
    expect(candidates[0].ctxSteps).toEqual([2048, 4096, 8192, 16384, 32768, 65536])
    expect(candidates[0].estVramBytes.kind).toBe('estimated')
  })

  it('never exceeds the declared context (A13)', () => {
    const { candidates } = generateCandidates(machine(), model({ ctxTrain: 8192 }), vulkan, WORKLOADS.coding)
    expect(candidates[0].ctxSteps).toEqual([2048, 4096, 8192])
    expect(candidates[0].skippedSteps.map((s) => s.ctx)).toEqual([16384, 32768, 65536])
    expect(candidates[0].skippedSteps[0].reason).toMatch(/above declared context 8K/)
  })

  it('14B: keeps one step above the VRAM estimate, prunes the rest with a reason', () => {
    const rules = { ...DEFAULT_CANDIDATE_RULES, ctxLadder: [...DEFAULT_CANDIDATE_RULES.ctxLadder, 131072] }
    const [c] = generateCandidates(machine(), q14b, vulkan, WORKLOADS.coding, rules).candidates
    expect(c.ctxSteps).toEqual([2048, 4096, 8192, 16384, 32768, 65536]) // 32K ≈ fits, 64K is the one kept over
    expect(c.notes).toContain('64K is above the VRAM estimate; kept to observe the cliff')
    expect(c.skippedSteps.at(-1)).toMatchObject({ ctx: 131072 })
    expect(c.skippedSteps.at(-1)!.reason).toMatch(/est\. VRAM .* > budget/)
  })

  it('70B: nothing fits; every combo rejected with a memory reason', () => {
    const { candidates, rejected } = generateCandidates(machine(), l70b, vulkan, WORKLOADS.coding)
    expect(candidates).toEqual([])
    expect(rejected.length).toBeGreaterThan(0)
    expect(rejected.every((r) => /est\. (VRAM|RAM)/.test(r.reason))).toBe(true)
  })

  it('no GPU → only ngl=0 (CPU)', () => {
    const { candidates } = generateCandidates({ ...machine(), gpuDevice: null }, llama8b, vulkan, WORKLOADS.coding)
    expect(candidates.map((c) => c.gpuLayers)).toEqual([0])
  })

  it('VRAM unavailable does not empty the list (X14)', () => {
    const m = { ...machine(), vramBytes: { value: null, kind: 'unavailable' as const, reason: 'test' } }
    const { candidates } = generateCandidates(m, q14b, vulkan, WORKLOADS.coding)
    expect(candidates.length).toBeGreaterThan(0)
    expect(candidates[0].notes).toContain('VRAM size unknown; not pruned by VRAM, runtime guards apply')
  })
})
