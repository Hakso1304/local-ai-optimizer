// KV layout for hybrid / sliding-window archs, from the real GGUF headers in D:\llm-models (read 2026-09-27):
//   Qwen3.8-27B-UD-Q4_K_M  arch qwen35: 65 layers, head_count 24, head_count_kv 4, key/value_length 256,
//                          full_attention_interval 4 (Gated-DeltaNet hybrid: only every 4th layer has a KV cache).
//   gemma-4-26B-A4B-it     arch gemma4: 30 layers, head_count 16, head_count_kv = per-layer array [8×5, 2, …],
//                          key/value_length 512 (global), *_swa 256, sliding_window 1024, sliding_window_pattern.
import { describe, expect, it } from 'vitest'
import type { ModelMeta } from '../../src/shared/bench-types'
import { DEFAULT_CANDIDATE_RULES, estimateMemory, generateCandidates, isLowConfidence, kvLayout } from '../../src/core/benchmark/candidates'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { machine } from './helpers'

const qwen35: ModelMeta = {
  id: 'D:/llm-models/Qwen3.8-27B-UD-Q4_K_M.gguf', name: 'Qwen3.8-27B', fileBytes: 16464440224, paramCount: 27e9, quant: 'Q4_K_M', arch: 'qwen35',
  ctxTrain: 262144, layers: 65, nEmbd: 5120, heads: 24, headsKv: 4, keyLength: 256, valueLength: 256, nVocab: 248320, slidingWindow: null
}
const gemmaPattern = Array.from({ length: 30 }, (_, i) => (i + 1) % 6 !== 0) // false = global layer (5, 11, 17, 23, 29)
const gemma4: ModelMeta = {
  id: 'D:/llm-models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf', name: 'Gemma-4-26B-A4B-It', fileBytes: 16947541728, paramCount: 26e9, quant: 'Q4_K_M', arch: 'gemma4',
  ctxTrain: 262144, layers: 30, nEmbd: 2816, heads: 16, headsKv: 16, keyLength: 512, valueLength: 512, nVocab: 262144, slidingWindow: 1024
}
const gemma4Layout: ModelMeta = {
  ...gemma4, headsKv: 8, headsKvPerLayer: gemmaPattern.map((swa) => (swa ? 8 : 2)), slidingWindowPattern: gemmaPattern, keyLengthSwa: 256, valueLengthSwa: 256
}
const heavy = { ...DEFAULT_CANDIDATE_RULES, heavyMode: true }

describe('kvLayout', () => {
  it('qwen35 without layout keys: upper bound (all 65 layers), flagged as fallback — never 0', () => {
    const k = kvLayout(qwen35, 1, 'f16')
    expect(k.bytes).toBe(65 * 4 * 512 * 2)
    expect(k.source).toMatch(/^fallback upper bound: no per-arch KV layout \(qwen35/)
    expect(isLowConfidence(qwen35)).toBe(true)
  })
  it('qwen35 with full_attention_interval 4: only 16/65 layers keep KV', () => {
    const k = kvLayout({ ...qwen35, fullAttentionInterval: 4 }, 1, 'f16')
    expect(k.bytes).toBe(16 * 4 * 512 * 2)
    expect(k.source).toBe('declared layout: 16/65 attention layers')
  })
  it('gemma4 with SWA pattern + per-layer heads: global layers grow with ctx, SWA layers cap at window + ubatch', () => {
    const at = (ctx: number) => kvLayout(gemma4Layout, ctx, 'f16').bytes
    expect(at(1)).toBe(5 * 2 * 1024 * 2 + 25 * 8 * 512 * 2) // both kinds hold 1 token
    expect(at(32768)).toBe(5 * 2 * 1024 * 2 * 32768 + 25 * 8 * 512 * 2 * 1536)
    expect(kvLayout(gemma4, 32768, 'f16').bytes).toBeGreaterThan(at(32768) * 10) // no layout → upper bound, far higher
  })
  it('missing head data → unknown, never throws', () => {
    const k = kvLayout({ ...qwen35, headsKv: 0 }, 4096, 'f16')
    expect(k).toMatchObject({ bytes: 0, unknown: true })
    expect(() => estimateMemory({ ...qwen35, headsKv: 0 }, 30, 4096, 'f16')).not.toThrow()
  })
})

describe('heavy-mode candidates on the real heavy models (16 GB VRAM)', () => {
  it('the upper bound never gives a smaller VRAM estimate than the declared layout (no silent 0)', () => {
    for (const ctx of [2048, 32768]) {
      expect(estimateMemory(gemma4, 20, ctx, 'f16').vramBytes).toBeGreaterThanOrEqual(estimateMemory(gemma4Layout, 20, ctx, 'f16').vramBytes)
      expect(estimateMemory(qwen35, 40, ctx, 'f16').vramBytes).toBeGreaterThanOrEqual(estimateMemory({ ...qwen35, fullAttentionInterval: 4 }, 40, ctx, 'f16').vramBytes)
    }
  })
  it('with the declared layout, gemma4 reaches much longer ctx than with the fallback', () => {
    const last = (m: ModelMeta) => Math.max(...generateCandidates(machine(), m, { backend: 'vulkan' }, WORKLOADS.document_analysis, heavy).candidates.flatMap((c) => c.ctxSteps))
    expect(last(gemma4Layout)).toBeGreaterThan(last(gemma4))
    const [first] = generateCandidates(machine(), gemma4, { backend: 'vulkan' }, WORKLOADS.document_analysis, heavy).candidates
    expect(first.notes.join(' ')).toMatch(/KV estimate: fallback upper bound/)
  })
  it('KV size unknown → only conservative probes (max ngl @2K, -nkvo, CPU baseline), reasons say so', () => {
    const { candidates } = generateCandidates(machine(), { ...qwen35, headsKv: 0 }, { backend: 'vulkan' }, WORKLOADS.max_quality, heavy)
    expect(candidates.length).toBeLessThanOrEqual(3)
    expect(candidates.every((c) => c.degradedReason!.endsWith('; KV size unknown'))).toBe(true)
    expect(candidates.some((c) => c.kvOffload === false)).toBe(true)
    expect(candidates.some((c) => c.gpuLayers === 0)).toBe(true)
  })
})
