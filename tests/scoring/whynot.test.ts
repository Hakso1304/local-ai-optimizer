// best.headline + Recommendation.whyNot on the real calibration fixtures (8B/14B full offload, heavy Qwen3.8-27B).
import { describe, expect, it } from 'vitest'
import type { QualityResult } from '../../src/shared/bench-types'
import { recommend } from '../../src/core/scoring/recommend'
import { inputs, load, machine } from './helpers'

const q5 = (id: string, rate: number): QualityResult[] =>
  ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context'].flatMap((category) => [0, 1, 2, 3, 4].map((i) => ({
    testId: `${id}-${category}-${i}`, category: category as QualityResult['category'], weight: 1, pass: i < rate * 5, score: 1, detail: ''
  })))

describe('14B (spill cliff at 32K) vs 8B full offload', () => {
  const f8 = load('calib-8b-rx9070.json'), f14 = load('calib-14b-rx9070.json')
  const M = machine(f14.vramBytes)
  const both = (q14: number) => [...inputs(f8), ...inputs(f14)].map((c) => ({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? q14 : 0.6) }))

  it('equal quality, Coding: the 14B loses on speed and its measured cliff/spill', () => {
    const r = recommend(both(0.6), M, 'coding')
    expect(r.best?.headline).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M @ 32K — 72.0 t/s, quality 60, no spill up to 64K')
    expect(r.whyNot).toEqual([
      { configId: 'qwen14b|ngl=all', model: 'Qwen2.5-14B-Instruct Q4_K_M',
        summary: 'Qwen2.5-14B-Instruct Q4_K_M: slower (51.2 vs 88.1 t/s at 16K), decode fell 51.2 → 26.4 t/s after 16K (shared-VRAM spill 1.05 GiB at 32K), practical context 16K vs 64K; Coding total 73 vs 79' },
      { configId: 'llama8b|ngl=20', model: 'Meta-Llama-3.1-8B-Instruct Q4_K_M',
        summary: 'Meta-Llama-3.1-8B-Instruct Q4_K_M (20/32 layers): ineligible — partial offload; a full-offload config of this model passed' }
    ])
  })

  it('14B quality 100, Reasoning: the faster 8B lost on quality', () => {
    const r = recommend(both(1), M, 'reasoning')
    expect(r.best?.headline).toBe('Qwen2.5-14B-Instruct Q4_K_M @ 16K — 51.2 t/s, quality 100, no spill up to 16K')
    expect(r.whyNot![0].summary).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M: faster (99.8 vs 57.1 t/s at 8K), practical context 64K vs 16K but quality −40 pts; Reasoning total 80 vs 97')
  })

  it('Document Analysis: a gated runner-up says which gate, with its numbers', () => {
    expect(recommend(both(1), M, 'document_analysis').whyNot![0].summary).toBe('Qwen2.5-14B-Instruct Q4_K_M: ineligible — practical context 16K is below 32K')
  })
})

describe('heavy: Qwen3.8-27B 55/65 layers vs Llama-3.1-8B', () => {
  const fh = load('calib-heavy-qwen38-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
  const MH = { ...machine(fh.vramBytes), vramInUseBytes: { value: fh.vramInUseBytes, kind: 'measured' as const } }
  const withQ = () => inputs(fh).map((c) => ({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? 1 : 0.6) }))

  it('Coding: the higher-quality 27B is explained once (not once per ngl), with both scoring contexts', () => {
    const r = recommend(withQ(), MH, 'coding')
    expect(r.best?.headline).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M @ 32K — 78.9 t/s, quality 60, no spill up to 32K')
    expect(r.whyNot!.map((w) => w.summary)).toEqual([
      'Qwen3.8-27B-UD-Q4_K_M (55/65 layers): quality +40 pts but slower (13.0 t/s at 8K vs 93.2 at 16K), practical context 8K vs 32K; Coding total 72 vs 80',
      'Qwen3.8-27B-UD-Q4_K_M (50/65 layers): ineligible — practical context 2K is below 8K'
    ])
  })

  it('Maximum Quality: partial-offload headline; the 8B lost on quality', () => {
    const r = recommend(withQ(), MH, 'max_quality')
    expect(r.best?.headline).toBe('Qwen3.8-27B-UD-Q4_K_M (55/65 layers) @ 8K — 13.0 t/s, quality 100, no spill up to 8K')
    expect(r.whyNot![0].summary).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M: faster (96.5 vs 13.0 t/s at 8K), practical context 32K vs 8K but quality −40 pts; Maximum Quality total 72 vs 97')
  })

  it('no winner → no whyNot', () => {
    expect(recommend([], MH, 'coding').whyNot).toEqual([])
  })
})
