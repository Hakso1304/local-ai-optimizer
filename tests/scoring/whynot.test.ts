// best.headline + Recommendation.whyNot (rule I-7.2: quality delta, speed at the scoring contexts, ceiling, gate/totals)
// on the real calibration fixtures (8B/14B full offload, heavy Qwen3.8-27B) with synthetic measured quality.
import { describe, expect, it } from 'vitest'
import { recommend } from '../../src/core/scoring/recommend'
import { inputs, load, machine, q5 } from './helpers'

describe('14B (spill cliff at 32K) vs 8B full offload', () => {
  const f8 = load('calib-8b-rx9070.json'), f14 = load('calib-14b-rx9070.json')
  const M = machine(f14.vramBytes)
  const both = (q14: number) => [...inputs(f8), ...inputs(f14)].map((c) => ({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? q14 : 0.6) }))

  it('equal quality, Coding: the 14B loses on speed and its measured cliff/spill', () => {
    const r = recommend(both(0.6), M, 'coding')
    expect(r.best?.headline).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M @ 32K — 72.0 t/s, quality 60 ± 20, no spill up to 64K')
    expect(r.whyNot).toEqual([
      { configId: 'qwen14b|ngl=all', model: 'Qwen2.5-14B-Instruct Q4_K_M',
        summary: 'Qwen2.5-14B-Instruct Q4_K_M: same quality (60 ± 20 vs 60 ± 20); decode 51.2 vs 88.1 t/s at 16K; practical context 16K vs 64K (decode fell 51.2 → 26.4 t/s after 16K, shared-VRAM spill 1.05 GiB at 32K); Coding total 73 vs 79' },
      { configId: 'llama8b|ngl=20', model: 'Meta-Llama-3.1-8B-Instruct Q4_K_M',
        summary: 'Meta-Llama-3.1-8B-Instruct Q4_K_M (20/32 layers): same quality (60 ± 20 vs 60 ± 20); decode 17.5 t/s at 8K vs 88.1 at 16K; practical context 8K vs 64K; ineligible — [I-3.9] partial offload; a full-offload config of this model ran (llama8b|ngl=all)' }
    ])
  })

  it('14B quality 100, Reasoning: the faster 8B lost on quality (quality delta leads)', () => {
    const r = recommend(both(1), M, 'reasoning')
    expect(r.best?.headline).toBe('Qwen2.5-14B-Instruct Q4_K_M @ 16K — 51.2 t/s, quality 100 ± 19, no spill up to 16K')
    expect(r.whyNot![0].summary).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M: quality −40 pts (60 ± 23 vs 100 ± 19); decode 99.8 vs 57.1 t/s at 8K; practical context 64K vs 16K; Reasoning total 80 vs 97')
  })

  it('Document Analysis: a gated runner-up says which gate (rule-cited), with its numbers', () => {
    expect(recommend(both(1), M, 'document_analysis').whyNot![0].summary).toBe(
      'Qwen2.5-14B-Instruct Q4_K_M: quality +40 pts (100 ± 16 vs 60 ± 19); decode 51.2 t/s at 16K vs 52.4 at 64K; practical context 16K vs 64K (decode fell 51.2 → 26.4 t/s after 16K, shared-VRAM spill 1.05 GiB at 32K); ineligible — [I-2.7] practical context 16K is below 32K')
  })
})

describe('heavy: Qwen3.8-27B 55/65 layers vs Llama-3.1-8B', () => {
  const fh = load('calib-heavy-qwen38-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
  const MH = { ...machine(fh.vramBytes), vramInUseBytes: { value: fh.vramInUseBytes, kind: 'measured' as const } }
  const withQ = () => inputs(fh).map((c) => ({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? 1 : 0.6) }))

  it('Coding, quality 100 vs 60 (bands apart): the 27B wins; the faster 8B says it was outranked on quality', () => {
    const r = recommend(withQ(), MH, 'coding')
    expect(r.best?.headline).toBe('Qwen3.8-27B-UD-Q4_K_M (55/65 layers) @ 8K — 13.0 t/s, quality 100 ± 17, no spill up to 8K')
    expect(r.whyNot!.map((w) => w.summary)).toEqual([
      'Meta-Llama-3.1-8B-Instruct Q4_K_M: quality −40 pts (60 ± 20 vs 100 ± 17); decode 93.2 t/s at 16K vs 13.0 at 8K; practical context 32K vs 8K; Coding total 80 vs 72 — outranked because quality decides (bands apart, [I-5.2])',
      'Qwen3.8-27B-UD-Q4_K_M (50/65 layers): same quality (100 ± 17 vs 100 ± 17); decode 10.7 t/s at 2K vs 13.0 at 8K; practical context 2K vs 8K; ineligible — [I-2.7] practical context 2K is below 8K'
    ])
  })

  it('Maximum Quality: partial-offload headline; the 8B lost on quality', () => {
    const r = recommend(withQ(), MH, 'max_quality')
    expect(r.best?.headline).toBe('Qwen3.8-27B-UD-Q4_K_M (55/65 layers) @ 8K — 13.0 t/s, quality 100 ± 12, no spill up to 8K')
    expect(r.whyNot![0].summary).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M: quality −40 pts (60 ± 14 vs 100 ± 12); decode 96.5 vs 13.0 t/s at 8K; practical context 32K vs 8K; Maximum Quality total 72 vs 97')
  })

  it('no winner → no whyNot', () => {
    expect(recommend([], MH, 'coding').whyNot).toEqual([])
  })
})
