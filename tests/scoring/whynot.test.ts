// best.headline + Recommendation.whyNot (rule I-7.3: paired quality difference, speed at the scoring rungs, coverage,
// gate / totals) on the real calibration fixtures (8B/14B full offload, heavy Qwen3.8-27B) with synthetic measured quality.
import { describe, expect, it } from 'vitest'
import { recommend } from '../../src/core/scoring/recommend'
import { inputs, load, machine, q5, withOffProof } from './helpers'

describe('14B (spill cliff at 32K) vs 8B full offload', () => {
  const f8 = load('calib-8b-rx9070.json'), f14 = load('calib-14b-rx9070.json')
  const M = machine(f14.vramBytes)
  const both = (q14: number) => [...inputs(f8), ...inputs(f14)].map((c) => ({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? q14 : 0.6) }))

  it('equal quality, Coding: the 14B loses with quality neutralized, and its measured cliff/spill is shown', () => {
    const r = recommend(both(0.6), M, 'coding')
    expect(r.best?.headline).toBe('Meta-Llama-3.1-8B-Instruct Q4_K_M @ 32K — 72.0 t/s, quality 60 [35, 81], no spill up to 64K')
    expect(r.whyNot!.map((w) => [w.configId, w.summary])).toEqual([
      ['qwen14b|ngl=all', '[I-7.3] Qwen2.5-14B-Instruct Q4_K_M: quality +0 [+0, +0] (60 [35, 81] vs 60 [35, 81]); decode 51.2 vs 88.1 t/s at 16K; largest clean context 16K vs 64K (decode fell 51.2 → 26.4 t/s after 16K, shared-VRAM spill 1.05 GiB at 32K); Coding totals without quality 49.0 vs 54.9 (quality neutralized)'],
      ['llama8b|ngl=20', '[I-7.3] Meta-Llama-3.1-8B-Instruct Q4_K_M (20/32 layers): quality +0 [+0, +0] (60 [35, 81] vs 60 [35, 81]); decode 17.5 t/s at 8K vs 88.1 at 16K; largest clean context 8K vs 64K; ineligible — [I-7.6] partial offload dominated by llama8b|ngl=all, a same-model full offload meeting the same hard constraints']
    ])
  })

  it('14B quality 100, Reasoning: the faster 8B lost on a paired quality difference that excludes 0', () => {
    const r = recommend(both(1), M, 'reasoning')
    expect(r.best?.headline).toBe('Qwen2.5-14B-Instruct Q4_K_M @ 16K — 51.2 t/s, quality 100 [72, 100], no spill up to 16K')
    expect(r.whyNot![0].summary).toBe('[I-7.3] Meta-Llama-3.1-8B-Instruct Q4_K_M: quality −40 [−71, −10] (60 [31, 83] vs 100 [72, 100]); decode 99.8 vs 57.1 t/s at 8K; largest clean context 64K vs 16K; Reasoning total 80 vs 97')
  })

  it('Document Analysis: a gated runner-up says which gate (rule-cited), with its numbers', () => {
    expect(recommend(both(1), M, 'document_analysis').whyNot![0].summary).toBe(
      '[I-7.3] Qwen2.5-14B-Instruct Q4_K_M: quality +40 [+13, +67] (100 [80, 100] vs 60 [36, 80]); decode 51.2 t/s at 16K vs 52.4 at 64K; largest clean context 16K vs 64K (decode fell 51.2 → 26.4 t/s after 16K, shared-VRAM spill 1.05 GiB at 32K); ineligible — [I-2.7] largest clean context 16K is below 32K (half the Document Analysis target)')
  })
})

describe('heavy: Qwen3.8-27B 55/65 layers vs Llama-3.1-8B', () => {
  const fh = load('calib-heavy-qwen38-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
  const MH = { ...machine(fh.vramBytes), vramInUseBytes: { value: fh.vramInUseBytes, kind: 'measured' as const } }
  const withQ = () => inputs(fh).map((c) => withOffProof({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? 1 : 0.6) }))

  it('Coding: the 27B (clean only to 8K) is provisional below the common 16K rung; its why-not says so', () => {
    const r = recommend(withQ(), MH, 'coding')
    expect(r.best?.configId).toBe('llama8b|ngl=all')
    expect(r.whyNot!.find((w) => w.configId === 'qwen38|ngl=55')!.summary).toMatch(/provisional — genSpeed unmatched-rung/)
  })

  it('Maximum Quality: partial-offload headline; the 8B lost on quality', () => {
    const r = recommend(withQ(), MH, 'max_quality')
    expect(r.best?.headline).toBe('Qwen3.8-27B-UD-Q4_K_M (55/65 layers) @ 8K — 13.0 t/s, quality 100 [87, 100], no spill up to 8K')
    expect(r.whyNot![0].summary).toBe('[I-7.3] Meta-Llama-3.1-8B-Instruct Q4_K_M: quality −40 [−60, −22] (60 [41, 76] vs 100 [87, 100]); decode 96.5 vs 13.0 t/s at 8K; largest clean context 32K vs 8K; Maximum Quality total 72 vs 97')
  })

  it('no winner → no whyNot', () => {
    expect(recommend([], MH, 'coding').whyNot).toEqual([])
  })
})
