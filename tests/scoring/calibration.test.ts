// Real measurements, RX 9070 XT 16 GB (docs/calibration-2026-09-27.md → tests/fixtures/scoring/calib-8b-rx9070.json).
import { describe, expect, it } from 'vitest'
import type { ModelMeta, QualityResult, WorkloadId } from '../../src/shared/bench-types'
import { estimateMemory, generateCandidates } from '../../src/core/benchmark/candidates'
import { detectCliffs } from '../../src/core/scoring/cliff'
import { componentScores } from '../../src/core/scoring/components'
import { recommend } from '../../src/core/scoring/recommend'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { inputs, load, machine, toRun } from './helpers'

const MiB = 1024 ** 2
const f = load('calib-8b-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
const M = { ...machine(f.vramBytes), vramInUseBytes: { value: f.vramInUseBytes, kind: 'measured' as const } }
const llama = f.models[0]
const FULL = 'llama8b|ngl=all'

/** Fixture runs + the config the generator really produces (so skippedSteps/notes are real). */
function candidates(workload: WorkloadId) {
  const gen = generateCandidates(M, llama, { backend: 'vulkan' }, WORKLOADS[workload]).candidates[0]
  return inputs(f).map((c) => (c.config.id === FULL ? { ...c, config: { ...gen, id: FULL } } : c))
}

describe('calibration: 8B Q4_K_M full offload 2K→64K', () => {
  it('gradual decode decay (−4…−27%/step) is not a cliff; ceiling is 64K', () => {
    const r = detectCliffs(f.runs.filter((x) => x.configId === FULL).map(toRun), f.vramBytes)
    expect(r.steps.every((s) => s.verdict === 'pass')).toBe(true)
    expect(r.practicalContextCeiling.value).toBe(65536)
  })

  it('KV estimate matches measured buffers within 5% (8B 0.125 MiB/token, Qwen2.5-1.5B 0.0273 MiB/token)', () => {
    const qwen: ModelMeta = { ...llama, id: 'qwen', layers: 28, nEmbd: 1536, heads: 12, headsKv: 2, keyLength: 128, valueLength: 128 }
    expect(estimateMemory(llama, 32, 1, 'f16').kvBytes / MiB).toBeCloseTo(0.125, 3)
    expect(Math.abs(estimateMemory(qwen, 28, 1, 'f16').kvBytes / MiB / 0.0273 - 1)).toBeLessThan(0.05)
  })

  it('VRAM estimate is within −5%…+25% of measured at every rung (the 1 GiB budget margin covers the residual)', () => {
    for (const r of f.runs.filter((x) => x.configId === FULL)) {
      const est = estimateMemory(llama, 32, r.ctx, 'f16').vramBytes
      expect(est / r.peakVramBytes!, `${r.ctx}`).toBeGreaterThanOrEqual(0.95)
      expect(est / r.peakVramBytes!, `${r.ctx}`).toBeLessThanOrEqual(1.25)
    }
  })

  // TTFT is the usability cliff (0.5 s@2K … 32 s@64K): each workload's recommended ctx is the largest step whose
  // full-prompt TTFT fits its tolerance, capped at its maxContext.
  const expected: Record<WorkloadId, number> = {
    fast_assistant: 4096, general_chat: 16384, reasoning: 16384, max_quality: 16384,
    coding: 32768, long_context_coding: 65536, document_analysis: 65536
  }
  for (const [w, ctx] of Object.entries(expected) as [WorkloadId, number][]) {
    it(`${w}: recommends full offload at ${ctx / 1024}K`, () => {
      const rec = recommend(candidates(w), M, w)
      expect(rec.best?.configId).toBe(FULL)
      expect(rec.best?.score.recommendedCtx).toBe(ctx)
      expect(rec.best?.score.referenceCtx).toBe(Math.min(ctx, WORKLOADS[w].targetContext)) // scoring stays at target
      expect(rec.reasons.some((x) => x.startsWith(`Recommended context ${ctx / 1024}K: TTFT`))).toBe(true)
    })
  }

  it('reasons name the TTFT at the chosen ctx and the memory bound (not "no cliff")', () => {
    const rec = recommend(candidates('coding'), M, 'coding')
    // Scored at the workload target (16K) but recommended at 32K: the reason shows both.
    expect(rec.reasons).toContain('Recommended context 32K: TTFT 12.0 s for a full prompt (tolerance 15 s), decode 72.0 t/s (scored at 16K: TTFT 4.9 s, decode 88.1 t/s)')
    const fast = recommend(candidates('fast_assistant'), M, 'fast_assistant') // recommended = scored = 4K → no suffix
    expect(fast.reasons.find((x) => x.startsWith('Recommended context 4K'))).not.toMatch(/scored at/)
    expect(rec.reasons.find((x) => x.startsWith('Practical context 64K'))).toMatch(/memory-bound at 64K \(128K: est\. VRAM .* > budget/)
  })

  it('partial offload (ngl 20: 17.5 t/s; ngl 0: 7.1 t/s) is never recommended over a passing full offload', () => {
    for (const w of Object.keys(expected) as WorkloadId[]) {
      const rec = recommend(candidates(w), M, w)
      for (const id of ['llama8b|ngl=20', 'llama8b|ngl=0']) {
        const s = rec.ranked.find((x) => x.configId === id)!
        expect(s.eligible, `${w} ${id}`).toBe(false)
        expect(s.gateFailures).toContain('partial offload; a full-offload config of this model passed')
      }
    }
  })

  it('unavailable telemetry scores neutral and is flagged, never 0 (<1 s requests get 0 typeperf samples)', () => {
    const full = inputs(f).find((x) => x.config.id === FULL)!
    const blind = { ...full, runs: full.runs.map((r) => ({ ...r, peakVramBytes: { value: null, kind: 'unavailable' as const, reason: 'no telemetry samples' } })) }
    const mem = componentScores(blind, M, WORKLOADS.coding).components.memory
    expect(mem.score).toBe(50)
    expect(mem.note).toMatch(/^unknown \(VRAM peak unavailable\)/)
  })
})

describe('calibration: Qwen2.5-14B Q4_K_M real spill cliff at 32K (per-PID telemetry)', () => {
  const f14 = load('calib-14b-rx9070.json')
  const q14 = f14.models[0]
  const FULL14 = 'qwen14b|ngl=all'
  const both = () => [...candidates('document_analysis'), ...inputs(f14)]
  // Synthetic measured quality: 14B passes every test (Q 100), 8B passes 3 of 5 per category (Q 60).
  const passAll = (modelId: string, best: boolean): QualityResult[] =>
    ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context'].flatMap((category) => [0, 1, 2, 3, 4].map((i) => {
      const pass = best || i < 3
      return { testId: `${modelId}-${category}-${i}`, category: category as QualityResult['category'], weight: 1, pass, score: pass ? 1 : 0, detail: '' }
    }))

  it('practical 16K, degraded 32K, limited by cliff; decode_drop + shared_spill fire, vram_spill corroborates (83% VRAM)', () => {
    const r = detectCliffs(f14.runs.filter((x) => x.configId === FULL14).map(toRun), f14.vramBytes)
    expect(r).toMatchObject({ practicalContextCeiling: { value: 16384 }, degradedContextCeiling: { value: 32768 }, limitedBy: 'cliff', spillFreeUpTo: 16384 })
    const at32 = r.steps.find((s) => s.ctx === 32768)!
    expect(at32.verdict).toBe('degraded')
    expect(at32.reasons.map((x) => x.code)).toEqual(['decode_drop', 'vram_spill', 'shared_spill'])
    expect(at32.reasons[0].message).toBe('decode TPS fell 48% between 16K and 32K (51.2 → 26.4 t/s)')
    expect(r.steps.at(-1)!.verdict).toBe('fail')
  })

  it('Document Analysis: 14B at 16K beats 8B when its measured quality is higher', () => {
    const inp = both().map((c) => ({ ...c, quality: passAll(c.model.id, c.model.id === q14.id) }))
    const rec = recommend(inp, M, 'document_analysis')
    expect(rec.best?.configId).toBe(FULL14)
    expect(rec.best?.score.recommendedCtx).toBe(16384) // the spilled 32K step is never recommended
  })

  it('Document Analysis on priors only: 8B wins (64K ceiling + faster prefill outweigh the 14B quality prior)', () => {
    expect(recommend(both(), M, 'document_analysis').best?.configId).toBe(FULL)
  })

  it('Fast Assistant never picks the 14B, even with higher quality', () => {
    for (const withQuality of [false, true]) {
      const inp = [...candidates('fast_assistant'), ...inputs(f14)].map((c) => ({ ...c, quality: withQuality ? passAll(c.model.id, c.model.id === q14.id) : [] }))
      expect(recommend(inp, M, 'fast_assistant').best?.configId).toBe(FULL)
    }
  })

  it('partial offload (ngl 30: 5.6 t/s) is never preferred over the spilled full offload (26.4 t/s)', () => {
    for (const w of ['document_analysis', 'long_context_coding', 'coding', 'general_chat'] as const) {
      const s = recommend(inputs(f14), M, w).ranked.find((x) => x.configId === 'qwen14b|ngl=30')!
      expect(s.eligible, w).toBe(false)
    }
  })
})
