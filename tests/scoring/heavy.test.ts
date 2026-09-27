// Heavy-model mode: a ~27B Q4 (16.1 GiB file) on the RX 9070 XT 16 GB / 31 GB RAM machine.
import { describe, expect, it } from 'vitest'
import type { CandidateInput, ModelMeta, QualityResult } from '../../src/shared/bench-types'
import { DEFAULT_CANDIDATE_RULES, generateCandidates, rulesForRequest } from '../../src/core/benchmark/candidates'
import { recommend } from '../../src/core/scoring/recommend'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { inputs, load, machine, toRun } from './helpers'

const GiB = 1024 ** 3
const f8 = load('calib-8b-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
const M = { ...machine(f8.vramBytes), vramInUseBytes: { value: f8.vramInUseBytes, kind: 'measured' as const } }
const m27: ModelMeta = {
  id: 'q27b', name: 'Synthetic 27B Q4_K_M', fileBytes: Math.round(16.1 * GiB), paramCount: 27.2e9, quant: 'Q4_K_M', arch: 'qwen2',
  ctxTrain: 32768, layers: 64, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064, slidingWindow: null
}
const heavy = { ...DEFAULT_CANDIDATE_RULES, heavyMode: true }

describe('heavy-model candidate generation (27B, 16.1 GiB, 16 GB VRAM)', () => {
  it('normal mode: nothing, with a "does not fit — enable heavy-model mode" reason', () => {
    const r = generateCandidates(M, m27, { backend: 'vulkan' }, WORKLOADS.max_quality)
    expect(r.candidates).toEqual([])
    expect(r.rejected.map((x) => x.reason).join(' ')).toMatch(/full GPU offload does not fit — enable heavy-model mode/)
  })

  it('heavy mode: ≤4 partial configs (max layers @2K, @target, -nkvo, CPU baseline), all flagged expectDegraded', () => {
    const { candidates } = generateCandidates(M, m27, { backend: 'vulkan' }, WORKLOADS.max_quality, heavy)
    expect(candidates.length).toBeLessThanOrEqual(4)
    expect(candidates.every((c) => c.expectDegraded && !c.gpuLayersAll)).toBe(true)
    const [short, target, nkvo, cpu] = candidates
    expect(short.gpuLayers).toBeGreaterThan(target.gpuLayers) // fewer layers fit once the target ctx's KV is on the GPU
    expect(target.ctxSteps.at(-1)).toBe(8192) // Maximum Quality target
    expect(nkvo).toMatchObject({ kvOffload: false, id: expect.stringMatching(/\|nkvo$/) })
    expect(nkvo.ctxSteps.at(-1)).toBeGreaterThan(target.ctxSteps.at(-1)!) // KV in RAM: reaches past the target ctx
    expect(cpu).toMatchObject({ gpuLayers: 0, device: null })
    expect(target.degradedReason).toMatch(/^weights 16\.1 GiB > VRAM budget 13\.7 GiB; \d+\/64 layers on GPU$/)
    expect(nkvo.degradedReason).toMatch(/KV cache in system RAM \(-nkvo\)$/)
  })

  it('heavy RAM check uses the resident part + a 2 GiB reserve (never below 2 GiB)', () => {
    const tight = { ...M, ramAvailableBytes: { value: 4 * GiB, kind: 'measured' as const } }
    const { candidates } = generateCandidates(tight, m27, { backend: 'vulkan' }, WORKLOADS.max_quality, { ...heavy, heavyRamReserveBytes: 0 })
    for (const c of candidates) expect(c.estRamBytes.value!, c.id).toBeLessThanOrEqual(2 * GiB) // 4 GiB − max(2, 0) GiB
  })

  it('rulesForRequest carries heavyMode so re-planning yields the same configIds', () => {
    expect(rulesForRequest({ heavyMode: true }).heavyMode).toBe(true)
    expect(rulesForRequest({}).heavyMode).toBe(false)
  })
})

describe('heavy-model recommendation (27B partial @ ~8 t/s vs 8B full offload)', () => {
  // Synthetic measured quality: 27B passes everything (Q 100), 8B passes 3 of 5 per category (Q 60).
  const quality = (id: string, best: boolean): QualityResult[] =>
    ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context'].flatMap((category) => [0, 1, 2, 3, 4].map((i) => ({
      testId: `${id}-${category}-${i}`, category: category as QualityResult['category'], weight: 1, pass: best || i < 3, score: 1, detail: ''
    })))
  const cand27 = generateCandidates(M, m27, { backend: 'vulkan' }, WORKLOADS.max_quality, heavy).candidates[1] // target-ctx config
  const run27 = (ctx: number, decode: number, prefill: number) => toRun({
    configId: cand27.id, model: m27.id, ctx, gpuLayers: cand27.gpuLayers, threads: 8, status: 'ok', promptTokens: ctx * 0.75, loadMs: 9000,
    ttftMs: ((ctx * 0.75) / prefill) * 1000, prefillTps: prefill, decodeTps: decode, totalMs: null,
    peakRamBytes: 3 * GiB, peakVramBytes: 13.5 * GiB, peakSharedGpuBytes: 0.05 * GiB, cpuAvgPct: 60, gpuAvgPct: 40
  })
  const heavy27: CandidateInput = { config: cand27, model: m27, runs: [run27(2048, 8.4, 650), run27(4096, 8.1, 620), run27(8192, 7.8, 580)], quality: quality(m27.id, true) }
  const full8 = inputs(f8).filter((c) => c.config.id === 'llama8b|ngl=all').map((c) => ({ ...c, quality: quality(c.model.id, false) }))
  const all = [heavy27, ...full8]

  it('Maximum Quality: the 27B partial offload wins on measured quality, with a degraded-speed reason', () => {
    const rec = recommend(all, M, 'max_quality')
    expect(rec.best?.configId).toBe(cand27.id)
    expect(rec.reasons).toContain(`Partial GPU offload (${cand27.gpuLayers}/64 layers) — degraded speed expected: decode 7.8 t/s (${cand27.degradedReason})`)
  })

  it('Fast Assistant and Coding: the 27B fails the decode gate; the 8B wins', () => {
    for (const w of ['fast_assistant', 'coding'] as const) {
      const rec = recommend(all, M, w)
      expect(rec.best?.configId, w).toBe('llama8b|ngl=all')
      expect(rec.ranked.find((s) => s.configId === cand27.id)!.gateFailures.join(' '), w).toMatch(/below the \d+ t\/s minimum/)
    }
  })
})
