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
    const { candidates, rejected } = generateCandidates(M, m27, { backend: 'vulkan' }, WORKLOADS.max_quality, heavy)
    expect(candidates.length).toBeLessThanOrEqual(4)
    expect(candidates.every((c) => c.expectDegraded && !c.gpuLayersAll)).toBe(true)
    const [short, target] = candidates
    // 8K target: the target's KV costs < 8 layers vs 2K → no -nkvo rung (it lost to dropping ~5 layers at 2K).
    expect(candidates.some((c) => c.kvOffload === false)).toBe(false)
    const long = generateCandidates(M, m27, { backend: 'vulkan' }, WORKLOADS.long_context_coding, heavy).candidates
    const nkvo = long.find((c) => c.kvOffload === false)!
    // 16.1 GiB file > 50% of 31 GiB RAM → the CPU baseline is skipped (it drove a real host to 1 GiB free).
    expect(candidates.some((c) => c.gpuLayers === 0)).toBe(false)
    expect(rejected.map((r) => r.reason)).toContain('CPU baseline skipped: model is >50% of system RAM')
    // With 64 GiB RAM the same model is < 50% of RAM → the CPU baseline is generated.
    const bigRam = { ...M, ramTotalBytes: { value: 64 * GiB, kind: 'declared' as const }, ramAvailableBytes: { value: 50 * GiB, kind: 'measured' as const } }
    const cpu = generateCandidates(bigRam, m27, { backend: 'vulkan' }, WORKLOADS.max_quality, heavy).candidates.find((c) => c.gpuLayers === 0)!
    expect(short.gpuLayers).toBeGreaterThan(target.gpuLayers) // fewer layers fit once the target ctx's KV is on the GPU
    expect(target.ctxSteps.at(-1)).toBe(8192) // Maximum Quality target
    expect(nkvo).toMatchObject({ kvOffload: false, id: expect.stringMatching(/\|nkvo$/) })
    expect(nkvo.ctxSteps.at(-1)).toBe(32768) // long target (declared 32K): KV in RAM reaches it
    expect(nkvo.degradedReason).toMatch(/KV on CPU: slower decode than dropping ~5 layers at short ctx; useful only for long context$/)
    expect(cpu).toMatchObject({ gpuLayers: 0, device: null })
    expect(target.degradedReason).toMatch(/^weights 16\.1 GiB > VRAM budget 13\.7 GiB; \d+\/64 layers on GPU$/)
    expect(nkvo.degradedReason).toMatch(/KV cache in system RAM \(-nkvo\); /)
  })

  it('heavy RAM check uses the resident part + a 4 GiB reserve (never below 4 GiB)', () => {
    const tight = { ...M, ramAvailableBytes: { value: 7 * GiB, kind: 'measured' as const } }
    const { candidates } = generateCandidates(tight, m27, { backend: 'vulkan' }, WORKLOADS.max_quality, { ...heavy, heavyRamReserveBytes: 0 })
    for (const c of candidates) expect(c.estRamBytes.value!, c.id).toBeLessThanOrEqual(3 * GiB) // 7 GiB − max(4, 0) GiB
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
      testId: `${category}-${i}`, category: category as QualityResult['category'], weight: 1, pass: best || i < 3, score: 1, detail: ''
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
    expect(rec.reasons).toContain(`[I-3.5] ${cand27.id}: ${cand27.gpuLayers}/64 layers on GPU: 7.8 t/s at 8K.`)
  })

  it('Fast Assistant and Coding: the 27B fails the decode gate; the 8B wins', () => {
    for (const w of ['fast_assistant', 'coding'] as const) {
      const rec = recommend(all, M, w)
      expect(rec.best?.configId, w).toBe('llama8b|ngl=all')
      expect(rec.ranked.find((s) => s.configId === cand27.id)!.gateFailures.join(' '), w).toMatch(/\[I-3\.1\] decode [\d.]+ t\/s at \d+K is below the .* gate \d+ t\/s/)
    }
  })
})

describe('heavy-model calibration: Qwen3.8-27B 55/65 layers vs Llama-3.1-8B full offload (real RX 9070 XT run)', () => {
  const fh = load('calib-heavy-qwen38-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
  const MH = { ...machine(fh.vramBytes), vramInUseBytes: { value: fh.vramInUseBytes, kind: 'measured' as const } }
  const Q = 'qwen38|ngl=55', L8 = 'llama8b|ngl=all'
  const q5 = (id: string, rate: number): QualityResult[] =>
    ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context'].flatMap((category) => [0, 1, 2, 3, 4].map((i) => ({
      testId: `${category}-${i}`, category: category as QualityResult['category'], weight: 1, pass: i < rate * 5, score: 1, detail: ''
    })))
  const withQ = (qQwen: number, q8: number) => inputs(fh).map((c) => ({ ...c, quality: q5(c.model.id, c.model.id.startsWith('qwen') ? qQwen : q8) }))

  it('the 55/65 partial offload (decode 12–13 t/s, no spill) stays eligible for Maximum Quality', () => {
    const r = recommend(withQ(1, 0.6), MH, 'max_quality').ranked.find((s) => s.configId === Q)!
    expect(r.eligible).toBe(true) // decode 13 ≥ Maximum Quality gate 2; TTFT 6.7 s ≤ 20 s at 8K
  })

  it('Maximum Quality: the 27B wins when its measured quality is higher', () => {
    const rec = recommend(withQ(1, 0.6), MH, 'max_quality')
    expect(rec.best?.configId).toBe(Q)
    expect(rec.reasons.join('\n')).toMatch(/\[I-7\.4\] Quality vs speed: chosen for quality: quality \+40 \[\+\d+, \+\d+\] vs llama8b\|ngl=all; decode 13\.0 vs 96\.5 t\/s/)
    expect(rec.reasons.join('\n')).toMatch(/\[I-5\.4\] Qwen3\.8-27B-UD-Q4_K_M: quality measured with thinking off \(T=0\)/)
  })

  it('Coding: the 27B stops at 8K below the common 16K rung → its speed basis is unmatched, so it is provisional (F2)', () => {
    for (const [a, b] of [[0.8, 0.8], [1, 0.6]]) {
      const r = recommend(withQ(a, b), MH, 'coding')
      expect(r.ranked.find((s) => s.configId === Q)!.eligible, `${a}/${b}`).toBe(true) // 13 t/s passes the 10 t/s Coding gate
      expect(r.best?.configId, `${a}/${b}`).toBe(L8)
      expect(r.decisionTrace!.candidates.find((c) => c.configId === Q)!.undecided.join(' ')).toMatch(/unmatched-rung/)
    }
  })

  it('Fast Assistant rejects the 27B on the decode gate (13 < 30 t/s)', () => {
    const rec = recommend(withQ(1, 0.6), MH, 'fast_assistant')
    expect(rec.best?.configId).toBe(L8)
    expect(rec.ranked.find((s) => s.configId === Q)!.gateFailures.join(' ')).toMatch(/\[I-3\.1\] decode 1[23]\.\d t\/s at \d+K is below the Fast Assistant gate 30 t\/s/)
  })
})

describe('heavy 2K rows (real run 09:39): -nkvo is dominated at short ctx; MoE partial offload is much cheaper', () => {
  const fh = load('calib-heavy-qwen38-rx9070.json')
  const dec = (id: string) => fh.runs.find((r) => r.configId === id && r.ctx === 2048)!.decodeTps!
  it('-nkvo decodes slower than simply dropping ~5 layers, in both families', () => {
    expect(dec('qwen38|ngl=57|nkvo')).toBeLessThan(dec('qwen38|ngl=50'))
    expect(dec('gemma4|ngl=26|nkvo')).toBeLessThan(dec('gemma4|ngl=21'))
  })
  it('the MoE (Gemma-4-26B-A4B) decodes ≥ 3.5× the dense 27B at a similar layer share', () => {
    expect(dec('gemma4|ngl=21') / dec('qwen38|ngl=50')).toBeGreaterThan(3.5) // 21/30 = 70 % vs 50/65 = 77 %
  })
  it('heavy candidates for a MoE carry the note; -nkvo (when generated) is last before the CPU baseline', () => {
    const g = fh.models.find((m) => m.id.startsWith('gemma4'))!
    const cs = generateCandidates(M, g, { backend: 'vulkan' }, WORKLOADS.long_context_coding, heavy).candidates
    expect(cs.every((c) => c.notes.some((n) => n.startsWith('MoE (128 experts, 8 active per token)')))).toBe(true)
    const nk = cs.findIndex((c) => c.kvOffload === false)
    if (nk >= 0) expect(cs.slice(nk + 1).every((c) => c.gpuLayers === 0)).toBe(true)
  })
})
