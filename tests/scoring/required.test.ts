// Required context (64K–128K large-scale coding), advisory latency, user decode gate, fallback, large_coding profile.
import { describe, expect, it } from 'vitest'
import type { CandidateInput, QualityResult } from '../../src/shared/bench-types'
import { generateCandidates } from '../../src/core/benchmark/candidates'
import { recommendForWorkload } from '../../src/core/scoring/recommend'
import { WORKLOADS, effectiveProfile } from '../../src/core/scoring/workloads'
import { inputs, load, machine, toRun } from './helpers'

const f8 = load('calib-8b-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
const M = { ...machine(f8.vramBytes), vramInUseBytes: { value: f8.vramInUseBytes, kind: 'measured' as const } }
const llama = f8.models[0]
const FULL = 'llama8b|ngl=all'
// Real config (so skippedSteps carry the planner's memory reasons), real 8B runs 2K→64K.
const gen = generateCandidates(M, llama, { backend: 'vulkan' }, effectiveProfile(WORKLOADS.long_context_coding, { requiredContext: 131072 })).candidates[0]
const eight = (): CandidateInput[] => inputs(f8).filter((c) => c.config.id === FULL).map((c) => ({ ...c, config: { ...gen, id: FULL } }))
const data = (cs: CandidateInput[]) => ({ candidates: cs, machine: M })

describe('required context', () => {
  it('8B practical 64K: eligible for required 64K, with the TTFT at 64K in the reasons', () => {
    const rec = recommendForWorkload(data(eight()), 'long_context_coding', { requiredContext: 65536 })
    expect(rec.best?.configId).toBe(FULL)
    expect(rec.best?.score.recommendedCtx).toBe(65536)
    expect(rec.reasons).toContain('Required context 64K: TTFT 32.4 s at 64K')
  })

  it('…ineligible for required 128K, with what limited it', () => {
    const rec = recommendForWorkload(data(eight()), 'long_context_coding', { requiredContext: 131072 })
    expect(rec.best).toBeNull()
    expect(rec.ranked[0].gateFailures).toContain('practical context 64K < required 128K (limited by memory)')
  })

  it('an explicit required context makes TTFT advisory: eligible, and the reason says it was accepted', () => {
    const rec = recommendForWorkload(data(eight()), 'coding', { requiredContext: 65536 }) // coding tolerance 15 s
    expect(rec.best?.configId).toBe(FULL)
    expect(rec.reasons).toContain("Required context 64K: TTFT 32.4 s at 64K (above the profile's 15 s tolerance; accepted because you required 64K)")
  })

  it('a user minDecodeTps replaces the profile gate (raise or lower)', () => {
    expect(recommendForWorkload(data(eight()), 'fast_assistant', { minDecodeTps: 200 }).best).toBeNull()
    const lowered = effectiveProfile(WORKLOADS.fast_assistant, { minDecodeTps: 5 })
    expect(lowered.minDecodeTps).toBe(5)
  })

  it('never empty just because everything is slow: falls back to the fastest config reaching the required context', () => {
    const rec = recommendForWorkload(data(eight()), 'long_context_coding', { requiredContext: 65536, minDecodeTps: 500 })
    expect(rec.best).toMatchObject({ configId: FULL, fallback: 'meets required context; below preferred speed' })
    expect(rec.reasons.join('\n')).toMatch(/^Meets required context; below preferred speed: llama8b\|ngl=all \(fastest config reaching 64K; decode .* below the 500 t\/s minimum\)$/m)
  })

  it('16 GB GPU, target 128K: q8_0 and -nkvo full-offload variants are generated, each noted', () => {
    const ids = generateCandidates(M, llama, { backend: 'vulkan' }, effectiveProfile(WORKLOADS.long_context_coding, { requiredContext: 131072 })).candidates
    expect(ids.map((c) => c.id.replace(`${llama.id}|`, ''))).toEqual(['ngl=all|kv=f16|t=8', 'ngl=all|kv=q8_0|t=8', 'ngl=all|kv=f16|t=8|nkvo'])
    expect(ids[1].notes.join(' ')).toMatch(/KV cache q8_0/)
    expect(ids[2]).toMatchObject({ kvOffload: false })
    expect(ids[1].ctxSteps.at(-1)).toBe(131072) // q8_0 KV (≈8.5 GiB) + weights fit the 13.7 GiB budget at 128K
    // KV in RAM: 16 GiB of f16 KV at 128K exceeds available − reserve (20 − 4 GiB) → skipped with the RAM reason.
    expect(ids[2].skippedSteps).toEqual([{ ctx: 131072, reason: 'skipped_memory: est. RAM 16.5 GiB > available − reserve 16.0 GiB' }])
  })
})

describe('large_coding profile (quality over speed)', () => {
  const fh = load('calib-heavy-qwen38-rx9070.json')
  const q5 = (id: string, rate: number): QualityResult[] =>
    ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context'].flatMap((category) => [0, 1, 2, 3, 4].map((i) => ({
      testId: `${id}-${category}-${i}`, category: category as QualityResult['category'], weight: 1, pass: i < rate * 5, score: 1, detail: ''
    })))
  // Qwen3.8-27B 55/65: measured 2K–8K; 16K–64K extrapolated flat (decode is set by the CPU-side layers), prefill ~600 t/s.
  const q = inputs(fh).find((c) => c.config.id === 'qwen38|ngl=55')!
  const extra = [16384, 32768, 65536].map((ctx) => toRun({ ...fh.runs.find((r) => r.configId === 'qwen38|ngl=55' && r.ctx === 8192)!, ctx, ttftMs: ((ctx * 0.75) / 600) * 1000, prefillTps: 600, decodeTps: 12.4 }))
  const big: CandidateInput = { ...q, runs: [...q.runs, ...extra], quality: q5(q.model.id, 1) }
  const all = () => [big, ...eight().map((c) => ({ ...c, quality: q5(c.model.id, 0.6) }))]

  it('picks the 27B (Q 100 at 12 t/s) over the 8B (Q 60 at ~100 t/s) when both reach 64K, and says why', () => {
    const rec = recommendForWorkload(data(all()), 'large_coding', { requiredContext: 65536 })
    expect(rec.best?.configId).toBe('qwen38|ngl=55')
    expect(rec.reasons.join('\n')).toMatch(/Chosen for quality over speed: decode 12\.4 t\/s \([\d.]+× slower than llama8b\|ngl=all\)/)
    expect(rec.reasons.join('\n')).toMatch(/Required context 64K: TTFT 81\.9 s at 64K \(above the profile's 180 s|Required context 64K: TTFT 81\.9 s at 64K$/m)
  })

  it('plain Coding still picks the 8B on the same data', () => {
    expect(recommendForWorkload(data(all()), 'coding').best?.configId).toBe(FULL)
  })

  it('large_coding: latency is advisory, gates are 8 t/s decode and quality 50', () => {
    expect(WORKLOADS.large_coding).toMatchObject({ targetContext: 65536, maxContext: 131072, minDecodeTps: 8, latencyAdvisory: true })
    expect(WORKLOADS.large_coding.weights.latency).toBe(0)
  })
})
