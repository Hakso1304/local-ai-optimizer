// Interpretation rules v1: every rule has a positive and a negative case (docs/INTERPRETATION.md, rules.v1.json).
import { describe, expect, it } from 'vitest'
import type { BenchmarkRunResult, CandidateInput, GenConfig, MachineLimits, WorkloadId } from '../src/shared/bench-types'
import { interpret, RULES, verdicts, type Insight } from '../src/core/interpret'
import { summarizeGen, type GenRow } from '../src/core/benchmark/gen'
import { generateCandidates } from '../src/core/benchmark/candidates'
import { recommend, recommendForWorkload } from '../src/core/scoring/recommend'
import { WORKLOADS, effectiveProfile, withProfile, DEFAULT_SCORING_CONFIG } from '../src/core/scoring/workloads'
import { inputs, load, machine, q5, toRun, withQuality } from './scoring/helpers'

const GiB = 1024 ** 3
const f8 = load('calib-8b-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
const f14 = load('calib-14b-rx9070.json')
const fh = load('calib-heavy-qwen38-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
const M = { ...machine(f8.vramBytes), vramInUseBytes: { value: f8.vramInUseBytes, kind: 'measured' as const } }
const MH = { ...machine(fh.vramBytes), vramInUseBytes: { value: fh.vramInUseBytes, kind: 'measured' as const } }
const FULL = 'llama8b|ngl=all'
const full8 = () => inputs(f8).filter((c) => c.config.id === FULL)
const eight = (w: WorkloadId = 'coding') => {
  const gen = generateCandidates(M, f8.models[0], { backend: 'vulkan' }, WORKLOADS[w]).candidates[0]
  return full8().map((c) => ({ ...c, config: { ...gen, id: FULL } }))
}

function panel(cands: CandidateInput[], w: WorkloadId, request: { requiredContext?: number; minDecodeTps?: number } = {}, m: MachineLimits = M): Insight[] {
  const cfg = withProfile(DEFAULT_SCORING_CONFIG, effectiveProfile(WORKLOADS[w], request))
  return interpret(verdicts({ candidates: cands, machine: m }, w, request, cfg))
}
const has = (ins: Insight[], id: string) => ins.some((i) => i.ruleId === id)
const text = (ins: Insight[], id: string) => ins.filter((i) => i.ruleId === id).map((i) => i.text).join('\n')
const gates = (cands: CandidateInput[], w: WorkloadId, request = {}, m: MachineLimits = M) => {
  const cfg = withProfile(DEFAULT_SCORING_CONFIG, effectiveProfile(WORKLOADS[w], request))
  return verdicts({ candidates: cands, machine: m }, w, request, cfg).ranked.flatMap((v) => v.failures.map((f) => f.ruleId))
}
const run = (ctx: number, o: Partial<Record<'decode' | 'ttft' | 'prefill' | 'vram' | 'shared', number | null>> = {}, extra: Partial<BenchmarkRunResult> = {}): BenchmarkRunResult => ({
  ...toRun({ configId: 'x', model: 'm', ctx, gpuLayers: 99, threads: 8, status: 'ok', promptTokens: ctx * 0.75, loadMs: 1000,
    ttftMs: o.ttft === undefined ? 100 + ctx / 10 : o.ttft, prefillTps: o.prefill ?? 3000, decodeTps: o.decode === undefined ? 90 : o.decode, totalMs: 2000,
    peakRamBytes: GiB, peakVramBytes: o.vram === undefined ? 8 * GiB : o.vram, peakSharedGpuBytes: o.shared === undefined ? 0 : o.shared, cpuAvgPct: 10, gpuAvgPct: 90 }), ...extra
})
const synth = (runs: BenchmarkRunResult[], over: Partial<CandidateInput['model']> = {}, cfg: Partial<CandidateInput['config']> = {}): CandidateInput => {
  const base = full8()[0]
  return { ...base, model: { ...base.model, ...over }, config: { ...base.config, ...cfg }, runs: runs.map((r) => ({ ...r, configId: cfg.id ?? base.config.id })), quality: q5(base.model.id, 0.6) }
}

// Generation configs: a thinking config with higher quality (reasoning 300 + answer 100 tokens) vs the baseline.
const genRows = (g: GenConfig, rate: number, reasoning: number, ms: number): GenRow[] =>
  q5('m', rate).map((r) => ({ ...r, genId: g.id, sample: 1, answerTokens: 100, reasoningTokens: reasoning, totalMs: ms }))
const OFF: GenConfig = { id: 'off', thinking: false, temperature: 0, source: 'default' }
const LOW: GenConfig = { id: 'think-low-t1', thinking: true, effort: 'low', temperature: 1, source: 'default' }
const MED: GenConfig = { id: 'think-medium-t1', thinking: true, effort: 'medium', temperature: 1, source: 'default' }
const withGens = (lowRate: number, medRate: number, medReasoning = 600) => {
  const c = full8()[0]
  const model = { ...c.model, supportsThinking: true, genKnobs: { supportsThinking: true, effortValues: ['low', 'medium'] } }
  return [{ ...c, model, quality: q5(c.model.id, 0.6), genQuality: [
    summarizeGen(OFF, genRows(OFF, 0.6, 0, 1000), 1), summarizeGen(LOW, genRows(LOW, lowRate, 300, 4000), 3), summarizeGen(MED, genRows(MED, medRate, medReasoning, 7000), 3)
  ] }]
}

describe('rules data', () => {
  it('ids are unique; every warn/critical rule carries an action (I-9.1)', () => {
    expect(new Set(RULES.map((r) => r.id)).size).toBe(RULES.length)
    for (const r of RULES) if (r.severity === 'warn' || r.severity === 'critical') expect(r.action, r.id).toBeTruthy()
  })
  it('I-9.1: every emitted warn/critical insight has an action', () => {
    for (const ins of [panel(withQuality(inputs(f8)), 'coding'), panel(withQuality([...inputs(f8), ...inputs(f14)]), 'document_analysis'), panel(inputs(f8), 'general_chat')]) {
      for (const i of ins) if (i.severity === 'warn' || i.severity === 'critical') expect(i.action, i.text).toBeTruthy()
    }
  })
  it('I-0.1: context and quality insights lead; speed follows (negative: a speed insight is never first)', () => {
    const ins = panel(withQuality(inputs(f8)), 'coding')
    const sec = (i: Insight) => RULES.find((r) => r.id === i.ruleId)!.section
    const firstOther = ins.findIndex((i) => sec(i) !== 2 && sec(i) !== 5)
    expect(ins.slice(0, firstOther).every((i) => sec(i) === 2 || sec(i) === 5)).toBe(true)
    expect(ins.slice(firstOther).some((i) => sec(i) === 2 || sec(i) === 5)).toBe(false)
    expect(sec(ins[0])).not.toBe(3)
  })
  it('every recommendation reason cites a rule id', () => {
    const recs = [
      recommend(withQuality([...inputs(f8), ...inputs(f14)]), M, 'coding'), recommend(inputs(f8), M, 'general_chat'), recommend([], M, 'coding'),
      recommend(withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 1 : 0.6)), MH, 'max_quality', undefined, [{ model: 'X', reason: 'does not fit' }]),
      recommendForWorkload({ candidates: withQuality(eight('long_context_coding')), machine: M }, 'long_context_coding', { requiredContext: 65536, minDecodeTps: 500 }),
      recommend(withGens(1, 1), M, 'reasoning')
    ]
    for (const r of recs) {
      expect(r.rulesVersion).toBe('interp-1')
      for (const x of r.reasons) expect(x, x).toMatch(/^\[I-\d/)
    }
  })
})

describe('§1 provenance', () => {
  it('I-1.1 / I-5.5: a prior-quality winner is provisional; measured is not', () => {
    const v = verdicts({ candidates: inputs(f8), machine: M }, 'general_chat')
    expect(v.provisional.map((p) => p.ruleId)).toEqual(['I-1.1', 'I-5.5'])
    expect(verdicts({ candidates: withQuality(inputs(f8)), machine: M }, 'general_chat').provisional).toEqual([])
  })
  it('I-1.1: an estimated quality never out-ranks a measured one (capped at the lowest measured)', () => {
    const cands = [...withQuality(full8(), 0.2), ...inputs(f14).filter((c) => c.config.id === 'qwen14b|ngl=all')] // 14B prior ~90 vs measured 20
    const v = verdicts({ candidates: cands, machine: M }, 'general_chat')
    expect(Math.round(v.ranked.find((x) => x.input.config.id === 'qwen14b|ngl=all')!.cs.components.quality.score)).toBe(20)
  })
  it('I-1.2: unobserved memory on a usable rung is noted, not penalised to 0', () => {
    expect(has(panel([synth([run(2048, { vram: null })])], 'general_chat'), 'I-1.2')).toBe(true)
    expect(has(panel(withQuality(full8()), 'general_chat'), 'I-1.2')).toBe(false)
  })
})

describe('§2 context', () => {
  it('I-2.1 ceiling is always stated for a winner; nothing without one', () => {
    expect(text(panel(withQuality(eight()), 'coding'), 'I-2.1')).toMatch(/^\[I-2\.1\] Practical context 64K \(measured\); model declares 128K; memory-bound at 64K/)
    expect(has(panel([], 'coding'), 'I-2.1')).toBe(false)
  })
  it('I-2.2 spill (14B at 32K) with the decode drop; none on the 8B', () => {
    expect(text(panel(withQuality(inputs(f14)), 'reasoning'), 'I-2.2')).toMatch(/1\.05 GiB moved to shared GPU memory at 32K; decode fell 51\.2 → 26\.4 t\/s/)
    expect(has(panel(withQuality(full8()), 'coding'), 'I-2.2')).toBe(false)
  })
  it('I-2.3 memory-bound (planner skipped 128K for VRAM); not without a memory skip', () => {
    expect(has(panel(withQuality(eight()), 'coding'), 'I-2.3')).toBe(true)
    expect(has(panel(withQuality(full8()), 'coding'), 'I-2.3')).toBe(false)
  })
  it('I-2.4 practical < 25 % of declared; not at 50 %', () => {
    const c = withQuality(inputs(load('sweep-cliff-16k-32k.json')))
    expect(has(panel(c, 'coding'), 'I-2.4')).toBe(true) // 16K of 128K
    expect(has(panel(withQuality(full8()), 'coding'), 'I-2.4')).toBe(false) // 64K of 128K
  })
  it('I-2.5 / I-2.9: required 128K not reachable (critical, gate); 64K reachable', () => {
    expect(has(panel(withQuality(eight('long_context_coding')), 'long_context_coding', { requiredContext: 131072 }), 'I-2.5')).toBe(true)
    expect(gates(withQuality(eight('long_context_coding')), 'long_context_coding', { requiredContext: 131072 })).toContain('I-2.9')
    expect(has(panel(withQuality(eight('long_context_coding')), 'long_context_coding', { requiredContext: 65536 }), 'I-2.5')).toBe(false)
    expect(gates(withQuality(eight('long_context_coding')), 'long_context_coding', { requiredContext: 65536 })).not.toContain('I-2.9')
  })
  it('I-2.6 transient dip that recovers is noted; a smooth sweep is not', () => {
    const dip = synth([run(2048, { decode: 90 }), run(4096, { decode: 40 }), run(8192, { decode: 85 })])
    expect(text(panel([dip], 'general_chat'), 'I-2.6')).toMatch(/Transient dip at 4K .* \(90\.0 → 40\.0 → 85\.0 t\/s/)
    expect(has(panel(withQuality(full8()), 'general_chat'), 'I-2.6')).toBe(false)
  })
  it('I-2.7 context floor gate (14B practical 16K < 32K for Document Analysis); the 8B passes it', () => {
    expect(gates(withQuality(inputs(f14)), 'document_analysis')).toContain('I-2.7')
    expect(gates(withQuality(full8()), 'document_analysis')).not.toContain('I-2.7')
  })
  it('I-2.8 / I-3.10 say which rung was scored and recommended, and why; nothing without a winner', () => {
    const ins = panel(withQuality(full8()), 'coding')
    expect(text(ins, 'I-2.8')).toBe('[I-2.8] Scored at 16K: largest passing rung ≤ target 16K.')
    expect(text(ins, 'I-3.10')).toBe('[I-3.10] Recommended -c 32K: largest passing rung ≤ 64K with measured TTFT within the 15 s tolerance.')
    expect(has(panel([], 'coding'), 'I-2.8')).toBe(false)
  })
  it('I-2.10 required-context fallback below the preferred speed; not when the speed is met', () => {
    const r = recommendForWorkload({ candidates: withQuality(eight('long_context_coding')), machine: M }, 'long_context_coding', { requiredContext: 65536, minDecodeTps: 500 })
    expect(r.reasons.join('\n')).toMatch(/^\[I-2\.10\] Meets required context; below preferred speed/m)
    const ok = recommendForWorkload({ candidates: withQuality(eight('long_context_coding')), machine: M }, 'long_context_coding', { requiredContext: 65536 })
    expect(ok.best?.fallback).toBeUndefined()
  })
})

describe('§3 speed', () => {
  it('I-3.1 decode band + gate wording (user floor = "preferred")', () => {
    expect(text(panel(withQuality(full8()), 'coding'), 'I-3.1')).toBe('[I-3.1] Decode 72.0 t/s at 32K — snappy (Coding gate 10 t/s).')
    expect(text(panel(withQuality(full8()), 'coding', { minDecodeTps: 20 }), 'I-3.1')).toMatch(/Coding preferred 20 t\/s/)
    expect(has(panel([], 'coding'), 'I-3.1')).toBe(false)
  })
  it('I-3.2 thinking that eats the answer rate warns; the baseline never does', () => {
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-3.2')).toMatch(/Reasoning consumes 300 tokens per answer \(thinking on \(effort low, T=1\.0\)\)/)
    expect(has(panel(withGens(0.2, 0.2), 'reasoning'), 'I-3.2')).toBe(false)
  })
  it('I-3.3 TTFT band; warn + "accepted because required" when over tolerance with a required ctx', () => {
    const ok = panel(withQuality(full8()), 'coding').find((i) => i.ruleId === 'I-3.3')!
    expect(ok.severity).toBe('info')
    const req = panel(withQuality(eight('coding')), 'coding', { requiredContext: 65536 }).find((i) => i.ruleId === 'I-3.3')!
    expect(req.severity).toBe('warn')
    expect(req.text).toMatch(/accepted because you required 64K/)
  })
  it('I-3.4 super-linear prefill slowdown; not on the 8B', () => {
    const c = synth([run(2048, { prefill: 3000 }), run(4096, { prefill: 1000 })])
    expect(has(panel([c], 'general_chat'), 'I-3.4')).toBe(true)
    expect(has(panel(withQuality(full8()), 'general_chat'), 'I-3.4')).toBe(false)
  })
  it('I-3.5 / I-3.9 partial offload is explained and gated when the full offload ran; neither for full offload only', () => {
    expect(text(panel(withQuality(inputs(f8)), 'coding'), 'I-3.5')).toMatch(/Partial GPU offload \(20\/32 layers\) for llama8b\|ngl=20 — degraded speed expected: decode 17\.5 t\/s vs 88\.1 t\/s with full offload/)
    expect(gates(withQuality(inputs(f8)), 'coding')).toContain('I-3.9')
    expect(has(panel(withQuality(full8()), 'coding'), 'I-3.5')).toBe(false)
    expect(gates(withQuality(full8()), 'coding')).not.toContain('I-3.9')
  })
  it('I-3.6 MoE note; none for dense', () => {
    expect(has(panel([synth([run(2048)], { expertCount: 128, expertUsedCount: 8 })], 'general_chat'), 'I-3.6')).toBe(true)
    expect(has(panel(withQuality(full8()), 'general_chat'), 'I-3.6')).toBe(false)
  })
  it('I-3.7 decode gate (Fast Assistant 30 t/s rejects the 27B); the 8B passes', () => {
    const cands = withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 1 : 0.6))
    const v = verdicts({ candidates: cands, machine: MH }, 'fast_assistant')
    expect(v.ranked.find((x) => x.input.config.id === 'qwen38|ngl=55')!.failures.map((f) => f.ruleId)).toContain('I-3.7')
    expect(v.ranked.find((x) => x.input.config.id === 'llama8b|ngl=all')!.failures.map((f) => f.ruleId)).not.toContain('I-3.7')
  })
  it('I-3.8 D07: an unknown TTFT fails the latency gate; advisory latency (large_coding) does not; measured passes', () => {
    const blind = synth([run(2048, { ttft: null }), run(4096, { ttft: null }), run(8192, { ttft: null })])
    expect(gates([blind], 'general_chat')).toContain('I-3.8')
    expect(gates([blind], 'large_coding')).not.toContain('I-3.8')
    expect(gates(withQuality(full8()), 'general_chat')).not.toContain('I-3.8')
  })
  it('I-3.10 D06: when no passing rung is within tolerance, the smallest PASS rung is reported and the TTFT gate fails', () => {
    const slow = synth([run(2048, { ttft: 9000 }), run(4096, { ttft: 12000 })])
    const v = verdicts({ candidates: [slow], machine: M }, 'general_chat')
    expect(v.ranked[0].cs).toMatchObject({ recommendedCtx: 2048, recommendedFits: false })
    expect(v.ranked[0].failures.map((f) => f.ruleId)).toContain('I-3.8')
  })
})

describe('§4 memory', () => {
  it('I-4.1 headroom: info normally, warn under 0.5 GiB', () => {
    expect(panel(withQuality(full8()), 'coding').find((i) => i.ruleId === 'I-4.1')!.severity).toBe('info')
    const tight = { ...M, vramBytes: { value: 9 * GiB, kind: 'declared' as const }, vramInUseBytes: { value: 0, kind: 'measured' as const } }
    expect(panel(withQuality(full8()), 'coding', {}, tight).find((i) => i.ruleId === 'I-4.1')!.severity).toBe('warn')
  })
  it('I-4.2 VRAM in use at plan > 1.5 GiB; not at 1.2 GiB', () => {
    expect(has(panel(withQuality(full8()), 'coding', {}, { ...M, vramInUseBytes: { value: 2 * GiB, kind: 'measured' } }), 'I-4.2')).toBe(true)
    expect(has(panel(withQuality(full8()), 'coding', {}, { ...M, vramInUseBytes: { value: 1.2 * GiB, kind: 'measured' } }), 'I-4.2')).toBe(false)
  })
  it('I-4.3 within 1 GiB of the RAM floor (warn) and guard_abort (critical); not with room', () => {
    const near = synth([run(2048, {}, { minRamAvailBytes: { value: 4.5 * GiB, kind: 'measured' } })])
    expect(text(panel([near], 'general_chat'), 'I-4.3')).toMatch(/came within 0\.50 GiB of the RAM safety floor/)
    const abort = synth([run(2048), { ...run(4096), status: 'fail', failureKind: 'guard_abort', reason: 'RAM available 1.0 GiB fell below the floor' }])
    expect(panel([abort], 'general_chat').find((i) => i.ruleId === 'I-4.3')!.severity).toBe('critical')
    expect(has(panel([synth([run(2048, {}, { minRamAvailBytes: { value: 20 * GiB, kind: 'measured' } })])], 'general_chat'), 'I-4.3')).toBe(false)
  })
  it('I-4.4 mmap note for a full offload; not without mmap', () => {
    expect(has(panel(withQuality(full8()), 'coding'), 'I-4.4')).toBe(true)
    expect(has(panel(withQuality(full8()).map((c) => ({ ...c, config: { ...c.config, mmap: false } })), 'coding'), 'I-4.4')).toBe(false)
  })
  it('I-4.5 WDDM spills before full (14B at 83 %); not without a spill', () => {
    expect(text(panel(withQuality(inputs(f14)), 'reasoning'), 'I-4.5')).toMatch(/at 83 %/)
    expect(has(panel(withQuality(full8()), 'coding'), 'I-4.5')).toBe(false)
  })
})

describe('§5 quality', () => {
  it('I-5.1 Q ± ci (n) with the band and per-category rates; not for a prior', () => {
    expect(text(panel(withQuality(full8()), 'coding'), 'I-5.1')).toBe('[I-5.1] Quality 60 ± 20 (15 graded items) — limited; instruction 3/5, coding 3/5, structured 3/5.')
    expect(has(panel(full8(), 'general_chat'), 'I-5.1')).toBe(false)
  })
  it('I-5.2 decisive (bands apart → quality wins) vs within the band (speed decided, warn + action); none without a rival', () => {
    const apart = verdicts({ candidates: withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 1 : 0.6)), machine: MH }, 'coding')
    expect(apart.quality).toMatchObject({ decisive: true })
    expect(apart.winner!.input.config.id).toBe('qwen38|ngl=55')
    const close = verdicts({ candidates: withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 0.8 : 0.6)), machine: MH }, 'coding')
    expect(close.quality).toMatchObject({ decisive: false })
    expect(close.winner!.input.config.id).toBe('llama8b|ngl=all')
    expect(interpret(close).find((i) => i.ruleId === 'I-5.2')).toMatchObject({ severity: 'warn', action: 'run-thorough-quality' })
    expect(verdicts({ candidates: withQuality(full8()), machine: M }, 'coding').quality).toBeNull()
  })
  it('I-5.3 weak categories (coding ≤ 66 % warns for coding workloads); none at 100 %', () => {
    expect(panel(withQuality(full8()), 'coding').find((i) => i.ruleId === 'I-5.3')!.severity).toBe('warn')
    expect(has(panel(withQuality(full8(), 1), 'coding'), 'I-5.3')).toBe(false)
  })
  it('I-5.4 thinking-capable model scored with thinking off; not for a non-thinking model or when thinking won', () => {
    const t = withQuality(full8()).map((c) => ({ ...c, model: { ...c.model, supportsThinking: true } }))
    expect(has(panel(t, 'coding'), 'I-5.4')).toBe(true)
    expect(has(panel(withQuality(full8()), 'coding'), 'I-5.4')).toBe(false)
    expect(has(panel(withGens(1, 1), 'reasoning'), 'I-5.4')).toBe(false)
  })
  it('I-5.6 small suite (n < 30); not with 30+ graded items', () => {
    expect(has(panel(withQuality(full8()), 'coding'), 'I-5.6')).toBe(true)
    const big = full8().map((c) => ({ ...c, quality: [...q5('a', 0.6), ...q5('b', 0.6)] })) // 10 per category × 3 = 30
    expect(has(panel(big, 'coding'), 'I-5.6')).toBe(false)
  })
  it('I-5.7 not-a-leaderboard once with measured quality; not on priors only', () => {
    expect(panel(withQuality(inputs(f8)), 'coding').filter((i) => i.ruleId === 'I-5.7')).toHaveLength(1)
    expect(has(panel(inputs(f8), 'general_chat'), 'I-5.7')).toBe(false)
  })
  it('I-5.8 minimum quality gate; I-5.9 unmeasured quality gates quality-weighted workloads only (D07)', () => {
    expect(gates(withQuality(full8(), 0.2), 'coding')).toContain('I-5.8')
    expect(gates(withQuality(full8(), 0.6), 'coding')).not.toContain('I-5.8')
    expect(gates(full8(), 'coding')).toContain('I-5.9')
    expect(gates(full8(), 'general_chat')).not.toContain('I-5.9')
  })
})

describe('§6 stability', () => {
  it('I-6.1 noisy reps (> 15 % apart); not at 2 %', () => {
    expect(has(panel([synth([run(2048, {}, { repDecodeTps: [90, 60] })])], 'general_chat'), 'I-6.1')).toBe(true)
    expect(has(panel([synth([run(2048, {}, { repDecodeTps: [90, 88] })])], 'general_chat'), 'I-6.1')).toBe(false)
  })
  it('I-6.2 mixed runtime versions; not with one', () => {
    const v = (rt: string) => ({ versions: { benchmark: 'bench-1.0.0', prompts: 'ladder-1', quality: 'qb-1.1.0', runtime: rt } })
    expect(has(panel([synth([run(2048, {}, v('b1')), run(4096, {}, v('b2'))])], 'general_chat'), 'I-6.2')).toBe(true)
    expect(has(panel([synth([run(2048, {}, v('b1')), run(4096, {}, v('b1'))])], 'general_chat'), 'I-6.2')).toBe(false)
  })
  it('I-6.3 failures (device_lost critical); none on a clean sweep', () => {
    const c = synth([run(2048), { ...run(4096), status: 'fail', failureKind: 'device_lost' }])
    expect(panel([c], 'general_chat').find((i) => i.ruleId === 'I-6.3')!.severity).toBe('critical')
    expect(has(panel(withQuality(full8()), 'general_chat'), 'I-6.3')).toBe(false)
  })
  it('I-6.4 cold run noted; not for warm rows', () => {
    expect(has(panel([synth([run(2048, {}, { warm: false })])], 'general_chat'), 'I-6.4')).toBe(true)
    expect(has(panel([synth([run(2048, {}, { warm: true })])], 'general_chat'), 'I-6.4')).toBe(false)
  })
  it('I-6.5 stability gate (1 of 3 rungs ≤ target usable); a clean sweep passes', () => {
    const c = synth([run(2048), { ...run(4096), status: 'fail', failureKind: 'request_error' }, { ...run(8192), status: 'fail', failureKind: 'request_error' }])
    expect(gates([c], 'general_chat')).toContain('I-6.5')
    expect(gates(withQuality(full8()), 'general_chat')).not.toContain('I-6.5')
  })
})

describe('§7 comparisons', () => {
  it('I-7.1 the winner line; I-7.2 why-not leads with the quality delta', () => {
    const r = recommend(withQuality([...full8(), ...inputs(f14).filter((c) => c.config.id === 'qwen14b|ngl=all')]), M, 'coding')
    expect(r.reasons[0]).toMatch(/^\[I-7\.1\] Best for Coding: llama8b\|ngl=all/)
    expect(r.whyNot![0].summary).toMatch(/^Qwen2\.5-14B-Instruct Q4_K_M: (same quality|quality [+−]\d+ pts) \(/)
    expect(recommend([], M, 'coding').reasons[0]).toBe('[I-7.1] No recommendation: no candidates were benchmarked')
  })
  it('I-7.3 quality over speed is stated with both bands; not when the winner is also the fastest', () => {
    const r = recommend(withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 1 : 0.6)), MH, 'max_quality')
    expect(r.reasons.join('\n')).toMatch(/\[I-7\.3\] Chosen for quality over speed: quality 100 ± \d+ vs 60 ± \d+; decode 13\.0 vs 96\.5 t\/s \(7\.4× slower/)
    expect(recommend(withQuality(full8()), M, 'coding').reasons.join('\n')).not.toMatch(/I-7\.3/)
  })
  it('I-7.4 same model, two quantizations, quality inside the band → prefer the smaller; not across parameter counts', () => {
    const a = withQuality(full8())[0]
    const b = { ...a, model: { ...a.model, id: 'q8', quant: 'Q8_0', name: 'Meta-Llama-3.1-8B-Instruct Q8_0', fileBytes: a.model.fileBytes * 1.8 }, config: { ...a.config, id: 'q8|all', modelId: 'q8' } }
    expect(text(panel([a, b], 'coding'), 'I-7.4')).toMatch(/prefer the smaller \(Meta-Llama-3\.1-8B-Instruct Q4_K_M\)/)
    expect(has(panel([a, { ...b, model: { ...b.model, paramCount: 14e9 } }], 'coding'), 'I-7.4')).toBe(false)
  })
})

describe('§8 generation configs', () => {
  it('I-8.1 the chosen thinking config and its delta; baseline wins → no I-8.1', () => {
    const r = recommend(withGens(1, 1), M, 'reasoning')
    expect(r.best?.gen?.config.id).toBe('think-low-t1')
    expect(r.best?.gen?.reason).toMatch(/^\[I-8\.1\] Meta-Llama-3\.1-8B-Instruct Q4_K_M: thinking on \(effort low, T=1\.0\): quality 100 ± \d+ vs 60 ± \d+ with thinking off; answers 4\.0× slower\.$/)
    expect(r.whyNot!.filter((w) => w.genId).map((w) => w.genId)).toEqual(['off', 'think-medium-t1'])
    expect(has(panel(withGens(0.2, 0.2), 'reasoning'), 'I-8.1')).toBe(false)
  })
  it('I-8.2 sampled config noted with its samples; not for the deterministic baseline', () => {
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-8.2')).toMatch(/is sampled \(seeded\), 3 per item/)
    expect(has(panel(withGens(0.2, 0.2), 'reasoning'), 'I-8.2')).toBe(false)
  })
  it('I-8.3 higher effort, more reasoning, no quality gain → prefer the lower; not when medium is clearly better', () => {
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-8.3')).toMatch(/effort medium adds reasoning \(600 vs 300 tokens\) .* prefer effort low/)
    expect(has(panel(withGens(0.2, 1, 600), 'reasoning'), 'I-8.3')).toBe(false)
  })
  it('a thinking config over the latency tolerance is not chosen (time to answer counts reasoning)', () => {
    const slow = withGens(1, 1).map((c) => ({ ...c, genQuality: c.genQuality!.map((g) => (g.gen.thinking ? { ...g, reasoningTokens: { value: 5000, kind: 'estimated' as const } } : g)) }))
    expect(verdicts({ candidates: slow, machine: M }, 'reasoning').winner!.gen!.gq.gen.id).toBe('off')
  })
})
