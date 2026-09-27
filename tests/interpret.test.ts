// Interpretation rules interp-2: every rule has a positive and a negative case (docs/INTERPRETATION.md v2, rules.v2.json).
// The review-w4g acceptance tests cover the conformance gaps; this file covers each rule id.
import { describe, expect, it } from 'vitest'
import type { BenchmarkRunResult, CandidateInput, GenConfig, MachineLimits, WorkloadId } from '../src/shared/bench-types'
import { interpret, RULES, RULES_VERSION, verdicts, type Insight, type InterpretData } from '../src/core/interpret'
import { summarizeGen, templateKwargsFor, type GenRow } from '../src/core/benchmark/gen'
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
const full8 = () => withQuality(inputs(f8).filter((c) => c.config.id === FULL))
const eight = (w: WorkloadId = 'coding') => {
  const gen = generateCandidates(M, f8.models[0], { backend: 'vulkan' }, WORKLOADS[w]).candidates[0]
  return full8().map((c) => ({ ...c, config: { ...gen, id: FULL } }))
}
const cfgFor = (w: WorkloadId, request = {}) => withProfile(DEFAULT_SCORING_CONFIG, effectiveProfile(WORKLOADS[w], request))
function panel(cands: CandidateInput[], w: WorkloadId, request: { requiredContext?: number; minDecodeTps?: number } = {}, m: MachineLimits = M, extra: Partial<InterpretData> = {}): Insight[] {
  return interpret(verdicts({ candidates: cands, machine: m, ...extra }, w, request, cfgFor(w, request)))
}
const has = (ins: Insight[], id: string) => ins.some((i) => i.ruleId === id)
const text = (ins: Insight[], id: string) => ins.filter((i) => i.ruleId === id).map((i) => i.text).join('\n')
const gates = (cands: CandidateInput[], w: WorkloadId, request = {}, m: MachineLimits = M) =>
  verdicts({ candidates: cands, machine: m }, w, request, cfgFor(w, request)).ranked.flatMap((v) => v.failures.map((f) => f.ruleId))
const run = (ctx: number, o: Partial<Record<'decode' | 'ttft' | 'prefill' | 'vram' | 'shared', number | null>> = {}, extra: Partial<BenchmarkRunResult> = {}): BenchmarkRunResult => ({
  ...toRun({ configId: 'x', model: 'm', ctx, gpuLayers: 99, threads: 8, status: 'ok', promptTokens: ctx * 0.75, loadMs: 1000,
    ttftMs: o.ttft === undefined ? 100 + ctx / 10 : o.ttft, prefillTps: o.prefill ?? 3000, decodeTps: o.decode === undefined ? 90 : o.decode, totalMs: 2000,
    peakRamBytes: GiB, peakVramBytes: o.vram === undefined ? 8 * GiB : o.vram, peakSharedGpuBytes: o.shared === undefined ? 0 : o.shared, cpuAvgPct: 10, gpuAvgPct: 90 }), ...extra
})
const synth = (runs: BenchmarkRunResult[], over: Partial<CandidateInput['model']> = {}, cfg: Partial<CandidateInput['config']> = {}): CandidateInput => {
  const base = full8()[0]
  return { ...base, model: { ...base.model, ...over }, config: { ...base.config, ...cfg }, runs: runs.map((r) => ({ ...r, configId: cfg.id ?? base.config.id })) }
}

// Two models measured on the same rungs (same prompt sizes): 'slow' (decode 20, quality rateA) vs 'fast' (decode 90, rateB).
const pair = (rateA: number, rateB: number): CandidateInput[] => {
  const mk = (id: string, decode: number, rate: number) => {
    const c = synth([run(2048, { decode }), run(4096, { decode }), run(8192, { decode })], { id: `m-${id}`, name: `Model ${id}` }, { id, modelId: `m-${id}` })
    return { ...c, quality: q5('x', rate) }
  }
  return [mk('slow', 20, rateA), mk('fast', 90, rateB)]
}

// Generation configs: thinking at two efforts (applied template kwargs verified) vs the deterministic baseline.
const genRows = (g: GenConfig, rate: number, reasoning: number, ms: number, applied = true): GenRow[] =>
  q5('m', rate).map((r) => ({ ...r, genId: g.id, sample: 1, answerTokens: 100, reasoningTokens: reasoning, totalMs: ms, tokenSource: 'runtime',
    ...(applied ? { appliedTemplateKwargs: templateKwargsFor(THINK_MODEL, g), templateHash: 'tpl-1', runtimeVersion: 'b11208', modelFingerprint: 'llama8b#1', acceptedSampling: { temperature: g.temperature } } : {}) }))
const THINK_MODEL = { genKnobs: { supportsThinking: true, effortValues: ['low', 'medium'] } } as CandidateInput['model']
const OFF: GenConfig = { id: 'off', thinking: false, temperature: 0, source: 'default' }
const LOW: GenConfig = { id: 'think-low-t1', thinking: true, effort: 'low', temperature: 1, source: 'default' }
const MED: GenConfig = { id: 'think-medium-t1', thinking: true, effort: 'medium', temperature: 1, source: 'default' }
const withGens = (lowRate: number, medRate: number, o: { medReasoning?: number; applied?: boolean } = {}) => {
  const c = full8()[0]
  const model = { ...c.model, supportsThinking: true, genKnobs: { supportsThinking: true, effortValues: ['low', 'medium'] } }
  return [{ ...c, model, quality: q5(c.model.id, 0.4), genQuality: [
    summarizeGen(OFF, genRows(OFF, 0.4, 0, 1000, o.applied ?? true), 1), summarizeGen(LOW, genRows(LOW, lowRate, 300, 4000, o.applied ?? true), 3),
    summarizeGen(MED, genRows(MED, medRate, o.medReasoning ?? 600, 7000, o.applied ?? true), 3)
  ] }]
}

describe('catalog', () => {
  it('interp-2: unique ids, origin tags; every warn/critical rule carries an action; no raise-min-decode (I-9.1)', () => {
    expect(RULES_VERSION).toBe('interp-2')
    expect(new Set(RULES.map((r) => r.id)).size).toBe(RULES.length)
    for (const r of RULES) {
      expect(['measured-calibration', 'policy', 'heuristic'], r.id).toContain(r.origin)
      if (r.severity === 'warn' || r.severity === 'critical') expect(r.action, r.id).toBeTruthy()
      expect(r.action).not.toBe('raise-min-decode')
    }
  })
  it('I-9.1: every emitted warn/critical insight has an action', () => {
    for (const ins of [panel(withQuality(inputs(f8)), 'coding'), panel(withQuality([...inputs(f8), ...inputs(f14)]), 'document_analysis'), panel(inputs(f8), 'general_chat')]) {
      for (const i of ins) if (i.severity === 'warn' || i.severity === 'critical') expect(i.action, i.text).toBeTruthy()
    }
  })
  it('I-0.1: critical first, then coverage/quality, speed, memory, comparisons (negative: speed never before coverage)', () => {
    const c = synth([run(2048), { ...run(4096), status: 'fail', failureKind: 'device_lost' }])
    const ins = panel([c, ...full8()], 'general_chat')
    expect(ins[0].severity).toBe('critical')
    const first3 = ins.findIndex((i) => RULES.find((r) => r.id === i.ruleId)!.section === 3)
    const last2 = ins.map((i) => RULES.find((r) => r.id === i.ruleId)!.section).lastIndexOf(2)
    expect(first3).toBeGreaterThan(last2)
  })
  it('I-0.2: evidence carries unit, kind, source/reason and rules version; unavailable stays null', () => {
    const ev = panel(full8(), 'coding').flatMap((i) => i.evidence)
    expect(ev.every((e) => e.rulesVersion === 'interp-2' && e.kind)).toBe(true)
    expect(ev.find((e) => e.metric === 'peakVramBytes')).toMatchObject({ unit: 'bytes' })
    const blind = panel([synth([run(2048, { vram: null })])], 'general_chat').flatMap((i) => i.evidence).find((e) => e.metric === 'peakVramBytes')!
    expect(blind).toMatchObject({ value: null, kind: 'unavailable', reason: 'not in fixture' })
  })
  it('every recommendation reason and why-not cites a v2 rule id', () => {
    const recs = [
      recommend(withQuality([...inputs(f8), ...inputs(f14)]), M, 'coding'), recommend(inputs(f8), M, 'general_chat'), recommend([], M, 'coding'),
      recommend(withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 1 : 0.6)), MH, 'max_quality', undefined, [{ model: 'X', reason: 'does not fit' }]),
      recommendForWorkload({ candidates: eight('long_context_coding'), machine: M }, 'long_context_coding', { requiredContext: 65536, minDecodeTps: 500 }),
      recommend(withGens(1, 1), M, 'reasoning')
    ]
    for (const r of recs) {
      expect(r.rulesVersion).toBe('interp-2')
      for (const x of [...r.reasons, ...(r.whyNot ?? []).map((w) => w.summary)]) expect(x, x).toMatch(/^\[I-\d+\.\d+\] /)
    }
  })
})

describe('§1 provenance', () => {
  it('I-1.1 an unknown TTFT is "not verified" and the constraint counts as not met; measured passes', () => {
    const blind = synth([run(2048, { ttft: null }), run(4096, { ttft: null }), run(8192, { ttft: null })])
    expect(text(panel([blind], 'general_chat'), 'I-1.1')).toMatch(/latency tolerance 8 s not verified .* counts as not met/)
    expect(has(panel(full8(), 'general_chat'), 'I-1.1')).toBe(false)
  })
  it('I-1.2 a prior-quality candidate is provisional, never the confirmed best; measured is confirmed', () => {
    const v = verdicts({ candidates: inputs(f8), machine: M }, 'general_chat')
    expect(v.winner).toBeNull()
    expect(v.provisionalWinner?.input.config.id).toBe(FULL)
    expect(text(interpret(v), 'I-1.2')).toMatch(/Provisional \(not confirmed\): llama8b\|ngl=all depends on quality \(estimated\)/)
    expect(verdicts({ candidates: full8(), machine: M }, 'general_chat').winner?.input.config.id).toBe(FULL)
  })
  it('I-1.2 priors are immutable: adding a measured candidate never changes another candidate\'s prior', () => {
    const prior = inputs(f14).filter((c) => c.config.id === 'qwen14b|ngl=all')
    const alone = verdicts({ candidates: prior, machine: M }, 'general_chat').ranked[0].cs.components.quality.score
    const mixed = verdicts({ candidates: [...prior, ...withQuality(inputs(f8).filter((c) => c.config.id === FULL), 0.2)], machine: M }, 'general_chat')
    expect(mixed.ranked.find((x) => x.input.config.id === 'qwen14b|ngl=all')!.cs.components.quality.score).toBe(alone)
  })
  it('I-1.3 no confirmed winner → the trace names the estimated term; a confirmed winner needs no such note', () => {
    expect(text(panel(inputs(f8), 'general_chat'), 'I-1.3')).toMatch(/No confirmed winner: the best provisional candidate llama8b\|ngl=all depends on quality/)
    expect(has(panel(full8(), 'general_chat'), 'I-1.3')).toBe(false)
  })
  it('I-1.4 an unavailable metric is shown with its recorded reason; measured memory is not', () => {
    const c = synth([run(2048, {}, { peakVramBytes: { value: null, kind: 'unavailable', reason: 'typeperf rows dropped: 3 impossible' } })])
    expect(text(panel([c], 'general_chat'), 'I-1.4')).toMatch(/Peak VRAM for .* at 2K is unavailable: typeperf rows dropped: 3 impossible/)
    expect(has(panel(full8(), 'general_chat'), 'I-1.4')).toBe(false)
  })
})

describe('§2 coverage', () => {
  it('I-2.1 coverage with the structured stop reason; nothing without candidates', () => {
    expect(text(panel(eight(), 'coding'), 'I-2.1')).toMatch(/largest clean context measured in this run 64K \(declared 128K\)\. Higher contexts were not attempted: 128K est\. VRAM/)
    expect(has(panel([], 'coding'), 'I-2.1')).toBe(false)
  })
  it('I-2.2 spill at its rung with the adjacent decode delta; none on the 8B', () => {
    expect(text(panel(withQuality(inputs(f14)), 'reasoning'), 'I-2.2')).toMatch(/exceeded 0\.25 GiB at 32K \(1\.05 GiB\); decode 51\.2 → 26\.4 t\/s vs 16K/)
    expect(has(panel(full8(), 'coding'), 'I-2.2')).toBe(false)
  })
  it('I-2.3 planned skip from the structured planner record; not from free text alone', () => {
    expect(text(panel(eight(), 'coding'), 'I-2.3')).toMatch(/128K was not attempted — estimated VRAM .* > budget .* \(planning snapshot\)/)
    expect(panel(eight(), 'coding').find((i) => i.ruleId === 'I-2.3')!.action).toBe('enable-kv-q8')
    expect(has(panel(full8(), 'coding'), 'I-2.3')).toBe(false)
  })
  it('I-2.4 < 25 % of declared: warn only when a failure/spill/cliff was observed; not at 50 %', () => {
    const c = withQuality(inputs(load('sweep-cliff-16k-32k.json')))
    expect(panel(c, 'coding').find((i) => i.ruleId === 'I-2.4')!.severity).toBe('warn')
    expect(has(panel(full8(), 'coding'), 'I-2.4')).toBe(false)
  })
  it('I-2.5 required context per model: critical when unmet, info when reached', () => {
    const miss = panel(eight('long_context_coding'), 'long_context_coding', { requiredContext: 131072 }).find((i) => i.ruleId === 'I-2.5')!
    expect(miss.severity).toBe('critical')
    expect(miss.text).toMatch(/Required context 128K: .* not tested/)
    expect(panel(eight('long_context_coding'), 'long_context_coding', { requiredContext: 65536 }).find((i) => i.ruleId === 'I-2.5')!.severity).toBe('info')
  })
  it('I-2.6 recovered dip, cause unverified; a smooth sweep has none', () => {
    const dip = synth([run(2048, { decode: 90 }), run(4096, { decode: 40 }), run(8192, { decode: 85 })])
    expect(text(panel([dip], 'general_chat'), 'I-2.6')).toMatch(/decode dipped at 4K \(90\.0 → 40\.0 t\/s\) and recovered on the next tested rung 8K \(85\.0 t\/s\); cause unverified/)
    expect(has(panel(full8(), 'general_chat'), 'I-2.6')).toBe(false)
  })
  it('I-2.7 context floor gate; the 8B passes', () => {
    expect(gates(withQuality(inputs(f14)), 'document_analysis')).toContain('I-2.7')
    expect(gates(full8(), 'document_analysis')).not.toContain('I-2.7')
  })
})

describe('§3 speed', () => {
  it('I-3.1 decode band with actual prompt tokens and the effective gate / user floor; the decode gate fails below it', () => {
    expect(text(panel(full8(), 'coding'), 'I-3.1')).toMatch(/decode 72\.0 t\/s at 32K \(\d+ prompt tokens\) — snappy \(Coding gate 10 t\/s\)/)
    expect(text(panel(full8(), 'coding', { minDecodeTps: 20 }), 'I-3.1')).toMatch(/Coding gate 10 t\/s; your floor 20 t\/s/)
    expect(gates(full8(), 'coding', { minDecodeTps: 200 })).toContain('I-3.1')
    expect(gates(full8(), 'coding')).not.toContain('I-3.1')
  })
  it('I-3.2 same-request effective vs raw rate for a chosen thinking config; never for the baseline', () => {
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-3.2')).toMatch(/reasoning consumed 300 tokens per answer; effective 25\.0 t\/s vs 100\.0 t\/s raw in the same requests \(token split runtime-reported \(measured\)\)/)
    expect(has(panel(withGens(0.4, 0.4), 'reasoning'), 'I-3.2')).toBe(false)
  })
  it('I-3.3 TTFT band; warn + "accepted because required" over tolerance with a required ctx; info within', () => {
    expect(panel(full8(), 'coding').find((i) => i.ruleId === 'I-3.3')!.severity).toBe('info')
    const req = panel(eight(), 'coding', { requiredContext: 65536 }).find((i) => i.ruleId === 'I-3.3')!
    expect(req).toMatchObject({ severity: 'note' }) // accepted because required: no remedy to suggest (G12)
    expect(req.text).toMatch(/accepted because you required 64K/)
  })
  it('I-3.4 prefill scaling reported with the prompt-token ratio and measured TTFT; not on the 8B', () => {
    expect(text(panel([synth([run(2048, { prefill: 3000 }), run(4096, { prefill: 1000 })])], 'general_chat'), 'I-3.4')).toMatch(/prefill 3000\.0 → 1000\.0 t\/s from 2K to 4K \(prompt tokens ×2\.00\); measured TTFT/)
    expect(has(panel(full8(), 'general_chat'), 'I-3.4')).toBe(false)
  })
  it('I-3.5 partial offload only for expectDegraded configs, compared at the same context only', () => {
    const heavy = synth([run(2048, { decode: 12 })], {}, { id: 'p', gpuLayersAll: false, gpuLayers: 20, expectDegraded: true })
    expect(text(panel([heavy, ...full8()], 'general_chat'), 'I-3.5')).toMatch(/^\[I-3\.5\] p: 20\/32 layers on GPU: 12\.0 t\/s at 2K vs 109\.5 t\/s \(llama8b\|ngl=all\) at the same context\.$/)
    expect(has(panel(withQuality(inputs(f8)), 'coding'), 'I-3.5')).toBe(false) // ngl 20/0 fixture rows are not expectDegraded
  })
  it('I-3.6 MoE note scoped to one observation; none for dense', () => {
    expect(text(panel([synth([run(2048)], { expertCount: 128, expertUsedCount: 8 })], 'general_chat'), 'I-3.6')).toMatch(/8\/128 experts per token; .*one cross-family observation/)
    expect(has(panel(full8(), 'general_chat'), 'I-3.6')).toBe(false)
  })
  it('I-3.7 recommended -c is a measured passing rung with its reason; unavailable without one', () => {
    expect(text(panel(full8(), 'coding'), 'I-3.7')).toBe('[I-3.7] Recommended -c 32K: largest passing rung ≤ 64K with measured TTFT within the 15 s tolerance.')
    expect(has(panel([], 'coding'), 'I-3.7')).toBe(false)
  })
})

describe('§4 memory', () => {
  it('I-4.1 budget basis from the planning snapshot (signed, with reserve); otherwise per-PID vs total, not evaluable', () => {
    expect(text(panel(eight(), 'coding'), 'I-4.1')).toMatch(/planning budget remaining [+−][\d.]+ GiB \(budget [\d.]+ GiB − reserve 1\.00 GiB − peak [\d.]+ GiB; per-process budget [\d.]+ GiB estimated — no measured budget on this machine yet; assuming 80 % of the adapter total \(some GPUs\/backends allow 100 %\); not applied to planning; in use at planning [\d.]+ GiB measured\)/)
    // a measured per-process budget tighter than the adapter budget becomes the basis and is named
    const tight = eight().map((c) => ({ ...c, config: { ...c.config, planning: { ...c.config.planning!, effectiveBudget: { value: 11.6 * GiB, kind: 'measured' as const, source: 'test' } } } }))
    expect(text(panel(tight, 'coding'), 'I-4.1')).toMatch(/planning budget remaining [+−][\d.]+ GiB \(effective per-process budget 11\.60 GiB \(measured on this GPU\/driver\/backend\) − peak [\d.]+ GiB; adapter budget [\d.]+ GiB − reserve 1\.00 GiB is looser;/)
    const bare = panel(full8(), 'coding').find((i) => i.ruleId === 'I-4.1')!
    expect(bare).toMatchObject({ evaluable: false })
    expect(bare.text).toMatch(/per-PID vs adapter total; planning budget not recorded — headroom not evaluable/)
  })
  it('I-4.2 VRAM in use > 1.5 GiB noted; unknown in-use says "not measured"; 1.2 GiB is silent', () => {
    expect(text(panel(full8(), 'coding', {}, { ...M, vramInUseBytes: { value: 2 * GiB, kind: 'measured' } }), 'I-4.2')).toMatch(/Planned while 2\.00 GiB of VRAM was already in use/)
    expect(text(panel(full8(), 'coding', {}, { ...M, vramInUseBytes: { value: null, kind: 'unavailable', reason: 'x' } }), 'I-4.2')).toMatch(/was not measured/)
    expect(has(panel(full8(), 'coding', {}, { ...M, vramInUseBytes: { value: 1.2 * GiB, kind: 'measured' } }), 'I-4.2')).toBe(false)
  })
  it('I-4.3 signed floor distance (a deficit is critical), guard_abort critical; room is silent', () => {
    const near = synth([run(2048, {}, { minRamAvailBytes: { value: 4.5 * GiB, kind: 'measured' }, ramFloorBytes: 4 * GiB, mmapCreditBytes: 2 * GiB })])
    expect(text(panel([near], 'general_chat'), 'I-4.3')).toMatch(/minimum RAM available 4\.50 GiB is \+0\.50 GiB vs the 4\.00 GiB floor \(mmap credit 2\.00 GiB applied separately by the guard\)/)
    const under = synth([run(2048, {}, { minRamAvailBytes: { value: 3 * GiB, kind: 'measured' }, ramFloorBytes: 4 * GiB })])
    expect(panel([under], 'general_chat').find((i) => i.ruleId === 'I-4.3')!.severity).toBe('critical')
    const abort = synth([run(2048), { ...run(4096), status: 'fail', failureKind: 'guard_abort', reason: 'RAM available 1.0 GiB fell below the floor' }])
    expect(panel([abort], 'general_chat').find((i) => i.ruleId === 'I-4.3')!.severity).toBe('critical')
    expect(has(panel([synth([run(2048, {}, { minRamAvailBytes: { value: 20 * GiB, kind: 'measured' }, ramFloorBytes: 4 * GiB })])], 'general_chat'), 'I-4.3')).toBe(false)
  })
  it('I-4.4 mmap note only with before / during / after-unload values; omitted otherwise', () => {
    const m = (v: number) => ({ value: v * GiB, kind: 'measured' as const })
    const life = synth([run(2048, {}, { ramAvailBeforeLoadBytes: m(20), minRamAvailDuringLoadBytes: m(15) }), run(4096, {}, { ramAvailBeforeLoadBytes: m(19.8) })])
    expect(text(panel([life], 'general_chat'), 'I-4.4')).toMatch(/RAM drop of 5\.00 GiB during load \(before 20\.00 GiB, minimum 15\.00 GiB\) .* released after unload \(19\.80 GiB\)/)
    expect(has(panel(full8(), 'coding'), 'I-4.4')).toBe(false)
  })
  it('I-4.5 saturation only with ≥ 3 plateau samples before the spill; not with one', () => {
    const c = withQuality(inputs(f14)).map((x) => ({ ...x, runs: x.runs.map((r) => (r.ctx === 32768 ? { ...r, peakVramPlateauSamples: 4, peakVramPlateauVersion: 'pre-spill-1' } : r)) }))
    expect(text(panel(c, 'reasoning'), 'I-4.5')).toMatch(/per-process budget ≈ [\d.]+ GiB on this GPU\/driver\/backend for this allocation \(measured: spill began there, ≈83 % of the adapter total, 4 plateau samples, this run\) — specific to GPU, driver, backend and allocation pattern, not a fixed share of the card\./)
    expect(has(panel(withQuality(inputs(f14)), 'reasoning'), 'I-4.5')).toBe(false)
  })
})

describe('§5 quality', () => {
  it('I-5.1 Q with the heuristic band, method and coverage counts; not for a prior', () => {
    expect(text(panel(full8(), 'coding'), 'I-5.1')).toMatch(/^\[I-5\.1\] Meta-Llama-3\.1-8B-Instruct Q4_K_M: Q 60 \(uncertainty \[\d+, \d+\], heuristic wilson-item unc-1; n=15 items \/ 15 skills \/ 1 samples\) — instruction 60 % \(5 items\), coding 60 % \(5 items\), structured 60 % \(5 items\)\.$/)
    expect(has(panel(inputs(f8).filter((c) => c.config.id === FULL), 'general_chat'), 'I-5.1')).toBe(false)
  })
  it('I-5.2 decisive paired difference vs "insufficient evidence" with the delta neutralized; none without a rival', () => {
    const apart = panel(pair(1, 0.6), 'max_quality').find((i) => i.ruleId === 'I-5.2')!
    expect(apart).toMatchObject({ severity: 'info' })
    expect(apart.text).toMatch(/the interval excludes 0/)
    const close = panel(pair(0.6, 0.6), 'max_quality').find((i) => i.ruleId === 'I-5.2')!
    expect(close).toMatchObject({ severity: 'warn', action: 'run-thorough-quality' })
    expect(close.text).toMatch(/insufficient evidence to distinguish quality on this suite .* the quality delta is neutralized/)
    expect(has(panel(full8(), 'coding'), 'I-5.2')).toBe(false)
  })
  it('I-5.3 weak / coding warning (inclusive boundaries, ≥ 3 items); nothing at 100 %', () => {
    expect(panel(full8(), 'coding').find((i) => i.ruleId === 'I-5.3')).toMatchObject({ severity: 'warn' })
    expect(has(panel(withQuality(inputs(f8).filter((c) => c.config.id === FULL), 1), 'coding'), 'I-5.3')).toBe(false)
  })
  it('I-5.4 thinking-capable model: the gen config used is stated; not for a plain model', () => {
    expect(text(panel(full8().map((c) => ({ ...c, model: { ...c.model, supportsThinking: true } })), 'coding'), 'I-5.4')).toMatch(/quality measured with thinking off \(T=0\)/)
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-5.4')).toMatch(/quality measured with thinking on \(effort low, T=1\.0\); also measured: thinking off/)
    expect(has(panel(full8(), 'coding'), 'I-5.4')).toBe(false)
  })
  it('I-5.5 a prior names its basis and makes the candidate provisional; measured has none', () => {
    expect(text(panel(inputs(f8).filter((c) => c.config.id === FULL), 'general_chat'), 'I-5.5')).toMatch(/no measured quality — the quality term is a prior \(prior from 8\.0B params/)
    expect(has(panel(full8(), 'general_chat'), 'I-5.5')).toBe(false)
  })
  it('I-5.6 limited coverage under 30 unique items (repeats do not count); none at 30 with ≥ 3 per category', () => {
    expect(text(panel(full8(), 'coding'), 'I-5.6')).toMatch(/15 unique items < 30/)
    const repeated = full8().map((c) => ({ ...c, quality: [...c.quality, ...c.quality].map((r, i) => ({ ...r, sample: i < c.quality.length ? 1 : 2 })) }))
    expect(text(panel(repeated, 'coding'), 'I-5.6')).toMatch(/15 unique items < 30/)
    const big = full8().map((c) => ({ ...c, quality: [...q5('a', 0.6), ...q5('b', 0.6).map((r) => ({ ...r, testId: `b-${r.testId}` }))] }))
    expect(has(panel(big, 'coding'), 'I-5.6')).toBe(false)
  })
  it('I-5.7 an infrastructure error quarantines quality (critical) and keeps performance; clean rows do not', () => {
    const c = full8().map((x) => ({ ...x, quality: x.quality.map((r, i) => (i ? r : { ...r, evaluationStatus: 'infra_error' as const })) }))
    const ins = panel(c, 'coding')
    expect(ins.find((i) => i.ruleId === 'I-5.7')).toMatchObject({ severity: 'critical', action: 'inspect-diagnostics' })
    expect(verdicts({ candidates: c, machine: M }, 'coding').ranked[0].cs.components.genSpeed.input.kind).toBe('measured')
    expect(has(panel(full8(), 'coding'), 'I-5.7')).toBe(false)
  })
  it('I-5.8 truncated answers are listed with the budget and count as failures; none when complete', () => {
    const c = full8().map((x) => ({ ...x, quality: x.quality.map((r, i) => (i ? r : { ...r, pass: true, evaluationStatus: 'truncated' as const, outputTruncated: true, maxTokens: 64 })) }))
    expect(text(panel(c, 'coding'), 'I-5.8')).toMatch(/1 answer\(s\) hit the token budget \(64 tokens\) and count as failures: instruction-0/)
    expect(has(panel(full8(), 'coding'), 'I-5.8')).toBe(false)
  })
  it('I-5.9 not-a-leaderboard once with measured quality; not on priors only', () => {
    expect(panel(withQuality(inputs(f8)), 'coding').filter((i) => i.ruleId === 'I-5.9')).toHaveLength(1)
    expect(has(panel(inputs(f8), 'general_chat'), 'I-5.9')).toBe(false)
  })
  it('I-5.10 minimum quality gate; Q 60 passes Coding', () => {
    expect(gates(withQuality(inputs(f8).filter((c) => c.config.id === FULL), 0.2), 'coding')).toContain('I-5.10')
    expect(gates(full8(), 'coding')).not.toContain('I-5.10')
  })
})

describe('§6 eligibility and stability', () => {
  it('I-6.0 / I-6.4 rows without a recorded warmup or with estimated decode are excluded and listed; warm rows are used', () => {
    const c = synth([run(2048), run(4096, {}, { warm: undefined }), run(8192, {}, { decodeTps: { value: 80, kind: 'estimated', source: 'wall clock' } })])
    const v = verdicts({ candidates: [c], machine: M }, 'general_chat')
    expect(v.ranked[0].scored.runs.map((r) => r.ctx)).toEqual([2048])
    expect(text(interpret(v), 'I-6.4')).toMatch(/4K: excluded from speed scoring — warmup not recorded[\s\S]*8K: excluded from speed scoring — decode is estimated/)
    expect(has(panel(full8(), 'general_chat'), 'I-6.4')).toBe(false)
  })
  it('I-6.1 rep spread over the median; 2 % is silent', () => {
    expect(text(panel([synth([run(2048, {}, { repDecodeTps: [90, 60] })])], 'general_chat'), 'I-6.1')).toMatch(/spread 40\.0 % of the median/)
    expect(has(panel([synth([run(2048, {}, { repDecodeTps: [90, 88] })])], 'general_chat'), 'I-6.1')).toBe(false)
  })
  it('I-6.2 mixed runtime or rules versions → rerun-comparable; one version is silent', () => {
    const ver = (rt: string, rules = 'interp-2') => ({ versions: { benchmark: 'bench-1.0.0', prompts: 'ladder-1', quality: 'qb-1.1.0', runtime: rt, rules } })
    expect(panel([synth([run(2048, {}, ver('b1')), run(4096, {}, ver('b2'))])], 'general_chat').find((i) => i.ruleId === 'I-6.2')).toMatchObject({ action: 'rerun-comparable' })
    expect(text(panel([synth([run(2048, {}, ver('b1', 'interp-1')), run(4096, {}, ver('b1'))])], 'general_chat'), 'I-6.2')).toMatch(/rules versions \(interp-1, interp-2\)/)
    expect(has(panel([synth([run(2048, {}, ver('b1')), run(4096, {}, ver('b1'))])], 'general_chat'), 'I-6.2')).toBe(false)
  })
  it('I-6.3 failures from all persisted runs, superseded retries marked, device loss critical; none on a clean sweep', () => {
    const lost = { ...run(4096), configId: FULL, runId: 'old', status: 'fail' as const, failureKind: 'device_lost' as const, supersededBy: 'new', endedAt: 2000 }
    const i = panel(full8(), 'general_chat', {}, M, { allRuns: [lost, ...full8()[0].runs] }).find((x) => x.ruleId === 'I-6.3')!
    expect(i).toMatchObject({ severity: 'critical' })
    expect(i.text).toMatch(/device_lost — GPU reset; results after t=2000 are suspect \(superseded by retry new\)/)
    expect(has(panel(full8(), 'general_chat'), 'I-6.3')).toBe(false)
  })
  it('I-6.5 stability gate (1 of 3 rungs usable); a clean sweep passes', () => {
    const c = synth([run(2048), { ...run(4096), status: 'fail', failureKind: 'request_error' }, { ...run(8192), status: 'fail', failureKind: 'request_error' }])
    expect(gates([c], 'general_chat')).toContain('I-6.5')
    expect(gates(full8(), 'general_chat')).not.toContain('I-6.5')
  })
})

describe('§7 comparisons', () => {
  it('I-7.1 common scoring rung; a candidate not reaching it is named (latency 0); single candidate: no scope line', () => {
    const cands = [...full8(), ...withQuality(inputs(f14).filter((c) => c.config.id === 'qwen14b|ngl=all'))]
    const v = verdicts({ candidates: cands, machine: M }, 'document_analysis')
    expect(v.scoringRung).toBe(65536)
    expect(text(interpret(v), 'I-7.1')).toMatch(/common scoring rung 64K .*not reaching it \(speed read lower, latency 0\): qwen14b\|ngl=all/)
    expect(has(panel(full8(), 'coding'), 'I-7.1')).toBe(false)
  })
  it('I-7.2 the decision trace records constraints, eligibility, neutralizations and the tie-break path', () => {
    const r = recommend(pair(0.6, 0.6), M, 'max_quality')
    const t = r.decisionTrace!
    expect(t.hardConstraints).toMatchObject({ minDecodeTps: { value: 2, source: 'workload' }, latencyToleranceMs: 20000 })
    expect(t.neutralizations[0]).toMatchObject({ a: 'slow', b: 'fast', winner: 'fast' })
    expect(t.comparisons[0]).toMatchObject({ a: 'slow', b: 'fast', basis: 'without-quality', winner: 'fast' })
    expect(t.steps.map((s) => s.kind)).toEqual(['total'])
    expect(r.reasons[0]).toMatch(/^\[I-7\.2\] Decision trace .*: best for Maximum Quality: fast .*1 quality neutralization/)
  })
  it('I-7.3 why-not carries the paired difference first, and cites the rule', () => {
    const r = recommend([...full8(), ...withQuality(inputs(f14).filter((c) => c.config.id === 'qwen14b|ngl=all'))], M, 'coding')
    expect(r.whyNot![0].summary).toMatch(/^\[I-7\.3\] Qwen2\.5-14B-Instruct Q4_K_M: quality [+−]\d+ \[[+−]\d+, [+−]\d+\] \(/)
    expect(recommend([], M, 'coding').whyNot).toEqual([])
  })
  it('I-7.4 chosen for quality with the paired interval; the faster winner states no quality win', () => {
    const r = recommend(withQuality(inputs(fh), (m) => (m.startsWith('qwen') ? 1 : 0.6)), MH, 'max_quality')
    expect(r.reasons.join('\n')).toMatch(/\[I-7\.4\] Quality vs speed: chosen for quality: quality \+40 \[\+\d+, \+\d+\] vs llama8b\|ngl=all; decode 13\.0 vs 96\.5 t\/s \(7\.4× slower\)/)
    expect(recommend(full8(), M, 'coding').reasons.join('\n')).not.toMatch(/I-7\.4/)
  })
  it('I-7.5 same base model → two quantizations, no preference; different fine-tunes → nothing; unverified → "related models"', () => {
    const a = full8()[0]
    const b = { ...a, model: { ...a.model, id: 'q8', quant: 'Q8_0', name: 'Meta-Llama-3.1-8B-Instruct Q8_0' }, config: { ...a.config, id: 'q8|all', modelId: 'q8' } }
    const id = (x: CandidateInput, base: string, ft?: string) => ({ ...x, model: { ...x.model, baseModelId: base, ...(ft ? { fineTuneId: ft } : {}) } })
    expect(text(panel([id(a, 'meta/llama-3.1-8b'), id(b, 'meta/llama-3.1-8b')], 'coding'), 'I-7.5')).toMatch(/same base model, two quantizations — quality difference .*; no non-inferiority margin is declared, so neither is preferred/)
    expect(has(panel([id(a, 'org/base', 'org/math'), id(b, 'org/base', 'org/legal')], 'coding'), 'I-7.5')).toBe(false)
    expect(text(panel([a, b], 'coding'), 'I-7.5')).toMatch(/related models \(same architecture and size; identity not verified\)/)
  })
  it('I-7.6 partial veto only by a same-model full offload meeting the same hard constraints', () => {
    expect(gates(withQuality(inputs(f8)), 'coding')).toContain('I-7.6')
    const short = full8().map((c) => ({ ...c, runs: c.runs.filter((r) => r.ctx <= 8192) }))
    const partial = withQuality([synth([run(2048, { decode: 40 }), run(32768, { decode: 40, ttft: 5000 })], {}, { id: 'p', gpuLayersAll: false, gpuLayers: 20 })])
    expect(gates([...short, ...partial], 'coding', { requiredContext: 32768 })).not.toContain('I-7.6')
  })
})

describe('§8 generation configs', () => {
  it('I-8.0 thinking configs without verified applied kwargs are not considered; verified ones are', () => {
    const v = verdicts({ candidates: withGens(1, 1, { applied: false }), machine: M }, 'general_chat')
    expect(v.winner!.gen!.gq.gen.id).toBe('off')
    expect(text(interpret(v), 'I-8.0')).toMatch(/thinking on \(effort low, T=1\.0\) not comparable — application contract not met: applied template kwargs not verified on every row/)
    expect(has(panel(withGens(1, 1), 'reasoning'), 'I-8.0')).toBe(false)
  })
  it('I-8.1 the chosen thinking config with its paired difference and both effective speeds; baseline kept → none', () => {
    // Max Quality has no latency weight: the cross-context time-to-answer projection (estimated, G08) is not decisive there.
    const r = recommend(withGens(1, 1), M, 'max_quality')
    expect(r.best?.gen?.config.id).toBe('think-low-t1')
    expect(r.best?.gen?.reason).toMatch(/^\[I-8\.1\] Meta-Llama-3\.1-8B-Instruct Q4_K_M: thinking on \(effort low, T=1\.0\): Q \+60 \[\+\d+, \+\d+\] vs thinking off; answers 4\.0× slower \(effective 25\.0 vs 100\.0 t\/s\)$/)
    expect(has(panel(withGens(0.4, 0.4), 'reasoning'), 'I-8.1')).toBe(false)
  })
  it('I-8.2 every sampled config is noted with its samples; the deterministic baseline alone is not', () => {
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-8.2')).toMatch(/thinking on \(effort low, T=1\.0\) is sampled \(seeded\), n=3 per item/)
    expect(has(panel(full8(), 'reasoning'), 'I-8.2')).toBe(false)
  })
  it('I-8.3 higher effort, more reasoning, paired difference includes 0 → prefer the lower; not when medium is clearly better', () => {
    expect(text(panel(withGens(1, 1), 'reasoning'), 'I-8.3')).toMatch(/effort medium adds reasoning \(600 vs 300 tokens\) with a quality difference \+0 \[\+0, \+0\] whose interval includes 0 — prefer effort low/)
    expect(has(panel(withGens(0.2, 1), 'reasoning'), 'I-8.3')).toBe(false)
  })
  it('a thinking config whose time to answer exceeds the tolerance is not chosen', () => {
    const slow = withGens(1, 1).map((c) => ({ ...c, genQuality: c.genQuality!.map((g) => (g.gen.thinking ? { ...g, reasoningMs: { value: 60000, kind: 'measured' as const } } : g)) }))
    expect(verdicts({ candidates: slow, machine: M }, 'general_chat').winner!.gen!.gq.gen.id).toBe('off')
  })
})
