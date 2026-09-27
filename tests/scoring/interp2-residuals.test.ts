// Worker #4's final re-review (docs/review-w4j-2026-09-27.md): each executed residual probe F1–F10 as a NEGATIVE test —
// the output the probe observed must no longer occur.
import { describe, expect, it } from 'vitest'
import type { BenchmarkRunResult, CandidateInput, GenQuality, MachineLimits, Metric, ModelMeta, QualityCategory, QualityResult, WorkloadId } from '../../src/shared/bench-types'
import { interpret, verdicts, type Insight } from '../../src/core/interpret'
import fixture from '../fixtures/scoring/session-single.json'
import { RULES } from '../../src/core/interpret/catalog'
import { proofRowId, type GenRow } from '../../src/core/benchmark/gen'

const GiB = 1024 ** 3
const m = (value: number): Metric => ({ value, kind: 'measured', source: 'synthetic' })
const na = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
const machine = (): MachineLimits => ({ vramBytes: m(16 * GiB), vramInUseBytes: m(GiB), ramTotalBytes: m(32 * GiB), ramAvailableBytes: m(24 * GiB), physicalCores: 8, gpuDevice: 'Vulkan0' })
const CATS: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']
const quality = (n = 10, pass: (i: number) => boolean = (i) => i < 8): QualityResult[] =>
  CATS.flatMap((category) => Array.from({ length: n }, (_, i) => ({ testId: `${category}-${i}`, category, weight: 1, pass: pass(i), score: pass(i) ? 1 : 0, detail: '', evaluationStatus: 'valid' } as QualityResult)))
const VER = { benchmark: 'bench-1.0.0', prompts: 'ladder-2', quality: 'qb-1.1.0', runtime: 'synthetic' }
const run = (configId: string, ctx: number, decode = 40, over: Partial<BenchmarkRunResult> = {}): BenchmarkRunResult => ({
  configId, ctx, promptTokens: ctx * 0.75, status: 'pass', failureKind: null, warm: true,
  loadTimeMs: m(500), ttftMs: m(1000), prefillTps: m(2000), decodeTps: m(decode), totalMs: m(2000),
  peakVramBytes: m(8 * GiB), peakSharedGpuBytes: m(0), peakRamBytes: m(GiB), minRamAvailBytes: m(20 * GiB), ramFloorBytes: 4 * GiB, avgGpuUtil: m(80), avgCpuUtil: m(10),
  versions: VER, ...over
})
function candidate(id: string, contexts = [8192], decode = 40, over: Partial<BenchmarkRunResult> = {}): CandidateInput {
  const model: ModelMeta = { ...(fixture.models[0] as ModelMeta), id, name: `Synthetic ${id}`, ctxTrain: 131072 }
  return {
    model, config: { id, modelId: id, device: 'Vulkan0', gpuLayers: model.layers, gpuLayersAll: true, kvType: 'f16', flashAttn: true, threads: 8, ctxSteps: contexts, skippedSteps: [], estVramBytes: m(8 * GiB), estRamBytes: m(GiB), notes: [], mmap: false },
    runs: contexts.map((ctx) => run(id, ctx, decode, over)), quality: quality()
  }
}
const V = (cs: CandidateInput[], w: WorkloadId = 'max_quality', extra = {}, req = {}) => verdicts({ candidates: cs, machine: machine(), ...extra }, w, req)
const text = (ins: Insight[], id: string) => ins.filter((i) => i.ruleId === id).map((i) => i.text).join('\n')

describe('F1 unknown RAM safety fails the hard predicate', () => {
  it('no recorded floor → not a confirmed winner (was: winner, confirmed, failures [])', () => {
    const c = candidate('unknown-floor', [8192], 40, { ramFloorBytes: undefined })
    const v = V([c])
    expect(v.winner).toBeNull()
    expect(v.ranked[0].failures.find((f) => f.ruleId === 'I-4.3')).toMatchObject({ hard: true, notVerified: 'RAM safety floor' })
  })
  it('a full offload with unknown safety cannot veto a safe partial config', () => {
    const full = candidate('full', [8192], 40, { ramFloorBytes: undefined })
    const partial = candidate('partial', [8192], 30)
    partial.model = full.model; partial.config = { ...partial.config, modelId: full.model.id, gpuLayersAll: false, gpuLayers: 20 }
    expect(V([full, partial]).ranked.find((x) => x.input.config.id === 'partial')!.failures.map((f) => f.ruleId)).not.toContain('I-7.6')
  })
})

describe('F2 comparable scope for decisive components', () => {
  it('a shorter ladder (4K only vs the 8K rung) does not confirm (was: winner=short, confirmed)', () => {
    const short = candidate('short', [4096], 100), exact = candidate('exact', [8192], 40)
    short.quality = quality(10, () => true)
    const v = V([short, exact])
    expect(v.scoringRung).toBe(8192)
    expect(v.ranked.find((x) => x.input.config.id === 'short')!.confirmed).toBe(false)
    expect(v.winner?.input.config.id).not.toBe('short')
  })
  it('a different prompt size at the same rung is unmatched; a different quality ctx / template is not paired', () => {
    const a = candidate('a'), b = candidate('b', [8192], 40, { promptTokens: 100 })
    b.quality = b.quality.map((r) => ({ ...r, ctx: 2048, templateHash: 'other' } as QualityResult)); a.quality = a.quality.map((r) => ({ ...r, ctx: 8192, templateHash: 'tpl' } as QualityResult))
    const v = V([a, b])
    expect(v.ranked.find((x) => x.input.config.id === 'b')!.undecided.map((u) => u.kind)).toContain('unmatched-prompt')
    const qb = v.trace.candidates.find((c) => c.configId === 'b')!.qualityVsWinner
    expect(qb).not.toBeNull()
    expect(qb!.difference).toBeNull()
    expect(qb!.reason).toMatch(/not comparable: quality contexts differ \(2048 vs 8192\)/)
  })
})

describe('F3 per-component basis and a stored counterfactual', () => {
  it('quality basis is the quality rows\' own ctx, not the speed rung (was: quality rung 8192)', () => {
    const c = candidate('a'); c.quality = c.quality.map((r) => ({ ...r, ctx: 2048 } as QualityResult))
    expect(V([c]).trace.candidates[0].basis.find((b) => b.component === 'quality')!.rung).toBe(2048)
  })
  it('a prior-only candidate has an evaluated counterfactual in the trace', () => {
    const prior = candidate('prior'); prior.quality = []
    const t = V([candidate('measured'), prior], 'general_chat').trace
    expect(t.counterfactuals.find((c) => c.configId === 'prior')).toMatchObject({ without: ['quality'], vs: 'measured', result: expect.stringMatching(/without those terms/) })
  })
})

describe('F4 the adjusted spill is rendered as observed', () => {
  it('I-2.2 discloses absorbed first-rung residual without treating it as proven spill-free capacity', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[0].peakSharedGpuRawBytes = m(0.5 * GiB)
    c.runs[0].baselineAbsorbedBytes = m(0.4 * GiB)
    c.runs[1].peakSharedGpuRawBytes = m(1.6 * GiB)
    const i = interpret(V([c])).find((x) => x.ruleId === 'I-2.2')!
    expect(i.text).toMatch(/0\.40 GiB first-rung shared residual was subtracted as baseline \(cause unverified\)/)
    expect(i.evidence).toEqual(expect.arrayContaining([expect.objectContaining({ metric: 'baselineAbsorbedBytes' })]))
    expect(i.text).not.toMatch(/first rung was spill-free|capacity confirmed/)
    c.runs[0].baselineAbsorbedBytes = na('shared reading unavailable')
    expect(text(interpret(V([c])), 'I-2.2')).toMatch(/first-rung baseline absorption was not recorded/)
    c.runs[0].baselineAbsorbedBytes = m(0)
    expect(text(interpret(V([c])), 'I-2.2')).toMatch(/0\.00 GiB first-rung shared residual was subtracted as baseline/)
  })
  it('unavailable adjusted metric is not described as "stayed below"', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[0].peakSharedGpuRawBytes = m(0.05 * GiB); c.runs[1].peakSharedGpuRawBytes = m(1.15 * GiB)
    c.runs[1].peakSharedGpuBytes = na('no adjusted metric')
    const t = text(interpret(V([c])), 'I-2.2')
    expect(t).not.toMatch(/stayed below/)
    expect(t).toMatch(/the adjusted spill is unavailable \(no adjusted metric\)/)
    expect(t).toMatch(/raw-growth-1/)
  })
})

describe('F5 the application contract', () => {
  it('nonempty applied kwargs without template/runtime/model identity and accepted sampling is not comparable', () => {
    const c = { ...candidate('think'), model: { ...candidate('think').model, genKnobs: { supportsThinking: true, effortValues: ['low'] } } }
    const rows = quality(10, () => true).map((r) => ({ ...r, genId: 'think-low', appliedTemplateKwargs: { enable_thinking: true } }))
    const g: GenQuality = { gen: { id: 'think-low', thinking: true, effort: 'low', temperature: 0, source: 'default' }, results: rows, samples: 1, stochastic: false, answerTokens: m(100), reasoningTokens: m(20), effectiveAnswerLatencyMs: m(2000), effectiveTps: m(50), reasoningMs: m(500), rawTps: m(60) }
    c.genQuality = [g]
    const v = V([c])
    expect(v.ranked[0].genOptions[0].comparable).toBe(false)
    expect(v.ranked[0].genOptions[0].why).toMatch(/templateHash not recorded.*runtime-accepted temperature not recorded on every row/)
    expect(v.winner?.gen ?? null).toBeNull()
  })
})

describe('F6 version proof without an expected version', () => {
  it('versionless rows are ineligible even when no session version is known (was: winner "no-version")', () => {
    const c = candidate('no-version'); c.runs = c.runs.map(({ versions: _v, ...r }) => r as BenchmarkRunResult)
    const v = V([c])
    expect(v.sessionVersion).toBeNull()
    expect(v.winner).toBeNull()
    expect(v.excluded[0].reasons.join(' ')).toMatch(/versions not recorded — no version proof/)
  })
})

describe('F7 category rates never round away repeated failures', () => {
  it('one coding item with pass + fail completions is 50 %, not "1/1"', () => {
    const c = candidate('a')
    c.quality = [...quality(10, () => true).filter((r) => r.category !== 'coding'),
      { testId: 'coding-0', category: 'coding', weight: 1, pass: true, score: 1, detail: '', sample: 1 } as QualityResult,
      { testId: 'coding-0', category: 'coding', weight: 1, pass: false, score: 0, detail: '', sample: 2 } as QualityResult]
    const t = text(interpret(V([c])), 'I-5.1')
    expect(t).not.toMatch(/coding 1\/1/)
    expect(t).toMatch(/coding 50 % \(1 item, 2 completions\)/)
  })
})

describe('F8 saturation needs the versioned pre-spill window', () => {
  it('a legacy plateau count (no producer version) is not evaluable (was: "spill began at ≈50 %")', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[1] = { ...c.runs[1], peakSharedGpuBytes: m(1 * GiB), peakVramBytes: m(8 * GiB), peakVramPlateauSamples: 3 }
    expect(text(interpret(V([c])), 'I-4.5')).toBe('')
    c.runs[1] = { ...c.runs[1], peakVramPlateauVersion: 'pre-spill-1' }
    expect(text(interpret(V([c])), 'I-4.5')).toMatch(/≈50 % of the adapter total/)
  })
})

describe('F9 remedy feasibility from the breakdown', () => {
  it('weights alone over budget → no enable-kv-q8 (was: enable-kv-q8)', () => {
    const c = candidate('a', [8192])
    c.config.skippedSteps = [{ ctx: 16384, reason: 'weights exceed budget', skip: { resource: 'vram', estimateBytes: 25 * GiB, budgetBytes: 14 * GiB, ruleId: 'I-2.3', weightsBytes: 22 * GiB, kvBytes: 2 * GiB, overheadBytes: GiB } }]
    const i = interpret(V([c])).find((x) => x.ruleId === 'I-2.3')!
    expect(i.action).not.toBe('enable-kv-q8')
    expect(i.action).toBe('enable-heavy-mode')
  })
  it('KV-dominated skip where q8_0 fits → enable-kv-q8; no breakdown → inspect-diagnostics', () => {
    const c = candidate('a', [8192])
    c.config.skippedSteps = [{ ctx: 16384, reason: 'kv', skip: { resource: 'vram', estimateBytes: 16 * GiB, budgetBytes: 14 * GiB, ruleId: 'I-2.3', weightsBytes: 8 * GiB, kvBytes: 7 * GiB, overheadBytes: GiB } }]
    expect(interpret(V([c])).find((x) => x.ruleId === 'I-2.3')!.action).toBe('enable-kv-q8')
    c.config.skippedSteps = [{ ctx: 16384, reason: 'old', skip: { resource: 'vram', estimateBytes: 16 * GiB, budgetBytes: 14 * GiB, ruleId: 'I-2.3' } }]
    expect(interpret(V([c])).find((x) => x.ruleId === 'I-2.3')!.action).toBe('inspect-diagnostics')
  })
})

describe('F10 coverage and required context from every candidate', () => {
  it('an excluded first-rung OOM config still gets its coverage line (was: only "ok")', () => {
    const ok = candidate('ok'), dead = candidate('dead', [2048], 40, { status: 'fail', failureKind: 'oom', decodeTps: na('oom') })
    const t = text(interpret(V([ok, dead])), 'I-2.1')
    expect(t).toMatch(/^\[I-2\.1\] ok: /m)
    expect(t).toMatch(/^\[I-2\.1\] dead: no clean context measured in this run .*excluded from ranking/m)
  })
  it('a sole passing 32K row without a recorded warmup is "not verified", never "tested and failed … (pass)"', () => {
    const c = candidate('solo', [32768], 40, { warm: undefined })
    const t = text(interpret(V([c], 'long_context_coding', {}, { requiredContext: 32768 })), 'I-2.5')
    expect(t).not.toMatch(/tested and failed at 32K \(pass\)/)
    expect(t).toMatch(/measured at 32K but not verified \(warmup not recorded\)/)
  })
})

describe('R1–R5 (review-w4k): scope, identity and version strictness', () => {
  const genBase = (id: string, over: Record<string, unknown> = {}) => quality(10, () => true).map((r) => ({ ...r, genId: id, appliedTemplateKwargs: { enable_thinking: id !== 'off' }, templateHash: 'tpl', runtimeVersion: 'b1', modelFingerprint: 'm1', acceptedSampling: { temperature: 0 }, ...over } as QualityResult))
  const gq = (id: string, thinking: boolean, results: QualityResult[]): GenQuality => ({ gen: { id, thinking, ...(thinking ? { effort: 'low' } : {}), temperature: 0, source: 'default' }, results, samples: 1, stochastic: false, answerTokens: m(100), reasoningTokens: m(thinking ? 20 : 0), effectiveAnswerLatencyMs: m(2000), effectiveTps: m(50), reasoningMs: m(thinking ? 400 : 0), rawTps: m(60) })
  const thinker = (rows: QualityResult[]) => {
    const c = candidate('think')
    c.model = { ...c.model, genKnobs: { supportsThinking: true, effortValues: ['low'] } }
    c.quality = genBase('off').map((r, i) => ({ ...r, pass: i % 2 === 0 }))
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, rows)]
    return c
  }
  it('R1: f16 vs q8_0 configs are not paired even when the rows do not repeat the KV type', () => {
    const a = candidate('a'), b = candidate('b'); b.config.kvType = 'q8_0'
    const q = V([a, b]).trace.candidates.find((c) => c.configId !== V([a, b]).winner?.input.config.id)!.qualityVsWinner
    expect(q).not.toBeNull()
    expect(q!.difference).toBeNull()
    expect(q!.reason).toMatch(/not comparable: KV types differ \((f16 vs q8_0|q8_0 vs f16)\)/)
  })
  it('R2: one conflicting template hash among otherwise complete rows breaks the contract', () => {
    const rows = genBase('think-low').map((r, i) => (i === 1 ? { ...r, templateHash: 'DIFFERENT' } as QualityResult : r))
    const v = V([thinker(rows)])
    expect(v.ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!.why).toMatch(/templateHash not uniform/)
    expect(v.winner?.gen?.gq.gen.id ?? 'off').toBe('off')
  })
  it('R2: accepted sampling that differs from the config breaks the contract', () => {
    const v = V([thinker(genBase('think-low', { acceptedSampling: { temperature: 0.8 } }))])
    expect(v.ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!.why).toMatch(/accepted temperature differs from the config's 0/)
  })
  it('R3: mixed quality contexts are reported as mixed, and equal counterfactual totals are "tied"', () => {
    const c = candidate('a'); c.quality = c.quality.map((r, i) => ({ ...r, ctx: i % 2 ? 4096 : 2048 } as QualityResult))
    const b = V([c]).trace.candidates[0].basis.find((x) => x.component === 'quality')!
    expect(b).toMatchObject({ rung: null, scope: 'mixed (2048, 4096)' })
    const prior = candidate('prior'); prior.quality = []
    const measured = candidate('measured')
    const cf = V([measured, { ...prior, runs: measured.runs.map((r) => ({ ...r, configId: 'prior' })) }], 'general_chat').trace.counterfactuals.find((x) => x.configId === 'prior')!
    expect(cf.total).toBe(cf.vsTotal)
    expect(cf.result).toBe('tied with measured without those terms')
  })
  it('R4: a versions object missing its prompts version is no version proof', () => {
    const c = candidate('partial-version'); c.runs = c.runs.map((r) => ({ ...r, versions: { ...VER, prompts: undefined as unknown as string } }))
    const v = V([c])
    expect(v.winner).toBeNull()
    expect(v.sessionVersion).toBeNull()
  })
  it('R5: the q8_0 remedy uses the estimator exact ratio (13.245 GiB budget does not fit 8 + 8·34/64 + 1)', () => {
    const c = candidate('a', [8192])
    c.config.skippedSteps = [{ ctx: 16384, reason: 'kv', skip: { resource: 'vram', estimateBytes: 17 * GiB, budgetBytes: 13.245 * GiB, ruleId: 'I-2.3', weightsBytes: 8 * GiB, kvBytes: 8 * GiB, overheadBytes: GiB } }]
    expect(interpret(V([c])).find((x) => x.ruleId === 'I-2.3')!.action).not.toBe('enable-kv-q8')
    c.config.skippedSteps[0].skip!.budgetBytes = 13.25 * GiB
    expect(interpret(V([c])).find((x) => x.ruleId === 'I-2.3')!.action).toBe('enable-kv-q8')
  })
})

describe('A/B/C (review-w4l): generation contract strictness', () => {
  const rowsFor = (id: string, over: Record<string, unknown> = {}) => quality(10, () => true).map((r) => ({ ...r, genId: id, templateHash: 'tpl', runtimeVersion: 'b1', modelFingerprint: 'm1', acceptedSampling: { temperature: 0 }, ...over } as QualityResult))
  const gq = (id: string, thinking: boolean, results: QualityResult[]): GenQuality => ({ gen: { id, thinking, ...(thinking ? { effort: 'low' } : {}), temperature: 0, source: 'default' }, results, samples: 1, stochastic: false, answerTokens: m(100), reasoningTokens: m(thinking ? 20 : 0), effectiveAnswerLatencyMs: m(2000), effectiveTps: m(50), reasoningMs: m(thinking ? 400 : 0), rawTps: m(60) })
  const thinkModel = (c: CandidateInput) => ({ ...c, model: { ...c.model, genKnobs: { supportsThinking: true, effortValues: ['low', 'high'] } } })
  const bound = (rows: QualityResult[], applied: Record<string, unknown>, temperature = 0): QualityResult[] => rows.map((r) => {
    const hash = 'a'.repeat(64), alt = 'b'.repeat(64)
    const tagged = { ...r, sample: 1, promptSha256: hash, appliedTemplateKwargs: applied,
      requestedSampling: { temperature, topP: null, topK: null, minP: null, seed: 1 }, acceptedSampling: { temperature, seed: 1 },
      templateKwargProof: Object.fromEntries(Object.entries(applied).map(([key, requested]) => [key, {
        requested, counterfactual: key === 'enable_thinking' ? !requested : requested === 'low' ? 'high' : 'low',
        requestedSha256: hash, counterfactualSha256: alt, status: 'proved'
      }])) }
    return { ...tagged, renderProof: { rowId: proofRowId(tagged), promptSha256: hash, renderedSha256: hash,
      counterfactualSha256: alt, counterfactuals: Object.fromEntries(Object.keys(applied).map((key) => [key, alt])),
      keys: Object.keys(applied), status: 'proved' } } as QualityResult
  })
  const renderOf = (r: QualityResult) => (r as QualityResult & { renderProof: NonNullable<GenRow['renderProof']> }).renderProof
  it('A: a T=0 baseline whose rows report accepted temperature 0.8 is not confirmed', () => {
    const c = candidate('a')
    c.quality = rowsFor('off', { acceptedSampling: { temperature: 0.8 } })
    const v = V([c])
    expect(v.winner).toBeNull()
    expect(v.ranked[0].undecided.find((u) => u.component === 'quality')).toMatchObject({ kind: 'contract-error', reason: expect.stringMatching(/accepted temperature differs from the config's 0/) })
  })
  it('A: the same contradiction on an "off" gen option excludes it (not silently accepted)', () => {
    const c = thinkModel(candidate('a'))
    c.quality = rowsFor('off', { appliedTemplateKwargs: { enable_thinking: false } })
    c.genQuality = [gq('off', false, rowsFor('off', { appliedTemplateKwargs: { enable_thinking: false }, acceptedSampling: { temperature: 0.8 } }))]
    expect(V([c]).ranked[0].genOptions[0]).toMatchObject({ comparable: false, why: expect.stringMatching(/accepted temperature differs/) })
  })
  it('B: rows applied with {enable_thinking:false, reasoning_effort:"high"} do not count for a think-low config', () => {
    const c = thinkModel(candidate('a'))
    c.quality = rowsFor('off', { appliedTemplateKwargs: { enable_thinking: false } })
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, rowsFor('think-low', { appliedTemplateKwargs: { enable_thinking: false, reasoning_effort: 'high' } }))]
    const o = V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!
    expect(o.comparable).toBe(false)
    expect(o.why).toMatch(/applied template kwargs .* differ from the config's \{"enable_thinking":true,"reasoning_effort":"low"\}/)
  })
  it('I-8.0 historical applied-kwargs claim without per-key proof is not comparable; proved rows are', () => {
    const c = thinkModel(candidate('a'))
    c.quality = bound(rowsFor('off'), { enable_thinking: false })
    const claimed = rowsFor('think-low', { appliedTemplateKwargs: { enable_thinking: true, reasoning_effort: 'low' } })
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, claimed)]
    const old = V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!
    expect(old).toMatchObject({ comparable: false, why: expect.stringMatching(/template.*proof|per-key.*proof/) })
    const proved = bound(claimed, { enable_thinking: true, reasoning_effort: 'low' })
    c.genQuality[1] = gq('think-low', true, proved)
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: true })
  })
  it('I-8.0 rejects imported proved effort with null counterfactual despite distinct render hashes', () => {
    const c = thinkModel(candidate('a'))
    c.quality = rowsFor('off', { appliedTemplateKwargs: { enable_thinking: false } })
    const proof = { enable_thinking: { requested: true, counterfactual: false, requestedSha256: 'a'.repeat(64), counterfactualSha256: 'b'.repeat(64), status: 'proved' },
      reasoning_effort: { requested: 'low', counterfactual: null, requestedSha256: 'a'.repeat(64), counterfactualSha256: 'b'.repeat(64), status: 'proved' } }
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, rowsFor('think-low', { appliedTemplateKwargs: { enable_thinking: true, reasoning_effort: 'low' }, templateKwargProof: proof }))]
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false, why: expect.stringMatching(/per-key template proof/) })
    const wrongDomain = { ...proof, reasoning_effort: { ...proof.reasoning_effort, counterfactual: true } }
    c.genQuality[1] = gq('think-low', true, rowsFor('think-low', { appliedTemplateKwargs: { enable_thinking: true, reasoning_effort: 'low' }, templateKwargProof: wrongDomain }))
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false, why: expect.stringMatching(/per-key template proof/) })
  })
  it('I-8.0 binds render proof to each row and rejects a copied proof from another prompt', () => {
    const c = thinkModel(candidate('a'))
    c.quality = bound(rowsFor('off'), { enable_thinking: false })
    const valid = bound(rowsFor('think-low'), { enable_thinking: true, reasoning_effort: 'low' })
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, valid)]
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')?.comparable).toBe(true)
    const copied = valid.map((r, i) => i === 1 ? { ...r, promptSha256: 'c'.repeat(64) } as QualityResult : r)
    c.genQuality[1] = gq('think-low', true, copied)
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false })
  })
  it('I-8.0 reconstructed render proof cannot become proved without the original prompt hash', () => {
    const c = thinkModel(candidate('a'))
    c.quality = bound(rowsFor('off'), { enable_thinking: false })
    const valid = bound(rowsFor('think-low'), { enable_thinking: true, reasoning_effort: 'low' })
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, valid)]
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')?.comparable).toBe(true)
    const reconstructed = valid.map((r, i) => i === 1 ? { ...r,
      proofProvenance: { originalPromptHashPresent: false },
      renderProof: { ...renderOf(r), status: 'reconstructed', promptSha256: renderOf(r).renderedSha256 }
    } as QualityResult : r)
    c.genQuality[1] = gq('think-low', true, reconstructed)
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false })
  })
  it('I-5.7 excludes an infra error despite valid template proof; I-5.8 keeps a proved truncation comparable', () => {
    const c = thinkModel(candidate('a'))
    c.quality = bound(rowsFor('off'), { enable_thinking: false })
    const valid = bound(rowsFor('think-low'), { enable_thinking: true, reasoning_effort: 'low' })
    for (const status of ['infra_error', 'truncated'] as const) {
      const rows = valid.map((r, i) => i === 1 ? { ...r, evaluationStatus: status, pass: false, score: 0,
        ...(status === 'truncated' ? { outputTruncated: true } : {}) } as QualityResult : r)
      c.genQuality = [gq('off', false, c.quality), gq('think-low', true, rows)]
      const option = V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!
      if (status === 'infra_error') {
        expect(option.comparable).toBe(false)
        expect(option.why).toMatch(/quality quarantined \(infrastructure error\)/)
        expect(option.cs.components.quality.input.kind).not.toBe('measured')
      } else {
        expect(option.comparable).toBe(true)
        expect(option.cs.components.quality.input.kind).toBe('measured')
      }
    }
  })
  it('I-8.0 quarantines one contradictory render row while valid rows remain comparable', () => {
    const c = thinkModel(candidate('a'))
    c.quality = bound(rowsFor('off'), { enable_thinking: false })
    const valid = bound(rowsFor('think-low'), { enable_thinking: true, reasoning_effort: 'low' })
    const contradicted = valid.map((r, i) => i === 1 ? { ...r,
      renderProof: { ...renderOf(r), status: 'contradicted' }
    } as QualityResult : r)
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, contradicted)]
    const option = V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!
    expect(option.comparable).toBe(true)
    expect(option.cs.components.quality.input.kind).toBe('measured')
  })
  it('I-8.0 requires runtime accepted sample seed to match the requested seed on every row', () => {
    const c = candidate('seed')
    const rows = rowsFor('off', { sample: 1, requestedSampling: { temperature: 0, topP: null, topK: null, minP: null, seed: 1 }, acceptedSampling: { temperature: 0, seed: 1 } })
    c.quality = rows
    expect(V([c]).ranked[0].qualityMeasured).toBe(true)
    c.quality = rows.map((r, i) => i === 1 ? { ...r, acceptedSampling: { temperature: 0, seed: 999 } } as QualityResult : r)
    expect(V([c]).ranked[0]).toMatchObject({ qualityMeasured: false, undecided: expect.arrayContaining([expect.objectContaining({ component: 'quality', kind: 'contract-error' })]) })
    c.quality = rows.map((r, i) => i === 1 ? { ...r, acceptedSampling: { temperature: 0 } } as QualityResult : r)
    expect(V([c]).ranked[0].qualityMeasured).toBe(false)
  })
  it('I-8.0 migrated single-toggle proof remains comparable; explicit contradictory proof rejects it', () => {
    const c = candidate('a')
    c.model = { ...c.model, genKnobs: { supportsThinking: true } }
    c.quality = bound(rowsFor('off'), { enable_thinking: false })
    const legacy = bound(rowsFor('think'), { enable_thinking: true })
    const on = { ...gq('think', true, legacy), gen: { id: 'think', thinking: true, temperature: 0, source: 'default' as const } }
    c.genQuality = [gq('off', false, c.quality), on]
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think')).toMatchObject({ comparable: true })
    const contradicted = legacy.map((r) => ({ ...r, templateKwargProof: { enable_thinking: { requested: true, counterfactual: false,
      requestedSha256: 'a'.repeat(64), counterfactualSha256: 'a'.repeat(64), status: 'unchanged' } } } as QualityResult))
    c.genQuality[1] = { ...on, results: contradicted }
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think')).toMatchObject({ comparable: false, why: expect.stringMatching(/template.*proof|per-key.*proof/) })
  })
  it('C: one row with ctx 2048 among rows without ctx is a partial scope, not a single measured rung', () => {
    const c = candidate('a'); c.quality = c.quality.map((r, i) => (i === 0 ? { ...r, ctx: 2048 } as QualityResult : r))
    const b = V([c]).trace.candidates[0].basis.find((x) => x.component === 'quality')!
    expect(b.rung).toBeNull()
    expect(b.scope).toBe('partial (1 of 60 rows carry ctx 2048)')
  })
})

describe('I-8.0 off/think comparator proof on both sides', () => {
  const rowsFor = (id: 'off' | 'think-low', proof: Record<string, unknown> = {}) => quality(10, () => true).map((r) => ({
    ...r, genId: id, templateHash: 'tpl', runtimeVersion: 'b1', modelFingerprint: 'm1',
    ...proof
  } as QualityResult))
  const gq = (id: 'off' | 'think-low', results: QualityResult[]): GenQuality => ({
    gen: { id, thinking: id !== 'off', ...(id === 'think-low' ? { effort: 'low' } : {}), temperature: 1, source: 'model-card' },
    results, samples: 1, stochastic: true, answerTokens: m(100), reasoningTokens: m(id === 'off' ? 0 : 20),
    effectiveAnswerLatencyMs: m(2000), effectiveTps: m(50), reasoningMs: m(id === 'off' ? 0 : 400), rawTps: m(60)
  })
  const withRowProof = (rows: QualityResult[]): QualityResult[] => rows.map((r) => {
    const x = r as QualityResult & { appliedTemplateKwargs?: Record<string, unknown>; acceptedSampling?: Record<string, unknown> | null; templateKwargProof?: GenRow['templateKwargProof'] }
    const applied = x.appliedTemplateKwargs ?? {}
    const hash = 'a'.repeat(64), alt = 'b'.repeat(64)
    const tagged = { ...r, sample: 1, promptSha256: hash,
      requestedSampling: { temperature: 1, topP: null, topK: null, minP: null, seed: 1 },
      acceptedSampling: x.acceptedSampling === null ? null : { ...x.acceptedSampling, seed: 1 },
      templateKwargProof: x.templateKwargProof ?? Object.fromEntries(Object.entries(applied).map(([key, requested]) => [key, {
        requested, counterfactual: key === 'enable_thinking' ? !requested : 'high', requestedSha256: hash,
        counterfactualSha256: alt, status: 'proved'
      }])) }
    return { ...tagged, renderProof: { rowId: proofRowId(tagged), promptSha256: hash, renderedSha256: hash,
      counterfactualSha256: alt, counterfactuals: Object.fromEntries(Object.keys(applied).map((key) => [key, alt])),
      keys: Object.keys(applied), status: Object.keys(applied).length ? 'proved' : 'unproved' } } as QualityResult
  })
  const assess = (offProof: Record<string, unknown>, missingOneAccepted = false) => {
    const c = candidate('a')
    c.model = { ...c.model, genKnobs: { supportsThinking: true, effortValues: ['low', 'high'] } }
    c.quality = withRowProof(rowsFor('off', offProof))
    if (missingOneAccepted) c.quality[0] = { ...c.quality[0], acceptedSampling: null } as QualityResult
    c.genQuality = [gq('off', c.quality), gq('think-low', withRowProof(rowsFor('think-low', {
      appliedTemplateKwargs: { enable_thinking: true, reasoning_effort: 'low' }, acceptedSampling: { temperature: 1 },
      templateKwargProof: {
        enable_thinking: { requested: true, counterfactual: false, requestedSha256: 'a'.repeat(64), counterfactualSha256: 'b'.repeat(64), status: 'proved' },
        reasoning_effort: { requested: 'low', counterfactual: 'high', requestedSha256: 'a'.repeat(64), counterfactualSha256: 'b'.repeat(64), status: 'proved' }
      }
    })))]
    const verdict = V([c])
    return { options: verdict.ranked[0].genOptions, chosen: verdict.ranked[0].gen?.gq.gen.id ?? null, verdict }
  }
  const offValid = { appliedTemplateKwargs: { enable_thinking: false }, acceptedSampling: { temperature: 1 } }

  it('compares a fully proven think config with a fully proven off control', () => {
    const { options } = assess(offValid)
    expect(options.find((g) => g.gq.gen.id === 'off')).toMatchObject({ comparable: true })
    expect(options.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: true })
  })
  it('does not compare T=1 think against T=1 off with one missing accepted temperature', () => {
    const { options, chosen } = assess(offValid, true)
    expect(options.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false, why: expect.stringMatching(/baseline.*runtime-accepted temperature|runtime-accepted temperature.*baseline/) })
    expect(chosen).not.toBe('think-low')
  })
  it('does not borrow measured off quality when requested T=1 but runtime accepted T=0', () => {
    const { options, chosen, verdict } = assess({ ...offValid, acceptedSampling: { temperature: 0 } })
    expect(options.find((g) => g.gq.gen.id === 'off')).toMatchObject({ comparable: false })
    expect(options.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false, why: expect.stringMatching(/baseline.*accepted temperature|accepted temperature.*baseline/) })
    expect(chosen).toBeNull()
    expect(verdict.ranked[0].cs.components.quality.input.kind).not.toBe('measured')
    expect(verdict.ranked[0].qualityMeasured).toBe(false)
    expect(verdict.ranked[0].undecided).toEqual(expect.arrayContaining([
      expect.objectContaining({ component: 'quality', kind: 'contract-error', reason: expect.stringMatching(/accepted temperature.*config's 1/) })
    ]))
    expect(verdict.winner).toBeNull()
  })
  it('does not compare valid think rows against off rows missing applied kwargs', () => {
    const { options, chosen } = assess({ acceptedSampling: { temperature: 1 } })
    expect(options.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false, why: expect.stringMatching(/baseline.*applied template kwargs|applied template kwargs.*baseline/) })
    expect(chosen).not.toBe('think-low')
  })
  it('excludes T=1 off and think comparison when off applied kwargs enable thinking', () => {
    const { options, chosen, verdict } = assess({ ...offValid, appliedTemplateKwargs: { enable_thinking: true } })
    expect(options.find((g) => g.gq.gen.id === 'off')).toMatchObject({ comparable: false })
    expect(options.find((g) => g.gq.gen.id === 'think-low')).toMatchObject({ comparable: false, why: expect.stringMatching(/baseline.*applied template kwargs|applied template kwargs.*baseline/) })
    expect(chosen).toBeNull()
    expect(verdict.winner).toBeNull()
  })
})

describe('I-2.8 placement spill (w4m: same-window reading required, heuristic)', () => {
  const B = (g: number) => ({ ...machine(), vramEffectiveBudgetBytes: m(g * GiB) })
  const shape = { peakVramBytes: m(11.6 * GiB), peakSharedGpuBytes: m(0), peakSharedGpuRawBytes: m(1.08 * GiB) }
  it('not re-measured: warn + restart-runtime, cause stated as unconfirmed (never "not a capacity limit")', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[1] = { ...c.runs[1], ...shape, adapterFreeAtSharedPeakBytes: m(3.1 * GiB) }
    const i = interpret(V([c])).find((x) => x.ruleId === 'I-2.8')!
    expect(i).toMatchObject({ severity: 'warn', action: 'restart-runtime' })
    expect(i.text).toMatch(/1\.08 GiB of this process was resident in shared memory while 3\.10 GiB dedicated VRAM was free in the same window — not re-measured: cause unconfirmed/)
    expect(i.text).not.toMatch(/not a capacity limit/)
  })
  it('w4m-5: without the same-window adapter reading the rule is not evaluable — a budget is no substitute', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[1] = { ...c.runs[1], ...shape, peakVramBytes: m(8 * GiB) }
    expect(interpret(V([c], 'max_quality', { machine: B(12.8) })).some((x) => x.ruleId === 'I-2.8')).toBe(false)
    c.runs[1] = { ...c.runs[1], adapterFreeAtSharedPeakBytes: m(0.2 * GiB) }
    expect(interpret(V([c])).some((x) => x.ruleId === 'I-2.8')).toBe(false) // card full
  })
  it('a cleared retry reports both observations and an unknown cause without a placement or capacity claim', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[1] = { ...c.runs[1], placementRetry: true, placementFirst: { ...shape, adapterFreeAtSharedPeakBytes: m(3.1 * GiB), decodeTps: m(37) } }
    const i = interpret(V([c])).find((x) => x.ruleId === 'I-2.8')!
    expect(i.severity).toBe('note')
    expect(i.action ?? null).toBeNull()
    expect(i.text).toMatch(/it cleared after a fresh restart \(0\.00 GiB shared, decode 40\.0 vs 37\.0 t\/s before\) — cause unknown \(likely a host-visible buffer\)/)
    expect(i.text).not.toMatch(/previous large load|driver placement|confirmed placement|capacity limit|capacity confirmed/)
  })
  it('w4m-4: a retry that still shows residency is never called placement', () => {
    const c = candidate('a', [2048, 4096])
    c.runs[1] = { ...c.runs[1], ...shape, placementRetry: true, placementFirst: { ...shape, adapterFreeAtSharedPeakBytes: m(3.1 * GiB), decodeTps: m(37) } }
    expect(interpret(V([c])).some((x) => x.ruleId === 'I-2.8')).toBe(false)
  })
  it('catalog: I-2.8 is heuristic until the A/B confirms', () => {
    expect(RULES.find((r) => r.id === 'I-2.8')).toMatchObject({ origin: 'heuristic', action: 'restart-runtime' })
  })
})

describe('w4m-6: required sampling fields must be present, finite and matching', () => {
  const rowsFor = (id: string, over: Record<string, unknown> = {}) => quality(10, () => true).map((r) => ({ ...r, genId: id, templateHash: 'tpl', runtimeVersion: 'b1', modelFingerprint: 'm1', ...over } as QualityResult))
  const think = (c: CandidateInput) => ({ ...c, model: { ...c.model, genKnobs: { supportsThinking: true, effortValues: ['low'] } } })
  const gq = (id: string, thinking: boolean, results: QualityResult[]): GenQuality => ({ gen: { id, thinking, ...(thinking ? { effort: 'low' } : {}), temperature: 0, source: 'default' }, results, samples: 1, stochastic: false, answerTokens: m(100), reasoningTokens: m(20), effectiveAnswerLatencyMs: m(2000), effectiveTps: m(50), reasoningMs: m(400), rawTps: m(60) })
  it('{seed} alone is not evaluable for a thinking config (temperature never reported)', () => {
    const c = think(candidate('a'))
    c.quality = rowsFor('off', { appliedTemplateKwargs: { enable_thinking: false }, acceptedSampling: { temperature: 0 } })
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, rowsFor('think-low', { appliedTemplateKwargs: { enable_thinking: true, reasoning_effort: 'low' }, acceptedSampling: { seed: 42 } }))]
    const o = V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!
    expect(o).toMatchObject({ comparable: false, why: expect.stringMatching(/runtime-accepted temperature not recorded on every row/) })
  })
  it('a non-finite temperature counts as absent, not as agreement', () => {
    const c = think(candidate('a'))
    c.quality = rowsFor('off', { appliedTemplateKwargs: { enable_thinking: false }, acceptedSampling: { temperature: 0 } })
    c.genQuality = [gq('off', false, c.quality), gq('think-low', true, rowsFor('think-low', { appliedTemplateKwargs: { enable_thinking: true, reasoning_effort: 'low' }, acceptedSampling: { temperature: Number.NaN } }))]
    expect(V([c]).ranked[0].genOptions.find((g) => g.gq.gen.id === 'think-low')!.comparable).toBe(false)
  })
})

describe('I-3.9 backend comparison', () => {
  it('HIP vs Vulkan for the same config: measured decode/TTFT at the largest common rung and observed allocations', () => {
    const vk = candidate('m', [2048, 4096], 40)
    const hip = candidate('m', [2048, 4096], 52)
    hip.config = { ...hip.config, id: 'm|hip', backend: 'hip', device: 'ROCm0' }
    hip.runs = hip.runs.map((r) => ({ ...r, configId: 'm|hip', peakVramBytes: m(14.5 * GiB) }))
    const t = text(interpret(V([vk, hip])), 'I-3.9')
    expect(t).toBe('[I-3.9] HIP vs Vulkan for m: decode 52.0 vs 40.0 t/s at 4K; TTFT 1000 vs 1000 ms; largest observed dedicated allocation 14.50 GiB vs 8.00 GiB (per-PID measured peak, not a capacity ceiling).')
    expect(text(interpret(V([vk])), 'I-3.9')).toBe('')
  })
})

describe('w4n-N3 insight: unknown re-measurement is unknown', () => {
  it('a retry without a shared reading is never "cleared" / "not a capacity limit"', () => {
    const c = candidate('a', [2048, 4096])
    const shape = { peakVramBytes: m(8 * GiB), peakSharedGpuBytes: m(0), peakSharedGpuRawBytes: m(1.4 * GiB) }
    c.runs[1] = { ...c.runs[1], peakSharedGpuBytes: na('counter lost'), peakSharedGpuRawBytes: na('counter lost'), placementRetry: true, placementFirst: { ...shape, adapterFreeAtSharedPeakBytes: m(6.92 * GiB), decodeTps: m(37) } }
    const i = interpret(V([c])).find((x) => x.ruleId === 'I-2.8')!
    expect(i).toMatchObject({ severity: 'warn', action: 'restart-runtime' })
    expect(i.text).toMatch(/that attempt had no shared-memory reading: cause unknown/)
    expect(i.text).not.toMatch(/cleared|not a capacity limit/)
  })
})
