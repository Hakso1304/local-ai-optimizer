// Executed scenarios from Worker #4's re-review of interp-2 (docs/review-w4i-2026-09-27.md), one per gap.
// G02/G14 (renderer) are #2's; G06 producer wiring is covered in tests/benchmark/runner-history.test.ts.
import { describe, expect, it } from 'vitest'
import type { BenchmarkRunResult, CandidateInput, GenQuality, MachineLimits, Metric, ModelMeta, QualityCategory, QualityResult, WorkloadId } from '../../src/shared/bench-types'
import { interpret, verdicts, type Insight } from '../../src/core/interpret'
import { recommend, recommendForWorkload } from '../../src/core/scoring/recommend'
import { DEFAULT_SCORING_CONFIG } from '../../src/core/scoring/workloads'
import fixture from '../fixtures/scoring/session-single.json'

const GiB = 1024 ** 3
const m = (value: number): Metric => ({ value, kind: 'measured', source: 'synthetic' })
const na = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
const machine = (): MachineLimits => ({ vramBytes: m(16 * GiB), vramInUseBytes: m(GiB), ramTotalBytes: m(32 * GiB), ramAvailableBytes: m(24 * GiB), physicalCores: 8, gpuDevice: 'Vulkan0' })
const CATS: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']
const quality = (n = 10, pass: (i: number) => boolean = (i) => i < 8, cats = CATS): QualityResult[] =>
  cats.flatMap((category) => Array.from({ length: n }, (_, i) => ({ testId: `${category}-${i}`, category, weight: 1, pass: pass(i), score: pass(i) ? 1 : 0, detail: '', evaluationStatus: 'valid' } as QualityResult)))
const VER = { benchmark: 'bench-1.0.0', prompts: 'ladder-2', quality: 'qb-1.1.0', runtime: 'synthetic' }
function run(configId: string, ctx: number, decode = 40, over: Partial<BenchmarkRunResult> = {}): BenchmarkRunResult {
  return { configId, ctx, promptTokens: ctx * 0.75, status: 'pass', failureKind: null, warm: true,
    loadTimeMs: m(500), ttftMs: m(1000), prefillTps: m(2000), decodeTps: m(decode), totalMs: m(2000),
    peakVramBytes: m(8 * GiB), peakSharedGpuBytes: m(0), peakRamBytes: m(GiB), minRamAvailBytes: m(20 * GiB), avgGpuUtil: m(80), avgCpuUtil: m(10),
    versions: VER, ...over }
}
function candidate(id: string, contexts = [8192], decode = 40, over: Partial<BenchmarkRunResult> = {}): CandidateInput {
  const model: ModelMeta = { ...(fixture.models[0] as ModelMeta), id, name: `Synthetic ${id}`, ctxTrain: 131072 }
  return {
    model, config: { id, modelId: id, device: 'Vulkan0', gpuLayers: model.layers, gpuLayersAll: true, kvType: 'f16', flashAttn: true, threads: 8, ctxSteps: contexts, skippedSteps: [], estVramBytes: m(8 * GiB), estRamBytes: m(GiB), notes: [], mmap: false },
    runs: contexts.map((ctx) => run(id, ctx, decode, over)), quality: quality()
  }
}
const panel = (cs: CandidateInput[], w: WorkloadId = 'max_quality', hw = machine(), extra = {}): Insight[] => interpret(verdicts({ candidates: cs, machine: hw, ...extra }, w))
const text = (ins: Insight[], id: string) => ins.filter((i) => i.ruleId === id).map((i) => i.text).join('\n')

describe('G01 safety and provenance before ranking', () => {
  it('a measured RAM minimum below the recorded floor makes the pick unsafe (hard), not a confirmed winner', () => {
    const c = candidate('a', [8192], 40, { minRamAvailBytes: m(3 * GiB), ramFloorBytes: 4 * GiB })
    const v = verdicts({ candidates: [c], machine: machine() }, 'max_quality')
    expect(v.winner).toBeNull()
    expect(v.ranked[0].failures.map((f) => f.ruleId)).toContain('I-4.3')
    expect(v.trace.candidates[0].safety.ramFloor).toBe('violated')
  })
  it('estimated peak VRAM and unmeasured spill never confirm: the memory term is not measured → provisional', () => {
    const c = candidate('a', [8192], 40, { peakVramBytes: { value: 8 * GiB, kind: 'estimated', source: 'estimate' }, peakSharedGpuBytes: na('counter missing') })
    const v = verdicts({ candidates: [c], machine: machine() }, 'max_quality')
    expect(v.winner).toBeNull()
    expect(v.provisionalWinner?.input.config.id).toBe('a')
    expect(v.ranked[0].cs.components.memory.input.kind).not.toBe('measured')
  })
  it('an unsafe full offload cannot veto a safe partial config of the same model', () => {
    const full = candidate('full', [8192], 40, { minRamAvailBytes: m(3 * GiB), ramFloorBytes: 4 * GiB })
    const partial = candidate('partial', [8192], 30)
    partial.model = full.model; partial.config = { ...partial.config, modelId: full.model.id, gpuLayersAll: false, gpuLayers: 20 }
    const v = verdicts({ candidates: [full, partial], machine: machine() }, 'max_quality')
    expect(v.ranked.find((x) => x.input.config.id === 'partial')!.failures.map((f) => f.ruleId)).not.toContain('I-7.6')
    expect(v.winner?.input.config.id).toBe('partial')
  })
})

describe('G03 the common rung is a measured basis, and comparisons check their scope', () => {
  it('a candidate read at another rung than the common one is provisional and the trace says so', () => {
    const short = candidate('short', [4096, 16384], 100), exact = candidate('exact', [8192], 40)
    const v = verdicts({ candidates: [short, exact], machine: machine() }, 'max_quality')
    expect(v.scoringRung).toBe(8192)
    const s = v.ranked.find((x) => x.input.config.id === 'short')!
    expect(s.cs.referenceCtx).toBe(4096)
    expect(s.confirmed).toBe(false)
    expect(s.undecided.map((u) => u.kind)).toContain('unmatched-rung')
    expect(v.winner?.input.config.id).toBe('exact')
  })
  it('different token budgets or generation configs on the shared items are not paired (neutralized with the reason)', () => {
    const a = candidate('a'), b = candidate('b')
    b.quality = b.quality.map((r) => ({ ...r, maxTokens: 999 })); a.quality = a.quality.map((r) => ({ ...r, maxTokens: 64 }))
    const v = verdicts({ candidates: [a, b], machine: machine() }, 'max_quality')
    expect(v.trace.comparisons[0]).toMatchObject({ basis: 'without-quality', reason: expect.stringMatching(/not comparable: token budgets differ/) })
  })
})

describe('G04 request overrides apply inside the shared entry point', () => {
  it('recommend() and verdicts() honour minDecodeTps numerically, like recommendForWorkload()', () => {
    const c = candidate('a')
    expect(recommend([c], machine(), 'max_quality', undefined, [], { minDecodeTps: 100 }).best).toBeNull()
    expect(recommendForWorkload({ candidates: [c], machine: machine() }, 'max_quality', { minDecodeTps: 100 }).best).toBeNull()
    expect(verdicts({ candidates: [c], machine: machine() }, 'max_quality', { minDecodeTps: 100 }).ranked[0].failures[0].text).toMatch(/below your floor 100 t\/s/)
  })
  it('requiredContext likewise gates numerically', () => {
    expect(verdicts({ candidates: [candidate('a', [8192])], machine: machine() }, 'max_quality', { requiredContext: 32768 }).winner).toBeNull()
  })
})

describe('G05 the trace records what actually decided', () => {
  it('a neutralized win shows the totals without quality in the deciding tie-break chain', () => {
    const higherQ = candidate('higherQ', [8192], 10), faster = candidate('faster', [8192], 40)
    higherQ.quality = quality(10, (i) => i <= 7); faster.quality = quality(10, (i) => i >= 2 && i <= 8)
    const r = recommend([higherQ, faster], machine(), 'general_chat')
    const t = r.decisionTrace!
    expect(r.best?.configId).toBe('faster')
    const cmpEv = t.comparisons.find((c) => c.basis === 'without-quality')!
    expect(cmpEv.winner).toBe('faster')
    expect(t.tieBreakChain[0]).toMatchObject({ step: 'total without quality', a: 'faster', b: 'higherQ', decided: true })
    expect(Number(t.tieBreakChain[0].a_value)).toBeGreaterThan(Number(t.tieBreakChain[0].b_value))
    expect(t.eligibleSet.map((e) => e.configId).sort()).toEqual(['faster', 'higherQ'])
    expect(t.thresholdsUsed['cliff.decodeDropRatio']).toBe(DEFAULT_SCORING_CONFIG.cliff.decodeDropRatio)
  })
  it('excluded candidates appear in the eligible set with their failing rule', () => {
    const dead = candidate('dead', [2048], 40, { status: 'fail', failureKind: 'oom', decodeTps: na('oom') })
    const t = recommend([candidate('a'), dead], machine(), 'max_quality').decisionTrace!
    expect(t.eligibleSet.find((e) => e.configId === 'dead')).toEqual({ configId: 'dead', failingRuleIds: ['I-6.3'] })
  })
})

describe('G07 memory and spill wording keep scope', () => {
  it('unavailable in-use VRAM at planning is stated as unavailable, never 0', () => {
    const c = candidate('a')
    c.config.planning = { vramTotalBytes: 16 * GiB, vramInUse: na('adapter counter unavailable'), planningVramBudgetBytes: 15 * GiB, planningReserveBytes: GiB, ramAvailableBytes: 24 * GiB, ramReserveBytes: 4 * GiB, candidateRulesVersion: 'cand-1.4' }
    const t = text(panel([c]), 'I-4.1')
    expect(t).toMatch(/in use at planning unavailable \(adapter counter unavailable\)/)
    expect(t).not.toMatch(/0\.00 GiB unavailable/)
  })
  it('a raw-growth spill is labelled raw growth, not adjusted spill, and marked provisional', () => {
    const c = candidate('a', [2048, 4096], 40)
    c.runs[0].peakSharedGpuRawBytes = m(0.05 * GiB); c.runs[1].peakSharedGpuRawBytes = m(1.15 * GiB)
    const t = text(panel([c]), 'I-2.2')
    expect(t).toMatch(/raw shared-GPU usage \(host-pinned excluded\) grew \+1\.10 GiB at 4K vs 2K — above the 1\.00 GiB growth rule; the adjusted spill stayed below 0\.25 GiB/)
    expect(t).toMatch(/provisional/)
  })
})

describe('G08 generation choice is deterministic and needs application proof', () => {
  const genRows = (id: string, reasoning: number, applied: Record<string, unknown> | undefined) =>
    quality().map((r) => ({ ...r, genId: id, sample: 1, answerTokens: 100, reasoningTokens: reasoning, totalMs: 2000, tokenSource: 'runtime', ...(applied ? { appliedTemplateKwargs: applied } : {}) }))
  const gq = (id: string, effort: string, reasoning: number, applied?: Record<string, unknown>): GenQuality => ({
    gen: { id, thinking: true, effort, temperature: 0, source: 'default' }, results: genRows(id, reasoning, applied), samples: 1, stochastic: false,
    answerTokens: m(100), reasoningTokens: m(reasoning), effectiveAnswerLatencyMs: m(2000), effectiveTps: m(50), reasoningMs: m(1000), rawTps: m(100)
  })
  const model = (c: CandidateInput) => ({ ...c, model: { ...c.model, genKnobs: { supportsThinking: true, effortValues: ['low', 'high'] } } })
  it('equal quality: the lower effort is kept whatever the input order', () => {
    for (const order of [['high', 'low'], ['low', 'high']]) {
      const c = model(candidate('a'))
      c.genQuality = order.map((e) => gq(`think-${e}`, e, e === 'high' ? 100 : 20, { enable_thinking: true, reasoning_effort: e }))
      expect(verdicts({ candidates: [c], machine: machine() }, 'max_quality').winner?.gen?.gq.gen.id, order.join()).toBe('think-low')
    }
  })
  it('an empty applied-kwargs record is not proof of application', () => {
    const c = model(candidate('a'))
    c.genQuality = [gq('think-low', 'low', 20, {})]
    expect(verdicts({ candidates: [c], machine: machine() }, 'max_quality').ranked[0].genOptions[0].comparable).toBe(false)
  })
})

describe('G09 version proof', () => {
  it('versionless rows are not speed-eligible when the session version is known', () => {
    const c = candidate('a'); c.runs = c.runs.map(({ versions: _v, ...r }) => r as BenchmarkRunResult)
    const v = verdicts({ candidates: [c], machine: machine(), sessionVersions: { benchmark: 'bench-1.0.0', prompts: 'ladder-2' } }, 'max_quality')
    expect(v.winner).toBeNull()
  })
  it('older superseded attempts in allRuns do not redefine the session version', () => {
    const c = candidate('a')
    const old = Array.from({ length: 5 }, (_, i) => ({ ...run('a', 8192), runId: `o${i}`, supersededBy: 'x', versions: { ...VER, prompts: 'ladder-1' } }))
    expect(verdicts({ candidates: [c], machine: machine(), allRuns: [...old, ...c.runs] }, 'max_quality').winner?.input.config.id).toBe('a')
  })
})

describe('G10 quality coverage and quarantine scope', () => {
  it('a category with one unique item (repeated) is limited coverage, and rates use unique items', () => {
    const c = candidate('a')
    c.quality = [...quality(10, () => true, ['instruction', 'reasoning', 'structured', 'extraction', 'context']),
      ...[1, 2, 3].map((s) => ({ testId: 'coding-0', category: 'coding' as const, weight: 1, pass: true, score: 1, detail: '', sample: s } as QualityResult))]
    const ins = panel([c])
    expect(text(ins, 'I-5.6')).toMatch(/< 3 items in coding/)
    expect(text(ins, 'I-5.1')).toMatch(/coding 1\/1/)
  })
  it('an infra error in a category the workload does not weight still quarantines the result', () => {
    const c = candidate('a')
    c.quality = c.quality.map((r, i) => (r.category === 'coding' && i % 10 === 0 ? { ...r, evaluationStatus: 'infra_error' } as QualityResult : r))
    const v = verdicts({ candidates: [c], machine: machine() }, 'general_chat')
    expect(v.ranked[0].cs.components.quality.quarantined).toBe(true)
    expect(v.winner).toBeNull()
  })
})

describe('G11 lifecycle and saturation need their observations', () => {
  it('no load-phase minimum → no mmap note (a drop during inference is not a load drop)', () => {
    const c = candidate('a', [2048, 4096])
    c.config.mmap = true
    c.runs[0] = { ...c.runs[0], ramAvailBeforeLoadBytes: m(20 * GiB), minRamAvailBytes: m(15 * GiB) }
    c.runs[1] = { ...c.runs[1], ramAvailBeforeLoadBytes: m(20 * GiB) }
    expect(text(panel([c]), 'I-4.4')).toBe('')
  })
})

describe('G12 action preconditions and gate insights', () => {
  it('a VRAM skip on a q8_0 config does not suggest KV q8 again', () => {
    const c = candidate('a', [8192])
    c.config.kvType = 'q8_0'
    c.config.skippedSteps = [{ ctx: 16384, reason: 'est. VRAM', skip: { resource: 'vram', estimateBytes: 20 * GiB, budgetBytes: 14 * GiB, ruleId: 'I-2.3' } }]
    const i = panel([c]).find((x) => x.ruleId === 'I-2.3')!
    expect(i.action).not.toBe('enable-kv-q8')
  })
  it('gate failures (context floor, quality minimum, stability) are emitted as actionable insights', () => {
    const c = candidate('a', [2048]); c.quality = quality(10, () => false)
    const ins = panel([c], 'coding')
    expect(ins.find((i) => i.ruleId === 'I-2.7')).toMatchObject({ action: 'inspect-diagnostics' })
    expect(ins.find((i) => i.ruleId === 'I-5.10')).toMatchObject({ action: 'run-thorough-quality' })
  })
})

describe('G13 coverage per candidate', () => {
  it('every configuration gets a coverage line, including one with no clean rung', () => {
    const a = candidate('a', [8192]), b = candidate('b', [2048], 40, { status: 'fail', failureKind: 'oom', decodeTps: na('oom') })
    const bad = candidate('c', [2048], 40, { status: 'degraded' as never })
    bad.model = a.model; bad.config.modelId = a.model.id
    const t = text(panel([a, bad, b]), 'I-2.1')
    expect(t).toMatch(/^\[I-2\.1\] a: largest clean context/m)
    expect(t).toMatch(/^\[I-2\.1\] c: /m)
  })
  it('a passing row at the required context without a recorded warmup is "not verified", not "tested and failed"', () => {
    const c = candidate('a', [8192, 32768]); c.runs[1] = { ...c.runs[1], warm: undefined }
    const t = text(panel([c], 'long_context_coding', machine(), {}), 'I-2.5')
    expect(t).toBe('')
    const req = interpret(verdicts({ candidates: [c], machine: machine() }, 'long_context_coding', { requiredContext: 32768 }))
    expect(text(req, 'I-2.5')).toMatch(/measured at 32K but not verified \(warmup not recorded\)/)
  })
})

describe('G15 executing policy is recorded and used', () => {
  it('a custom cliff config drives the text and thresholdsUsed', () => {
    const cfg = { ...DEFAULT_SCORING_CONFIG, cliff: { ...DEFAULT_SCORING_CONFIG.cliff, sharedSpillBytes: 512 * 1024 ** 2 } }
    const c = candidate('a', [2048, 4096], 40)
    c.runs[1] = { ...c.runs[1], peakSharedGpuBytes: m(0.7 * GiB) }
    const v = verdicts({ candidates: [c], machine: machine() }, 'max_quality', {}, cfg)
    expect(v.trace.thresholdsUsed['cliff.sharedSpillBytes']).toBe(512 * 1024 ** 2)
    expect(text(interpret(v), 'I-2.2')).toMatch(/exceeded 0\.50 GiB at 4K/)
  })
})
