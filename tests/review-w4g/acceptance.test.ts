import { beforeAll, describe, expect, it } from 'vitest'
import fixture from '../fixtures/scoring/session-single.json'
import { RULES, RULES_VERSION, interpret, verdicts } from '../../src/core/interpret'
import { recommend, recommendForWorkload } from '../../src/core/scoring/recommend'
import { componentScores } from '../../src/core/scoring/components'
import { detectCliffs } from '../../src/core/scoring/cliff'
import { categoryFlags, qualityUncertainty, type UncertaintyRow } from '../../src/core/scoring/uncertainty'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import type { BenchmarkRunResult, CandidateInput, GenQuality, MachineLimits, Metric, ModelMeta, QualityCategory } from '../../src/shared/bench-types'

// Synthetic observations only. No runners, telemetry, database, mocks or host state.
// Future fields are additive fixture contracts, not casts hiding current output values.
const GiB = 1024 ** 3
const m = (value: number): Metric => ({ value, kind: 'measured', source: 'synthetic acceptance fixture' })
const na = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
const machine = (): MachineLimits => ({ vramBytes: m(16 * GiB), vramInUseBytes: m(GiB), ramTotalBytes: m(32 * GiB), ramAvailableBytes: m(24 * GiB), physicalCores: 8, gpuDevice: 'Vulkan0' })
const categories: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']
function quality(n = 10, passes = 8): UncertaintyRow[] {
  return categories.flatMap(category => Array.from({ length: n }, (_, i) => ({ testId: `${category}-${i}`, skillId: `${category}-${i}`, generatorSeed: 1, sample: 1, category, weight: 1, pass: i < passes, score: i < passes ? 1 : 0, detail: '', evaluationStatus: 'valid' as const })))
}
function run(configId: string, ctx = 8192, decode = 40): BenchmarkRunResult {
  return { configId, ctx, promptTokens: ctx * .75, status: 'pass', failureKind: null, warm: true,
    loadTimeMs: m(500), ttftMs: m(1000), prefillTps: m(2000), decodeTps: m(decode), totalMs: m(2000),
    peakVramBytes: m(8 * GiB), peakSharedGpuBytes: m(0), peakRamBytes: m(GiB), minRamAvailBytes: m(20 * GiB), avgGpuUtil: m(80), avgCpuUtil: m(10),
    versions: { benchmark: 'synthetic-1', prompts: 'synthetic-1', quality: 'synthetic-1', runtime: 'synthetic-1' } }
}
function candidate(id = 'a', contexts = [8192]): CandidateInput {
  const model: ModelMeta = { ...fixture.models[0], id, name: `Synthetic ${id}`, ctxTrain: 131072 }
  return { model, config: { id, modelId: id, device: 'Vulkan0', gpuLayers: model.layers, gpuLayersAll: true, kvType: 'f16', flashAttn: true, threads: 8, ctxSteps: contexts, skippedSteps: [], estVramBytes: m(8 * GiB), estRamBytes: m(GiB), notes: [], mmap: false }, runs: contexts.map(ctx => run(id, ctx)), quality: quality() }
}
const inspect = (cs: CandidateInput[], hw = machine()) => interpret(verdicts({ candidates: cs, machine: hw }, 'max_quality'))
function gen(id: string, thinking: boolean, results = quality()): GenQuality {
  return { gen: { id, thinking, temperature: 0, source: 'default' }, results, samples: 1, stochastic: false, answerTokens: m(80), reasoningTokens: m(thinking ? 40 : 0), reasoningMs: m(thinking ? 500 : 0), rawTps: m(60), effectiveAnswerLatencyMs: m(thinking ? 2000 : 1000), effectiveTps: m(thinking ? 40 : 80) }
}

// Output additions required by I-7.2; adapt this adapter if the engine chooses other
// names, preserving every semantic assertion. No synthetic trace is supplied.
interface Trace {
  scoringRung: number
  eligibleSet: { configId: string; failingRuleIds: string[] }[]
  neutralizations: unknown[]
  tieBreakChain: unknown[]
  rulesVersion: string
  thresholdsUsed: Record<string, unknown>
  alternatives: Record<string, { configId: string | null; ruleId: string }>
}
const traceOf = (value: unknown) => (value as { decisionTrace?: Trace }).decisionTrace

beforeAll(() => {
  // These checks are deliberately NOT expected failures: fixture/API regressions
  // must fail the suite rather than masquerade as the intended acceptance defect.
  expect(componentScores(candidate(), machine(), WORKLOADS.max_quality).usable).toBe(true)
  expect(recommend([candidate()], machine(), 'max_quality').best?.configId).toBe('a')
  expect(detectCliffs(candidate().runs, 16 * GiB).practicalContextCeiling.value).toBe(8192)
})

describe('W4G v2 acceptance — remove it.fails only when each contract is implemented', () => {
  it('G01 repeats preserve the band and the report labels its heuristic coverage', () => {
    const c = candidate(), repeated = { ...c, quality: [...c.quality, ...c.quality, ...c.quality] }
    const oracle = qualityUncertainty(c.quality, { instruction: 1, reasoning: 1, coding: 1, structured: 1, extraction: 1, context: 1 })
    expect(qualityUncertainty(repeated.quality, { instruction: 1, reasoning: 1, coding: 1, structured: 1, extraction: 1, context: 1 })).toEqual(oracle)
    const a = componentScores(c, machine(), WORKLOADS.max_quality).components.quality
    const b = componentScores(repeated, machine(), WORKLOADS.max_quality).components.quality
    expect.soft(b.n).toBe(a.n)
    expect.soft(b.ci95).toBe(a.ci95)
    // Repeat invariance was fixed concurrently; the same gap's reporting contract
    // remains unmet. Do not remove the passing invariance assertions.
    const text = inspect([repeated]).filter(i => i.ruleId === 'I-5.1').map(i => i.text).join(' ')
    expect.soft(text).toMatch(/heuristic/i)
    expect.soft(text).toMatch(/skills/i)
    expect.soft(text).toMatch(/samples|completions/i)
  })

  it('G02 required context never overrides an explicit user decode floor', () => {
    const c = candidate('slow', [32768]); c.runs[0].decodeTps = m(10)
    const rec = recommendForWorkload({ candidates: [c], machine: machine() }, 'coding', { requiredContext: 32768, minDecodeTps: 20 })
    expect(rec.best).toBeNull()
    expect(rec.ranked.find(x => x.configId === 'slow')?.eligible).toBe(false)
  })

  it('G03 partial offload survives when full offload misses the required context', () => {
    const full = candidate('full', [8192]), partial = candidate('partial', [8192, 32768])
    partial.model = full.model; partial.config.modelId = full.model.id
    partial.config.gpuLayersAll = false; partial.config.gpuLayers = 20
    const rec = recommendForWorkload({ candidates: [full, partial], machine: machine() }, 'coding', { requiredContext: 32768, minDecodeTps: 20 })
    expect(rec.best?.configId).toBe('partial')
    expect(rec.ranked.find(x => x.configId === 'partial')?.eligible).toBe(true)
  })

  it('G04 infra_error quality is quarantined without discarding valid performance', () => {
    const c = candidate(); c.quality = quality().map((r, i) => i ? r : { ...r, evaluationStatus: 'infra_error' })
    const v = verdicts({ candidates: [c], machine: machine() }, 'max_quality')
    expect.soft(v.winner).toBeNull()
    expect.soft(v.ranked.find(x => x.input.config.id === 'a')?.cs.components.genSpeed.input.value).toBe(40)
    expect.soft(interpret(v).some(i => i.ruleId === 'I-5.7' && i.severity === 'critical' && /invalid|quarantin|infra/i.test(i.text))).toBe(true)
  })

  it('G05 a model prior is immutable when another measured candidate is added', () => {
    const prior = candidate('prior'); prior.quality = []
    const low = candidate('measured'); low.quality = quality(10, 1)
    const score = (cs: CandidateInput[]) => verdicts({ candidates: cs, machine: machine() }, 'general_chat').ranked.find(x => x.input.config.id === 'prior')!.cs.components.quality.score
    expect(score([prior, low])).toBe(score([prior]))
  })

  it('G06 device loss remains critical after a superseding successful retry', () => {
    const c = candidate()
    // Expected session input: allRuns: (BenchmarkRunResult & {runId: string,
    // supersededBy?: string, startedAt: number, endedAt: number})[]. runs is latest-only.
    const old = { ...run('a'), runId: 'old', status: 'fail' as const, failureKind: 'device_lost' as const, supersededBy: 'new', startedAt: 1000, endedAt: 2000 }
    const latest = { ...c.runs[0], runId: 'new', startedAt: 3000, endedAt: 4000 }
    const data = { candidates: [{ ...c, runs: [latest] }], machine: machine(), allRuns: [old, latest] }
    const insights = interpret(verdicts(data, 'max_quality'))
    expect(insights.some(i => i.ruleId === 'I-6.3' && i.severity === 'critical' && /device.lost|GPU reset/i.test(i.text))).toBe(true)
  })

  it('G07 unknown in-use VRAM cannot produce numeric planning headroom', () => {
    const hw = machine(); hw.vramInUseBytes = na('adapter counter unavailable')
    const headroom = inspect([candidate()], hw).filter(i => i.ruleId === 'I-4.1')
    // Omission or an explicit unavailable/named per-PID-vs-total basis is allowed.
    expect(headroom.every(i => /unavailable|not evaluable|not verified|per.PID.*(?:adapter|total)/i.test(i.text))).toBe(true)
  })

  it('G07b floor distance preserves a one GiB deficit as negative', () => {
    const c = candidate(); c.runs[0].minRamAvailBytes = m(3 * GiB)
    // Expected §12 input: planningSnapshot.ramFloorBytes, mmapCreditBytes.
    const data = { candidates: [c], machine: machine(), planningSnapshot: { ramFloorBytes: 4 * GiB, mmapCreditBytes: 0 } }
    const text = interpret(verdicts(data, 'max_quality')).filter(i => i.ruleId === 'I-4.3').map(i => i.text).join(' ')
    expect(text).toMatch(/(?:[-−]1(?:\.0+)?\s*GiB|1(?:\.0+)?\s*GiB\s+below)/i)
  })

  it('G08 unrelated fine-tunes are not declared equivalent quantizations', () => {
    const a = candidate('finetune-a'), b = candidate('finetune-b')
    // Expected ModelMeta.baseModelId/fineTuneId identify content beyond architecture.
    a.model = { ...a.model, baseModelId: 'org/base', fineTuneId: 'org/math' } as ModelMeta
    b.model = { ...b.model, quant: 'Q8_0', fileBytes: 9 * GiB, baseModelId: 'org/base', fineTuneId: 'org/legal' } as ModelMeta
    const notes = inspect([a, b]).filter(i => /quantization difference|prefer the smaller/i.test(i.text))
    expect(notes).toEqual([])
  })

  it('G09 a user-capped clean sweep is coverage, not a proven usable limit', () => {
    const c = candidate('capped', [8192])
    // Expected session input: stopReason: 'user-cap'; largest clean tested remains 8K.
    const data = { candidates: [c], machine: machine(), stopReason: 'user-cap' as const }
    const notes = interpret(verdicts(data, 'max_quality')).filter(i => i.ruleId === 'I-2.1' || i.ruleId === 'I-2.4')
    expect.soft(notes.every(i => i.severity !== 'warn' && !/usable limit|machine.s limit/i.test(i.text))).toBe(true)
    expect.soft(notes.map(i => i.text).join(' ')).toMatch(/higher contexts were not attempted/i)
  })

  const meanings = [
    ['I-1.2', /confirmed/i, /provisional/i],
    ['I-5.7', /infra|harness|evaluat/i, /invalid|quarantin/i],
    ['I-5.9', /relative signal/i, /not a leaderboard/i],
    ['I-7.1', /comparab|scope/i, /workload|machine/i],
    ['I-7.2', /decision trace/i, /constraint|tie.break|neutraliz/i],
    ['I-7.4', /quality/i, /speed|slower/i],
    ['I-7.6', /partial/i, /constraint|eligible/i],
  ] as const
  it.each(meanings)('G10 rule-id integrity: %s retains its v2 meaning', (id, first, second) => {
    const r = RULES.find(r => r.id === id)
    // Soft assertions inspect text even while interp-1 is still the active version.
    expect.soft(RULES_VERSION).toBe('interp-2')
    expect.soft(r?.text).toMatch(first)
    expect.soft(r?.text).toMatch(second)
  })

  it('G10b decision trace records scoring rung, eligible set and actual tie-break chain', () => {
    const rec = recommend([candidate('a'), candidate('b')], machine(), 'max_quality')
    const trace = traceOf(rec)
    expect(trace).toBeDefined()
    expect(trace!.scoringRung).toBe(8192)
    expect(trace!.eligibleSet).toEqual(expect.arrayContaining([{ configId: 'a', failingRuleIds: [] }, { configId: 'b', failingRuleIds: [] }]))
    expect(Array.isArray(trace!.neutralizations)).toBe(true)
    expect(trace!.tieBreakChain.length).toBeGreaterThan(0)
    expect(trace!.rulesVersion).toBe(rec.rulesVersion)
    expect(Object.keys(trace!.thresholdsUsed).length).toBeGreaterThan(0)
  })

  it('G11 alternative ranking choices carry their deciding rule verdict', () => {
    const a = candidate('a'), b = candidate('b'); b.runs[0].decodeTps = m(60)
    const rec = recommend([a, b], machine(), 'max_quality'), trace = traceOf(rec)
    expect(trace?.alternatives).toBeDefined()
    for (const [kind, id] of Object.entries(rec.alternatives)) {
      expect(trace!.alternatives[kind].configId).toBe(id)
      expect(trace!.alternatives[kind].ruleId).toMatch(/^I-\d+\.\d+$/)
    }
    expect(rec.alternatives.fastest).toBe('b')
  })

  it('G12 every recommendation, why-not and generation reason cites a rule', () => {
    const a = candidate('a'), b = candidate('b')
    a.genQuality = [gen('off', false, quality(10, 6)), gen('think', true, quality(10, 10))]
    const rec = recommend([a, b], machine(), 'max_quality')
    const texts = [...rec.reasons, ...(rec.whyNot ?? []).map(x => x.summary), ...(rec.best?.gen ? [rec.best.gen.reason] : [])]
    expect(rec.whyNot?.length).toBeGreaterThan(0)
    expect(texts.filter(text => !/\[I-\d+\.\d+\]/.test(text))).toEqual([])
  })

  it('G13 first-rung spill never suggests a nonexistent clean use-context action', () => {
    const c = candidate(); c.runs[0].peakSharedGpuBytes = m(GiB)
    const report = detectCliffs(c.runs, 16 * GiB)
    expect(report.spillFreeUpTo).toBeNull()
    const spill = inspect([c]).filter(i => i.ruleId === 'I-2.2')
    expect(spill.length).toBeGreaterThan(0)
    expect(spill.some(i => /^use-context(?:\s|$)/.test(i.action ?? ''))).toBe(false)
  })

  it('G14 category boundaries are inclusive and fewer than three items is insufficient', () => {
    const c = candidate(); c.quality = [...quality(3, 1).filter(r => r.category === 'reasoning'), ...quality(3, 2).filter(r => r.category === 'coding')]
    const flags = categoryFlags(c.quality)
    expect(flags.find(f => f.category === 'reasoning')?.flags).toContain('weak')
    expect(flags.find(f => f.category === 'coding')?.flags).toContain('coding warning')
    const cfg = { ...WORKLOADS.max_quality, promptSetIds: ['reasoning', 'coding'] as QualityCategory[] }
    const boundary = inspect([c]).filter(i => i.ruleId === 'I-5.3').map(i => i.text).join(' ')
    expect.soft(boundary).toMatch(/reasoning/i)
    const codingCandidate = candidate(); codingCandidate.quality = c.quality.filter(r => r.category === 'coding')
    const codingNotes = interpret(verdicts({ candidates: [codingCandidate], machine: machine() }, 'coding')).filter(i => i.ruleId === 'I-5.3')
    expect.soft(codingNotes.some(i => i.severity === 'warn' && /coding/i.test(i.text))).toBe(true)
    const thin = candidate(); thin.quality = quality(1, 0).filter(r => r.category === 'reasoning')
    // max_quality has minQuality=0, so the incomplete category must still be reported.
    const text = inspect([thin]).filter(i => i.ruleId === 'I-5.3').map(i => i.text).join(' ')
    expect.soft(text).toMatch(/insufficient coverage/i)
    expect(componentScores(c, machine(), cfg).components.quality.score).toBeGreaterThan(0)
  })

  it('G14b rep spread uses median rather than maximum', () => {
    const c = candidate(); c.runs[0].repDecodeTps = [10, 11.7]; c.runs[0].decodeTps = m(10.85)
    // 1.7 / 10.85 = 15.67% >15%; 1.7 / 11.7 = 14.53% incorrectly misses it.
    expect(inspect([c]).some(i => i.ruleId === 'I-6.1' && i.severity === 'warn')).toBe(true)
  })

  it('G15 a recovered decode dip states cause unverified, not likely contention', () => {
    const c = candidate('dip', [2048, 4096, 8192])
    c.runs[0].decodeTps = m(40); c.runs[1].decodeTps = m(20); c.runs[2].decodeTps = m(40)
    const text = inspect([c]).filter(i => i.ruleId === 'I-2.6').map(i => i.text).join(' ')
    expect.soft(text).toMatch(/cause unverified/i)
    expect.soft(text).not.toMatch(/likely contention/i)
  })
})
