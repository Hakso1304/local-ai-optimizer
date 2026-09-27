import { describe, expect, it } from 'vitest'
import type { ModelMeta, QualityResult } from '../../src/shared/bench-types'
import { BASELINE_GEN, genConfigsFor, genLabel, samplingFor, splitReasoning, summarizeGen, templateKwargsFor } from '../../src/core/benchmark/gen'
import { measuredQuality } from '../../src/core/scoring/components'

const base = { id: 'm', name: 'M', layers: 32 } as ModelMeta
const effort = { ...base, genKnobs: { supportsThinking: true, effortValues: ['low', 'medium', 'high', 'xhigh'] } }

describe('genConfigsFor', () => {
  it('non-thinking model: only the deterministic baseline', () => {
    expect(genConfigsFor(base)).toEqual([BASELINE_GEN])
  })
  it('effort levels: baseline, lowest, medium — T=1.0 without a model card; cap 3', () => {
    const g = genConfigsFor(effort)
    expect(g.map((x) => x.id)).toEqual(['off', 'think-low-t1', 'think-medium-t1'])
    expect(g[1]).toMatchObject({ thinking: true, effort: 'low', temperature: 1, source: 'default' })
  })
  it('model-card sampling; bare supportsThinking (no effort) → 2 configs; genSearch false / explicit override', () => {
    const card = { ...base, genKnobs: { supportsThinking: true, recommended: { temperature: 0.6, topP: 0.95, topK: 20 } } }
    expect(genConfigsFor(card)).toEqual([BASELINE_GEN, { id: 'think-t0.6', thinking: true, temperature: 0.6, topP: 0.95, topK: 20, source: 'model-card' }])
    expect(genConfigsFor({ ...base, supportsThinking: true }).map((x) => x.id)).toEqual(['off', 'think-t1'])
    expect(genConfigsFor(effort, { genSearch: false })).toEqual([BASELINE_GEN])
    expect(genConfigsFor(effort, { genConfigs: [{ id: '', thinking: true, effort: 'high', temperature: 0.7, source: 'template' }] }).map((x) => x.id)).toEqual(['think-high-t0.7'])
  })
  it('template kwargs use the template variable names; sampling uses llama-server names; label', () => {
    const [, low] = genConfigsFor(effort)
    expect(templateKwargsFor(effort, low)).toEqual({ enable_thinking: true, reasoning_effort: 'low' })
    expect(templateKwargsFor({ ...effort, genKnobs: { ...effort.genKnobs, effortKw: 'thinking_level' } }, low)).toEqual({ enable_thinking: true, thinking_level: 'low' })
    expect(templateKwargsFor(effort, BASELINE_GEN)).toEqual({ enable_thinking: false })
    expect(templateKwargsFor(base, BASELINE_GEN)).toBeUndefined()
    expect(samplingFor({ ...low, topP: 0.9 })).toEqual({ temperature: 1, top_p: 0.9 })
    expect(genLabel(low)).toBe('thinking on (effort low, T=1.0)')
    expect(genLabel(BASELINE_GEN)).toBe('thinking off (T=0)')
  })
})

describe('reasoning split + summary', () => {
  it('splits <think>, a prompt-opened think, and Gemma thought channels', () => {
    expect(splitReasoning('<think>abcd</think>xy')).toEqual({ reasoningChars: 19, answerChars: 2 })
    expect(splitReasoning('abcd</think>xy')).toEqual({ reasoningChars: 12, answerChars: 2 })
    expect(splitReasoning('<|channel>thought ab<channel|>Answer: 4').answerChars).toBe(9)
    expect(splitReasoning('plain')).toEqual({ reasoningChars: 0, answerChars: 5 })
  })
  it('summarizeGen: medians; stochastic flag; unknown tokens stay unavailable', () => {
    const row = (a: number | null, r: number | null, ms: number) => ({ testId: 't', category: 'coding', weight: 1, pass: true, score: 1, detail: '', answerTokens: a, reasoningTokens: r, totalMs: ms }) as QualityResult & { answerTokens: number | null }
    const [, low] = genConfigsFor(effort)
    const g = summarizeGen(low, [row(10, 90, 2000), row(20, 180, 4000), row(30, 270, 6000)], 3)
    expect(g).toMatchObject({ stochastic: true, samples: 3, answerTokens: { value: 20 }, reasoningTokens: { value: 180, kind: 'estimated' }, effectiveAnswerLatencyMs: { value: 4000 }, effectiveTps: { value: 5 } })
    expect(summarizeGen(BASELINE_GEN, [row(null, null, 1)], 1).answerTokens.kind).toBe('unavailable')
  })
})

describe('measuredQuality (uncertainty.ts unc-1: unique items are the unit)', () => {
  const rows = (cat: QualityResult['category'], n: number, pass: number): QualityResult[] =>
    Array.from({ length: n }, (_, i) => ({ testId: `${cat}${i}`, category: cat, weight: 1, pass: i < pass, score: 1, detail: '' }))
  const ok = (r: ReturnType<typeof measuredQuality>) => { if (!r.ok) throw new Error(r.reason); return r }
  it('value = weighted pass rate; the band narrows with more unique items and is never 0 wide at 100 %', () => {
    const small = ok(measuredQuality(rows('coding', 5, 5), ['coding'])).u
    const big = ok(measuredQuality(rows('coding', 60, 60), ['coding'])).u
    expect(small).toMatchObject({ q: 100, n: 5, method: 'wilson-item', version: 'unc-1' })
    expect(100 - small.lower).toBeGreaterThan(100 - big.lower)
    expect(big.lower).toBeLessThan(100)
    expect(measuredQuality(rows('coding', 5, 3), ['instruction']).ok).toBe(false)
  })
  it('repeated samples do NOT count as more independent items (Gap 1)', () => {
    const once = rows('reasoning', 4, 2), thrice = [...once, ...once, ...once].map((r, i) => ({ ...r, sample: Math.floor(i / 4) + 1 }))
    const a = ok(measuredQuality(once, ['reasoning'])), b = ok(measuredQuality(thrice, ['reasoning']))
    expect(b.u.n).toBe(4)
    expect(b.coverage).toMatchObject({ uniqueItems: 4, completions: 12 })
    expect(b.u.lower).toBeCloseTo(a.u.lower, 6)
  })
  it('an infrastructure error quarantines the result (I-5.7); truncation counts as a failure (I-5.8)', () => {
    const infra = [...rows('coding', 5, 5)]; (infra[0] as QualityResult & { evaluationStatus: string }).evaluationStatus = 'infra_error'
    expect(measuredQuality(infra, ['coding'])).toMatchObject({ ok: false, quarantined: true })
    const trunc = rows('coding', 5, 5).map((r, i) => (i === 0 ? { ...r, evaluationStatus: 'truncated' as const, outputTruncated: true } : r))
    expect(ok(measuredQuality(trunc, ['coding'])).u.q).toBe(80)
  })
})
