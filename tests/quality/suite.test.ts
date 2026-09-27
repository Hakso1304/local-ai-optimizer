import { describe, expect, it } from 'vitest'
import {
  buildQualityPrompts, categoryPassRates, defaultTestSet, evaluate, generateFiller, qualityScore, type QualityResult
} from '../../src/core/quality'

// Canned model outputs per test: realistic good answers (incl. fences/markdown) and plausible wrong ones.
const GOOD: Record<string, string> = {
  'IF-01': 'BANANA',
  'IF-02': 'red,green,blue',
  'IF-03': 'The ocean is vast today.',
  'RS-01': '27 pens is 9 groups of 3.\n9 × $4 = $36.\nAnswer: 36',
  'RS-02': 'Dave > Alice > Bob > Carol.\n\n**Answer:** Carol',
  'RS-03': '2^3 = 8 ≡ 1 (mod 7), so 2^20 = (2^3)^6 · 2^2 ≡ 4.\nFinal answer: `4`',
  'RS-04': '100 mod 7 = 2, Wednesday + 2 = Friday.\nanswer: Friday.',
  'CD-01': '```javascript\nfunction isPalindrome(s) {\n  const t = s.toLowerCase().replace(/[^a-z0-9]/g, "");\n  return t === [...t].reverse().join("");\n}\n```',
  'CD-02': '```js\nfunction fib(n) {\n  let a = 0n, b = 1n;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}\n```',
  'CD-03': 'function groupBy(arr, key) {\n  const out = {};\n  for (const it of arr) (out[it[key]] ??= []).push(it);\n  return out;\n}',
  'SO-01': '```json\n{"title": "Dune", "year": 1965, "tags": ["sci-fi", "classic"]}\n```',
  'SO-02': '[{"name":"Tom","age":31},{"name":"Ana","age":27},{"name":"Lee","age":45}]',
  'EX-01': 'INV-20931;1,428.00 EUR',
  'EX-02': 'ops@acme.io\nbill.pay@acme.co.uk\nold_admin@acme.io',
  'CR-10': 'HELIOTROPE-5',
  'CR-50': 'The secret project name is HELIOTROPE-5.',
  'CR-90': '**HELIOTROPE-5**'
}
const BAD: Record<string, string> = {
  'IF-01': 'Banana',
  'IF-02': 'Red, Green, Blue.',
  'IF-03': 'The ocean is deep.',
  'RS-01': '36', // bare answer, no Answer: line → format violation
  'RS-02': 'Answer: Carol\nWait, let me recheck.\nAnswer: Dave', // last Answer line wins
  'RS-03': 'Answer: 2',
  'RS-04': 'It is Friday or Thursday.\nAnswer: Thursday',
  'CD-01': 'function isPalindrome(s) { return s === s.split("").reverse().join(""); }',
  'CD-02': 'function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }',
  'CD-03': 'function groupBy(arr, key) { return arr; }',
  'SO-01': '{"title": "Dune", "year": "1965", "tags": ["sci-fi"]}',
  'SO-02': 'Here you go: [{"name":"Tom","age":31}]',
  'EX-01': 'INV-20931;1,200.00 EUR',
  'EX-02': 'ops@acme.io\nbill.pay@acme.co.uk',
  'CR-10': 'I could not find a project name.',
  'CR-50': 'HELIOTROPE',
  'CR-90': 'HELIOTROPE-9'
}

describe('quality suite v1', () => {
  const tests = defaultTestSet.tests

  it('is well-formed: unique ids, known categories, weights sum to 1', () => {
    expect(new Set(tests.map((t) => t.id)).size).toBe(tests.length)
    const cats = Object.keys(defaultTestSet.categoryWeights)
    for (const t of tests) expect(cats, t.id).toContain(t.category)
    for (const c of cats) expect(tests.some((t) => t.category === c), c).toBe(true)
    expect(Object.values(defaultTestSet.categoryWeights).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9)
    expect(Object.keys(GOOD).sort()).toEqual(tests.map((t) => t.id).sort())
  })

  it.each(tests.map((t) => [t.id, t] as const))('%s: good output passes, bad output fails', (id, t) => {
    const good = evaluate(t, GOOD[id])
    expect(good.pass, good.detail).toBe(true)
    expect(good.score).toBe(1)
    const bad = evaluate(t, BAD[id])
    expect(bad.pass, bad.detail).toBe(false)
    expect(bad.score).toBeLessThan(1)
  })

  it('CD-02 bad (naive recursion, Number) fails on fib(90) rather than hanging', () => {
    const r = evaluate(tests.find((t) => t.id === 'CD-02')!, BAD['CD-02'])
    expect(r.detail).toMatch(/timed out|0\/4|\d\/4/)
  })

  it('buildQualityPrompts fills needle templates deterministically at the requested depth', () => {
    const a = buildQualityPrompts()
    expect(a).toEqual(buildQualityPrompts())
    expect(a).toHaveLength(tests.length)
    const pos = ['CR-10', 'CR-50', 'CR-90'].map((id) => {
      const c = a.find((p) => p.testId === id)!.messages[0].content
      expect(c.split('HELIOTROPE-5').length - 1).toBe(1)
      const doc = c.slice(c.indexOf('<document>'), c.indexOf('</document>'))
      return doc.indexOf('HELIOTROPE-5') / doc.length
    })
    expect(pos[0]).toBeGreaterThan(0.05); expect(pos[0]).toBeLessThan(0.15)
    expect(pos[1]).toBeGreaterThan(0.45); expect(pos[1]).toBeLessThan(0.55)
    expect(pos[2]).toBeGreaterThan(0.85); expect(pos[2]).toBeLessThan(0.95)
    for (const p of a) expect(p).toMatchObject({ temperature: 0, seed: 1 })
  })

  it('filler length tracks fillerTokens and contains no digits', () => {
    const f = generateFiller(3000, 42).join(' ')
    expect(f.length / 4).toBeGreaterThanOrEqual(3000)
    expect(f.length / 4).toBeLessThan(3100)
    expect(/\d/.test(f)).toBe(false)
    expect(generateFiller(500, 42)).not.toEqual(generateFiller(500, 43))
    const small = buildQualityPrompts(defaultTestSet, { fillerTokens: 500 }).find((p) => p.testId === 'CR-50')!
    expect(small.messages[0].content.length).toBeLessThan(2600)
  })

  it('thinking mode boosts only reasoning/coding maxTokens ×4', () => {
    const base = buildQualityPrompts()
    const think = buildQualityPrompts(defaultTestSet, { thinking: true })
    base.forEach((b, i) => {
      const k = b.category === 'reasoning' || b.category === 'coding' ? 4 : 1
      expect(think[i].maxTokens).toBe(b.maxTokens * k)
    })
  })
})

describe('qualityScore', () => {
  const r = (category: QualityResult['category'], pass: boolean, weight = 1): QualityResult => ({ testId: 'x', category, weight, pass, score: pass ? 1 : 0, detail: '' })

  it('100 for all good, 0 for all bad over the real suite', () => {
    const t = defaultTestSet.tests
    expect(qualityScore(t.map((x) => evaluate(x, GOOD[x.id])))).toBeCloseTo(100, 9)
    expect(qualityScore(t.map((x) => evaluate(x, BAD[x.id])))).toBe(0)
  })
  it('weights categories, not test counts', () => {
    // 4 passing reasoning tests vs 1 failing context test: 0.25/(0.25+0.1) = 71.43, not 80.
    const s = qualityScore([r('reasoning', true), r('reasoning', true), r('reasoning', true), r('reasoning', true), r('context', false)])
    expect(s).toBeCloseTo((100 * 0.25) / 0.35, 6)
  })
  it('per-test weight applies within a category', () => {
    expect(categoryPassRates([r('coding', true, 3), r('coding', false, 1)]).coding).toBeCloseTo(0.75, 9)
  })
  it('null (not NaN / 0) when there is nothing to score; zero-weight tests ignored', () => {
    expect(qualityScore([])).toBeNull()
    expect(qualityScore([r('coding', true, 0)])).toBeNull()
  })
  it('is order independent', () => {
    const xs = [r('coding', true), r('context', false), r('instruction', true), r('coding', false)]
    expect(qualityScore(xs)).toBe(qualityScore([...xs].reverse()))
  })
})
