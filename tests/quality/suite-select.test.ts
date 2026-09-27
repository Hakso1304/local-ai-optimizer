import { describe, expect, it } from 'vitest'
import { buildQualityPrompts, defaultTestSet, evaluate, suiteFor } from '../../src/core/quality'
import { runChecker } from '../../src/core/quality/checkers'

describe('suiteFor (quality mode → suite)', () => {
  it("'quick' and unset stay on qb-1.1.0 (17 fixed tests, no seed)", () => {
    for (const mode of ['quick', undefined] as const) {
      const s = suiteFor(mode, 42)
      expect(s.suite).toBe('qb-1.1.0')
      expect(s.tests).toHaveLength(17)
      expect(s.suiteSeed).toBeNull()
      expect(s.tests).toEqual(defaultTestSet.tests)
    }
  })

  it("'thorough' resolves qb-2.0.0: 60 concrete items, generator version and seed recorded", () => {
    const s = suiteFor('thorough', 1234)
    expect(s.suite).toBe('qb-2.0.0')
    expect(s.tests).toHaveLength(60)
    expect(s.suiteSeed).toBe(1234)
    expect(s.generatorVersion).toBe('qbg-2.0.0')
    expect(s.tests.every((t) => t.checker && (t.prompt || t.template))).toBe(true)
    const byCat = Object.fromEntries(Object.keys(s.categoryWeights).map((c) => [c, s.tests.filter((t) => t.category === c).length]))
    expect(byCat).toEqual({ instruction: 12, reasoning: 12, coding: 12, structured: 8, extraction: 8, context: 8 })
  })

  it('same seed → identical suite; another seed changes only the generated items', () => {
    const a = suiteFor('thorough', 7), b = suiteFor('thorough', 7), c = suiteFor('thorough', 8)
    expect(a).toEqual(b)
    const gen = (s: typeof a) => s.tests.filter((t) => (t as { instanceSeed?: number }).instanceSeed !== undefined)
    expect(gen(a)).toHaveLength(13)
    expect(gen(a).map((t) => t.prompt)).not.toEqual(gen(c).map((t) => t.prompt))
    const stat = (s: typeof a) => s.tests.filter((t) => (t as { instanceSeed?: number }).instanceSeed === undefined)
    expect(stat(a)).toEqual(stat(c))
  })

  it('prompts and checkers come from the same resolved suite; every checker runs without throwing', () => {
    const s = suiteFor('thorough', 99)
    const prompts = buildQualityPrompts(s)
    expect(prompts.map((p) => p.testId)).toEqual(s.tests.map((t) => t.id))
    for (const t of s.tests) {
      if (t.checker.type === 'jsCode') continue // async sandbox path, covered in tests/quality-v2
      expect(() => runChecker(t.checker, '')).not.toThrow()
      expect(evaluate(t, '').pass).toBe(false)
    }
  })
})
