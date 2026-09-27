import { describe, expect, it } from 'vitest'
import { coverage, qualityUncertainty, pairedDifference, includesZero, nonInferior, categoryFlags, type UncertaintyRow, type PairedInterval } from '../../src/core/scoring/uncertainty'

const weights = { reasoning: 1, coding: 1 }
const row = (id: number, pass = true, extra: Partial<UncertaintyRow> = {}): UncertaintyRow => ({ testId: `item-${id}`, category: 'reasoning', weight: 1, pass, score: pass ? 1 : 0, detail: '', ...extra })
const ten = Array.from({ length: 10 }, (_, i) => row(i))
const pair = (a: UncertaintyRow[], b: UncertaintyRow[]): PairedInterval => {
  const result = pairedDifference(a, b, weights)
  if ('interval' in result) throw new Error(result.reason)
  return result
}

describe('coverage and evidence hygiene', () => {
  it('distinguishes items, seeds, samples, skills, invalid and truncated rows', () => {
    const rows = [row(0), row(0, true, { sample: 2 }), row(0, true, { generatorSeed: 0 }), row(1, false, { evaluationStatus: 'infra_error' }), row(2, false, { evaluationStatus: 'unrun' }), row(3, true, { outputTruncated: true }), row(4, true, { evaluationStatus: 'truncated' })]
    const c = coverage(rows)
    expect(c).toMatchObject({ uniqueItems: 6, uniqueSkills: 5, seeds: 1, completions: 5, validItems: 4, categoriesCovered: ['reasoning'], infraErrors: 1, truncated: 2 })
    expect(c.samplesPerItem['["item-0",null]']).toBe(2)
    expect(c.samplesPerItem['["item-2",null]']).toBe(0)
    expect(coverage([...rows].reverse())).toEqual(c)
  })
  it('quarantines infrastructure failures rather than scoring them', () => {
    const invalid = [...ten, row(20, false, { evaluationStatus: 'infra_error' })]
    expect(() => qualityUncertainty(invalid, weights)).toThrow('quarantined')
    expect(() => pairedDifference(invalid, ten, weights)).toThrow('quarantined')
    expect(() => categoryFlags(invalid)).toThrow('quarantined')
  })
  it('excludes unrun rows and forces truncated passes to failures', () => {
    expect(qualityUncertainty([row(1, true, { evaluationStatus: 'unrun' }), row(2, true, { outputTruncated: true })], weights)).toMatchObject({ q: 0, n: 1 })
  })
  it('rejects mixed generation scopes and inconsistent item metadata', () => {
    expect(() => qualityUncertainty([row(1), row(2, true, { genId: 'thinking' })], weights)).toThrow('genId')
    expect(() => qualityUncertainty([row(1), row(1, true, { skillId: 'other' })], weights)).toThrow('metadata')
  })
})

describe('quality uncertainty', () => {
  it('matches the Wilson known answer for ten independent successes', () => {
    const band = qualityUncertainty(ten, weights)
    expect(band).toMatchObject({ q: 100, n: 10, method: 'wilson-item', unit: 'item', version: 'unc-1', level: .95 })
    expect(band.lower).toBeCloseTo(72.2467, 3)
    expect(band.upper).toBeCloseTo(100, 10)
  })
  it('matches the balanced Wilson interval and failure symmetry', () => {
    const balanced = qualityUncertainty(ten.map((r, i) => ({ ...r, pass: i < 5 })), weights)
    expect(balanced.lower).toBeCloseTo(23.6593, 3)
    expect(balanced.upper).toBeCloseTo(76.3407, 3)
    expect(qualityUncertainty(ten.map(r => ({ ...r, pass: false })), weights).upper).toBeCloseTo(27.7533, 3)
  })
  it('repeats do not shrink bands, including fractional item means', () => {
    const rows = [...ten, row(0, false)]
    expect(qualityUncertainty([...ten, ...ten, ...ten], weights)).toEqual(qualityUncertainty(ten, weights))
    expect(qualityUncertainty([...rows, ...rows], weights)).toEqual(qualityUncertainty(rows, weights))
    expect(qualityUncertainty(rows, weights).q).toBeCloseTo(95)
  })
  it('weights categories and items rather than graded rows', () => {
    const rows = [row(1), row(2, false, { weight: 3 }), row(3, true, { category: 'coding' })]
    expect(qualityUncertainty(rows, { reasoning: 3, coding: 1 }).q).toBeCloseTo(43.75)
  })
  it('discounts uneven item weights instead of treating tiny weights as full evidence', () => {
    const uneven = ten.map((r, i) => ({ ...r, weight: i ? 1 : 1000 }))
    expect(qualityUncertainty(uneven, weights).lower).toBeLessThan(qualityUncertainty(ten, weights).lower)
  })
  it('correlated skill blocks widen the band compared with naive independent items', () => {
    const naive = Array.from({ length: 40 }, (_, i) => row(i, i < 20))
    const clustered = naive.map((r, i) => ({ ...r, skillId: `skill-${Math.floor(i / 10)}` }))
    const a = qualityUncertainty(naive, weights), b = qualityUncertainty(clustered, weights)
    expect(b).toMatchObject({ method: 'cluster-bootstrap', unit: 'skill', n: 4 })
    expect(b.upper - b.lower).toBeGreaterThan(a.upper - a.lower)
    expect(qualityUncertainty([...clustered].reverse(), weights)).toEqual(b)
    expect(qualityUncertainty([...clustered, ...clustered], weights)).toEqual(b)
  })
  it('uses a conservative full band with only one multi-item skill', () => {
    expect(qualityUncertainty(ten.map(r => ({ ...r, skillId: 'one' })), weights)).toMatchObject({ lower: 0, upper: 100, n: 1, unit: 'skill' })
  })
  it('keeps extreme endpoints bounded, including very large weights', () => {
    for (const pass of [true, false]) {
      const result = qualityUncertainty(ten.map(r => ({ ...r, pass, weight: Number.MAX_VALUE })), { reasoning: Number.MAX_VALUE })
      expect(result.lower).toBeGreaterThanOrEqual(0)
      expect(result.upper).toBeLessThanOrEqual(100)
      expect(result.q).toBe(pass ? 100 : 0)
    }
  })
  it('rejects missing evidence and invalid weights without inventing zero quality', () => {
    expect(() => qualityUncertainty([], weights)).toThrow('No positively weighted')
    expect(() => qualityUncertainty(ten, {})).toThrow('No positively weighted')
    for (const w of [-1, NaN, Infinity]) expect(() => qualityUncertainty(ten, { reasoning: w })).toThrow()
  })
})

describe('paired differences and decisions', () => {
  it('identical evidence has exactly zero paired difference and band', () => {
    const rows = ten.map((r, i) => ({ ...r, pass: i < 6 }))
    expect(pair(rows, rows)).toEqual({ diff: 0, lower: 0, upper: 0, sharedItems: 10, method: 'cluster-bootstrap' })
  })
  it('uses only matching item AND seed, ignoring unmatched successes', () => {
    const a = [...ten, row(20, true, { generatorSeed: 1 })]
    const b = [...ten, row(20, false, { generatorSeed: 2 })]
    expect(pair(a, b)).toMatchObject({ diff: 0, lower: 0, upper: 0, sharedItems: 10 })
  })
  it('returns a null interval with reason for fewer than five shared items', () => {
    expect(pairedDifference(ten.slice(0, 4), ten, weights)).toEqual({ interval: null, reason: 'insufficient shared items', sharedItems: 4 })
    expect(pairedDifference([], ten, weights)).toMatchObject({ interval: null, sharedItems: 0 })
    expect(pairedDifference(ten, ten, {})).toMatchObject({ interval: null, sharedItems: 0 })
    expect(() => pairedDifference([], [], { reasoning: NaN })).toThrow()
    expect(pairedDifference(ten, ten.map(r => ({ ...r, generatorSeed: 0 })), weights)).toMatchObject({ interval: null, sharedItems: 0 })
  })
  it('does not infer variation from a single shared skill', () => {
    const rows = ten.map(r => ({ ...r, skillId: 'one' }))
    expect(pairedDifference(rows, rows, weights)).toMatchObject({ interval: null, reason: 'insufficient shared skills' })
  })
  it('is deterministic, clustered, bounded and invariant to repeats and ordering', () => {
    const a = ten.map((r, i) => ({ ...r, skillId: `skill-${Math.floor(i / 2)}`, pass: i < 6 }))
    const b = a.map(r => ({ ...r, pass: false }))
    const result = pair(a, b)
    expect(result.diff).toBeCloseTo(60)
    expect(result.lower).toBeCloseTo(20)
    expect(result.upper).toBeCloseTo(100)
    expect(pair([...a].reverse(), [...b].reverse())).toEqual(result)
    expect(pair([...a, ...a], [...b, ...b])).toEqual(result)
    expect(pair(b, a).diff).toBeCloseTo(-60)
    expect(pair(ten, ten.map(r => ({ ...r, pass: false })))).toMatchObject({ diff: 100, lower: 100, upper: 100 })
  })
  it('rejects incompatible paired item metadata', () => {
    expect(() => pair(ten, ten.map(r => ({ ...r, weight: 2 })))).toThrow('metadata')
  })
  it('uses inclusive zero and noninferiority boundaries in percentage points', () => {
    expect(includesZero({ lower: 0, upper: 4 })).toBe(true)
    expect(includesZero({ lower: -4, upper: 0 })).toBe(true)
    expect(includesZero({ lower: 1, upper: 4 })).toBe(false)
    expect(nonInferior({ lower: -3, upper: 1 }, 3)).toBe(true)
    expect(nonInferior({ lower: -3.01, upper: 1 }, 3)).toBe(false)
    expect(() => nonInferior({ lower: 0, upper: 1 }, -1)).toThrow()
    expect(() => includesZero({ lower: NaN, upper: 1 })).toThrow()
    expect(() => includesZero({ lower: 2, upper: 1 })).toThrow()
  })
})

describe('category flags', () => {
  it('uses inclusive thirds and only unique valid items as coverage', () => {
    const rows = [row(0), row(1, false), row(2, false), row(3, true, { category: 'coding' }), row(4, true, { category: 'coding' }), row(5, false, { category: 'coding' })]
    const flags = categoryFlags(rows)
    expect(flags.find(f => f.category === 'reasoning')?.flags).toEqual(['weak'])
    expect(flags.find(f => f.category === 'coding')?.flags).toEqual(['coding warning'])
    expect(flags.find(f => f.category === 'context')).toMatchObject({ rate: null, validItems: 0, flags: ['insufficient coverage'] })
    expect(categoryFlags([row(0, false), row(0, false), row(0, false)])[1]).toMatchObject({ validItems: 1, flags: ['insufficient coverage'] })
  })
})
