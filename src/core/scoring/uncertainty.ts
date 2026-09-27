/**
 * unc-1: uncertainty HEURISTICS, not validated 95% confidence guarantees.
 * Evidence identity is (testId, generatorSeed); missing seed is distinct from 0.
 * Repeated completions collapse to item mean p_i (binary pass, truncated => 0).
 * Within category c, b_i = itemWeight_i / sum(itemWeight in c); then
 * a_i = categoryWeight_c * b_i / sum(categoryWeight over observed categories).
 * Q = 100 sum(a_i p_i). Missing categories are not fabricated as failures.
 * Wilson: p=Q/100, effective item n=1/sum(a_i^2), z=1.959963984540054;
 * center=(p+z²/(2n))/(1+z²/n), half=z sqrt(p(1-p)/n+z²/(4n²))/(1+z²/n).
 * Kish effective n discounts unequal weights, never exceeds unique item count.
 * With multi-item skills, resample whole skills uniformly with replacement,
 * preserving original a_i within blocks; each draw uses sum(a_i p_i)/sum(a_i).
 * Report linearly interpolated .025/.975 quantiles of 8192 draws, fixed PRNG seed.
 * This ratio bootstrap allows category composition to vary across draws.
 * Paired bootstrap uses d_i=p_Ai-p_Bi on the intersection ONLY, recomputing a_i
 * there; resample shared skill blocks and report 100 sum(a_i d_i)/sum(a_i).
 * Independence is assumed across items for Wilson, across skills for bootstrap,
 * NEVER across repeats. Correlated skills, suite selection, and small samples
 * violate these assumptions; all-equal bootstrap samples have degenerate bands.
 * One skill cannot estimate between-skill variation: quality band is [0,100],
 * paired inference is unavailable. n is actual independent-unit count, not Kish n.
 * Callers must partition by suite/candidate/context/gen configuration and enforce
 * guide §7 comparability. Mixed genId values in one input and infra_error rows
 * are rejected, not silently scored. Legacy missing status means valid.
 */
import type { QualityCategory, QualityResult } from '../quality'

export interface UncertaintyRow extends QualityResult {
  skillId?: string
  generatorSeed?: number | string
  sample?: number
  genId?: string
  evaluationStatus?: 'valid' | 'infra_error' | 'unrun' | 'truncated'
  outputTruncated?: boolean
}
export type CategoryWeights = Partial<Record<QualityCategory, number>>
export interface Interval { lower: number; upper: number }
export interface QualityInterval extends Interval {
  q: number
  method: 'wilson-item' | 'cluster-bootstrap'
  version: 'unc-1'
  unit: 'item' | 'skill'
  level: 0.95
  n: number
}
export interface PairedInterval extends Interval {
  diff: number
  sharedItems: number
  method: 'cluster-bootstrap'
}
export interface UnavailablePair {
  interval: null
  reason: 'insufficient shared items' | 'insufficient shared skills'
  sharedItems: number
}
const categories: QualityCategory[] = ['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context']
const compare = (a: string, b: string): number => a < b ? -1 : a > b ? 1 : 0
const key = (r: UncertaintyRow): string => JSON.stringify([r.testId, r.generatorSeed ?? null])
const skill = (r: UncertaintyRow): string => r.skillId ?? r.testId
const valid = (r: UncertaintyRow): boolean => r.evaluationStatus === undefined || r.evaluationStatus === 'valid' || r.evaluationStatus === 'truncated'
const truncated = (r: UncertaintyRow): boolean => r.evaluationStatus === 'truncated' || r.outputTruncated === true
const clamp = (x: number, lo = 0, hi = 100): number => Math.max(lo, Math.min(hi, x))

export function coverage(rows: readonly UncertaintyRow[]) {
  const samples = new Map<string, number>()
  const validKeys = new Set<string>()
  for (const r of rows) {
    const id = key(r)
    if (!samples.has(id)) samples.set(id, 0)
    if (valid(r)) { samples.set(id, samples.get(id)! + 1); validKeys.add(id) }
  }
  return {
    uniqueItems: samples.size,
    uniqueSkills: new Set(rows.map(skill)).size,
    seeds: new Set(rows.filter(r => r.generatorSeed !== undefined).map(r => JSON.stringify(r.generatorSeed))).size,
    completions: [...samples.values()].reduce((a, b) => a + b, 0),
    samplesPerItem: Object.fromEntries([...samples].sort(([a], [b]) => compare(a, b))),
    validItems: validKeys.size,
    categoriesCovered: categories.filter(c => rows.some(r => r.category === c && valid(r))),
    infraErrors: rows.filter(r => r.evaluationStatus === 'infra_error').length,
    truncated: rows.filter(r => valid(r) && truncated(r)).length,
  }
}

interface Item { id: string; skill: string; category: QualityCategory; weight: number; p: number; mass: number }
function items(rows: readonly UncertaintyRow[]): Item[] {
  if (rows.some(r => r.evaluationStatus === 'infra_error')) throw new Error('Quality quarantined: infra_error')
  if (new Set(rows.map(r => r.genId ?? null)).size > 1) throw new Error('Partition quality rows by genId')
  const groups = new Map<string, UncertaintyRow[]>()
  for (const r of rows) {
    if (!categories.includes(r.category) || !r.testId || !skill(r) || !Number.isFinite(r.weight) || r.weight < 0 || typeof r.pass !== 'boolean') throw new TypeError('Invalid quality row')
    if (r.generatorSeed !== undefined && typeof r.generatorSeed !== 'string' && !Number.isFinite(r.generatorSeed)) throw new TypeError('Invalid generator seed')
    if (r.evaluationStatus !== undefined && !['valid', 'unrun', 'truncated'].includes(r.evaluationStatus)) throw new TypeError('Invalid evaluation status')
    if (!valid(r)) continue
    const id = key(r)
    const group = groups.get(id) ?? []
    group.push(r); groups.set(id, group)
  }
  return [...groups].sort(([a], [b]) => compare(a, b)).map(([id, rs]) => {
    const r = rs[0]
    if (rs.some(s => s.category !== r.category || skill(s) !== skill(r) || s.weight !== r.weight)) throw new Error('Conflicting item metadata')
    return { id, skill: skill(r), category: r.category, weight: r.weight, p: rs.filter(s => s.pass && !truncated(s)).length / rs.length, mass: 0 }
  })
}
function weighted(input: Item[], weights: CategoryWeights, allowEmpty = false): Item[] {
  for (const w of Object.values(weights)) if (!Number.isFinite(w) || w < 0) throw new TypeError('Weights must be finite and nonnegative')
  const active = input.filter(i => i.weight > 0 && (weights[i.category] ?? 0) > 0)
  if (!active.length) {
    if (allowEmpty) return []
    throw new RangeError('No positively weighted valid items')
  }
  // Normalize by maxima first to avoid overflow with large but finite weights.
  const present = categories.filter(c => active.some(i => i.category === c))
  const maxCategory = Math.max(...present.map(c => weights[c]!))
  const totalCategory = present.reduce((s, c) => s + weights[c]! / maxCategory, 0)
  const result: Item[] = []
  for (const c of present) {
    const members = active.filter(i => i.category === c)
    const maxItem = Math.max(...members.map(i => i.weight))
    const totalItem = members.reduce((s, i) => s + i.weight / maxItem, 0)
    for (const i of members) result.push({ ...i, mass: (weights[c]! / maxCategory / totalCategory) * (i.weight / maxItem / totalItem) })
  }
  // Weights below floating-point representability contribute no evidence.
  return result.filter(i => i.mass > 0).sort((a, b) => compare(a.id, b.id))
}
function blocks(input: Item[]): Item[][] {
  const groups = new Map<string, Item[]>()
  for (const i of input) { const g = groups.get(i.skill) ?? []; g.push(i); groups.set(i.skill, g) }
  return [...groups].sort(([a], [b]) => compare(a, b)).map(([, g]) => g)
}
function mean(input: Item[]): number {
  return input.reduce((s, i) => s + i.mass * i.p, 0) / input.reduce((s, i) => s + i.mass, 0)
}
function bootstrap(groups: Item[][]): Interval {
  let seed = 0x756e6331
  const random = () => {
    seed = (seed + 0x6d2b79f5) >>> 0
    let t = Math.imul(seed ^ (seed >>> 15), seed | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const summaries = groups.map(g => ({ mass: g.reduce((s, i) => s + i.mass, 0), sum: g.reduce((s, i) => s + i.mass * i.p, 0) }))
  const draws: number[] = []
  for (let b = 0; b < 8192; b++) {
    let sum = 0, mass = 0
    for (let j = 0; j < groups.length; j++) {
      const g = summaries[Math.floor(random() * groups.length)]
      sum += g.sum; mass += g.mass
    }
    draws.push(100 * sum / mass)
  }
  draws.sort((a, b) => a - b)
  const quantile = (p: number) => {
    const index = p * (draws.length - 1), lo = Math.floor(index), f = index - lo
    return draws[lo] * (1 - f) + draws[Math.ceil(index)] * f
  }
  return { lower: quantile(.025), upper: quantile(.975) }
}

export function qualityUncertainty(rows: readonly UncertaintyRow[], weights: CategoryWeights): QualityInterval {
  const data = weighted(items(rows), weights), groups = blocks(data)
  const q = clamp(100 * mean(data))
  if (groups.length < data.length) {
    const band = groups.length < 2 ? { lower: 0, upper: 100 } : bootstrap(groups)
    return { q, lower: clamp(band.lower), upper: clamp(band.upper), method: 'cluster-bootstrap', version: 'unc-1', unit: 'skill', level: .95, n: groups.length }
  }
  const n = Math.min(data.length, 1 / data.reduce((s, i) => s + i.mass ** 2, 0))
  const p = q / 100, z = 1.959963984540054, z2 = z * z, denominator = 1 + z2 / n
  const center = (p + z2 / (2 * n)) / denominator
  const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
  return { q, lower: clamp(100 * (center - half)), upper: clamp(100 * (center + half)), method: 'wilson-item', version: 'unc-1', unit: 'item', level: .95, n: data.length }
}

export function pairedDifference(rowsA: readonly UncertaintyRow[], rowsB: readonly UncertaintyRow[], weights: CategoryWeights): PairedInterval | UnavailablePair {
  const a = items(rowsA), b = new Map(items(rowsB).map(i => [i.id, i]))
  const shared: Item[] = []
  for (const i of a) {
    const j = b.get(i.id)
    if (!j) continue
    if (i.category !== j.category || i.skill !== j.skill || i.weight !== j.weight) throw new Error('Conflicting paired item metadata')
    shared.push({ ...i, p: i.p - j.p })
  }
  // Only positively weighted matched items support the comparison.
  const data = weighted(shared, weights, true)
  const sharedItems = data.length
  if (sharedItems < 5) return { interval: null, reason: 'insufficient shared items', sharedItems }
  const groups = blocks(data)
  if (groups.length < 2) return { interval: null, reason: 'insufficient shared skills', sharedItems }
  const band = bootstrap(groups)
  return { diff: clamp(100 * mean(data), -100, 100), lower: clamp(band.lower, -100, 100), upper: clamp(band.upper, -100, 100), sharedItems, method: 'cluster-bootstrap' }
}
function checkInterval(interval: Interval): void {
  if (!Number.isFinite(interval.lower) || !Number.isFinite(interval.upper) || interval.lower > interval.upper) throw new TypeError('Invalid interval')
}
export function includesZero(interval: Interval): boolean {
  checkInterval(interval)
  return interval.lower <= 0 && interval.upper >= 0
}
/** A-B is non-inferior at a caller-declared percentage-point margin, inclusively. */
export function nonInferior(interval: Interval, margin: number): boolean {
  checkInterval(interval)
  if (!Number.isFinite(margin) || margin < 0) throw new TypeError('Margin must be finite and nonnegative')
  return interval.lower >= -margin
}

export function categoryFlags(rows: readonly UncertaintyRow[], options = { weakAtMost: 1 / 3, codingAtMost: 2 / 3, minItems: 3 }) {
  if (![options.weakAtMost, options.codingAtMost].every(x => Number.isFinite(x) && x >= 0 && x <= 1) || !Number.isInteger(options.minItems) || options.minItems < 1) throw new TypeError('Invalid category thresholds')
  const data = items(rows)
  return categories.map(category => {
    const members = data.filter(i => i.category === category && i.weight > 0)
    const rate = members.length ? mean(weighted(members, { [category]: 1 })) : null
    const flags: ('insufficient coverage' | 'weak' | 'coding warning')[] = []
    if (members.length < options.minItems) flags.push('insufficient coverage')
    else {
      if (rate! <= options.weakAtMost) flags.push('weak')
      if (category === 'coding' && rate! <= options.codingAtMost) flags.push('coding warning')
    }
    return { category, validItems: members.length, rate, flags }
  })
}
