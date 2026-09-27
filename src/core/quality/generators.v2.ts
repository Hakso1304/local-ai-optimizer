import type { QualityCategory, QualityTest, QualityTestSet } from './index'
import type { CheckerSpec } from './checkers'

/** Additive metadata: resolved items still satisfy the existing QualityTest interface. */
export interface V2Metadata {
  difficulty: 1 | 2 | 3
  skill: string
  variant: number
}
export type V2Test = QualityTest & V2Metadata & { instanceSeed?: number; generatorVersion?: string }
export interface GeneratorParams { variant?: 1 | 2 | 3 }
export type GeneratorName = 'arithmetic' | 'wordProblem' | 'extraction' | 'schemaRecord' | 'codeConstants'
export interface GeneratorEntry extends V2Metadata {
  id: string
  category: QualityCategory
  weight: number
  maxTokens: number
  generator: GeneratorName
  params: GeneratorParams
}
export type V2Entry = V2Test | GeneratorEntry
export interface V2Manifest extends Omit<QualityTestSet, 'tests'> { tests: V2Entry[] }
export const GENERATOR_VERSION = 'qbg-2.0.0'

function seed32(seed: number): number {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('seed must be a uint32')
  return seed
}
// Mulberry32, integer state only. No global state, wall clock, locale, filesystem or model identity.
function random(seed: number) {
  let a = seed32(seed)
  return (lo: number, hi: number) => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), a | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return lo + Math.floor((((t ^ (t >>> 14)) >>> 0) / 4294967296) * (hi - lo + 1))
  }
}
function variant(params: GeneratorParams): 1 | 2 | 3 {
  const v = params.variant ?? 1
  if (v !== 1 && v !== 2 && v !== 3) throw new Error('variant must be 1, 2 or 3')
  return v
}
const answerLine = 'You may explain your reasoning. End with a final line exactly in the form Answer: X, where X is the integer answer.'
const integerAnswer = (n: number): CheckerSpec => ({ type: 'finalAnswer', inner: { type: 'regex', pattern: `^${n}$` } })
function item(name: GeneratorName, seed: number, v: 1 | 2 | 3, category: QualityCategory, prompt: string, checker: CheckerSpec): V2Test {
  return { id: `GEN-${name}-${v}-${seed}`, category, weight: 1, maxTokens: category === 'coding' ? 640 : category === 'reasoning' ? 768 : 256,
    difficulty: v, skill: name, variant: v, prompt, checker, instanceSeed: seed, generatorVersion: GENERATOR_VERSION }
}

export function arithmetic(seed: number, params: GeneratorParams = {}): V2Test {
  const r = random(seed), v = variant(params)
  const a = r(11, 79), b = r(3, 19), c = r(2, 9)
  const expression = v === 1 ? `${a} + ${b} * ${c}` : v === 2 ? `(${a} - ${b}) * ${c}` : `(${a * c} / ${c}) + (${b} * ${c})`
  const expected = v === 2 ? (a - b) * c : a + b * c
  return item('arithmetic', seed, v, 'reasoning', `Compute ${expression} using standard arithmetic precedence. ${answerLine}`, integerAnswer(expected))
}

export function wordProblem(seed: number, params: GeneratorParams = {}): V2Test {
  const r = random(seed), v = variant(params)
  const packs = r(4, 15), each = r(3, 12), removed = r(1, each - 1), price = r(2, 9)
  let question: string, expected: number
  if (v === 1) {
    question = `A library receives ${packs} boxes with ${each} notebooks each. It gives away ${removed} notebooks in total. How many notebooks remain?`
    expected = packs * each - removed
  } else if (v === 2) {
    question = `A club buys ${packs} packs of ${each} tickets. Each ticket costs ${price} dollars. The seller gives one discount of ${removed} dollars on the entire purchase. What is the final cost in whole dollars?`
    expected = packs * each * price - removed
  } else {
    const people = each, total = packs * each + removed
    question = `There are ${total} tokens. Reserve ${removed} tokens first, then divide the remaining tokens equally among ${people} people. How many tokens does each person receive?`
    expected = packs
  }
  return item('wordProblem', seed, v, 'reasoning', `${question} ${answerLine}`, integerAnswer(expected))
}

export function extraction(seed: number, params: GeneratorParams = {}): V2Test {
  const r = random(seed), v = variant(params)
  const names = ['Aster', 'Birch', 'Cedar', 'Dahlia', 'Elm', 'Fir']
  const targetIndex = r(0, names.length - 1), target = names[targetIndex], other = names[(targetIndex + 1) % names.length]
  const year = r(2025, 2034), month = r(1, 12), day = r(2, 20)
  const date = (d: number) => `${year}-${String(month).padStart(2, '0')}-${String(d).padStart(2, '0')}`
  const ref = `EV-${r(10000, 99999)}`, latest = date(day + 2)
  const lines = [
    `Notice ${ref}: ${target} was initially scheduled for ${date(day)}.`,
    `${other} is confirmed for ${date(day + 3)}; this is a different event.`,
    `A proposed date ${date(day + 1)} for ${target} was rejected.`,
    `Final approved update: ${target} is confirmed for ${latest}.`
  ]
  // Different document order, same explicit status semantics; never infer latest from line position.
  if (v === 2) lines.reverse()
  if (v === 3) lines.push(`Archived reminder: the old ${date(day)} date for ${target} is obsolete.`)
  const expected = { event: target, date: latest, reference: ref }
  return item('extraction', seed, v, 'extraction', `Read these notices. Return a JSON object with event, date (YYYY-MM-DD), and reference for ${target}, using only its final approved date and its notice reference. Do not use rejected or obsolete dates.\n${lines.join('\n')}`, { type: 'jsonEqual', expected })
}

export function schemaRecord(seed: number, params: GeneratorParams = {}): V2Test {
  const r = random(seed), v = variant(params)
  const suffix = () => r(0, 0xffffff).toString(16).padStart(6, '0')
  const titleKey = `label_${suffix()}`, countKey = `count_${suffix()}`, stateKey = `ready_${suffix()}`
  const label = `unit-${r(1000, 9999)}`, count = r(2, 37), ready = v !== 2
  return item('schemaRecord', seed, v, 'structured',
    `Produce a JSON object matching this record. The string field "${titleKey}" must be "${label}". The integer field "${countKey}" must be ${count}. The boolean field "${stateKey}" must be ${ready}. All three fields are required. Additional fields are allowed; do not wrap numbers or booleans in quotes. Output JSON only.`,
    { type: 'jsonSchema', schema: { type: 'object', required: [titleKey, countKey, stateKey], properties: {
      [titleKey]: { type: 'string', enum: [label] }, [countKey]: { type: 'integer', enum: [count] }, [stateKey]: { type: 'boolean', enum: [ready] }
    } } })
}

export function codeConstants(seed: number, params: GeneratorParams = {}): V2Test {
  const r = random(seed), v = variant(params), a = r(2, 9), b = r(3, 17)
  const xs = [-100, -b, -1, 0, 1, b, 100, r(18, 70)]
  let contract: string, expected: (x: number) => number
  if (v === 1) {
    contract = `return ${a} * x + ${b}`; expected = (x) => a * x + b
  } else if (v === 2) {
    contract = `return x clipped to the inclusive interval [-${a}, ${b}]`; expected = (x) => Math.max(-a, Math.min(b, x))
  } else {
    contract = `return the unique integer r with 0 <= r < ${b} such that x - r is divisible by ${b} (nonnegative mathematical modulo, including negative x)`
    expected = (x) => ((x % b) + b) % b
  }
  return item('codeConstants', seed, v, 'coding', `Write a JavaScript function transform(x). For any integer x from -100 through 100, ${contract}. Return a number, not a string. Output only code.`,
    { type: 'jsCode', timeoutMs: 1000, cases: xs.map((x) => ({ expr: `transform(${x})`, expected: expected(x) })) })
}

export const GENERATORS_V2 = { arithmetic, wordProblem, extraction, schemaRecord, codeConstants } as const

/** Stable per-id seed: independent of suite order and identical across competing models/gen configs. */
export function seedForItem(suiteSeed: number, id: string): number {
  let h = (2166136261 ^ seed32(suiteSeed)) >>> 0
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619) >>> 0
  return h
}

/** Resolve ONCE before both prompt building and evaluation. Do not feed raw generator entries to v1 index.ts. */
export function materializeV2(entry: V2Entry, suiteSeed: number): V2Test {
  seed32(suiteSeed)
  if (!('generator' in entry)) return structuredClone(entry)
  if (!Object.hasOwn(GENERATORS_V2, entry.generator)) throw new Error(`unknown v2 generator: ${entry.generator}`)
  const generated = GENERATORS_V2[entry.generator](seedForItem(suiteSeed, entry.id), entry.params)
  if (generated.category !== entry.category) throw new Error(`${entry.id}: generator category mismatch`)
  return { ...generated, id: entry.id, category: entry.category, weight: entry.weight, maxTokens: entry.maxTokens,
    difficulty: entry.difficulty, skill: entry.skill, variant: entry.variant }
}

export function resolveSuiteV2(manifest: V2Manifest, suiteSeed: number): QualityTestSet & { tests: V2Test[] } {
  seed32(suiteSeed)
  if (manifest.suite !== 'qb-2.0.0') throw new Error('expected qb-2.0.0 manifest')
  const ids = new Set<string>()
  for (const t of manifest.tests) {
    if (ids.has(t.id)) throw new Error(`duplicate v2 item: ${t.id}`)
    ids.add(t.id)
  }
  return { ...manifest, categoryWeights: { ...manifest.categoryWeights }, tests: manifest.tests.map((t) => materializeV2(t, suiteSeed)) }
}
