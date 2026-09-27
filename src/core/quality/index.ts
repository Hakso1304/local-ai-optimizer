import { runChecker, runCheckerAsync, type CheckerSpec } from './checkers'
import suiteV1 from './tests.v1.json'
import suiteV2 from './tests.v2.json'
import { GENERATOR_VERSION, resolveSuiteV2, type V2Manifest } from './generators.v2'

export type QualityCategory = 'instruction' | 'reasoning' | 'coding' | 'structured' | 'extraction' | 'context'

export interface NeedleParams {
  depth: number // 0..1, fraction of the filler before the needle sentence
  seed: number
  needle: string
}

export interface QualityTest {
  id: string
  category: QualityCategory
  weight: number
  maxTokens: number
  prompt?: string
  template?: 'needle'
  params?: NeedleParams
  checker: CheckerSpec
}

export interface QualityTestSet {
  version: string
  suite: string
  categoryWeights: Record<QualityCategory, number>
  tests: QualityTest[]
}

export interface QualityPrompt {
  testId: string
  category: QualityCategory
  messages: { role: 'user'; content: string }[]
  maxTokens: number
  temperature: 0
  seed: 1
}

export interface QualityResult {
  testId: string
  category: QualityCategory
  weight: number
  pass: boolean
  score: number
  detail: string
}

export const defaultTestSet = suiteV1 as QualityTestSet

export type QualityMode = 'quick' | 'thorough'
/** The suite a session runs, resolved once: prompts AND checkers come from this same object (docs/quality-v2.md). */
export interface SelectedSuite extends QualityTestSet {
  /** v2 only: the persisted seed the 13 generated items were derived from (shared by every candidate). */
  suiteSeed: number | null
  generatorVersion: string | null
}

/** Default / 'thorough' → qb-2.0.0 (60 items: 47 static + 13 generated from suiteSeed; ≈3.5× the quick suite's runtime).
 *  'quick' (explicit opt-in) → qb-1.1.0 (17 fixed tests). Credible quality is the default (product decision 2026-09-27). */
export function suiteFor(mode: QualityMode | undefined, suiteSeed: number): SelectedSuite {
  if (mode === 'quick') return { ...defaultTestSet, suiteSeed: null, generatorVersion: null }
  return { ...resolveSuiteV2(suiteV2 as unknown as V2Manifest, suiteSeed >>> 0), suiteSeed: suiteSeed >>> 0, generatorVersion: GENERATOR_VERSION }
}

// Deterministic 32-bit PRNG (mulberry32): same seed → same filler on every machine/build.
function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), a | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Lowercase words only, no digits or hyphens, so nothing in the filler can resemble a needle.
const ADJ = ['quiet', 'old', 'narrow', 'bright', 'distant', 'wooden', 'busy', 'gentle', 'plain', 'heavy', 'small', 'green']
const NOUN = ['farmer', 'river', 'teacher', 'market', 'bridge', 'garden', 'merchant', 'village', 'harbor', 'library', 'traveler', 'orchard']
const VERB = ['watched', 'crossed', 'painted', 'visited', 'measured', 'followed', 'repaired', 'described', 'passed', 'noticed']
const TAIL = ['before the rain began', 'during the long winter', 'on a calm morning', 'after the harvest', 'near the end of the road', 'while the bells rang']

/** Seeded filler of about `tokens` tokens (≈4 chars/token), as sentences. Same seed → same text. */
export function generateFiller(tokens: number, seed: number): string[] {
  const r = rng(seed)
  const pick = <T>(xs: T[]) => xs[Math.floor(r() * xs.length)]
  const out: string[] = []
  let chars = 0
  while (chars / 4 < tokens) {
    const s = `The ${pick(ADJ)} ${pick(NOUN)} ${pick(VERB)} the ${pick(ADJ)} ${pick(NOUN)} ${pick(TAIL)}.`
    out.push(s)
    chars += s.length + 1
  }
  return out
}

export function needlePrompt(p: NeedleParams, fillerTokens: number): string {
  const sentences = generateFiller(fillerTokens, p.seed)
  sentences.splice(Math.round(p.depth * sentences.length), 0, `The secret project name is ${p.needle}.`)
  return `Below is a document. Read it and answer the question that follows.\n\n<document>\n${sentences.join(' ')}\n</document>\n\nQuestion: What is the secret project name? Answer with the name only.`
}

export interface BuildOptions {
  /** Needle filler length in tokens (default 3000, fits ctx 8192). */
  fillerTokens?: number
  /** Thinking models: reasoning/coding maxTokens ×4 (DESIGN §4). */
  thinking?: boolean
}

export function buildQualityPrompts(set: QualityTestSet = defaultTestSet, opts: BuildOptions = {}): QualityPrompt[] {
  const fillerTokens = opts.fillerTokens ?? 3000
  return set.tests.map((t) => {
    let content: string
    if (t.template === 'needle') {
      if (!t.params) throw new Error(`${t.id}: needle template without params`)
      content = needlePrompt(t.params, fillerTokens)
    } else if (t.prompt) content = t.prompt
    else throw new Error(`${t.id}: neither prompt nor template`)
    const boost = opts.thinking && (t.category === 'reasoning' || t.category === 'coding') ? 4 : 1
    return { testId: t.id, category: t.category, messages: [{ role: 'user', content }], maxTokens: t.maxTokens * boost, temperature: 0, seed: 1 }
  })
}

/** Check one model output (message.content, not reasoning_content). Never throws on model output.
 *  Sync: jsCode runs in-process (CPU-timeout safe, NOT memory safe) — use evaluateAsync for real model output. */
export function evaluate(test: QualityTest, output: string): QualityResult {
  const r = runChecker(test.checker, output ?? '')
  return { testId: test.id, category: test.category, weight: test.weight, ...r }
}

/** Production path: jsCode tests run in a memory-capped child process (sandbox.ts). */
export async function evaluateAsync(test: QualityTest, output: string): Promise<QualityResult> {
  const r = await runCheckerAsync(test.checker, output ?? '')
  return { testId: test.id, category: test.category, weight: test.weight, ...r }
}

/** Per category: Σ(weight·pass) / Σ weight over that category's results. */
export function categoryPassRates(results: QualityResult[]): Partial<Record<QualityCategory, number>> {
  const acc: Partial<Record<QualityCategory, { w: number; p: number }>> = {}
  for (const r of results) {
    const a = (acc[r.category] ??= { w: 0, p: 0 })
    a.w += r.weight
    a.p += r.pass ? r.weight : 0
  }
  return Object.fromEntries(Object.entries(acc).filter(([, a]) => a!.w > 0).map(([c, a]) => [c, a!.p / a!.w]))
}

/**
 * Q = 100 · Σ_c W_c · passRate_c / Σ_c W_c, over categories that have results.
 * W_c = fixed category weights (suite.categoryWeights), so a category with many tests can't dominate;
 * renormalizing over present categories keeps a partial run on the same 0–100 scale.
 * Uses binary pass (DESIGN §4), not partial score. Returns null when there is nothing to score.
 */
export function qualityScore(results: QualityResult[], weights: Record<QualityCategory, number> = defaultTestSet.categoryWeights): number | null {
  const rates = categoryPassRates(results)
  let num = 0
  let den = 0
  // Fixed category order (weights' key order) so float summation doesn't depend on result order.
  for (const c of Object.keys(weights) as QualityCategory[]) {
    const rate = rates[c]
    if (rate === undefined) continue
    num += weights[c] * rate
    den += weights[c]
  }
  return den > 0 ? (100 * num) / den : null
}
