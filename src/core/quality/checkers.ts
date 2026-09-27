import vm from 'node:vm'
import { runSandboxed } from './sandbox'

// Deterministic output checkers. Every checker takes the raw model message content and returns
// {pass, score 0..1, detail}; none of them throws on model output.

export interface CheckResult {
  pass: boolean
  score: number
  detail: string
}

export type CheckerSpec =
  | { type: 'exact'; expected: string; caseSensitive?: boolean }
  | { type: 'regex'; pattern: string; flags?: string }
  | { type: 'number'; expected: number; tolerance?: number }
  | { type: 'containsAll'; items: string[]; caseSensitive?: boolean }
  | { type: 'needle'; needle: string }
  | { type: 'wordCount'; min: number; max: number }
  | { type: 'jsonSchema'; schema: JsonSchema }
  | { type: 'jsonEqual'; expected: unknown }
  | { type: 'jsCode'; cases: { expr: string; expected: unknown }[]; timeoutMs?: number }
  /** Reasoning allowed; the LAST "Answer:" line is extracted and checked with `inner`. */
  | { type: 'finalAnswer'; inner: Extract<CheckerSpec, { type: 'exact' | 'number' | 'regex' }> }

export interface JsonSchema {
  type?: 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array' | 'null'
  required?: string[]
  properties?: Record<string, JsonSchema>
  items?: JsonSchema
  minItems?: number
  enum?: unknown[]
}

const res = (pass: boolean, score: number, detail: string): CheckResult => ({ pass, score, detail })
const clip = (s: string, n = 80) => JSON.stringify(s.length > n ? `${s.slice(0, n)}…` : s)

/** Drop reasoning blocks some templates leave in content, closed or not, then trim:
 *  Qwen/DeepSeek `<think>…</think>` and Gemma 4 `<|channel>thought…<channel|>`. A template that opens the block in the
 *  prompt (Qwen3.x thinking: `…assistant\n<think>\n`) makes the reply start INSIDE it — only the close tag is in the
 *  content — so a leading segment up to a close tag with no open tag before it is reasoning too (session 5: every
 *  thinking answer was graded on its reasoning text). */
export function stripThinking(s: string): string {
  return s
    .replace(/^(?:(?!<think>)[\s\S])*?<\/think>/i, '')
    .replace(/^(?:(?!<\|channel>)[\s\S])*?<channel\|>/i, '')
    .replace(/<think>[\s\S]*?(<\/think>|$)/gi, '').replace(/<\|channel>thought[\s\S]*?(<channel\|>|$)/gi, '').trim()
}

/** Contents of the first ``` fence (preferring a js/javascript one); the whole string if there is no fence. */
export function stripFence(s: string, prefer?: RegExp): string {
  const fences = [...s.matchAll(/```([\w+-]*)[^\n]*\n([\s\S]*?)(```|$)/g)]
  const hit = (prefer && fences.find((f) => prefer.test(f[1]))) || fences[0]
  return (hit ? hit[2] : s).trim()
}

/** Per line: trim, collapse spaces, strip wrapping markdown/quotes and trailing .!; drop empty lines. */
function normalize(s: string, caseSensitive: boolean): string {
  const lines = s.split(/\r?\n/).map((l) =>
    l.trim().replace(/\s+/g, ' ').replace(/^[*_`"']+|[*_`"']+$/g, '').replace(/[.!]+$/, '').trim()
  )
  const out = lines.filter(Boolean).join('\n')
  return caseSensitive ? out : out.toLowerCase()
}

export function exactMatch(output: string, expected: string, caseSensitive = false): CheckResult {
  const got = normalize(stripThinking(output), caseSensitive)
  const ok = got === normalize(expected, caseSensitive)
  return res(ok, ok ? 1 : 0, ok ? 'exact match' : `expected ${clip(expected)}, got ${clip(got)}`)
}

export function regex(output: string, pattern: string, flags = ''): CheckResult {
  const got = stripThinking(output)
  const ok = new RegExp(pattern, flags).test(got)
  return res(ok, ok ? 1 : 0, ok ? `matches /${pattern}/` : `${clip(got)} does not match /${pattern}/`)
}

/** Last number in the output (thousands separators allowed) compared to expected. */
export function numberMatch(output: string, expected: number, tolerance = 0): CheckResult {
  const nums = stripThinking(output).match(/-?\d[\d,]*(?:\.\d+)?/g)
  if (!nums) return res(false, 0, 'no number in output')
  const got = Number(nums[nums.length - 1].replace(/,/g, ''))
  const ok = Math.abs(got - expected) <= tolerance
  return res(ok, ok ? 1 : 0, ok ? `number ${got}` : `expected ${expected}, got ${got}`)
}

/** Text after the last `Answer:` (case-insensitive; tolerates **bold**, `code`, leading "Final"), or null. */
export function lastAnswerLine(output: string): string | null {
  const lines = stripThinking(output).split(/\r?\n/)
  for (let i = lines.length - 1; i >= 0; i--) {
    const m = /^[\s>*_`#-]*(?:final\s+)?answer\s*[*_`]*\s*[:：]\s*[*_`]*\s*(.*?)\s*$/i.exec(lines[i])
    if (m) return m[1].replace(/[*_`]+$/g, '').trim()
  }
  return null
}

export function finalAnswer(output: string, inner: Extract<CheckerSpec, { type: 'exact' | 'number' | 'regex' }>): CheckResult {
  const a = lastAnswerLine(output)
  if (a === null) return res(false, 0, `no final "Answer:" line in ${clip(stripThinking(output).slice(-80))}`)
  const r = runChecker(inner, a)
  return { ...r, detail: `Answer: ${clip(a, 40)} — ${r.detail}` }
}

export function containsAll(output: string, items: string[], caseSensitive = false): CheckResult {
  const f = (s: string) => (caseSensitive ? s : s.toLowerCase())
  const hay = f(stripThinking(output))
  const missing = items.filter((i) => !hay.includes(f(i)))
  const score = items.length ? (items.length - missing.length) / items.length : 1
  return res(missing.length === 0, score, missing.length ? `missing: ${missing.join(', ')}` : 'all items present')
}

/** Full needle = 1. Only the part before the first '-' (e.g. HELIOTROPE without -5) = 0.5, fail. */
const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
const NEGATION = /\b(not|no|never|neither|nor|isn't|wasn't|aren't|doesn't|don't|cannot|can't)\b|n't\b/i

/** The needle must be retrieved as the answer: whole token (HELIOTROPE-59 ≠ HELIOTROPE-5), no negation earlier in its
 *  sentence ("…is not HELIOTROPE-5"), and no other needle-shaped candidate (WORD-123) named alongside it. */
export function needle(output: string, value: string): CheckResult {
  const text = stripThinking(output)
  const hay = text.toLowerCase()
  const tok = new RegExp(`(?<![\\w-])${esc(value)}(?![\\w-])`, 'i')
  const m = tok.exec(text)
  if (m) {
    const sentence = text.slice(0, m.index).split(/[.!?\n]/).pop() ?? ''
    if (NEGATION.test(sentence)) return res(false, 0, `needle ${value} appears negated: ${clip(sentence + value)}`)
    const others = [...new Set((text.match(/(?<![\w-])[A-Za-z]{3,}-\d+(?![\w-])/g) ?? []).map((x) => x.toUpperCase()))].filter((x) => x !== value.toUpperCase())
    if (others.length) return res(false, 0, `ambiguous: ${value} named together with ${others.join(', ')}`)
    return res(true, 1, `needle ${value} retrieved`)
  }
  const stem = value.split('-')[0].toLowerCase()
  if (stem !== value.toLowerCase() && hay.includes(stem)) return res(false, 0.5, `partial needle: found ${stem} but not ${value}`)
  return res(false, 0, `needle not found in ${clip(hay)}`)
}

export function wordCount(output: string, min: number, max: number): CheckResult {
  const n = stripThinking(output).replace(/[^\w\s'-]/g, ' ').trim().split(/\s+/).filter(Boolean).length
  const ok = n >= min && n <= max
  return res(ok, ok ? 1 : 0, `${n} words (want ${min === max ? min : `${min}-${max}`})`)
}

function parseJson(output: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const text = stripFence(stripThinking(output), /^json$/i)
  try {
    return { ok: true, value: JSON.parse(text) }
  } catch (e) {
    return { ok: false, error: `invalid JSON: ${(e as Error).message}` }
  }
}

/** Minimal validator for {type, required, properties, items, minItems, enum}. Returns error paths. */
export function validateSchema(schema: JsonSchema, v: unknown, path = '$'): string[] {
  const errs: string[] = []
  if (schema.type) {
    const t = v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v
    const ok = schema.type === 'integer' ? Number.isInteger(v) : schema.type === t
    if (!ok) return [`${path}: expected ${schema.type}, got ${t}`]
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(v))) errs.push(`${path}: not in enum`)
  if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
    const o = v as Record<string, unknown>
    for (const k of schema.required ?? []) if (!(k in o)) errs.push(`${path}.${k}: required`)
    for (const [k, s] of Object.entries(schema.properties ?? {})) if (k in o) errs.push(...validateSchema(s, o[k], `${path}.${k}`))
  }
  if (Array.isArray(v)) {
    if (schema.minItems !== undefined && v.length < schema.minItems) errs.push(`${path}: ${v.length} items < minItems ${schema.minItems}`)
    if (schema.items) v.forEach((x, i) => errs.push(...validateSchema(schema.items!, x, `${path}[${i}]`)))
  }
  return errs
}

export function jsonSchema(output: string, schema: JsonSchema): CheckResult {
  const p = parseJson(output)
  if (!p.ok) return res(false, 0, p.error)
  const errs = validateSchema(schema, p.value)
  return res(errs.length === 0, errs.length ? 0 : 1, errs.length ? errs.slice(0, 3).join('; ') : 'valid against schema')
}

/** JSON with object keys sorted, so key order doesn't matter but array order does. */
export function canonical(v: unknown): string {
  return JSON.stringify(v, (_k, x) =>
    x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : 1))) : x
  )
}

export function jsonEqual(output: string, expected: unknown): CheckResult {
  const p = parseJson(output)
  if (!p.ok) return res(false, 0, p.error)
  const ok = canonical(p.value) === canonical(expected)
  return res(ok, ok ? 1 : 0, ok ? 'JSON equal' : `expected ${clip(canonical(expected))}, got ${clip(canonical(p.value))}`)
}

/** Code from the first ```js/javascript fence (else first fence, else the bare text), ESM exports stripped. */
export function extractCode(output: string): string {
  return stripFence(stripThinking(output), /^(js|javascript|jsx|mjs)$/i)
    .replace(/^\s*export\s+default\s+/gm, '')
    .replace(/^\s*export\s+(?=(async\s+)?(function|const|let|var|class)\b)/gm, '')
}

type JsCase = { expr: string; expected: unknown }

/** Model code + one probe per case; evaluates to a JSON string array (one JSON.stringify per case). */
function harnessScript(output: string, cases: JsCase[]): string {
  const probes = cases
    .map((c) => `try{__r.push(JSON.stringify(${c.expr}))}catch(e){__r.push("!"+String(e&&e.message))}`)
    .join('\n')
  return `var module={exports:{}},exports=module.exports;\n${extractCode(output)}\n;(function(){var __r=[];\n${probes}\nreturn JSON.stringify(__r)})()`
}

function scoreHarness(raw: unknown, cases: JsCase[]): CheckResult {
  let got: (string | null)[]
  try {
    got = typeof raw === 'string' ? JSON.parse(raw) : []
  } catch {
    got = []
  }
  if (!Array.isArray(got) || got.length !== cases.length) return res(false, 0, 'harness result tampered or missing')
  const fails: string[] = []
  cases.forEach((c, i) => {
    const want = JSON.stringify(c.expected)
    if (got[i] !== want) fails.push(`${c.expr} → ${got[i] ?? 'undefined'} (want ${want})`)
  })
  const passed = cases.length - fails.length
  return res(fails.length === 0, cases.length ? passed / cases.length : 0, fails.length ? `${passed}/${cases.length}: ${fails.slice(0, 2).join('; ')}` : `${passed}/${cases.length} cases`)
}

/**
 * In-process: run model code in a fresh V8 context and evaluate each case expression after it.
 * Isolation: DONT_CONTEXTIFY context (no host objects inside, so no `this.constructor.constructor`
 * escape), string code generation disabled (no eval/new Function), no require/process/fetch/timers,
 * sync timeout covers microtasks. Only a primitive string crosses back to the host.
 * NOT memory-safe: an allocation bomb aborts the calling process. Real model output → jsCodeAsync.
 */
export function jsCode(output: string, cases: JsCase[], timeoutMs = 2000): CheckResult {
  let raw: unknown
  try {
    const ctx = vm.createContext(vm.constants.DONT_CONTEXTIFY, {
      codeGeneration: { strings: false, wasm: false },
      microtaskMode: 'afterEvaluate'
    })
    raw = vm.runInContext(harnessScript(output, cases), ctx, { timeout: timeoutMs })
  } catch (e) {
    return res(false, 0, `execution failed: ${String((e as Error)?.message ?? e)}`)
  }
  return scoreHarness(raw, cases)
}

/** Same checks as jsCode, but in a memory-capped child process (see sandbox.ts). Never throws. */
export async function jsCodeAsync(output: string, cases: JsCase[], timeoutMs = 2000, memoryMb?: number): Promise<CheckResult> {
  const o = await runSandboxed(harnessScript(output, cases), { timeoutMs, memoryMb })
  return o.ok ? scoreHarness(o.raw, cases) : res(false, 0, `execution failed: ${o.error}`)
}

export function runChecker(spec: CheckerSpec, output: string): CheckResult {
  switch (spec.type) {
    case 'exact': return exactMatch(output, spec.expected, spec.caseSensitive)
    case 'regex': return regex(output, spec.pattern, spec.flags)
    case 'number': return numberMatch(output, spec.expected, spec.tolerance)
    case 'containsAll': return containsAll(output, spec.items, spec.caseSensitive)
    case 'needle': return needle(output, spec.needle)
    case 'wordCount': return wordCount(output, spec.min, spec.max)
    case 'jsonSchema': return jsonSchema(output, spec.schema)
    case 'jsonEqual': return jsonEqual(output, spec.expected)
    case 'jsCode': return jsCode(output, spec.cases, spec.timeoutMs)
    case 'finalAnswer': return finalAnswer(output, spec.inner)
    default: throw new Error(`unknown checker type ${(spec as { type: string }).type}`)
  }
}

/** Production path: jsCode goes through the child-process sandbox, everything else is sync. */
export async function runCheckerAsync(spec: CheckerSpec, output: string): Promise<CheckResult> {
  return spec.type === 'jsCode' ? jsCodeAsync(output, spec.cases, spec.timeoutMs) : runChecker(spec, output)
}
