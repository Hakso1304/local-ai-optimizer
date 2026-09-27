// The active rule catalog (rules.v2.json, 'interp-2'). Every id means exactly what docs/INTERPRETATION.md v2 says;
// rules.v1.json is kept only as the record of what stored interp-1 recommendations cited.
import data from './rules.v2.json'

export type Severity = 'info' | 'note' | 'warn' | 'critical'
export type Origin = 'measured-calibration' | 'policy' | 'heuristic'
export interface Rule {
  id: string; key: string; section: number; severity: Severity; metric: string; origin: Origin; calibration?: string
  params: Record<string, number>; text: string; action?: string
}

export const RULES_VERSION: string = data.version
export const RULES: Rule[] = data.rules as Rule[]
const BY_KEY = new Map(RULES.map((r) => [r.key, r]))

export function rule(key: string): Rule {
  const r = BY_KEY.get(key)
  if (!r) throw new Error(`unknown interpretation rule ${key}`)
  return r
}
const fill = (t: string, vars: Record<string, unknown>) => t.replace(/\{(\w+)\}/g, (m, k: string) => (vars[k] === undefined ? m : String(vars[k])))
/** The rule's text template filled, prefixed with its id: "[I-2.1] …". */
export const cite = (key: string, vars: Record<string, unknown> = {}): string => `[${rule(key).id}] ${fill(rule(key).text, vars)}`
/** A free sentence under a rule's id (for secondary lines that share the rule's meaning). */
export const tag = (key: string, text: string): string => `[${rule(key).id}] ${text}`
export const P = (key: string, name: string): number => rule(key).params[name]

/** Typed actions (guide §9). Serialized as "<type>[ <argument>]", e.g. "use-context 16K". */
export type ActionType =
  | 'use-context' | 'enable-kv-q8' | 'enable-heavy-mode' | 'lower-required-context' | 'try-smaller-quant' | 'try-thinking-config'
  | 'run-thorough-quality' | 'rerun-idle' | 'rerun-comparable' | 'retry-telemetry' | 'inspect-diagnostics'
  /** unload + fresh server + re-measure the rung once (placement spill, I-2.8) */
  | 'restart-runtime'
export const action = (type: ActionType, arg?: string): string => (arg ? `${type} ${arg}` : type)
