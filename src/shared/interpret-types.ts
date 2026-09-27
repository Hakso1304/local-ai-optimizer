// Interpretation insights, shape per docs/INTERPRETATION.md §0/§10 (the engine lives in src/core/interpret). Types only.
import type { ProvenanceKind } from './bench-types'

export type Severity = 'info' | 'note' | 'warn' | 'critical'

export interface Evidence {
  metric: string
  value: number | string | null
  kind: ProvenanceKind
  ctx?: number
  configId?: string
}

export interface Insight {
  /** Guide rule id, e.g. "I-2.1" (section 2). */
  ruleId: string
  /** Stable rule key, e.g. "ctx.ceiling". */
  key?: string
  severity: Severity
  metric: string
  /** Rendered sentence, starting with its "[I-x.y]" tag. */
  text: string
  evidence: Evidence[]
  /** Action vocabulary of §9, e.g. "enable-heavy-mode", "use-context 32768", "download <model>". */
  action?: string
  configId?: string
}

/** Guide section of a rule id ("I-3.1" → 3); null when unparseable. */
export const sectionOf = (ruleId: string): number | null => {
  const m = /^I-(\d+)\./.exec(ruleId)
  return m ? Number(m[1]) : null
}
