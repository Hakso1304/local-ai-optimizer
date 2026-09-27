import { describe, expect, it } from 'vitest'
import { sectionOf, type Insight } from '../src/shared/interpret-types'
import { cardInsights } from '../src/renderer/src/InterpretPanel'

const ins = (ruleId: string, severity: Insight['severity'], key = ''): Insight => ({ ruleId, key, severity, metric: 'm', text: `[${ruleId}] x`, evidence: [] })

describe('interpretation UI helpers (INTERPRETATION.md §10)', () => {
  it('sectionOf reads the guide section from the rule id', () => {
    expect(sectionOf('I-2.1')).toBe(2)
    expect(sectionOf('I-10.3')).toBe(10)
    expect(sectionOf('bogus')).toBeNull()
  })

  it('cardInsights: ceiling line, decode band, every critical then at most 2 warnings', () => {
    const c = cardInsights([ins('I-3.1', 'info'), ins('I-2.1', 'info', 'ctx.ceiling'), ins('I-4.2', 'warn'), ins('I-5.3', 'critical'), ins('I-6.1', 'warn'), ins('I-6.3', 'critical'), ins('I-4.4', 'warn')])
    expect(c.ceiling?.ruleId).toBe('I-2.1')
    expect(c.decode?.ruleId).toBe('I-3.1')
    expect(c.alerts.map((i) => i.ruleId)).toEqual(['I-5.3', 'I-6.3', 'I-4.2', 'I-6.1']) // criticals never capped
    expect(cardInsights(undefined)).toEqual({ ceiling: null, decode: null, alerts: [] })
  })
})
