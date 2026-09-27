import { describe, expect, it } from 'vitest'
import { sectionOf, type Insight } from '../src/shared/interpret-types'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { ENGINE_RULES, InterpretPanel, Reason, RulesVersion, cardInsights } from '../src/renderer/src/InterpretPanel'
import v1 from '../src/core/interpret/rules.v1.json'
import v2 from '../src/core/interpret/rules.v2.json'

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

describe('interpretation UI rendering (G02 versioned citations, G14 critical-first)', () => {
  const noop = () => {}
  const actions = { enableHeavyMode: noop, runThoroughQuality: noop, tryThinkingConfig: noop, download: noop, useContext: noop }
  const textOf = (rules: { rules: { id: string; text: string }[] }, id: string) => rules.rules.find((r) => r.id === id)!.text.replace(/\{\w+\}/g, '…')
  const html = (el: Parameters<typeof renderToStaticMarkup>[0]) => renderToStaticMarkup(el).replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&amp;/g, '&')

  it('tooltips resolve against the record\'s rules version; the active catalog by default', () => {
    expect(ENGINE_RULES).toBe(v2.version)
    expect(textOf(v1, 'I-5.7')).not.toBe(textOf(v2, 'I-5.7')) // the id changed meaning between versions
    expect(html(createElement(Reason, { text: '[I-5.7] x' }))).toContain(textOf(v2, 'I-5.7'))
    expect(html(createElement(Reason, { text: '[I-5.7] x', rules: 'interp-1' }))).toContain(textOf(v1, 'I-5.7'))
    expect(html(createElement(RulesVersion.Provider, { value: 'interp-1' }, createElement(Reason, { text: '[I-5.7] x' })))).toContain(textOf(v1, 'I-5.7'))
  })

  it('every critical renders before the section groups; not-evaluable is greyed without an action', () => {
    const list = [ins('I-2.1', 'info'), ins('I-3.1', 'warn'), { ...ins('I-6.1', 'critical') }, { ...ins('I-4.2', 'note'), evaluable: false, action: 'use-context 16K' }]
    const out = html(createElement(InterpretPanel, { insights: list, actions }))
    const pos = (s: string) => out.indexOf(s)
    expect(pos('Critical')).toBeGreaterThan(-1)
    expect(pos('I-6.1')).toBeLessThan(pos('I-2.1'))
    expect(out.match(/I-6\.1<\/span>/g)).toHaveLength(1) // not duplicated in its section
    expect(out).toContain('not-evaluable')
    expect(out).toContain('not evaluable from stored data')
    expect(out).not.toContain('Use 16K') // no action button on a not-evaluable rule
  })
})
