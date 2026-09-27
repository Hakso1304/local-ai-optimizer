// Results → "Interpretation" (docs/INTERPRETATION.md §10): insights grouped by guide section 2–8, evidence
// expandable, actions as buttons where the app can execute them. Also the [I-x.y] tag tooltips used in reasons.
import type { ReactNode } from 'react'
import { sectionOf, type Insight } from '../../shared/interpret-types'
import { Prov, fmtCtx } from './ui'

const SECTIONS: Record<number, string> = {
  1: 'Provenance', 2: 'Context ladder and practical ceiling', 3: 'Speed', 4: 'Memory', 5: 'Quality', 6: 'Stability and reproducibility',
  7: 'Comparing candidates', 8: 'Generation configuration'
}

// Rule texts for the [I-x.y] tooltips. Eager glob: empty (no tooltip text) if the engine's rules file isn't there.
const ruleFiles = import.meta.glob<{ version?: string; rules?: { id: string; text: string; section: number }[] }>('../../core/interpret/rules.v1.json', { eager: true, import: 'default' })
const RULE_TEXT = new Map((Object.values(ruleFiles)[0]?.rules ?? []).map((r) => [r.id, r.text] as const))

/** Renders a reason/insight string; a leading "[I-x.y]" becomes a tag whose tooltip is the guide rule. */
export function Reason({ text }: { text: string }) {
  const m = /^\[(I-\d+\.\d+)\]\s*/.exec(text)
  if (!m) return <>{text}</>
  const s = sectionOf(m[1])
  const rule = RULE_TEXT.get(m[1])?.replace(/\{\w+\}/g, '…')
  const tip = `Guide rule ${m[1]} (docs/INTERPRETATION.md §${s ?? '?'})${rule ? `: ${rule}` : ''}`
  return <><span className="rule-tag" title={tip}>{m[1]}</span> {text.slice(m[0].length)}</>
}

/** Executable actions (§9). Anything else is shown as a plain hint. */
export interface InsightActions {
  enableHeavyMode(): void
  runThoroughQuality(): void
  tryThinkingConfig(): void
  download(model: string): void
  useContext(ctx: number): void
}

function actionButton(i: Insight, a: InsightActions): ReactNode {
  if (!i.action) return null
  const [name, ...rest] = i.action.split(' ')
  const arg = rest.join(' ')
  // use-context = stay at or below the last clean rung: an explicit arg, else the smallest context in the evidence.
  const ctxs = i.evidence.map((e) => e.ctx).filter((c): c is number => c != null)
  // typed action "<type>[ <arg>]": sizes may be written as 16K / 32768
  const k = /^(\d+)\s*K$/i.exec(arg)
  const argCtx = k ? Number(k[1]) * 1024 : Number(arg)
  const ctx = argCtx || (ctxs.length ? Math.min(...ctxs) : undefined)
  switch (name) {
    case 'enable-heavy-mode': return <button className="mini" onClick={a.enableHeavyMode}>Enable heavy mode</button>
    case 'run-thorough-quality': return <button className="mini" onClick={a.runThoroughQuality}>Run thorough quality</button>
    case 'try-thinking-config': return <button className="mini" onClick={a.tryThinkingConfig}>Search generation settings</button>
    case 'download': return <button className="mini" onClick={() => a.download(arg)}>Download{arg ? ` ${arg}` : ''}</button>
    case 'use-context': return ctx ? <button className="mini" onClick={() => a.useContext(ctx)}>Use {fmtCtx(ctx)} in export</button> : null
    default: return <span className="pill" title="suggested next step">{i.action}</span>
  }
}

const SEV_ORDER = { critical: 0, warn: 1, note: 2, info: 3 } as const

export function InterpretPanel({ insights, actions, tag = '' }: { insights: Insight[]; actions: InsightActions; tag?: string }) {
  if (!insights.length) return null
  const groups = new Map<number, Insight[]>()
  for (const i of insights) { const s = sectionOf(i.ruleId) ?? 7; groups.set(s, [...(groups.get(s) ?? []), i]) }
  // §0: context (2) and quality (5) lead, then the rest in guide order.
  const order = [2, 5, 3, 4, 6, 7, 8, 1].filter((s) => groups.has(s))
  return (
    <>
      <h2>Interpretation{tag}</h2>
      {order.map((s) => (
        <div key={s} className="interp-group">
          <h3>§{s} {SECTIONS[s] ?? ''}</h3>
          <ul>
            {groups.get(s)!.sort((x, y) => SEV_ORDER[x.severity] - SEV_ORDER[y.severity]).map((i, k) => (
              <li key={`${i.ruleId}-${k}`} className={`sev-${i.severity}${i.evaluable === false ? ' not-evaluable' : ''}`}>
                <span className={`sev sev-${i.severity}`}>{i.severity}</span> <Reason text={i.text} />
                {i.evaluable === false ? <span className="muted"> (not evaluable from stored data)</span> : actionButton(i, actions)}
                {i.evidence.length > 0 && (
                  <details>
                    <summary className="muted">evidence ({i.evidence.length})</summary>
                    <table className="kv">
                      <tbody>
                        {i.evidence.map((e, n) => (
                          <tr key={n}><td>{e.metric}</td><td>{e.value ?? '—'} <Prov kind={e.kind} /></td><td className="muted">{e.ctx != null ? `@${fmtCtx(e.ctx)}` : ''} {e.configId ?? ''}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </details>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </>
  )
}

/** The rules version this build's engine uses (null if the rules file isn't there). */
export const ENGINE_RULES: string | null = (Object.values(ruleFiles)[0] as { version?: string } | undefined)?.version ?? null

/** Rules version a recommendation was made with; null = before the rule engine (no insights stored). */
export const rulesOf = (rec: unknown): string | null => (rec as { rulesVersion?: string } | null)?.rulesVersion ?? null

/** Recommendation.insights (persisted with the recommendation by the interpretation engine); [] for older ones. */
export const insightsOf = (rec: unknown): Insight[] => ((rec as { insights?: Insight[] } | null)?.insights ?? [])

/** Dashboard card: the ceiling line (I-2.x), the decode band (I-3.1) and at most 2 warn/critical insights (§10). */
export function cardInsights(insights: Insight[] | undefined) {
  const list = insights ?? []
  return {
    ceiling: list.find((i) => sectionOf(i.ruleId) === 2 && /ceiling/i.test(i.key ?? i.metric)) ?? list.find((i) => sectionOf(i.ruleId) === 2) ?? null,
    decode: list.find((i) => i.ruleId === 'I-3.1') ?? null,
    // every critical insight, then at most 2 warnings (W4f: criticals are never capped)
    alerts: [...list.filter((i) => i.severity === 'critical' && i.evaluable !== false), ...list.filter((i) => i.severity === 'warn' && i.evaluable !== false).slice(0, 2)]
  }
}
