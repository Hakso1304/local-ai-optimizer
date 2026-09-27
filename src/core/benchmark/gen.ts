// Generation-config search: which (thinking, effort, sampling) settings the quality suite runs a model with.
// Pure: no I/O. The baseline (thinking off, T=0) always runs, so every model has a deterministic, comparable score.
import type { GenConfig, GenKnobs, GenQuality, Metric, ModelMeta, QualityResult } from '../../shared/bench-types'
import type { SessionRequest } from '../../shared/bench-events'

export const MAX_GEN_CONFIGS = 3
export const BASELINE_GEN: GenConfig = { id: 'off', thinking: false, temperature: 0, source: 'default' }

/** Knobs from the GGUF template / model card; a bare `supportsThinking` flag counts as thinking without effort levels. */
export const knobsOf = (m: ModelMeta): GenKnobs => m.genKnobs ?? { supportsThinking: !!m.supportsThinking }

const genId = (g: Omit<GenConfig, 'id'>): string =>
  g.thinking ? `think${g.effort ? `-${g.effort}` : ''}-t${g.temperature}` : g.temperature === 0 ? 'off' : `off-t${g.temperature}`

/** Middle effort: a value named like "medium", else the middle of the list (never the lowest). */
function middle(values: string[]): string {
  const named = values.find((v) => /^med/i.test(v))
  if (named && named !== values[0]) return named
  return values[Math.max(1, Math.floor((values.length - 1) / 2))]
}

/** [baseline] + for thinking models [on @ lowest effort, on @ middle effort], at the model card's sampling (else T=1.0). */
export function genConfigsFor(m: ModelMeta, req: Pick<SessionRequest, 'genSearch' | 'genConfigs'> = {}): GenConfig[] {
  if (req.genConfigs?.length) return req.genConfigs.slice(0, MAX_GEN_CONFIGS).map((g) => ({ ...g, id: g.id || genId(g) }))
  const k = knobsOf(m)
  if (!k.supportsThinking || req.genSearch === false) return [BASELINE_GEN]
  const rec = k.recommended ?? {}
  const sampling = {
    temperature: rec.temperature ?? 1.0,
    ...(rec.topP !== undefined ? { topP: rec.topP } : {}),
    ...(rec.topK !== undefined ? { topK: rec.topK } : {}),
    ...(rec.minP !== undefined ? { minP: rec.minP } : {}),
    source: (k.recommended ? 'model-card' : 'default') as GenConfig['source']
  }
  const efforts = k.effortValues ?? []
  const on = (effort?: string): GenConfig => { const g = { thinking: true, ...(effort ? { effort } : {}), ...sampling }; return { id: genId(g), ...g } }
  const out = [BASELINE_GEN, on(efforts[0])]
  if (efforts.length >= 2) out.push(on(middle(efforts)))
  return out.slice(0, MAX_GEN_CONFIGS)
}

/** chat_template_kwargs for this config, named as the template names them. undefined for non-thinking templates. */
export function templateKwargsFor(m: ModelMeta, g: GenConfig): Record<string, unknown> | undefined {
  const k = knobsOf(m)
  if (!k.supportsThinking) return undefined
  return { enable_thinking: g.thinking, ...(g.thinking && g.effort ? { [k.effortKw ?? 'reasoning_effort']: g.effort } : {}) }
}

/** Sampling fields for runPrompt / the exported request body (llama-server names). */
export function samplingFor(g: GenConfig): { temperature: number; top_p?: number; top_k?: number; min_p?: number } {
  return {
    temperature: g.temperature,
    ...(g.topP !== undefined ? { top_p: g.topP } : {}),
    ...(g.topK !== undefined ? { top_k: g.topK } : {}),
    ...(g.minP !== undefined ? { min_p: g.minP } : {})
  }
}

/** "thinking on (effort low, T=1.0)" / "thinking off (T=0)". */
export function genLabel(g: GenConfig): string {
  const s = [g.effort ? `effort ${g.effort}` : null, `T=${g.temperature === 0 ? '0' : g.temperature.toFixed(1)}`,
    g.topP !== undefined ? `top_p ${g.topP}` : null, g.topK !== undefined ? `top_k ${g.topK}` : null].filter(Boolean).join(', ')
  return `thinking ${g.thinking ? 'on' : 'off'} (${s})`
}

/** Reasoning vs answer split of a raw completion: Qwen/DeepSeek `<think>…</think>` and Gemma 4 thought channels. */
export function splitReasoning(text: string): { reasoningChars: number; answerChars: number } {
  let reasoning = 0
  // Templates that open `<think>` in the prompt itself leave only the closing tag in the completion.
  const close = text.indexOf('</think>')
  if (close >= 0 && !text.slice(0, close).includes('<think>')) { reasoning = close + 8; text = text.slice(close + 8) }
  const answer = text.replace(/<think>[\s\S]*?(<\/think>|$)|<\|channel>thought[\s\S]*?(<channel\|>|$)/gi, (x) => { reasoning += x.length; return '' })
  return { reasoningChars: reasoning, answerChars: answer.trim().length }
}

/** A graded quality row as stored: which gen config and sample produced it, plus that request's token/time counts. */
export type GenRow = QualityResult & { genId?: string; sample?: number; answerTokens?: number | null; reasoningTokens?: number | null; totalMs?: number | null }

const median = (xs: (number | null | undefined)[]): number | null => {
  const v = xs.filter((x): x is number => typeof x === 'number' && Number.isFinite(x)).sort((a, b) => a - b)
  return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null
}

/** Rows of one gen config → GenQuality (medians per request; unknown stays unavailable, never 0). */
export function summarizeGen(gen: GenConfig, rows: GenRow[], samples: number): GenQuality {
  const m = (v: number | null, source: string, reason: string): Metric => (v === null ? { value: null, kind: 'unavailable', reason } : { value: v, kind: 'measured', source })
  const ans = median(rows.map((r) => r.answerTokens)), rea = gen.thinking ? median(rows.map((r) => r.reasoningTokens)) : 0
  const ms = median(rows.map((r) => r.totalMs))
  const tps = median(rows.map((r) => (r.answerTokens != null && r.totalMs ? (r.answerTokens * 1000) / r.totalMs : null)))
  const src = `quality suite median of ${rows.length} requests`
  return {
    gen, results: rows, samples, stochastic: gen.temperature > 0,
    answerTokens: m(ans, src, 'no token counts'),
    reasoningTokens: rea === null ? { value: null, kind: 'unavailable', reason: 'no token counts' } : { value: rea, kind: gen.thinking ? 'estimated' : 'measured', source: gen.thinking ? `${src}; split by reasoning/answer text length` : 'thinking off' },
    effectiveAnswerLatencyMs: m(ms, src, 'no request timings'),
    effectiveTps: m(tps, `${src}; answer tokens / total s`, 'no token counts')
  }
}
