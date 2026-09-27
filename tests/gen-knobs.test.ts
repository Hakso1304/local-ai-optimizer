// Test scope: excerpts of the real chat templates on this machine (not the whole files), a temp sidecar file, and a
// synthetic token stream. No network: generation_config.json fetching is checked by hand against huggingface.co.
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { templateKnobs } from '../src/core/models/gguf'
import { parseGenerationConfig, readSidecar, writeSidecar } from '../src/core/hub/modelcard'
import { thinkCounter } from '../src/core/runtimes/llamacpp'
import { genQualityOf } from '../src/core/storage/sessions'
import type { ModelMeta } from '../src/shared/bench-types'

// Excerpts of the real templates on this machine (Qwen3.8-27B, gemma-4-26B-A4B).
const QWEN = `{%- if enable_thinking is undefined or enable_thinking is true %}
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort == 'high' %}{%- set resolved_reasoning_effort = 'xhigh' %}{%- endif %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{{- raise_exception('bad') }}{%- endif %}
{%- if preserve_thinking is undefined or preserve_thinking is true %}{%- endif %}`
const GEMMA = `{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set preserve_thinking = preserve_thinking | default(false) -%}
{%- if enable_thinking -%}{{- '<|think|>\\n' -}}{%- endif -%}`

describe('generation knobs', () => {
  it('reads thinking / effort kwargs and their values from the chat template (not the template itself)', () => {
    expect(templateKnobs(QWEN)).toEqual({
      templateKwNames: ['enable_thinking', 'preserve_thinking', 'reasoning_effort'],
      genKnobs: { supportsThinking: true, effortKw: 'reasoning_effort', effortValues: ['low', 'medium', 'high', 'xhigh'] }
    })
    expect(templateKnobs(GEMMA)).toEqual({ templateKwNames: ['enable_thinking', 'preserve_thinking'], genKnobs: { supportsThinking: true } })
    expect(templateKnobs('{{ messages }}')).toEqual({ templateKwNames: [], genKnobs: { supportsThinking: false } })
    expect(templateKnobs('{% if thinking_budget is defined %}{% endif %}{{ enable_thinking }}').genKnobs.thinkingBudgetKw).toBe('thinking_budget')
  })

  it('model card: generation_config.json → sampling fields, cached in <model>.meta.json', () => {
    expect(parseGenerationConfig({ temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, bos_token_id: 1 })).toEqual({ temperature: 0.6, topP: 0.95, topK: 20, minP: 0 })
    expect(parseGenerationConfig({ bos_token_id: 1 })).toBeNull()
    const dir = mkdtempSync(join(tmpdir(), 'lao-card-'))
    try {
      const model = join(dir, 'm.gguf')
      writeFileSync(model, 'x')
      writeSidecar(model, { repoId: 'Qwen/Qwen3-8B' })
      writeSidecar(model, { generation: { temperature: 0.6 }, fetchedAt: 't' })
      expect(readSidecar(model)).toEqual({ repoId: 'Qwen/Qwen3-8B', generation: { temperature: 0.6 }, fetchedAt: 't' })
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('counts streamed tokens inside thinking regions (Qwen think opened in the prompt, Gemma thought channel)', () => {
    const run = (prompt: string, chunks: string[]) => {
      const t = thinkCounter(prompt)
      for (const c of chunks) { if (t.inside) t.tokens++; t.feed(c) }
      return t.seen ? t.tokens : null
    }
    expect(run('<|im_start|>assistant\n<think>\n', ['Let', ' me', ' think', '</think>', 'Answer'])).toBe(4) // incl. the closing tag chunk
    expect(run('user: hi', ['<|channel>thought', '\nplan', ' more', '<channel|>', 'Hi'])).toBe(3)
    expect(run('user: hi', ['<thi', 'nk>', 'a', '</think>', 'b'])).toBe(2) // tag split across chunks
    expect(run('user: hi', ['Hello', ' there'])).toBeNull() // no thinking markers at all
  })

  it('genQualityOf rebuilds each generation config from its deterministic id and summarizes its rows', () => {
    const model = { id: 'm', name: 'm', supportsThinking: true, genKnobs: { supportsThinking: true } } as unknown as ModelMeta
    const row = (genId: string | undefined, pass: boolean, extra = {}) => ({ testId: 'IF-01', category: 'instruction' as const, weight: 1, pass, score: pass ? 1 : 0, detail: '', ...(genId ? { genId } : {}), ...extra })
    const out = genQualityOf(model, { workload: 'coding', modelIds: ['m'] }, [
      row(undefined, false, { answerTokens: 10, reasoningTokens: 0, totalMs: 1000 }),
      row('think-t1', true, { sample: 1, answerTokens: 10, reasoningTokens: 90, totalMs: 5000 }),
      row('think-t1', true, { sample: 2, answerTokens: 12, reasoningTokens: 80, totalMs: 4000 })
    ])
    const off = out.find((g) => g.gen.id === 'off')!, on = out.find((g) => g.gen.id !== 'off')!
    expect(off).toMatchObject({ samples: 1, qualityScore: 0 })
    expect(on.samples).toBe(2)
    expect(on.qualityScore).toBe(100)
    expect(on.gen.thinking).toBe(true)
  })
})
