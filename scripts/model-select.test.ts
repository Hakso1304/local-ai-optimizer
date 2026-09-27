import { describe, expect, it } from 'vitest'
import type { ModelInfo } from '../src/shared/types'
import { resolveSelectedModels } from './model-select'

describe('runner and manifest model selection', () => {
  const infos = [
    { name: 'fake-model', path: 'D:\\llm-models\\fake-model.gguf' },
    { name: 'other-model', path: 'D:\\llm-models\\subdir\\other-model.gguf' }
  ] as ModelInfo[]
  it('selects the scanner result by exact name for a new session', () => {
    expect(resolveSelectedModels(infos, ['fake-model'], false).map((info) => info.path)).toEqual(['D:\\llm-models\\fake-model.gguf'])
    expect(() => resolveSelectedModels(infos, ['FAKE-MODEL'], false)).toThrow(/not found/)
  })
  it('selects by the exact stored path for resume', () => {
    expect(resolveSelectedModels(infos, ['D:\\llm-models\\subdir\\other-model.gguf'], true)).toEqual([infos[1]])
    expect(() => resolveSelectedModels(infos, ['D:\\wrong\\other-model.gguf'], true)).toThrow(/not found/)
  })
})
