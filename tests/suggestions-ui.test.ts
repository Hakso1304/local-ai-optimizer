import { describe, expect, it } from 'vitest'
import { modelSuggestions } from '../src/main/suggestions'
import { dirOf } from '../src/renderer/src/Suggestions'
import { WORKLOADS } from '../src/core/scoring/workloads'
import type { ModelInfo } from '../src/shared/types'
import { machine } from './scoring/helpers'

describe('sibling-quant suggestions (app side)', () => {
  it('dirOf: the model\'s own folder is the download destination (Windows and POSIX paths)', () => {
    expect(dirOf(String.raw`D:\llm-models\Qwen3.8-27B-UD-Q4_K_M.gguf`)).toBe(String.raw`D:\llm-models`)
    expect(dirOf('/m/sub/a.gguf')).toBe('/m/sub')
  })

  it('models without usable GGUF metadata (or without siblings) yield no suggestions, never a throw', () => {
    const broken = { id: 'x', name: 'x', path: 'D:/m/x.gguf', sizeBytes: 1, runtime: 'llamacpp', meta: null, metaError: 'bad header' } as ModelInfo
    expect(modelSuggestions(machine(), [broken], WORKLOADS.coding)).toEqual([])
    expect(modelSuggestions(machine(), [], WORKLOADS.coding)).toEqual([])
  })
})
