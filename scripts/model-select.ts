/** One model-root and name/path resolver for the runner and immutable gate. */
import type { ModelInfo } from '../src/shared/types'

export const MODELS_DIR = 'D:\\llm-models'

export function resolveSelectedModels(infos: ModelInfo[], names: string[], resume: boolean): ModelInfo[] {
  return names.map((name) => {
    const info = infos.find((candidate) => resume ? candidate.path === name : candidate.name === name)
    if (!info) throw new Error(`${name} not found in ${MODELS_DIR}`)
    return info
  })
}
