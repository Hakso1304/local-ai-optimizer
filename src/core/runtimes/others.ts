import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { RuntimeDetection } from '../../shared/types'
import { NotImplementedError, getJson, type InferenceBackend } from './types'

// Ollama / LM Studio: detect() only for now. Both expose local HTTP APIs; "available" means the API answered.

abstract class HttpDetectBackend implements InferenceBackend {
  abstract readonly id: 'ollama' | 'lmstudio'
  protected abstract url: string
  protected abstract modelDirs(): string[]
  protected abstract modelNames(json: unknown): string[]

  async detect(): Promise<RuntimeDetection> {
    const modelDir = this.modelDirs().find((d) => existsSync(d))
    try {
      const json = await getJson<unknown>(this.url, 1_500)
      return { id: this.id, status: 'available', source: `GET ${this.url}`, models: this.modelNames(json), modelDir }
    } catch (e) {
      const why = (e as Error).name === 'TimeoutError' ? 'timed out after 1500ms' : (e as Error).message
      const hint = modelDir ? ` (model dir exists: ${modelDir} — installed but not running?)` : ''
      return { id: this.id, status: 'unavailable', source: `GET ${this.url}`, modelDir, error: `API not reachable: ${why}${hint}` }
    }
  }

  async enumerateModels(): Promise<never> { throw new NotImplementedError(`${this.id}.enumerateModels`) }
  async loadModel(): Promise<never> { throw new NotImplementedError(`${this.id}.loadModel`) }
  async unloadModel(): Promise<void> { throw new NotImplementedError(`${this.id}.unloadModel`) }
  async runPrompt(): Promise<never> { throw new NotImplementedError(`${this.id}.runPrompt`) }
  async getRuntimeStats(): Promise<never> { throw new NotImplementedError(`${this.id}.getRuntimeStats`) }
  configure(): void { throw new NotImplementedError(`${this.id}.configure`) }
  async cancel(): Promise<void> { throw new NotImplementedError(`${this.id}.cancel`) }
  async healthCheck(): Promise<never> { throw new NotImplementedError(`${this.id}.healthCheck`) }
}

export class OllamaBackend extends HttpDetectBackend {
  readonly id = 'ollama' as const
  protected url = 'http://127.0.0.1:11434/api/tags'
  protected modelDirs = () => [process.env.OLLAMA_MODELS ?? '', join(homedir(), '.ollama', 'models')].filter(Boolean)
  protected modelNames = (j: unknown) => ((j as { models?: { name: string }[] }).models ?? []).map((m) => m.name)
}

export class LmStudioBackend extends HttpDetectBackend {
  readonly id = 'lmstudio' as const
  protected url = 'http://127.0.0.1:1234/v1/models'
  protected modelDirs = () => [join(homedir(), '.lmstudio', 'models'), join(homedir(), '.cache', 'lm-studio', 'models')]
  protected modelNames = (j: unknown) => ((j as { data?: { id: string }[] }).data ?? []).map((m) => m.id)
}
