import type { LoadResult, ModelInfo, PromptResult, RuntimeDetection } from '../../shared/types'

export type { LoadResult, ModelInfo, PromptResult }

export interface LoadConfig {
  modelPath: string
  contextSize: number
  gpuLayers: number // -1 / 99 = all
  threads?: number
  batchSize?: number
  port?: number
  device: string // llama-server --device id, e.g. Vulkan0 (from listDevices; never let it default to all devices)
  extraArgs?: string[]
  /** Aborting kills the starting server and rejects with 'cancelled' (the /health wait can take up to 120 s). */
  signal?: AbortSignal
}

export interface PromptRequest {
  prompt: string
  maxTokens: number
  /** Persistent session/guard cancellation, including before the request starts. */
  signal?: AbortSignal
  temperature?: number
  /** Sampling (camelCase here → top_p / top_k / min_p in /completion); absent = llama-server defaults. */
  topP?: number
  topK?: number
  minP?: number
  seed?: number
  timeoutMs?: number
}

export interface RuntimeStats {
  [key: string]: unknown
}

export interface HealthStatus {
  ok: boolean
  detail: string
}

export interface InferenceBackend {
  readonly id: RuntimeDetection['id']
  detect(): Promise<RuntimeDetection>
  enumerateModels(dirs: string[]): Promise<ModelInfo[]>
  loadModel(cfg: LoadConfig): Promise<LoadResult>
  unloadModel(): Promise<void>
  runPrompt(req: PromptRequest, onToken?: (t: string) => void): Promise<PromptResult>
  getRuntimeStats(): Promise<RuntimeStats>
  configure(opts: Record<string, unknown>): void
  cancel(): Promise<void>
  healthCheck(): Promise<HealthStatus>
}

export class NotImplementedError extends Error {
  constructor(what: string) {
    super(`${what} is not implemented yet`)
    this.name = 'NotImplementedError'
  }
}

/** HTTP GET JSON with a hard timeout. Throws on network error, timeout or non-2xx. */
export async function getJson<T>(url: string, timeoutMs: number, signal?: AbortSignal): Promise<T> {
  const timeout = AbortSignal.timeout(timeoutMs)
  const res = await fetch(url, { signal: signal ? AbortSignal.any([signal, timeout]) : timeout })
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`)
  return (await res.json()) as T
}
