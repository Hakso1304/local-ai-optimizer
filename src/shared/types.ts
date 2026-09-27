// Types shared between main, preload and renderer. No runtime code here.

export type Status = 'available' | 'unavailable' | 'unsupported'

/** Every measured value carries where it came from. value is null unless status is 'available'
 *  (except 'unsupported' may carry a definitive negative, e.g. cuda {available:false}). */
export interface Sourced<T> {
  value: T | null
  status: Status
  source: string
  error?: string
}

export type GpuVendor = 'nvidia' | 'amd' | 'intel' | 'other'

export interface GpuInfo {
  name: string
  vendor: GpuVendor
  pnpDeviceId: string
  driverVersion: string | null
  dedicatedVramBytes: Sourced<number>
  /** Heuristic guess (name pattern / small VRAM), not a hardware fact. */
  isIntegrated: boolean
}

export interface DiskInfo {
  mount: string
  totalBytes: number
  freeBytes: number
}

export interface RuntimeDetection {
  id: 'llamacpp' | 'ollama' | 'lmstudio'
  status: Status
  source: string
  version?: string
  path?: string
  models?: string[]
  modelDir?: string
  error?: string
}

export interface SystemProfile {
  scannedAt: string
  os: Sourced<{ name: string; version: string; build: string }>
  cpu: Sourced<{ model: string; physicalCores: number; logicalCores: number }>
  ram: Sourced<{ totalBytes: number; availableBytes: number }>
  gpus: Sourced<GpuInfo[]>
  cuda: Sourced<{ available: boolean; driverCudaVersion?: string }>
  disks: Sourced<DiskInfo[]>
  runtimes: RuntimeDetection[]
}

export interface ModelInfo {
  id: string // absolute path; stable per file
  name: string
  path: string
  sizeBytes: number
  runtime: 'llamacpp'
}

/** One measured prompt. ttftMs/totalMs are wall clock (measured here); prefill/decode come from
 *  the runtime's own `timings` (declared). null = runtime did not report it. */
export interface PromptResult {
  ttftMs: number | null
  promptTokens: number | null
  prefillMs: number | null
  prefillTps: number | null
  decodeTokens: number | null
  decodeMs: number | null
  decodeTps: number | null
  totalMs: number
  text: string
  stopType: string | null
  timedOut: boolean
  error: string | null
}

/** loadTimeMs is wall clock spawn -> /health ok (measured). `declared` is parsed from the
 *  runtime's startup log; MiB values keyed by buffer name as printed (e.g. Vulkan0, CPU_Mapped). */
export interface LoadResult {
  loadTimeMs: number
  declared: {
    layersOffloaded: number | null
    layersTotal: number | null
    modelBufferMiB: Record<string, number>
    kvBufferMiB: Record<string, number>
    computeBufferMiB: Record<string, number>
  }
}

export interface SmokeResult {
  load: LoadResult
  prompt: PromptResult
}

/** API exposed to the renderer by preload (window.api). */
export interface RendererApi {
  scanSystem(): Promise<SystemProfile>
  detectRuntimes(): Promise<RuntimeDetection[]>
  listModels(): Promise<ModelInfo[]>
  benchSmoke(modelPath: string): Promise<SmokeResult>
}
