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
  cuda: Sourced<{ available: boolean; version?: string }>
  disks: Sourced<DiskInfo[]>
  runtimes: RuntimeDetection[]
}

/** API exposed to the renderer by preload (window.api). */
export interface RendererApi {
  scanSystem(): Promise<SystemProfile>
  detectRuntimes(): Promise<RuntimeDetection[]>
}
