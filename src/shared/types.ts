// Types shared between main, preload and renderer. No runtime code here.
import type { BenchmarkRunResult, CandidateConfig, CliffReport, ModelMeta, QualityResult, Recommendation, WorkloadId, WorkloadProfile, WorkloadScore } from './bench-types'
import type { SessionEvent, SessionRequest, TelemetrySample } from './bench-events'

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
  /** GPU temperature/power source: nvidia-smi (NVIDIA only). Filled by main's system:scan. */
  nvidiaSmi?: { available: boolean; reason: string | null; cudaVersion: string | null }
}

/** Read from the GGUF header without loading the model. Plain fields are DECLARED (from the file);
 *  `estimated` holds our formulas (pruning only, never shown as fact). */
export interface GgufMetadata {
  ggufVersion: number
  arch: string | null
  name: string | null
  sizeLabel: string | null
  parameterCount: { value: number | null; kind: 'declared' | 'unavailable'; source: string }
  contextLength: number | null
  blockCount: number | null
  headCount: number | null
  headCountKv: number | null
  embeddingLength: number | null
  fileType: number | null
  quantName: string | null // from general.file_type, else parsed from the filename
  fileSizeBytes: number
  keyLength: number | null
  valueLength: number | null
  /** <arch>.vocab_size, else the length of tokenizer.ggml.tokens. */
  nVocab: number | null
  slidingWindow: number | null
  /** <arch>.attention.head_count_kv when it is a per-layer array (then headCountKv = its max). */
  headCountKvPerLayer: number[] | null
  /** Hybrid archs (qwen35): only every Nth layer is full attention and holds KV. */
  fullAttentionInterval: number | null
  /** true = sliding-window layer. */
  slidingWindowPattern: boolean[] | null
  keyLengthSwa: number | null
  valueLengthSwa: number | null
  headDim: { value: number | null; kind: 'declared' | 'estimated' }
  estimated: { kvCacheBytesPerToken: number | null }
}

export interface ModelInfo {
  id: string // absolute path; stable per file
  name: string
  path: string
  sizeBytes: number
  /** Where the file was found. All are benchmarked through our llama-server (Ollama blobs load directly with -m). */
  runtime: 'llamacpp' | 'ollama' | 'lmstudio'
  /** Ollama model name ("llama3.1:8b") when runtime = ollama. */
  ollamaName?: string
  meta: GgufMetadata | null
  metaError?: string
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

/** userData/settings.json */
export interface AppSettings {
  workload?: WorkloadId
  modelDirs?: string[]
}

export type StartResult = { ok: true; sessionId: string } | { ok: false; error: string }

/** API exposed to the renderer by preload (window.api). */
export interface RendererApi {
  getSettings(): Promise<AppSettings>
  setWorkload(w: WorkloadId): Promise<AppSettings>
  listWorkloads(): Promise<WorkloadProfile[]>
  listSessions(): Promise<SessionSummary[]>
  getSession(id: number): Promise<SessionDetail | null>
  /** OS save dialog for exported text; saved = chosen path, or null when cancelled. */
  saveFile(defaultName: string, content: string): Promise<{ saved: string | null }>
  /** Raw telemetry samples of one benchmark_run row. */
  telemetryForRun(runId: number): Promise<TelemetrySample[]>
  latestRecommendation(w: WorkloadId): Promise<{ sessionId: number; recommendation: Recommendation } | null>
  startBench(req: SessionRequest): Promise<StartResult>
  cancelBench(): Promise<{ ok: boolean; error?: string }>
  /** Stop between steps; the session ends 'paused' and is resumable. */
  pauseBench(): Promise<{ ok: boolean; error?: string }>
  /** Continue a paused/cancelled/interrupted/failed session; measured steps are reused unless retried/rerun. */
  resumeBench(sessionId: number, opts?: { retryFailed?: boolean; rerunConfigIds?: string[] }): Promise<StartResult>
  /** Download + install the llama.cpp runtime (first run / packaged app). Progress via onRuntimeProgress. */
  installRuntime(): Promise<RuntimeDetection>
  onRuntimeProgress(cb: (msg: string) => void): () => void
  /** Subscribe to bench:event; returns an unsubscribe function. */
  onBenchEvent(cb: (e: SessionEvent) => void): () => void
  scanSystem(): Promise<SystemProfile>
  detectRuntimes(): Promise<RuntimeDetection[]>
  listModels(): Promise<ModelInfo[]>
  /** modelId → null (fits in normal mode for this workload) or the reason it has no normal-mode candidate. */
  modelFit(w: WorkloadId): Promise<Record<string, string | null>>
  benchSmoke(modelPath: string): Promise<SmokeResult>
}

// ---- Stored sessions (read side). Payload contract for benchmark_session / benchmark_run / recommendation rows. ----

/** benchmark_session.payload */
export interface SessionPayload {
  workload: WorkloadId
  /** DEMO data (LAO_SEED_DEMO): must be flagged in every view and never used as a real recommendation. */
  demo?: boolean
  label?: string
  /** Declared VRAM of the bench GPU, for cliff/spill rules. */
  vramBytes: number | null
  candidates: { config: CandidateConfig; model: ModelMeta }[]
  /** Original request (for resume). */
  request?: SessionRequest
  /** The scan the candidate plan was generated from; resume re-plans from it so configIds/ctxSteps match. */
  machine?: SystemProfile
  /** Set when status = failed. */
  error?: string
}

export interface SessionSummary {
  id: number
  createdAt: string
  status: string
  workload: WorkloadId
  demo: boolean
  label: string | null
  error: string | null
  candidateCount: number
  bestConfigId: string | null
}

export interface SessionCandidate {
  config: CandidateConfig
  model: ModelMeta
  runs: BenchmarkRunResult[]
  /** benchmark_run row ids, parallel to runs (for telemetryForRun). */
  runIds: number[]
  /** Recomputed from the stored runs on read (pure scoring code), not stored. */
  cliff: CliffReport
  score: WorkloadScore | null
  quality: QualityResult[]
}

export interface SessionDetail {
  session: SessionSummary
  candidates: SessionCandidate[]
  recommendation: Recommendation | null
}
