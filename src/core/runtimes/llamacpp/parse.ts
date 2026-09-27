// Pure parsers for llama-server output. No I/O here so they can be tested from fixtures.
import type { LoadResult, PromptResult } from '../../../shared/types'
import { INTEGRATED_GPU_NAME } from '../../system/scanner'

/** Split an SSE text buffer into complete `data:` payloads; returns the unfinished remainder. */
export function parseSse(buf: string): { events: unknown[]; rest: string } {
  const lines = buf.split('\n')
  const rest = lines.pop() ?? ''
  const events: unknown[] = []
  for (const raw of lines) {
    const line = raw.replace(/\r$/, '')
    if (!line.startsWith('data:')) continue
    const data = line.slice(5).trim()
    if (!data || data === '[DONE]') continue
    events.push(JSON.parse(data))
  }
  return { events, rest }
}

export interface LlamaTimings {
  cache_n?: number
  prompt_n?: number
  prompt_ms?: number
  prompt_per_second?: number
  predicted_n?: number
  predicted_ms?: number
  predicted_per_second?: number
}

export interface CompletionChunk {
  content?: string
  stop?: boolean
  stop_type?: string
  timings?: LlamaTimings
  error?: { message?: string } | string
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** Map the final stream chunk (the one with `timings`) plus our wall-clock measurements. */
export function toPromptResult(
  final: CompletionChunk | null,
  m: { ttftMs: number | null; totalMs: number; text: string; timedOut: boolean; error: string | null }
): PromptResult {
  const t = final?.timings ?? {}
  return {
    ttftMs: m.ttftMs,
    promptTokens: num(t.prompt_n),
    prefillMs: num(t.prompt_ms),
    prefillTps: num(t.prompt_per_second),
    decodeTokens: num(t.predicted_n),
    decodeMs: num(t.predicted_ms),
    decodeTps: num(t.predicted_per_second),
    totalMs: m.totalMs,
    text: m.text,
    stopType: final?.stop_type ?? null,
    timedOut: m.timedOut,
    error: m.error
  }
}

export const emptyDeclared = (): LoadResult['declared'] => ({
  layersOffloaded: null,
  layersTotal: null,
  modelBufferMiB: {},
  kvBufferMiB: {},
  computeBufferMiB: {}
})

/** Fold one startup log line into `acc`. Formats observed on b11208:
 *  `offloaded 25/25 layers to GPU`, `Vulkan0 model buffer size = 500.79 MiB`,
 *  `Vulkan0 KV buffer size = 24.00 MiB`, `Vulkan0 compute buffer size = 34.01 MiB`. */
export function parseLogLine(line: string, acc: LoadResult['declared']): void {
  const off = /offloaded (\d+)\/(\d+) layers to GPU/.exec(line)
  if (off) {
    acc.layersOffloaded = Number(off[1])
    acc.layersTotal = Number(off[2])
    return
  }
  const buf = /(\S+) (model|KV|compute) buffer size\s*=\s*([\d.]+) MiB/.exec(line)
  if (!buf) return
  const map = buf[2] === 'model' ? acc.modelBufferMiB : buf[2] === 'KV' ? acc.kvBufferMiB : acc.computeBufferMiB
  map[buf[1]] = Number(buf[3])
}

export interface LlamaDevice {
  id: string // value for --device, e.g. Vulkan0
  name: string
  totalMiB: number
  freeMiB: number
}

/** Parse `--list-devices` lines like `  Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)`. */
export function parseDevices(out: string): LlamaDevice[] {
  const devs: LlamaDevice[] = []
  for (const m of out.matchAll(/^\s*(\S+): (.+?) \((\d+) MiB, (\d+) MiB free\)\s*$/gm)) {
    devs.push({ id: m[1], name: m[2], totalMiB: Number(m[3]), freeMiB: Number(m[4]) })
  }
  return devs
}

/** First device whose name doesn't look integrated. Memory can't decide it: iGPUs report shared RAM (16 GB on the dev box). */
export function pickDiscreteDevice(devs: LlamaDevice[]): LlamaDevice | null {
  return devs.find((d) => !INTEGRATED_GPU_NAME.test(d.name)) ?? null
}

export type ExitReason = 'oom' | 'device_lost' | 'crash'

export function classifyExit(tail: string[]): ExitReason {
  const text = tail.join('\n')
  // OOM first: an allocation failure is usually the root cause of a later device loss.
  if (/failed to allocate|out of memory|ErrorOutOfDeviceMemory/i.test(text)) return 'oom'
  if (/DeviceLost|ErrorDeviceLost/i.test(text)) return 'device_lost'
  return 'crash'
}
