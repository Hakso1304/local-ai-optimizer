// Loads ACCEPTANCE §2.3-shaped fixtures into scoring inputs. Plain numbers → measured Metric, null → unavailable.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type {
  BenchmarkRunResult, CandidateConfig, CandidateInput, FailureKind, MachineLimits, Metric, ModelMeta, QualityResult, RunStatus
} from '../../src/shared/bench-types'

export interface FixtureRun {
  configId: string; model: string; ctx: number; gpuLayers: number; threads: number; status: string
  promptTokens: number | null; loadMs: number | null; ttftMs: number | null; prefillTps: number | null; decodeTps: number | null
  totalMs: number | null; peakRamBytes: number | null; peakVramBytes: number | null; peakSharedGpuBytes: number | null
  cpuAvgPct: number | null; gpuAvgPct: number | null
}
export interface Fixture { vramBytes: number; models: ModelMeta[]; runs: FixtureRun[]; expected?: Record<string, unknown> }

export const load = (name: string): Fixture =>
  JSON.parse(readFileSync(join(__dirname, '../fixtures/scoring', name), 'utf8'))

const m = (v: number | null): Metric => (v === null ? { value: null, kind: 'unavailable', reason: 'not in fixture' } : { value: v, kind: 'measured' })

// ACCEPTANCE A11 status vocabulary → RunStatus + FailureKind.
const STATUS: Record<string, [RunStatus, FailureKind | null]> = {
  ok: ['pass', null], failed: ['fail', 'exit_1'], oom: ['fail', 'oom'], device_lost: ['fail', 'device_lost'],
  crashed: ['fail', 'crash'], timeout: ['timeout', 'req_timeout'], cancelled: ['cancelled', null]
}

export const toRun = (r: FixtureRun): BenchmarkRunResult => ({
  configId: r.configId, ctx: r.ctx, promptTokens: r.promptTokens, status: STATUS[r.status][0], failureKind: STATUS[r.status][1],
  loadTimeMs: m(r.loadMs), ttftMs: m(r.ttftMs), prefillTps: m(r.prefillTps), decodeTps: m(r.decodeTps), totalMs: m(r.totalMs),
  peakVramBytes: m(r.peakVramBytes), peakSharedGpuBytes: m(r.peakSharedGpuBytes), peakRamBytes: m(r.peakRamBytes),
  avgGpuUtil: m(r.gpuAvgPct), avgCpuUtil: m(r.cpuAvgPct)
})

export const machine = (vramBytes = 17095983104): MachineLimits => ({
  vramBytes: { value: vramBytes, kind: 'declared' }, vramInUseBytes: { value: 0, kind: 'measured' },
  ramTotalBytes: { value: 31 * 1024 ** 3, kind: 'declared' }, ramAvailableBytes: { value: 20 * 1024 ** 3, kind: 'measured' },
  physicalCores: 8, gpuDevice: 'Vulkan0'
})

/** One CandidateInput per configId, in first-appearance order. */
export function inputs(f: Fixture): CandidateInput[] {
  const ids = [...new Set(f.runs.map((r) => r.configId))]
  return ids.map((id) => {
    const rows = f.runs.filter((r) => r.configId === id)
    const model = f.models.find((x) => x.id === rows[0].model)!
    const config: CandidateConfig = {
      id, modelId: model.id, device: 'Vulkan0', gpuLayers: rows[0].gpuLayers, gpuLayersAll: rows[0].gpuLayers >= model.layers,
      kvType: 'f16', flashAttn: true, threads: rows[0].threads, ctxSteps: rows.map((r) => r.ctx), skippedSteps: [],
      estVramBytes: { value: null, kind: 'unavailable' }, estRamBytes: { value: null, kind: 'unavailable' }, notes: []
    }
    return { config, model, runs: rows.map(toRun), quality: [] }
  })
}

/** Synthetic measured quality: 5 items per category, the first `rate × 5` pass (policy tests, not measurements). */
export const q5 = (modelId: string, rate: number): QualityResult[] =>
  (['instruction', 'reasoning', 'coding', 'structured', 'extraction', 'context'] as const).flatMap((category) => [0, 1, 2, 3, 4].map((i) => ({
    testId: `${modelId}-${category}-${i}`, category, weight: 1, pass: i < rate * 5, score: i < rate * 5 ? 1 : 0, detail: ''
  })))
/** Give every candidate measured quality (rate per model id; default 0.6 = Q 60). */
export const withQuality = (list: CandidateInput[], rate: number | ((modelId: string) => number) = 0.6): CandidateInput[] =>
  list.map((c) => ({ ...c, quality: q5(c.model.id, typeof rate === 'number' ? rate : rate(c.model.id)) }))
