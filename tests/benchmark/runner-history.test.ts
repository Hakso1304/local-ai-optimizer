// Executed re-review scenarios (docs/review-w4i-2026-09-27.md) that touch the RUNNER's producer path.
// Formerly `it.fails` (G06/G09); fixed in #1's re-review round.
// Minimal fake backend/storage copied from session.test.ts (kept separate: new test files only).
import { describe, expect, it } from 'vitest'
import type { SessionEvent, SessionRequest } from '../../src/shared/bench-events'
import type { BenchmarkRunResult, ModelMeta, Recommendation } from '../../src/shared/bench-types'
import type { SystemProfile } from '../../src/shared/types'
import type { LoadConfig, PromptResult } from '../../src/core/runtimes/types'
import type { ExitInfo } from '../../src/core/runtimes/llamacpp'
import type { TelemetrySample } from '../../src/core/telemetry/sampler'
import { runSession, type SessionBackend, type SessionDeps, type SessionStorage } from '../../src/core/benchmark/session'
import { ladderPrompt } from '../../src/core/benchmark/prompts'
import { speedIneligible } from '../../src/core/interpret/verdicts'
import { load } from '../scoring/helpers'

const GiB = 1024 ** 3
const model: ModelMeta = { ...load('sweep-smooth.json').models[0], id: 'C:/models/llama8b.gguf', ctxTrain: 32768 }
const machine: SystemProfile = {
  scannedAt: 't0',
  os: { value: { name: 'Windows 11', version: '10', build: '26200' }, status: 'available', source: 'test' },
  cpu: { value: { model: 'Ryzen', physicalCores: 8, logicalCores: 16 }, status: 'available', source: 'test' },
  ram: { value: { totalBytes: 31 * GiB, availableBytes: 20 * GiB }, status: 'available', source: 'test' },
  gpus: { value: [{ name: 'RX 9070 XT', vendor: 'amd', pnpDeviceId: 'x', driverVersion: null, isIntegrated: false, dedicatedVramBytes: { value: 17095983104, status: 'available', source: 'registry' } }], status: 'available', source: 'test' },
  cuda: { value: { available: false }, status: 'unsupported', source: 'test' },
  disks: { value: [], status: 'available', source: 'test' },
  runtimes: []
}

type Step = { prompt?: 'device_lost' }
function fakeBackend(script: (ctx: number) => Step) {
  let ctx = 0
  const b: { -readonly [K in keyof SessionBackend]: SessionBackend[K] } = {
    pid: undefined as number | undefined,
    lastExit: null as ExitInfo | null,
    async loadModel(c: LoadConfig) {
      ctx = c.contextSize
      b.lastExit = null
      b.pid = 1000 + ctx
      return { loadTimeMs: 900, declared: { layersOffloaded: 33, layersTotal: 33, modelBufferMiB: {}, kvBufferMiB: {}, computeBufferMiB: {} } }
    },
    async unloadModel() { b.pid = undefined },
    async warmup() {},
    async runPrompt(req): Promise<PromptResult> {
      const base = { promptTokens: 100, prefillMs: 10, decodeTokens: req.maxTokens, decodeMs: 1000, text: 'BANANA', stopType: 'eos' }
      if (req.prompt === ladderPrompt(ctx) && script(ctx).prompt === 'device_lost') {
        b.lastExit = { code: 3221225477, reason: 'device_lost', tail: ['vk::DeviceLostError'] }
        return { ...base, ttftMs: null, prefillTps: null, decodeTps: null, totalMs: 5, timedOut: false, error: 'server exited (device_lost)' }
      }
      return { ...base, ttftMs: 100 + ctx / 10, prefillTps: 3000, decodeTps: 90, totalMs: 2000, timedOut: false, error: null }
    },
    async applyTemplate(m) { return m.map((x) => x.content).join('\n') },
    async cancel() {}
  }
  return b
}

const sample = (ctx: number): TelemetrySample => ({
  ts: ctx, cpuPct: 10, ramAvailBytes: 20 * GiB, gpuUtilPct: 95, vramDedicatedBytes: 9 * GiB, vramSharedBytes: GiB,
  procRamPrivateBytes: GiB, procVramDedicatedBytes: 5 * GiB + ctx * 131072, procVramSharedBytes: 10 * 1024 ** 2
})

/** Append-only store: every attempt persists (a retry adds a row; nothing is overwritten), like db.ts. */
function memStorage(seed: BenchmarkRunResult[] = []) {
  const s = { runs: [...seed] as BenchmarkRunResult[], recs: [] as Recommendation[] }
  const storage: SessionStorage = {
    createSession: () => 's1',
    setSessionStatus: () => {},
    listRuns: () => [...s.runs],
    saveRun: (_id, run) => { s.runs.push(run) },
    listQuality: () => [],
    saveQuality: () => {},
    saveRecommendation: (_id, rec) => { s.recs.push(rec) }
  }
  return { s, storage }
}

async function run(script: (ctx: number) => Step, req: Partial<SessionRequest>, seed: BenchmarkRunResult[] = []) {
  const { s, storage } = memStorage(seed)
  const events: SessionEvent[] = []
  let t = 0
  const deps: SessionDeps = {
    backend: () => fakeBackend(script), startSampler: (pid) => { const xs = [sample(pid - 1000)]; return { samples: xs, unavailable: {}, stop: () => xs } },
    storage, machine, gpuDevice: 'Vulkan0', models: [model], clock: { now: () => t++ },
    evaluate: async (x) => ({ testId: x.id, category: x.category, weight: x.weight, pass: true, score: 1, detail: '' })
  }
  const rec = await runSession({ workload: 'general_chat', modelIds: [model.id], runQuality: false, ...req }, deps, (e) => events.push(e))
  return { rec, s, events }
}

const texts = (r: Recommendation | null) => (r?.insights ?? []).map((i) => i.text).join('\n')

describe('G06 — live/resumed recommendations see every persisted attempt (I-6.3)', () => {
  const ladder = [2048, 4096, 8192]

  it('control: the first (live) session reports the device loss it just recorded', async () => {
    const first = await run((ctx) => (ctx === 4096 ? { prompt: 'device_lost' } : {}), { ladder })
    expect(first.s.runs.some((r) => r.failureKind === 'device_lost')).toBe(true)
    expect(texts(first.rec)).toMatch(/device_lost/)
  })

  it('a resumed session whose retry succeeded still reports the superseded device_lost as critical', async () => {
    const first = await run((ctx) => (ctx === 4096 ? { prompt: 'device_lost' } : {}), { ladder })
    const resumed = await run(() => ({}), { ladder, resumeSessionId: 's1', retryFailed: true }, first.s.runs)
    // Precondition: the history really holds both attempts at 4K (the failure and the successful retry).
    expect(resumed.s.runs.filter((r) => r.ctx === 4096).map((r) => r.failureKind ?? r.status)).toEqual(expect.arrayContaining(['device_lost', 'pass']))
    // The saved recommendation must be computed from ALL persisted attempts, not only the latest row per step.
    const insight = (resumed.rec?.insights ?? []).find((i) => /device_lost/.test(i.text))
    expect(insight, texts(resumed.rec)).toBeDefined()
    expect(insight!.severity).toBe('critical')
  })
})

describe('G09 — versionless rows are not speed-eligible (I-6.0)', () => {
  const row = (versions?: BenchmarkRunResult['versions']): BenchmarkRunResult => ({
    configId: 'c', ctx: 2048, promptTokens: 100, status: 'pass', failureKind: null, warm: true,
    loadTimeMs: { value: 900, kind: 'measured' }, ttftMs: { value: 300, kind: 'measured' }, prefillTps: { value: 3000, kind: 'measured' },
    decodeTps: { value: 90, kind: 'measured' }, totalMs: { value: 2000, kind: 'measured' }, peakVramBytes: { value: 5 * GiB, kind: 'measured' },
    peakSharedGpuBytes: { value: 0, kind: 'measured' }, peakRamBytes: { value: GiB, kind: 'measured' },
    avgGpuUtil: { value: 90, kind: 'measured' }, avgCpuUtil: { value: 10, kind: 'measured' },
    ...(versions ? { versions } : {})
  } as BenchmarkRunResult)

  it('control: a row with the session version is eligible', () => {
    expect(speedIneligible(row({ benchmark: 'bench-1', prompts: 'ladder-1', quality: 'qb-2.0.0', runtime: 'b1' } as BenchmarkRunResult['versions']), 'bench-1/ladder-1')).toBeNull()
  })

  it('a completed row with no versions has no version proof and must not enter speed scoring', () => {
    expect(speedIneligible(row(undefined), 'bench-1/ladder-1')).not.toBeNull()
  })

  it('a resumed session whose stored rows lost their versions must not confirm a winner from them', async () => {
    const ladder = [2048, 4096, 8192]
    const first = await run(() => ({}), { ladder, runQuality: true })
    // Control: with versions the same evidence DOES confirm a winner (otherwise this test would prove nothing).
    expect(first.rec?.best?.configId, texts(first.rec)).toBeDefined()
    const stripped = first.s.runs.map(({ versions: _v, ...r }) => r as BenchmarkRunResult)
    const resumed = await run(() => ({}), { ladder, runQuality: true, resumeSessionId: 's1' }, stripped)
    expect(resumed.s.runs.every((r) => !r.versions)).toBe(true) // reused, not re-measured
    // Every speed row is versionless → no version proof → no confirmed winner (it may be provisional).
    expect(resumed.rec?.best ?? null).toBeNull()
  })
})
