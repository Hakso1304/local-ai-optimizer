import { describe, expect, it } from 'vitest'
import type { SessionEvent, SessionRequest } from '../../src/shared/bench-events'
import type { BenchmarkRunResult, ModelMeta, QualityResult, Recommendation } from '../../src/shared/bench-types'
import type { SystemProfile } from '../../src/shared/types'
import type { LoadConfig, PromptResult } from '../../src/core/runtimes/types'
import type { ExitInfo } from '../../src/core/runtimes/llamacpp'
import type { TelemetrySample } from '../../src/core/telemetry/sampler'
import { runSession, type RunDetail, type SessionBackend, type SessionDeps, type SessionStorage } from '../../src/core/benchmark/session'
import { ladderPrompt } from '../../src/core/benchmark/prompts'
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

/** Per-ctx script: load failure, prompt failure, or rates. hook runs inside the measured prompt. */
interface Step { load?: 'oom'; prompt?: 'device_lost' | 'timeout'; decode?: number; prefill?: number; hook?: () => void }

function fakeBackend(script: (ctx: number, cfg: LoadConfig) => Step) {
  let ctx = 0
  let cfg: LoadConfig | null = null
  const calls = { loads: [] as LoadConfig[], templates: 0, cancels: 0, unloads: 0 }
  const b: { -readonly [K in keyof SessionBackend]: SessionBackend[K] } & { calls: typeof calls } = {
    calls,
    pid: undefined as number | undefined,
    lastExit: null as ExitInfo | null,
    async loadModel(c) {
      calls.loads.push(c)
      ctx = c.contextSize
      cfg = c
      b.lastExit = null
      if (script(ctx, c).load === 'oom') {
        b.lastExit = { code: 1, reason: 'oom', tail: ['ggml_vulkan: ErrorOutOfDeviceMemory'] }
        throw new Error('llama-server exited with code 1 (oom)')
      }
      b.pid = 1000 + ctx
      return { loadTimeMs: 900, declared: { layersOffloaded: 33, layersTotal: 33, modelBufferMiB: {}, kvBufferMiB: {}, computeBufferMiB: {} } }
    },
    async unloadModel() { calls.unloads++; b.pid = undefined },
    async warmup() {},
    async runPrompt(req): Promise<PromptResult> {
      const s = script(ctx, cfg!)
      const base = { promptTokens: 100, prefillMs: 10, decodeTokens: req.maxTokens, decodeMs: 1000, text: 'BANANA', stopType: 'limit' }
      const isMeasured = req.prompt === ladderPrompt(ctx)
      if (isMeasured) s.hook?.()
      if (isMeasured && s.prompt === 'device_lost') {
        b.lastExit = { code: 3221225477, reason: 'device_lost', tail: ['vk::DeviceLostError'] }
        return { ...base, ttftMs: null, prefillTps: null, decodeTps: null, totalMs: 5, timedOut: false, error: 'server exited (device_lost)' }
      }
      if (isMeasured && s.prompt === 'timeout') return { ...base, ttftMs: null, prefillTps: null, decodeTps: null, totalMs: 5, timedOut: true, error: 'timed out' }
      return { ...base, ttftMs: 100 + ctx / 10, prefillTps: s.prefill ?? 3000, decodeTps: s.decode ?? 90, totalMs: 2000, timedOut: false, error: null }
    },
    async applyTemplate(m) { calls.templates++; return m.map((x) => x.content).join('\n') },
    async cancel() { calls.cancels++ }
  }
  return b
}

const sample = (ctx: number): TelemetrySample => ({
  ts: ctx, cpuPct: 10, ramAvailBytes: 20 * GiB, gpuUtilPct: 95, vramDedicatedBytes: 9 * GiB, vramSharedBytes: GiB,
  procRamPrivateBytes: GiB, procVramDedicatedBytes: 5 * GiB + ctx * 131072, procVramSharedBytes: 10 * 1024 ** 2
})

function memStorage(seed: BenchmarkRunResult[] = []) {
  const s = {
    status: [] as string[],
    runs: [...seed] as BenchmarkRunResult[],
    details: [] as RunDetail[],
    quality: [] as { modelId: string; configId: string; ctx: number; results: QualityResult[] }[],
    recs: [] as Recommendation[]
  }
  const storage: SessionStorage = {
    createSession: () => 's1',
    setSessionStatus: (_id, st) => { s.status.push(st) },
    listRuns: () => [...s.runs],
    saveRun: (_id, run, detail) => { s.runs.push(run); s.details.push(detail) },
    listQuality: (_id, modelId) => s.quality.find((q) => q.modelId === modelId)?.results ?? [],
    saveQuality: (_id, modelId, configId, ctx, results) => { s.quality.push({ modelId, configId, ctx, results }) },
    saveRecommendation: (_id, rec) => { s.recs.push(rec) }
  }
  return { s, storage }
}

const passAll: SessionDeps['evaluate'] = async (t) => ({ testId: t.id, category: t.category, weight: t.weight, pass: true, score: 1, detail: '' })

async function run(script: (ctx: number, cfg: LoadConfig) => Step, req: Partial<SessionRequest> = {}, extra: Partial<SessionDeps> = {}, seed: BenchmarkRunResult[] = []) {
  const backend = fakeBackend(script)
  const { s, storage } = memStorage(seed)
  const events: SessionEvent[] = []
  let t = 0
  const rec = await runSession(
    { workload: 'coding', modelIds: [model.id], runQuality: false, ...req },
    { backend: () => backend, startSampler: (pid) => { const xs = [sample(pid - 1000)]; return { samples: xs, unavailable: {}, stop: () => xs } },
      storage, machine, gpuDevice: 'Vulkan0', models: [model], clock: { now: () => t++ }, evaluate: passAll, ...extra },
    (e) => events.push(e)
  )
  return { rec, events, s, backend }
}

const ctxOf = (b: ReturnType<typeof fakeBackend>) => b.calls.loads.map((l) => l.contextSize)

describe('runSession', () => {
  it('runs the full ladder, one server launch per step, with explicit device/ctx args', async () => {
    const { rec, s, backend } = await run(() => ({}))
    expect(ctxOf(backend)).toEqual([2048, 4096, 8192, 16384, 32768])
    expect(backend.calls.loads[0]).toMatchObject({ modelPath: model.id, device: 'Vulkan0', gpuLayers: 999, threads: 8 })
    expect(backend.calls.loads[0].extraArgs).toEqual(['-ub', '512', '-fa', 'on'])
    expect(s.runs.every((r) => r.status === 'pass')).toBe(true)
    expect(s.runs[0].peakVramBytes).toMatchObject({ kind: 'measured' })
    expect(s.runs[0].decodeTps).toMatchObject({ value: 90, kind: 'measured' })
    expect(rec?.best?.configId).toBe(`${model.id}|ngl=all|kv=f16|t=8`)
    expect(s.status).toEqual(['running', 'done'])
  })

  it('stops the ladder at OOM and records the step as fail/oom with the stderr tail', async () => {
    const { s, backend, events } = await run((ctx) => (ctx === 16384 ? { load: 'oom' } : {}))
    expect(ctxOf(backend)).toEqual([2048, 4096, 8192, 16384])
    const last = s.runs.at(-1)!
    expect(last).toMatchObject({ ctx: 16384, status: 'fail', failureKind: 'oom' })
    expect(s.details.at(-1)!.stderrTail).toContain('ggml_vulkan: ErrorOutOfDeviceMemory')
    const done = events.find((e) => e.type === 'candidate:done')!
    expect(done).toMatchObject({ status: 'done', reason: 'stopped after fail (oom) at 16384' })
  })

  it('a cliff-degraded step does not stop the ladder; the next step still runs', async () => {
    const { s, backend, events } = await run((ctx) => ({ decode: ctx >= 8192 ? 30 : 90 }))
    expect(ctxOf(backend)).toEqual([2048, 4096, 8192, 16384]) // 8K cliff, 16K runs, then 2 consecutive degraded → stop
    const verdicts = events.filter((e) => e.type === 'step:done').map((e) => (e as { verdict: string }).verdict)
    expect(verdicts).toEqual(['pass', 'pass', 'degraded', 'degraded'])
    expect(s.runs.map((r) => r.status)).toEqual(['pass', 'pass', 'pass', 'pass'])
  })

  it('device loss is recorded and ends the ladder', async () => {
    const { s } = await run((ctx) => (ctx === 4096 ? { prompt: 'device_lost' } : {}))
    expect(s.runs.map((r) => [r.ctx, r.status, r.failureKind])).toEqual([[2048, 'pass', null], [4096, 'fail', 'device_lost']])
  })

  it('timeout is recorded as timeout/req_timeout', async () => {
    const { s } = await run((ctx) => (ctx === 8192 ? { prompt: 'timeout' } : {}))
    expect(s.runs.at(-1)).toMatchObject({ ctx: 8192, status: 'timeout', failureKind: 'req_timeout' })
  })

  it('cancel mid-ladder → session:cancelled, partial results persisted, no recommendation', async () => {
    const ac = new AbortController()
    const { rec, s, events, backend } = await run((ctx) => (ctx === 8192 ? { hook: () => ac.abort() } : {}), {}, { signal: ac.signal })
    expect(rec).toBeNull()
    expect(backend.calls.cancels).toBeGreaterThan(0)
    expect(s.runs.map((r) => [r.ctx, r.status])).toEqual([[2048, 'pass'], [4096, 'pass'], [8192, 'cancelled']])
    expect(events.at(-1)?.type).toBe('session:cancelled')
    expect(s.status.at(-1)).toBe('cancelled')
    expect(s.recs).toEqual([])
    expect(backend.pid).toBeUndefined() // unloaded
  })

  it('resume skips (configId, ctx) steps already persisted', async () => {
    const ac = new AbortController()
    const first = await run((ctx) => (ctx === 8192 ? { hook: () => ac.abort() } : {}), {}, { signal: ac.signal })
    const kept = first.s.runs.filter((r) => r.status === 'pass')
    const { backend, rec } = await run(() => ({}), { resumeSessionId: 's1' }, {}, kept)
    expect(ctxOf(backend)).toEqual([8192, 16384, 32768])
    expect(rec?.best?.score.referenceCtx).toBe(16384)
    expect(rec?.best?.score.recommendedCtx).toBe(32768) // coding: largest step within 15 s TTFT, ≤ 32K
  })

  it('emits events in order', async () => {
    const { events } = await run((ctx) => (ctx > 4096 ? { load: 'oom' } : {}), { ladder: [2048, 4096, 8192] })
    const types = events.map((e) => (e.type === 'phase' ? `phase:${e.phase}` : e.type)).filter((t) => t !== 'telemetry' && t !== 'log')
    expect(types).toEqual([
      'session:started', 'candidate:started', 'phase:ladder',
      'step:started', 'phase:load', 'phase:warmup', 'phase:measure', 'token-rate', 'token-rate', 'step:done',
      'step:started', 'phase:load', 'phase:warmup', 'phase:measure', 'token-rate', 'token-rate', 'step:done',
      'step:started', 'phase:load', 'step:done',
      'candidate:done', 'session:done'
    ])
    expect(events.every((e) => e.sessionId === 's1')).toBe(true)
  })

  it('runQuality=false skips quality; true runs the suite once per model at min(target, ceiling)', async () => {
    const off = await run(() => ({}))
    expect(off.backend.calls.templates).toBe(0)
    expect(off.s.quality).toEqual([])
    const on = await run(() => ({}), { runQuality: true })
    expect(on.s.quality).toHaveLength(1)
    expect(on.s.quality[0]).toMatchObject({ ctx: 16384 })
    expect(on.s.quality[0].results).toHaveLength(17)
    expect(on.rec?.best?.score.breakdown.find((b) => b.component === 'quality')?.input.kind).toBe('measured')
  })

  it('produces a recommendation from a 2-candidate run (f16 vs q8_0 KV for long context)', async () => {
    const { rec, s } = await run((_ctx, c) => ({ decode: c.extraArgs?.includes('q8_0') ? 70 : 90 }), { workload: 'long_context_coding' })
    expect(new Set(s.runs.map((r) => r.configId)).size).toBe(2)
    expect(rec?.ranked).toHaveLength(2)
    expect(rec?.best?.configId).toBe(`${model.id}|ngl=all|kv=f16|t=8`)
    expect(rec?.alternatives.fastest).toBe(`${model.id}|ngl=all|kv=f16|t=8`)
  })

  it('skips a step whose RAM estimate breaks the live floor (A25) without loading it', async () => {
    const { s, backend } = await run(() => ({}), {}, { readRamAvailableBytes: () => 1 * GiB })
    expect(backend.calls.loads).toEqual([])
    expect(s.runs[0]).toMatchObject({ status: 'fail', failureKind: 'skipped_memory' })
  })
})
