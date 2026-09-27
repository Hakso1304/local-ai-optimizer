import { describe, expect, it } from 'vitest'
import type { SessionEvent, SessionRequest } from '../../src/shared/bench-events'
import type { BenchmarkRunResult, ModelMeta, QualityResult, Recommendation } from '../../src/shared/bench-types'
import type { SystemProfile } from '../../src/shared/types'
import type { LoadConfig, PromptResult } from '../../src/core/runtimes/types'
import type { ExitInfo } from '../../src/core/runtimes/llamacpp'
import type { TelemetrySample } from '../../src/core/telemetry/sampler'
import { runSession, type RunDetail, type SessionBackend, type SessionDeps, type SessionStorage } from '../../src/core/benchmark/session'
import { ladderPrompt } from '../../src/core/benchmark/prompts'
import { ConfigDriftError } from '../../src/core/runtimes/llamacpp'
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
interface Step { load?: 'oom' | 'drift' | 'slow'; prompt?: 'device_lost' | 'timeout'; warmupBlocks?: boolean; decode?: number; prefill?: number; hook?: () => void }

function fakeBackend(script: (ctx: number, cfg: LoadConfig) => Step) {
  let ctx = 0
  let cfg: LoadConfig | null = null
  const calls = { loads: [] as LoadConfig[], templates: 0, cancels: 0, unloads: 0, order: [] as string[] }
  let onCancel: (() => void) | null = null
  const b: { -readonly [K in keyof SessionBackend]: SessionBackend[K] } & { calls: typeof calls } = {
    calls,
    pid: undefined as number | undefined,
    lastExit: null as ExitInfo | null,
    async loadModel(c) {
      calls.loads.push(c)
      ctx = c.contextSize
      cfg = c
      b.lastExit = null
      const s = script(ctx, c)
      if (s.load === 'oom') {
        b.lastExit = { code: 1, reason: 'oom', tail: ['ggml_vulkan: ErrorOutOfDeviceMemory'] }
        throw new Error('llama-server exited with code 1 (oom)')
      }
      b.pid = 1000 + ctx
      if (s.load === 'drift') throw new ConfigDriftError(`config_drift: requested -c ${ctx} but server serves n_ctx 16384`)
      if (s.load === 'slow') await new Promise((r) => setTimeout(r, 150)) // pid exists, /health not yet ok
      calls.order.push('load-resolved')
      return { loadTimeMs: 900, declared: { layersOffloaded: 33, layersTotal: 33, modelBufferMiB: {}, kvBufferMiB: {}, computeBufferMiB: {} } }
    },
    async unloadModel() { calls.unloads++; b.pid = undefined },
    async warmup() {
      if (!script(ctx, cfg!).warmupBlocks) return
      const cancelled = await new Promise<boolean>((r) => { onCancel = () => r(true); setTimeout(() => r(false), 1000) })
      if (cancelled) throw new Error('warmup failed: cancelled')
    },
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
    async cancel() { calls.cancels++; onCancel?.() }
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

  it('resume re-runs cancelled and RAM-skipped steps instead of stopping on them', async () => {
    const ac = new AbortController()
    const first = await run((ctx) => (ctx === 8192 ? { hook: () => ac.abort() } : {}), {}, { signal: ac.signal })
    expect(first.s.runs.at(-1)).toMatchObject({ ctx: 8192, status: 'cancelled' })
    const { backend, rec } = await run(() => ({}), { resumeSessionId: 's1' }, {}, first.s.runs) // includes the cancelled 8K row
    expect(ctxOf(backend)).toEqual([8192, 16384, 32768])
    expect(rec?.best).not.toBeNull()
    const skipped = await run(() => ({}), { ladder: [2048] }, { readRamAvailableBytes: () => 1 * GiB })
    const again = await run(() => ({}), { resumeSessionId: 's1', ladder: [2048] }, {}, skipped.s.runs)
    expect(ctxOf(again.backend)).toEqual([2048])
  })

  it('heavy mode: a 16 GiB model gets a partial-offload ladder; -nkvo reaches the launch args', async () => {
    const m27 = { ...model, id: 'C:/models/q27b.gguf', fileBytes: Math.round(16.1 * GiB), layers: 64, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064 }
    const normal = await run(() => ({}), { workload: 'max_quality', modelIds: [m27.id], ladder: [2048] }, { models: [m27] })
    expect(normal.backend.calls.loads).toEqual([]) // does not fit → nothing to run
    const h = await run(() => ({}), { workload: 'max_quality', modelIds: [m27.id], ladder: [2048], heavyMode: true }, { models: [m27] })
    const args = h.backend.calls.loads.map((l) => `${l.gpuLayers} ${l.device} ${(l.extraArgs ?? []).join(' ')}`)
    expect(args.some((a) => a.endsWith('-nkvo'))).toBe(true)
    expect(h.backend.calls.loads.every((l) => l.gpuLayers < 64)).toBe(true)
    expect(h.backend.calls.loads.some((l) => l.gpuLayers === 0 && l.device === 'none')).toBe(true)
  })

  it('pause mid-ladder: finishes the current step, unloads, status paused; resume continues from the next step', async () => {
    const pause = new AbortController()
    const first = await run((ctx) => (ctx === 8192 ? { hook: () => pause.abort() } : {}), {}, { pauseSignal: pause.signal })
    expect(first.rec).toBeNull()
    expect(first.s.runs.map((r) => [r.ctx, r.status])).toEqual([[2048, 'pass'], [4096, 'pass'], [8192, 'pass']]) // 8K completed, not cut
    expect(first.backend.calls.cancels).toBe(0) // never mid-request
    expect(first.backend.pid).toBeUndefined()
    expect(first.s.status.at(-1)).toBe('paused')
    expect(first.events.at(-1)?.type).toBe('session:paused')
    expect(first.events.find((e) => e.type === 'candidate:done')).toMatchObject({ status: 'paused' })
    const resumed = await run(() => ({}), { resumeSessionId: 's1' }, {}, first.s.runs)
    expect(ctxOf(resumed.backend)).toEqual([16384, 32768])
    expect(resumed.rec?.best).not.toBeNull()
  })

  it('retryFailed re-runs the failed step only (not config_drift); without it the failure is reused', async () => {
    const first = await run((ctx) => (ctx === 8192 ? { prompt: 'timeout' } : {}))
    expect(first.s.runs.at(-1)).toMatchObject({ ctx: 8192, status: 'timeout' })
    const plain = await run(() => ({}), { resumeSessionId: 's1' }, {}, first.s.runs)
    expect(ctxOf(plain.backend)).toEqual([]) // reuses the timeout → ladder stops there again
    const retry = await run(() => ({}), { resumeSessionId: 's1', retryFailed: true }, {}, first.s.runs)
    expect(ctxOf(retry.backend)).toEqual([8192, 16384, 32768]) // 2K/4K reused, 8K retried, ladder continues
    const drift = await run((ctx) => (ctx === 4096 ? { load: 'drift' } : {}))
    const again = await run(() => ({}), { resumeSessionId: 's1', retryFailed: true }, {}, drift.s.runs)
    expect(ctxOf(again.backend)).toEqual([]) // same config → same drift: not retried
  })

  it('rerunConfigIds re-runs only the selected configuration', async () => {
    const first = await run(() => ({}), { workload: 'long_context_coding' })
    const ids = [...new Set(first.s.runs.map((r) => r.configId))]
    expect(ids).toHaveLength(2)
    const re = await run(() => ({}), { workload: 'long_context_coding', resumeSessionId: 's1', rerunConfigIds: [ids[1]] }, {}, first.s.runs)
    expect(re.backend.calls.loads.every((l) => (l.extraArgs ?? []).includes('q8_0') === ids[1].includes('kv=q8_0'))).toBe(true)
    expect(re.backend.calls.loads.length).toBe(first.s.runs.filter((r) => r.configId === ids[1]).length)
    expect(re.s.runs.length).toBe(first.s.runs.length + re.backend.calls.loads.length) // new rows appended; reads keep the last
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
    const { s, backend, rec } = await run(() => ({}), {}, { readRamAvailableBytes: () => 1 * GiB })
    expect(backend.calls.loads).toEqual([])
    expect(s.runs[0]).toMatchObject({ status: 'fail', failureKind: 'skipped_memory' })
    // Every step skipped by the RAM guard → say so, with the guard's arithmetic (not just "no successful runs").
    expect(rec?.best).toBeNull()
    expect(rec?.reasons[0]).toBe('No recommendation: nothing was run — the RAM guard skipped every step before loading')
    expect(rec?.reasons[1]).toMatch(/^est\. RAM 5\.1 GiB > available 1\.0 GiB − floor 2\.5 GiB$/)
    expect(rec?.excluded[0].reasons[0]).toMatch(/^2K: run fail \(skipped_memory\): est\. RAM/)
  })
  it('guard reason wins over the "cancelled" error its own cancel() causes (guard_abort, not request_error)', async () => {
    const spill = { ...sample(2048), procVramSharedBytes: 3 * GiB }
    const { s } = await run(() => ({ warmupBlocks: true }), { ladder: [2048] }, {
      startSampler: () => ({ samples: [spill], unavailable: {}, stop: () => [spill] }), config: { guardPollMs: 5 }
    })
    expect(s.runs[0]).toMatchObject({ status: 'fail', failureKind: 'guard_abort' })
    expect(s.details[0].reason).toBe('shared GPU memory spill 3.0 GiB exceeded the abort limit')
  })

  it('RAM floor credits mmap pages of GPU-offloaded weights (not pressure); trips when no credit applies', async () => {
    const low = { ...sample(2048), ramAvailBytes: 1 * GiB }
    const deps = { startSampler: () => ({ samples: [low], unavailable: {}, stop: () => [low] }), config: { guardPollMs: 5 } }
    // Full offload of the 4.6 GiB 8B: 1 GiB available + 4.6 GiB reclaimable file pages > 2.5 GiB floor → no abort.
    const full = await run(() => ({ warmupBlocks: true }), { ladder: [2048] }, deps)
    expect(full.s.runs[0].status).toBe('pass')
    // A tiny model gets no meaningful credit → the same 1 GiB trips the floor.
    const tiny = { ...model, fileBytes: 100 * 1024 ** 2 }
    const t = await run(() => ({ warmupBlocks: true }), { ladder: [2048] }, { ...deps, models: [tiny] })
    expect(t.s.runs[0]).toMatchObject({ status: 'fail', failureKind: 'guard_abort' })
    expect(t.s.details[0].reason).toBe('RAM available 1.0 GiB fell below the floor')
  })

  it('ConfigDriftError from loadModel → fail/config_drift', async () => {
    const { s } = await run((ctx) => (ctx === 4096 ? { load: 'drift' } : {}))
    expect(s.runs.map((r) => [r.ctx, r.status, r.failureKind])).toEqual([[2048, 'pass', null], [4096, 'fail', 'config_drift']])
    expect(s.details.at(-1)!.reason).toMatch(/serves n_ctx 16384/)
  })

  it('starts the sampler during load, as soon as the new pid exists', async () => {
    let backendRef: ReturnType<typeof fakeBackend> | null = null
    await run(() => ({}), { ladder: [2048] }, {
      startSampler: (pid) => { backendRef!.calls.order.push(`sampler:${pid}`); const xs = [sample(2048)]; return { samples: xs, unavailable: {}, stop: () => xs } },
      backend: () => (backendRef = fakeBackend(() => ({ load: 'slow' })))
    })
    expect(backendRef!.calls.order).toEqual(['sampler:3048', 'load-resolved'])
  })

  it('short step: waits for the first real sample; none at all → peaks unavailable (never fabricated)', async () => {
    const late = await run(() => ({}), { ladder: [2048] }, {
      startSampler: () => { const xs: TelemetrySample[] = []; setTimeout(() => xs.push(sample(2048)), 150); return { samples: xs, unavailable: {}, stop: () => xs } },
      config: { firstSampleWaitMs: 2000 }
    })
    expect(late.s.runs[0].peakVramBytes).toMatchObject({ kind: 'measured' })
    const none = await run(() => ({}), { ladder: [2048] }, {
      startSampler: () => ({ samples: [], unavailable: {}, stop: () => [] }), config: { firstSampleWaitMs: 100 }
    })
    expect(none.s.runs[0].peakVramBytes).toMatchObject({ value: null, kind: 'unavailable' })
    expect(none.s.runs[0].status).toBe('pass')
  })
})
