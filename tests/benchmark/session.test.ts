import { describe, expect, it } from 'vitest'
import type { SessionEvent, SessionRequest } from '../../src/shared/bench-events'
import type { BenchmarkRunResult, ModelMeta, QualityResult, Recommendation } from '../../src/shared/bench-types'
import type { SystemProfile } from '../../src/shared/types'
import type { LoadConfig, PromptRequest, PromptResult } from '../../src/core/runtimes/types'
import type { ExitInfo } from '../../src/core/runtimes/llamacpp'
import type { TelemetrySample } from '../../src/core/telemetry/sampler'
import { runSession, type RunDetail, type SessionBackend, type SessionDeps, type SessionStorage } from '../../src/core/benchmark/session'
import { ladderPrompt } from '../../src/core/benchmark/prompts'
import { generateCandidates, machineFromProfile } from '../../src/core/benchmark/candidates'
import { WORKLOADS } from '../../src/core/scoring/workloads'
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
interface Step { noTimings?: boolean; hostMiB?: number; load?: 'oom' | 'drift' | 'slow'; prompt?: 'device_lost' | 'timeout'; warmupBlocks?: boolean; decode?: number; prefill?: number; hook?: () => void }

function fakeBackend(script: (ctx: number, cfg: LoadConfig) => Step) {
  let ctx = 0
  let cfg: LoadConfig | null = null
  const calls = { loads: [] as LoadConfig[], templates: 0, templateOpts: [] as unknown[], cancels: 0, unloads: 0, order: [] as string[] }
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
      return { loadTimeMs: 900, declared: { layersOffloaded: 33, layersTotal: 33, modelBufferMiB: (s.hostMiB ? { Vulkan0: 9000, CPU: s.hostMiB } : {}) as Record<string, number>, kvBufferMiB: {}, computeBufferMiB: {} } }
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
      if (s.noTimings) return { ...base, promptTokens: null, prefillMs: null, decodeTokens: null, decodeMs: null, streamedTokens: 100, ttftMs: 1000, prefillTps: null, decodeTps: null, totalMs: 3000, timedOut: false, error: null } as PromptResult
      return { ...base, ttftMs: 100 + ctx / 10, prefillTps: s.prefill ?? 3000, decodeTps: s.decode ?? 90, totalMs: 2000, timedOut: false, error: null }
    },
    async applyTemplate(m, opts) { calls.templates++; calls.templateOpts.push(opts); return m.map((x) => x.content).join('\n') },
    async cancel() { calls.cancels++; onCancel?.() }
  }
  return b
}

const sample = (ctx: number): TelemetrySample => ({
  ts: ctx, cpuPct: 10, ramAvailBytes: 20 * GiB, gpuUtilPct: 95, vramDedicatedBytes: 9 * GiB, vramSharedBytes: GiB,
  procRamPrivateBytes: GiB, procVramDedicatedBytes: 5 * GiB + ctx * 131072, procVramSharedBytes: 10 * 1024 ** 2
})

function memStorage(seed: BenchmarkRunResult[] = [], seedQuality: { modelId: string; configId: string; ctx: number; results: QualityResult[] }[] = []) {
  const s = {
    status: [] as string[],
    runs: [...seed] as BenchmarkRunResult[],
    details: [] as RunDetail[],
    quality: [...seedQuality] as { modelId: string; configId: string; ctx: number; results: QualityResult[] }[],
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

async function run(script: (ctx: number, cfg: LoadConfig) => Step, req: Partial<SessionRequest> = {}, extra: Partial<SessionDeps> = {}, seed: BenchmarkRunResult[] = [], seedQuality: Parameters<typeof memStorage>[1] = []) {
  const backend = fakeBackend(script)
  const { s, storage } = memStorage(seed, seedQuality)
  const events: SessionEvent[] = []
  let t = 0
  const rec = await runSession(
    { workload: 'general_chat', modelIds: [model.id], runQuality: false, ...req },
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
    expect(rec?.best?.score.referenceCtx).toBe(8192) // general chat target
    expect(rec?.best?.score.recommendedCtx).toBe(16384) // general chat: largest passing rung ≤ 16K within the 8 s TTFT
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
    const h = await run(() => ({}), { workload: 'long_context_coding', modelIds: [m27.id], ladder: [2048], heavyMode: true }, { models: [m27] })
    const args = h.backend.calls.loads.map((l) => `${l.gpuLayers} ${l.device} ${(l.extraArgs ?? []).join(' ')}`)
    expect(args.some((a) => / -nkvo( |$)/.test(a))).toBe(true)
    // Run order: KV-on-GPU rungs first, the -nkvo rung after them.
    const kinds = h.backend.calls.loads.map((l) => ((l.extraArgs ?? []).includes('-nkvo') ? 'nkvo' : 'gpu'))
    expect(kinds.lastIndexOf('gpu')).toBeLessThan(kinds.indexOf('nkvo'))
    // Partial-offload (heavy) loads run without mmap; the 8B-style full offload keeps the default.
    expect(h.backend.calls.loads.every((l) => (l.extraArgs ?? []).join(' ').includes('-lm none'))).toBe(true)
    const full = await run(() => ({}), { ladder: [2048] })
    expect(full.backend.calls.loads.every((l) => !(l.extraArgs ?? []).includes('-lm'))).toBe(true)
    expect(h.backend.calls.loads.every((l) => l.gpuLayers < 64)).toBe(true)
    expect(h.backend.calls.loads.some((l) => l.gpuLayers === 0)).toBe(false) // CPU baseline skipped: file > 50% RAM
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

  it('D1: unloads the previous step before the live RAM pre-check (its mmap must not count against the next rung)', async () => {
    let backendRef: ReturnType<typeof fakeBackend> | null = null
    // While a server is loaded "available" is low (its mmap'd weights); once unloaded it is back to 20 GiB.
    const { s } = await run(() => ({}), {}, {
      backend: () => (backendRef = fakeBackend(() => ({}))),
      readRamAvailableBytes: () => (backendRef!.pid !== undefined ? 2 * GiB : 20 * GiB)
    })
    expect(s.runs.map((r) => r.status)).toEqual(['pass', 'pass', 'pass', 'pass', 'pass'])
    expect(backendRef!.calls.unloads).toBeGreaterThanOrEqual(5)
  })

  it('D9: GPU/CPU means use only rows covering a request, not load-phase or idle rows', async () => {
    const xs: TelemetrySample[] = []
    const { s } = await run(() => ({ hook: () => xs.push({ ...sample(2048), ts: Date.now() + 1, gpuUtilPct: 90, cpuPct: 20 }) }), { ladder: [2048] }, {
      startSampler: () => { xs.length = 0; xs.push({ ...sample(2048), ts: Date.now() - 5000, gpuUtilPct: 0, cpuPct: 100 }); return { samples: xs, unavailable: {}, stop: () => xs } }
    })
    expect(s.runs[0].avgGpuUtil).toMatchObject({ value: 90, kind: 'measured' }) // the 0 % load-phase row is excluded
    expect(s.runs[0].avgCpuUtil.value).toBe(20)
    expect(s.runs[0].peakVramBytes.kind).toBe('measured') // peaks still use every row
  })

  it('session:started carries the planned candidate count (progress total)', async () => {
    const { events } = await run(() => ({}), { workload: 'long_context_coding' })
    expect(events.find((e) => e.type === 'session:started')).toMatchObject({ candidates: 2 })
  })

  it('quality runs on the best-offload usable candidate (most GPU layers), not on the first one the plan ran', async () => {
    const m27 = { ...model, id: 'C:/models/q27b.gguf', fileBytes: Math.round(16.1 * GiB), layers: 64, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064 }
    const h = await run(() => ({}), { workload: 'max_quality', modelIds: [m27.id], ladder: [2048], heavyMode: true, runQuality: true }, { models: [m27] })
    const ran = [...new Set(h.s.runs.map((r) => r.configId))]
    const most = Math.max(...h.s.runs.map((r) => Number(/ngl=(\d+)/.exec(r.configId)?.[1] ?? 0)))
    expect(ran.length).toBeGreaterThan(1)
    expect(h.s.quality).toHaveLength(1)
    expect(h.s.quality[0].configId).toContain(`ngl=${most}|`)
  })

  it('gen-config search: baseline + thinking @ lowest and middle effort on one load, model-card sampling, 3 seeded samples', async () => {
    const genKnobs = { supportsThinking: true, effortValues: ['low', 'medium', 'high', 'xhigh'], recommended: { temperature: 0.6, topP: 0.95 } }
    const seen: PromptRequest[] = []
    let ref: ReturnType<typeof fakeBackend> | null = null
    const backend = () => {
      const b = fakeBackend(() => ({}))
      const rp = b.runPrompt.bind(b)
      b.runPrompt = async (q) => { seen.push(q); return rp(q) }
      return (ref = b)
    }
    const m = { ...model, supportsThinking: true, genKnobs }
    const r = await run(() => ({}), { workload: 'coding', runQuality: true, ladder: [2048] }, { models: [m], backend })
    const kw = ref!.calls.templateOpts.map((o) => JSON.stringify((o as { templateKwargs?: unknown } | undefined)?.templateKwargs))
    expect(kw.length).toBe(17 + 17 * 3 * 2)
    expect([...new Set(kw)]).toEqual(['{"enable_thinking":false}', '{"enable_thinking":true,"reasoning_effort":"low"}', '{"enable_thinking":true,"reasoning_effort":"medium"}'])
    expect(ref!.calls.loads).toHaveLength(2) // one ladder step + ONE quality load for all three gen configs
    const stoch = seen.filter((q) => q.temperature === 0.6) as (PromptRequest & { topP?: number })[]
    expect(stoch.every((q) => q.topP === 0.95 && q.maxTokens >= 1024)).toBe(true)
    expect([...new Set(stoch.map((q) => q.seed))].sort()).toEqual([1, 2, 3])
    const rows = r.s.quality[0].results as (QualityResult & { genId: string })[]
    expect([...new Set(rows.map((x) => x.genId))]).toEqual(['off', 'think-low-t0.6', 'think-medium-t0.6'])
    const quick = await run(() => ({}), { workload: 'coding', runQuality: true, ladder: [2048], qualityMode: 'quick' }, { models: [m] })
    expect(quick.backend.calls.templates).toBe(17 * 3)
    const off = await run(() => ({}), { workload: 'coding', runQuality: true, ladder: [2048], genSearch: false }, { models: [m] })
    expect(off.backend.calls.templates).toBe(17)
  })

  it('thinking models: quality templates use enable_thinking=false; others get no kwargs', async () => {
    const thinking = await run(() => ({}), { workload: 'fast_assistant', runQuality: true, ladder: [2048, 4096], genSearch: false }, { models: [{ ...model, supportsThinking: true }] })
    expect(thinking.backend.calls.templateOpts.length).toBe(17)
    expect(thinking.backend.calls.templateOpts.every((o) => JSON.stringify(o) === '{"templateKwargs":{"enable_thinking":false}}')).toBe(true)
    expect(thinking.rec?.reasons.join('\n')).toMatch(/\[I-5\.4\] .*: quality measured with thinking disabled \(chat template enable_thinking=false\)/)
    const plain = await run(() => ({}), { runQuality: true, ladder: [2048] })
    expect(plain.backend.calls.templateOpts.every((o) => o === undefined)).toBe(true)
  })

  it('resume with a stored plan uses its configIds verbatim, even if planning would now produce different ones', async () => {
    const first = await run(() => ({}), { ladder: [2048, 4096] })
    const planned = [...new Set(first.s.runs.map((r) => r.configId))]
    expect(planned).toEqual([`${model.id}|ngl=all|kv=f16|t=8`])
    // The stored plan came from an older build: same model, a different config id (e.g. another heavy-mode ngl).
    const stored = { ...generateCandidates(machineFromProfile(machine, 'Vulkan0'), model, { backend: 'vulkan' }, WORKLOADS.coding).candidates[0] }
    const old = { ...stored, id: `${model.id}|ngl=all|kv=f16|t=8|old`, ctxSteps: [2048, 4096] }
    const seeded = first.s.runs.map((r) => ({ ...r, configId: old.id }))
    const re = await run(() => ({}), { resumeSessionId: 's1', ladder: [2048, 4096] }, { plan: [old] }, seeded)
    expect(re.backend.calls.loads).toEqual([]) // both stored steps reused under the stored id
    expect(re.rec?.ranked.map((x) => x.configId)).toEqual([old.id])
    const fresh = await run(() => ({}), { resumeSessionId: 's1', ladder: [2048, 4096] }, {}, seeded) // no plan → regenerated
    expect(ctxOf(fresh.backend)).toEqual([2048, 4096])
  })

  it('H1: the RAM guard runs during LOAD (mmap ramps there) and aborts the step before any request', async () => {
    const tiny = { ...model, fileBytes: 100 * 1024 ** 2 } // no mmap credit
    const low = { ...sample(2048), ramAvailBytes: 1 * GiB }
    const { s, backend } = await run(() => ({ load: 'slow' }), { ladder: [2048, 4096] }, {
      models: [tiny], startSampler: () => ({ samples: [low], unavailable: {}, stop: () => [low] }), config: { guardPollMs: 5 }
    })
    expect(s.runs).toHaveLength(1)
    expect(s.runs[0]).toMatchObject({ ctx: 2048, status: 'fail', failureKind: 'guard_abort' })
    expect(s.details[0].reason).toBe('RAM available 1.0 GiB fell below the floor')
    expect(backend.calls.templates).toBe(0)
    expect(backend.calls.unloads).toBeGreaterThanOrEqual(2) // killed during load, and again before returning
  })

  it('H2: heavy mode runs full-offload configs first, then partial configs most-offloaded first', async () => {
    const m27 = { ...model, id: 'C:/models/q27b.gguf', fileBytes: Math.round(16.1 * GiB), layers: 64, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064 }
    const h = await run(() => ({}), { workload: 'max_quality', modelIds: [model.id, m27.id], ladder: [2048], heavyMode: true }, { models: [model, m27] })
    const order = [...new Set(h.s.runs.map((r) => r.configId))]
    expect(order[0]).toBe(`${model.id}|ngl=all|kv=f16|t=8`)
    const ngl = order.slice(1).map((id) => Number(/ngl=(\d+)/.exec(id)![1]))
    expect(ngl).toEqual([...ngl].sort((a, b) => b - a))
  })

  it('H3: a model with no candidates is named in the recommendation reasons, not only the log', async () => {
    const m27 = { ...model, id: 'C:/models/q27b.gguf', name: 'Qwen3.8-27B', fileBytes: Math.round(16.1 * GiB), layers: 64 }
    const r = await run(() => ({}), { workload: 'max_quality', modelIds: [model.id, m27.id] }, { models: [model, m27] })
    expect(r.rec?.reasons.find((x) => x.startsWith('[I-2.3] Not benchmarked: Qwen3.8-27B — '))).toMatch(/enable heavy-model mode/)
  })

  it('heavy configs: a >2 GiB shared spill is recorded (degraded + reason) and the ladder moves to the next config — no abort', async () => {
    const m27 = { ...model, id: 'C:/models/q27b.gguf', fileBytes: Math.round(16.1 * GiB), layers: 64, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064 }
    const spill = { ...sample(2048), procVramDedicatedBytes: 14 * GiB, procVramSharedBytes: Math.round(2.31 * GiB) } // dedicated ≥ 80 %: real spill
    const h = await run(() => ({}), { workload: 'max_quality', modelIds: [m27.id], ladder: [2048, 4096], heavyMode: true }, {
      models: [m27], startSampler: () => ({ samples: [spill], unavailable: {}, stop: () => [spill] }), config: { guardPollMs: 5, heavyGuardPollMs: 5 }
    })
    expect(h.s.runs.every((r) => r.failureKind !== 'guard_abort')).toBe(true)
    expect(h.s.runs.filter((r) => r.ctx === 4096)).toEqual([]) // each config stopped after its spilled 2K step
    expect(new Set(h.s.runs.map((r) => r.configId)).size).toBeGreaterThan(1) // …and the next config still ran
    const done = h.events.filter((e) => e.type === 'candidate:done') as { reason: string | null }[]
    expect(done[0].reason).toBe('spilled 2.31 GiB into shared GPU memory at 2048; next configuration')
  })

  it('restarts a sampler once when its pid columns are missing, and persists sampler errors in the run detail', async () => {
    let restarts = 0
    const mk = () => {
      const xs = [sample(2048)]
      const errors: string[] = []
      return {
        samples: xs, unavailable: {}, errors, stop: () => xs,
        get hasPidColumns() { return restarts === 0 ? false : true },
        restart() { restarts++; errors.push(`typeperf restarted after ${xs.length} samples (pid columns missing)`) }
      }
    }
    const { s } = await run(() => ({}), { ladder: [2048] }, { startSampler: () => mk(), config: { guardPollMs: 5 } })
    expect(restarts).toBe(1)
    expect(s.details[0].samplerErrors).toEqual(['typeperf restarted after 1 samples (pid columns missing)'])
    const ok = await run(() => ({}), { ladder: [2048] })
    expect(ok.s.details[0].samplerErrors).toEqual([]) // fakes without the new fields keep working
  })

  it('spill: host-pinned buffers (load log) are subtracted — 3.5 GiB of CPU layers + flat shared is not spill', async () => {
    const xs = (ctx: number) => [{ ...sample(ctx), procVramDedicatedBytes: 14 * GiB, procVramSharedBytes: Math.round(3.5 * GiB) }]
    const { s } = await run(() => ({ hostMiB: 3.5 * 1024 }), { ladder: [2048, 4096] }, { startSampler: (pid) => { const v = xs(pid - 1000); return { samples: v, unavailable: {}, stop: () => v } } })
    expect(s.runs.map((r) => r.peakSharedGpuBytes.value)).toEqual([0, 0])
    expect(s.runs[0].peakSharedGpuRawBytes?.value).toBe(Math.round(3.5 * GiB))
    expect(s.runs[0].hostPinnedBytes).toMatchObject({ value: 3.5 * GiB, kind: 'declared' })
    expect(s.runs.every((r) => r.status === 'pass')).toBe(true)
  })

  it('spill: growth of shared memory while dedicated VRAM is saturated IS spill; with free VRAM it is not', async () => {
    const at = (ctx: number, ded: number) => [{ ...sample(ctx), procVramDedicatedBytes: ded, procVramSharedBytes: ctx >= 4096 ? Math.round(1.5 * GiB) : Math.round(0.02 * GiB) }]
    const sat = await run(() => ({}), { ladder: [2048, 4096] }, { startSampler: (pid) => { const v = at(pid - 1000, 14 * GiB); return { samples: v, unavailable: {}, stop: () => v } } })
    expect(sat.s.runs[1].peakSharedGpuBytes.value).toBe(Math.round(1.5 * GiB)) // step 1 was saturated → no baseline is subtracted
    const free = await run(() => ({}), { ladder: [2048, 4096] }, { startSampler: (pid) => { const v = at(pid - 1000, 8 * GiB); return { samples: v, unavailable: {}, stop: () => v } } })
    expect(free.s.runs[1].peakSharedGpuBytes.value).toBe(0)
    expect(free.s.runs[1].peakSharedGpuBytes.source).toMatch(/dedicated below saturation: not spill/)
  })

  it('required context: the ladder runs every rung up to it (UI ladder cap ignored), nothing above; CR-04-long added', async () => {
    const m = { ...model, ctxTrain: 131072 }
    const r = await run(() => ({}), { workload: 'long_context_coding', requiredContext: 65536, ladder: [2048, 4096], runQuality: true }, { models: [m] })
    const f16 = `${m.id}|ngl=all|kv=f16|t=8`
    expect(r.s.runs.filter((x) => x.configId === f16).map((x) => x.ctx)).toEqual([2048, 4096, 8192, 16384, 32768, 65536])
    expect(r.s.quality[0].results.map((x) => x.testId)).toContain('CR-04-long')
    expect(r.backend.calls.loads.some((l) => l.contextSize === 131072)).toBe(false)
  })

  it('no CR-04-long without a required context ≥ 32K; skipped (with a reason) when no config reaches it', async () => {
    const m = { ...model, ctxTrain: 131072 }
    const plain = await run(() => ({}), { runQuality: true, ladder: [2048, 4096] }, { models: [m] })
    expect(plain.s.quality[0].results.some((x) => x.testId === 'CR-04-long')).toBe(false)
    // 128K: f16 over VRAM, q8_0 reaches it in the plan, but the fake run fails the 128K load → no config reaches 128K.
    const r = await run((ctx) => (ctx === 131072 ? { load: 'oom' } : {}), { workload: 'long_context_coding', requiredContext: 131072, runQuality: true }, { models: [m] })
    expect(r.s.quality[0].results.some((x) => x.testId === 'CR-04-long')).toBe(false)
    expect(r.rec?.reasons.some((x) => /^\[I-2\.5\] Long-context needle CR-04-long at 128K skipped for .*: practical context 64K < 128K$/.test(x))).toBe(true)
  })

  it('F2: every loadModel gets the session signal (cancel during load is immediate)', async () => {
    const ac = new AbortController()
    const { backend } = await run(() => ({}), { ladder: [2048], runQuality: true }, { signal: ac.signal })
    expect(backend.calls.loads.length).toBeGreaterThan(1) // ladder + quality
    expect(backend.calls.loads.every((l) => l.signal === ac.signal)).toBe(true)
  })

  it('F3: the RAM guard reads OS free RAM on its own — no typeperf rows needed — and fails safe when blind', async () => {
    const tiny = { ...model, fileBytes: 100 * 1024 ** 2 }
    const noRows = { startSampler: () => ({ samples: [], unavailable: {}, stop: () => [] }), config: { guardPollMs: 5 }, models: [tiny] }
    let ref: ReturnType<typeof fakeBackend> | null = null
    const low = await run(() => ({ load: 'slow' }), { ladder: [2048] }, {
      ...noRows, backend: () => (ref = fakeBackend(() => ({ load: 'slow' }))), readRamAvailableBytes: () => (ref!.pid !== undefined ? 1 * GiB : 20 * GiB)
    })
    expect(low.s.runs[0]).toMatchObject({ status: 'fail', failureKind: 'guard_abort' })
    expect(low.s.details[0].reason).toBe('RAM available 1.0 GiB (OS) fell below the floor')
    const blind = await run(() => ({ load: 'slow' }), { ladder: [2048] }, { ...noRows, readRamAvailableBytes: () => { throw new Error('no counter') } })
    expect(blind.s.runs[0]).toMatchObject({ status: 'fail', failureKind: 'guard_abort' })
    expect(blind.s.details[0].reason).toMatch(/^RAM guard inputs unreadable/)
  })

  it('F1: the quality phase is guarded — RAM falling below the floor during the suite discards it', async () => {
    const tiny = { ...model, fileBytes: 100 * 1024 ** 2 }
    let ref: ReturnType<typeof fakeBackend> | null = null
    const r = await run(() => ({}), { ladder: [2048], runQuality: true }, {
      models: [tiny], config: { guardPollMs: 5 }, backend: () => (ref = fakeBackend(() => ({}))),
      readRamAvailableBytes: () => (ref!.calls.loads.length >= 2 && ref!.pid !== undefined ? 1 * GiB : 20 * GiB) // low only once the quality load is up
    })
    expect(r.s.runs[0].status).toBe('pass')
    expect(r.s.quality).toEqual([]) // incomplete suite is never stored
    expect(r.events.some((e) => e.type === 'log' && /fell below the floor during quality suite/.test(e.msg))).toBe(true)
  })

  it('F5: resume reuses stored quality only when it is the complete current suite', async () => {
    const first = await run(() => ({}), { ladder: [2048], runQuality: true })
    const stored = first.s.quality[0]
    expect(stored.results).toHaveLength(17)
    const resume = (q: typeof first.s.quality) => run(() => ({}), { ladder: [2048], runQuality: true, resumeSessionId: 's1' }, {}, first.s.runs, q)
    const full = await resume([stored])
    expect(full.backend.calls.templates).toBe(0) // complete + same suite → reused
    const partial = await resume([{ ...stored, results: stored.results.slice(0, 1) }])
    expect(partial.backend.calls.templates).toBe(17) // incomplete → re-run
    const legacy = await resume([{ ...stored, results: stored.results.map(({ ...r }) => { delete (r as { suite?: string }).suite; return r }) }])
    expect(legacy.backend.calls.templates).toBe(17) // no suite version → re-run
  })

  it('F12: without llama timings, TPS are estimated from the streamed-token count and the prompt size', async () => {
    const { s } = await run(() => ({ noTimings: true }), { ladder: [2048] })
    expect(s.runs[0].decodeTps).toMatchObject({ value: 50, kind: 'estimated' }) // 100 tokens / (3000 − 1000) ms
    expect(s.runs[0].prefillTps).toMatchObject({ value: 1536, kind: 'estimated' }) // 0.75 × 2048 tokens / 1 s
    expect(s.runs[0].status).toBe('pass')
  })

  it('a fatal unload error (ServerStuckError) is a hard session stop: session:failed, no further loads, no cleanup unload', async () => {
    let ref: ReturnType<typeof fakeBackend> | null = null
    let unloadsAfterStuck = 0
    let stuck = false
    const r = await run(() => ({}), {}, {
      backend: () => {
        const b = (ref = fakeBackend(() => ({})))
        const orig = b.unloadModel.bind(b)
        b.unloadModel = async () => {
          if (stuck) { unloadsAfterStuck++; return }
          if (b.pid === 1000 + 4096) { stuck = true; throw Object.assign(new Error('llama-server pid 5096 still alive after taskkill /F'), { fatal: true }) }
          return orig()
        }
        return b
      }
    })
    expect(r.rec).toBeNull()
    expect(r.events.at(-1)).toMatchObject({ type: 'session:failed', error: 'llama-server pid 5096 still alive after taskkill /F' })
    expect(r.s.status.at(-1)).toBe('failed')
    expect(ctxOf(ref!)).toEqual([2048, 4096]) // stopped before loading 8K
    expect(unloadsAfterStuck).toBe(0)
  })

  it('F12 (#2 shape): no timings but stream token counts → estimated TPS from those counts', async () => {
    const b = fakeBackend(() => ({}))
    b.runPrompt = async () => ({ ttftMs: 500, promptTokens: 1500, prefillMs: null, prefillTps: null, decodeTokens: 64, decodeMs: null, decodeTps: null, totalMs: 1500, text: 'x', stopType: 'limit', timedOut: false, error: null })
    const { s } = await run(() => ({}), { ladder: [2048] }, { backend: () => b })
    expect(s.runs[0].decodeTps).toMatchObject({ value: 64, kind: 'estimated' }) // 64 tokens / 1 s
    expect(s.runs[0].prefillTps).toMatchObject({ value: 3000, kind: 'estimated' }) // 1500 tokens / 0.5 s
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
    expect(on.s.quality[0]).toMatchObject({ ctx: 8192 }) // min(target 8K, ceiling), a passed rung (D09)
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
    expect(rec?.reasons[0]).toBe('[I-7.1] No recommendation: nothing was run — the RAM guard skipped every step before loading')
    expect(rec?.reasons[1]).toMatch(/^\[I-4\.3\] est\. RAM 0\.5 GiB > available 1\.0 GiB − floor 4\.0 GiB$/) // resident only; floor max(4 GiB, 8%)
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
