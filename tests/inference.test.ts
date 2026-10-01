// Test scope: MOCKED — llama-server (tests/fixtures/fake-llama-server.cjs: canned /health, /props, SSE, 400s, deaths;
// no model, no GPU) and, in one test, a child that ignores kill(). REAL — node processes, sockets, the SSE parser,
// the stderr classifier and Windows process cleanup in separate tests. Proves process/HTTP orchestration, not inference.
import { spawn, type ChildProcess } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { ConfigDriftError, LlamaCppBackend, ServerStuckError, type ProcessIdentity, type ProcessTree } from '../src/core/runtimes/llamacpp'
import { classifyExit, emptyDeclared, parseDevices, parseLogLine, parseSse, pickBenchmarkDevice, pickDiscreteDevice, toPromptResult, type CompletionChunk } from '../src/core/runtimes/llamacpp/parse'

// Shape of a real llama-server /completion stream (b11208), split at awkward byte boundaries.
const STREAM =
  'data: {"content":"A","stop":false}\n\ndata: {"content":" GPU","stop":false}\n\n' +
  'data: {"content":"","stop":true,"stop_type":"limit","tokens_predicted":32,"tokens_evaluated":14,' +
  '"timings":{"cache_n":0,"prompt_n":14,"prompt_ms":8.2,"prompt_per_token_ms":0.59,"prompt_per_second":1707.3,' +
  '"predicted_n":32,"predicted_ms":95.1,"predicted_per_token_ms":2.97,"predicted_per_second":336.5}}\n\n'

describe('SSE + timings', () => {
  it('reassembles events across chunk boundaries and maps the final timings', () => {
    let buf = ''
    const events: CompletionChunk[] = []
    for (let i = 0; i < STREAM.length; i += 7) {
      const r = parseSse(buf + STREAM.slice(i, i + 7))
      buf = r.rest
      events.push(...(r.events as CompletionChunk[]))
    }
    expect(buf).toBe('')
    expect(events.map((e) => e.content).join('')).toBe('A GPU')
    const final = events.find((e) => e.stop)!
    const r = toPromptResult(final, { ttftMs: 12.5, totalMs: 110, text: 'A GPU', timedOut: false, error: null })
    expect(r).toEqual({
      ttftMs: 12.5, promptTokens: 14, prefillMs: 8.2, prefillTps: 1707.3, decodeTokens: 32, decodeMs: 95.1, decodeTps: 336.5,
      totalMs: 110, text: 'A GPU', stopType: 'limit', timedOut: false, error: null
    })
  })

  it('leaves unreported values null instead of inventing them', () => {
    const r = toPromptResult(null, { ttftMs: null, totalMs: 5, text: '', timedOut: true, error: 'timed out after 5ms' })
    expect(r.decodeTps).toBeNull()
    expect(r.stopType).toBeNull()
    expect(r.timedOut).toBe(true)
  })
})

describe('startup log + exit classification', () => {
  it('parses offload and per-device buffer sizes', () => {
    const d = emptyDeclared()
    for (const l of [
      'load_tensors: offloaded 25/25 layers to GPU',
      'load_tensors:      Vulkan0 model buffer size =   500.79 MiB',
      'load_tensors:   CPU_Mapped model buffer size =   137.94 MiB',
      'llama_kv_cache:    Vulkan0 KV buffer size =    24.00 MiB',
      'llama_context:    Vulkan0 compute buffer size =    34.01 MiB'
    ]) parseLogLine(l, d)
    expect(d).toEqual({
      layersOffloaded: 25, layersTotal: 25,
      modelBufferMiB: { Vulkan0: 500.79, CPU_Mapped: 137.94 }, kvBufferMiB: { Vulkan0: 24 }, computeBufferMiB: { Vulkan0: 34.01 }
    })
  })

  it('classifies OOM, device lost, and other crashes', () => {
    expect(classifyExit(['ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory'])).toBe('oom')
    expect(classifyExit(['llama_model_load: failed to allocate buffer'])).toBe('oom')
    expect(classifyExit(['ggml_vulkan: vk::Queue::submit: ErrorDeviceLost'])).toBe('device_lost')
    expect(classifyExit(['GGML_ASSERT(n_tokens > 0) failed'])).toBe('crash')
    // Vulkan C codes (W4 F9)
    expect(classifyExit(['ggml_vulkan: VK_ERROR_DEVICE_LOST'])).toBe('device_lost')
    expect(classifyExit(['vkAllocateMemory: VK_ERROR_OUT_OF_DEVICE_MEMORY'])).toBe('oom')
  })

  it('parses --list-devices and picks the discrete GPU over the iGPU', () => {
    const devs = parseDevices([
      'Available devices:',
      '  Vulkan0: AMD Radeon(TM) Graphics (16187 MiB, 15377 MiB free)',
      '  Vulkan1: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)'
    ].join('\r\n'))
    expect(devs).toHaveLength(2)
    expect(pickDiscreteDevice(devs)).toMatchObject({ id: 'Vulkan1', name: 'AMD Radeon RX 9070 XT', totalMiB: 16304 })
    expect(pickBenchmarkDevice(devs, true)?.id).toBe('Vulkan1')
    expect(pickBenchmarkDevice(devs.slice(0, 1), true)?.id).toBe('Vulkan0')
    expect(pickBenchmarkDevice(devs.slice(0, 1), false)).toBeNull()
    expect(pickBenchmarkDevice(parseDevices('Vulkan0: Microsoft Basic Render Driver (512 MiB, 256 MiB free)'), true)).toBeNull()
  })
})

/** Poll until cond() holds or the (generous) deadline passes; returns whether it held. Replaces fixed sleeps. */
async function waitFor(cond: () => boolean, ms = 15_000): Promise<boolean> {
  for (const end = Date.now() + ms; Date.now() < end; await new Promise((r) => setTimeout(r, 25))) if (cond()) return true
  return cond()
}
const exited = (p: { exitCode: number | null; signalCode: NodeJS.Signals | null }) => p.exitCode !== null || p.signalCode !== null

describe('LlamaCppBackend process handling (fake server, no GPU)', { timeout: 20_000 }, () => {
  const fake = join(__dirname, 'fixtures', 'fake-llama-server.cjs')
  const pidFile = join(tmpdir(), `lao-test-${process.pid}.pid`)
  // Deterministic in-process ProcessTree: identity fixed at spawn, live while the ChildProcess runs, no descendants.
  // Without it every load/unload ran 9–12 serial PowerShell CIM calls inside the 20 s budget (the flake). Real CIM
  // identity is covered by tests/runtime/process-tree-windows.test.ts.
  const backend = (mode: string) => {
    let child: ChildProcess | null = null
    let identity: ProcessIdentity | null = null
    const live = (pid: number) => !!child && child.pid === pid && !exited(child)
    const processTree: ProcessTree = {
      descendants: async () => [],
      inspect: async (pid) => (live(pid) ? identity : null),
      killVerified: async (record) => {
        if (!live(record.pid) || record.startedAt !== identity?.startedAt) return false
        child!.kill()
        return true
      },
      isAlive: async (pid) => live(pid),
      kill: async () => { throw new Error('unverified numeric-PID kill') }
    }
    return new LlamaCppBackend('unused', {
      pidFile, processTree,
      spawnFn: (_cmd, args, opts) => {
        child = spawn(process.execPath, [fake, ...args], { ...opts, env: { ...process.env, FAKE_MODE: mode } })
        identity = { pid: child.pid!, name: 'node.exe', startedAt: new Date().toISOString() }
        return child
      }
    })
  }
  const cfg = { modelPath: join(tmpdir(), 'm.gguf'), contextSize: 2048, gpuLayers: 99, device: 'Vulkan0' }
  let b: LlamaCppBackend | null = null
  afterEach(async () => { await b?.unloadModel() })

  it('fails fast and classifies when the server dies before /health', async () => {
    b = backend('die-oom')
    const t0 = Date.now()
    await expect(b.loadModel(cfg)).rejects.toThrow(/exited with code 3 \(oom\)/)
    expect(Date.now() - t0).toBeLessThan(60_000) // well below the 120 s /health deadline it must not wait for
    expect(b.lastExit).toMatchObject({ code: 3, reason: 'oom' })
    expect(existsSync(pidFile)).toBe(false)
  })

  it('loadModel is abortable during the /health wait and kills the child', async () => {
    b = backend('slow')
    const ctl = new AbortController()
    setTimeout(() => ctl.abort(), 500)
    const t0 = Date.now()
    await expect(b.loadModel({ ...cfg, signal: ctl.signal })).rejects.toThrow(/^cancelled$/)
    expect(Date.now() - t0).toBeLessThan(60_000) // abort, not the 120 s /health deadline
    expect(b.pid).toBeUndefined()
    expect(existsSync(pidFile)).toBe(false)
    await expect(b.loadModel({ ...cfg, signal: ctl.signal })).rejects.toThrow(/^cancelled$/) // already aborted: no spawn
  })

  it('safety: a health endpoint that never responds is cancellable and its child is reaped', async () => {
    b = backend('hang-health')
    const ctl = new AbortController()
    setTimeout(() => ctl.abort(), 100)
    await expect(b.loadModel({ ...cfg, signal: ctl.signal })).rejects.toThrow(/^cancelled$/)
    expect(b.pid).toBeUndefined()
    expect(existsSync(pidFile)).toBe(false)
  })

  it('safety: a request that never responds times out; unloading reaps the child', async () => {
    b = backend('hang-completion')
    await b.loadModel(cfg)
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 1, timeoutMs: 100 })
    expect(r.timedOut).toBe(true)
    expect(r.error).toMatch(/timed out|abort/i)
    await b.unloadModel()
    expect(b.pid).toBeUndefined()
    expect(existsSync(pidFile)).toBe(false)
  })

  it('rejects a server whose /props reports a different model', async () => {
    b = backend('wrong')
    await expect(b.loadModel(cfg)).rejects.toThrow(/wrong server answered/)
  })

  it('loads, streams a prompt, and cleans up the pid file on unload', async () => {
    b = backend('ok')
    const load = await b.loadModel(cfg)
    expect(load.declared.layersOffloaded).toBe(25)
    expect(load.declared.modelBufferMiB).toEqual({ Vulkan0: 500.79 })
    expect(existsSync(pidFile)).toBe(true)
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 1, timeoutMs: 5_000 })
    expect(r).toMatchObject({ text: 'Hi', decodeTokens: 1, stopType: 'limit', error: null })
    expect(r.ttftMs).toBeGreaterThan(0)
    expect(await b.tokenize('one two three')).toBe(3)
    // date_string is always pinned; caller kwargs are added (and can override)
    expect(await b.applyTemplate([{ role: 'user', content: 'hi' }], { templateKwargs: { enable_thinking: false } }))
      .toBe('[{"date_string":"01 Jan 2025","enable_thinking":false}] hi')
    await b.unloadModel()
    expect(existsSync(pidFile)).toBe(false)
    expect((await b.runPrompt({ prompt: 'x', maxTokens: 1 })).error).toBe('no model loaded')
  })

  it('fails with config_drift when the server serves a smaller n_ctx than requested', async () => {
    b = backend('drift')
    const err = await b.loadModel(cfg).catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ConfigDriftError)
    expect((err as ConfigDriftError).failureKind).toBe('config_drift')
    expect(String(err)).toMatch(/requested -c 2048 but server serves n_ctx 1024/)
  })

  it('reports HTTP 400 with the server error text', async () => {
    b = backend('http400')
    await b.loadModel(cfg)
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 1 })
    expect(r.error).toMatch(/HTTP 400: exceed_context_size_error: request \(40000 tokens\) exceeds/)
    expect(b.lastExit).toBeNull() // server still alive: runner classifies this as request_error, never crash
  })

  it('templateHash = sha256 of /props chat_template, per load; acceptedSampling = what the server reports applying (F5)', async () => {
    b = backend('ok')
    const lr = await b.loadModel(cfg)
    expect(lr.templateHash).toBe(createHash('sha256').update('{{ messages }}').digest('hex'))
    expect(b.templateHash).toBe(lr.templateHash)
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 1, temperature: 0.6, topK: 20, seed: 7 })
    // the fake applies top_k 40 whatever is asked: applied ≠ requested must be visible
    expect(r.acceptedSampling).toEqual({ temperature: 0.6, top_p: 0.95, top_k: 40, min_p: 0.05, seed: 7 })
    await b.unloadModel()
    b = backend('notemplate')
    expect((await b.loadModel(cfg)).templateHash).toBeNull()
    await b.unloadModel()
    b = backend('notimings') // a final chunk without generation_settings → not reported, never guessed
    await b.loadModel(cfg)
    expect((await b.runPrompt({ prompt: 'x', maxTokens: 1 })).acceptedSampling).toBeNull()
  })

  it('without timings, token counts fall back to streamed chunks and /tokenize (W4 F12)', async () => {
    b = backend('notimings')
    await b.loadModel(cfg)
    const r = await b.runPrompt({ prompt: 'one two three four', maxTokens: 3 })
    expect(r).toMatchObject({ decodeTokens: 3, promptTokens: 4, decodeTps: null, prefillTps: null, error: null })
  })

  it('unloadModel keeps the handle and pid file when the identity-verified server survives teardown (W4 F4)', async () => {
    const { createServer } = await import('node:http')
    const { EventEmitter } = await import('node:events')
    const srv = createServer((q, s) => s.end(q.url === '/health' ? '{"status":"ok"}' : JSON.stringify({ model_path: cfg.modelPath, default_generation_settings: { n_ctx: 2048 } })))
    await new Promise<void>((r) => srv.listen(0, '127.0.0.1', () => r()))
    const port = (srv.address() as { port: number }).port
    // A child that ignores kill(): all process-tree checks are injected, so this PID is never sent to the OS.
    const stuck = Object.assign(new EventEmitter(), { pid: 999_999_1, exitCode: null, signalCode: null, stdout: null, stderr: null, kill: () => true })
    const identity = { pid: stuck.pid, name: 'fake-server.exe', startedAt: '2026-09-28T00:00:00.0000000Z' }
    const processTree: ProcessTree = {
      descendants: async () => [], kill: async () => {}, isAlive: async () => stuck.exitCode === null,
      inspect: async () => stuck.exitCode === null ? identity : null,
      killVerified: async () => true // ignored; post-kill identity remains live
    }
    const sb = new LlamaCppBackend('unused', { pidFile, spawnFn: () => stuck as never, processTree })
    try {
      await sb.loadModel({ ...cfg, port })
      await expect(sb.unloadModel()).rejects.toBeInstanceOf(ServerStuckError)
      expect(sb.pid).toBe(999_999_1) // handle kept
      expect(existsSync(pidFile)).toBe(true) // and the pid file, for killStaleServer
    } finally {
      stuck.exitCode = 0 as never
      stuck.emit('exit', 0)
      stuck.emit('close', 0)
      await sb.unloadModel()
      srv.close()
    }
  })

  it('rejects a second concurrent prompt instead of clobbering the cancel slot', async () => {
    b = backend('ok')
    await b.loadModel(cfg)
    const [r1, r2] = await Promise.all([b.runPrompt({ prompt: 'x', maxTokens: 1 }), b.runPrompt({ prompt: 'y', maxTokens: 1 })])
    expect(r1.error).toBeNull()
    expect(r2.error).toMatch(/already in flight/)
  })

  it('reports the classified exit when the server died while idle, without fetching', async () => {
    b = backend('ok')
    await b.loadModel(cfg)
    process.kill(b.pid!)
    expect(await waitFor(() => b!.lastExit !== null)).toBe(true) // the close event, however late it arrives
    expect((await b.runPrompt({ prompt: 'x', maxTokens: 1 })).error).toMatch(/^server exited \(crash, code/)
  })
})

describe('stale server cleanup verifies identity (W4c D11)', { timeout: 60_000 }, () => {
  it('staleMatches: same exe path and start time within 30 s, else never', async () => {
    const { staleMatches } = await import('../src/core/runtimes/llamacpp')
    const rec = { pid: 5, exePath: 'C:/rt/llama-server.exe', startedAt: '2026-09-27T10:00:00.000Z' }
    expect(staleMatches(rec, { path: 'C:\\RT\\llama-server.exe', startedAt: '2026-09-27T10:00:03.000Z' })).toBe(true)
    expect(staleMatches(rec, { path: 'C:\\other\\llama-server.exe', startedAt: '2026-09-27T10:00:03.000Z' })).toBe(false)
    expect(staleMatches(rec, { path: 'C:/rt/llama-server.exe', startedAt: '2026-09-27T11:00:00.000Z' })).toBe(false) // reused pid
    expect(staleMatches(rec, null)).toBe(false)
  })

  it('killStaleServer kills a verified record, spares a mismatch and an old plain-number file', async () => {
    const { killStaleServer } = await import('../src/core/runtimes/llamacpp')
    const { writeFileSync: w } = await import('node:fs')
    const { spawn: sp } = await import('node:child_process')
    const pf = join(tmpdir(), `lao-stale-${process.pid}.pid`)
    const child = () => sp(process.execPath, ['-e', 'setTimeout(() => {}, 60000)'])
    const a = child(), b = child(), c = child()
    try {
      // start-time checks need the children to exist: wait until each has a pid (spawned)
      expect(await waitFor(() => [a, b, c].every((x) => x.pid !== undefined))).toBe(true)
      w(pf, JSON.stringify({ pid: b.pid, exePath: 'C:/not/node.exe', startedAt: new Date().toISOString() }))
      expect(await killStaleServer(pf)).toMatch(/not the recorded/)
      w(pf, String(c.pid))
      expect(await killStaleServer(pf)).toMatch(/could not be verified/)
      w(pf, JSON.stringify({ pid: a.pid, exePath: process.execPath, startedAt: new Date().toISOString() }))
      expect(await killStaleServer(pf)).toMatch(/killed stale/)
      expect(await waitFor(() => exited(a))).toBe(true)
      expect(b.exitCode).toBeNull()
      expect(c.exitCode).toBeNull()
    } finally {
      for (const x of [a, b, c]) x.kill()
    }
  })
})

describe('child environment (docs/HIP-BACKEND.md)', () => {
  it('O4: strips mixed and lowercase managed-memory keys before child spawn', async () => {
    const { LlamaCppBackend: B, serverEnv } = await import('../src/core/runtimes/llamacpp')
    const base = { PATH: 'x', ggml_cuda_enable_unified_memory: '1', GgMl_CuDa_EnAbLe_UnIfIeD_MeMoRy: '1' }
    expect(serverEnv(base)).toEqual({ PATH: 'x' })
    let env: NodeJS.ProcessEnv | undefined
    const previous = { ...process.env }
    try {
      Object.assign(process.env, base)
      const b = new B('unused', { spawnFn: (_c, _a, opts) => { env = opts.env; throw new Error('captured') } })
      await b.loadModel({ modelPath: 'm.gguf', contextSize: 2048, gpuLayers: 999, device: 'ROCm0', threads: 8 }).catch(() => {})
    } finally {
      for (const key of Object.keys(process.env)) if (!(key in previous)) delete process.env[key]
      Object.assign(process.env, previous)
    }
    expect(env).toBeDefined()
    expect(Object.keys(env!).some((k) => k.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')).toBe(false)
  })

  it('never passes GGML_CUDA_ENABLE_UNIFIED_MEMORY to llama-server', async () => {
    const { LlamaCppBackend: B, serverEnv } = await import('../src/core/runtimes/llamacpp')
    expect(serverEnv({ PATH: 'x', GGML_CUDA_ENABLE_UNIFIED_MEMORY: '1' })).toEqual({ PATH: 'x' })
    const prev = process.env.GGML_CUDA_ENABLE_UNIFIED_MEMORY
    process.env.GGML_CUDA_ENABLE_UNIFIED_MEMORY = '1'
    let env: NodeJS.ProcessEnv | undefined
    const b = new B('unused', { spawnFn: (_c, _a, opts) => { env = opts.env; throw new Error('captured') } })
    await b.loadModel({ modelPath: 'm.gguf', contextSize: 2048, gpuLayers: 999, device: 'ROCm0', threads: 8 }).catch(() => {})
    if (prev === undefined) delete process.env.GGML_CUDA_ENABLE_UNIFIED_MEMORY; else process.env.GGML_CUDA_ENABLE_UNIFIED_MEMORY = prev
    expect(env).toBeDefined()
    expect(env).not.toHaveProperty('GGML_CUDA_ENABLE_UNIFIED_MEMORY')
  })
})
