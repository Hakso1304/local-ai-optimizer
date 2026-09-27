import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync, readFileSync, rmSync } from 'node:fs'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { finished } from 'node:stream/promises'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'
import { LlamaCppBackend, ServerStuckError, type ProcessTree } from '../../src/core/runtimes/llamacpp'

// Fake llama-server child: the adapter only sees this object (spawnFn seam) plus a real HTTP server we control.
// Odd pid: Windows pids are multiples of 4, so a taskkill escalation can never hit a real process.
class FakeChild extends EventEmitter {
  stdout = new PassThrough()
  stderr = new PassThrough()
  pid = 424243
  exitCode: number | null = null
  signalCode: NodeJS.Signals | null = null
  kills = 0
  exitOnKill = true
  kill(): boolean {
    this.kills++
    if (this.exitOnKill) void this.exit(null, 'SIGTERM')
    return true
  }
  /** Like a real child: 'exit', then 'close' once both stdio streams have drained. */
  async exit(code: number | null, signal: NodeJS.Signals | null = null, stderr: string[] = []): Promise<void> {
    for (const l of stderr) this.stderr.write(`${l}\n`)
    this.stdout.end()
    this.stderr.end()
    this.exitCode = code
    this.signalCode = signal
    this.emit('exit', code, signal)
    await Promise.all([finished(this.stdout), finished(this.stderr)])
    this.emit('close', code, signal)
  }
}

type Handler = (req: IncomingMessage, res: ServerResponse) => void
let server: Server
let port = 0
let handler: Handler
const MODEL = join(tmpdir(), 'lifecycle-model.gguf')
const pidFile = join(tmpdir(), `lao-lifecycle-${process.pid}.pid`)

const json = (res: ServerResponse, code: number, body: unknown) => res.writeHead(code, { 'content-type': 'application/json' }).end(JSON.stringify(body))
/** Healthy server serving MODEL; `completion` handles POST /completion. */
const healthy = (completion?: Handler, modelPath = MODEL): Handler => (req, res) => {
  if (req.url === '/health') return json(res, 200, { status: 'ok' })
  if (req.url === '/props') return json(res, 200, { model_path: modelPath })
  if (req.url === '/completion' && completion) return completion(req, res)
  res.writeHead(404).end()
}
const sse = (res: ServerResponse, ev: unknown) => res.write(`data: ${JSON.stringify(ev)}\n\n`)
const FINAL = { content: '', stop: true, stop_type: 'limit', timings: { prompt_n: 14, prompt_ms: 8.2, prompt_per_second: 1707.3, predicted_n: 2, predicted_ms: 5.9, predicted_per_second: 336.5 } }
const hang: Handler = (_req, res) => { res.writeHead(200, { 'content-type': 'text/event-stream' }); res.flushHeaders() }

let child: FakeChild
let b: LlamaCppBackend
const cfg = { modelPath: MODEL, contextSize: 2048, gpuLayers: 99, device: 'Vulkan0' }
const load = () => b.loadModel({ ...cfg, port })

beforeAll(async () => {
  server = createServer((req, res) => handler(req, res))
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r))
  port = (server.address() as AddressInfo).port
})
afterAll(() => { server.closeAllConnections(); server.close() })
beforeEach(() => {
  child = new FakeChild()
  const spawned: FakeChild[] = []
  const processTree: ProcessTree = {
    descendants: async () => [],
    kill: async (pid) => { spawned.find((p) => p.pid === pid)?.kill() },
    isAlive: async (pid) => { const p = spawned.find((x) => x.pid === pid); return !!p && p.exitCode === null && p.signalCode === null }
  }
  b = new LlamaCppBackend('unused', { pidFile, spawnFn: () => { spawned.push(child); return child as unknown as ChildProcess }, processTree })
})
afterEach(async () => {
  child.exitOnKill = true
  await b.unloadModel()
  server.closeAllConnections()
  rmSync(pidFile, { force: true })
})

describe('loadModel', () => {
  it('rejects a combined abort signal promptly while health stays 503 and reaps its child', async () => {
    handler = (_req, res) => json(res, 503, { status: 'loading' })
    const session = new AbortController(), guard = new AbortController()
    const signal = AbortSignal.any([session.signal, guard.signal])
    const started = Date.now()
    const pending = b.loadModel({ ...cfg, port, signal })
    setTimeout(() => guard.abort(new Error('RAM guard')), 50)
    await expect(pending).rejects.toThrow(/cancelled/i)
    expect(Date.now() - started).toBeLessThan(500)
    expect(signal.aborted).toBe(true)
    expect(session.signal.aborted).toBe(false)
    expect(child.kills).toBe(1)
    expect(existsSync(pidFile)).toBe(false)
  })

  it.each([
    ['oom', ['ggml_vulkan: Device memory allocation of size 9000000000 failed.', 'ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory']],
    ['device_lost', ['ggml_vulkan: vk::Queue::submit: ErrorDeviceLost']],
    ['crash', ['GGML_ASSERT(n_tokens > 0) failed']]
  ])('child dies before /health ok → rejects fast as %s', async (reason, lines) => {
    handler = (_req, res) => json(res, 503, { error: { message: 'Loading model' } })
    setTimeout(() => void child.exit(3, null, lines), 150)
    const t0 = Date.now()
    await expect(load()).rejects.toThrow(new RegExp(`exited with code 3 \\(${reason}\\)`))
    expect(Date.now() - t0).toBeLessThan(3000)
    expect(b.lastExit).toMatchObject({ code: 3, reason })
    expect(b.lastExit!.tail).toEqual(expect.arrayContaining(lines))
    // The failed load unloads and checks descendants before clearing the owned PID record.
    expect(existsSync(pidFile)).toBe(false)
  })

  it('writes the pid file while running', async () => {
    handler = healthy()
    await load()
    // W4c D11: the record carries the exe path and start time so a reused pid is never killed later.
    const rec = JSON.parse(readFileSync(pidFile, 'utf8'))
    expect(rec.pid).toBe(child.pid)
    expect(typeof rec.exePath).toBe('string')
    expect(Number.isFinite(Date.parse(rec.startedAt))).toBe(true)
  })

  it.each([
    ['different model_path', healthy(undefined, 'C:\\other\\model.gguf')],
    ['no /props', ((req, res) => (req.url === '/health' ? json(res, 200, { status: 'ok' }) : res.writeHead(404).end())) as Handler]
  ])('health ok but %s → "wrong server", our child is killed', async (_n, h) => {
    handler = h
    await expect(load()).rejects.toThrow(/wrong server answered/)
    expect(child.kills).toBe(1)
    expect(existsSync(pidFile)).toBe(false)
  })

  it('accepts the same model path with different case / separators (Windows)', async () => {
    handler = healthy(undefined, MODEL.toUpperCase())
    await expect(load()).resolves.toMatchObject({ loadTimeMs: expect.any(Number) })
  })
})

describe('runPrompt', () => {
  it('streams SSE with timings: text, TTFT and runtime-reported TPS', async () => {
    handler = healthy((_req, res) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      setTimeout(() => { sse(res, { content: 'A', stop: false }); sse(res, { content: ' GPU', stop: false }) }, 200)
      setTimeout(() => { sse(res, FINAL); res.end() }, 300)
    })
    await load()
    const tokens: string[] = []
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 5000 }, (t) => tokens.push(t))
    expect(tokens).toEqual(['A', ' GPU'])
    expect(r).toMatchObject({
      text: 'A GPU', promptTokens: 14, prefillMs: 8.2, prefillTps: 1707.3, decodeTokens: 2, decodeTps: 336.5,
      stopType: 'limit', timedOut: false, error: null
    })
    expect(r.ttftMs!).toBeGreaterThanOrEqual(180)
    expect(r.totalMs).toBeGreaterThan(r.ttftMs!)
  })

  it('server never answers → timedOut after the request timeout', async () => {
    handler = healthy(() => {}) // no headers ever
    await load()
    const t0 = Date.now()
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 300 })
    expect(r).toMatchObject({ timedOut: true, error: 'timed out after 300ms', decodeTps: null })
    expect(Date.now() - t0).toBeLessThan(2000)
  })

  it('stream stalls mid-way → timedOut, partial text kept, no fabricated timings', async () => {
    handler = healthy((_req, res) => { hang(_req, res); sse(res, { content: 'par', stop: false }) })
    await load()
    const r = await b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 400 })
    expect(r).toMatchObject({ timedOut: true, text: 'par', decodeTps: null, prefillTps: null })
    expect(r.ttftMs).not.toBeNull()
  })

  it('cancel() aborts an in-flight request and runPrompt resolves', async () => {
    handler = healthy(hang)
    await load()
    const p = b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 10_000 })
    setTimeout(() => void b.cancel(), 100)
    const t0 = Date.now()
    expect(await p).toMatchObject({ timedOut: false, error: 'cancelled' })
    expect(Date.now() - t0).toBeLessThan(1500)
    await expect(b.cancel()).resolves.toBeUndefined() // idle cancel is a no-op
  })

  it('server crashes mid-run → request aborted with the classified exit', async () => {
    handler = healthy(hang)
    await load()
    const p = b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 10_000 })
    setTimeout(() => void child.exit(-1073741819, null, ['ggml_vulkan: vk::Queue::submit: ErrorDeviceLost']), 100)
    const r = await p
    expect(r.error).toMatch(/exited with code -1073741819 \(device_lost\)/)
    expect(r.timedOut).toBe(false)
    expect(b.lastExit?.reason).toBe('device_lost')
  })

  it('an SSE error event becomes error, not an exception', async () => {
    handler = healthy((_req, res) => { hang(_req, res); sse(res, { error: { message: 'context size exceeded' } }); res.end() })
    await load()
    expect(await b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 2000 })).toMatchObject({ error: 'context size exceeded' })
  })

  it('HTTP error status becomes error', async () => {
    handler = healthy((_req, res) => json(res, 500, { error: 'boom' }))
    await load()
    expect((await b.runPrompt({ prompt: 'x', maxTokens: 2, timeoutMs: 2000 })).error).toMatch(/HTTP 500/)
  })
})

describe('unloadModel', () => {
  it('shares one in-flight tree reap across concurrent unload callers', async () => {
    let enter!: () => void, release!: () => void
    const entered = new Promise<void>((r) => { enter = r })
    const gate = new Promise<void>((r) => { release = r })
    let scans = 0, rootKills = 0
    const processTree: ProcessTree = {
      descendants: async () => { if (++scans === 1) { enter(); await gate } return [] },
      kill: async (pid) => { if (pid === child.pid) { rootKills++; child.kill() } },
      isAlive: async (pid) => pid === child.pid && child.exitCode === null && child.signalCode === null
    }
    b = new LlamaCppBackend('unused', { pidFile, spawnFn: () => child as unknown as ChildProcess, processTree })
    handler = healthy()
    await load()
    const first = b.unloadModel()
    await entered
    const second = b.unloadModel()
    release()
    await Promise.all([first, second])
    expect(rootKills).toBe(1)
    expect(scans).toBe(2) // one initial tree snapshot and one verification, not two reap sequences
    expect(existsSync(pidFile)).toBe(false)
  })

  it('Q2: reaps an owned descendant even when the server parent exits promptly', async () => {
    const descendantPid = 424245
    let descendantAlive = true
    const calls: number[] = []
    const startedAt = new Date().toISOString()
    const processTree: ProcessTree = {
      descendants: async (pid) => { expect(pid).toBe(child.pid); return [{ pid: descendantPid, name: 'fake-child', startedAt }] },
      kill: async (pid, opts) => { expect(opts).toEqual({ tree: true, force: true }); calls.push(pid); if (pid === descendantPid) descendantAlive = false },
      isAlive: async (pid) => pid === descendantPid && descendantAlive
    }
    b = new LlamaCppBackend('unused', { pidFile, spawnFn: () => child as unknown as ChildProcess, processTree })
    handler = healthy()
    await load()
    await child.exit(0)
    await b.unloadModel()
    expect(child.kills).toBe(0)
    expect(calls).toContain(descendantPid)
    expect(descendantAlive).toBe(false)
    expect(existsSync(pidFile)).toBe(false)
  })

  it('Q2: reports a surviving descendant after parent exit and retains cleanup identity', async () => {
    const descendantPid = 424245
    let descendantAlive = true
    const startedAt = new Date().toISOString()
    const processTree: ProcessTree = {
      descendants: async () => [{ pid: descendantPid, name: 'fake-child', startedAt }],
      kill: async () => {},
      isAlive: async (pid) => pid === descendantPid && descendantAlive
    }
    b = new LlamaCppBackend('unused', { pidFile, spawnFn: () => child as unknown as ChildProcess, processTree })
    handler = healthy()
    await load()
    await child.exit(0)
    try {
      await expect(b.unloadModel()).rejects.toBeInstanceOf(ServerStuckError)
      expect(existsSync(pidFile)).toBe(true)
    } finally {
      descendantAlive = false // fake process is released before the shared cleanup hook
    }
  })

  it('kills once, resolves when the child exits, clears the pid file; does not report a self-exit', async () => {
    handler = healthy()
    await load()
    await b.unloadModel()
    expect(child.kills).toBe(1)
    expect(child.exitCode === null && child.signalCode === 'SIGTERM').toBe(true)
    expect(existsSync(pidFile)).toBe(false)
    expect(b.lastExit).toBeNull()
  })

  it('is a no-op when nothing is loaded or the child already exited', async () => {
    await expect(b.unloadModel()).resolves.toBeUndefined()
    handler = healthy()
    await load()
    await child.exit(0)
    await expect(b.unloadModel()).resolves.toBeUndefined()
    expect(child.kills).toBe(0)
  })

  it('loadModel on a loaded backend unloads the previous child first', async () => {
    handler = healthy()
    await load()
    const first = child
    child = new FakeChild()
    child.pid = 424247
    await load()
    expect(first.kills).toBe(1)
  })
})
