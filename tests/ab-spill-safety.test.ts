import { describe, expect, it, vi } from 'vitest'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { assertNewArtifact, baseArgv, bounded, bufferExtrema, idle, launch, outputPathFor, ramFloor, reserveLoopbackPort, safeEnv, stopOwned, unifiedMemoryKeys, verifyProps, watchRam, type CollisionEvidence, type LaunchProbe } from '../scripts/ab-spill'
import type { OwnedProcess } from '../scripts/owned-process'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'

const GiB = 1024 ** 3
const fakeNodeTree = (): ProcessTree => {
  const known = new Map<number, ProcessIdentity>()
  const alive = (pid: number) => { try { process.kill(pid, 0); return true } catch { return false } }
  return {
    descendants: async () => [], isAlive: async (pid) => alive(pid),
    kill: async () => { throw new Error('numeric-PID kill must not be used') },
    inspect: async (pid) => {
      if (!alive(pid)) return null
      if (!known.has(pid)) known.set(pid, { pid, name: 'node.exe', startedAt: '2026-09-28T00:00:00.0000000Z' })
      return known.get(pid)!
    },
    killVerified: async (record) => {
      if (known.get(record.pid)?.startedAt !== record.startedAt || !alive(record.pid)) return false
      process.kill(record.pid)
      return true
    }
  }
}

describe('ab-spill safety helpers (injected fakes; no GPU or executable)', () => {
  it('builds every future A/B argv with app mmap/cache policy and verbosity 4', () => {
    const valueAfter = (argv: string[], flag: string) => argv[argv.indexOf(flag) + 1]
    const base = baseArgv(65536)
    const q8 = baseArgv(131072, ['-ctk', 'q8_0', '-ctv', 'q8_0'])
    const ub256 = baseArgv(65536).map((arg, i, argv) => argv[i - 1] === '-ub' ? '256' : arg)
    for (const argv of [base, q8, ub256]) {
      expect(argv.filter((x) => x === '-lv')).toHaveLength(1)
      expect(valueAfter(argv, '-lv')).toBe('4')
      expect(valueAfter(argv, '-lm')).toBe('none')
      expect(valueAfter(argv, '--cache-ram')).toBe('0')
      expect(valueAfter(argv, '-dev')).toBe('Vulkan0')
    }
    expect(valueAfter(base, '-c')).toBe('65536')
    expect(valueAfter(base, '-ub')).toBe('512')
    expect(valueAfter(q8, '-c')).toBe('131072')
    expect(valueAfter(q8, '-ctk')).toBe('q8_0')
    expect(valueAfter(q8, '-ctv')).toBe('q8_0')
    expect(valueAfter(ub256, '-ub')).toBe('256')
  })

  it('uses a repaired output path and refuses the old partial artifact or any existing explicit path', () => {
    const old = 'docs/ab-spill-2026-09-28.json'
    const repaired = outputPathFor([])
    expect(repaired).toBe('docs/ab-spill-repaired-2026-09-28.json')
    expect(repaired).not.toBe(old)
    expect(outputPathFor(['--hip'])).toBe('docs/ab-hip-2026-09-28.json')
    expect(outputPathFor(['--igpu'])).toBe('docs/ab-igpu-2026-09-28.json')
    expect(outputPathFor([old])).toBe(old)
    const exists = vi.fn((path: string) => path === old)
    expect(() => assertNewArtifact(old, exists)).toThrow(/refusing to overwrite existing A\/B artifact/)
    expect(() => assertNewArtifact(repaired, exists)).not.toThrow()
    expect(exists.mock.calls.map(([path]) => path)).toEqual([old, repaired])
  })

  it('removes every Windows spelling of unified-memory env, leaving unrelated keys intact', () => {
    const base = { PATH: 'x', GGML_CUDA_ENABLE_UNIFIED_MEMORY: '1', ggml_cuda_enable_unified_memory: '1', GgMl_CuDa_EnAbLe_UnIfIeD_MeMoRy: '1' }
    expect(unifiedMemoryKeys(base)).toHaveLength(3)
    expect(safeEnv(base)).toEqual({ PATH: 'x' })
    expect(unifiedMemoryKeys({ PATH: 'x' })).toEqual([])
  })

  it('composes request deadline with parent cancellation', async () => {
    const parent = new AbortController()
    const byTime = bounded(parent.signal, 10)
    await new Promise((r) => setTimeout(r, 30))
    expect(byTime.aborted).toBe(true)
    expect(parent.signal.aborted).toBe(false)
    const byParent = bounded(parent.signal, 10_000)
    parent.abort(new Error('cancelled'))
    expect(byParent.aborted).toBe(true)
  })

  it('checks the 4 GiB RAM floor, watches a later drop, aborts and kills exactly once', async () => {
    expect(() => ramFloor(() => 4 * GiB)).not.toThrow()
    expect(() => ramFloor(() => 3.99 * GiB)).toThrow(/< 4 GiB/)
    const values = [5, 5, 3.5, 3].map((x) => x * GiB)
    const read = () => values.shift() ?? 3 * GiB
    const controller = new AbortController()
    const kill = vi.fn()
    const watch = watchRam(controller, kill, read, 5)
    try {
      await new Promise((r) => setTimeout(r, 30))
      expect(controller.signal.aborted).toBe(true)
      expect(kill).toHaveBeenCalledTimes(1)
      expect(watch.reason()).toMatch(/RAM available 3\.5 GiB < 4 GiB/)
      expect(watch.minimum()).toBeLessThan(4 * GiB)
    } finally { watch.stop() }
    expect(kill).toHaveBeenCalledTimes(1)
  })

  it('checks RAM and server ownership throughout idle, including a server appearing later', async () => {
    await expect(idle(1, () => 3 * GiB, () => 0)).rejects.toThrow(/< 4 GiB/)
    await expect(idle(1, () => 5 * GiB, () => 1)).rejects.toThrow(/already running/)
    let reads = 0
    await expect(idle(5, () => (++reads >= 2 ? 3 * GiB : 5 * GiB), () => 0)).rejects.toThrow(/< 4 GiB/)
    let polls = 0
    await expect(idle(5, () => 5 * GiB, () => (++polls >= 2 ? 1 : 0))).rejects.toThrow(/appeared during idle/)
    await expect(idle(1, () => 5 * GiB, () => 0)).resolves.toBeUndefined()
  })

  it('awaits identity-verified owned teardown and reports a survivor', async () => {
    let release!: () => void
    const stop = vi.fn(() => new Promise<void>((done) => { release = done }))
    const owned = { root: { pid: 1234, name: 'node.exe', startedAt: '2026-09-28T00:00:00Z' },
      descendants: new Map(), snapshot: async () => {}, stop } satisfies OwnedProcess
    let settled = false
    const pending = stopOwned(owned).then(() => { settled = true })
    await Promise.resolve()
    expect(settled).toBe(false)
    expect(stop).toHaveBeenCalledTimes(1)
    release()
    await expect(pending).resolves.toBeUndefined()
    expect(settled).toBe(true)
    await expect(stopOwned({ ...owned, stop: async () => { throw new Error('owned child survived teardown') } }))
      .rejects.toThrow(/survived teardown/)
  })

  it('excludes CPU and host buffers from the largest device allocation', () => {
    expect(bufferExtrema([{ dev: 'CPU', mib: 900 }, { dev: 'CPU_Host', mib: 850 }, { dev: 'Vulkan0', mib: 100 }]))
      .toEqual({ largestDeviceBufferMiB: 100, largestHostBufferMiB: 900 })
    expect(bufferExtrema([{ dev: 'CPU', mib: 900 }]))
      .toEqual({ largestDeviceBufferMiB: null, largestHostBufferMiB: 900 })
  })
})

describe('ab-spill launch with a fake Node HTTP child (no GPU)', { timeout: 15_000 }, () => {
  const fake = join(__dirname, 'fixtures', 'fake-llama-server.cjs')
  const safeProbe: LaunchProbe = {
    readRam: () => 5 * GiB, countServers: () => 0, devices: () => [],
    startSampler: () => ({ rows: [], stop: async () => {} }),
    processTree: fakeNodeTree(),
    ownerOfPort: async () => process.env.FAKE_PID_FILE && existsSync(process.env.FAKE_PID_FILE)
      ? Number(readFileSync(process.env.FAKE_PID_FILE, 'utf8')) : null,
    requestTimeoutMs: 100, healthTimeoutMs: 50, loadTimeoutMs: 500, watchIntervalMs: 10
  }
  const args = [fake, '-m', 'fake.gguf', '-c', '2048']
  const isGone = (pid: number) => { try { process.kill(pid, 0); return false } catch { return true } }
  async function withFake(mode: string, work: (pidFile: string, phaseFile: string) => Promise<void>) {
    const dir = mkdtempSync(join(tmpdir(), 'lao-ab-fake-'))
    const pidFile = join(dir, 'pid.txt'), phaseFile = join(dir, 'phase.txt')
    const old = { mode: process.env.FAKE_MODE, pid: process.env.FAKE_PID_FILE, phase: process.env.FAKE_PHASE_FILE }
    process.env.FAKE_MODE = mode; process.env.FAKE_PID_FILE = pidFile; process.env.FAKE_PHASE_FILE = phaseFile
    try { await work(pidFile, phaseFile) } finally {
      for (const [key, value] of [['FAKE_MODE', old.mode], ['FAKE_PID_FILE', old.pid], ['FAKE_PHASE_FILE', old.phase]] as const) {
        if (value === undefined) delete process.env[key]; else process.env[key] = value
      }
      rmSync(dir, { recursive: true, force: true })
    }
  }

  it('bounds a never-responding health endpoint and reaps the spawned child', async () => withFake('hang-health', async (pidFile) => {
    const row = await launch('health hang', args, null, process.execPath, { ...safeProbe, loadTimeoutMs: 300 })
    expect(row.error).toMatch(/health unavailable|abort|timeout/i)
    expect(existsSync(pidFile)).toBe(true)
    expect(isGone(Number(readFileSync(pidFile, 'utf8')))).toBe(true)
  }))

  it('bounds a never-responding completion request and reaps the child', async () => withFake('hang-completion', async (pidFile, phaseFile) => {
    const row = await launch('request hang', args, 16, process.execPath, { ...safeProbe, loadTimeoutMs: 2_000 })
    expect(existsSync(phaseFile)).toBe(true)
    expect(row.error).toMatch(/timeout|aborted/i)
    expect(isGone(Number(readFileSync(pidFile, 'utf8')))).toBe(true)
  }))

  it('RAM watchdog aborts during load and reaps the child', async () => withFake('hang-health', async (pidFile) => {
    const row = await launch('load RAM', args, null, process.execPath, { ...safeProbe, loadTimeoutMs: 2_000,
      readRam: () => existsSync(pidFile) ? 3 * GiB : 5 * GiB })
    expect(row.ramAbort).toMatch(/< 4 GiB/)
    expect(isGone(Number(readFileSync(pidFile, 'utf8')))).toBe(true)
  }))

  it('RAM watchdog aborts an active request and reaps the child', async () => withFake('hang-completion', async (pidFile, phaseFile) => {
    const row = await launch('request RAM', args, 16, process.execPath, { ...safeProbe, requestTimeoutMs: 2_000, loadTimeoutMs: 2_000,
      readRam: () => existsSync(phaseFile) ? 3 * GiB : 5 * GiB })
    expect(existsSync(phaseFile)).toBe(true)
    expect(row.ramAbort).toMatch(/< 4 GiB/)
    expect(isGone(Number(readFileSync(pidFile, 'utf8')))).toBe(true)
  }))
})

describe('ab-spill evidence from a fake Node child (no GPU)', { timeout: 15_000 }, () => {
  const fake = join(__dirname, 'fixtures', 'fake-ab-spill-server.cjs')
  const probe: LaunchProbe = {
    readRam: () => 5 * GiB, countServers: () => 0, devices: () => ['fake MiB'],
    startSampler: () => ({ rows: [], stop: async () => {} }),
    processTree: fakeNodeTree(),
    ownerOfPort: async () => process.env.FAKE_AB_PID_FILE && existsSync(process.env.FAKE_AB_PID_FILE)
      ? Number(readFileSync(process.env.FAKE_AB_PID_FILE, 'utf8')) : null,
    requestTimeoutMs: 500, loadTimeoutMs: 1_000, healthTimeoutMs: 100, watchIntervalMs: 20,
    settleMs: 0, postMs: 0
  }
  const runFake = async (mode: string, promptTokens: number | null, over: Partial<LaunchProbe> = {}) => {
    const dir = mkdtempSync(join(tmpdir(), 'lao-ab-props-'))
    const pidFile = join(dir, 'pid.txt'), previous = process.env.FAKE_AB_PID_FILE
    process.env.FAKE_AB_PID_FILE = pidFile
    try {
      return await launch(mode, [fake, '-m', 'fake.gguf', '-c', '2048', '--fake-mode', mode],
        promptTokens, process.execPath, { ...probe, ...over })
    } finally {
      if (previous === undefined) delete process.env.FAKE_AB_PID_FILE
      else process.env.FAKE_AB_PID_FILE = previous
      rmSync(dir, { recursive: true, force: true })
    }
  }

  it('reserves a real loopback port and accepts only matching model/context props', async () => {
    const port = await reserveLoopbackPort()
    expect(port).toBeGreaterThan(0)
    expect(port).toBeLessThanOrEqual(65535)
    const argv = ['-m', 'fake.gguf', '-c', '2048']
    expect(verifyProps({ model_path: resolve('FAKE.gguf'), default_generation_settings: { n_ctx: 2048 } }, argv))
      .toEqual({ modelPath: resolve('FAKE.gguf'), contextSize: 2048 })
    expect(() => verifyProps({ model_path: resolve('foreign.gguf'), default_generation_settings: { n_ctx: 2048 } }, argv))
      .toThrow(/model mismatch/)
    expect(() => verifyProps({ model_path: resolve('fake.gguf'), default_generation_settings: { n_ctx: 1024 } }, argv))
      .toThrow(/context mismatch/)
  })

  it('verifies the fake listener PID and served props before accepting a load', async () => {
    const row = await runFake('streams', null)
    expect(row.error).toBeNull()
    expect(row.listenerOwnerPid).toBeGreaterThan(0)
    expect(row.servedProps).toEqual({ modelPath: resolve('fake.gguf'), contextSize: 2048 })
  })

  it.each([
    ['wrong-model', /\/props model mismatch/],
    ['wrong-ctx', /\/props context mismatch/]
  ] as const)('rejects %s props before recording a successful load', async (mode, reason) => {
    const row = await runFake(mode, 16)
    expect(row.error).toMatch(reason)
    expect(row.servedProps).toBeNull()
    expect(row.reps).toEqual([])
  })

  it('rejects a foreign listener between reservation and spawn', async () => {
    const owner = vi.fn(async () => 4242)
    const row = await runFake('streams', 16, { ownerOfPort: owner })
    expect(row.error).toMatch(/acquired by another listener before spawn/)
    expect(row.listenerOwnerPid).toBeNull()
    expect(row.servedProps).toBeNull()
    expect(row.reps).toEqual([])
    expect(owner).toHaveBeenCalledTimes(1)
  })

  it('rejects a listener owned by a different PID after health, before props', async () => {
    let checks = 0
    const row = await runFake('streams', 16, { ownerOfPort: async () => ++checks === 1 ? null : 4242 })
    expect(row.error).toMatch(/not owned server PID/)
    expect(row.listenerOwnerPid).toBe(4242)
    expect(row.servedProps).toBeNull()
    expect(row.reps).toEqual([])
  })

  it('never accepts measurements from a replacement listener after the owned child exits during settle', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'lao-ab-replacement-'))
    const pidFile = join(dir, 'pid.txt'), previous = process.env.FAKE_AB_PID_FILE
    process.env.FAKE_AB_PID_FILE = pidFile
    let port = 0, requests = 0, replacementStarted = false
    const replacement = createServer((req, res) => {
      requests++
      res.setHeader('content-type', 'application/json')
      if (req.url === '/tokenize') return res.end(JSON.stringify({ tokens: Array(16).fill(1) }))
      if (req.url === '/completion') return res.end(JSON.stringify({ content: 'foreign', timings: {
        prompt_n: 16, prompt_ms: 12.5, prompt_per_second: 1280, predicted_per_second: 42 } }))
      return res.end(JSON.stringify({ status: 'ok' }))
    })
    const gone = (pid: number) => { try { process.kill(pid, 0); return false } catch { return true } }
    const monitor = setInterval(() => {
      if (!port || !existsSync(pidFile) || replacementStarted) return
      if (!gone(Number(readFileSync(pidFile, 'utf8')))) return
      replacementStarted = true
      replacement.listen(port, '127.0.0.1')
    }, 5)
    try {
      const row = await launch('replacement', [fake, '-m', 'fake.gguf', '-c', '2048', '--fake-mode', 'exit-after-props'],
        16, process.execPath, { ...probe, settleMs: 500,
          reservePort: async () => (port = await reserveLoopbackPort()),
          ownerOfPort: async () => {
            if (replacement.listening) return 4242
            if (existsSync(pidFile)) {
              const pid = Number(readFileSync(pidFile, 'utf8'))
              if (!gone(pid)) return pid
            }
            return null
          } })
      expect(row.error ?? '').toMatch(/server exited|listener|ownership|abort|cancel/i)
      expect(row.reps).toEqual([])
      expect(requests).toBe(0)
    } finally {
      clearInterval(monitor)
      if (replacement.listening) await new Promise<void>((done) => replacement.close(() => done()))
      if (previous === undefined) delete process.env.FAKE_AB_PID_FILE
      else process.env.FAKE_AB_PID_FILE = previous
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('drains and preserves stdout plus stderr and parses buffer declarations from both streams', async () => {
    const row = await runFake('streams', null)
    expect(row.error).toBeNull()
    expect(row.stdoutLog).toContain('Vulkan0 model buffer size = 800.50 MiB')
    expect(row.stderrLog).toContain('Vulkan0 KV buffer size = 256.25 MiB')
    expect(row.buffersMiB).toEqual({
      model: [{ dev: 'Vulkan0', mib: 800.5 }], kv: [{ dev: 'Vulkan0', mib: 256.25 }], compute: [{ dev: 'Vulkan0', mib: 64 }]
    })
    expect(row.largestBufferMiB).toBe(800.5)
    expect(row.stdoutLog).toContain('offloaded 25/25 layers')
    expect(row.stderrLog).toContain('Vulkan0 compute buffer size = 64.00 MiB')
  })

  it('records adapter provenance and device-only max even when a larger host buffer is logged', async () => {
    const selectedAdapter = { luid: 'fake-luid', name: 'Fake Adapter', pnpDeviceId: 'PCI\\FAKE',
      totalBytes: 16 * GiB, totalSource: 'test scanner', mappingStatus: 'inferred-single-discrete' as const }
    const row = await runFake('mixed-buffers', null, { selectedAdapter })
    expect(row.error).toBeNull()
    expect(row.selectedAdapter).toEqual(selectedAdapter)
    expect(row.adapterTotalGiB).toBe(16)
    expect(row.largestBufferMiB).toBe(100)
    expect(row.largestDeviceBufferMiB).toBe(100)
    expect(row.largestHostBufferMiB).toBe(900)
    expect(row.residencyScope).toMatch(/raw counters only.*no capacity/)
  })

  it('waits for the sampler to close before completing fake launch teardown', async () => {
    let release!: () => void
    let stopped = false, finished = false
    const gate = new Promise<void>((resolve) => { release = resolve })
    const pending = runFake('streams', null, { startSampler: () => ({ rows: [], stop: async () => { stopped = true; await gate } }) })
      .then((row) => { finished = true; return row })
    try {
      await vi.waitFor(() => expect(stopped).toBe(true), { timeout: 3000 })
      expect(finished).toBe(false)
    } finally { release() }
    const row = await pending
    expect(row.error).toBeNull()
    expect(finished).toBe(true)
  })

  it('reports missing buffer declarations as unavailable rather than zero', async () => {
    const row = await runFake('no-buffers', null)
    expect(row.error).toBeNull()
    expect(row.buffersMiB).toEqual({ model: [], kv: [], compute: [] })
    expect(row.largestBufferMiB).toBeNull()
    expect(row.pidSharedBaselineGiB).toBeNull()
  })

  it('records server prefill timing separately from unmeasured client TTFT', async () => {
    const measured = await runFake('streams', 16)
    expect(measured.error).toBeNull()
    expect(measured.reps).toHaveLength(2)
    expect(measured.reps.every((r) => r.prefillMs === 12.5 && r.prefillTps === 1280 && r.clientTtftMs === null && r.requestWallMs >= 0)).toBe(true)
    const missing = await runFake('no-timings', 16)
    expect(missing.error).toBeNull()
    expect(missing.reps).toHaveLength(2)
    expect(missing.reps.every((r) => r.prefillMs === null && r.prefillTps === null && r.clientTtftMs === null && r.requestWallMs >= 0)).toBe(true)
  })

  it('idle collision reports injected PID, path and command line without launching or killing it', async () => {
    const evidence: CollisionEvidence = {
      observedAt: '2026-09-28T01:55:57.840Z', tasklist: { observedAt: '2026-09-28T01:55:57.800Z', processes: [{ pid: 4242, image: 'llama-server.exe' }] },
      cim: [{ pid: 4242, parentPid: 1212, path: 'C:/foreign/llama-server.exe', commandLine: 'llama-server.exe --version', creationDate: '2026-09-28T01:55:55Z' }]
    }
    const inspect = vi.fn(() => evidence)
    await expect(idle(0, () => 5 * GiB, () => 1, inspect)).rejects.toThrow(/4242.*foreign.*--version/)
    expect(inspect).toHaveBeenCalledTimes(1)
    let calls = 0
    await expect(idle(5, () => 5 * GiB, () => ++calls > 1 ? 1 : 0, inspect)).rejects.toThrow(/appeared during idle.*4242/)
    expect(inspect).toHaveBeenCalledTimes(2)
  })
})
