import { describe, expect, it, vi } from 'vitest'
import type { ChildProcess } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { assertNewArtifact, bounded, idle, launch, outputPathFor, ramFloor, safeEnv, stopOwned, unifiedMemoryKeys, watchRam, type CollisionEvidence, type LaunchProbe } from '../scripts/ab-spill'

const GiB = 1024 ** 3

describe('ab-spill safety helpers (injected fakes; no GPU or executable)', () => {
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

  it('reaps an owned child, escalates to force-kill, and reports a survivor', async () => {
    const fake = (kill: () => void) => ({ pid: 1234, kill }) as unknown as ChildProcess
    let alive = true
    const gentle = vi.fn(() => { alive = false })
    await expect(stopOwned(fake(gentle), Promise.resolve(), () => alive, vi.fn())).resolves.toBeUndefined()
    expect(gentle).toHaveBeenCalledTimes(1)

    alive = true
    const force = vi.fn(() => { alive = false })
    await expect(stopOwned(fake(vi.fn()), Promise.resolve(), () => alive, force)).resolves.toBeUndefined()
    expect(force).toHaveBeenCalledWith(1234)

    await expect(stopOwned(fake(vi.fn()), Promise.resolve(), () => true, vi.fn())).rejects.toThrow(/survived teardown/)
    await expect(stopOwned({ pid: undefined } as ChildProcess, Promise.resolve(), () => false, vi.fn())).rejects.toThrow(/no PID/)
  })
})

describe('ab-spill launch with a fake Node HTTP child (no GPU)', { timeout: 15_000 }, () => {
  const fake = join(__dirname, 'fixtures', 'fake-llama-server.cjs')
  const safeProbe: LaunchProbe = {
    readRam: () => 5 * GiB, countServers: () => 0, devices: () => [],
    startSampler: () => ({ rows: [], stop: async () => {} }),
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
    requestTimeoutMs: 500, loadTimeoutMs: 1_000, healthTimeoutMs: 100, watchIntervalMs: 20,
    settleMs: 0, postMs: 0
  }
  const runFake = (mode: string, promptTokens: number | null) => launch(mode, [fake, '--fake-mode', mode], promptTokens, process.execPath, probe)

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
