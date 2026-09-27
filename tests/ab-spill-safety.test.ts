import { describe, expect, it, vi } from 'vitest'
import type { ChildProcess } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { bounded, idle, launch, ramFloor, safeEnv, stopOwned, unifiedMemoryKeys, watchRam, type LaunchProbe } from '../scripts/ab-spill'

const GiB = 1024 ** 3

describe('ab-spill safety helpers (injected fakes; no GPU or executable)', () => {
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
