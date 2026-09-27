import { spawn, type ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readVramInUse } from '../src/core/telemetry/sampler'

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>()
  return { ...actual, spawn: vi.fn() }
})

function fakeTypeperf() {
  const child = new EventEmitter() as EventEmitter & {
    stdout: PassThrough; stderr: PassThrough; exitCode: number | null; signalCode: string | null; kill: ReturnType<typeof vi.fn>
  }
  child.stdout = new PassThrough()
  child.stderr = new PassThrough()
  child.exitCode = null
  child.signalCode = null
  const close = () => {
    child.exitCode = 0
    child.stdout.end()
    child.stderr.end()
    child.emit('close', 0)
  }
  child.kill = vi.fn(() => { setImmediate(close); return true })
  vi.mocked(spawn).mockReturnValue(child as unknown as ChildProcess)
  return { child, close }
}

describe('one-shot VRAM probe cancellation (fake typeperf, no GPU)', () => {
  beforeEach(() => vi.mocked(spawn).mockReset())

  it('does not spawn a child for an already aborted probe', async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(readVramInUse(10_000, controller.signal)).resolves.toBeNull()
    expect(spawn).not.toHaveBeenCalled()
  })

  it('aborts an in-flight probe promptly, kills its child, and returns unavailable only after close', async () => {
    const { child } = fakeTypeperf()
    const controller = new AbortController()
    const probe = readVramInUse(10_000, controller.signal)
    expect(spawn).toHaveBeenCalledWith('typeperf', expect.any(Array), { windowsHide: true })
    controller.abort()
    expect(child.kill).toHaveBeenCalledTimes(1)
    const result = await Promise.race([
      probe,
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error('abort left typeperf pending')), 250))
    ])
    expect(result).toBeNull()
    expect(child.exitCode).toBe(0)
    expect(child.stdout.readableEnded).toBe(true)
  })

  it('keeps a completed measurement after a later abort and does not kill the closed child', async () => {
    const { child, close } = fakeTypeperf()
    const controller = new AbortController()
    const probe = readVramInUse(10_000, controller.signal)
    child.stdout.write(String.raw`"(PDH-CSV 4.0)","\\HOST\GPU Adapter Memory(luid_0x00000000_0x00016058_phys_0)\Dedicated Usage"` + '\n')
    child.stdout.write('"t","123456789"\n')
    await new Promise<void>((resolve) => setImmediate(resolve))
    close()
    const measured = await probe
    expect(measured).toEqual({ bytes: 123456789, luid: '0x00000000_0x00016058' })
    controller.abort()
    await Promise.resolve()
    expect(child.kill).not.toHaveBeenCalled()
    await expect(probe).resolves.toEqual(measured)
  })
})
