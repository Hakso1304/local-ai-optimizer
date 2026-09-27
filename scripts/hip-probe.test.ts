import { EventEmitter } from 'node:events'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ChildProcess } from 'node:child_process'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'
import { runHipProbe } from './hip-probe'

const GiB = 1024 ** 3
const dirs: string[] = []
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })
function fixture() {
  const dir = mkdtempSync(join(tmpdir(), 'lao-hip-probe-')); dirs.push(dir)
  const exe = join(dir, 'llama-server.exe'), out = join(dir, 'probe.json')
  writeFileSync(exe, 'fake HIP executable'); writeFileSync(join(dir, 'fake.dll'), 'fake DLL')
  const identity: ProcessIdentity = { pid: 975310, name: 'fake.exe', startedAt: '2026-09-28T00:00:00.0000000Z' }
  let live = true
  const child = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean; stdout: PassThrough; stderr: PassThrough }
  child.pid = identity.pid; child.exitCode = null; child.signalCode = null; child.stdout = new PassThrough(); child.stderr = new PassThrough()
  child.kill = vi.fn(() => { live = false; child.exitCode = 0; child.emit('close', 0, null); return true })
  const tree: ProcessTree = { descendants: async () => [], kill: async () => { throw new Error('numeric kill forbidden') }, isAlive: async () => live,
    inspect: async () => live ? identity : null, killVerified: async () => { child.kill(); return true } }
  return { dir, exe, out, child, tree, close: () => { live = false; child.exitCode = 0; child.emit('close', 0, null) } }
}

describe('HIP device probe with an injected fake process (no vendor launch)', () => {
  it('launches exactly --list-devices hidden, captures both streams, hashes runtime, and writes once', async () => {
    const f = fixture()
    const spawnFn = vi.fn((_exe, args, opts) => {
      expect(_exe).toBe(f.exe); expect(args).toEqual(['--list-devices'])
      expect(opts).toMatchObject({ windowsHide: true, shell: false })
      expect(Object.keys(opts.env ?? {}).some((k) => k.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')).toBe(false)
      setTimeout(() => { f.child.stdout.write('ROCm0: fake (1024 MiB, 512 MiB free)\n'); f.child.stderr.write('stderr detail\n'); f.close() }, 10)
      return f.child as unknown as ChildProcess
    }) as unknown as typeof import('node:child_process').spawn
    const result = await runHipProbe(f.exe, f.out, { spawnFn, processTree: f.tree, readRam: () => 6 * GiB,
      hash: async () => 'A'.repeat(64), env: { Path: 'x' }, timeoutMs: 1000, watchIntervalMs: 5 })
    expect(result).toMatchObject({ exitCode: 0, error: null, argv: ['--list-devices'], stdout: 'ROCm0: fake (1024 MiB, 512 MiB free)\n', stderr: 'stderr detail\n', ramMinGiB: 6,
      devices: [{ id: 'ROCm0', name: 'fake', backend: 'hip' }], rocm0: { id: 'ROCm0', name: 'fake', backend: 'hip' },
      wrapperSha256: 'A'.repeat(64), runtimeIntegrity: { verified: true, error: null }, teardown: { verified: true, survivors: [] } })
    expect(result.verifiedTeardownAt).toMatch(/^2026-/)
    expect(result.dlls).toHaveLength(1)
    expect(JSON.parse(readFileSync(f.out, 'utf8'))).toEqual(result)
    expect(spawnFn).toHaveBeenCalledTimes(1)
    await expect(runHipProbe(f.exe, f.out, { spawnFn, processTree: f.tree })).rejects.toThrow(/overwrite/)
  })

  it('records timeout and teardown failure without hiding the partial output', async () => {
    const f = fixture()
    const spawnFn = vi.fn(() => f.child as unknown as ChildProcess) as unknown as typeof import('node:child_process').spawn
    await expect(runHipProbe(f.exe, f.out, { spawnFn, processTree: f.tree, readRam: () => 6 * GiB,
      hash: async () => 'A'.repeat(64), timeoutMs: 80, watchIntervalMs: 5 })).rejects.toThrow()
    const row = JSON.parse(readFileSync(f.out, 'utf8')) as { error: string; abortReason: string; argv: string[]; runtimeIntegrity: { verified: boolean }; teardown: { verified: boolean } }
    expect(row.argv).toEqual(['--list-devices'])
    expect(row.error).toMatch(/deadline|teardown|scan/i)
    expect(row.abortReason).toMatch(/deadline/i)
    expect(row.runtimeIntegrity.verified).toBe(true)
    expect(row.teardown.verified).toBe(false)
  })

  it('refuses mixed-case unified-memory keys before spawn', async () => {
    const f = fixture(), spawnFn = vi.fn()
    await expect(runHipProbe(f.exe, f.out, { spawnFn, env: { gGmL_cUdA_EnAbLe_UnIfIeD_MeMoRy: '1' } })).rejects.toThrow(/unified-memory/)
    expect(spawnFn).not.toHaveBeenCalled()
  })
})
