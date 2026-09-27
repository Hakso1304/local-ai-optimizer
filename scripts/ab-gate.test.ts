import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { PassThrough } from 'node:stream'
import type { ChildProcess } from 'node:child_process'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'
import { abCaseSpecs } from './ab-spill'
import { launchAbGate, type AbGateManifest } from './ab-gate'

const GiB = 1024 ** 3
const dirs: string[] = []
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })
const digest = (x: unknown) => createHash('sha256').update(JSON.stringify(x)).digest('hex').toUpperCase()
function fixture(mode: 'hip' | 'igpu' = 'hip') {
  const cwd = mkdtempSync(join(tmpdir(), 'lao-ab-gate-')); dirs.push(cwd)
  const out = join(cwd, 'ab.json'), args = [mode === 'hip' ? '--hip' : '--igpu', '--request-cap-ms', '300000', '--ram-abort-gib', '4', out]
  const command = { cwd, args, out }, caseSpecs = abCaseSpecs(mode)
  const models = [...new Set([...abCaseSpecs('hip'), ...abCaseSpecs('igpu')].map((s) => s.argv[s.argv.indexOf('-m') + 1]))].sort()
  const paths: Record<string, string> = {
    node: process.execPath, packageLock: join(cwd, 'package-lock.json'), installedLock: join(cwd, 'node_modules', '.package-lock.json'),
    tsxCli: join(cwd, 'node_modules', 'tsx', 'dist', 'cli.mjs'), abSpill: join(cwd, 'scripts', 'ab-spill.ts'),
    ownedProcess: join(cwd, 'scripts', 'owned-process.ts'), abGate: join(cwd, 'scripts', 'ab-gate.ts'), sourceZip: join(cwd, 'source.zip'),
    'runtime:vulkan': join(cwd, 'vendor', 'llama.cpp', 'llama-server.exe'), 'runtime:hip': join(cwd, 'vendor', 'llama.cpp-hip', 'llama-server.exe')
  }
  for (const [i, model] of models.entries()) paths[`model:${i}`] = model
  const manifest: AbGateManifest = { kind: 'local-ai-optimizer/ab-gate-v1', head: 'a'.repeat(40), mode, createdAt: new Date().toISOString(),
    command, commandSha256: digest(command), caseSpecs, caseSpecsSha256: digest(caseSpecs),
    dependencyScope: 'node_modules lockfile only; installed file contents not individually pinned',
    files: Object.entries(paths).map(([role, path]) => ({ role, path: resolve(path), bytes: 1, sha256: 'A'.repeat(64) })) }
  return { cwd, out, manifest, paths }
}

describe('standalone A/B gate (fake Node child, no vendor)', () => {
  it('records the exact full HIP and iGPU case argv lists', () => {
    const hip = abCaseSpecs('hip'), igpu = abCaseSpecs('igpu')
    expect(hip).toHaveLength(4); expect(igpu).toHaveLength(2)
    expect(hip.map((c) => c.argv[c.argv.indexOf('-dev') + 1])).toEqual(['Vulkan0', 'ROCm0', 'Vulkan0', 'ROCm0'])
    expect(igpu[1].argv.slice(-6)).toEqual(['-ngl', '999', '-dev', 'Vulkan0,Vulkan1', '-ts', '49,16'])
    for (const spec of [...hip, ...igpu]) {
      expect(spec.argv[spec.argv.indexOf('-lv') + 1]).toBe('4')
      expect(spec.argv[spec.argv.indexOf('-lm') + 1]).toBe('none')
    }
  })

  it('launches one hidden shell-free runner, verifies twice and awaits teardown', async () => {
    const f = fixture()
    const identity: ProcessIdentity = { pid: 885522, name: 'node.exe', startedAt: '2026-09-28T00:00:00.0000000Z' }
    let live = true
    const child = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean; stdout: PassThrough; stderr: PassThrough }
    child.pid = identity.pid; child.exitCode = null; child.signalCode = null; child.stdout = new PassThrough(); child.stderr = new PassThrough()
    child.kill = vi.fn(() => { live = false; child.exitCode = 0; child.emit('close', 0); return true })
    const tree: ProcessTree = { descendants: async () => [], kill: async () => { throw new Error('numeric kill forbidden') }, isAlive: async () => live,
      inspect: async () => live ? identity : null, killVerified: async () => { child.kill(); return true } }
    const spawnFn = vi.fn((_exe, argv, opts) => {
      expect(_exe).toBe(process.execPath)
      expect(argv).toEqual([f.paths.tsxCli, f.paths.abSpill, ...f.manifest.command.args])
      expect(opts).toMatchObject({ cwd: f.cwd, windowsHide: true, shell: false })
      setTimeout(() => { writeFileSync(f.out, '{"results":[]}'); live = false; child.exitCode = 0; child.stdout.end(); child.stderr.end(); child.emit('close', 0) }, 10)
      return child as unknown as ChildProcess
    }) as unknown as typeof import('node:child_process').spawn
    const verify = vi.fn(async (_manifest: AbGateManifest, _signal?: AbortSignal, _allowOutput?: boolean) => {})
    const row = await launchAbGate(f.manifest, { spawnFn, tree, readRam: () => 16 * GiB, countNamed: async () => 0, verify })
    expect(row).toEqual({ exitCode: 0, minRamGiB: 16 })
    expect(verify).toHaveBeenCalledTimes(2)
    expect(verify.mock.calls[1]?.[2]).toBe(true)
  })

  it('rejects a changed case argv or runtime selection before spawning', async () => {
    const f = fixture(), spawnFn = vi.fn()
    f.manifest.caseSpecs[0].argv.push('--changed')
    await expect(launchAbGate(f.manifest, { spawnFn, readRam: () => 16 * GiB, countNamed: async () => 0, verify: async () => {} })).rejects.toThrow(/case argv/)
    expect(spawnFn).not.toHaveBeenCalled()
    expect(dirname(f.manifest.files.find((r) => r.role === 'runtime:hip')!.path)).toContain('llama.cpp-hip')
  })
})
