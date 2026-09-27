import { EventEmitter } from 'node:events'
import { type ChildProcess, type SpawnOptions } from 'node:child_process'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createManifest, launchManifest, verifyManifest, type MeasurementManifest } from '../scripts/measurement-launcher'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'

const GiB = 1024 ** 3
const dirs: string[] = []
afterEach(() => { for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true }) })

function fixture() {
  const cwd = mkdtempSync(join(tmpdir(), 'lao-gate-'))
  dirs.push(cwd)
  const put = (path: string, value: string) => { writeFileSync(path, value); return path }
  mkdirSync(join(cwd, 'node_modules'))
  const runtimeDir = join(cwd, 'fake-runtime')
  mkdirSync(runtimeDir)
  const head = 'a'.repeat(40)
  put(join(cwd, 'HEAD.txt'), head)
  const dbPath = put(join(cwd, 'fake.db'), 'sqlite fixture')
  const runtimeExe = put(join(runtimeDir, 'fake-server.exe'), 'fake executable bytes')
  const dll = put(join(runtimeDir, 'fake.dll'), 'fake dependency bytes')
  const config = {
    snapshotHead: head, cwd, dbPath, runtimeExe,
    args: ['--db', dbPath, '--request-cap-ms', '1000', '--ram-abort-gib', '4'],
    sourceZip: put(join(cwd, 'source.zip'), 'fake source archive'),
    tsxCli: put(join(cwd, 'tsx-cli.js'), 'fake tsx'),
    launcher: put(join(cwd, 'launcher.ts'), 'fake launcher source'),
    runner: put(join(cwd, 'runner.ts'), 'fake runner source'),
    model: put(join(cwd, 'model.gguf'), 'fake model bytes')
  }
  put(join(cwd, 'package-lock.json'), '{}')
  put(join(cwd, 'node_modules', '.package-lock.json'), '{}')
  return { config, dll, out: join(cwd, 'manifest.json') }
}

describe('measurement launcher gate (fake files and processes only)', () => {
  it('pins every role and DLL, refuses overwrite, and detects byte and DLL-set tampering', async () => {
    const { config, dll, out } = fixture()
    const manifest = await createManifest(config, out)
    expect(manifest.files.map((f) => f.role)).toEqual(expect.arrayContaining(['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll']))
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(manifest)
    await expect(verifyManifest(manifest)).resolves.toBeUndefined()
    await expect(createManifest(config, out)).rejects.toThrow(/EEXIST/)
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(manifest)
    writeFileSync(config.model, 'tampered model bytes')
    await expect(verifyManifest(manifest)).rejects.toThrow(/model changed/)
    writeFileSync(config.model, 'fake model bytes')
    writeFileSync(dll, 'tampered DLL bytes')
    await expect(verifyManifest(manifest)).rejects.toThrow(/dll:fake.dll changed/)
  })

  it('uses one hidden shell-free child, verifies on both sides, and awaits identity-bound teardown', async () => {
    const { config } = fixture()
    const root: ProcessIdentity = { pid: 332211, name: 'node.exe', startedAt: '2026-09-28T00:00:00.000Z' }
    const descendant: ProcessIdentity = { pid: 332212, parentPid: root.pid, name: 'fake.exe', startedAt: '2026-09-28T00:00:01.000Z' }
    let rootLive = true, childLive = true
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async () => [descendant], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root : pid === descendant.pid && childLive ? descendant : null,
      killVerified: async (record) => { killed.push(record.pid); if (record.pid === root.pid) rootLive = false; if (record.pid === descendant.pid) childLive = false; return true }
    }
    const child = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean }
    child.pid = root.pid; child.exitCode = null; child.signalCode = null
    child.kill = vi.fn(() => { rootLive = false; child.exitCode = 0; child.emit('close', 0); return true })
    const spawnFn = vi.fn((_exe: string, _argv: string[], opts: SpawnOptions) => {
      expect(opts.shell).toBe(false)
      expect(opts.windowsHide).toBe(true)
      setTimeout(() => { child.exitCode = 0; rootLive = false; child.emit('close', 0) }, 30)
      return child as unknown as ChildProcess
    })
    const verify = vi.fn(async () => {})
    const manifest: MeasurementManifest = { kind: 'local-ai-optimizer/measurement-gate-v1', createdAt: new Date().toISOString(), snapshotHead: config.snapshotHead,
      command: { cwd: config.cwd, dbPath: config.dbPath, args: config.args },
      files: ['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll'].map((role) =>
        ({ role: role as MeasurementManifest['files'][number]['role'], path: config.tsxCli, bytes: 1, sha256: '0'.repeat(64) })) }
    const result = await launchManifest(manifest, { spawnFn: spawnFn as unknown as typeof import('node:child_process').spawn,
      tree, countNamed: async () => 0, readRam: () => 16 * GiB, verify })
    expect(result).toEqual({ exitCode: 0, minRamGiB: 16 })
    expect(spawnFn).toHaveBeenCalledTimes(1)
    expect(verify).toHaveBeenCalledTimes(2)
    expect(killed).toEqual([descendant.pid])
  })

  it('fails the gate if an owned fake descendant survives teardown', async () => {
    const { config } = fixture()
    const root: ProcessIdentity = { pid: 887766, name: 'node.exe', startedAt: '2026-09-28T00:00:00.000Z' }
    const childId: ProcessIdentity = { pid: 887767, parentPid: root.pid, name: 'fake.exe', startedAt: '2026-09-28T00:00:01.000Z' }
    const p = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean }
    p.pid = root.pid; p.exitCode = null; p.signalCode = null; p.kill = () => true
    let rootLive = true
    const tree: ProcessTree = {
      descendants: async () => [childId], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root : pid === childId.pid ? childId : null,
      killVerified: async () => true
    }
    const manifest: MeasurementManifest = { kind: 'local-ai-optimizer/measurement-gate-v1', createdAt: new Date().toISOString(), snapshotHead: config.snapshotHead,
      command: { cwd: config.cwd, dbPath: config.dbPath, args: config.args },
      files: ['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll'].map((role) =>
        ({ role: role as MeasurementManifest['files'][number]['role'], path: config.tsxCli, bytes: 1, sha256: '0'.repeat(64) })) }
    const spawnFn = () => { setTimeout(() => { rootLive = false; p.exitCode = 0; p.emit('close', 0) }, 30); return p as unknown as ChildProcess }
    await expect(launchManifest(manifest, { spawnFn: spawnFn as typeof import('node:child_process').spawn,
      tree, countNamed: async () => 0, readRam: () => 16 * GiB, verify: async () => {} }))
      .rejects.toThrow(/measurement gate failed after execution/)
  })
})
