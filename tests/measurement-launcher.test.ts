import { EventEmitter } from 'node:events'
import { type ChildProcess, type SpawnOptions } from 'node:child_process'
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createManifest, launchManifest, verifyManifest, type MeasurementManifest } from '../scripts/measurement-launcher'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'

const modelScan = vi.hoisted(() => ({ selectedPath: '' }))
vi.mock('../src/core/models/gguf', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/core/models/gguf')>()
  return { ...actual, findGgufModels: async () => modelScan.selectedPath ? [{ name: 'fake-model', path: modelScan.selectedPath }] : [] }
})

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
    exporter: put(join(cwd, 'exporter.ts'), 'fake exporter source'),
    exportOut: join(cwd, 'fresh-export.json'),
    model: put(join(cwd, 'model.gguf'), 'fake model bytes')
  }
  put(join(cwd, 'package-lock.json'), '{}')
  put(join(cwd, 'package.json'), '{"name":"fake-archive"}')
  put(join(cwd, 'node_modules', '.package-lock.json'), '{}')
  return { config, dll, out: join(cwd, 'manifest.json') }
}

function fakeManifest(config: ReturnType<typeof fixture>['config']): MeasurementManifest {
  return { kind: 'local-ai-optimizer/measurement-gate-v1', createdAt: new Date().toISOString(), snapshotHead: config.snapshotHead,
    command: { cwd: config.cwd, dbPath: config.dbPath, args: config.args, exportOut: config.exportOut },
    files: ['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'exporter', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll'].map((role) =>
      ({ role: role as MeasurementManifest['files'][number]['role'], path: config.tsxCli, bytes: 1, sha256: '0'.repeat(64) })) }
}

function storedZip(path: string, name: string, content: Buffer) {
  const file = Buffer.from(name)
  let crc = 0xffffffff
  for (const byte of content) {
    crc ^= byte
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
  }
  crc = (crc ^ 0xffffffff) >>> 0
  const local = Buffer.alloc(30)
  local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4)
  local.writeUInt32LE(crc, 14); local.writeUInt32LE(content.length, 18); local.writeUInt32LE(content.length, 22)
  local.writeUInt16LE(file.length, 26)
  const central = Buffer.alloc(46)
  central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6)
  central.writeUInt32LE(crc, 16); central.writeUInt32LE(content.length, 20); central.writeUInt32LE(content.length, 24)
  central.writeUInt16LE(file.length, 28)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(1, 8); end.writeUInt16LE(1, 10)
  end.writeUInt32LE(central.length + file.length, 12)
  end.writeUInt32LE(local.length + file.length + content.length, 16)
  writeFileSync(path, Buffer.concat([local, file, content, central, file, end]))
}

function scenarioH(pinnedWrong = false) {
  const { config, out } = fixture()
  const runtime = join(config.cwd, 'vendor', 'llama.cpp')
  mkdirSync(runtime, { recursive: true })
  config.runtimeExe = join(runtime, 'llama-server.exe')
  writeFileSync(config.runtimeExe, 'inert Vulkan server')
  writeFileSync(join(runtime, 'ggml.dll'), 'inert Vulkan DLL')
  const wrongRoot = join(config.cwd, 'wrong-root'), modelRoot = join(config.cwd, 'model-root')
  mkdirSync(wrongRoot); mkdirSync(modelRoot)
  const wrongModel = join(wrongRoot, 'fake-model.gguf')
  writeFileSync(wrongModel, 'wrong-model bytes')
  const runnerSelectedModel = join(modelRoot, 'fake-model.gguf')
  writeFileSync(runnerSelectedModel, 'runner-selected bytes')
  modelScan.selectedPath = runnerSelectedModel // injected scanner result, no real model-root scan
  config.model = pinnedWrong ? wrongModel : runnerSelectedModel
  config.args = ['H', '--db', config.dbPath, '--request-cap-ms', '1000', '--ram-abort-gib', '4', '--backend', 'vulkan', '--models', 'fake-model']
  storedZip(config.sourceZip, 'package.json', readFileSync(join(config.cwd, 'package.json')))
  return { config, out, wrongRoot, modelRoot, runnerSelectedModel }
}

describe('measurement launcher gate (fake files and processes only)', () => {
  it('Y2: rejects a manifest pinned to wrong-root when the runner resolver selects model-root', async () => {
    const { config, out, runnerSelectedModel } = scenarioH(true)
    expect(config.model).not.toBe(runnerSelectedModel)
    let manifest: MeasurementManifest
    try { manifest = await createManifest(config, out) }
    catch (error) {
      expect((error as Error).message).toMatch(/model|selection|path/i)
      return // early create-time refusal is also a valid gate
    }
    await expect(verifyManifest(manifest)).rejects.toThrow(/model|selection|path/i)
  })

  it('Y3: rejects a tsconfig.json added after a valid scenario-H manifest', async () => {
    const { config, out } = scenarioH()
    const manifest = await createManifest(config, out)
    await expect(verifyManifest(manifest)).resolves.toBeUndefined()
    writeFileSync(join(config.cwd, 'tsconfig.json'), '{"compilerOptions":{"paths":{"@/*":["other/*"]}}}')
    await expect(verifyManifest(manifest)).rejects.toThrow(/tsconfig|source|configuration|manifest|changed/i)
  })

  it('W2: rejects a changed extracted source file that was not a listed manifest role', async () => {
    const { config, out } = fixture()
    const sourceDir = join(config.cwd, 'src', 'core', 'benchmark')
    mkdirSync(sourceDir, { recursive: true })
    const importedSource = join(sourceDir, 'session.ts')
    writeFileSync(importedSource, 'export const scope = 1\n')
    const manifest = await createManifest(config, out)
    await expect(verifyManifest(manifest)).resolves.toBeUndefined()
    writeFileSync(importedSource, 'export const scope = 2\n')
    await expect(verifyManifest(manifest)).rejects.toThrow(/source|session|changed|hash|manifest/i)
  })

  it.each([
    ['backend', ['--backend', 'hip']],
    ['model', ['--models', 'other.gguf']]
  ] as const)('W3: rejects a changed %s selection after manifest creation', async (_name, changedArgs) => {
    const { config, out } = fixture()
    const manifest = await createManifest(config, out)
    await expect(verifyManifest(manifest)).resolves.toBeUndefined()
    manifest.command.args.push(...changedArgs)
    await expect(verifyManifest(manifest)).rejects.toThrow(/backend|model|command|manifest|selection/i)
  })

  it('pins every role and DLL, refuses overwrite, and detects byte and DLL-set tampering', async () => {
    const { config, dll, out } = fixture()
    const manifest = await createManifest(config, out)
    expect(manifest.files.map((f) => f.role)).toEqual(expect.arrayContaining(['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'exporter', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll']))
    expect(manifest.command.exportOut).toBe(config.exportOut)
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(manifest)
    await expect(verifyManifest(manifest)).resolves.toBeUndefined()
    await expect(createManifest(config, out)).rejects.toThrow(/EEXIST/)
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(manifest)
    writeFileSync(config.model, 'tampered model bytes')
    await expect(verifyManifest(manifest)).rejects.toThrow(/model changed/)
    writeFileSync(config.model, 'fake model bytes')
    writeFileSync(dll, 'tampered DLL bytes')
    await expect(verifyManifest(manifest)).rejects.toThrow(/dll:fake.dll changed/)
    writeFileSync(config.exportOut, 'existing export sentinel')
    await expect(createManifest(config, join(config.cwd, 'second-manifest.json'))).rejects.toThrow(/refusing to overwrite session export/)
    expect(readFileSync(config.exportOut, 'utf8')).toBe('existing export sentinel')
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
    const child = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean; stdout: PassThrough; stderr: PassThrough }
    child.pid = root.pid; child.exitCode = null; child.signalCode = null
    child.stdout = new PassThrough(); child.stderr = new PassThrough()
    child.kill = vi.fn(() => { rootLive = false; child.exitCode = 0; child.emit('close', 0); return true })
    const spawnFn = vi.fn((_exe: string, _argv: string[], opts: SpawnOptions) => {
      expect(opts.shell).toBe(false)
      expect(opts.windowsHide).toBe(true)
      setTimeout(() => { child.stdout.end('SESSION_ID=7\n'); child.stderr.end(); child.exitCode = 0; rootLive = false; child.emit('close', 0) }, 30)
      return child as unknown as ChildProcess
    })
    const verify = vi.fn(async () => {})
    const manifest: MeasurementManifest = { kind: 'local-ai-optimizer/measurement-gate-v1', createdAt: new Date().toISOString(), snapshotHead: config.snapshotHead,
      command: { cwd: config.cwd, dbPath: config.dbPath, args: config.args, exportOut: config.exportOut },
      files: ['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'exporter', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll'].map((role) =>
        ({ role: role as MeasurementManifest['files'][number]['role'], path: config.tsxCli, bytes: 1, sha256: '0'.repeat(64) })) }
    const result = await launchManifest(manifest, { spawnFn: spawnFn as unknown as typeof import('node:child_process').spawn,
      tree, countNamed: async () => 0, readRam: () => 16 * GiB, verify })
    expect(result).toEqual({ exitCode: 0, sessionId: '7', sessionStartLine: 'SESSION_ID=7', minRamGiB: 16 })
    expect(spawnFn).toHaveBeenCalledTimes(1)
    expect(verify).toHaveBeenCalledTimes(2)
    expect(killed).toEqual([descendant.pid])
  })

  it('fails the gate if an owned fake descendant survives teardown', async () => {
    const { config } = fixture()
    const root: ProcessIdentity = { pid: 887766, name: 'node.exe', startedAt: '2026-09-28T00:00:00.000Z' }
    const childId: ProcessIdentity = { pid: 887767, parentPid: root.pid, name: 'fake.exe', startedAt: '2026-09-28T00:00:01.000Z' }
    const p = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean; stdout: PassThrough; stderr: PassThrough }
    p.pid = root.pid; p.exitCode = null; p.signalCode = null; p.kill = () => true
    p.stdout = new PassThrough(); p.stderr = new PassThrough()
    let rootLive = true
    const tree: ProcessTree = {
      descendants: async () => [childId], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root : pid === childId.pid ? childId : null,
      killVerified: async () => true
    }
    const manifest: MeasurementManifest = { kind: 'local-ai-optimizer/measurement-gate-v1', createdAt: new Date().toISOString(), snapshotHead: config.snapshotHead,
      command: { cwd: config.cwd, dbPath: config.dbPath, args: config.args, exportOut: config.exportOut },
      files: ['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'exporter', 'runtimeExe', 'model', 'sourceZip', 'dll:fake.dll'].map((role) =>
        ({ role: role as MeasurementManifest['files'][number]['role'], path: config.tsxCli, bytes: 1, sha256: '0'.repeat(64) })) }
    const spawnFn = () => { setTimeout(() => { p.stdout.end('SESSION_ID=7\n'); p.stderr.end(); rootLive = false; p.exitCode = 0; p.emit('close', 0) }, 30); return p as unknown as ChildProcess }
    await expect(launchManifest(manifest, { spawnFn: spawnFn as typeof import('node:child_process').spawn,
      tree, countNamed: async () => 0, readRam: () => 16 * GiB, verify: async () => {} }))
      .rejects.toThrow(/measurement gate failed after execution/)
  })

  it.each([
    ['two explicit markers', ['SESSION_ID=7', 'SESSION_ID=8']],
    ['marker and event disagree', ['SESSION_ID=7', '{"type":"session:started","sessionId":"8"}']]
  ] as const)('W4: rejects %s before exporting another session', async (_name, lines) => {
    const { config } = fixture()
    const root: ProcessIdentity = { pid: 771122, name: 'node.exe', startedAt: '2026-09-28T00:00:00.000Z' }
    const p = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean; stdout: PassThrough; stderr: PassThrough }
    p.pid = root.pid; p.exitCode = null; p.signalCode = null; p.kill = () => true
    p.stdout = new PassThrough(); p.stderr = new PassThrough()
    let rootLive = true
    const tree: ProcessTree = {
      descendants: async () => [], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root : null,
      killVerified: async () => true
    }
    const spawnFn = () => {
      setTimeout(() => { p.stdout.end(lines.join('\n') + '\n'); p.stderr.end(); rootLive = false; p.exitCode = 0; p.emit('close', 0) }, 30)
      return p as unknown as ChildProcess
    }
    await expect(launchManifest(fakeManifest(config), { spawnFn: spawnFn as typeof import('node:child_process').spawn,
      tree, countNamed: async () => 0, readRam: () => 16 * GiB, verify: async () => {} }))
      .rejects.toThrow(/session|identity|marker|conflict/i)
    expect(existsSync(config.exportOut)).toBe(false)
  })
})
