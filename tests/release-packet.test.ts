import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const GiB = 1024 ** 3
const state = vi.hoisted(() => ({
  inventory: [] as { pid: number; name: string; commandLine: string }[],
  scopes: { Process: [] as string[], User: [] as string[], Machine: [] as string[] },
  ram: 16 * 1024 ** 3,
  vram: { bytes: 1.15 * 1024 ** 3, luid: 'fake-luid' } as { bytes: number; luid: string } | null,
  porcelain: '',
  baseline: { code: 0, passed: 12, failed: 0, skipped: 0, total: 12 }
}))

vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>()
  const fs = await import('node:fs')
  const util = await import('node:util')
  const fake = ((_file: string, argv: string[], _opts: unknown, callback: (error: Error | null, stdout: string, stderr: string) => void) => {
    const output = argv.find((x) => x.startsWith('--outputFile='))?.slice('--outputFile='.length)
    if (!output) throw new Error('unexpected real child-process request')
    fs.writeFileSync(output, JSON.stringify({ numPassedTests: state.baseline.passed, numFailedTests: state.baseline.failed,
      numPendingTests: state.baseline.skipped, numTotalTests: state.baseline.total }))
    const error = state.baseline.code ? Object.assign(new Error('fake test failure'), { code: state.baseline.code }) : null
    queueMicrotask(() => callback(error, '', ''))
  }) as unknown as typeof actual.execFile
  Object.assign(fake, { [util.promisify.custom]: async (file: string, argv: string[]) => {
    if (file === 'git') {
      if (argv[0] === 'rev-parse') return { stdout: 'a'.repeat(40) + '\n', stderr: '' }
      if (argv[0] === 'status') return { stdout: state.porcelain, stderr: '' }
      if (argv[0] === 'ls-files') return { stdout: 'package.json\0', stderr: '' }
    }
    if (file === 'powershell.exe') {
      const script = Buffer.from(argv.at(-1)!, 'base64').toString('utf16le')
      if (script.includes('Win32_Process')) return { stdout: JSON.stringify(state.inventory), stderr: '' }
      if (script.includes('GetEnvironmentVariables')) return { stdout: JSON.stringify(state.scopes), stderr: '' }
      if (script.includes('FAKE_RELEASE_SCAN')) return { stdout: '{}', stderr: '' }
    }
    throw new Error(`unexpected external command: ${file}`)
  } })
  return { ...actual, execFile: fake }
})

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs')>()
  const { Readable } = await import('node:stream')
  return { ...actual, createReadStream: () => Readable.from(['fake fixture bytes']) }
})
vi.mock('node:os', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:os')>()
  return { ...actual, freemem: () => state.ram }
})
vi.mock('../src/core/system/scanner', () => ({
  SCAN_SCRIPT: 'FAKE_RELEASE_SCAN',
  parseScan: () => ({ gpus: { value: [{ name: 'AMD Radeon RX 9070 XT', vendor: 'amd', isIntegrated: false,
    pnpDeviceId: 'PCI\\FAKE', driverVersion: 'fake', dedicatedVramBytes: { value: 16 * GiB, source: 'fake' } }] } })
}))
vi.mock('../src/core/telemetry/sampler', () => ({ readVramInUse: async () => state.vram }))

type Packet = { status: 'READY' | 'HOLD'; checks: { name: string; status: 'READY' | 'HOLD'; reason?: string }[] }
const dirs: string[] = []
const originalArgv = [...process.argv]
const originalExitCode = process.exitCode

beforeEach(() => {
  state.inventory = []
  state.scopes = { Process: [], User: [], Machine: [] }
  state.ram = 16 * GiB
  state.vram = { bytes: 1.15 * GiB, luid: 'fake-luid' }
  state.porcelain = ''
  state.baseline = { code: 0, passed: 12, failed: 0, skipped: 0, total: 12 }
  process.exitCode = 0
  vi.spyOn(console, 'log').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
})
afterEach(() => {
  process.argv = [...originalArgv]
  process.exitCode = originalExitCode
  vi.restoreAllMocks()
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

async function packet(change?: (paths: { model: string; previous: string; out: string }) => void) {
  const dir = mkdtempSync(join(tmpdir(), 'lao-release-packet-'))
  dirs.push(dir)
  const backend = join(dir, 'backend')
  mkdirSync(backend)
  writeFileSync(join(backend, 'llama-server.exe'), 'inert fake executable')
  writeFileSync(join(backend, 'ggml.dll'), 'inert fake DLL')
  writeFileSync(join(backend, 'release-tag.txt'), 'b11208')
  const model = join(dir, 'fake.gguf')
  writeFileSync(model, 'inert fake model')
  const previous = join(dir, 'previous.json')
  writeFileSync(previous, JSON.stringify({ teardown: { at: new Date(Date.now() - 120_000).toISOString(), competingProcesses: 0 } }))
  const out = join(dir, 'packet.json')
  change?.({ model, previous, out })
  process.argv = [process.execPath, 'release-packet.ts', '--out', out, '--backend', backend, '--model', model,
    '--baseline-tests', 'tests/telemetry-vram-cancel.test.ts', '--previous-packet', previous]
  vi.resetModules()
  await import('../scripts/release-packet')
  await vi.waitFor(() => expect(existsSync(out)).toBe(true), { timeout: 5_000 })
  return { result: JSON.parse(readFileSync(out, 'utf8')) as Packet, out }
}

const check = (result: Packet, name: string) => {
  const found = result.checks.find((x) => x.name === name)
  expect(found, `missing release check ${name}`).toBeDefined()
  return found!
}

describe('read-only release packet (injected OS, git, typeperf and baseline runner)', () => {
  it('emits READY only when every fake gate is ready', async () => {
    const { result } = await packet()
    expect(result.status).toBe('READY')
    expect(result.checks).toHaveLength(10)
    expect(result.checks.every((x) => x.status === 'READY')).toBe(true)
    expect(process.exitCode).toBe(0)
  })

  it.each([
    ['llama-server', { pid: 41, name: 'llama-server.exe', commandLine: 'fake' }],
    ['run-session', { pid: 42, name: 'node.exe', commandLine: 'node run-session.ts' }]
  ] as const)('holds when fake %s is in the process inventory', async (_name, processRow) => {
    state.inventory = [processRow]
    const { result } = await packet()
    expect(result.status).toBe('HOLD')
    expect(check(result, 'process inventory').status).toBe('HOLD')
    expect(check(result, 'process inventory after fingerprints').status).toBe('HOLD')
    expect(process.exitCode).toBe(1)
  })

  it.each([
    ['Process', 'ggml_cuda_enable_unified_memory'],
    ['User', 'GgMl_CuDa_EnAbLe_UnIfIeD_MeMoRy'],
    ['Machine', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY']
  ] as const)('holds for a %s-scope unified-memory key, even with an empty value', async (scope, key) => {
    state.scopes[scope] = [key]
    const { result } = await packet()
    expect(result.status).toBe('HOLD')
    expect(check(result, 'environment').status).toBe('HOLD')
  })

  it('holds when available RAM is below 12 GiB', async () => {
    state.ram = 11.99 * GiB
    const { result } = await packet()
    expect(result.status).toBe('HOLD')
    expect(check(result, 'RAM and adapter baseline').reason).toMatch(/below 12 GiB/)
  })

  it('holds for dirty pinned source or scripts porcelain', async () => {
    state.porcelain = ' M src/core/benchmark/session.ts\n?? scripts/extra.ts\n'
    const { result } = await packet()
    expect(result.status).toBe('HOLD')
    expect(check(result, 'source checkout').status).toBe('HOLD')
  })

  it.each([
    ['failed test', { code: 1, passed: 11, failed: 1, skipped: 0, total: 12 }],
    ['unnamed skip', { code: 0, passed: 11, failed: 0, skipped: 1, total: 12 }]
  ] as const)('holds for a baseline %s', async (_name, baseline) => {
    state.baseline = { ...baseline }
    const { result } = await packet()
    expect(result.status).toBe('HOLD')
    expect(check(result, 'fix/test baseline').status).toBe('HOLD')
  })

  it('holds until 60 seconds after a verified teardown', async () => {
    const { result } = await packet(({ previous }) => writeFileSync(previous,
      JSON.stringify({ teardown: { at: new Date(Date.now() - 5_000).toISOString(), competingProcesses: 0 } })))
    expect(result.status).toBe('HOLD')
    expect(check(result, 'previous teardown idle').status).toBe('HOLD')
  })

  it('holds for a missing model path', async () => {
    const { result } = await packet(({ model }) => rmSync(model))
    expect(result.status).toBe('HOLD')
    expect(check(result, 'model GGUF files').status).toBe('HOLD')
  })

  it('holds when the one-shot VRAM baseline is unavailable', async () => {
    state.vram = null
    const { result } = await packet()
    expect(result.status).toBe('HOLD')
    expect(check(result, 'RAM and adapter baseline').reason).toMatch(/dedicated VRAM reading unavailable/)
  })

  it('refuses a pre-existing output name without changing any sentinel bytes', async () => {
    const { result, out } = await packet()
    expect(result.status).toBe('READY')
    const original = readFileSync(out)
    process.exitCode = 0
    vi.resetModules()
    await import('../scripts/release-packet')
    await vi.waitFor(() => expect(process.exitCode).toBe(1), { timeout: 5_000 })
    expect(readFileSync(out)).toEqual(original)
    expect(existsSync(out)).toBe(true)
  })
})
