import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { EventEmitter } from 'node:events'
import type { ChildProcess } from 'node:child_process'
import { describe, expect, it } from 'vitest'
import { parseCudaVersion, parseQueryLine, probeNvidiaSmi, startNvidiaSampler, type Exec } from '../src/core/telemetry/nvidia'
import { pickReleaseAsset, type ReleaseAsset } from '../src/core/runtimes/llamacpp/assets'

const MiB = 1024 ** 2
// Real output on the AMD dev machine (stale NVIDIA driver), captured 2026-09-27 — exit code 4.
const DENIED = 'NVIDIA-SMI has failed because you do not have sufficient permissions. Please try running as an administrator.'

describe('parseQueryLine', () => {
  it('parses a normal row (MiB → bytes)', () => {
    expect(parseQueryLine('0, NVIDIA GeForce RTX 4090, 37, 1234, 24564, 45, 120.50')).toEqual({
      index: 0, name: 'NVIDIA GeForce RTX 4090', gpuUtilPct: 37, vramUsedBytes: 1234 * MiB, vramTotalBytes: 24564 * MiB, tempC: 45, powerW: 120.5
    })
  })
  it('N/A, [N/A], [Not Supported] and glitch values → null', () => {
    expect(parseQueryLine('1, Tesla T4, [N/A], 100, 15360, N/A, [Not Supported]')).toMatchObject({ index: 1, gpuUtilPct: null, vramUsedBytes: 100 * MiB, tempC: null, powerW: null })
    expect(parseQueryLine('0, X, 13099175457793, 1, 2, 999, -5')).toMatchObject({ gpuUtilPct: null, tempC: null, powerW: null })
  })
  it('keeps commas in names; rejects error text and blanks', () => {
    expect(parseQueryLine('0, Quadro RTX 8000, Max-Q, 5, 10, 20, 40, 30.1')?.name).toBe('Quadro RTX 8000, Max-Q')
    expect(parseQueryLine(DENIED)).toBeNull()
    expect(parseQueryLine('')).toBeNull()
  })
  it('two GPUs', () => {
    const rows = ['0, RTX 3080, 90, 9000, 10240, 70, 300.2', '1, RTX 3060, 0, 200, 12288, 35, 15.0'].map(parseQueryLine)
    expect(rows.map((r) => [r?.index, r?.name])).toEqual([[0, 'RTX 3080'], [1, 'RTX 3060']])
  })
  it('reads the driver CUDA version from the banner', () => {
    expect(parseCudaVersion('| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6     |')).toEqual({ major: 12, minor: 6 })
    expect(parseCudaVersion(DENIED)).toBeNull()
  })
})

describe('probeNvidiaSmi', () => {
  const exec = (out: Partial<Awaited<ReturnType<Exec>>>): Exec => async () => ({ code: 0, stdout: '', stderr: '', ...out })
  it('the dev machine case: failure text is unavailable with its reason, whatever the exit code (F5)', async () => {
    for (const code of [0, 4]) {
      const p = await probeNvidiaSmi(exec({ code, stdout: DENIED }))
      expect(p).toMatchObject({ available: false, reason: DENIED })
    }
  })
  it('missing binary / non-zero exit → unavailable', async () => {
    expect(await probeNvidiaSmi(exec({ code: null, stderr: 'nvidia-smi not found' }))).toMatchObject({ available: false, reason: 'exit code n/a: nvidia-smi not found' })
    expect(await probeNvidiaSmi(exec({ code: 9, stdout: 'weird' }))).toMatchObject({ available: false, reason: 'exit code 9: weird' })
  })
  it('healthy output → gpus + CUDA version', async () => {
    const e: Exec = async (_f, args) => args.length
      ? { code: 0, stdout: '0, RTX 4090, 1, 500, 24564, 40, 30.0\n', stderr: '' }
      : { code: 0, stdout: 'Driver Version: 560.94   CUDA Version: 12.6', stderr: '' }
    const p = await probeNvidiaSmi(e)
    expect(p).toMatchObject({ available: true, cudaVersion: { major: 12, minor: 6 } })
    if (p.available) expect(p.gpus[0].vramTotalBytes).toBe(24564 * MiB)
  })
})

describe('startNvidiaSampler', () => {
  const fake = () => {
    const p = Object.assign(new EventEmitter(), { stdout: new PassThrough(), stderr: new PassThrough(), kill: () => true }) as unknown as ChildProcess
    return p
  }
  it('keeps rows of the chosen GPU and flags failure text', async () => {
    const p = fake()
    const s = startNvidiaSampler({ gpuIndex: 1 }, () => p)
    p.stdout!.push('0, A, 10, 100, 1000, 40, 50\n1, B, 20, 200, 2000, 41, 60\n')
    p.stdout!.push(`${DENIED}\n`)
    await new Promise((r) => setImmediate(r))
    expect(s.samples.map((x) => [x.gpuUtilPct, x.vramUsedBytes, x.tempC, x.powerW])).toEqual([[20, 200 * MiB, 41, 60]])
    expect(s.unavailable).toBe(DENIED)
    expect(s.stop()).toHaveLength(1)
  })
  it('early exit is reported', () => {
    const p = fake()
    const s = startNvidiaSampler({}, () => p)
    p.emit('close', 4)
    expect(s.unavailable).toBe('nvidia-smi exited early (code 4) after 0 samples')
  })
})

describe('pickReleaseAsset (real b11208 asset list)', () => {
  const assets = (JSON.parse(readFileSync(join(__dirname, 'fixtures/llamacpp-release-b11208.json'), 'utf8')) as { assets: ReleaseAsset[] }).assets
  const names = (p: ReturnType<typeof pickReleaseAsset>) => [p.main?.name ?? null, p.extra?.name ?? null, p.fallback?.name ?? null]

  it('AMD / Intel / other → Vulkan, with Vulkan fallback', () => {
    for (const vendor of ['amd', 'intel', 'other'] as const) {
      expect(names(pickReleaseAsset(assets, { vendor }))).toEqual(['llama-b11208-bin-win-vulkan-x64.zip', null, 'llama-b11208-bin-win-vulkan-x64.zip'])
    }
  })
  it('NVIDIA, driver CUDA 13 → CUDA 13.4 x64 + its cudart (not the arm64 one)', () => {
    const p = pickReleaseAsset(assets, { vendor: 'nvidia', cudaMajor: 13 })
    expect(names(p)).toEqual(['llama-b11208-bin-win-cuda-13.4-x64.zip', 'cudart-llama-bin-win-cuda-13.4-x64.zip', 'llama-b11208-bin-win-vulkan-x64.zip'])
    expect(p.reason).toMatch(/same directory/)
  })
  it('NVIDIA, driver CUDA 12 or unknown → CUDA 12.4', () => {
    for (const cudaMajor of [12, undefined]) {
      expect(names(pickReleaseAsset(assets, { vendor: 'nvidia', cudaMajor }))[0]).toBe('llama-b11208-bin-win-cuda-12.4-x64.zip')
    }
  })
  it('NVIDIA driver too old for any CUDA build, or cudart missing → Vulkan with a reason', () => {
    expect(pickReleaseAsset(assets, { vendor: 'nvidia', cudaMajor: 11 })).toMatchObject({ main: { name: 'llama-b11208-bin-win-vulkan-x64.zip' }, reason: expect.stringMatching(/CUDA 11/) })
    const noCudart = assets.filter((a) => !a.name.startsWith('cudart-'))
    expect(pickReleaseAsset(noCudart, { vendor: 'nvidia', cudaMajor: 13 }).main?.name).toBe('llama-b11208-bin-win-vulkan-x64.zip')
  })
})
