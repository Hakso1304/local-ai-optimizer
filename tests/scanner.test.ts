import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { scanSystem, type ScanDeps } from '../src/core/system/scanner'

// Real output captured from the dev machine (RX 9070 XT + iGPU, stale RTX 3080 registry entry).
const fixture = readFileSync(join(__dirname, 'fixtures/scan-rx9070.json'), 'utf8')

const deps = (ps: string | Error, smi: string | Error = new Error('nvidia-smi not found')): ScanDeps => ({
  powershell: async () => { if (ps instanceof Error) throw ps; return ps },
  exec: async () => { if (smi instanceof Error) throw smi; return smi }
})

describe('scanSystem', () => {
  it('parses fixture into a SystemProfile', async () => {
    const p = await scanSystem(deps(fixture))
    expect(p.os.value).toEqual({ name: 'Microsoft Windows 11 Pro', version: '10.0.26200', build: '26200' })
    expect(p.cpu.value).toEqual({ model: 'AMD Ryzen 7 9800X3D 8-Core Processor', physicalCores: 8, logicalCores: 16 })
    expect(p.ram.value?.totalBytes).toBe(32627000 * 1024)
    expect(p.disks.value?.map((d) => d.mount)).toEqual(['C:', 'D:', 'E:', 'G:'])
  })

  it('reports only present GPUs with true VRAM from registry, ignoring stale entries', async () => {
    const gpus = (await scanSystem(deps(fixture))).gpus.value!
    expect(gpus.map((g) => g.name)).toEqual(['AMD Radeon RX 9070 XT', 'AMD Radeon(TM) Graphics'])
    expect(gpus.some((g) => g.name.includes('NVIDIA'))).toBe(false)
    expect(gpus[0]).toMatchObject({ vendor: 'amd', isIntegrated: false })
    expect(gpus[0].dedicatedVramBytes).toMatchObject({ status: 'available', value: 17095983104 }) // not the 4GB WMI overflow
    expect(gpus[1].isIntegrated).toBe(true)
  })

  it('marks CUDA unsupported when no NVIDIA adapter is present', async () => {
    const p = await scanSystem(deps(fixture))
    expect(p.cuda).toMatchObject({ status: 'unsupported', value: { available: false } })
  })

  it('treats nvidia-smi failure as unavailable, not a crash', async () => {
    const withNvidia = fixture.replace('VEN_1002\\u0026DEV_7550', 'VEN_10DE\\u0026DEV_2206')
    const p = await scanSystem(deps(withNvidia, new Error('insufficient permissions')))
    expect(p.gpus.value![0].vendor).toBe('nvidia')
    expect(p.cuda.status).toBe('unavailable')
    expect(p.cuda.value).toBeNull()
    expect(p.cuda.error).toMatch(/insufficient permissions/)
  })

  it('parses CUDA version from nvidia-smi output', async () => {
    const withNvidia = fixture.replace('VEN_1002\\u0026DEV_7550', 'VEN_10DE\\u0026DEV_2206')
    const p = await scanSystem(deps(withNvidia, '| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6 |'))
    expect(p.cuda).toMatchObject({ status: 'available', value: { available: true, version: '12.6' } })
  })

  it('isolates a failed section', async () => {
    const raw = JSON.parse(fixture)
    raw.disks = { ok: false, error: 'Access denied' }
    raw.gpuReg = { ok: false, error: 'registry denied' }
    const p = await scanSystem(deps(JSON.stringify(raw)))
    expect(p.disks).toMatchObject({ status: 'unavailable', value: null, error: 'Access denied' })
    expect(p.gpus.status).toBe('available')
    expect(p.gpus.value![0].dedicatedVramBytes).toMatchObject({ status: 'unavailable', value: null })
    expect(p.os.status).toBe('available')
  })

  it('never throws when PowerShell itself fails', async () => {
    const p = await scanSystem(deps(new Error('powershell.exe timed out after 30000ms')))
    for (const k of ['os', 'cpu', 'ram', 'gpus', 'disks'] as const) {
      expect(p[k].status).toBe('unavailable')
      expect(p[k].error).toMatch(/timed out/)
    }
  })
})
