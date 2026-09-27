import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { TypeperfParser, counterPaths, parseColumn, peaks, type TelemetrySample } from '../src/core/telemetry/sampler'

// Real typeperf output from the dev box (RX 9070 XT luid ..16058, iGPU ..190BD); hostname replaced by HOST.
const fx = (name: string) => readFileSync(join(__dirname, 'fixtures', 'telemetry', name), 'utf8').split('\n').filter(Boolean)
const run = (lines: string[], opts: ConstructorParameters<typeof TypeperfParser>[0]) => {
  const p = new TypeperfParser(opts)
  const samples = lines.map((l) => p.line(l)).filter((s): s is TelemetrySample => s != null)
  return { p, samples }
}

describe('typeperf parser', () => {
  it('parses a pid-scoped llama-server capture: per-process VRAM on the dGPU, util from Compute 0', () => {
    const { p, samples } = run(fx('llama-pid.csv'), { pid: 26768 })
    expect(p.unavailable).toEqual({})
    expect(p.luid).toBe('0x00000000_0x00016058') // most per-pid dedicated, not the iGPU where it holds 118 KB
    const s = samples[2] // 3rd data row: during the measured prompt
    expect(s).toMatchObject({
      cpuPct: 30.528955,
      ramAvailBytes: 16427 * 1024 * 1024,
      gpuUtilPct: 79.326318, // Vulkan compute shows as "Compute 0", not 3D
      vramDedicatedBytes: 1894682624,
      vramSharedBytes: 860495872,
      procVramDedicatedBytes: 670892032,
      procVramSharedBytes: 6844416,
      procRamPrivateBytes: 101216256
    })
  })

  it('parses an idle capture without pid and reports per-process fields as unavailable', () => {
    const { p, samples } = run(fx('idle-nopid.csv'), {})
    expect(samples.length).toBeGreaterThanOrEqual(3)
    expect(p.luid).toBe('0x00000000_0x00016058') // largest adapter dedicated usage
    expect(Object.keys(p.unavailable).sort()).toEqual(['procRamPrivateBytes', 'procVramDedicatedBytes', 'procVramSharedBytes'])
    expect(samples[0].procVramDedicatedBytes).toBeNull()
    expect(samples[0].vramDedicatedBytes).toBeGreaterThan(0)
    expect(samples[0].gpuUtilPct).not.toBeNull()
  })

  it('marks counters absent from the header (e.g. localized names) unavailable, never 0', () => {
    const header = '"(PDH-CSV 4.0)","\\\\HOST\\Memory\\Available MBytes"'
    const { p, samples } = run([header, '"09/27/2026 17:14:13.137","16352.000000","-1"', '"09/27/2026 17:14:14.137"," "'], { pid: 1 })
    expect(p.unavailable.cpuPct).toMatch(/missing/)
    expect(p.unavailable.gpuUtilPct).toMatch(/missing/)
    expect(samples[0]).toMatchObject({ ramAvailBytes: 16352 * 1024 * 1024, cpuPct: null, gpuUtilPct: null, vramDedicatedBytes: null })
    expect(samples[1].ramAvailBytes).toBeNull() // blank cell
  })

  it('ignores non-CSV status lines and builds pid-safe counter paths', () => {
    const p = new TypeperfParser({})
    expect(p.line('Exiting, please wait...')).toBeNull()
    expect(counterPaths({ pid: 12 }).filter((c) => c.includes('pid_12_*'))).toHaveLength(3)
    expect(parseColumn('\\\\HOST\\GPU Engine(pid_1_luid_0x00000000_0x00016058_phys_0_eng_2_engtype_Compute 0)\\Utilization Percentage'))
      .toMatchObject({ object: 'GPU Engine', counter: 'Utilization Percentage', luid: '0x00000000_0x00016058', engtype: 'Compute 0' })
  })

  it('peaks: max/min per field, means ignore nulls', () => {
    const { samples } = run(fx('llama-pid.csv'), { pid: 26768 })
    const pk = peaks(samples)
    expect(pk.n).toBe(samples.length)
    expect(pk.max.procVramDedicatedBytes).toBe(Math.max(...samples.map((s) => s.procVramDedicatedBytes!)))
    expect(peaks([{ ...samples[0], gpuUtilPct: null }]).meanGpuUtilPct).toBeNull()
  })
})
