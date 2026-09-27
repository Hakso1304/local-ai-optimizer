import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { TypeperfParser, counterPaths, parseColumn, peaks, withNvidia, type TelemetrySample } from '../src/core/telemetry/sampler'

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

  it('hasPidColumns: null before the header, true for a pid capture, false when the pid GPU instances are missing', () => {
    const p = new TypeperfParser({ pid: 26768 })
    expect(p.hasPidColumns).toBeNull()
    for (const l of fx('llama-pid.csv')) p.line(l)
    expect(p.hasPidColumns).toBe(true)
    const q = new TypeperfParser({ pid: 1 }) // header captured with no pid instances (same shape as a too-early start)
    for (const l of fx('idle-nopid.csv')) q.line(l)
    expect(q.hasPidColumns).toBe(false)
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

  it('drops out-of-range percentages (PDH glitch) instead of reporting them', () => {
    const header = String.raw`"(PDH-CSV 4.0)","\\HOST\Processor(_Total)\% Processor Time","\\HOST\GPU Engine(pid_1_luid_0x00000000_0x00016058_phys_0_eng_0_engtype_3D)\Utilization Percentage"`
    const p = new TypeperfParser({ gpuLuid: '0x00000000_0x00016058' })
    const rows = [header, '"t","12.5","13000000000000.0"', '"t","-1","40.0"', '"t","20","40.0"'].map((l) => p.line(l)).filter(Boolean)
    expect(rows).toHaveLength(1) // both glitch rows dropped whole
    expect(rows[0]).toMatchObject({ cpuPct: 20, gpuUtilPct: 40 })
    expect(p.droppedRows).toBe(2)
    expect(p.dropped.glitch).toBe(2)
    // a slight overshoot under full load is real data, clamped — not a glitch row
    expect(p.line('"t","100.4","100.9"')).toMatchObject({ cpuPct: 100, gpuUtilPct: 100 })
  })

  it('accepts trailing -1 placeholders but drops rows that do not line up with the header', () => {
    const header = String.raw`"(PDH-CSV 4.0)","\\H\Memory\Available MBytes","\\H\Processor(_Total)\% Processor Time"`
    const p = new TypeperfParser({})
    p.line(header)
    expect(p.line('"t","11433.0","7.5","-1"')).toMatchObject({ ramAvailBytes: 11433 * 1024 * 1024, cpuPct: 7.5 }) // real: missing Process V2 instance
    expect(p.line('"t","11433.0"')).toBeNull() // short row
    expect(p.line('"t","65.6","11433.0","7.5"')).toBeNull() // extra non-placeholder cell: misaligned
    expect(p.droppedRows).toBe(2)
  })

  it('does not lock onto an adapter where the pid holds only a few MB (ngl=0 case)', () => {
    const H = String.raw`"(PDH-CSV 4.0)","\\H\GPU Adapter Memory(luid_0x00000000_0x00016058_phys_0)\Dedicated Usage","\\H\GPU Adapter Memory(luid_0x00000000_0x000190BD_phys_0)\Dedicated Usage","\\H\GPU Process Memory(pid_7_luid_0x00000000_0x00016058_phys_0)\Dedicated Usage","\\H\GPU Process Memory(pid_7_luid_0x00000000_0x000190BD_phys_0)\Dedicated Usage"`
    const p = new TypeperfParser({ pid: 7 })
    p.line(H)
    const s = p.line('"t","1800000000","0","13090816","13500000"')!
    expect(p.luid).toBe('0x00000000_0x00016058') // adapter-max fallback, not the iGPU
    expect(s.vramDedicatedBytes).toBe(1800000000)
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

  it('withNvidia merges nearest nvidia-smi temp/power into PDH samples and fills a missing GPU util', () => {
    const base = { cpuPct: 1, ramAvailBytes: 1, vramDedicatedBytes: 1, vramSharedBytes: 1, procRamPrivateBytes: 1, procVramDedicatedBytes: 1, procVramSharedBytes: 1 }
    const pdhSamples: TelemetrySample[] = [{ ts: 1000, gpuUtilPct: null, ...base }, { ts: 9000, gpuUtilPct: 50, ...base }]
    const pdh = { samples: pdhSamples, unavailable: {}, errors: [], hasPidColumns: true, restart: () => {}, stop: () => pdhSamples }
    const nv = { samples: [{ ts: 1200, gpuUtilPct: 88, vramUsedBytes: 1, tempC: 71, powerW: 250 }], stop: () => [] }
    const m = withNvidia(pdh, nv)
    expect(m.samples[0]).toMatchObject({ tempC: 71, powerW: 250, gpuUtilPct: 88 })
    expect(m.samples[1].tempC).toBeUndefined() // no nvidia sample within 1.5 s: left absent, not invented
    expect(peaks(m.stop()).max.tempC).toBe(71)
    expect(withNvidia(pdh, null)).toBe(pdh)
    // settled rows are enriched once and the same object is returned on later reads (no O(n·m) re-map per poll)
    const nv2 = { samples: [{ ts: 1100, gpuUtilPct: 1, vramUsedBytes: 1, tempC: 60, powerW: 1 }, { ts: 20000, gpuUtilPct: 1, vramUsedBytes: 1, tempC: 61, powerW: 1 }], stop: () => [] }
    const w = withNvidia(pdh, nv2)
    expect(w.samples[0]).toBe(w.samples[0])
    expect(w.samples[0].tempC).toBe(60)
  })

  it('H7 shape: every row one cell short of the header (an instance vanished) is counted as misaligned in a row', () => {
    // Real shape from #3's heavy runs: samplerErrors "rows dropped: 20 misaligned … (0 kept)".
    const lines = fx('llama-pid.csv')
    const p = new TypeperfParser({ pid: 26768 })
    p.line(lines[0])
    const short = (l: string) => l.split('","').slice(0, -1).join('","') + '"'
    for (const l of lines.slice(1, 4)) expect(p.line(short(l))).toBeNull()
    expect(p.consecutiveMisaligned).toBe(3) // the sampler restarts typeperf at 3
    expect(p.dropped.misaligned).toBe(3)
    expect(p.line(lines[4])).not.toBeNull() // a good row resets the streak
    expect(p.consecutiveMisaligned).toBe(0)
  })
})

