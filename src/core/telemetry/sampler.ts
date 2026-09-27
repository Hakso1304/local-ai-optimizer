import { spawn, type ChildProcess } from 'node:child_process'
import { createInterface } from 'node:readline'
import type { NvidiaSample } from './nvidia'

// typeperf-based sampler (docs/DESIGN.md §1.1–1.2). typeperf fixes its instance set at start, so start
// this AFTER the llama-server pid is known. Every numeric field: number = MEASURED, null = unavailable
// (reason in Sampler.unavailable). Nothing is ever filled in.
// ponytail: English counter names only; add the WMI (Win32_PerfFormattedData_GPUPerformanceCounters_*)
// fallback when a localized Windows shows up — for now those fields just report unavailable.

export interface TelemetrySample {
  ts: number // epoch ms when the row arrived (typeperf's own timestamp is locale-formatted)
  cpuPct: number | null
  ramAvailBytes: number | null
  gpuUtilPct: number | null // target adapter, max over 3D / Compute N engine groups (sum within a group)
  vramDedicatedBytes: number | null // adapter total — includes other apps
  vramSharedBytes: number | null
  procRamPrivateBytes: number | null
  procVramDedicatedBytes: number | null
  procVramSharedBytes: number | null // per-pid spill signal (adapter shared has a ~1.1–1.5 GB baseline)
  /** nvidia-smi only (merged by withNvidia); absent/null elsewhere — AMD has no non-admin source. */
  tempC?: number | null
  powerW?: number | null
}
export type Field = Exclude<keyof TelemetrySample, 'ts'>
const FIELDS: Field[] = ['cpuPct', 'ramAvailBytes', 'gpuUtilPct', 'vramDedicatedBytes', 'vramSharedBytes', 'procRamPrivateBytes', 'procVramDedicatedBytes', 'procVramSharedBytes', 'tempC', 'powerW']
const PROC_LUID_MIN_BYTES = 128 * 1024 * 1024
const PROC_FIELDS: Field[] = ['procRamPrivateBytes', 'procVramDedicatedBytes', 'procVramSharedBytes']

export interface SamplerOpts {
  pid?: number
  procName?: string // image name without .exe, for \Process V2(name:pid); default llama-server
  gpuLuid?: string // "0x00000000_0x00016058"; default: see pickLuid
  intervalMs?: number // typeperf granularity is whole seconds (min 1)
}

export function counterPaths(o: SamplerOpts): string[] {
  const c = [
    '\\Processor(_Total)\\% Processor Time',
    '\\Memory\\Available MBytes',
    '\\GPU Adapter Memory(*)\\Dedicated Usage',
    '\\GPU Adapter Memory(*)\\Shared Usage'
  ]
  if (o.pid == null) return [...c, '\\GPU Engine(*engtype_3D)\\Utilization Percentage', '\\GPU Engine(*engtype_Compute*)\\Utilization Percentage']
  const p = `pid_${o.pid}_*` // trailing _ so pid 12 doesn't match pid 123
  return [
    ...c,
    `\\GPU Engine(${p})\\Utilization Percentage`,
    `\\GPU Process Memory(${p})\\Dedicated Usage`,
    `\\GPU Process Memory(${p})\\Shared Usage`,
    `\\Process V2(${o.procName ?? 'llama-server'}:${o.pid})\\Working Set - Private`
  ]
}

interface Col { object: string; instance: string; counter: string; luid: string | null; engtype: string | null }

const INSTANCE = /luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+(?:_eng_\d+_engtype_(.+))?/i

/** `\\HOST\Object(instance)\Counter` → parts. */
export function parseColumn(h: string): Col | null {
  const m = /^\\\\[^\\]+\\([^(\\]+)(?:\((.*)\))?\\(.+)$/.exec(h)
  if (!m) return null
  const inst = INSTANCE.exec(m[2] ?? '')
  return { object: m[1], instance: m[2] ?? '', counter: m[3], luid: inst?.[1]?.toLowerCase() ?? null, engtype: inst?.[2] ?? null }
}

const splitCsv = (line: string) => line.trim().replace(/^"|"$/g, '').split('","')

/** Stateful CSV → sample parser (first line is the header). Pure apart from its own state, for fixture tests. */
export class TypeperfParser {
  cols: (Col | null)[] | null = null
  luid: string | null
  private luidLocked: boolean
  readonly unavailable: Partial<Record<Field, string>> = {}
  droppedRows = 0

  constructor(private o: SamplerOpts) {
    this.luid = o.gpuLuid?.toLowerCase() ?? null
    this.luidLocked = this.luid != null
  }

  line(raw: string): TelemetrySample | null {
    if (!raw.startsWith('"')) return null // blank lines, localized status/error text
    const cells = splitCsv(raw)
    if (!this.cols) {
      this.cols = cells.map(parseColumn)
      this.checkHeader()
      return null
    }
    // Exact-instance counters that don't exist (e.g. \Process V2(name:pid) before/after the process) are left out
    // of the header but still emit a "-1" cell at their position; ours is last, so it shows up as a trailing extra.
    // Any other length mismatch means the row doesn't line up with the header: drop it rather than mis-assign cells.
    const extra = cells.slice(this.cols.length)
    if (cells.length < this.cols.length || extra.some((x) => x.replace(/"/g, '').trim() !== '-1')) { this.droppedRows++; return null }
    const vals = new Map<Col, number>()
    let glitch = false
    this.cols.forEach((c, i) => {
      const v = Number(cells[i]?.trim())
      if (!c || cells[i]?.trim() === '' || !Number.isFinite(v)) return
      if (/Percent|^% /.test(c.counter) && (v < 0 || v > 100)) glitch = true
      vals.set(c, v)
    })
    // PDH glitch rows (1.3e13 % util in #3's calibration; 62 MB "available" RAM + 794 KB private WS in the E2E run)
    // carry an impossible percentage and garbage in the other cells too: drop the whole row, never report it.
    if (glitch) { this.droppedRows++; return null }
    return this.sample(vals)
  }

  private checkHeader(): void {
    const has = (obj: string, ctr: string) => this.cols!.some((c) => c?.object === obj && c.counter === ctr)
    const need: [Field, boolean][] = [
      ['cpuPct', has('Processor', '% Processor Time')],
      ['ramAvailBytes', has('Memory', 'Available MBytes')],
      ['gpuUtilPct', has('GPU Engine', 'Utilization Percentage')],
      ['vramDedicatedBytes', has('GPU Adapter Memory', 'Dedicated Usage')],
      ['vramSharedBytes', has('GPU Adapter Memory', 'Shared Usage')],
      ['procRamPrivateBytes', has('Process V2', 'Working Set - Private')],
      ['procVramDedicatedBytes', has('GPU Process Memory', 'Dedicated Usage')],
      ['procVramSharedBytes', has('GPU Process Memory', 'Shared Usage')]
    ]
    for (const [f, ok] of need) {
      if (ok) continue
      this.unavailable[f] = this.o.pid == null && PROC_FIELDS.includes(f)
        ? 'no pid given'
        : 'counter missing from typeperf header (process/instance not present at start, or localized counter names)'
    }
  }

  /** Target adapter: explicit gpuLuid; else where our pid holds the most dedicated VRAM (locked once seen);
   *  else the adapter with the most dedicated usage (the iGPU has ~0 dedicated). DESIGN §1.1. */
  private pickLuid(vals: Map<Col, number>): string | null {
    if (this.luidLocked) return this.luid
    const best = (obj: string) => {
      let top: [string | null, number] = [null, -1]
      for (const [c, v] of vals) if (c.object === obj && c.counter === 'Dedicated Usage' && c.luid && v > top[1]) top = [c.luid, v]
      return top
    }
    const [procLuid, procVal] = best('GPU Process Memory')
    // ≥128 MiB: at ngl=0 llama-server holds ~13 MB on BOTH adapters, which picked the iGPU in the E2E run.
    if (procLuid && procVal >= PROC_LUID_MIN_BYTES) {
      this.luidLocked = true
      return (this.luid = procLuid)
    }
    return (this.luid = best('GPU Adapter Memory')[0])
  }

  private sample(vals: Map<Col, number>): TelemetrySample {
    const luid = this.pickLuid(vals)
    const one = (obj: string, ctr: string) => { for (const [c, v] of vals) if (c.object === obj && c.counter === ctr) return v; return null }
    const sumLuid = (obj: string, ctr: string) => {
      let s: number | null = null
      for (const [c, v] of vals) if (c.object === obj && c.counter === ctr && c.luid === luid) s = (s ?? 0) + v
      return s
    }
    const groups = new Map<string, number>()
    for (const [c, v] of vals) {
      if (c.object !== 'GPU Engine' || c.luid !== luid || !c.engtype || !/^(3D|Compute \d+)$/.test(c.engtype)) continue
      groups.set(c.engtype, (groups.get(c.engtype) ?? 0) + v)
    }
    const mb = one('Memory', 'Available MBytes')
    return {
      ts: Date.now(),
      cpuPct: one('Processor', '% Processor Time'),
      ramAvailBytes: mb == null ? null : mb * 1024 * 1024,
      gpuUtilPct: groups.size ? Math.max(...groups.values()) : null,
      vramDedicatedBytes: sumLuid('GPU Adapter Memory', 'Dedicated Usage'),
      vramSharedBytes: sumLuid('GPU Adapter Memory', 'Shared Usage'),
      procRamPrivateBytes: one('Process V2', 'Working Set - Private'),
      procVramDedicatedBytes: sumLuid('GPU Process Memory', 'Dedicated Usage'),
      procVramSharedBytes: sumLuid('GPU Process Memory', 'Shared Usage')
    }
  }
}

export interface Peaks {
  max: Record<Field, number | null>
  min: Record<Field, number | null>
  meanGpuUtilPct: number | null
  meanCpuPct: number | null
  n: number
}

export function peaks(samples: TelemetrySample[]): Peaks {
  const agg = (f: Field, pick: (a: number, b: number) => number) =>
    samples.reduce<number | null>((acc, s) => (s[f] == null ? acc : acc == null ? s[f] : pick(acc, s[f]!)), null)
  const mean = (f: Field) => {
    const xs = samples.map((s) => s[f]).filter((v): v is number => v != null)
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null
  }
  const max = {} as Record<Field, number | null>
  const min = {} as Record<Field, number | null>
  for (const f of FIELDS) { max[f] = agg(f, Math.max); min[f] = agg(f, Math.min) }
  return { max, min, meanGpuUtilPct: mean('gpuUtilPct'), meanCpuPct: mean('cpuPct'), n: samples.length }
}

export interface Sampler {
  readonly samples: TelemetrySample[]
  readonly unavailable: Partial<Record<Field, string>>
  /** typeperf failures / early exit, human readable. */
  readonly errors: string[]
  stop(): TelemetrySample[]
}

/** Stop hooks of every running sampler child (typeperf, nvidia-smi -lms). */
const active = new Set<() => void>()
/** Kill every running sampler child (app quit). */
export function stopAllSamplers(): void {
  for (const stop of active) stop()
  active.clear()
}
// Safety net if stop() is never reached (hard kill of the app): typeperf exits by itself after this many samples.
const MAX_SAMPLES = 6 * 3600

export function startSampler(o: SamplerOpts = {}): Sampler {
  const parser = new TypeperfParser(o)
  const samples: TelemetrySample[] = []
  const errors: string[] = []
  const tail: string[] = []
  const si = String(Math.max(1, Math.round((o.intervalMs ?? 1000) / 1000)))
  let stopped = false
  const child: ChildProcess = spawn('typeperf', [...counterPaths(o), '-si', si, '-sc', String(MAX_SAMPLES)], { windowsHide: true })
  const kill = () => { child.kill() }
  active.add(kill)
  child.on('close', () => active.delete(kill))
  createInterface({ input: child.stdout! }).on('line', (l) => {
    const s = parser.line(l)
    if (s) samples.push(s)
    else if (l.trim() && !l.startsWith('"')) tail.push(l.trim())
  })
  child.stderr?.on('data', (d: Buffer) => tail.push(d.toString().trim()))
  child.on('error', (e) => errors.push(`typeperf failed to start: ${e.message}`))
  child.on('close', (code) => {
    if (stopped) return
    // ponytail: typeperf messages are in the console codepage (cp949 here) and may read garbled; exit code is reliable.
    errors.push(`typeperf exited early (code ${code}) after ${samples.length} samples: ${tail.slice(-3).join(' | ')}`)
    if (!parser.cols) for (const f of FIELDS) parser.unavailable[f] ??= 'typeperf produced no header'
  })
  return {
    samples,
    unavailable: parser.unavailable,
    errors,
    stop() {
      stopped = true
      child.kill()
      return samples
    }
  }
}

/** Merge nvidia-smi samples (nearest within 1.5 s) into the PDH stream: adds tempC/powerW, fills a missing GPU util.
 *  A row is enriched once, when it is settled (an nvidia sample ≥1.5 s newer exists, or the sampler stopped). */
export function withNvidia(pdh: Sampler, nv: { readonly samples: NvidiaSample[]; stop(): unknown } | null): Sampler {
  if (!nv) return pdh
  const stopNv = () => { nv.stop() }
  active.add(stopNv)
  const enrich = (s: TelemetrySample): TelemetrySample => {
    let best: NvidiaSample | null = null
    for (const n of nv.samples) if (Math.abs(n.ts - s.ts) <= 1500 && (!best || Math.abs(n.ts - s.ts) < Math.abs(best.ts - s.ts))) best = n
    return best ? { ...s, tempC: best.tempC, powerW: best.powerW, gpuUtilPct: s.gpuUtilPct ?? best.gpuUtilPct } : s
  }
  const settled: TelemetrySample[] = []
  let stopped = false
  const view = () => {
    const lastNv = nv.samples.at(-1)?.ts ?? -Infinity
    while (settled.length < pdh.samples.length && (stopped || lastNv >= pdh.samples[settled.length].ts + 1500)) settled.push(enrich(pdh.samples[settled.length]))
    return settled.length === pdh.samples.length ? settled : [...settled, ...pdh.samples.slice(settled.length).map(enrich)]
  }
  return {
    get samples() { return view() },
    unavailable: pdh.unavailable,
    errors: pdh.errors,
    stop() {
      stopNv()
      active.delete(stopNv)
      pdh.stop()
      stopped = true
      return view()
    }
  }
}
