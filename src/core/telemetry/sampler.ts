import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
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
  /** Fake child seam for deterministic lifecycle tests. */
  spawnFn?: (cmd: string, args: string[], opts: SpawnOptions) => ChildProcess
  stopChild?: (child: ChildProcess) => Promise<void>
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
  /** Why rows were dropped (for RunDetail.samplerErrors): misaligned vs glitch. */
  readonly dropped = { misaligned: 0, glitch: 0 }
  /** Misaligned rows in a row since the last good one: >= 3 means the instance set changed under typeperf (a GPU
   *  engine instance seen at start vanished, so every row is one cell short) and only a fresh typeperf helps. */
  consecutiveMisaligned = 0

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
    if (cells.length < this.cols.length || extra.some((x) => x.replace(/"/g, '').trim() !== '-1')) { this.droppedRows++; this.dropped.misaligned++; this.consecutiveMisaligned++; return null }
    this.consecutiveMisaligned = 0
    const vals = new Map<Col, number>()
    let glitch = false
    this.cols.forEach((c, i) => {
      const v = Number(cells[i]?.trim())
      if (!c || cells[i]?.trim() === '' || !Number.isFinite(v)) return
      const pct = /Percent|^% /.test(c.counter)
      // Real engine counters overshoot 100 a little under full load; only absurd values mark a glitch row. Rows with
      // every engine at 100.x were all dropped before, which left whole 60 s steps with 0 samples (#3's heavy run).
      if (pct && (v < 0 || v > 1000)) glitch = true
      vals.set(c, pct ? Math.min(v, 100) : v)
    })
    // PDH glitch rows (1.3e13 % util in #3's calibration; 62 MB "available" RAM + 794 KB private WS in the E2E run)
    // carry an impossible percentage and garbage in the other cells too: drop the whole row, never report it.
    if (glitch) { this.droppedRows++; this.dropped.glitch++; return null }
    return this.sample(vals)
  }

  /** null until the header arrived; with a pid: whether its GPU Process Memory columns exist. typeperf fixes the
   *  instance set at start, so a sampler started before llama-server created its GPU instances never gets them. */
  get hasPidColumns(): boolean | null {
    return this.cols ? this.cols.some((c) => c?.object === 'GPU Process Memory') : null
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
  /** typeperf failures / early exit, human readable. Survives restart(). */
  readonly errors: string[]
  /** null until typeperf printed its header; false = pid-scoped GPU columns missing (call restart() once the model is loaded). */
  readonly hasPidColumns: boolean | null
  /** Respawn typeperf with the same counters (new instance set); samples collected so far are kept. */
  restart(): void
  stop(): Promise<TelemetrySample[]>
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
  const samples: TelemetrySample[] = []
  const errors: string[] = []
  const si = String(Math.max(1, Math.round((o.intervalMs ?? 1000) / 1000)))
  let stopped = false
  let stopping: Promise<TelemetrySample[]> | null = null
  let parser = new TypeperfParser(o)
  let realigns = 0
  let misalignedBefore = 0 // misaligned rows of replaced typeperf processes, for the final drop note
  let child: ChildProcess
  const children = new Set<ChildProcess>()
  const stopChild = o.stopChild ?? (async (c: ChildProcess) => {
    if (c.exitCode !== null || c.signalCode !== null) return
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('typeperf did not exit after stop')), 3000)
      c.once('close', () => { clearTimeout(timer); resolve() })
      c.kill()
    })
  })

  const spawnOne = () => {
    const p = new TypeperfParser(o) // fresh header per process: the instance set can differ
    parser = p
    const tail: string[] = []
    const c: ChildProcess = (o.spawnFn ?? spawn)('typeperf', [...counterPaths(o), '-si', si, '-sc', String(MAX_SAMPLES)], { windowsHide: true })
    child = c
    children.add(c)
    const kill = () => { c.kill() }
    active.add(kill)
    c.on('close', () => { active.delete(kill); children.delete(c) })
    createInterface({ input: c.stdout! }).on('line', (l) => {
      const s = p.line(l)
      if (s) samples.push(s)
      // H7: whole steps lost every row as misaligned (20/20, 45/45 in #3's heavy runs). Re-snapshot the instance set.
      if (p.consecutiveMisaligned >= 3 && child === c && !stopped && realigns < 5) {
        realigns++
        misalignedBefore += p.dropped.misaligned
        errors.push(`typeperf rows no longer match the header (instance set changed); restarted (${realigns})`)
        spawnOne()
        c.kill()
        return
      }
      else if (l.trim() && !l.startsWith('"')) tail.push(l.trim())
    })
    c.stderr?.on('data', (d: Buffer) => tail.push(d.toString().trim()))
    c.on('error', (e) => errors.push(`typeperf failed to start: ${e.message}`))
    c.on('close', (code) => {
      if (stopped || child !== c) return // stopped or replaced by restart()
      // ponytail: typeperf messages are in the console codepage (cp949 here) and may read garbled; exit code is reliable.
      errors.push(`typeperf exited early (code ${code}) after ${samples.length} samples: ${tail.slice(-3).join(' | ')}`)
      if (!p.cols) for (const f of FIELDS) p.unavailable[f] ??= 'typeperf produced no header'
    })
  }
  spawnOne()

  return {
    samples,
    errors,
    get unavailable() { return parser.unavailable },
    get hasPidColumns() { return parser.hasPidColumns },
    restart() {
      if (stopped) return
      const old = child
      errors.push(`typeperf restarted after ${samples.length} samples (pid columns ${parser.hasPidColumns === false ? 'missing' : 'present'})`)
      spawnOne()
      old.kill()
    },
    stop() {
      if (stopping) return stopping
      stopped = true
      stopping = (async () => {
        await Promise.all([...children].map(stopChild))
        const misaligned = parser.dropped.misaligned + misalignedBefore
        const { glitch } = parser.dropped
        if (misaligned + glitch) errors.push(`typeperf rows dropped: ${misaligned} misaligned with the header, ${glitch} with impossible values (${samples.length} kept)`)
        return samples
      })()
      return stopping
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
    get unavailable() { return pdh.unavailable },
    get hasPidColumns() { return pdh.hasPidColumns },
    restart: () => pdh.restart(),
    errors: pdh.errors,
    async stop() {
      await Promise.resolve(nv.stop())
      active.delete(stopNv)
      await pdh.stop()
      stopped = true
      return view()
    }
  }
}

/** One-shot dedicated VRAM in use on the busiest adapter (= the discrete GPU; iGPUs hold ~0 dedicated), measured
 *  before a session so planning can subtract what other apps already use. null when typeperf fails — never guessed. */
export function readVramInUse(timeoutMs = 20_000, signal?: AbortSignal): Promise<{ bytes: number; luid: string } | null> {
  if (signal?.aborted) return Promise.resolve(null)
  return new Promise((done) => {
    const parser = new TypeperfParser({})
    const child = spawn('typeperf', ['\\GPU Adapter Memory(*)\\Dedicated Usage', '-sc', '1'], { windowsHide: true })
    let result: { bytes: number; luid: string } | null = null
    let cancelled = false, settled = false
    const stop = () => { cancelled = true; child.kill() }
    const timer = setTimeout(stop, timeoutMs)
    const finish = (value: { bytes: number; luid: string } | null) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal?.removeEventListener('abort', stop)
      done(value)
    }
    signal?.addEventListener('abort', stop, { once: true })
    if (signal?.aborted) stop()
    child.stdout && createInterface({ input: child.stdout }).on('line', (l) => {
      const s = parser.line(l)
      if (s?.vramDedicatedBytes != null && parser.luid) result = { bytes: s.vramDedicatedBytes, luid: parser.luid }
    })
    child.on('error', () => finish(null))
    child.on('close', () => finish(cancelled ? null : result))
  })
}
