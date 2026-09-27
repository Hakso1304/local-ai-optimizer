import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { createInterface } from 'node:readline'

// nvidia-smi telemetry (DESIGN §1.3). UNTESTED on real NVIDIA hardware: verified against fixture output only.
// On the AMD dev machine nvidia-smi exists (stale driver) and prints "NVIDIA-SMI has failed because you do not have
// sufficient permissions…" — exit code 4 via bash, reported as 0 elsewhere (F5) — so text, not exit code, decides.

export const QUERY = 'index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'

export interface NvidiaGpuRow {
  index: number
  name: string
  gpuUtilPct: number | null
  vramUsedBytes: number | null
  vramTotalBytes: number | null
  tempC: number | null
  powerW: number | null
}

export interface NvidiaSample {
  ts: number
  gpuUtilPct: number | null
  vramUsedBytes: number | null
  tempC: number | null
  powerW: number | null
}

const FAILED = /has failed|sufficient permissions|couldn't communicate with the nvidia driver|no devices were found/i
const MiB = 1024 ** 2

/** "N/A", "[N/A]", "[Not Supported]", "[Unknown Error]", "" → null; out-of-range → null (a glitch is not data). */
function cell(s: string | undefined, max = Infinity): number | null {
  const v = Number((s ?? '').trim())
  return (s ?? '').trim() !== '' && Number.isFinite(v) && v >= 0 && v <= max ? v : null
}

/** One `--format=csv,noheader,nounits` line → row, or null for anything else (error text, blank). Name may contain
 *  commas, so the numeric columns are taken from the end. */
export function parseQueryLine(line: string): NvidiaGpuRow | null {
  const c = line.split(',').map((x) => x.trim())
  if (c.length < 7 || !/^\d+$/.test(c[0])) return null
  const [util, used, total, temp, power] = c.slice(-5)
  const mib = (x: string) => { const v = cell(x); return v === null ? null : v * MiB }
  return {
    index: Number(c[0]), name: c.slice(1, -5).join(', '),
    gpuUtilPct: cell(util, 100), vramUsedBytes: mib(used), vramTotalBytes: mib(total), tempC: cell(temp, 150), powerW: cell(power, 2000)
  }
}

/** "CUDA Version: 12.4" from the plain `nvidia-smi` banner (the max CUDA the driver supports). */
export function parseCudaVersion(text: string): { major: number; minor: number } | null {
  const m = /CUDA Version\s*:\s*(\d+)\.(\d+)/i.exec(text)
  return m ? { major: Number(m[1]), minor: Number(m[2]) } : null
}

export type ExecOut = { code: number | null; stdout: string; stderr: string }
export type Exec = (file: string, args: string[], timeoutMs: number) => Promise<ExecOut>

/** Never rejects: missing binary / timeout become code null + message in stderr. */
const execCapture: Exec = (file, args, timeoutMs) =>
  new Promise((resolve) => {
    execFile(file, args, { timeout: timeoutMs, windowsHide: true, encoding: 'utf8' }, (err, stdout, stderr) => {
      const e = err as (NodeJS.ErrnoException & { killed?: boolean }) | null
      const code = !e ? 0 : typeof e.code === 'number' ? e.code : null
      const why = e && code === null ? (e.code === 'ENOENT' ? 'nvidia-smi not found' : e.killed ? `timed out after ${timeoutMs}ms` : e.message) : ''
      resolve({ code, stdout: stdout ?? '', stderr: [stderr, why].filter(Boolean).join('\n') })
    })
  })

export type NvidiaProbe =
  | { available: true; gpus: NvidiaGpuRow[]; cudaVersion: { major: number; minor: number } | null; source: string }
  | { available: false; reason: string; source: string }

/** Available only if exit code is 0, no failure text, and every query line parses. */
export async function probeNvidiaSmi(exec: Exec = execCapture): Promise<NvidiaProbe> {
  const source = `nvidia-smi --query-gpu=${QUERY}`
  const q = await exec('nvidia-smi', [`--query-gpu=${QUERY}`, '--format=csv,noheader,nounits'], 10_000)
  const text = `${q.stdout}\n${q.stderr}`.trim()
  const firstLine = text.split(/\r?\n/).find((l) => l.trim()) ?? ''
  if (FAILED.test(text)) return { available: false, reason: firstLine, source }
  if (q.code !== 0) return { available: false, reason: `exit code ${q.code ?? 'n/a'}${firstLine ? `: ${firstLine}` : ''}`, source }
  const lines = q.stdout.split(/\r?\n/).filter((l) => l.trim())
  const gpus = lines.map(parseQueryLine)
  if (!gpus.length || gpus.some((g) => g === null)) return { available: false, reason: `unparseable output: ${firstLine.slice(0, 200)}`, source }
  const banner = await exec('nvidia-smi', [], 10_000)
  return { available: true, gpus: gpus as NvidiaGpuRow[], cudaVersion: parseCudaVersion(banner.stdout), source }
}

export interface NvidiaSampler {
  readonly samples: NvidiaSample[]
  /** null while healthy; the reason once nvidia-smi failed or exited early. */
  readonly unavailable: string | null
  stop(): NvidiaSample[]
}

type SpawnFn = (cmd: string, args: string[]) => ChildProcess

/** Streams one row per interval for `gpuIndex` (default 0). spawnFn is a test seam. */
export function startNvidiaSampler(o: { intervalMs?: number; gpuIndex?: number } = {}, spawnFn: SpawnFn = (c, a) => spawn(c, a, { windowsHide: true })): NvidiaSampler {
  const samples: NvidiaSample[] = []
  const gpuIndex = o.gpuIndex ?? 0
  let unavailable: string | null = null
  let stopped = false
  const child = spawnFn('nvidia-smi', [`--query-gpu=${QUERY}`, '--format=csv,noheader,nounits', `-lms`, String(Math.max(100, o.intervalMs ?? 1000))])
  const onLine = (line: string) => {
    if (!line.trim()) return
    if (FAILED.test(line)) { unavailable ??= line.trim(); return }
    const r = parseQueryLine(line)
    if (r && r.index === gpuIndex) samples.push({ ts: Date.now(), gpuUtilPct: r.gpuUtilPct, vramUsedBytes: r.vramUsedBytes, tempC: r.tempC, powerW: r.powerW })
  }
  for (const s of [child.stdout, child.stderr]) if (s) createInterface({ input: s }).on('line', onLine)
  child.on('error', (e) => { unavailable ??= `nvidia-smi failed to start: ${e.message}` })
  child.on('close', (code) => { if (!stopped) unavailable ??= `nvidia-smi exited early (code ${code}) after ${samples.length} samples` })
  return {
    samples,
    get unavailable() { return unavailable },
    stop() {
      stopped = true
      child.kill()
      return samples
    }
  }
}
