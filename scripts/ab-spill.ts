// A/B for the ~11.6 GiB dedicated ceiling seen on 2026-09-28 (8B f16 64K spilled at 11.6 GiB with the GPU idle, where
// on 2026-09-27 it held 12.6 GiB with no spill). Usage: npx tsx scripts/ab-spill.ts [out.json]
//   A1 fresh server after >=2 min idle GPU, 36,572-token prompt (the 0.56·ctx fill of the 09-28 runs)
//   A2 fresh server after >=2 min idle GPU, prompt tokenized to 49,152 (0.75·ctx)
//   B1 immediately after a q8_0 -c 131072 load of the same model, then the A1 launch
// Identical argv: -c 65536 -ngl 999 -dev Vulkan0 -t 8 -b 2048 -ub 512 -fa on -fit off --parallel 1
//   -lm none --cache-ram 0. Warmup + 2 reps. The 2026-09-28 repaired run uses these app-comparable mmap/cache flags;
//   docs/ab-spill-2026-09-28.json is the earlier partial run with different argv and is never overwritten.
// --hip: Vulkan-vs-HIP ladder A/B instead — 8B f16 at 32K and 64K (0.56·ctx prompt), identical argv except the exe and
//   -dev Vulkan0 / ROCm0; first reports whether the HIP build enumerates ROCm0 (stops there if not).
// --igpu: can the iGPU's UMA memory replace the CPU layers? Qwen3.8-27B at 8K: (a) -ngl 49 on Vulkan0, rest on CPU
//   (session 3's clean config) vs (b) -ngl 999 -dev Vulkan0,Vulkan1 -ts 49,16 (the 16 CPU layers on the iGPU). The
//   per-device layer split is read from the load log. No per-device KV experiment: b11208 has no KV placement flag
//   (-mg places KV only with -sm row, which Vulkan does not implement).
// Telemetry: typeperf 1 s, per-PID dedicated/shared + adapter dedicated/shared (all LUIDs). At spill onset (per-PID shared
// > first sample + 256 MiB) records per-PID dedicated, adapter dedicated, adapter free and adapter total.
import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { existsSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { freemem } from 'node:os'
import { resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { promisify } from 'node:util'
import { generateFiller } from '../src/core/quality'
import type { ProcessTree } from '../src/core/runtimes/llamacpp'
import { scanSystem } from '../src/core/system/scanner'
import { readVramInUse } from '../src/core/telemetry/sampler'
import { validateHarnessLimits } from './harness-limits'
import { trackOwnedProcess, type OwnedProcess } from './owned-process'

// Same safety bounds as the session harness: every request <= 5 min, host RAM watchdog 4 GiB (kills the server).
const optionValue = (name: string): string | undefined => {
  const i = process.argv.indexOf(name)
  return i < 0 ? undefined : process.argv[i + 1]
}
const limits = validateHarnessLimits({ requestCapMs: Number(optionValue('--request-cap-ms') ?? 300_000), ramAbortGib: Number(optionValue('--ram-abort-gib') ?? 4) })
const REQUEST_TIMEOUT_MS = limits.requestCapMs
const RAM_ABORT_BYTES = limits.ramAbortGib * 1024 ** 3
const LOAD_TIMEOUT_MS = 120_000
const PROBE_TIMEOUT_MS = 30_000
const execFileAsync = promisify(execFile)
const osProbe = (file: string, args: string[], timeout: number, signal?: AbortSignal) =>
  execFileAsync(file, args, { encoding: 'utf8', windowsHide: true, timeout, signal, env: safeEnv(), maxBuffer: 16 * 1024 * 1024 })
/** Free ephemeral loopback port. A later listener check closes the bind race. */
export function reserveLoopbackPort(signal?: AbortSignal): Promise<number> {
  return new Promise((done, fail) => {
    const server = createServer()
    const timer = setTimeout(() => { server.close(); fail(new Error('loopback port reservation timed out')) }, 5_000)
    const abort = () => { server.close(); fail(signal?.reason ?? new Error('port reservation aborted')) }
    const cleanup = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort) }
    if (signal?.aborted) { cleanup(); return abort() }
    signal?.addEventListener('abort', abort, { once: true })
    server.once('error', (e) => { cleanup(); fail(e) })
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') { cleanup(); server.close(); return fail(new Error('loopback reservation returned no TCP port')) }
      server.close((e) => { cleanup(); if (e) fail(e); else done(address.port) })
    })
  })
}
/** Read-only owner query; ambiguity or unavailable OS data fails closed. */
export async function loopbackOwner(port: number, signal?: AbortSignal): Promise<number | null> {
  const command = `$p=@(Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalAddress -in @('127.0.0.1','0.0.0.0','::') } | Select-Object -ExpandProperty OwningProcess -Unique); if($p.Count -eq 1){$p[0]}`
  const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], { encoding: 'utf8', windowsHide: true, timeout: 5_000, signal, env: safeEnv() })
  const raw = stdout.trim()
  return /^\d+$/.test(raw) ? Number(raw) : null
}
export function verifyProps(props: unknown, argv: string[]): { modelPath: string; contextSize: number } {
  const args = argv as string[]
  const model = args[args.indexOf('-m') + 1]
  const ctx = Number(args[args.indexOf('-c') + 1])
  if (!model || !Number.isSafeInteger(ctx) || ctx <= 0) throw new Error('launch argv lacks valid -m and -c identity')
  const p = props as { model_path?: unknown; default_generation_settings?: { n_ctx?: unknown } } | null
  const actual = typeof p?.model_path === 'string' ? p.model_path : null
  if (!actual || resolve(actual).toLowerCase() !== resolve(model).toLowerCase()) throw new Error(`/props model mismatch: expected ${model}, got ${actual ?? 'missing'}`)
  const servedCtx = p?.default_generation_settings?.n_ctx
  if (servedCtx !== ctx) throw new Error(`/props context mismatch: requested ${ctx}, served ${String(servedCtx)}`)
  return { modelPath: actual, contextSize: ctx }
}
/** Windows environment names are case-insensitive. Keep this independent of uncommitted runtime changes. */
export const unifiedMemoryKeys = (base: NodeJS.ProcessEnv = process.env) => Object.keys(base).filter((k) => k.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')
export const safeEnv = (base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv => Object.fromEntries(Object.entries(base).filter(([k]) => k.toUpperCase() !== 'GGML_CUDA_ENABLE_UNIFIED_MEMORY'))
export function ramFloor(read: () => number = freemem): number {
  const available = read()
  if (available < RAM_ABORT_BYTES) throw new Error(`RAM available ${g(available)} GiB < ${limits.ramAbortGib} GiB`)
  return available
}
export const bounded = (signal: AbortSignal, ms = REQUEST_TIMEOUT_MS) => AbortSignal.any([signal, AbortSignal.timeout(ms)])
export async function stopOwned(owned: OwnedProcess): Promise<void> { await owned.stop() }
export function watchRam(controller: AbortController, kill: () => void, read: () => number = freemem, intervalMs = 500) {
  let minimum = read()
  let reason: string | null = null
  const check = () => {
    const free = read()
    minimum = Math.min(minimum, free)
    if (!reason && free < RAM_ABORT_BYTES) {
      reason = `RAM available ${g(free)} GiB < ${limits.ramAbortGib} GiB`
      controller.abort(new Error(reason))
      kill()
    }
  }
  check()
  const timer = setInterval(check, intervalMs)
  return { stop: () => clearInterval(timer), minimum: () => minimum, reason: () => reason }
}

const EXE = 'vendor/llama.cpp/llama-server.exe'
const HIP_EXE = 'vendor/llama.cpp-hip/llama-server.exe'
const HIP = process.argv.includes('--hip')
const IGPU = process.argv.includes('--igpu')
const QWEN = 'D:\\llm-models\\Qwen3.8-27B-UD-Q4_K_M.gguf'
const MODEL = 'D:\\llm-models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
const GiB = 1024 ** 3
const MiB = 1024 ** 2
export interface SelectedAdapter {
  luid: string | null
  name: string | null
  pnpDeviceId: string | null
  totalBytes: number | null
  totalSource: string | null
  mappingStatus: 'inferred-single-discrete' | 'unverified'
}
let selectedAdapter: SelectedAdapter | null = null
export function bufferExtrema(rows: { dev: string; mib: number }[]): { largestDeviceBufferMiB: number | null; largestHostBufferMiB: number | null } {
  const host = (dev: string) => /^CPU(?:$|[_/])|(?:^|[_/])Host$/i.test(dev)
  const max = (xs: number[]) => xs.length ? Math.max(...xs) : null
  return { largestDeviceBufferMiB: max(rows.filter((r) => !host(r.dev)).map((r) => r.mib)), largestHostBufferMiB: max(rows.filter((r) => host(r.dev)).map((r) => r.mib)) }
}
async function inspectSelectedAdapter(): Promise<SelectedAdapter> {
  const [profile, vram] = await Promise.all([scanSystem(), readVramInUse()])
  experimentController?.signal.throwIfAborted()
  const discrete = profile.gpus.value?.filter((g) => !g.isIntegrated) ?? []
  const gpu = discrete.length === 1 ? discrete[0] : null
  return { luid: vram?.luid ?? null, name: gpu?.name ?? null, pnpDeviceId: gpu?.pnpDeviceId ?? null,
    totalBytes: gpu?.dedicatedVramBytes.value ?? null, totalSource: gpu?.dedicatedVramBytes.source ?? null,
    mappingStatus: gpu && vram ? 'inferred-single-discrete' : 'unverified' }
}
export function outputPathFor(args: string[]): string {
  const positionals = args.filter((a, i) => !a.startsWith('--') && !['--request-cap-ms', '--ram-abort-gib'].includes(args[i - 1]))
  return positionals[0] ?? (args.includes('--hip') ? 'docs/ab-hip-2026-09-28.json' : args.includes('--igpu') ? 'docs/ab-igpu-2026-09-28.json' : 'docs/ab-spill-repaired-2026-09-28.json')
}
export function assertNewArtifact(path: string, exists: (path: string) => boolean = existsSync): void {
  if (exists(path)) throw new Error(`refusing to overwrite existing A/B artifact: ${path}`)
}
const out = outputPathFor(process.argv.slice(2))
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const g = (b: number | null) => (b == null ? null : +(b / GiB).toFixed(2))
let experimentController: AbortController | null = null
let activeOwnedChild: ChildProcess | null = null
export const baseArgv = (ctx: number, extra: string[] = [], dev = 'Vulkan0') => ['-m', MODEL, '-c', String(ctx), '-ngl', '999', '-dev', dev, '-t', '8', '-b', '2048', '-ub', '512', '-fa', 'on', '-fit', 'off', '--parallel', '1', '-lm', 'none', '--cache-ram', '0', '-lv', '4', ...extra]

/** Backend's own view (llama-server --list-devices): total/free MiB per device. */
async function listDevices(exe = EXE, signal?: AbortSignal): Promise<string[]> {
  ramFloor()
  const { stdout, stderr } = await osProbe(exe, ['--list-devices'], PROBE_TIMEOUT_MS, signal)
  const lines = `${stdout}\n${stderr}`.split(/\r?\n/).filter((l) => /MiB/.test(l)).map((l) => l.trim())
  if (!lines.length) throw new Error(`${exe} --list-devices returned no devices`)
  return lines
}
async function vulkanHeaps(signal?: AbortSignal): Promise<string> {
  try { ramFloor(); return (await osProbe('vulkaninfo', ['--summary'], PROBE_TIMEOUT_MS, signal)).stdout.split(/\r?\n/).filter((l) => /heap|MEMORY_HEAP|size\s*=/i.test(l)).slice(0, 30).join('\n') } catch { return 'vulkaninfo not available (skipped)' }
}

type SeenServer = { pid: number; image: string }
let lastServerScan: { observedAt: string; processes: SeenServer[] } = { observedAt: '', processes: [] }
const servers = async (signal?: AbortSignal) => {
  const observedAt = new Date().toISOString()
  const processes = (await osProbe('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], 5_000, signal)).stdout
    .split(/\r?\n/).map((l) => /^"(llama-server\.exe)","(\d+)"/i.exec(l)).filter((m): m is RegExpExecArray => m !== null)
    .map((m) => ({ image: m[1], pid: Number(m[2]) }))
  lastServerScan = { observedAt, processes }
  return processes.length
}
export interface CollisionEvidence {
  observedAt: string
  tasklist: { observedAt: string; processes: SeenServer[] }
  cim: { pid: number; parentPid: number | null; path: string | null; commandLine: string | null; creationDate: string | null }[]
  probeError?: string
}
/** Read-only diagnostics for a foreign server. Never terminate a process found here. */
export async function collisionEvidence(signal?: AbortSignal): Promise<CollisionEvidence> {
  const evidence: CollisionEvidence = { observedAt: new Date().toISOString(), tasklist: lastServerScan, cim: [] }
  try {
    const command = '$ErrorActionPreference="Stop"; ConvertTo-Json -InputObject @(Get-CimInstance Win32_Process -Filter "Name=\'llama-server.exe\'" | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine,CreationDate) -Compress -Depth 3'
    const raw = (await osProbe('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', command], 5_000, signal)).stdout.trim()
    const rows = raw ? JSON.parse(raw) as Record<string, unknown>[] | Record<string, unknown> : []
    evidence.cim = (Array.isArray(rows) ? rows : [rows]).map((r) => ({
      pid: Number(r.ProcessId), parentPid: r.ParentProcessId == null ? null : Number(r.ParentProcessId),
      path: typeof r.ExecutablePath === 'string' ? r.ExecutablePath : null,
      commandLine: typeof r.CommandLine === 'string' ? r.CommandLine : null,
      creationDate: typeof r.CreationDate === 'string' ? r.CreationDate : null
    }))
  } catch (e) { evidence.probeError = e instanceof Error ? e.message : String(e) }
  return evidence
}

interface Row { t: number; pidDed: number | null; pidShr: number | null; adapterDed: Record<string, number>; adapterShr: Record<string, number> }
function sampler(pid: number) {
  const ctr = [`\\GPU Process Memory(pid_${pid}_*)\\Dedicated Usage`, `\\GPU Process Memory(pid_${pid}_*)\\Shared Usage`, '\\GPU Adapter Memory(*)\\Dedicated Usage', '\\GPU Adapter Memory(*)\\Shared Usage']
  const c = spawn('typeperf', [...ctr, '-si', '1'], { windowsHide: true, env: safeEnv() })
  const owned = trackOwnedProcess(c)
  void owned.catch(() => {})
  const rows: Row[] = []
  let cols: { obj: string; ctr: string; luid: string | null }[] | null = null
  createInterface({ input: c.stdout }).on('line', (l) => {
    if (!l.startsWith('"')) return
    const cells = l.trim().replace(/^"|"$/g, '').split('","')
    if (!cols) { cols = cells.map((h) => { const m = /^\\\\[^\\]+\\([^(\\]+)(?:\((.*)\))?\\(.+)$/.exec(h); return { obj: m?.[1] ?? '', ctr: m?.[3] ?? '', luid: /luid_(0x[0-9a-f]+_0x[0-9a-f]+)/i.exec(m?.[2] ?? '')?.[1]?.toLowerCase() ?? null } }); return }
    const r: Row = { t: Date.now(), pidDed: null, pidShr: null, adapterDed: {}, adapterShr: {} }
    cols.forEach((col, i) => {
      const v = Number(cells[i]); if (!cells[i]?.trim() || !Number.isFinite(v)) return
      if (col.obj === 'GPU Process Memory') { if (col.ctr === 'Dedicated Usage') r.pidDed = (r.pidDed ?? 0) + v; else r.pidShr = (r.pidShr ?? 0) + v }
      else if (col.obj === 'GPU Adapter Memory' && col.luid) (col.ctr === 'Dedicated Usage' ? r.adapterDed : r.adapterShr)[col.luid] = v
    })
    rows.push(r)
  })
  return { rows, stop: async () => { await stopOwned(await owned) } }
}

async function prompt(port: number, tokens: number, signal: AbortSignal, timeoutMs = REQUEST_TIMEOUT_MS, beforeRequest: () => Promise<void> = async () => {}): Promise<{ text: string; n: number }> {
  const tok = async (s: string) => { await beforeRequest(); return ((await (await fetch(`http://127.0.0.1:${port}/tokenize`, { signal: bounded(signal, timeoutMs), method: 'POST', body: JSON.stringify({ content: s }) })).json()) as { tokens: unknown[] }).tokens.length }
  let est = tokens, text = '', n = 0
  for (let i = 0; i < 6; i++) {
    text = generateFiller(est, 65536).join(' ') + '\n\nContinue the story in the same style:\n'
    n = await tok(text)
    if (Math.abs(n - tokens) <= Math.max(16, tokens * 0.002)) break
    est = Math.max(16, Math.floor(est * (tokens / n)))
  }
  return { text, n }
}

/** Dependency seam for no-GPU fake-server tests. Production always uses the default OS probes and safety limits. */
export interface LaunchProbe {
  readRam?: () => number
  countServers?: () => number | Promise<number>
  devices?: (exe: string) => string[] | Promise<string[]>
  startSampler?: typeof sampler
  requestTimeoutMs?: number
  loadTimeoutMs?: number
  healthTimeoutMs?: number
  watchIntervalMs?: number
  settleMs?: number
  postMs?: number
  reservePort?: (signal: AbortSignal) => Promise<number>
  ownerOfPort?: (port: number, signal: AbortSignal) => Promise<number | null>
  processTree?: ProcessTree
  selectedAdapter?: SelectedAdapter
}
export async function launch(label: string, argv: string[], promptTokens: number | null, exe = EXE, probe: LaunchProbe = {}) {
  const readRam = probe.readRam ?? freemem
  const controller = new AbortController()
  const parentSignal = experimentController?.signal
  const parentAbort = () => controller.abort(parentSignal?.reason ?? new Error('experiment aborted'))
  if (parentSignal?.aborted) parentAbort()
  else parentSignal?.addEventListener('abort', parentAbort, { once: true })
  const countServers = probe.countServers ?? (() => servers(controller.signal))
  const finalServers = probe.countServers ?? (() => servers())
  const requestTimeoutMs = Math.min(probe.requestTimeoutMs ?? REQUEST_TIMEOUT_MS, REQUEST_TIMEOUT_MS)
  const loadTimeoutMs = Math.min(probe.loadTimeoutMs ?? LOAD_TIMEOUT_MS, LOAD_TIMEOUT_MS)
  const healthTimeoutMs = Math.min(probe.healthTimeoutMs ?? 5_000, 5_000)
  const settleMs = Math.min(Math.max(probe.settleMs ?? 2_500, 0), 2_500)
  const postMs = Math.min(Math.max(probe.postMs ?? 1_500, 0), 1_500)
  let port: number | null = null
  let listenerOwnerPid: number | null = null
  let servedProps: { modelPath: string; contextSize: number } | null = null
  const t0 = Date.now()
  const ramBefore = readRam()
  let devicesBefore: string[] = []
  let p: ChildProcess | null = null
  let closed: Promise<void> | null = null
  let ownership: Promise<OwnedProcess> | null = null
  let s: ReturnType<typeof sampler> | null = null
  // Drain both pipes: b11208 may emit load/buffer lines on stdout. Keep the text for audit, even when parsing fails.
  let stdoutLog = '', stderrLog = ''
  const append = (kind: 'stdout' | 'stderr', chunk: Buffer) => {
    const text = chunk.toString('utf8')
    if (kind === 'stdout') stdoutLog += text
    else stderrLog += text
  }
  let error: string | null = null
  let loadMs = 0
  let promptN: number | null = null
  const reps: { decodeTps: number | null; prefillTps: number | null; promptN: number | null; prefillMs: number | null; clientTtftMs: null; requestWallMs: number }[] = []
  const watchdog = watchRam(controller, () => { console.log('RAM WATCHDOG: cancelling request and killing server'); p?.kill() }, readRam, probe.watchIntervalMs ?? 500)
  try {
    ramFloor(readRam)
    if (await countServers() > 0) throw new Error('another llama-server is running')
    devicesBefore = await (probe.devices ?? ((path) => listDevices(path, controller.signal)))(exe)
    ramFloor(readRam)
    port = await (probe.reservePort ?? reserveLoopbackPort)(controller.signal)
    if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new Error(`invalid reserved loopback port: ${String(port)}`)
    // A listener appearing between reservation and spawn is a conflict, never ours.
    if (await (probe.ownerOfPort ?? loopbackOwner)(port, controller.signal) !== null) throw new Error(`reserved port ${port} acquired by another listener before spawn`)
    p = spawn(exe, [...argv, '--port', String(port), '--host', '127.0.0.1'], { windowsHide: true, env: safeEnv() })
    if (experimentController) activeOwnedChild = p
    ownership = trackOwnedProcess(p, probe.processTree)
    void ownership.catch(() => {}) // finally awaits the same rejection and records teardown failure
    let serverClosed = false
    const assertOwnedListener = async () => {
      if (serverClosed || !p?.pid) throw new Error('owned server exited before request')
      const currentOwner = await (probe.ownerOfPort ?? loopbackOwner)(port!, controller.signal)
      listenerOwnerPid = currentOwner
      if (currentOwner !== p.pid) throw new Error(`port ${port} listener PID ${String(currentOwner)} is not owned server PID ${p.pid}`)
    }
    closed = new Promise<void>((resolve) => { p!.once('close', () => { serverClosed = true; resolve() }); p!.once('error', () => { serverClosed = true; resolve() }) })
    await ownership
    p.stdout?.on('data', (d: Buffer) => append('stdout', d))
    p.stderr?.on('data', (d: Buffer) => append('stderr', d))
    const loadStart = Date.now()
    let ready = false
    while (Date.now() - loadStart < loadTimeoutMs) {
      if (controller.signal.aborted) throw controller.signal.reason
      if (serverClosed) throw new Error('server exited during load')
      try {
        if ((await fetch(`http://127.0.0.1:${port}/health`, { signal: bounded(controller.signal, healthTimeoutMs) })).ok) { ready = true; break }
      } catch (e) { if (controller.signal.aborted) throw e }
      await sleep(250)
    }
    if (!ready) throw new Error(`server health unavailable after ${loadTimeoutMs} ms`)
    await assertOwnedListener()
    const props = await (await fetch(`http://127.0.0.1:${port}/props`, { signal: bounded(controller.signal, healthTimeoutMs) })).json()
    servedProps = verifyProps(props, argv)
    await assertOwnedListener()
    loadMs = Date.now() - t0
    if (!p.pid) throw new Error('server started without PID')
    s = (probe.startSampler ?? sampler)(p.pid)
    await sleep(settleMs)
    if (controller.signal.aborted) throw controller.signal.reason
    await assertOwnedListener()
    if (promptTokens) {
      const pr = await prompt(port, promptTokens, controller.signal, requestTimeoutMs, assertOwnedListener)
      promptN = pr.n
      for (const phase of ['warmup', 'rep1', 'rep2']) {
        if (controller.signal.aborted) throw controller.signal.reason
        await assertOwnedListener()
        const r0 = Date.now()
        const j = (await (await fetch(`http://127.0.0.1:${port}/completion`, { signal: bounded(controller.signal, requestTimeoutMs), method: 'POST', body: JSON.stringify({ prompt: pr.text, n_predict: phase === 'warmup' ? 8 : 128, temperature: 0, seed: 1, cache_prompt: false }) })).json()) as { timings?: { prompt_n?: number; prompt_per_second?: number; predicted_per_second?: number; prompt_ms?: number } }
        if (phase !== 'warmup') reps.push({ decodeTps: j.timings?.predicted_per_second ?? null, prefillTps: j.timings?.prompt_per_second ?? null, promptN: j.timings?.prompt_n ?? null, prefillMs: j.timings?.prompt_ms ?? null, clientTtftMs: null, requestWallMs: Date.now() - r0 })
      }
    }
    await sleep(postMs)
    if (controller.signal.aborted) throw controller.signal.reason
    await assertOwnedListener()
  } catch (e) {
    error = e instanceof Error ? e.message : String(e)
  } finally {
    controller.abort()
    if (s) try { await s.stop() } catch (e) { error = [error, `sampler teardown: ${(e as Error).message}`].filter(Boolean).join('; ') }
    if (p && ownership) try { await stopOwned(await ownership) } catch (e) {
      p.kill() // ChildProcess handle only; no numeric-PID kill when identity verification failed
      if (closed) await new Promise<void>((resolve) => {
        const timer = setTimeout(resolve, 10_000)
        void closed!.then(() => { clearTimeout(timer); resolve() })
      })
      error = [error, `server teardown: ${(e as Error).message}`].filter(Boolean).join('; ')
    }
    try { if (await finalServers() > 0) error = [error, 'llama-server remains after this case'].filter(Boolean).join('; ') }
    catch (e) { error = [error, `server reaping could not be verified: ${(e as Error).message}`].filter(Boolean).join('; ') }
    watchdog.stop()
    if (activeOwnedChild === p) activeOwnedChild = null
    parentSignal?.removeEventListener('abort', parentAbort)
  }
  const ramMin = Math.min(watchdog.minimum(), readRam())
  const rows = s?.rows ?? []
  const adapter = probe.selectedAdapter ?? selectedAdapter
  const luid = adapter?.luid ?? null
  const adapterTotal = adapter?.totalBytes ?? null
  const base = rows.find((r) => r.pidShr != null)?.pidShr ?? null
  const onset = base == null ? null : rows.find((r) => r.pidShr != null && r.pidShr > base + 256 * MiB) ?? null
  const peak = (f: (r: Row) => number | null) => { const v = rows.map(f).filter((x): x is number => x != null); return v.length ? Math.max(...v) : null }
  const log = `${stdoutLog}\n${stderrLog}`
  const bufs = (kind: string) => [...log.matchAll(/(\S+) (model|KV|compute) buffer size\s*=\s*([\d.]+) MiB/g)].filter((m) => m[2] === kind).map((m) => ({ dev: m[1], mib: Number(m[3]) }))
  const bufferRows = [...bufs('model'), ...bufs('KV'), ...bufs('compute')]
  const extrema = bufferExtrema(bufferRows)
  const res = {
    label, exe, argv: argv.join(' '), port, listenerOwnerPid, servedProps, error, ramAbort: watchdog.reason(), loadMs, promptTokensRequested: promptTokens, promptTokensActual: promptN, reps,
    samples: rows.length, adapterLuid: luid, selectedAdapter: adapter, adapterTotalGiB: g(adapterTotal),
    peakPidDedicatedGiB: g(peak((r) => r.pidDed)), peakPidSharedGiB: g(peak((r) => r.pidShr)), pidSharedBaselineGiB: g(base),
    peakAdapterDedicatedGiB: g(peak((r) => (luid ? r.adapterDed[luid] ?? null : null))),
    spillOnset: onset ? { atSec: +((onset.t - t0) / 1000).toFixed(1), pidDedicatedGiB: g(onset.pidDed), pidSharedGiB: g(onset.pidShr), adapterDedicatedGiB: g(luid ? onset.adapterDed[luid] : null), adapterFreeGiB: g(luid && adapterTotal != null && onset.adapterDed[luid] != null ? adapterTotal - onset.adapterDed[luid] : null) } : null,
    buffersMiB: { model: bufs('model'), kv: bufs('KV'), compute: bufs('compute') },
    layersPerDevice: [...log.matchAll(/layer\s+\d+ assigned to device (\S+?),?\s/g)].reduce<Record<string, number>>((a, m) => ((a[m[1]] = (a[m[1]] ?? 0) + 1), a), {}),
    offloadLines: log.split(/\r?\n/).filter((l) => /offload(ing|ed) \d+/.test(l)).map((l) => l.trim()),
    largestBufferMiB: extrema.largestDeviceBufferMiB, largestDeviceBufferMiB: extrema.largestDeviceBufferMiB, largestHostBufferMiB: extrema.largestHostBufferMiB,
    residencyScope: 'raw counters only; adapter mapping inferred when one discrete GPU; no capacity or learned-budget claim',
    // Compatibility fields: both logs are fully retained, so truncation is always false.
    stdoutLog, stderrLog, stdoutTruncated: false, stderrTruncated: false,
    listDevicesBefore: devicesBefore, ramAvailBeforeGiB: g(ramBefore), ramAvailMinGiB: g(ramMin)
  }
  console.log(JSON.stringify(res))
  return res
}

export async function idle(ms: number, read: () => number = freemem, countServers: () => number | Promise<number> = () => servers(experimentController?.signal), inspectCollision: () => CollisionEvidence | Promise<CollisionEvidence> = () => collisionEvidence(experimentController?.signal)) {
  experimentController?.signal.throwIfAborted()
  if (await countServers() > 0) throw new Error(`llama-server already running; collision=${JSON.stringify(await inspectCollision())}`)
  console.log(`idle ${ms / 1000}s (GPU quiet)`)
  ramFloor(read)
  const end = Date.now() + ms
  while (Date.now() < end) {
    await sleep(Math.min(500, end - Date.now()))
    experimentController?.signal.throwIfAborted()
    ramFloor(read)
    if (await countServers() > 0) throw new Error(`llama-server appeared during idle; stop this experiment; collision=${JSON.stringify(await inspectCollision())}`)
  }
}
function assertCase(row: Awaited<ReturnType<typeof launch>>) {
  if (row.error || row.ramAbort) throw new Error(`${row.label}: ${row.error ?? row.ramAbort}; stopping experiment`)
}

async function hipAb() {
  // Unified memory would let HIP page to host RAM silently — the ceiling comparison would be meaningless.
  if (unifiedMemoryKeys().length) throw new Error(`${unifiedMemoryKeys().join(', ')} set in this environment; unset it first`)
  const hipDevices = await listDevices(HIP_EXE, experimentController?.signal)
  const vulkanDevices = await listDevices(EXE, experimentController?.signal)
  console.log(`HIP --list-devices: ${JSON.stringify(hipDevices)}`)
  const results = []
  try {
    if (hipDevices.some((l) => /ROCm0/.test(l))) {
      for (const ctx of [32768, 65536]) for (const [exe, dev] of [[EXE, 'Vulkan0'], [HIP_EXE, 'ROCm0']]) {
        await idle(120_000)
        const row = await launch(`${dev} f16 ${ctx / 1024}K, ${Math.round(0.558 * ctx)}-token prompt`, baseArgv(ctx, [], dev), Math.round(0.558 * ctx), exe)
        results.push(row); assertCase(row)
      }
    }
  } finally {
    writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), model: MODEL, hipDevices, vulkanDevices, unifiedMemoryEnvKeys: unifiedMemoryKeys(), safeEnvStripsUnifiedMemory: true, results }, null, 1), { flag: 'wx' })
  }
  console.log(`wrote ${out}; leftover llama-server ${await servers()}`)
}

async function igpuAb() {
  const devices = await listDevices(EXE, experimentController?.signal)
  console.log(`--list-devices: ${JSON.stringify(devices)}`)
  const common = ['-m', QWEN, '-c', '8192', '-t', '8', '-b', '2048', '-ub', '512', '-fa', 'on', '-lm', 'none', '-fit', 'off', '--parallel', '1', '--cache-ram', '0', '-lv', '4']
  const results = []
  try {
    await idle(120_000)
    const a = await launch('a Qwen3.8-27B 8K -ngl 49 Vulkan0, rest on CPU', [...common, '-ngl', '49', '-dev', 'Vulkan0'], 4572)
    results.push(a); assertCase(a)
    if (devices.some((l) => /Vulkan1/.test(l))) {
      await idle(120_000)
      const b = await launch('b Qwen3.8-27B 8K -ngl 999 Vulkan0,Vulkan1 -ts 49,16 (CPU layers on the iGPU)', [...common, '-ngl', '999', '-dev', 'Vulkan0,Vulkan1', '-ts', '49,16'], 4572)
      results.push(b); assertCase(b)
    }
  } finally {
    writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), model: QWEN, devices, results }, null, 1), { flag: 'wx' })
  }
  console.log(`wrote ${out}; leftover llama-server ${await servers()}`)
}

async function main() {
  const controller = new AbortController()
  experimentController = controller
  const supervisor = watchRam(controller, () => activeOwnedChild?.kill())
  let failure: unknown = null
  try {
  controller.signal.throwIfAborted()
  if (unifiedMemoryKeys().length) throw new Error(`${unifiedMemoryKeys().join(', ')} set in this environment; unset it first`)
  assertNewArtifact(out)
  selectedAdapter = await inspectSelectedAdapter()
  console.log(`selected adapter: ${JSON.stringify(selectedAdapter)}`)
  if (HIP) return hipAb()
  if (IGPU) return igpuAb()
  const results = []
  const heaps = await vulkanHeaps(experimentController?.signal)
  let stopReason: string | null = null
  try {
    await idle(120_000)
    const a1 = await launch('A1 fresh, 36,572-token prompt', baseArgv(65536), 36572)
    results.push(a1); assertCase(a1)
    await idle(120_000)
    const a2 = await launch('A2 fresh, 49,152-token prompt (0.75·ctx)', baseArgv(65536), 49152)
    results.push(a2); assertCase(a2)
    await idle(120_000)
    const b1a = await launch('B1a q8_0 -c 131072 load (primes placement), no prompt', baseArgv(131072, ['-ctk', 'q8_0', '-ctv', 'q8_0']), null)
    results.push(b1a); assertCase(b1a)
    const b1b = await launch('B1b A1 launch immediately after the q8_0 128K load', baseArgv(65536), 36572)
    results.push(b1b); assertCase(b1b)
    await idle(120_000)
    // B2: same as A1 but -ub 256 (smaller compute buffer).
    const b2 = await launch('B2 fresh, -ub 256, 36,572-token prompt', baseArgv(65536).map((a, i, xs) => (xs[i - 1] === '-ub' ? '256' : a)), 36572)
    results.push(b2); assertCase(b2)
  } catch (e) {
    stopReason = e instanceof Error ? e.message : String(e)
    throw e
  } finally {
    writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), model: MODEL, vulkaninfoHeaps: heaps, stopReason, results }, null, 1), { flag: 'wx' })
  }
  console.log(`wrote ${out}; leftover llama-server ${await servers()}`)
  } catch (e) { failure = e; throw e }
  finally {
    const cleanupErrors: unknown[] = []
    try { if (await servers() > 0) cleanupErrors.push(new Error('llama-server.exe remains after experiment')) }
    catch (e) { cleanupErrors.push(e) }
    supervisor.stop() // remains active through all case and final process reaping
    experimentController = null
    selectedAdapter = null
    if (cleanupErrors.length) throw new AggregateError(failure === null ? cleanupErrors : [failure, ...cleanupErrors], 'experiment teardown could not be verified')
  }
}

// Importing this module for fake-process safety tests never starts a GPU experiment.
if (process.argv[1] && /(?:^|[\\/])ab-spill\.ts$/i.test(process.argv[1])) void main().catch((e) => { console.error(e); process.exitCode = 1 })
