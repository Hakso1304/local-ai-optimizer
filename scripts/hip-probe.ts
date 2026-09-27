/** Supervised HIP enumeration. This launches only llama-server --list-devices. */
import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { freemem } from 'node:os'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { ProcessTree } from '../src/core/runtimes/llamacpp'
import { parseDevices } from '../src/core/runtimes/llamacpp/parse'
import { hashFile } from './measurement-launcher'
import { trackOwnedProcess, type OwnedProcess } from './owned-process'

const GiB = 1024 ** 3
const MAX_MS = 30_000
export interface ProbeFingerprint { path: string; bytes: number; sha256: string }
export interface HipProbeResult {
  kind: 'local-ai-optimizer/hip-device-probe-v1'
  exe: ProbeFingerprint
  dlls: ProbeFingerprint[]
  argv: ['--list-devices']
  startedAt: string
  finishedAt: string
  elapsedMs: number
  pid: number | null
  exitCode: number | null
  exitSignal: string | null
  stdout: string
  stderr: string
  devices: { id: string; name: string; backend: 'hip' }[]
  rocm0: { id: string; name: string; backend: 'hip' } | null
  wrapperSha256: string
  runtimeIntegrity: { verified: boolean; error: string | null }
  verifiedTeardownAt: string | null
  teardown: { at: string; verified: boolean; survivors: string[] }
  ramMinGiB: number
  abortReason: string | null
  error: string | null
}
export interface HipProbeDeps {
  spawnFn?: typeof spawn
  processTree?: ProcessTree
  readRam?: () => number
  hash?: typeof hashFile
  signal?: AbortSignal
  timeoutMs?: number // fake tests may shorten; production CLI always uses 30000.
  watchIntervalMs?: number
  env?: NodeJS.ProcessEnv
}
const message = (e: unknown) => e instanceof Error ? e.message : String(e)
async function fingerprints(exe: string, hash: typeof hashFile, signal: AbortSignal) {
  const file = async (path: string): Promise<ProbeFingerprint> => ({ path: resolve(path), bytes: statSync(path).size, sha256: await hash(path, signal) })
  const dllNames = readdirSync(dirname(exe)).filter((n) => n.toLowerCase().endsWith('.dll')).sort()
  if (!dllNames.length) throw new Error('HIP runtime has no DLLs to fingerprint')
  return { exe: await file(exe), dlls: await Promise.all(dllNames.map((name) => file(resolve(dirname(exe), name)))) }
}
export async function runHipProbe(exePath: string, outPath: string, deps: HipProbeDeps = {}): Promise<HipProbeResult> {
  const exe = resolve(exePath), out = resolve(outPath)
  if (existsSync(out)) throw new Error(`refusing to overwrite HIP probe artifact: ${out}`)
  const env = deps.env ?? process.env
  const unsafe = Object.keys(env).filter((k) => k.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')
  if (unsafe.length) throw new Error(`unified-memory environment key set: ${unsafe.join(', ')}`)
  if (!existsSync(exe)) throw new Error(`HIP executable missing: ${exe}`)
  const safeEnv = Object.fromEntries(Object.entries(env).filter(([k]) => k.toUpperCase() !== 'GGML_CUDA_ENABLE_UNIFIED_MEMORY'))
  const readRam = deps.readRam ?? freemem, hash = deps.hash ?? hashFile
  const controller = new AbortController()
  const parentAbort = () => controller.abort(deps.signal?.reason ?? new Error('HIP probe aborted'))
  if (deps.signal?.aborted) parentAbort()
  else deps.signal?.addEventListener('abort', parentAbort, { once: true })
  const startedAt = new Date(), start = Date.now()
  const totalMs = Math.min(deps.timeoutMs ?? MAX_MS, MAX_MS)
  const hardEnd = start + totalMs
  const childBudget = Math.max(1, totalMs - Math.min(10_000, Math.floor(totalMs / 3)))
  const within = async <T>(work: Promise<T>, label: string): Promise<T> => {
    const remaining = hardEnd - Date.now()
    if (remaining <= 0) throw new Error(`HIP probe total 30000 ms deadline before ${label}`)
    let timer: ReturnType<typeof setTimeout> | null = null
    try { return await Promise.race([work, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error(`HIP probe total 30000 ms deadline during ${label}`)), remaining) })]) }
    finally { if (timer) clearTimeout(timer) }
  }
  let minRam = readRam(), child: ChildProcess | null = null, owned: OwnedProcess | null = null
  let stdout = '', stderr = '', exitCode: number | null = null, exitSignal: string | null = null, failure: unknown = null
  let before: Awaited<ReturnType<typeof fingerprints>> | null = null
  let wrapperSha256 = '', teardownVerified = false, teardownAt = '', survivors: string[] = []
  let integrity: HipProbeResult['runtimeIntegrity'] = { verified: false, error: 'not checked' }
  const checkRam = () => {
    const available = readRam(); minRam = Math.min(minRam, available)
    if (available < 4 * GiB && !controller.signal.aborted) controller.abort(new Error(`HIP probe RAM ${Math.round(available / GiB * 100) / 100} GiB < 4 GiB`))
  }
  const watch = setInterval(checkRam, deps.watchIntervalMs ?? 500)
  const deadline = setTimeout(() => { if (!controller.signal.aborted) controller.abort(new Error('HIP probe exceeded 30000 ms total deadline')) }, totalMs)
  const childDeadline = setTimeout(() => { if (!controller.signal.aborted) controller.abort(new Error('HIP --list-devices child exceeded bounded deadline')) }, childBudget)
  let closed: Promise<void> | null = null
  try {
    checkRam(); controller.signal.throwIfAborted()
    wrapperSha256 = await within(hash(fileURLToPath(import.meta.url), controller.signal), 'wrapper hash')
    before = await within(fingerprints(exe, hash, controller.signal), 'preflight runtime hash')
    controller.signal.throwIfAborted()
    child = (deps.spawnFn ?? spawn)(exe, ['--list-devices'], { shell: false, windowsHide: true, env: safeEnv, stdio: ['ignore', 'pipe', 'pipe'] })
    closed = new Promise<void>((ok, fail) => {
      child!.once('close', (code, signal) => { exitCode = code; exitSignal = signal; ok() })
      child!.once('error', fail)
    })
    void closed.catch(() => {})
    child.stdout?.on('data', (chunk: Buffer) => { stdout += chunk.toString('utf8') })
    child.stderr?.on('data', (chunk: Buffer) => { stderr += chunk.toString('utf8') })
    owned = await within(trackOwnedProcess(child, deps.processTree), 'process ownership')
    const aborted = new Promise<never>((_, reject) => {
      if (controller.signal.aborted) return reject(controller.signal.reason)
      controller.signal.addEventListener('abort', () => reject(controller.signal.reason), { once: true })
    })
    await within(Promise.race([closed, aborted]), 'child close')
    clearTimeout(childDeadline)
    controller.signal.throwIfAborted()
    if (exitCode !== 0) throw new Error(`HIP --list-devices exited ${String(exitCode)} (${exitSignal ?? 'no signal'})`)
  } catch (e) { failure = e }
  finally {
    if (failure && !controller.signal.aborted) controller.abort(failure)
    if (owned) try { await within(owned.stop(), 'owned-process teardown'); teardownVerified = true }
    catch (e) { survivors.push(message(e)); failure = failure ? new AggregateError([failure, e], 'HIP probe and teardown failed') : e }
    else if (child) {
      child.kill()
      try { if (closed) await within(closed, 'unverified child close') }
      catch (e) { survivors.push(message(e)); failure = failure ? new AggregateError([failure, e], 'HIP child did not close') : e }
      failure ??= new Error('HIP child identity was never verified')
    }
    teardownAt = new Date().toISOString()
    if (before) try {
      const fresh = new AbortController()
      const remaining = Math.max(1, hardEnd - Date.now())
      const after = await within(fingerprints(exe, hash, AbortSignal.any([fresh.signal, AbortSignal.timeout(remaining)])), 'postflight runtime hash')
      if (JSON.stringify(after) !== JSON.stringify(before)) throw new Error('HIP executable or DLL set changed during probe')
      integrity = { verified: true, error: null }
    } catch (e) { integrity = { verified: false, error: message(e) }; failure = failure ? new AggregateError([failure, e], 'HIP runtime changed during probe') : e }
    try { checkRam() } catch (e) { failure = failure ? new AggregateError([failure, e], 'HIP RAM check failed') : e }
    if (controller.signal.aborted && !failure) failure = controller.signal.reason
    if (!controller.signal.aborted) controller.abort(new Error('HIP probe finished'))
    clearInterval(watch); clearTimeout(deadline); clearTimeout(childDeadline)
    deps.signal?.removeEventListener('abort', parentAbort)
  }
  const result: HipProbeResult = {
    kind: 'local-ai-optimizer/hip-device-probe-v1', exe: before?.exe ?? { path: exe, bytes: 0, sha256: '' }, dlls: before?.dlls ?? [], argv: ['--list-devices'],
    startedAt: startedAt.toISOString(), finishedAt: new Date().toISOString(), elapsedMs: Date.now() - start,
    pid: child?.pid ?? null, exitCode, exitSignal, stdout, stderr,
    devices: parseDevices(`${stdout}\n${stderr}`).map(({ id, name }) => ({ id, name, backend: 'hip' as const })),
    rocm0: parseDevices(`${stdout}\n${stderr}`).filter((d) => d.id === 'ROCm0').map(({ id, name }) => ({ id, name, backend: 'hip' as const }))[0] ?? null,
    wrapperSha256, runtimeIntegrity: integrity, verifiedTeardownAt: teardownVerified ? teardownAt : null,
    teardown: { at: teardownAt, verified: teardownVerified, survivors }, ramMinGiB: +(minRam / GiB).toFixed(2),
    abortReason: controller.signal.reason && message(controller.signal.reason) !== 'HIP probe finished' ? message(controller.signal.reason) : null,
    error: failure ? message(failure) : null
  }
  writeFileSync(out, JSON.stringify(result, null, 2), { flag: 'wx' })
  if (failure) throw failure
  return result
}

async function main(): Promise<void> {
  if (process.argv.length !== 4 || process.argv[2] !== '--out') throw new Error('usage: hip-probe.ts --out <new.json>')
  const result = await runHipProbe('vendor/llama.cpp-hip/llama-server.exe', process.argv[3])
  console.log(`HIP device probe wrote ${process.argv[3]}: exit ${result.exitCode}, ${result.elapsedMs} ms`)
}
if (process.argv[1] && /(?:^|[\\/])hip-probe\.ts$/i.test(process.argv[1])) void main().catch((e) => { console.error(e); process.exitCode = 1 })
