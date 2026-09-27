/** Immutable gate for the standalone --hip and --igpu A/B harnesses (no app session). */
import { createHash } from 'node:crypto'
import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { freemem } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { promisify } from 'node:util'
import type { ProcessTree } from '../src/core/runtimes/llamacpp'
import { abCaseSpecs, type AbCaseSpec } from './ab-spill'
import { hashFile, sourceStatus, verifyArchiveSource } from './measurement-launcher'
import { trackOwnedProcess, type OwnedProcess } from './owned-process'

const GiB = 1024 ** 3
const execFileAsync = promisify(execFile)
type Mode = 'hip' | 'igpu'
interface Fingerprint { role: string; path: string; bytes: number; sha256: string }
export interface AbGateManifest {
  kind: 'local-ai-optimizer/ab-gate-v1'
  head: string
  mode: Mode
  createdAt: string
  command: { cwd: string; args: string[]; out: string }
  commandSha256: string
  caseSpecs: AbCaseSpec[]
  caseSpecsSha256: string
  dependencyScope: 'node_modules lockfile only; installed file contents not individually pinned'
  files: Fingerprint[]
}
export interface AbGateConfig { head: string; mode: Mode; cwd: string; sourceZip: string; tsxCli: string; out: string }
const digest = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex').toUpperCase()
const samePath = (a: string, b: string) => resolve(a).toLowerCase() === resolve(b).toLowerCase()
const message = (e: unknown) => e instanceof Error ? e.message : String(e)
async function fingerprint(role: string, path: string, signal?: AbortSignal): Promise<Fingerprint> {
  const absolute = resolve(path)
  if (!existsSync(absolute)) throw new Error(`${role} missing: ${absolute}`)
  return { role, path: absolute, bytes: statSync(absolute).size, sha256: await hashFile(absolute, signal) }
}
function selectedModels(): string[] {
  const all = [...abCaseSpecs('hip'), ...abCaseSpecs('igpu')]
  return [...new Set(all.map((spec) => spec.argv[spec.argv.indexOf('-m') + 1]))].sort()
}
function runtimePath(cwd: string, backend: 'vulkan' | 'hip') {
  return join(cwd, 'vendor', backend === 'hip' ? 'llama.cpp-hip' : 'llama.cpp', 'llama-server.exe')
}
function fixedArgs(mode: Mode, out: string): string[] { return [mode === 'hip' ? '--hip' : '--igpu', '--request-cap-ms', '300000', '--ram-abort-gib', '4', out] }
function validateShape(manifest: AbGateManifest): void {
  if (manifest.kind !== 'local-ai-optimizer/ab-gate-v1' || !/^[0-9a-f]{40}$/i.test(manifest.head)) throw new Error('invalid A/B gate identity')
  if (manifest.mode !== 'hip' && manifest.mode !== 'igpu') throw new Error('invalid A/B mode')
  if (!isAbsolute(manifest.command.cwd) || !isAbsolute(manifest.command.out)) throw new Error('A/B command paths must be absolute')
  if (digest(manifest.command) !== manifest.commandSha256 || JSON.stringify(manifest.command.args) !== JSON.stringify(fixedArgs(manifest.mode, manifest.command.out))) throw new Error('A/B command changed after gate creation')
  if (digest(manifest.caseSpecs) !== manifest.caseSpecsSha256 || JSON.stringify(manifest.caseSpecs) !== JSON.stringify(abCaseSpecs(manifest.mode))) throw new Error('A/B case argv list changed')
  const roles = new Map(manifest.files.map((f) => [f.role, f]))
  if (roles.size !== manifest.files.length) throw new Error('duplicate A/B fingerprint roles')
  for (const backend of ['vulkan', 'hip'] as const) {
    if (!roles.get(`runtime:${backend}`) || !samePath(roles.get(`runtime:${backend}`)!.path, runtimePath(manifest.command.cwd, backend))) throw new Error(`${backend} runtime selection differs from manifest`)
  }
  const models = selectedModels()
  for (const [i, model] of models.entries()) if (!roles.get(`model:${i}`) || !samePath(roles.get(`model:${i}`)!.path, model)) throw new Error(`hardcoded GGUF ${i} differs from A/B manifest`)
  for (const role of ['node', 'packageLock', 'installedLock', 'tsxCli', 'abSpill', 'ownedProcess', 'abGate', 'sourceZip']) if (!roles.has(role)) throw new Error(`A/B manifest lacks ${role}`)
  for (const [role, expected] of [
    ['node', process.execPath], ['packageLock', join(manifest.command.cwd, 'package-lock.json')],
    ['installedLock', join(manifest.command.cwd, 'node_modules', '.package-lock.json')],
    ['tsxCli', join(manifest.command.cwd, 'node_modules', 'tsx', 'dist', 'cli.mjs')],
    ['abSpill', join(manifest.command.cwd, 'scripts', 'ab-spill.ts')],
    ['ownedProcess', join(manifest.command.cwd, 'scripts', 'owned-process.ts')],
    ['abGate', join(manifest.command.cwd, 'scripts', 'ab-gate.ts')]
  ]) if (!samePath(roles.get(role)!.path, expected)) throw new Error(`A/B ${role} path differs from executed dependency`)
}
export async function createAbGate(config: AbGateConfig, manifestPath: string): Promise<AbGateManifest> {
  if (config.mode !== 'hip' && config.mode !== 'igpu') throw new Error('invalid A/B mode')
  const cwd = resolve(config.cwd), out = resolve(config.out)
  if (!isAbsolute(config.out) || existsSync(out)) throw new Error(`A/B output must be a new absolute path: ${out}`)
  if (existsSync(manifestPath)) throw new Error(`refusing existing A/B manifest: ${manifestPath}`)
  if (readFileSync(join(cwd, 'HEAD.txt'), 'utf8').trim() !== config.head) throw new Error('snapshot HEAD differs from A/B gate')
  const sources = await sourceStatus(cwd, config.head)
  if (!sources.length) throw new Error('A/B snapshot has no tracked source files')
  await verifyArchiveSource(config.sourceZip, cwd, sources)
  const files = await Promise.all([
    fingerprint('node', process.execPath), fingerprint('packageLock', join(cwd, 'package-lock.json')),
    fingerprint('installedLock', join(cwd, 'node_modules', '.package-lock.json')),
    fingerprint('tsxCli', config.tsxCli), fingerprint('abSpill', join(cwd, 'scripts', 'ab-spill.ts')),
    fingerprint('ownedProcess', join(cwd, 'scripts', 'owned-process.ts')),
    fingerprint('abGate', join(cwd, 'scripts', 'ab-gate.ts')), fingerprint('sourceZip', config.sourceZip),
    fingerprint('runtime:vulkan', runtimePath(cwd, 'vulkan')), fingerprint('runtime:hip', runtimePath(cwd, 'hip'))
  ])
  for (const [i, model] of selectedModels().entries()) files.push(await fingerprint(`model:${i}`, model))
  for (const backend of ['vulkan', 'hip'] as const) {
    const dir = dirname(runtimePath(cwd, backend))
    const dlls = readdirSync(dir).filter((name) => name.toLowerCase().endsWith('.dll')).sort()
    if (!dlls.length) throw new Error(`${backend} runtime has no DLLs`)
    for (const dll of dlls) files.push(await fingerprint(`dll:${backend}:${dll}`, join(dir, dll)))
  }
  for (const name of sources) files.push(await fingerprint(`source:${name}`, join(cwd, name)))
  const command = { cwd, args: fixedArgs(config.mode, out), out }
  const caseSpecs = abCaseSpecs(config.mode)
  const manifest: AbGateManifest = { kind: 'local-ai-optimizer/ab-gate-v1', head: config.head, mode: config.mode, createdAt: new Date().toISOString(),
    command, commandSha256: digest(command), caseSpecs, caseSpecsSha256: digest(caseSpecs),
    dependencyScope: 'node_modules lockfile only; installed file contents not individually pinned', files }
  validateShape(manifest)
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), { flag: 'wx' })
  return manifest
}
export async function verifyAbGate(manifest: AbGateManifest, signal?: AbortSignal, allowOutput = false): Promise<void> {
  validateShape(manifest)
  if (readFileSync(join(manifest.command.cwd, 'HEAD.txt'), 'utf8').trim() !== manifest.head) throw new Error('A/B snapshot HEAD changed')
  if (!allowOutput && existsSync(manifest.command.out)) throw new Error(`A/B output already exists: ${manifest.command.out}`)
  const roles = new Map(manifest.files.map((f) => [f.role, f]))
  const names = await sourceStatus(manifest.command.cwd, manifest.head)
  const recorded = manifest.files.filter((f) => f.role.startsWith('source:')).map((f) => f.role.slice(7)).sort()
  if (JSON.stringify(names) !== JSON.stringify(recorded)) throw new Error('A/B source tree file set changed')
  await verifyArchiveSource(roles.get('sourceZip')!.path, manifest.command.cwd, names)
  for (const backend of ['vulkan', 'hip'] as const) {
    const namesNow = readdirSync(dirname(runtimePath(manifest.command.cwd, backend))).filter((n) => n.toLowerCase().endsWith('.dll')).sort()
    const namesPinned = manifest.files.filter((f) => f.role.startsWith(`dll:${backend}:`)).map((f) => f.role.slice(`dll:${backend}:`.length)).sort()
    if (JSON.stringify(namesNow) !== JSON.stringify(namesPinned)) throw new Error(`${backend} DLL set changed`)
  }
  for (const original of manifest.files) {
    signal?.throwIfAborted()
    const current = await fingerprint(original.role, original.path, signal)
    if (current.bytes !== original.bytes || current.sha256 !== original.sha256) throw new Error(`${original.role} changed: ${original.path}`)
  }
}
async function countNamed(name: string) {
  const { stdout } = await execFileAsync('tasklist', ['/FI', `IMAGENAME eq ${name}`, '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true, timeout: 5_000 })
  return stdout.split(/\r?\n/).filter((s) => s.toLowerCase().startsWith(`"${name.toLowerCase()}"`)).length
}
export interface AbGateDeps { spawnFn?: typeof spawn; tree?: ProcessTree; readRam?: () => number; countNamed?: typeof countNamed; verify?: typeof verifyAbGate }
export async function launchAbGate(manifest: AbGateManifest, deps: AbGateDeps = {}): Promise<{ exitCode: number; minRamGiB: number }> {
  const controller = new AbortController(), readRam = deps.readRam ?? freemem, count = deps.countNamed ?? countNamed, verify = deps.verify ?? verifyAbGate
  const min = { bytes: readRam() }
  let child: ChildProcess | null = null, owned: OwnedProcess | null = null, closed: Promise<number> | null = null, failure: unknown = null, exitCode = 1
  const watch = setInterval(() => { const free = readRam(); min.bytes = Math.min(min.bytes, free); if (free < 4 * GiB && !controller.signal.aborted) controller.abort(new Error('A/B gate RAM below 4 GiB')) }, 500)
  const onSignal = (name: string) => controller.abort(new Error(`A/B gate received ${name}`))
  const sigint = () => onSignal('SIGINT'), sigterm = () => onSignal('SIGTERM'), sigbreak = () => onSignal('SIGBREAK')
  process.once('SIGINT', sigint); process.once('SIGTERM', sigterm); process.once('SIGBREAK', sigbreak)
  try {
    validateShape(manifest)
    if (Object.keys(process.env).some((k) => k.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')) throw new Error('unified-memory environment key set')
    if (readRam() < 12 * GiB) throw new Error('A/B gate preflight RAM below 12 GiB')
    if (await count('llama-server.exe') || await count('typeperf.exe')) throw new Error('competing GPU measurement process')
    await verify(manifest, controller.signal)
    controller.signal.throwIfAborted()
    const roles = new Map(manifest.files.map((f) => [f.role, f]))
    child = (deps.spawnFn ?? spawn)(process.execPath, [roles.get('tsxCli')!.path, roles.get('abSpill')!.path, ...manifest.command.args],
      { cwd: manifest.command.cwd, shell: false, windowsHide: true, env: Object.fromEntries(Object.entries(process.env).filter(([k]) => k.toUpperCase() !== 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')), stdio: ['ignore', 'pipe', 'pipe'] })
    child.stdout?.pipe(process.stdout, { end: false }); child.stderr?.pipe(process.stderr, { end: false })
    closed = new Promise<number>((ok, fail) => { child!.once('close', (code) => ok(code ?? 1)); child!.once('error', fail) })
    void closed.catch(() => {})
    owned = await trackOwnedProcess(child, deps.tree)
    const aborted = new Promise<never>((_, reject) => {
      if (controller.signal.aborted) return reject(controller.signal.reason)
      controller.signal.addEventListener('abort', () => reject(controller.signal.reason), { once: true })
    })
    exitCode = await Promise.race([closed, aborted])
    controller.signal.throwIfAborted()
    if (exitCode !== 0) throw new Error(`A/B harness exited ${exitCode}`)
  } catch (e) { failure = e }
  finally {
    const errors: unknown[] = []
    if (owned) try { await owned.stop() } catch (e) { errors.push(e) }
    else if (child) {
      child.kill()
      if (closed) try { await Promise.race([closed, new Promise<never>((_, reject) => setTimeout(() => reject(new Error('unverified A/B runner did not close')), 10_000))]) } catch (e) { errors.push(e) }
      errors.push(new Error('A/B runner identity was never verified'))
    }
    try { if (await count('llama-server.exe') || await count('typeperf.exe')) errors.push(new Error('A/B server or sampler survived')) } catch (e) { errors.push(e) }
    try { await verify(manifest, undefined, true) } catch (e) { errors.push(e) }
    clearInterval(watch)
    process.off('SIGINT', sigint); process.off('SIGTERM', sigterm); process.off('SIGBREAK', sigbreak)
    if (controller.signal.aborted && !failure) errors.push(controller.signal.reason)
    if (errors.length) failure = new AggregateError(failure ? [failure, ...errors] : errors, 'A/B gate teardown or verification failed')
  }
  if (failure) throw failure
  if (!existsSync(manifest.command.out)) throw new Error('A/B harness exited without its exclusive artifact')
  return { exitCode, minRamGiB: +(min.bytes / GiB).toFixed(2) }
}
async function main() {
  const mode = process.argv[2], file = process.argv[3]
  if (!file) throw new Error('usage: ab-gate.ts --create|--verify|--launch <config-or-manifest.json> [new-manifest.json]')
  if (mode === '--create') { const row = await createAbGate(JSON.parse(readFileSync(file, 'utf8')) as AbGateConfig, process.argv[4]); console.log(`A/B manifest ${process.argv[4]} ${row.head} ${row.files.length} files`); return }
  const manifest = JSON.parse(readFileSync(file, 'utf8')) as AbGateManifest
  if (mode === '--verify') { await verifyAbGate(manifest); console.log(`A/B manifest verified ${manifest.head}`); return }
  if (mode !== '--launch') throw new Error(`unknown A/B gate mode ${mode}`)
  console.log(`A/B gate complete ${JSON.stringify(await launchAbGate(manifest))}`)
}
if (process.argv[1] && /(?:^|[\\/])ab-gate\.ts$/i.test(process.argv[1])) void main().catch((e) => { console.error(e); process.exitCode = 1 })
