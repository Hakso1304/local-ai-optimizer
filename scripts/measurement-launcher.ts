/** Immutable hardware gate. Create a manifest offline, then verify it on both sides of one launch. */
import { createHash } from 'node:crypto'
import { execFile, spawn, type ChildProcess } from 'node:child_process'
import { createReadStream, existsSync, lstatSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { freemem } from 'node:os'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { createRequire } from 'node:module'
import { DatabaseSync } from 'node:sqlite'
import * as ts from 'typescript'
import { createInterface } from 'node:readline'
import { promisify } from 'node:util'
import type { ProcessTree } from '../src/core/runtimes/llamacpp'
import { existingDbPath } from './harness-paths'
import { validateHarnessLimits } from './harness-limits'
import { trackOwnedProcess, type OwnedProcess } from './owned-process'
import { findGgufModels } from '../src/core/models/gguf'
import { MODELS_DIR, resolveSelectedModels } from './model-select'

const GiB = 1024 ** 3
const execFileAsync = promisify(execFile)
const unzipper = createRequire(import.meta.url)('unzipper') as { Open: { file(path: string): Promise<{ files: { path: string; type: string; buffer(): Promise<Buffer> }[] }> } }
const rolePaths = ['node', 'packageLock', 'installedLock', 'tsxCli', 'launcher', 'runner', 'exporter', 'runtimeExe', 'model', 'sourceZip'] as const
export type FileRole = typeof rolePaths[number] | `dll:${string}` | `model:${string}` | `source:${string}`
export interface Fingerprint { role: FileRole; path: string; bytes: number; sha256: string }
export interface MeasurementManifest {
  kind: 'local-ai-optimizer/measurement-gate-v1'
  createdAt: string
  snapshotHead: string
  command: { cwd: string; args: string[]; dbPath: string; exportOut: string }
  commandSha256?: string
  dependencyScope?: 'node_modules lockfile only; installed file contents not individually pinned'
  sourceHead?: string
  files: Fingerprint[]
}
export interface ManifestConfig {
  snapshotHead: string
  cwd: string
  args: string[]
  dbPath: string
  sourceZip: string
  tsxCli: string
  launcher: string
  runner: string
  exporter: string
  runtimeExe: string
  model: string
  models?: string[]
  exportOut: string
}
const commandHash = (command: MeasurementManifest['command']) => createHash('sha256').update(JSON.stringify(command)).digest('hex').toUpperCase()
const samePath = (a: string, b: string) => resolve(a).toLowerCase() === resolve(b).toLowerCase()
function sourcePaths(cwd: string): string[] {
  const out: string[] = []
  const walk = (relative: string) => {
    const absolute = join(cwd, relative)
    if (!existsSync(absolute)) return
    const stat = lstatSync(absolute)
    if (stat.isSymbolicLink()) throw new Error(`source tree contains a link: ${relative}`)
    if (stat.isFile()) { out.push(relative.replace(/\\/g, '/')); return }
    if (!stat.isDirectory()) throw new Error(`source tree contains unsupported entry: ${relative}`)
    for (const name of readdirSync(absolute).sort()) walk(join(relative, name))
  }
  for (const root of ['src', 'scripts', 'package.json']) walk(root)
  const configs = readdirSync(cwd).filter((name) => /^tsconfig(?:\.[^.]+)*\.json$/i.test(name) || /^tsx\.config\./i.test(name))
  const seenConfigs = new Set<string>()
  const configChain = (configPath: string) => {
    const absolute = resolve(cwd, configPath)
    const within = relative(cwd, absolute)
    if (!within || within.startsWith('..') || isAbsolute(within) || !existsSync(absolute)) throw new Error(`tsx configuration extends outside snapshot: ${configPath}`)
    if (seenConfigs.has(absolute)) return
    seenConfigs.add(absolute)
    walk(within)
    if (!within.toLowerCase().endsWith('.json')) return
    const parsed = ts.readConfigFile(absolute, ts.sys.readFile)
    if (parsed.error) throw new Error(`tsx configuration cannot be parsed: ${configPath}`)
    for (const ext of ([] as string[]).concat(parsed.config.extends ?? [])) {
      if (typeof ext !== 'string' || !ext.startsWith('.')) throw new Error(`tsx configuration extends unpinned package: ${String(ext)}`)
      const candidate = resolve(dirname(absolute), ext)
      configChain(candidate.toLowerCase().endsWith('.json') ? candidate : `${candidate}.json`)
    }
  }
  for (const config of configs) configChain(config)
  return out.sort()
}
export async function sourceStatus(cwd: string, head: string): Promise<string[]> {
  const files = sourcePaths(cwd)
  if (!existsSync(join(cwd, '.git'))) return files // git archive has no .git; the extracted tree is checked against its pinned ZIP below.
  const run = async (args: string[]) => (await execFileAsync('git', args, { cwd, encoding: 'utf8', windowsHide: true, timeout: 5_000 })).stdout.trim()
  const actualHead = await run(['rev-parse', 'HEAD'])
  if (actualHead !== head) throw new Error(`source git HEAD ${actualHead} differs from manifest ${head}`)
  const dirty = await run(['status', '--porcelain', '--untracked-files=all', '--', 'src', 'scripts', 'package.json'])
  if (dirty) throw new Error(`source git status is not clean: ${dirty}`)
  const tracked = (await execFileAsync('git', ['ls-files', '-z', '--', 'src', 'scripts', 'package.json'], { cwd, encoding: 'utf8', windowsHide: true, timeout: 5_000 })).stdout.split('\0').filter(Boolean).sort()
  if (JSON.stringify(tracked) !== JSON.stringify(files)) throw new Error('source tree differs from git-tracked file list')
  return tracked
}
export async function verifyArchiveSource(zipPath: string, cwd: string, names: string[]): Promise<void> {
  const archive = await unzipper.Open.file(zipPath)
  const entries = archive.files.filter((e) => e.type === 'File' && (e.path === 'package.json' || e.path.startsWith('src/') || e.path.startsWith('scripts/')))
  const archiveNames = entries.map((e) => e.path).sort()
  if (JSON.stringify(archiveNames) !== JSON.stringify(names)) throw new Error('source archive file list differs from extracted source tree')
  for (const entry of entries) {
    const archived = createHash('sha256').update(await entry.buffer()).digest('hex').toUpperCase()
    const extracted = await hashFile(join(cwd, entry.path))
    if (archived !== extracted) throw new Error(`source archive differs from extracted ${entry.path}`)
  }
}

export async function hashFile(path: string, signal?: AbortSignal): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path, { signal })) { signal?.throwIfAborted(); hash.update(chunk as Buffer) }
  return hash.digest('hex').toUpperCase()
}
async function fingerprint(role: FileRole, path: string, signal?: AbortSignal): Promise<Fingerprint> {
  const absolute = resolve(path)
  if (!existsSync(absolute)) throw new Error(`${role} missing: ${absolute}`)
  return { role, path: absolute, bytes: statSync(absolute).size, sha256: await hashFile(absolute, signal) }
}
const flag = (args: string[], name: string) => { const i = args.indexOf(name); return i < 0 ? undefined : args[i + 1] }
function checkCommand(manifest: MeasurementManifest): void {
  const { cwd, args, dbPath, exportOut } = manifest.command
  if (!isAbsolute(cwd) || !isAbsolute(dbPath) || !isAbsolute(exportOut) || !existsSync(cwd) || !existsSync(dirname(exportOut))) throw new Error('manifest cwd, DB and export path must be absolute with existing directories')
  if (existingDbPath(dbPath, true) !== resolve(dbPath) || flag(args, '--db') !== dbPath) throw new Error('launcher --db argument differs from existing manifest DB')
  validateHarnessLimits({ requestCapMs: Number(flag(args, '--request-cap-ms')), ramAbortGib: Number(flag(args, '--ram-abort-gib')) })
  if (manifest.commandSha256 && commandHash(manifest.command) !== manifest.commandSha256) throw new Error('manifest command changed after creation')
  if (Object.keys(process.env).some((k) => k.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')) throw new Error('unified-memory environment variable is set')
}
async function checkSelection(manifest: MeasurementManifest): Promise<void> {
  const { args, cwd } = manifest.command
  if (args[0] !== 'H') return // fake fixtures and future non-session gates have no model selector.
  const count = (flagName: string) => args.filter((a) => a === flagName).length
  if (count('--backend') !== 1 || count('--models') > 1 || count('--resume') > 1) throw new Error('backend/model selector is absent or ambiguous')
  const backend = flag(args, '--backend')
  if (backend !== 'vulkan' && backend !== 'hip') throw new Error('manifest backend selection invalid')
  const runtime = manifest.files.find((f) => f.role === 'runtimeExe')
  const expectedRuntime = join(cwd, 'vendor', backend === 'hip' ? 'llama.cpp-hip' : 'llama.cpp', 'llama-server.exe')
  if (!runtime || !samePath(runtime.path, expectedRuntime)) throw new Error(`manifest backend executable differs from ${backend} selection`)
  const requested = flag(args, '--models')?.split(',').filter(Boolean) ?? []
  const models = manifest.files.filter((f) => f.role === 'model' || f.role.startsWith('model:'))
  const resume = flag(args, '--resume')
  if (!requested.length && !resume) throw new Error('manifest model selection missing')
  let selectedNames = requested
  if (resume) {
    if (!/^\d+$/.test(resume)) throw new Error('manifest resume session ID invalid')
    const db = new DatabaseSync(manifest.command.dbPath, { readOnly: true })
    try {
      const row = db.prepare('SELECT payload FROM benchmark_session WHERE id = ?').get(Number(resume)) as { payload: string } | undefined
      const payload = row ? JSON.parse(row.payload) as { request?: { modelIds?: unknown } } : null
      if (!Array.isArray(payload?.request?.modelIds) || !payload.request.modelIds.every((p) => typeof p === 'string')) throw new Error(`manifest resume session ${resume} has no model paths`)
      selectedNames = payload.request.modelIds as string[]
    } finally { db.close() }
  }
  const infos = await findGgufModels([MODELS_DIR])
  const selectedPaths = resolveSelectedModels(infos, selectedNames, !!resume).map((info) => info.path)
  if (selectedPaths.length !== models.length) throw new Error('manifest model count differs from command')
  for (let i = 0; i < selectedPaths.length; i++) {
    if (!models[i] || !samePath(models[i].path, selectedPaths[i])) throw new Error(`manifest model path differs from selected ${selectedPaths[i]}`)
  }
}
function fileMap(manifest: MeasurementManifest): Map<FileRole, Fingerprint> {
  const map = new Map(manifest.files.map((f) => [f.role, f]))
  for (const role of rolePaths) if (!map.has(role)) throw new Error(`manifest lacks ${role}`)
  if (map.size !== manifest.files.length) throw new Error('manifest contains duplicate file roles')
  return map
}
export async function createManifest(config: ManifestConfig, out: string): Promise<MeasurementManifest> {
  if (!out) throw new Error('manifest output path required')
  if (!config.exportOut) throw new Error('manifest exportOut required')
  if (!isAbsolute(config.exportOut)) throw new Error('manifest exportOut must be absolute')
  if (existsSync(config.exportOut)) throw new Error(`refusing to overwrite session export: ${config.exportOut}`)
  const cwd = resolve(config.cwd), dbPath = existingDbPath(config.dbPath, true)!
  const args = [...config.args]
  const argDb = existingDbPath(flag(args, '--db'), true)
  if (argDb !== dbPath) throw new Error('manifest DB path differs from runner --db argument')
  args[args.indexOf('--db') + 1] = dbPath // shell:false never expands %APPDATA% itself
  const paths: Record<typeof rolePaths[number], string> = {
    node: process.execPath, packageLock: join(cwd, 'package-lock.json'), installedLock: join(cwd, 'node_modules', '.package-lock.json'),
    tsxCli: config.tsxCli, launcher: config.launcher, runner: config.runner, exporter: config.exporter, runtimeExe: config.runtimeExe, model: config.model, sourceZip: config.sourceZip
  }
  const files: Fingerprint[] = []
  for (const role of rolePaths) files.push(await fingerprint(role, paths[role]))
  for (const [i, model] of (config.models ?? [config.model]).entries()) {
    if (i === 0) { if (!samePath(model, config.model)) throw new Error('manifest first model differs from model role'); continue }
    files.push(await fingerprint(`model:${i}`, model))
  }
  const dllDir = dirname(resolve(config.runtimeExe))
  for (const name of readdirSync(dllDir).filter((n) => n.toLowerCase().endsWith('.dll')).sort()) files.push(await fingerprint(`dll:${name}`, join(dllDir, name)))
  if (!files.some((f) => f.role.startsWith('dll:'))) throw new Error('runtime has no DLLs to verify')
  const sourceNames = await sourceStatus(cwd, config.snapshotHead)
  if (args[0] === 'H' && !sourceNames.length) throw new Error('session snapshot has no tracked source files')
  for (const name of sourceNames) files.push(await fingerprint(`source:${name}`, join(cwd, name)))
  if (args[0] === 'H') await verifyArchiveSource(config.sourceZip, cwd, sourceNames)
  const command = { cwd, args, dbPath, exportOut: resolve(config.exportOut) }
  const manifest: MeasurementManifest = { kind: 'local-ai-optimizer/measurement-gate-v1', createdAt: new Date().toISOString(), snapshotHead: config.snapshotHead,
    command, commandSha256: commandHash(command), sourceHead: config.snapshotHead,
    dependencyScope: 'node_modules lockfile only; installed file contents not individually pinned', files }
  checkCommand(manifest)
  await checkSelection(manifest)
  const headFile = join(cwd, 'HEAD.txt')
  if (readFileSync(headFile, 'utf8').trim() !== config.snapshotHead) throw new Error('snapshot HEAD.txt differs from manifest')
  writeFileSync(out, JSON.stringify(manifest, null, 2), { flag: 'wx' })
  return manifest
}
export async function verifyManifest(manifest: MeasurementManifest, signal?: AbortSignal): Promise<void> {
  if (manifest.kind !== 'local-ai-optimizer/measurement-gate-v1' || !/^[0-9a-f]{40}$/i.test(manifest.snapshotHead)) throw new Error('invalid measurement manifest')
  checkCommand(manifest)
  await checkSelection(manifest)
  const map = fileMap(manifest)
  if (resolve(map.get('node')!.path).toLowerCase() !== resolve(process.execPath).toLowerCase()) throw new Error('Node executable differs from manifest')
  if (readFileSync(join(manifest.command.cwd, 'HEAD.txt'), 'utf8').trim() !== manifest.snapshotHead) throw new Error('snapshot HEAD changed')
  const dllDir = dirname(map.get('runtimeExe')!.path)
  const actualDlls = readdirSync(dllDir).filter((n) => n.toLowerCase().endsWith('.dll')).sort()
  const recordedDlls = manifest.files.filter((f) => f.role.startsWith('dll:')).map((f) => f.role.slice(4)).sort()
  if (JSON.stringify(actualDlls) !== JSON.stringify(recordedDlls)) throw new Error('runtime DLL set changed')
  const sourceNames = await sourceStatus(manifest.command.cwd, manifest.snapshotHead)
  const recordedSources = manifest.files.filter((f) => f.role.startsWith('source:')).map((f) => f.role.slice(7)).sort()
  if (JSON.stringify(sourceNames) !== JSON.stringify(recordedSources)) throw new Error('source tree file set changed')
  if (manifest.command.args[0] === 'H') {
    if (manifest.sourceHead !== manifest.snapshotHead || !manifest.commandSha256) throw new Error('source/command manifest identity missing')
    await verifyArchiveSource(map.get('sourceZip')!.path, manifest.command.cwd, sourceNames)
  }
  for (const original of manifest.files) {
    signal?.throwIfAborted()
    const current = await fingerprint(original.role, original.path, signal)
    if (current.bytes !== original.bytes || current.sha256 !== original.sha256) throw new Error(`${original.role} changed: ${original.path}`)
  }
}

async function countNamed(image: string): Promise<number> {
  const { stdout } = await execFileAsync('tasklist', ['/FI', `IMAGENAME eq ${image}`, '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true, timeout: 5_000 })
  return stdout.split(/\r?\n/).filter((l) => l.toLowerCase().startsWith(`"${image.toLowerCase()}"`)).length
}
export interface LaunchDeps {
  spawnFn?: typeof spawn
  tree?: ProcessTree
  countNamed?: typeof countNamed
  readRam?: () => number
  verify?: typeof verifyManifest
}
export async function launchManifest(manifest: MeasurementManifest, deps: LaunchDeps = {}): Promise<{ exitCode: number; sessionId: string; sessionStartLine: string; minRamGiB: number }> {
  const readRam = deps.readRam ?? freemem, count = deps.countNamed ?? countNamed, verify = deps.verify ?? verifyManifest
  const controller = new AbortController()
  const minRam = { bytes: readRam() }
  let child: ChildProcess | null = null
  let owned: OwnedProcess | null = null
  let closed: Promise<number> | null = null
  let exitCode = 1
  let sessionId: string | null = null, sessionStartLine: string | null = null
  let explicitMarkers = 0
  let launchError: unknown = null
  const checkRam = () => {
    const free = readRam(); minRam.bytes = Math.min(minRam.bytes, free)
    if (free < 4 * GiB && !controller.signal.aborted) { controller.abort(new Error(`host RAM ${Math.round(free / GiB * 100) / 100} GiB < 4 GiB`)); child?.kill() }
  }
  const timer = setInterval(checkRam, 500)
  const onSignal = (name: string) => { controller.abort(new Error(`launcher received ${name}`)); child?.kill() }
  const sigint = () => onSignal('SIGINT'), sigterm = () => onSignal('SIGTERM'), sigbreak = () => onSignal('SIGBREAK')
  process.once('SIGINT', sigint); process.once('SIGTERM', sigterm); process.once('SIGBREAK', sigbreak)
  try {
    checkRam(); controller.signal.throwIfAborted()
    checkCommand(manifest)
    controller.signal.throwIfAborted()
    if (readRam() < 12 * GiB) throw new Error('preflight RAM below 12 GiB')
    if (await count('llama-server.exe') || await count('typeperf.exe')) throw new Error('competing llama-server/typeperf process before launch')
    await verify(manifest, controller.signal) // last preflight action before process and model launch
    controller.signal.throwIfAborted()
    const map = fileMap(manifest), cwd = manifest.command.cwd
    child = (deps.spawnFn ?? spawn)(process.execPath, [map.get('tsxCli')!.path, map.get('runner')!.path, ...manifest.command.args], { cwd, shell: false, windowsHide: true,
      env: Object.fromEntries(Object.entries(process.env).filter(([k]) => k.toUpperCase() !== 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')), stdio: ['inherit', 'pipe', 'pipe'] })
    const forward = (stream: NodeJS.ReadableStream | null, dest: NodeJS.WriteStream, parse: boolean) => {
      if (!stream) return
      createInterface({ input: stream }).on('line', (line) => {
        dest.write(`${line}\n`)
        if (!parse) return
        const explicit = /^(?:\s*\d+(?:\.\d+)?s\s+)?SESSION_ID=(\d+)$/.exec(line)
        if (explicit) {
          explicitMarkers++
          if (explicitMarkers > 1 || (sessionId && sessionId !== explicit[1])) {
            controller.abort(new Error(`conflicting or repeated SESSION_ID marker: ${line}`))
            return
          }
          sessionId = explicit[1]; sessionStartLine = line; return
        }
        const at = line.indexOf('{')
        if (at < 0) return
        try {
          const event = JSON.parse(line.slice(at)) as { type?: string; sessionId?: string }
          if (event.type === 'session:started' && /^\d+$/.test(event.sessionId ?? '')) {
            if (sessionId && sessionId !== event.sessionId) throw new Error('multiple session IDs in runner output')
            sessionId = event.sessionId!; sessionStartLine ??= line
          }
        } catch (e) { if ((e as Error).message === 'multiple session IDs in runner output') controller.abort(e) }
      })
    }
    forward(child.stdout, process.stdout, true)
    forward(child.stderr, process.stderr, false)
    closed = new Promise<number>((resolve, reject) => { child!.once('close', (code) => resolve(code ?? 1)); child!.once('error', reject) })
    void closed.catch(() => {})
    owned = await trackOwnedProcess(child, deps.tree)
    const aborted = new Promise<never>((_, reject) => {
      if (controller.signal.aborted) return reject(controller.signal.reason)
      controller.signal.addEventListener('abort', () => reject(controller.signal.reason), { once: true })
    })
    exitCode = await Promise.race([closed, aborted])
    controller.signal.throwIfAborted()
    if (exitCode !== 0) throw new Error(`runner exited ${exitCode}`)
    if (!sessionId || !sessionStartLine) throw new Error('runner exited without a session:started ID in its own log')
  } catch (e) { launchError = e; throw e }
  finally {
    const errors: unknown[] = []
    if (owned) try { await owned.stop() } catch (e) { errors.push(e) }
    else if (child) {
      child.kill() // original ChildProcess handle, never a numeric PID with unverified identity
      if (closed) try { await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('unverified runner child did not close')), 10_000)
        void closed!.then(() => { clearTimeout(timer); resolve() }, (e) => { clearTimeout(timer); reject(e) })
      }) } catch (e) { errors.push(e) }
      errors.push(new Error('runner identity was never verified'))
    }
    // Every exit path waits for runner cleanup and checks named hardware children.
    try { if (await count('llama-server.exe') || await count('typeperf.exe')) errors.push(new Error('llama-server/typeperf survivor after launcher exit')) } catch (e) { errors.push(e) }
    try { await verify(manifest) } catch (e) { errors.push(e) } // detects mutable junctions after execution
    clearInterval(timer)
    process.off('SIGINT', sigint); process.off('SIGTERM', sigterm); process.off('SIGBREAK', sigbreak)
    if (errors.length) throw new AggregateError(launchError === null ? errors : [launchError, ...errors], 'measurement gate failed after execution')
  }
  return { exitCode, sessionId: sessionId!, sessionStartLine: sessionStartLine!, minRamGiB: +(minRam.bytes / GiB).toFixed(2) }
}

async function main(): Promise<void> {
  const mode = process.argv[2], file = process.argv[3]
  if (!file) throw new Error('usage: measurement-launcher.ts --create|--verify|--launch <manifest-or-config.json> [manifest-out.json]')
  if (mode === '--create') { const manifest = await createManifest(JSON.parse(readFileSync(file, 'utf8')) as ManifestConfig, process.argv[4]); console.log(`manifest ${process.argv[4]} HEAD ${manifest.snapshotHead} files ${manifest.files.length}`); return }
  const manifest = JSON.parse(readFileSync(file, 'utf8')) as MeasurementManifest
  if (mode === '--verify') { await verifyManifest(manifest); console.log(`manifest verified ${manifest.snapshotHead}`); return }
  if (mode !== '--launch') throw new Error(`unknown mode ${mode}`)
  const result = await launchManifest(manifest)
  if (existsSync(manifest.command.exportOut)) throw new Error(`refusing to overwrite session export: ${manifest.command.exportOut}`)
  const files = fileMap(manifest)
  const { stdout } = await execFileAsync(process.execPath, [files.get('tsxCli')!.path, files.get('exporter')!.path, '--session', result.sessionId,
    '--db', manifest.command.dbPath, '--head', manifest.snapshotHead, '--out', manifest.command.exportOut],
  { cwd: manifest.command.cwd, windowsHide: true, timeout: 120_000, env: Object.fromEntries(Object.entries(process.env).filter(([k]) => k.toUpperCase() !== 'GGML_CUDA_ENABLE_UNIFIED_MEMORY')) })
  process.stdout.write(stdout)
  console.log(`export SHA256 ${await hashFile(manifest.command.exportOut)}`)
  console.log(`measurement launcher done ${JSON.stringify(result)}`)
}
if (process.argv[1] && /(?:^|[\\/])measurement-launcher\.ts$/i.test(process.argv[1])) void main().catch((e) => { console.error(e); process.exitCode = 1 })
