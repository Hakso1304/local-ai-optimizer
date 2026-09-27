/** Read-only, software-only release inventory. Usage:
 *  npx tsx scripts/release-packet.ts --out <new.json> --backend <runtime-dir> --model <gguf> [--model <gguf> ...]
 *  This records evidence for a human release decision; it does not launch a runtime or authorize a lease.
 */
import { execFile } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import { createReadStream, existsSync, lstatSync, readFileSync, readdirSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { freemem, tmpdir } from 'node:os'
import { dirname, extname, join, resolve } from 'node:path'
import { promisify } from 'node:util'
import { parseScan, SCAN_SCRIPT } from '../src/core/system/scanner'
import { readVramInUse } from '../src/core/telemetry/sampler'

const execFileAsync = promisify(execFile)
const GiB = 1024 ** 3
const root = resolve(import.meta.dirname, '..')
type Status = 'READY' | 'HOLD'
type Check = { name: string; status: Status; at: string; evidence?: unknown; reason?: string }
const checks: Check[] = []
const packet: Record<string, unknown> = { schema: 'release-packet-v1', startedAt: new Date().toISOString(), checkout: root,
  purpose: 'pre-launch inventory only; Fable releases a named invocation separately', checks }

function args(): { out: string; backends: string[]; models: string[]; baselineTests: string; previous: string | null } {
  const values = { out: '', backends: [] as string[], models: [] as string[], baselineTests: '', previous: null as string | null }
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i], value = process.argv[i + 1]
    if (!value || value.startsWith('--')) throw new Error(`${key} requires a value`)
    if (key === '--out') { if (values.out) throw new Error('duplicate --out'); values.out = value }
    else if (key === '--backend') values.backends.push(value)
    else if (key === '--model') values.models.push(value)
    else if (key === '--baseline-tests') { if (values.baselineTests) throw new Error('duplicate --baseline-tests'); values.baselineTests = value }
    else if (key === '--previous-packet' || key === '--previous-dump') {
      if (values.previous) throw new Error('choose one previous teardown record')
      values.previous = value
    }
    else throw new Error(`unknown option ${key}`)
  }
  if (!values.out) throw new Error('--out is required')
  return values
}

const run = async (file: string, argv: string[], timeout = 10_000, cwd = root) =>
  (await execFileAsync(file, argv, { cwd, windowsHide: true, shell: false, timeout, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 })).stdout
const ps = (script: string, timeout = 15_000) => run('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', Buffer.from(`$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.Encoding]::UTF8;${script}`, 'utf16le').toString('base64')], timeout)
const sha256 = async (path: string) => {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path, { signal: AbortSignal.timeout(300_000) })) hash.update(chunk as Buffer)
  return hash.digest('hex')
}
const fileEvidence = async (path: string) => {
  const absolute = resolve(path), realPath = realpathSync(absolute), stat = statSync(realPath)
  if (!stat.isFile()) throw new Error(`not a file: ${absolute}`)
  return { path: absolute, realPath, size: stat.size, sha256: await sha256(realPath) }
}
async function check(name: string, fn: () => Promise<{ ready: boolean; evidence: unknown; reason?: string }>) {
  const at = new Date().toISOString()
  try {
    const result = await fn()
    checks.push({ name, at, status: result.ready ? 'READY' : 'HOLD', evidence: result.evidence, ...(result.reason ? { reason: result.reason } : {}) })
  } catch (error) { checks.push({ name, at, status: 'HOLD', reason: error instanceof Error ? error.message : String(error) }) }
  const last = checks.at(-1)!
  console.log(`${last.status} ${name}${last.reason ? `: ${last.reason}` : ''}`)
}

async function source() {
  const head = (await run('git', ['rev-parse', 'HEAD'])).trim()
  const porcelain = await run('git', ['status', '--porcelain=v1', '--untracked-files=all', '--', 'src', 'scripts', 'package.json', 'package-lock.json'])
  const tracked = (await run('git', ['ls-files', '-z', '--', 'src', 'scripts', 'package.json', 'package-lock.json'])).split('\0').filter(Boolean).sort()
  if (!/^[0-9a-f]{40}$/.test(head) || !tracked.length) throw new Error('git HEAD or tracked source list unavailable')
  const files = [] as { path: string; size: number; sha256: string }[]
  for (const path of tracked) {
    const evidence = await fileEvidence(join(root, path))
    files.push({ path, size: evidence.size, sha256: evidence.sha256 })
  }
  const contentManifestSha256 = createHash('sha256').update(JSON.stringify(files)).digest('hex')
  return { ready: !porcelain.trim(), evidence: { head, porcelain, trackedFileCount: files.length, contentManifestSha256, files },
    reason: porcelain.trim() ? 'source checkout has changes in pinned paths' : undefined }
}

async function baselineTests(value: string) {
  if (!value) return { ready: false, evidence: null, reason: '--baseline-tests is required' }
  const files = value.trim().split(/\s+/)
  if (!files.length || files.some((file) => !/^tests[\\/][\w./\\-]+\.test\.tsx?$/.test(file) || file.includes('..') || !existsSync(join(root, file)))) {
    return { ready: false, evidence: { requested: value }, reason: 'baseline tests must name existing scoped test files' }
  }
  const reportPath = join(tmpdir(), `release-packet-vitest-${randomUUID()}.json`)
  const argv = [join(root, 'node_modules', 'vitest', 'vitest.mjs'), 'run', ...files, '--maxWorkers=1', '--reporter=json', `--outputFile=${reportPath}`, '--silent']
  const startedAt = new Date().toISOString()
  const result = await new Promise<{ code: number; stdout: string; stderr: string }>((done) => {
    execFile(process.execPath, argv, { cwd: root, windowsHide: true, shell: false, timeout: 180_000, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 },
      (error, stdout, stderr) => done({ code: error ? (typeof error.code === 'number' ? error.code : 1) : 0,
        stdout: String(stdout), stderr: String(stderr) }))
  })
  const endedAt = new Date().toISOString()
  let counts: { passed: number; failed: number; skipped: number; total: number } | null = null
  try {
    const report = JSON.parse(readFileSync(reportPath, 'utf8')) as {
      numPassedTests: number; numFailedTests: number; numPendingTests: number; numTotalTests: number
    }
    if ([report.numPassedTests, report.numFailedTests, report.numPendingTests, report.numTotalTests].every(Number.isSafeInteger))
      counts = { passed: report.numPassedTests, failed: report.numFailedTests, skipped: report.numPendingTests, total: report.numTotalTests }
  } catch { /* Reporter failure is a HOLD. */ }
  finally { rmSync(reportPath, { force: true }) }
  const ready = result.code === 0 && !!counts && counts.passed > 0 && counts.failed === 0 && counts.skipped === 0
  return { ready, evidence: { command: { executable: process.execPath, argv, cwd: root, shell: false, windowsHide: true, timeoutMs: 180_000 },
    startedAt, endedAt, exitCode: result.code, counts, reporter: 'vitest json (temporary reporter file removed)',
    stdoutSha256: createHash('sha256').update(result.stdout).digest('hex'),
    stderrSha256: createHash('sha256').update(result.stderr).digest('hex'), stderrTail: result.stderr.slice(-1000) },
  reason: ready ? undefined : 'baseline test command failed or reporter counts unavailable' }
}

async function idle(previous: string | null) {
  if (!previous) return { ready: false, evidence: null, reason: 'previous teardown record required to prove 60 s idle' }
  const file = await fileEvidence(previous)
  const document = JSON.parse(readFileSync(file.realPath, 'utf8')) as { teardown?: { at?: unknown; competingProcesses?: unknown; verified?: unknown }; verifiedTeardownAt?: unknown }
  const at = document.teardown?.at ?? document.verifiedTeardownAt
  const parsed = typeof at === 'string' ? Date.parse(at) : NaN
  const ageMs = Date.now() - parsed
  const verified = document.teardown?.competingProcesses === 0 || document.teardown?.verified === true
  const ready = Number.isFinite(parsed) && ageMs >= 60_000 && verified
  return { ready, evidence: { previous: file, teardownAt: at ?? null, teardownVerified: verified, ageMs: Number.isFinite(ageMs) ? ageMs : null,
    requiredIdleMs: 60_000 }, reason: ready ? undefined : 'verified teardown or 60 s idle unavailable' }
}

async function dependencies() {
  const node = await fileEvidence(process.execPath)
  const tsx = await fileEvidence(join(root, 'node_modules', 'tsx', 'dist', 'cli.mjs'))
  const packageLock = await fileEvidence(join(root, 'package-lock.json'))
  const installedLock = await fileEvidence(join(root, 'node_modules', '.package-lock.json'))
  const modules = join(root, 'node_modules')
  const junctionTarget = realpathSync(modules)
  const vendor = join(root, 'vendor')
  return { ready: true, evidence: { node: { ...node, version: process.version }, tsx, packageLock, installedLock,
    nodeModules: { path: modules, junctionTarget, isLink: lstatSync(modules).isSymbolicLink(),
      limitation: 'remaining node_modules contents pinned by lockfiles only, not individually authenticated' },
    vendor: { path: vendor, junctionTarget: realpathSync(vendor), isLink: lstatSync(vendor).isSymbolicLink() } } }
}

async function processes() {
  const script = "$rows=@(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.Name -ieq 'llama-server.exe' -or $_.Name -ieq 'typeperf.exe' -or (($_.Name -ieq 'node.exe' -or $_.Name -ieq 'tsx.exe') -and $_.CommandLine -match '(?i)(run-session|ab-spill|measurement-launcher)(\\.ts|\\.js|\\.mjs)') } | Select-Object @{n='pid';e={[int]$_.ProcessId}},@{n='name';e={$_.Name}},@{n='commandLine';e={$_.CommandLine}});@($rows) | ConvertTo-Json -Depth 3 -Compress"
  const raw = (await ps(script, 15_000)).trim()
  const parsed = JSON.parse(raw || '[]') as unknown
  const rows = Array.isArray(parsed) ? parsed : parsed ? [parsed] : []
  return { ready: rows.length === 0, evidence: { at: new Date().toISOString(), rows }, reason: rows.length ? `${rows.length} competing process(es)` : undefined }
}

async function environment() {
  const script = "$o=[ordered]@{};foreach($scope in @('Process','User','Machine')){$e=[Environment]::GetEnvironmentVariables([EnvironmentVariableTarget]::$scope);$o[$scope]=@($e.Keys | Where-Object { [string]$_ -ieq 'GGML_CUDA_ENABLE_UNIFIED_MEMORY' } | ForEach-Object { [string]$_ })};$o | ConvertTo-Json -Depth 3 -Compress"
  const scopes = JSON.parse(await ps(script, 10_000)) as Record<'Process' | 'User' | 'Machine', string[]>
  const found = Object.values(scopes).flat()
  return { ready: found.length === 0, evidence: { at: new Date().toISOString(), keysByScope: scopes }, reason: found.length ? 'unified-memory environment key present' : undefined }
}

async function machine() {
  const availableBytes = freemem()
  const raw = await ps(SCAN_SCRIPT, 30_000)
  const profile = parseScan(raw)
  const gpus = profile.gpus.value ?? []
  const discrete = gpus.filter((gpu) => !gpu.isIntegrated)
  const gpu = discrete.length === 1 ? discrete[0] : null
  // The one-shot reads the busiest adapter LUID. Mapping is inferred only when one discrete adapter exists.
  const vram = await readVramInUse(20_000)
  const baselineGiB = vram ? vram.bytes / GiB : null
  const reasons = [] as string[]
  if (availableBytes < 12 * GiB) reasons.push('available RAM below 12 GiB')
  if (!gpu || gpu.vendor !== 'amd' || !/RX\s*9070\s*XT/i.test(gpu.name)) reasons.push('expected single RX 9070 XT not verified')
  if (!vram || baselineGiB === null) reasons.push('dedicated VRAM reading unavailable')
  else if (baselineGiB < 1.0 || baselineGiB > 1.3) reasons.push('idle dedicated VRAM outside 1.0-1.3 GiB review band')
  return { ready: reasons.length === 0, evidence: { at: new Date().toISOString(), availableBytes, requiredRamBytes: 12 * GiB,
    adapters: gpus.map((g) => ({ name: g.name, vendor: g.vendor, pnpDeviceId: g.pnpDeviceId, driverVersion: g.driverVersion,
      dedicatedVramBytes: g.dedicatedVramBytes })), mapping: gpu && vram ? 'inferred-single-discrete' : 'unverified',
    selectedAdapter: gpu?.name ?? null, dedicatedVramInUse: vram, baselineGiB, expectedBaselineGiB: [1.1, 1.2], reviewBandGiB: [1.0, 1.3] },
  reason: reasons.join('; ') || undefined }
}

function dllPaths(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) out.push(...dllPaths(path))
    else if (entry.isFile() && extname(entry.name).toLowerCase() === '.dll') out.push(path)
  }
  return out.sort()
}

async function runtimes(backends: string[]) {
  if (!backends.length) return { ready: false, evidence: [], reason: 'at least one --backend directory is required' }
  const entries = [] as unknown[]
  const reasons: string[] = []
  for (const input of backends) {
    const dir = realpathSync(resolve(input))
    const exePath = join(dir, 'llama-server.exe')
    const tagPath = join(dir, 'release-tag.txt')
    if (!existsSync(exePath)) { reasons.push(`${input}: llama-server.exe missing`); continue }
    const executable = await fileEvidence(exePath)
    const dlls = [] as Awaited<ReturnType<typeof fileEvidence>>[]
    for (const path of dllPaths(dir)) dlls.push(await fileEvidence(path))
    const build = existsSync(tagPath) ? readFileSync(tagPath, 'utf8').trim() : null
    if (!build) reasons.push(`${input}: release-tag.txt build unavailable`)
    entries.push({ input, directory: dir, junctionTarget: dir, build, executable, dlls })
  }
  return { ready: !reasons.length, evidence: entries, reason: reasons.join('; ') || undefined }
}

async function models(paths: string[]) {
  if (!paths.length) return { ready: false, evidence: [], reason: 'at least one --model path is required' }
  const files = [] as unknown[]
  const reasons: string[] = []
  for (const input of paths) {
    try {
      const file = await fileEvidence(input)
      if (extname(file.realPath).toLowerCase() !== '.gguf') reasons.push(`${input}: expected GGUF extension`)
      files.push({ ...file, modelDirectory: { path: dirname(file.path), junctionTarget: realpathSync(dirname(file.path)),
        isLink: lstatSync(dirname(file.path)).isSymbolicLink() } })
    } catch (error) { reasons.push(`${input}: ${error instanceof Error ? error.message : String(error)}`) }
  }
  return { ready: !reasons.length, evidence: files, reason: reasons.join('; ') || undefined }
}

async function main() {
  const cli = args()
  const out = resolve(cli.out)
  if (existsSync(out)) throw new Error(`packet output already exists: ${out}`)
  packet.output = out
  packet.arguments = { backends: cli.backends.map((x) => resolve(x)), models: cli.models.map((x) => resolve(x)),
    baselineTests: cli.baselineTests, previous: cli.previous ? resolve(cli.previous) : null }
  await check('fix/test baseline', () => baselineTests(cli.baselineTests))
  await check('previous teardown idle', () => idle(cli.previous))
  await check('source checkout', source)
  await check('dependencies', dependencies)
  await check('process inventory', processes)
  await check('environment', environment)
  await check('RAM and adapter baseline', machine)
  await check('runtime executable and DLLs', () => runtimes(cli.backends))
  await check('model GGUF files', () => models(cli.models))
  await check('process inventory after fingerprints', processes)
  packet.completedAt = new Date().toISOString()
  packet.status = checks.every((c) => c.status === 'READY') ? 'READY' : 'HOLD'
  writeFileSync(out, JSON.stringify(packet, null, 2) + '\n', { flag: 'wx' })
  console.log(`${packet.status} packet ${out}`)
  if (packet.status === 'HOLD') process.exitCode = 1
}

void main().catch((error) => { console.error(`HOLD release packet: ${error instanceof Error ? error.message : String(error)}`); process.exitCode = 1 })
