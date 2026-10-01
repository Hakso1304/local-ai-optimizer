import { spawn, spawnSync, type ChildProcess, type SpawnOptions } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createWriteStream, existsSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync, mkdirSync } from 'node:fs'
import { findGgufModels } from '../../models/gguf'
import { createServer, type AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { join, dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as WebReadableStream } from 'node:stream/web'
import { runPowerShell, runProcess } from '../../exec'
import type { GpuVendor, RuntimeDetection } from '../../../shared/types'
import { expectedSha256, pickPrismAsset, pickReleaseAsset, pickRocmAsset, type ReleaseAsset } from './assets'
import { fetchOrExplain, getJson, type HealthStatus, type InferenceBackend, type LoadConfig, type LoadResult, type ModelInfo, type PromptRequest, type PromptResult, type RuntimeStats } from '../types'
import { acceptedSampling, classifyExit, emptyDeclared, parseDevices, parseLogLine, parseSse, toPromptResult, type CompletionChunk, type ExitReason, type LlamaDevice } from './parse'

const RELEASES_URL = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20'
// The fork publishes proper (non-prerelease) releases, so /latest is fine here.
const PRISM_RELEASE_URL = 'https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest'
export const VULKAN_ASSET = /^llama-.+-bin-win-vulkan-x64\.zip$/

interface GhRelease {
  tag_name: string
  draft: boolean
  assets: ReleaseAsset[]
}

/** Pick the newest release that actually ships a Windows Vulkan x64 zip.
 *  Note: /releases/latest is NOT usable — ggml-org marks every bNNNN build as prerelease and
 *  "latest" points at an asset-less release (observed 2026-09: v0.5.0 with only nightly-tag.txt). */
export function pickVulkanAsset(releases: GhRelease[]): { tag: string; name: string; url: string; size: number } | null {
  for (const r of releases) {
    if (r.draft) continue
    const a = r.assets.find((x) => VULKAN_ASSET.test(x.name))
    if (a) return { tag: r.tag_name, name: a.name, url: a.browser_download_url, size: a.size }
  }
  return null
}

/** Server came up with different settings than requested (maps to FailureKind 'config_drift'). */
/** Tracks whether streamed text is inside a thinking region. A chunk counts as reasoning when the region was open
 *  before it arrived (the tag chunks themselves are the boundary). Qwen templates may open <think> in the prompt. */
export function thinkCounter(prompt: string) {
  const OPEN = /<think>|<\|channel>thought/g, CLOSE = /<\/think>|<channel\|>/g
  const openedInPrompt = /<think>\s*$/.test(prompt)
  const c = { inside: openedInPrompt, seen: openedInPrompt, tokens: 0, tail: '',
    feed(s: string) {
      const t = c.tail + s
      // last tag in the chunk decides the state
      let last = -1, open = c.inside
      for (const m of t.matchAll(OPEN)) if (m.index! > last) { last = m.index!; open = true }
      for (const m of t.matchAll(CLOSE)) if (m.index! > last) { last = m.index!; open = false }
      if (last >= 0) c.seen = true
      c.inside = open
      c.tail = t.slice(-16) // tags can be split across chunks
    } }
  return c
}

/** Fixed date for chat templates (determinism across days). */
export const TEMPLATE_DATE = '01 Jan 2025'

/** The server could not be killed (kill + taskkill /F): stop the session, don't start another server. */
export class ServerStuckError extends Error {
  readonly fatal = true
}

export class ConfigDriftError extends Error {
  readonly failureKind = 'config_drift' as const
}

/** llama-server error body `{error:{type,message}}` → "type: message"; else the raw text, truncated. */
function serverError(body: string): string {
  try {
    const e = (JSON.parse(body) as { error?: { type?: string; message?: string } }).error
    if (e?.message) return `${e.type ?? 'error'}: ${e.message}`.slice(0, 300)
  } catch { /* not JSON */ }
  return body.slice(0, 300)
}

export interface ExitInfo {
  code: number | null
  reason: ExitReason
  tail: string[]
}

type SpawnFn = (cmd: string, args: string[], opts: SpawnOptions) => ChildProcess

export interface ProcessIdentity { pid: number; parentPid?: number; name: string; startedAt: string }
export interface ProcessTree {
  descendants(pid: number): Promise<ProcessIdentity[]>
  kill(pid: number, opts: { tree: boolean; force: boolean }): Promise<void>
  isAlive(pid: number): Promise<boolean>
  alivePids?(pids: number[]): Promise<number[]>
  inspect?(pid: number, timeoutMs?: number): Promise<ProcessIdentity | null>
  killVerified?(record: ProcessIdentity): Promise<boolean>
}

const defaultProcessTree: ProcessTree = {
  async descendants(pid) {
    const raw = await runPowerShell("$all = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,@{n='CreationDate';e={$_.CreationDate.ToUniversalTime().ToString('o')}}); $all | ConvertTo-Json -Compress", 10_000)
    const parsed = JSON.parse(raw || '[]') as { ProcessId: number; ParentProcessId: number; Name: string; CreationDate: string } | { ProcessId: number; ParentProcessId: number; Name: string; CreationDate: string }[]
    const all = Array.isArray(parsed) ? parsed : [parsed]
    const root = await defaultProcessTree.inspect!(pid)
    const seen = new Map<number, ProcessIdentity | null>([[pid, root]]), out: ProcessIdentity[] = []
    for (let i = 0; i < all.length; i++) {
      let added = false
      for (const p of all) if (!seen.has(p.ProcessId) && seen.has(p.ParentProcessId)) {
        const parent = seen.get(p.ParentProcessId)
        if (parent && (await defaultProcessTree.inspect!(parent.pid))?.startedAt !== parent.startedAt) continue
        const identity = await defaultProcessTree.inspect!(p.ProcessId)
        if (!identity || identity.parentPid !== p.ParentProcessId || identity.startedAt !== p.CreationDate) continue
        seen.set(p.ProcessId, identity)
        out.push(identity)
        added = true
      }
      if (!added) break
    }
    return out
  },
  async kill(pid, opts) {
    await runProcess('taskkill', ['/PID', String(pid), ...(opts.tree ? ['/T'] : []), ...(opts.force ? ['/F'] : [])], 10_000)
  },
  async isAlive(pid) {
    const raw = await runPowerShell(`if (Get-Process -Id ${pid} -ErrorAction SilentlyContinue) { 'true' } else { 'false' }`, 5_000)
    return raw.trim().toLowerCase() === 'true'
  },
  async alivePids(pids) {
    if (!pids.length) return []
    const ids = pids.filter((pid) => Number.isSafeInteger(pid) && pid > 0)
    const raw = await runPowerShell(`@(Get-Process -Id ${ids.join(',')} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id) | ConvertTo-Json -Compress`, 5_000)
    const parsed = JSON.parse(raw || '[]') as number | number[] | null
    return Array.isArray(parsed) ? parsed : typeof parsed === 'number' ? [parsed] : []
  },
  async inspect(pid, timeoutMs = 5_000) {
    if (!Number.isSafeInteger(pid) || pid <= 0) return null
    // Use CIM for both PID discovery and creation identity. Get-Process
    // StartTime has finer precision, and comparing its raw ticks to CIM's
    // truncated CreationDate rejects the same live process.
    const raw = await runPowerShell(`$w=Get-CimInstance Win32_Process -Filter 'ProcessId = ${pid}' -ErrorAction Stop;if($w){@{pid=[int]$w.ProcessId;parentPid=[int]$w.ParentProcessId;name=[string]$w.Name;startedAt=$w.CreationDate.ToUniversalTime().ToString('o')} | ConvertTo-Json -Compress}`, timeoutMs)
    if (!raw.trim()) return null
    const record = JSON.parse(raw) as ProcessIdentity
    return record.pid === pid && Number.isSafeInteger(record.parentPid) && typeof record.startedAt === 'string' ? record : null
  },
  async killVerified(record) {
    if (!Number.isSafeInteger(record.pid) || record.pid <= 0 || !/^\d{4}-\d\d-\d\dT[\d:.]+Z$/.test(record.startedAt)) throw new Error('invalid process identity')
    const script = `$p=Get-Process -Id ${record.pid} -ErrorAction SilentlyContinue;if(-not $p){'gone';exit};$want=[datetime]::Parse('${record.startedAt}').ToUniversalTime();if([math]::Abs($p.StartTime.ToUniversalTime().Ticks - $want.Ticks) -ge 10){'mismatch';exit};$p.Kill();$p.WaitForExit(3000) | Out-Null;'killed'`
    return (await runPowerShell(script, 6_000)).trim() === 'killed'
  }
}

/** Windows PID + creation-time inspection for supervised external process trees. */
export const windowsProcessTree: ProcessTree = defaultProcessTree

const sameProcess = (a: ProcessIdentity | null, b: ProcessIdentity) => !!a && a.pid === b.pid && a.startedAt === b.startedAt &&
  (a.parentPid === undefined || b.parentPid === undefined || a.parentPid === b.parentPid)

export interface OwnedScanOptions {
  tree: ProcessTree
  root: number
  rootIdentity: ProcessIdentity | null
  children: Map<number, ProcessIdentity>
  rootWasLive: boolean
  ownedExitAt: number | null
}

/** Reconcile a fresh scan against identities captured while the root lived.
 *  Call reap() only after checking the caller's deadline. Unknown ancestry fails closed. */
export async function reconcileOwnedProcessScan(options: OwnedScanOptions): Promise<{
  remaining: ProcessIdentity[]
  reap(): Promise<void>
}> {
  const { tree, root, rootIdentity, children, rootWasLive, ownedExitAt } = options
  const scan = () => tree.descendants(root)
  const stillOwned = async (record: ProcessIdentity) => {
    if (tree.inspect) return sameProcess(await tree.inspect(record.pid), record)
    if (!await tree.isAlive(record.pid)) return false
    // Legacy injected seam: production uses inspect + killVerified.
    return (await scan()).some((current) => sameProcess(current, record))
  }
  const ancestryValid = async (record: ProcessIdentity, seen = new Set<number>()): Promise<boolean> => {
    if (record.parentPid === undefined || record.parentPid === root) return true
    if (seen.has(record.pid)) return false
    seen.add(record.pid)
    const parent = children.get(record.parentPid)
    return !!parent && await stillOwned(parent) && await ancestryValid(parent, seen)
  }
  // Y1: after the root exits, a row whose parent is not the root is ours only through a captured parent that is
  // still the same live process (pid + creation identity), chained up to the root. Rows may chain through each other.
  const chainsToOwned = async (record: ProcessIdentity): Promise<boolean> => {
    const parent = record.parentPid === undefined ? undefined : children.get(record.parentPid)
    return !!tree.inspect && !!parent && sameProcess(await tree.inspect(parent.pid), parent) &&
      await ancestryValid(parent) && sameProcess(await tree.inspect(record.pid), record)
  }
  const pending: ProcessIdentity[] = []
  const found = await scan()
  for (const child of found) {
    const known = children.get(child.pid)
    if (known && sameProcess(child, known)) continue
    if (known && !sameProcess(child, known) && !rootWasLive) continue // recorded PID now belongs to a foreign process
    if (!known && !rootWasLive && child.parentPid !== undefined && child.parentPid !== root) { pending.push(child); continue }
    if (!known && rootWasLive && rootIdentity && tree.inspect &&
        sameProcess(await tree.inspect(root), rootIdentity) &&
        sameProcess(await tree.inspect(child.pid), child) &&
        (child.parentPid === root || (child.parentPid !== undefined &&
          children.has(child.parentPid) && sameProcess(await tree.inspect(child.parentPid), children.get(child.parentPid)!) &&
          await ancestryValid(children.get(child.parentPid)!)))) {
      children.set(child.pid, child)
      continue
    }
    const born = Date.parse(child.startedAt)
    // A child of the exited root PID born after that exit has a reused parent PID: provably foreign, never killed.
    if (!rootWasLive && Number.isFinite(born) && ownedExitAt !== null && born > ownedExitAt) continue
    throw new ServerStuckError(`llama-server pid ${root} has unverified descendant ${child.pid}`)
  }
  for (let progress = true; progress && pending.length;) {
    progress = false
    for (const record of [...pending]) if (await chainsToOwned(record)) {
      children.set(record.pid, record)
      pending.splice(pending.indexOf(record), 1)
      progress = true
    }
  }
  // No owned parent chain: not ours to kill, and not provably foreign either → fail closed, survivor reported.
  if (pending.length) throw new ServerStuckError(`llama-server pid ${root} has unverified descendant ${pending.map((x) => x.pid).join(', ')} (no owned parent identity after root exit)`)
  const remaining: ProcessIdentity[] = []
  for (const child of children.values()) if (await stillOwned(child)) remaining.push(child)
  // Deepest first: a child is killed while its parent still proves its ancestry.
  remaining.reverse()
  return { remaining, async reap() {
    for (const record of remaining) {
      if (!await ancestryValid(record)) throw new ServerStuckError(`descendant pid ${record.pid} lost its verified parent identity`)
      if (!await stillOwned(record)) continue
      if (tree.killVerified) await tree.killVerified(record)
      else if ((await scan()).some((current) => sameProcess(current, record))) await tree.kill(record.pid, { tree: true, force: true })
    }
  } }
}

/** The child's environment: the parent's minus GGML_CUDA_ENABLE_UNIFIED_MEMORY. Managed memory lets a CUDA/HIP
 *  build oversubscribe VRAM into system RAM, which would make "fits in VRAM" unmeasurable (docs/HIP-BACKEND.md). */
export function serverEnv(base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  // Windows environment names are case-insensitive. Delete every spelling so a
  // lowercase inherited key cannot re-enable managed-memory oversubscription.
  const env = { ...base }
  for (const key of Object.keys(env)) if (key.toUpperCase() === 'GGML_CUDA_ENABLE_UNIFIED_MEMORY') delete env[key]
  return env
}

/** Ask the OS for a free loopback port (tiny race until llama-server binds it; /props check catches a squatter).
 *  With `port`: that port if it is free, else rejects. */
export function freePort(port = 0): Promise<number> {
  return new Promise((ok, fail) => {
    const s = createServer()
    s.on('error', fail)
    s.listen(port, '127.0.0.1', () => {
      const { port } = s.address() as AddressInfo
      s.close(() => ok(port))
    })
  })
}

/** Kill a llama-server left behind by a previous app run (pid persisted in pidFile). Returns what it did. */
/** What the pid file records about the server we started (W4c D11): a reused pid must never be killed. */
export interface PidRecord { pid: number; exePath: string; startedAt: string }

/** Kill only if the live process is the one we recorded: same exe path and started within 30 s of the record.
 *  Anything unverifiable (old plain-number file, process gone, no path/start time) → don't kill. Pure, for tests. */
export function staleMatches(rec: PidRecord, proc: { path: string | null; startedAt: string | null } | null): boolean {
  if (!proc?.path || !proc.startedAt) return false
  const t = Date.parse(proc.startedAt) - Date.parse(rec.startedAt)
  return resolve(proc.path).toLowerCase() === resolve(rec.exePath).toLowerCase() && Number.isFinite(t) && Math.abs(t) <= 30_000
}

export function readPidRecord(pidFile: string): PidRecord | null {
  try {
    const j = JSON.parse(readFileSync(pidFile, 'utf8')) as Partial<PidRecord>
    return Number.isInteger(j.pid) && (j.pid as number) > 0 && typeof j.exePath === 'string' && typeof j.startedAt === 'string' ? (j as PidRecord) : null
  } catch { return null } // plain-number file from an older build: unverifiable
}

/** Kill a llama-server left behind by a previous app run (recorded in pidFile). Returns what it did. */
export async function killStaleServer(pidFile: string): Promise<string | null> {
  if (!existsSync(pidFile)) return null
  const rec = readPidRecord(pidFile)
  rmSync(pidFile, { force: true })
  if (!rec) return 'stale pid file could not be verified (no exe path / start time); nothing killed'
  const q = await runPowerShell(`$p = Get-Process -Id ${rec.pid} -ErrorAction SilentlyContinue; if ($p) { @{ path = $p.Path; startedAt = $p.StartTime.ToUniversalTime().ToString('o') } | ConvertTo-Json -Compress }`, 15_000).catch(() => '')
  let proc: { path: string | null; startedAt: string | null } | null = null
  try { proc = q.trim() ? (JSON.parse(q.trim()) as typeof proc) : null } catch { proc = null }
  if (!proc) return null // gone
  if (!staleMatches(rec, proc)) return `pid ${rec.pid} is not the recorded llama-server (reused pid); nothing killed`
  await runProcess('taskkill', ['/PID', String(rec.pid), '/T', '/F'], 10_000)
  return `killed stale llama-server pid ${rec.pid}`
}

const alive = (p: ChildProcess) => p.exitCode === null && p.signalCode === null

function waitExit(p: ChildProcess, ms: number): Promise<boolean> {
  return new Promise((done) => {
    if (!alive(p)) return done(true)
    const t = setTimeout(() => done(!alive(p)), ms)
    p.once('exit', () => { clearTimeout(t); done(true) })
  })
}

export class LlamaCppBackend implements InferenceBackend {
  readonly id = 'llamacpp' as const
  private proc: ChildProcess | null = null
  private port = 0
  private abort: AbortController | null = null
  private unloading: ChildProcess | null = null
  private unloadPromise: Promise<void> | null = null
  private loadDeclarationListener: ((declared: LoadResult['declared']) => void) | null = null
  private exeOverride?: string
  private pidFile?: string
  private spawnFn: SpawnFn
  private processTree: ProcessTree
  private ownedRootPid: number | null = null
  private ownedExitAt: number | null = null // exclusion bound only: a child born after exit cannot be owned
  private ownedRootIdentity: ProcessIdentity | null = null
  private ownedChildren = new Map<number, ProcessIdentity>()
  /** Last 100 output lines, for crash diagnostics. */
  readonly log: string[] = []
  /** Set when the server exits on its own (not via unloadModel). */
  lastExit: ExitInfo | null = null
  /** sha256 of the loaded model's chat_template (/props, read once per load); null = none reported / not loaded. */
  templateHash: string | null = null

  /** pidFile: where the running server's pid is persisted (see killStaleServer). spawnFn: test seam. */
  constructor(private vendorDir: string, opts: { exePath?: string; pidFile?: string; spawnFn?: SpawnFn; processTree?: ProcessTree } = {}) {
    this.exeOverride = opts.exePath
    this.pidFile = opts.pidFile
    this.spawnFn = opts.spawnFn ?? spawn
    this.processTree = opts.processTree ?? defaultProcessTree
  }

  get exePath(): string {
    return this.exeOverride ?? join(this.vendorDir, 'llama-server.exe')
  }

  setLoadDeclarationListener(listener: ((declared: LoadResult['declared']) => void) | null): void {
    this.loadDeclarationListener = listener
  }

  configure(opts: { exePath?: string; vendorDir?: string }): void {
    if (opts.exePath !== undefined) this.exeOverride = opts.exePath || undefined
    if (opts.vendorDir) this.vendorDir = opts.vendorDir
  }

  async detect(): Promise<RuntimeDetection> {
    const base = { id: this.id, path: this.exePath, source: 'llama-server --version' }
    if (!existsSync(this.exePath)) return { ...base, status: 'unavailable', error: 'llama-server.exe not found (run ensureRuntime or set a path)' }
    try {
      const { stdout, stderr } = await runProcess(this.exePath, ['--version'], 15_000)
      const out = `${stdout}\n${stderr}`
      const version = /version:\s*(.+)/.exec(out)?.[1]?.trim()
      const tagFile = join(dirname(this.exePath), 'release-tag.txt')
      const tag = existsSync(tagFile) ? readFileSync(tagFile, 'utf8').trim() : undefined
      return { ...base, status: 'available', version: [tag, version].filter(Boolean).join(' / ') || 'unknown' }
    } catch (e) {
      return { ...base, status: 'unavailable', error: (e as Error).message }
    }
  }

  /** Download + extract the newest official Windows build for this GPU into vendorDir if not installed.
   *  NVIDIA with a CUDA-capable driver → CUDA build + cudart (same dir); anything else, or any CUDA failure → Vulkan.
   *  Installed = release-tag.txt ("<tag> <build>") exists; it is written last, after an atomic rename. */
  async ensureRuntime(log: (m: string) => void = () => {}, pref: { vendor: GpuVendor; cudaMajor?: number; hip?: boolean; prism?: boolean; tag?: string } = { vendor: 'other' }): Promise<RuntimeDetection> {
    if (existsSync(join(this.vendorDir, 'release-tag.txt'))) return this.detect()
    // Opt-in PrismML fork (ternary models) into its own vendorDir. Marker "<tag> prism-vulkan".
    if (pref.prism) {
      const rel = await getJson<GhRelease>(PRISM_RELEASE_URL, 15_000)
      const asset = pickPrismAsset(rel.assets)
      if (!asset) throw new Error(`no win-vulkan-x64 build in PrismML release ${rel.tag_name}`)
      log(`PrismML ternary build ${asset.name}`)
      await this.installAssets(rel.tag_name, [asset], 'prism-vulkan', log)
      return this.detect()
    }
    // Opt-in ROCm/HIP build into this (separate) vendorDir, at the installed Vulkan build's tag when given so an A/B
    // compares backends of the same llama.cpp build. Marker "<tag> hip".
    if (pref.hip) {
      const rel = pref.tag
        ? await getJson<GhRelease>(`${RELEASES_URL.split('?')[0]}/tags/${encodeURIComponent(pref.tag)}`, 15_000)
        : (await getJson<GhRelease[]>(RELEASES_URL, 15_000)).find((r) => !r.draft && pickRocmAsset(r.assets))
      const asset = rel ? pickRocmAsset(rel.assets) : null
      if (!rel || !asset) throw new Error(`no win-rocm-x64 build in ${pref.tag ? `release ${pref.tag}` : 'the latest releases'}`)
      log(`AMD GPU: opt-in ROCm/HIP build ${asset.name}`)
      await this.installAssets(rel.tag_name, [asset], 'hip', log)
      return this.detect()
    }
    const releases = await getJson<GhRelease[]>(RELEASES_URL, 15_000)
    const rel = releases.find((r) => !r.draft && pickReleaseAsset(r.assets, pref).main)
    if (!rel) throw new Error(`no release among the latest ${releases.length} has a usable Windows build`)
    const pick = pickReleaseAsset(rel.assets, pref)
    log(pick.reason)
    const cuda = /-cuda-([\d.]+)-x64\.zip$/.exec(pick.main!.name)?.[1]
    try {
      await this.installAssets(rel.tag_name, [pick.main!, ...(pick.extra ? [pick.extra] : [])], cuda ? `cuda-${cuda}` : 'vulkan', log)
    } catch (e) {
      if (!cuda || !pick.fallback) throw e
      log(`CUDA build failed (${(e as Error).message}); installing the Vulkan build instead`)
      await this.installAssets(rel.tag_name, [pick.fallback], 'vulkan', log)
    }
    return this.detect()
  }

  /** Download zips, extract all into vendorDir.tmp, check llama-server runs, then swap it into place. */
  private async installAssets(tag: string, assets: ReleaseAsset[], build: string, log: (m: string) => void): Promise<void> {
    const tmp = `${this.vendorDir}.tmp`
    const zips: string[] = []
    try {
      rmSync(tmp, { recursive: true, force: true })
      mkdirSync(tmp, { recursive: true })
      for (const asset of assets) {
        const zip = join(tmpdir(), asset.name)
        zips.push(zip)
        log(`downloading ${asset.name} (${(asset.size / 1e6).toFixed(1)} MB)`)
        const res = await fetchOrExplain(asset.browser_download_url, { signal: AbortSignal.timeout(15 * 60_000) })
        if (!res.ok || !res.body) throw new Error(`download ${asset.browser_download_url} -> HTTP ${res.status}`)
        let received = 0
        let shown = -1
        const sha = createHash('sha256')
        const body = Readable.fromWeb(res.body as WebReadableStream).on('data', (c: Buffer) => {
          received += c.length
          sha.update(c)
          const pct = Math.floor((received * 100) / asset.size / 5) * 5
          if (pct !== shown) { shown = pct; log(`downloading ${asset.name} ${pct}%`) }
        })
        await pipeline(body, createWriteStream(zip))
        const got = statSync(zip).size
        if (got !== asset.size) throw new Error(`download truncated: ${got} of ${asset.size} bytes`)
        const want = expectedSha256(asset), have = sha.digest('hex')
        if (want && want !== have) throw new Error(`sha256 mismatch for ${asset.name}: got ${have}, expected ${want}`)
        log(want ? `sha256 verified (${have.slice(0, 12)}…)` : `sha256 ${have} (no published digest to compare)`)
        log(`extracting ${asset.name}`)
        const q = (s: string) => `'${s.replace(/'/g, "''")}'`
        await runPowerShell(`Expand-Archive -LiteralPath ${q(zip)} -DestinationPath ${q(tmp)} -Force`, 5 * 60_000)
      }
      // ponytail: assumes flat zip layout (true for current releases); hoist from a subfolder if that changes.
      const exe = join(tmp, 'llama-server.exe')
      if (!existsSync(exe)) throw new Error(`llama-server.exe missing after extract; got: ${readdirSync(tmp).slice(0, 10).join(', ')}`)
      await runProcess(exe, ['--version'], 30_000) // a CUDA build without its runtime DLLs fails here
      rmSync(this.vendorDir, { recursive: true, force: true }) // no marker = partial/old install
      renameSync(tmp, this.vendorDir)
      writeFileSync(join(this.vendorDir, 'release-tag.txt'), `${tag} ${build}`)
      log(`installed llama.cpp ${tag} (${build})`)
    } finally {
      for (const z of zips) rmSync(z, { force: true })
      rmSync(tmp, { recursive: true, force: true })
    }
  }

  /** `llama-server --list-devices`: ground truth for which Vulkan device ids exist. */
  async listDevices(): Promise<LlamaDevice[]> {
    const { stdout, stderr } = await runProcess(this.exePath, ['--list-devices'], 30_000)
    return parseDevices(`${stdout}\n${stderr}`)
  }

  /** Start llama-server for one model on a free port; resolves once /health is OK and /props shows our model. */
  /** Base URL of the running server (its built-in web UI is at /); null when nothing is loaded. */
  get url(): string | null { return this.proc ? `http://127.0.0.1:${this.port}` : null }

  async loadModel(cfg: LoadConfig): Promise<LoadResult> {
    if (this.proc || this.ownedRootPid) await this.unloadModel()
    if (cfg.signal?.aborted) throw new Error('cancelled')
    this.port = cfg.port ?? (await freePort())
    if (cfg.signal?.aborted) throw new Error('cancelled')
    this.log.length = 0
    this.lastExit = null
    this.templateHash = null
    const args = ['-m', cfg.modelPath, '-c', String(cfg.contextSize), '-ngl', String(cfg.gpuLayers), '--host', '127.0.0.1', '--port', String(this.port)]
    // -fit off: otherwise llama-server silently changes ctx/ngl to fit. -lv 4: device/offload lines we parse.
    // --cache-ram 0: the host-RAM prompt cache (default 8 GiB) saves every previous slot state on a new task — the
    // quality phase's 60+ tasks grew a 27B hybrid's RAM by ≈ 8.6 GiB over its ladder steps (3 tasks per launch).
    args.push('-fit', 'off', '--parallel', '1', '--cache-ram', '0', '--device', cfg.device, '-lv', '4', '--metrics')
    if (cfg.threads) args.push('-t', String(cfg.threads))
    if (cfg.batchSize) args.push('-b', String(cfg.batchSize))
    args.push(...(cfg.extraArgs ?? []))
    const declared = emptyDeclared()
    const t0 = performance.now()
    const p = this.spawnFn(this.exePath, args, { windowsHide: true, env: serverEnv() })
    this.proc = p
    this.ownedRootPid = p.pid ?? null
    this.ownedExitAt = null
    this.ownedRootIdentity = null
    this.ownedChildren.clear()
    if (this.pidFile && p.pid) writeFileSync(this.pidFile, JSON.stringify({ pid: p.pid, exePath: resolve(this.exePath), startedAt: new Date().toISOString() } satisfies PidRecord))
    const onLine = (line: string) => {
      if (!line) return
      parseLogLine(line, declared)
      this.loadDeclarationListener?.(declared)
      this.log.push(line)
      if (this.log.length > 100) this.log.shift()
    }
    for (const s of [p.stdout, p.stderr]) if (s) createInterface({ input: s, crlfDelay: Infinity }).on('line', onLine)
    let exited: string | null = null
    // 'close' (not 'exit') so the last stderr lines are in the tail before we classify.
    p.on('close', (code) => {
      if (this.proc !== p || this.unloading === p) return // being unloaded: unloadModel finishes the cleanup
      this.proc = null
      this.ownedExitAt = Date.now()
      this.lastExit = { code, reason: classifyExit(this.log), tail: [...this.log] }
      exited = `llama-server exited with code ${code} (${this.lastExit.reason})`
      this.abort?.abort(new Error(exited))
    })
    p.on('error', (e) => {
      exited = `llama-server failed to start: ${e.message}`
      if (this.proc === p) this.proc = null
    })
    if (p.pid && this.processTree.inspect) {
      try {
        const root = await this.processTree.inspect(p.pid, 15_000)
        if (!root || !alive(p)) {
          await waitExit(p, 1_000)
          throw new Error('spawned server identity could not be verified')
        }
        this.ownedRootIdentity = root
      } catch (e) {
        await this.unloadModel()
        if (exited) throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`)
        throw new ServerStuckError(`cannot verify spawned server identity: ${(e as Error).message}`)
      }
    }

    const deadline = Date.now() + 120_000
    let healthy = false
    while (!healthy && Date.now() < deadline) {
      if (cfg.signal?.aborted) {
        await this.unloadModel()
        throw new Error('cancelled')
      }
      if (exited) { await this.unloadModel(); throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`) }
      healthy = (await this.healthCheck(cfg.signal)).ok
      if (!healthy) await new Promise<void>((resolve) => {
        const timer = setTimeout(done, 100)
        const onAbort = () => { clearTimeout(timer); done() }
        function done() { cfg.signal?.removeEventListener('abort', onAbort); resolve() }
        cfg.signal?.addEventListener('abort', onAbort, { once: true })
        if (cfg.signal?.aborted) onAbort()
      })
    }
    if (!healthy) {
      await this.unloadModel()
      throw new Error('llama-server did not become healthy within 120s')
    }
    const loadTimeMs = performance.now() - t0
    const props = await getJson<{ model_path?: string; chat_template?: string; default_generation_settings?: { n_ctx?: number } }>(`http://127.0.0.1:${this.port}/props`, 2_000, cfg.signal).catch(() => null)
    if (cfg.signal?.aborted) { await this.unloadModel(); throw new Error('cancelled') }
    if (exited) { await this.unloadModel(); throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`) }
    const same = (a: string) => resolve(a).toLowerCase() === resolve(cfg.modelPath).toLowerCase()
    if (!props?.model_path || !same(props.model_path)) {
      await this.unloadModel()
      throw new Error(`wrong server answered on port ${this.port}: /props model_path=${props?.model_path ?? 'n/a'}`)
    }
    // #3's 14B run: -c 49152 on a 32K-trained model allocated a 48K KV but served n_ctx 32768 → every request HTTP 400.
    const nCtx = props.default_generation_settings?.n_ctx
    if (typeof nCtx === 'number' && nCtx < cfg.contextSize) {
      await this.unloadModel()
      throw new ConfigDriftError(`config_drift: requested -c ${cfg.contextSize} but server serves n_ctx ${nCtx}`)
    }
    // I-8.0: identity of the template the server renders with (read once per load; null = /props had none).
    this.templateHash = typeof props.chat_template === 'string' ? createHash('sha256').update(props.chat_template).digest('hex') : null
    // Capture descendants while the owned parent is still live. Once it exits,
    // a matching numeric parent PID is no longer evidence of ancestry.
    if (p.pid && alive(p)) {
      if (this.processTree.inspect && (!this.ownedRootIdentity ||
          (await this.processTree.inspect(p.pid, 15_000))?.startedAt !== this.ownedRootIdentity.startedAt)) {
        throw new ServerStuckError('root identity changed before child snapshot')
      }
      try {
        const children = await this.processTree.descendants(p.pid)
        for (const child of children) if (Number.isSafeInteger(child.pid) && Number.isFinite(Date.parse(child.startedAt))) this.ownedChildren.set(child.pid, child)
      } catch { /* unloading will require a fresh verified enumeration */ }
    }
    return { loadTimeMs, declared, templateHash: this.templateHash }
  }

  /** Reap the owned server tree, including children left behind by an exited parent. */
  unloadModel(): Promise<void> {
    if (this.unloadPromise) return this.unloadPromise
    const pending = this.reapOwnedTree()
    this.unloadPromise = pending
    void pending.then(
      () => { if (this.unloadPromise === pending) this.unloadPromise = null },
      () => { if (this.unloadPromise === pending) this.unloadPromise = null }
    )
    return pending
  }

  private async reapOwnedTree(): Promise<void> {
    const p = this.proc
    const root = this.ownedRootPid ?? p?.pid
    if (root === undefined || root === null) {
      if (p && alive(p)) { p.kill(); if (!(await waitExit(p, 3_000))) throw new ServerStuckError('llama-server without a pid did not exit') }
      if (this.proc === p) this.proc = null
      this.clearPid()
      return
    }
    if (p) this.unloading = p
    const children = new Map(this.ownedChildren)
    const reconcile = (rootWasLive: boolean) => reconcileOwnedProcessScan({
      tree: this.processTree, root, rootIdentity: this.ownedRootIdentity,
      children, rootWasLive, ownedExitAt: this.ownedExitAt
    })
    try {
      const rootLive = !!p && alive(p)
      if (rootLive && this.processTree.inspect && (!this.ownedRootIdentity || !sameProcess(await this.processTree.inspect(root), this.ownedRootIdentity))) {
        throw new ServerStuckError(`llama-server pid ${root} identity changed before unload`)
      }
      const initial = await reconcile(rootLive) // enumeration failure is a fatal inability to verify cleanup
      for (const [pid, child] of children) this.ownedChildren.set(pid, child)
      // After the parent exits, only children captured while it was alive are
      // owned. A new child of the same numeric parent PID is foreign.
      await initial.reap()
      if (rootLive) {
        if (this.processTree.killVerified) await this.processTree.killVerified(this.ownedRootIdentity!)
        else await this.processTree.kill(root, { tree: true, force: true })
        if (p && alive(p) && !(await waitExit(p, 3_000))) throw new ServerStuckError(`llama-server pid ${root} still alive after tree kill`)
      }
      const deadline = Date.now() + 3_000
      while (true) {
        const verified = await reconcile(false) // every scan is evidence; unknown survivors cannot be discarded
        if (!verified.remaining.length) break
        if (Date.now() >= deadline) throw new ServerStuckError(`llama-server pid ${root} left descendants alive: ${verified.remaining.map((x) => x.pid).join(', ')}`)
        await verified.reap()
        await new Promise((r) => setTimeout(r, 100))
      }
    } catch (e) {
      throw e instanceof ServerStuckError ? e : new ServerStuckError(`llama-server pid ${root} cleanup could not be verified: ${(e as Error).message}`)
    } finally {
      this.unloading = null
    }
    if (this.proc === p) this.proc = null
    this.ownedRootPid = null
    this.ownedExitAt = null
    this.ownedRootIdentity = null
    this.ownedChildren.clear()
    this.clearPid()
  }

  /** PID of the running server (for the telemetry sampler). */
  get pid(): number | undefined {
    return this.proc?.pid
  }

  /** Synchronous best-effort kill for app quit / process exit handlers. */
  killSync(): void {
    const killIdentity = (record: ProcessIdentity) => {
      if (!Number.isSafeInteger(record.pid) || !/^\d{4}-\d\d-\d\dT[\d:.]+Z$/.test(record.startedAt)) return
      const script = `$p=Get-Process -Id ${record.pid} -ErrorAction SilentlyContinue;if($p -and $p.StartTime.ToUniversalTime().Ticks -eq [datetime]::Parse('${record.startedAt}').ToUniversalTime().Ticks){$p.Kill();$p.WaitForExit(3000) | Out-Null}`
      spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', script], { windowsHide: true, timeout: 5_000 })
    }
    for (const child of this.ownedChildren.values()) killIdentity(child)
    if (this.proc && alive(this.proc)) {
      if (this.ownedRootIdentity) killIdentity(this.ownedRootIdentity)
      else this.proc.kill() // ChildProcess handle, never a numeric PID after exit
    }
  }

  private clearPid(): void {
    if (this.pidFile) rmSync(this.pidFile, { force: true })
  }

  async healthCheck(signal?: AbortSignal): Promise<HealthStatus> {
    try {
      const j = await getJson<{ status?: string }>(`http://127.0.0.1:${this.port}/health`, 1_500, signal)
      return { ok: j.status === 'ok', detail: JSON.stringify(j) }
    } catch (e) {
      return { ok: false, detail: (e as Error).message }
    }
  }

  async enumerateModels(dirs: string[]): Promise<ModelInfo[]> {
    return findGgufModels(dirs)
  }

  /** Stream one /completion. Request-level failures (HTTP, timeout, cancel, crash) land in `error`, never thrown. */
  async runPrompt(req: PromptRequest, onToken?: (t: string) => void): Promise<PromptResult> {
    const early = (error: string) => toPromptResult(null, { ttftMs: null, totalMs: 0, text: '', timedOut: false, error })
    if (req.signal?.aborted) return early('request cancelled before generation')
    if (!this.proc) return early(this.lastExit ? `server exited (${this.lastExit.reason}, code ${this.lastExit.code})` : 'no model loaded')
    if (this.abort) return early('another prompt is already in flight') // --parallel 1; cancel() has a single slot
    const timeoutMs = req.timeoutMs ?? 120_000
    const ctl = new AbortController()
    const requestSignal = req.signal ? AbortSignal.any([ctl.signal, req.signal]) : ctl.signal
    this.abort = ctl
    const t0 = performance.now()
    let ttftMs: number | null = null
    let streamed = 0
    const think = thinkCounter(req.prompt)
    let text = ''
    let final: CompletionChunk | null = null
    let error: string | null = null
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; ctl.abort(new Error(`timed out after ${timeoutMs}ms`)) }, timeoutMs)
    try {
      const res = await fetch(`http://127.0.0.1:${this.port}/completion`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          prompt: req.prompt, n_predict: req.maxTokens, temperature: req.temperature ?? 0, seed: req.seed ?? 42,
          ...(req.topP !== undefined ? { top_p: req.topP } : {}),
          ...(req.topK !== undefined ? { top_k: req.topK } : {}),
          ...(req.minP !== undefined ? { min_p: req.minP } : {}),
          stream: true, cache_prompt: false
        }),
        signal: requestSignal
      })
      if (!res.ok || !res.body) throw new Error(`POST /completion -> HTTP ${res.status}: ${serverError(await res.text())}`)
      const dec = new TextDecoder()
      let buf = ''
      for await (const bytes of res.body) {
        const { events, rest } = parseSse(buf + dec.decode(bytes, { stream: true }))
        buf = rest
        for (const ev of events as CompletionChunk[]) {
          if (ev.error) throw new Error(typeof ev.error === 'string' ? ev.error : (ev.error.message ?? JSON.stringify(ev.error)))
          if (ev.content) {
            streamed++
            if (think.inside) think.tokens++
            think.feed(ev.content)
            ttftMs ??= performance.now() - t0
            text += ev.content
            onToken?.(ev.content)
          }
          if (ev.stop) final = ev
        }
      }
      if (!final) error = 'stream ended without a final (stop) chunk'
    } catch (e) {
      const reason: unknown = requestSignal.aborted ? requestSignal.reason : e
      error = reason instanceof Error ? reason.message : String(reason)
    } finally {
      clearTimeout(timer)
      if (this.abort === ctl) this.abort = null
    }
    const totalMs = performance.now() - t0
    const f = final as CompletionChunk | null
    const needPrompt = !error && f !== null && f.timings?.prompt_n == null && f.tokens_evaluated == null
    const promptTokens = needPrompt ? await this.tokenize(req.prompt, req.signal).catch(() => null) : null
    const runtimeDecode = (typeof f?.timings?.predicted_n === 'number' && Number.isFinite(f.timings.predicted_n)) || (typeof f?.tokens_predicted === 'number' && Number.isFinite(f.tokens_predicted))
    return { ...toPromptResult(final, { ttftMs, totalMs, text, timedOut, error, streamedTokens: streamed, promptTokens }), streamedTokens: streamed, decodeTokenSource: runtimeDecode ? 'runtime' as const : 'streamed' as const, reasoningTokens: think.seen ? think.tokens : null, reasoningTokenSource: 'streamed' as const, acceptedSampling: acceptedSampling(f) }
  }

  /** One short discarded request so the measured run doesn't pay first-dispatch costs. Pass the measured
   *  prompt: a 1-token warmup left prefill at 47 tok/s on the 10-token measured run (b11208, RX 9070 XT). */
  async warmup(prompt = 'Hello', timeoutMs = 120_000, signal?: AbortSignal): Promise<void> {
    const r = await this.runPrompt({ prompt, maxTokens: 8, timeoutMs, signal })
    if (r.error) throw new Error(`warmup failed: ${r.error}`)
  }

  /** Token count of `text` with the loaded model's tokenizer (POST /tokenize). */
  async tokenize(text: string, signal?: AbortSignal): Promise<number> {
    const res = await fetch(`http://127.0.0.1:${this.port}/tokenize`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ content: text }), signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000)
    })
    if (!res.ok) throw new Error(`POST /tokenize -> HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`)
    return ((await res.json()) as { tokens: unknown[] }).tokens.length
  }

  /** Apply the model's chat template (POST /apply-template) so chat-format prompts can go through runPrompt. */
  /** Chat template → raw prompt. date_string is always pinned: templates like Llama 3.1's inject today's date via
   *  strftime_now, which would make quality prompts differ by day. templateKwargs (e.g. enable_thinking: false) win. */
  async applyTemplate(messages: { role: string; content: string }[], opts: { templateKwargs?: Record<string, unknown>; signal?: AbortSignal } = {}): Promise<string> {
    const chat_template_kwargs = { date_string: TEMPLATE_DATE, ...opts.templateKwargs }
    const res = await fetch(`http://127.0.0.1:${this.port}/apply-template`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages, chat_template_kwargs }), signal: opts.signal ? AbortSignal.any([opts.signal, AbortSignal.timeout(10_000)]) : AbortSignal.timeout(10_000)
    })
    if (!res.ok) throw new Error(`POST /apply-template -> HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`)
    return ((await res.json()) as { prompt: string }).prompt
  }

  /** /metrics (Prometheus text -> name:value) and /slots, whichever answer. */
  async getRuntimeStats(): Promise<RuntimeStats> {
    const base = `http://127.0.0.1:${this.port}`
    const [metrics, slots] = await Promise.allSettled([
      fetch(`${base}/metrics`, { signal: AbortSignal.timeout(2_000) }).then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const out: Record<string, number> = {}
        for (const line of (await r.text()).split('\n')) {
          const m = /^([^#\s]\S*)\s+(\S+)/.exec(line)
          if (m) out[m[1]] = Number(m[2])
        }
        return out
      }),
      getJson<unknown>(`${base}/slots`, 2_000)
    ])
    const stats: RuntimeStats = {}
    if (metrics.status === 'fulfilled') stats.metrics = metrics.value
    if (slots.status === 'fulfilled') stats.slots = slots.value
    stats.status = Object.keys(stats).length ? 'available' : 'unavailable'
    return stats
  }

  async cancel(): Promise<void> {
    this.abort?.abort(new Error('cancelled'))
  }
}
