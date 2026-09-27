import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
import { createWriteStream, existsSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync, mkdirSync } from 'node:fs'
import { createServer, type AddressInfo } from 'node:net'
import { tmpdir } from 'node:os'
import { basename, join, dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as WebReadableStream } from 'node:stream/web'
import { runPowerShell, runProcess } from '../../exec'
import type { RuntimeDetection } from '../../../shared/types'
import { getJson, type HealthStatus, type InferenceBackend, type LoadConfig, type LoadResult, type ModelInfo, type PromptRequest, type PromptResult, type RuntimeStats } from '../types'
import { classifyExit, emptyDeclared, parseDevices, parseLogLine, parseSse, toPromptResult, type CompletionChunk, type ExitReason, type LlamaDevice } from './parse'

const RELEASES_URL = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20'
export const VULKAN_ASSET = /^llama-.+-bin-win-vulkan-x64\.zip$/

interface GhRelease {
  tag_name: string
  draft: boolean
  assets: { name: string; browser_download_url: string; size: number }[]
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

/** Recursively list *.gguf files under each existing dir. Missing dirs are skipped. */
export function findGgufModels(dirs: string[]): ModelInfo[] {
  const out = new Map<string, ModelInfo>()
  for (const dir of dirs) {
    if (!existsSync(dir)) continue
    for (const rel of readdirSync(dir, { recursive: true, encoding: 'utf8' })) {
      // ponytail: split shards (-0000N-of-0000M) and mmproj files are listed as-is; group/filter once GGUF metadata lands.
      if (!rel.toLowerCase().endsWith('.gguf')) continue
      const path = join(dir, rel)
      const st = statSync(path)
      if (st.isFile()) out.set(path, { id: path, name: basename(path, '.gguf'), path, sizeBytes: st.size, runtime: 'llamacpp' })
    }
  }
  return [...out.values()]
}

/** TODO(next task): parse GGUF header (arch, params, quant, ctx_train). */
export function readGgufMetadata(_path: string): Record<string, unknown> | null {
  return null
}

export interface ExitInfo {
  code: number | null
  reason: ExitReason
  tail: string[]
}

type SpawnFn = (cmd: string, args: string[], opts: SpawnOptions) => ChildProcess

/** Ask the OS for a free loopback port (tiny race until llama-server binds it; /props check catches a squatter). */
export function freePort(): Promise<number> {
  return new Promise((ok, fail) => {
    const s = createServer()
    s.on('error', fail)
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address() as AddressInfo
      s.close(() => ok(port))
    })
  })
}

/** Kill a llama-server left behind by a previous app run (pid persisted in pidFile). Returns what it did. */
export async function killStaleServer(pidFile: string): Promise<string | null> {
  if (!existsSync(pidFile)) return null
  const pid = Number(readFileSync(pidFile, 'utf8').trim())
  rmSync(pidFile, { force: true })
  if (!Number.isInteger(pid) || pid <= 0) return null
  const { stdout } = await runProcess('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], 10_000)
  if (!/^"llama-server\.exe"/im.test(stdout)) return null // gone, or pid reused by something else
  await runProcess('taskkill', ['/PID', String(pid), '/T', '/F'], 10_000)
  return `killed stale llama-server pid ${pid}`
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
  private exeOverride?: string
  private pidFile?: string
  private spawnFn: SpawnFn
  /** Last 100 output lines, for crash diagnostics. */
  readonly log: string[] = []
  /** Set when the server exits on its own (not via unloadModel). */
  lastExit: ExitInfo | null = null

  /** pidFile: where the running server's pid is persisted (see killStaleServer). spawnFn: test seam. */
  constructor(private vendorDir: string, opts: { exePath?: string; pidFile?: string; spawnFn?: SpawnFn } = {}) {
    this.exeOverride = opts.exePath
    this.pidFile = opts.pidFile
    this.spawnFn = opts.spawnFn ?? spawn
  }

  get exePath(): string {
    return this.exeOverride ?? join(this.vendorDir, 'llama-server.exe')
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

  /** Download + extract the newest official Windows Vulkan build into vendorDir if not installed.
   *  Installed = release-tag.txt exists; it is written last, after an atomic rename of the extracted dir. */
  async ensureRuntime(log: (m: string) => void = () => {}): Promise<RuntimeDetection> {
    const marker = join(this.vendorDir, 'release-tag.txt')
    if (existsSync(marker)) return this.detect()
    const releases = await getJson<GhRelease[]>(RELEASES_URL, 15_000)
    const asset = pickVulkanAsset(releases)
    if (!asset) throw new Error(`no release among the latest ${releases.length} has a win-vulkan-x64 zip`)
    const zip = join(tmpdir(), asset.name)
    const tmp = `${this.vendorDir}.tmp`
    try {
      log(`downloading ${asset.name} (${(asset.size / 1e6).toFixed(1)} MB)`)
      const res = await fetch(asset.url, { signal: AbortSignal.timeout(15 * 60_000) })
      if (!res.ok || !res.body) throw new Error(`download ${asset.url} -> HTTP ${res.status}`)
      await pipeline(Readable.fromWeb(res.body as WebReadableStream), createWriteStream(zip))
      const got = statSync(zip).size
      if (got !== asset.size) throw new Error(`download truncated: ${got} of ${asset.size} bytes`)
      log(`extracting to ${this.vendorDir}`)
      rmSync(tmp, { recursive: true, force: true })
      mkdirSync(tmp, { recursive: true })
      const q = (s: string) => `'${s.replace(/'/g, "''")}'`
      await runPowerShell(`Expand-Archive -LiteralPath ${q(zip)} -DestinationPath ${q(tmp)} -Force`, 5 * 60_000)
      // ponytail: assumes flat zip layout (true for current releases); hoist from a subfolder if that changes.
      if (!existsSync(join(tmp, 'llama-server.exe'))) {
        throw new Error(`llama-server.exe missing after extract; got: ${readdirSync(tmp).slice(0, 10).join(', ')}`)
      }
      rmSync(this.vendorDir, { recursive: true, force: true }) // no marker = partial/old install
      renameSync(tmp, this.vendorDir)
      writeFileSync(marker, asset.tag)
    } finally {
      rmSync(zip, { force: true })
      rmSync(tmp, { recursive: true, force: true })
    }
    return this.detect()
  }

  /** `llama-server --list-devices`: ground truth for which Vulkan device ids exist. */
  async listDevices(): Promise<LlamaDevice[]> {
    const { stdout, stderr } = await runProcess(this.exePath, ['--list-devices'], 30_000)
    return parseDevices(`${stdout}\n${stderr}`)
  }

  /** Start llama-server for one model on a free port; resolves once /health is OK and /props shows our model. */
  async loadModel(cfg: LoadConfig): Promise<LoadResult> {
    if (this.proc) await this.unloadModel()
    this.port = cfg.port ?? (await freePort())
    this.log.length = 0
    this.lastExit = null
    const args = ['-m', cfg.modelPath, '-c', String(cfg.contextSize), '-ngl', String(cfg.gpuLayers), '--host', '127.0.0.1', '--port', String(this.port)]
    // -fit off: otherwise llama-server silently changes ctx/ngl to fit. -lv 4: device/offload lines we parse.
    args.push('-fit', 'off', '--parallel', '1', '--device', cfg.device, '-lv', '4', '--metrics')
    if (cfg.threads) args.push('-t', String(cfg.threads))
    if (cfg.batchSize) args.push('-b', String(cfg.batchSize))
    args.push(...(cfg.extraArgs ?? []))
    const declared = emptyDeclared()
    const t0 = performance.now()
    const p = this.spawnFn(this.exePath, args, { windowsHide: true })
    this.proc = p
    if (this.pidFile && p.pid) writeFileSync(this.pidFile, String(p.pid))
    const onLine = (line: string) => {
      if (!line) return
      parseLogLine(line, declared)
      this.log.push(line)
      if (this.log.length > 100) this.log.shift()
    }
    for (const s of [p.stdout, p.stderr]) if (s) createInterface({ input: s, crlfDelay: Infinity }).on('line', onLine)
    let exited: string | null = null
    // 'close' (not 'exit') so the last stderr lines are in the tail before we classify.
    p.on('close', (code) => {
      if (this.proc !== p) return // killed by unloadModel
      this.proc = null
      this.clearPid()
      this.lastExit = { code, reason: classifyExit(this.log), tail: [...this.log] }
      exited = `llama-server exited with code ${code} (${this.lastExit.reason})`
      this.abort?.abort(new Error(exited))
    })
    p.on('error', (e) => {
      exited = `llama-server failed to start: ${e.message}`
      if (this.proc === p) this.proc = null
    })

    const deadline = Date.now() + 120_000
    let healthy = false
    while (!healthy && Date.now() < deadline) {
      if (exited) throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`)
      healthy = (await this.healthCheck()).ok
      if (!healthy) await new Promise((r) => setTimeout(r, 100))
    }
    if (!healthy) {
      await this.unloadModel()
      throw new Error('llama-server did not become healthy within 120s')
    }
    const loadTimeMs = performance.now() - t0
    const props = await getJson<{ model_path?: string }>(`http://127.0.0.1:${this.port}/props`, 2_000).catch(() => null)
    if (exited) throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`)
    const same = (a: string) => resolve(a).toLowerCase() === resolve(cfg.modelPath).toLowerCase()
    if (!props?.model_path || !same(props.model_path)) {
      await this.unloadModel()
      throw new Error(`wrong server answered on port ${this.port}: /props model_path=${props?.model_path ?? 'n/a'}`)
    }
    return { loadTimeMs, declared }
  }

  /** Kill the server; escalate to taskkill /T /F after 5s. Throws if it is still alive afterwards. */
  async unloadModel(): Promise<void> {
    const p = this.proc
    this.proc = null
    if (!p || p.pid === undefined || !alive(p)) return this.clearPid()
    p.kill()
    if (!(await waitExit(p, 5_000))) {
      await runProcess('taskkill', ['/PID', String(p.pid), '/T', '/F'], 10_000).catch(() => {})
      if (!(await waitExit(p, 3_000))) throw new Error(`llama-server pid ${p.pid} still alive after taskkill /F`)
    }
    this.clearPid()
  }

  /** Synchronous best-effort kill for app quit / process exit handlers. */
  killSync(): void {
    if (this.proc?.pid && alive(this.proc)) this.proc.kill()
  }

  private clearPid(): void {
    if (this.pidFile) rmSync(this.pidFile, { force: true })
  }

  async healthCheck(): Promise<HealthStatus> {
    try {
      const j = await getJson<{ status?: string }>(`http://127.0.0.1:${this.port}/health`, 1_500)
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
    const timeoutMs = req.timeoutMs ?? 120_000
    const ctl = new AbortController()
    this.abort = ctl
    const t0 = performance.now()
    let ttftMs: number | null = null
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
          stream: true, cache_prompt: false
        }),
        signal: ctl.signal
      })
      if (!res.ok || !res.body) throw new Error(`POST /completion -> HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`)
      const dec = new TextDecoder()
      let buf = ''
      for await (const bytes of res.body) {
        const { events, rest } = parseSse(buf + dec.decode(bytes, { stream: true }))
        buf = rest
        for (const ev of events as CompletionChunk[]) {
          if (ev.error) throw new Error(typeof ev.error === 'string' ? ev.error : (ev.error.message ?? JSON.stringify(ev.error)))
          if (ev.content) {
            ttftMs ??= performance.now() - t0
            text += ev.content
            onToken?.(ev.content)
          }
          if (ev.stop) final = ev
        }
      }
      if (!final) error = 'stream ended without a final (stop) chunk'
    } catch (e) {
      const reason: unknown = ctl.signal.aborted ? ctl.signal.reason : e
      error = reason instanceof Error ? reason.message : String(reason)
    } finally {
      clearTimeout(timer)
      if (this.abort === ctl) this.abort = null
    }
    return toPromptResult(final, { ttftMs, totalMs: performance.now() - t0, text, timedOut, error })
  }

  /** One short discarded request so the measured run doesn't pay first-dispatch costs. Pass the measured
   *  prompt: a 1-token warmup left prefill at 47 tok/s on the 10-token measured run (b11208, RX 9070 XT). */
  async warmup(prompt = 'Hello'): Promise<void> {
    const r = await this.runPrompt({ prompt, maxTokens: 8, timeoutMs: 60_000 })
    if (r.error) throw new Error(`warmup failed: ${r.error}`)
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
