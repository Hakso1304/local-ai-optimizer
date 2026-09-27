import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
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
import { pickReleaseAsset, type ReleaseAsset } from './assets'
import { getJson, type HealthStatus, type InferenceBackend, type LoadConfig, type LoadResult, type ModelInfo, type PromptRequest, type PromptResult, type RuntimeStats } from '../types'
import { acceptedSampling, classifyExit, emptyDeclared, parseDevices, parseLogLine, parseSse, toPromptResult, type CompletionChunk, type ExitReason, type LlamaDevice } from './parse'

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
  private exeOverride?: string
  private pidFile?: string
  private spawnFn: SpawnFn
  /** Last 100 output lines, for crash diagnostics. */
  readonly log: string[] = []
  /** Set when the server exits on its own (not via unloadModel). */
  lastExit: ExitInfo | null = null
  /** sha256 of the loaded model's chat_template (/props, read once per load); null = none reported / not loaded. */
  templateHash: string | null = null

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

  /** Download + extract the newest official Windows build for this GPU into vendorDir if not installed.
   *  NVIDIA with a CUDA-capable driver → CUDA build + cudart (same dir); anything else, or any CUDA failure → Vulkan.
   *  Installed = release-tag.txt ("<tag> <build>") exists; it is written last, after an atomic rename. */
  async ensureRuntime(log: (m: string) => void = () => {}, pref: { vendor: GpuVendor; cudaMajor?: number } = { vendor: 'other' }): Promise<RuntimeDetection> {
    if (existsSync(join(this.vendorDir, 'release-tag.txt'))) return this.detect()
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
        const res = await fetch(asset.browser_download_url, { signal: AbortSignal.timeout(15 * 60_000) })
        if (!res.ok || !res.body) throw new Error(`download ${asset.browser_download_url} -> HTTP ${res.status}`)
        let received = 0
        let shown = -1
        const body = Readable.fromWeb(res.body as WebReadableStream).on('data', (c: Buffer) => {
          received += c.length
          const pct = Math.floor((received * 100) / asset.size / 5) * 5
          if (pct !== shown) { shown = pct; log(`downloading ${asset.name} ${pct}%`) }
        })
        await pipeline(body, createWriteStream(zip))
        const got = statSync(zip).size
        if (got !== asset.size) throw new Error(`download truncated: ${got} of ${asset.size} bytes`)
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
  async loadModel(cfg: LoadConfig): Promise<LoadResult> {
    if (this.proc) await this.unloadModel()
    if (cfg.signal?.aborted) throw new Error('cancelled')
    this.port = cfg.port ?? (await freePort())
    this.log.length = 0
    this.lastExit = null
    this.templateHash = null
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
    if (this.pidFile && p.pid) writeFileSync(this.pidFile, JSON.stringify({ pid: p.pid, exePath: resolve(this.exePath), startedAt: new Date().toISOString() } satisfies PidRecord))
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
      if (this.proc !== p || this.unloading === p) return // being unloaded: unloadModel finishes the cleanup
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
      if (cfg.signal?.aborted) {
        await this.unloadModel()
        throw new Error('cancelled')
      }
      if (exited) throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`)
      healthy = (await this.healthCheck()).ok
      if (!healthy) await new Promise((r) => setTimeout(r, 100))
    }
    if (!healthy) {
      await this.unloadModel()
      throw new Error('llama-server did not become healthy within 120s')
    }
    const loadTimeMs = performance.now() - t0
    const props = await getJson<{ model_path?: string; chat_template?: string; default_generation_settings?: { n_ctx?: number } }>(`http://127.0.0.1:${this.port}/props`, 2_000).catch(() => null)
    if (exited) throw new Error(`${exited}; last log: ${this.log.slice(-10).join(' | ')}`)
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
    return { loadTimeMs, declared, templateHash: this.templateHash }
  }

  /** Kill the server; escalate to taskkill /T /F after 5s. Throws if it is still alive afterwards. */
  /** Kill the server; escalate to taskkill /T /F after 5s. The handle and pid file are only dropped once the exit
   *  is confirmed; if the process survives, ServerStuckError is thrown and both are kept (W4 F4), so a later unload
   *  or the next app start (killStaleServer) can still reach it. The runner must treat it as a session stop. */
  async unloadModel(): Promise<void> {
    const p = this.proc
    if (!p || p.pid === undefined || !alive(p)) {
      if (this.proc === p) this.proc = null
      return this.clearPid()
    }
    this.unloading = p
    try {
      p.kill()
      if (!(await waitExit(p, 5_000))) {
        await runProcess('taskkill', ['/PID', String(p.pid), '/T', '/F'], 10_000).catch(() => {})
        if (!(await waitExit(p, 3_000))) throw new ServerStuckError(`llama-server pid ${p.pid} still alive after taskkill /F`)
      }
    } finally {
      this.unloading = null
    }
    if (this.proc === p) this.proc = null
    this.clearPid()
  }

  /** PID of the running server (for the telemetry sampler). */
  get pid(): number | undefined {
    return this.proc?.pid
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
    const early = (error: string) => toPromptResult(null, { ttftMs: null, totalMs: 0, text: '', timedOut: false, error })
    if (!this.proc) return early(this.lastExit ? `server exited (${this.lastExit.reason}, code ${this.lastExit.code})` : 'no model loaded')
    if (this.abort) return early('another prompt is already in flight') // --parallel 1; cancel() has a single slot
    const timeoutMs = req.timeoutMs ?? 120_000
    const ctl = new AbortController()
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
        signal: ctl.signal
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
      const reason: unknown = ctl.signal.aborted ? ctl.signal.reason : e
      error = reason instanceof Error ? reason.message : String(reason)
    } finally {
      clearTimeout(timer)
      if (this.abort === ctl) this.abort = null
    }
    const totalMs = performance.now() - t0
    const f = final as CompletionChunk | null
    const needPrompt = !error && f !== null && f.timings?.prompt_n == null && f.tokens_evaluated == null
    const promptTokens = needPrompt ? await this.tokenize(req.prompt).catch(() => null) : null
    return { ...toPromptResult(final, { ttftMs, totalMs, text, timedOut, error, streamedTokens: streamed, promptTokens }), streamedTokens: streamed, reasoningTokens: think.seen ? think.tokens : null, acceptedSampling: acceptedSampling(f) }
  }

  /** One short discarded request so the measured run doesn't pay first-dispatch costs. Pass the measured
   *  prompt: a 1-token warmup left prefill at 47 tok/s on the 10-token measured run (b11208, RX 9070 XT). */
  async warmup(prompt = 'Hello', timeoutMs = 120_000): Promise<void> {
    const r = await this.runPrompt({ prompt, maxTokens: 8, timeoutMs })
    if (r.error) throw new Error(`warmup failed: ${r.error}`)
  }

  /** Token count of `text` with the loaded model's tokenizer (POST /tokenize). */
  async tokenize(text: string): Promise<number> {
    const res = await fetch(`http://127.0.0.1:${this.port}/tokenize`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ content: text }), signal: AbortSignal.timeout(30_000)
    })
    if (!res.ok) throw new Error(`POST /tokenize -> HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`)
    return ((await res.json()) as { tokens: unknown[] }).tokens.length
  }

  /** Apply the model's chat template (POST /apply-template) so chat-format prompts can go through runPrompt. */
  /** Chat template → raw prompt. date_string is always pinned: templates like Llama 3.1's inject today's date via
   *  strftime_now, which would make quality prompts differ by day. templateKwargs (e.g. enable_thinking: false) win. */
  async applyTemplate(messages: { role: string; content: string }[], opts: { templateKwargs?: Record<string, unknown> } = {}): Promise<string> {
    const chat_template_kwargs = { date_string: TEMPLATE_DATE, ...opts.templateKwargs }
    const res = await fetch(`http://127.0.0.1:${this.port}/apply-template`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages, chat_template_kwargs }), signal: AbortSignal.timeout(10_000)
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
