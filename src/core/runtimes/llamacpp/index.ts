import { spawn, type ChildProcess } from 'node:child_process'
import { createWriteStream, existsSync, readdirSync, readFileSync, rmSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as WebReadableStream } from 'node:stream/web'
import { runPowerShell, runProcess } from '../../exec'
import type { RuntimeDetection } from '../../../shared/types'
import { NotImplementedError, getJson, type HealthStatus, type InferenceBackend, type LoadConfig, type ModelRef } from '../types'

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

export class LlamaCppBackend implements InferenceBackend {
  readonly id = 'llamacpp' as const
  private proc: ChildProcess | null = null
  private port = 8089
  readonly log: string[] = []

  constructor(
    private vendorDir: string,
    private exeOverride?: string
  ) {}

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

  /** Download + extract the newest official Windows Vulkan build into vendorDir if not present. */
  async ensureRuntime(log: (m: string) => void = () => {}): Promise<RuntimeDetection> {
    if (existsSync(join(this.vendorDir, 'llama-server.exe'))) return this.detect()
    const releases = await getJson<GhRelease[]>(RELEASES_URL, 15_000)
    const asset = pickVulkanAsset(releases)
    if (!asset) throw new Error(`no release among the latest ${releases.length} has a win-vulkan-x64 zip`)
    log(`downloading ${asset.name} (${(asset.size / 1e6).toFixed(1)} MB)`)
    const zip = join(tmpdir(), asset.name)
    const res = await fetch(asset.url, { signal: AbortSignal.timeout(15 * 60_000) })
    if (!res.ok || !res.body) throw new Error(`download ${asset.url} -> HTTP ${res.status}`)
    await pipeline(Readable.fromWeb(res.body as WebReadableStream), createWriteStream(zip))
    log(`extracting to ${this.vendorDir}`)
    mkdirSync(this.vendorDir, { recursive: true })
    const q = (s: string) => `'${s.replace(/'/g, "''")}'`
    await runPowerShell(`Expand-Archive -LiteralPath ${q(zip)} -DestinationPath ${q(this.vendorDir)} -Force`, 5 * 60_000)
    rmSync(zip, { force: true })
    // ponytail: assumes flat zip layout (true for current releases); hoist from a subfolder if that changes.
    if (!existsSync(join(this.vendorDir, 'llama-server.exe'))) {
      throw new Error(`llama-server.exe missing after extract; got: ${readdirSync(this.vendorDir).slice(0, 10).join(', ')}`)
    }
    writeFileSync(join(this.vendorDir, 'release-tag.txt'), asset.tag)
    return this.detect()
  }

  /** Start llama-server for one model and wait until /health is OK. */
  async loadModel(cfg: LoadConfig): Promise<void> {
    if (this.proc) await this.unloadModel()
    this.port = cfg.port ?? 8089
    this.log.length = 0
    const args = ['-m', cfg.modelPath, '-c', String(cfg.contextSize), '-ngl', String(cfg.gpuLayers), '--host', '127.0.0.1', '--port', String(this.port)]
    if (cfg.threads) args.push('-t', String(cfg.threads))
    if (cfg.batchSize) args.push('-b', String(cfg.batchSize))
    args.push(...(cfg.extraArgs ?? []))
    const p = spawn(this.exePath, args, { windowsHide: true })
    this.proc = p
    const onData = (d: Buffer) => {
      for (const line of d.toString().split(/\r?\n/)) if (line) this.log.push(line)
      if (this.log.length > 2000) this.log.splice(0, this.log.length - 2000)
    }
    p.stdout?.on('data', onData)
    p.stderr?.on('data', onData)
    let exited: string | null = null
    p.on('exit', (code) => { exited = `llama-server exited with code ${code}`; if (this.proc === p) this.proc = null })
    p.on('error', (e) => { exited = `llama-server failed to start: ${e.message}` })

    const deadline = Date.now() + 120_000
    while (Date.now() < deadline) {
      if (exited) throw new Error(`${exited}; last log: ${this.log.slice(-5).join(' | ')}`)
      if ((await this.healthCheck()).ok) return
      await new Promise((r) => setTimeout(r, 250))
    }
    await this.unloadModel()
    throw new Error('llama-server did not become healthy within 120s')
  }

  async unloadModel(): Promise<void> {
    const p = this.proc
    this.proc = null
    if (!p || p.exitCode !== null) return
    await new Promise<void>((resolve) => {
      const t = setTimeout(resolve, 5_000)
      p.once('exit', () => { clearTimeout(t); resolve() })
      p.kill()
    })
  }

  async healthCheck(): Promise<HealthStatus> {
    try {
      const j = await getJson<{ status?: string }>(`http://127.0.0.1:${this.port}/health`, 1_500)
      return { ok: j.status === 'ok', detail: JSON.stringify(j) }
    } catch (e) {
      return { ok: false, detail: (e as Error).message }
    }
  }

  async enumerateModels(): Promise<ModelRef[]> { throw new NotImplementedError('llamacpp.enumerateModels') }
  async runPrompt(): Promise<never> { throw new NotImplementedError('llamacpp.runPrompt') }
  async getRuntimeStats(): Promise<never> { throw new NotImplementedError('llamacpp.getRuntimeStats') }
  async cancel(): Promise<void> { throw new NotImplementedError('llamacpp.cancel') }
}
