// A/B for the ~11.6 GiB dedicated ceiling seen on 2026-09-28 (8B f16 64K spilled at 11.6 GiB with the GPU idle, where
// on 2026-09-27 it held 12.6 GiB with no spill). Usage: npx tsx scripts/ab-spill.ts [out.json]
//   A1 fresh server after >=2 min idle GPU, 36,572-token prompt (the 0.56·ctx fill of the 09-28 runs)
//   A2 fresh server after >=2 min idle GPU, prompt tokenized to 49,152 (0.75·ctx)
//   B1 immediately after a q8_0 -c 131072 load of the same model, then the A1 launch
// Identical argv: -c 65536 -ngl 999 -dev Vulkan0 -t 8 -b 2048 -ub 512 -fa on -fit off --parallel 1. Warmup + 2 reps.
// --hip: Vulkan-vs-HIP ladder A/B instead — 8B f16 at 32K and 64K (0.56·ctx prompt), identical argv except the exe and
//   -dev Vulkan0 / ROCm0; first reports whether the HIP build enumerates ROCm0 (stops there if not).
// --igpu: can the iGPU's UMA memory replace the CPU layers? Qwen3.8-27B at 8K: (a) -ngl 49 on Vulkan0, rest on CPU
//   (session 3's clean config) vs (b) -ngl 999 -dev Vulkan0,Vulkan1 -ts 49,16 (the 16 CPU layers on the iGPU). The
//   per-device layer split is read from the load log. No per-device KV experiment: b11208 has no KV placement flag
//   (-mg places KV only with -sm row, which Vulkan does not implement).
// Telemetry: typeperf 1 s, per-PID dedicated/shared + adapter dedicated/shared (all LUIDs). At spill onset (per-PID shared
// > first sample + 256 MiB) records per-PID dedicated, adapter dedicated, adapter free and adapter total.
import { execFileSync, spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { freemem } from 'node:os'
import { createInterface } from 'node:readline'
import { generateFiller } from '../src/core/quality'

const EXE = 'vendor/llama.cpp/llama-server.exe'
const HIP_EXE = 'vendor/llama.cpp-hip/llama-server.exe'
const HIP = process.argv.includes('--hip')
const IGPU = process.argv.includes('--igpu')
const QWEN = 'D:\\llm-models\\Qwen3.8-27B-UD-Q4_K_M.gguf'
const MODEL = 'D:\\llm-models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
const ADAPTER_TOTAL = 17095983104 // RX 9070 XT qwMemorySize (scanner, registry)
const GiB = 1024 ** 3
const MiB = 1024 ** 2
const out = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? (HIP ? 'docs/ab-hip-2026-09-28.json' : IGPU ? 'docs/ab-igpu-2026-09-28.json' : 'docs/ab-spill-2026-09-28.json')
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const g = (b: number | null) => (b == null ? null : +(b / GiB).toFixed(2))
const baseArgv = (ctx: number, extra: string[] = [], dev = 'Vulkan0') => ['-m', MODEL, '-c', String(ctx), '-ngl', '999', '-dev', dev, '-t', '8', '-b', '2048', '-ub', '512', '-fa', 'on', '-fit', 'off', '--parallel', '1', ...extra]

/** Backend's own view (llama-server --list-devices): total/free MiB per device. */
function listDevices(exe = EXE): string[] {
  try { return execFileSync(exe, ['--list-devices'], { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }).split(/\r?\n/).filter((l) => /MiB/.test(l)).map((l) => l.trim()) } catch (e) { return [`error: ${(e as Error).message.slice(0, 120)}`] }
}
function vulkanHeaps(): string {
  try { return execFileSync('vulkaninfo', ['--summary'], { encoding: 'utf8', windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] }).split(/\r?\n/).filter((l) => /heap|MEMORY_HEAP|size\s*=/i.test(l)).slice(0, 30).join('\n') } catch { return 'vulkaninfo not available (skipped)' }
}

const servers = () => execFileSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true }).split('\n').filter((l) => /^"llama-server\.exe"/i.test(l)).length

interface Row { t: number; pidDed: number | null; pidShr: number | null; adapterDed: Record<string, number>; adapterShr: Record<string, number> }
function sampler(pid: number) {
  const ctr = [`\\GPU Process Memory(pid_${pid}_*)\\Dedicated Usage`, `\\GPU Process Memory(pid_${pid}_*)\\Shared Usage`, '\\GPU Adapter Memory(*)\\Dedicated Usage', '\\GPU Adapter Memory(*)\\Shared Usage']
  const c = spawn('typeperf', [...ctr, '-si', '1'], { windowsHide: true })
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
  return { rows, stop: () => c.kill() }
}

async function prompt(port: number, tokens: number): Promise<{ text: string; n: number }> {
  const tok = async (s: string) => ((await (await fetch(`http://127.0.0.1:${port}/tokenize`, { method: 'POST', body: JSON.stringify({ content: s }) })).json()) as { tokens: unknown[] }).tokens.length
  let est = tokens, text = '', n = 0
  for (let i = 0; i < 6; i++) {
    text = generateFiller(est, 65536).join(' ') + '\n\nContinue the story in the same style:\n'
    n = await tok(text)
    if (Math.abs(n - tokens) <= Math.max(16, tokens * 0.002)) break
    est = Math.max(16, Math.floor(est * (tokens / n)))
  }
  return { text, n }
}

async function launch(label: string, argv: string[], promptTokens: number | null, exe = EXE) {
  const port = 19100 + Math.floor(Math.random() * 500)
  const t0 = Date.now()
  const ramBefore = freemem()
  const devicesBefore = listDevices(exe)
  const p = spawn(exe, [...argv, '--port', String(port), '--host', '127.0.0.1'], { windowsHide: true })
  let log = ''
  p.stderr.on('data', (d) => { log += d.toString() })
  for (let i = 0; i < 480; i++) { try { if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) break } catch {} await sleep(250) }
  const loadMs = Date.now() - t0
  const s = sampler(p.pid!)
  await sleep(2500)
  const reps: { decodeTps: number | null; prefillTps: number | null; promptN: number | null; ttftMs: number | null }[] = []
  let promptN: number | null = null
  if (promptTokens) {
    const pr = await prompt(port, promptTokens)
    promptN = pr.n
    for (const phase of ['warmup', 'rep1', 'rep2']) {
      const r0 = Date.now()
      const j = (await (await fetch(`http://127.0.0.1:${port}/completion`, { method: 'POST', body: JSON.stringify({ prompt: pr.text, n_predict: phase === 'warmup' ? 8 : 128, temperature: 0, seed: 1, cache_prompt: false }) })).json()) as { timings?: { prompt_n?: number; prompt_per_second?: number; predicted_per_second?: number; prompt_ms?: number } }
      if (phase !== 'warmup') reps.push({ decodeTps: j.timings?.predicted_per_second ?? null, prefillTps: j.timings?.prompt_per_second ?? null, promptN: j.timings?.prompt_n ?? null, ttftMs: j.timings?.prompt_ms ?? Date.now() - r0 })
    }
  }
  await sleep(1500)
  s.stop()
  const ramMin = freemem()
  p.kill()
  for (let i = 0; i < 40 && servers() > 0; i++) await sleep(250)
  const rows = s.rows
  const luid = Object.entries(rows[0]?.adapterDed ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null
  const base = rows.find((r) => r.pidShr != null)?.pidShr ?? 0
  const onset = rows.find((r) => r.pidShr != null && r.pidShr > base + 256 * MiB) ?? null
  const peak = (f: (r: Row) => number | null) => { const v = rows.map(f).filter((x): x is number => x != null); return v.length ? Math.max(...v) : null }
  const bufs = (kind: string) => [...log.matchAll(/(\S+) (model|KV|compute) buffer size\s*=\s*([\d.]+) MiB/g)].filter((m) => m[2] === kind).map((m) => ({ dev: m[1], mib: Number(m[3]) }))
  const res = {
    label, exe, argv: argv.join(' '), loadMs, promptTokensRequested: promptTokens, promptTokensActual: promptN, reps,
    samples: rows.length, adapterLuid: luid, adapterTotalGiB: g(ADAPTER_TOTAL),
    peakPidDedicatedGiB: g(peak((r) => r.pidDed)), peakPidSharedGiB: g(peak((r) => r.pidShr)), pidSharedBaselineGiB: g(base),
    peakAdapterDedicatedGiB: g(peak((r) => (luid ? r.adapterDed[luid] ?? null : null))),
    spillOnset: onset ? { atSec: +((onset.t - t0) / 1000).toFixed(1), pidDedicatedGiB: g(onset.pidDed), pidSharedGiB: g(onset.pidShr), adapterDedicatedGiB: g(luid ? onset.adapterDed[luid] : null), adapterFreeGiB: g(luid ? ADAPTER_TOTAL - onset.adapterDed[luid] : null) } : null,
    buffersMiB: { model: bufs('model'), kv: bufs('KV'), compute: bufs('compute') },
    layersPerDevice: [...log.matchAll(/layer\s+\d+ assigned to device (\S+?),?\s/g)].reduce<Record<string, number>>((a, m) => ((a[m[1]] = (a[m[1]] ?? 0) + 1), a), {}),
    offloadLines: log.split(/\r?\n/).filter((l) => /offload(ing|ed) \d+/.test(l)).map((l) => l.trim()),
    largestBufferMiB: Math.max(0, ...[...bufs('model'), ...bufs('KV'), ...bufs('compute')].map((b) => b.mib)),
    listDevicesBefore: devicesBefore, ramAvailBeforeGiB: g(ramBefore), ramAvailAfterGiB: g(ramMin)
  }
  console.log(JSON.stringify(res))
  return res
}

async function idle(ms: number) {
  if (servers() > 0) throw new Error('llama-server already running')
  console.log(`idle ${ms / 1000}s (GPU quiet)`)
  await sleep(ms)
}

async function hipAb() {
  // Unified memory would let HIP page to host RAM silently — the ceiling comparison would be meaningless.
  if (process.env.GGML_CUDA_ENABLE_UNIFIED_MEMORY) throw new Error('GGML_CUDA_ENABLE_UNIFIED_MEMORY is set; unset it first')
  const hipDevices = listDevices(HIP_EXE)
  console.log(`HIP --list-devices: ${JSON.stringify(hipDevices)}`)
  const results = []
  if (hipDevices.some((l) => /ROCm0/.test(l))) {
    for (const ctx of [32768, 65536]) for (const [exe, dev] of [[EXE, 'Vulkan0'], [HIP_EXE, 'ROCm0']]) {
      await idle(120_000)
      results.push(await launch(`${dev} f16 ${ctx / 1024}K, ${Math.round(0.558 * ctx)}-token prompt`, baseArgv(ctx, [], dev), Math.round(0.558 * ctx), exe))
    }
  }
  writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), model: MODEL, hipDevices, vulkanDevices: listDevices(), unifiedMemoryEnv: process.env.GGML_CUDA_ENABLE_UNIFIED_MEMORY ?? null, results }, null, 1))
  console.log(`wrote ${out}; leftover llama-server ${servers()}`)
}

async function igpuAb() {
  const devices = listDevices()
  console.log(`--list-devices: ${JSON.stringify(devices)}`)
  const common = ['-m', QWEN, '-c', '8192', '-t', '8', '-b', '2048', '-ub', '512', '-fa', 'on', '-lm', 'none', '-fit', 'off', '--parallel', '1', '--cache-ram', '0', '-lv', '4']
  const results = []
  await idle(120_000)
  results.push(await launch('a Qwen3.8-27B 8K -ngl 49 Vulkan0, rest on CPU', [...common, '-ngl', '49', '-dev', 'Vulkan0'], 4572))
  if (devices.some((l) => /Vulkan1/.test(l))) {
    await idle(120_000)
    results.push(await launch('b Qwen3.8-27B 8K -ngl 999 Vulkan0,Vulkan1 -ts 49,16 (CPU layers on the iGPU)', [...common, '-ngl', '999', '-dev', 'Vulkan0,Vulkan1', '-ts', '49,16'], 4572))
  }
  writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), model: QWEN, devices, results }, null, 1))
  console.log(`wrote ${out}; leftover llama-server ${servers()}`)
}

void (async () => {
  if (HIP) return hipAb()
  if (IGPU) return igpuAb()
  const results = []
  await idle(120_000)
  results.push(await launch('A1 fresh, 36,572-token prompt', baseArgv(65536), 36572))
  await idle(120_000)
  results.push(await launch('A2 fresh, 49,152-token prompt (0.75·ctx)', baseArgv(65536), 49152))
  await idle(120_000)
  results.push(await launch('B1a q8_0 -c 131072 load (primes placement), no prompt', baseArgv(131072, ['-ctk', 'q8_0', '-ctv', 'q8_0']), null))
  results.push(await launch('B1b A1 launch immediately after the q8_0 128K load', baseArgv(65536), 36572))
  await idle(120_000)
  // B2: same as A1 but -ub 256 (smaller compute buffer): if the ceiling moves with the largest allocation, it is
  // placement of big buffers, not a fixed per-process budget.
  results.push(await launch('B2 fresh, -ub 256, 36,572-token prompt', baseArgv(65536).map((a, i, xs) => (xs[i - 1] === '-ub' ? '256' : a)), 36572))
  writeFileSync(out, JSON.stringify({ when: new Date().toISOString(), model: MODEL, vulkaninfoHeaps: vulkanHeaps(), results }, null, 1))
  console.log(`wrote ${out}; leftover llama-server ${servers()}`)
})()
