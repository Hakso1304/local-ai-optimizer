// Real-machine calibration matrix (GPU). Usage: npx tsx scripts/calibrate.ts [outMd]
// Measures load/prefill/decode via LlamaCppBackend + adapter/process telemetry via typeperf, writes raw tables.
// ponytail: own typeperf reader because src/core/telemetry/sampler.ts had not landed when this was written;
// switch to startSampler() once it is committed.
import { execFileSync, spawn } from 'node:child_process'
import { readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createInterface } from 'node:readline'
import { LlamaCppBackend } from '../src/core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../src/core/runtimes/llamacpp/parse'
import { generateFiller } from '../src/core/quality'

const MODELS = 'D:\\llm-models'
const QWEN = join(MODELS, 'qwen2.5-1.5b-instruct-q4_k_m.gguf')
const LLAMA = join(MODELS, 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
const N_PREDICT = 128
const REQ_TIMEOUT = 5 * 60_000
const RAM_FLOOR = 4 * 1024 ** 3
const out = process.argv[2] ?? join('docs', 'calibration-2026-09-27.md')
const pidFile = join(tmpdir(), 'lao-calibrate.pid')
const GiB = (b: number | null) => (b == null ? 'n/a' : (b / 1024 ** 3).toFixed(2))
const f1 = (x: number | null | undefined) => (x == null ? 'n/a' : x.toFixed(1))
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

// ---- telemetry (typeperf) ----
interface Sample { ts: number; ded: Record<string, number>; shr: Record<string, number>; ramB: number | null; cpu: number | null; gpuUtil: number | null }
const LUID = /luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+(?:_eng_\d+_engtype_(.+))?/i

function typeperf(pid: number | null, count?: number) {
  const ctrs = ['\\GPU Adapter Memory(*)\\Dedicated Usage', '\\GPU Adapter Memory(*)\\Shared Usage', '\\Memory\\Available MBytes', '\\Processor(_Total)\\% Processor Time']
  if (pid) ctrs.push(`\\GPU Engine(pid_${pid}_*)\\Utilization Percentage`)
  const child = spawn('typeperf', [...ctrs, '-si', '1', ...(count ? ['-sc', String(count)] : [])], { windowsHide: true })
  const samples: Sample[] = []
  let cols: { obj: string; ctr: string; luid: string | null; eng: string | null }[] | null = null
  createInterface({ input: child.stdout }).on('line', (l) => {
    if (!l.startsWith('"')) return
    const cells = l.trim().replace(/^"|"$/g, '').split('","')
    if (!cols) {
      cols = cells.map((h) => {
        const m = /^\\\\[^\\]+\\([^(\\]+)(?:\((.*)\))?\\(.+)$/.exec(h)
        const i = LUID.exec(m?.[2] ?? '')
        return { obj: m?.[1] ?? '', ctr: m?.[3] ?? '', luid: i?.[1]?.toLowerCase() ?? null, eng: i?.[2] ?? null }
      })
      return
    }
    const s: Sample = { ts: Date.now(), ded: {}, shr: {}, ramB: null, cpu: null, gpuUtil: null }
    const groups = new Map<string, number>()
    cols.forEach((c, i) => {
      const v = Number(cells[i])
      if (!cells[i]?.trim() || !Number.isFinite(v)) return
      if (c.obj === 'GPU Adapter Memory' && c.luid) (c.ctr === 'Dedicated Usage' ? s.ded : s.shr)[c.luid] = v
      else if (c.obj === 'Memory') s.ramB = v * 1024 * 1024
      else if (c.obj === 'Processor') s.cpu = v
      else if (c.obj === 'GPU Engine' && c.eng && /^(3D|Compute \d+)$/.test(c.eng)) groups.set(c.eng, (groups.get(c.eng) ?? 0) + v)
    })
    if (pid) s.gpuUtil = groups.size ? Math.max(...groups.values()) : null
    samples.push(s)
  })
  const done = new Promise<void>((r) => child.on('close', () => r()))
  return { samples, stop: async () => { child.kill(); await done }, done }
}

function llamaServers(): number {
  const o = execFileSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], { encoding: 'utf8' })
  return o.split('\n').filter((l) => /^"llama-server\.exe"/i.test(l)).length
}
async function waitNoServer(): Promise<void> {
  for (let i = 0; i < 20; i++) {
    if (llamaServers() === 0) return
    console.log('llama-server already running (another worker?) — waiting 30s')
    await sleep(30_000)
  }
  throw new Error('llama-server still running after 10 min; aborting')
}

// ---- matrix ----
interface Run { phase: string; ttftMs: number | null; promptN: number | null; prefillTps: number | null; decodeTps: number | null; prefillMs: number | null; decodeMs: number | null; totalMs: number; wallGapMs: number | null; peakDed: number | null; sharedDelta: number | null; ramMin: number | null; gpuUtil: number | null; cpuAvg: number | null; error: string | null; timedOut: boolean }
interface Row { model: string; ctx: number; ngl: number; status: string; loadMs: number | null; layers: string; modelMiB: string; kvMiB: string; computeMiB: string; base: { ded: number; shr: number; ram: number }; runs: Run[]; loadPeakDed: number | null; tail: string[] }

let luid = ''
const rows: Row[] = []

async function baseline() {
  const t = typeperf(null, 3)
  await t.done
  const last = t.samples[t.samples.length - 1]
  if (!luid) luid = Object.entries(last.ded).sort((a, b) => b[1] - a[1])[0][0] // dGPU = most dedicated in use
  return { ded: last.ded[luid], shr: last.shr[luid], ram: last.ramB ?? NaN }
}

function window(s: Sample[], t0: number, t1: number, base: { shr: number }) {
  const w = s.filter((x) => x.ts >= t0 && x.ts <= t1 + 1500)
  const max = (xs: (number | null | undefined)[]) => { const v = xs.filter((x): x is number => x != null); return v.length ? Math.max(...v) : null }
  const min = (xs: (number | null | undefined)[]) => { const v = xs.filter((x): x is number => x != null); return v.length ? Math.min(...v) : null }
  const shr = max(w.map((x) => x.shr[luid]))
  return { peakDed: max(w.map((x) => x.ded[luid])), sharedDelta: shr == null ? null : shr - base.shr, ramMin: min(w.map((x) => x.ramB)), n: w.length }
}

async function prompt(port: number, ctx: number): Promise<string> {
  const target = Math.floor(ctx * 0.75)
  const tok = async (s: string) => ((await (await fetch(`http://127.0.0.1:${port}/tokenize`, { method: 'POST', body: JSON.stringify({ content: s }) })).json()) as { tokens: unknown[] }).tokens.length
  let est = target
  let text = ''
  for (let i = 0; i < 3; i++) {
    text = generateFiller(est, 1).join(' ') + '\n\nSummarize the text above in one paragraph.'
    const n = await tok(text)
    if (Math.abs(n - target) / target < 0.03 && n + N_PREDICT < ctx) break
    est = Math.floor(est * (target / n))
  }
  return text
}

async function measure(model: string, ctx: number, ngl: number, device: string): Promise<Row> {
  await waitNoServer()
  const base = await baseline()
  const row: Row = { model: model.split('\\').pop()!.replace('.gguf', ''), ctx, ngl, status: 'ok', loadMs: null, layers: 'n/a', modelMiB: '', kvMiB: '', computeMiB: '', base, runs: [], loadPeakDed: null, tail: [] }
  console.log(`\n=== ${row.model} ctx=${ctx} ngl=${ngl}  base ded=${GiB(base.ded)} shr=${GiB(base.shr)} ram=${GiB(base.ram)}`)
  const adapter = typeperf(null)
  const b = new LlamaCppBackend(join('vendor', 'llama.cpp'), { pidFile })
  let proc: ReturnType<typeof typeperf> | null = null
  try {
    const tl0 = Date.now()
    const load = await b.loadModel({ modelPath: model, contextSize: ctx, gpuLayers: ngl, device })
    row.loadMs = load.loadTimeMs
    row.loadPeakDed = window(adapter.samples, tl0, Date.now(), base).peakDed
    const d = load.declared
    row.layers = d.layersOffloaded == null ? 'n/a' : `${d.layersOffloaded}/${d.layersTotal}`
    const kv = (m: Record<string, number>) => Object.entries(m).map(([k, v]) => `${k}:${v.toFixed(0)}`).join(' ') || 'n/a'
    row.modelMiB = kv(d.modelBufferMiB); row.kvMiB = kv(d.kvBufferMiB); row.computeMiB = kv(d.computeBufferMiB)
    const pid = Number(readFileSync(pidFile, 'utf8'))
    proc = typeperf(pid)
    const port = Number((b as unknown as { port: number }).port)
    const text = await prompt(port, ctx)
    await sleep(1500) // let the pid sampler emit its header
    for (const phase of ['warmup', 'run1', 'run2']) {
      const t0 = Date.now()
      const r = await b.runPrompt({ prompt: text, maxTokens: N_PREDICT, temperature: 0, seed: 1, timeoutMs: REQ_TIMEOUT })
      const t1 = Date.now()
      const w = window(adapter.samples, t0, t1, base)
      const pw = proc.samples.filter((x) => x.ts >= t0 && x.ts <= t1 + 1500)
      const util = pw.map((x) => x.gpuUtil).filter((x): x is number => x != null)
      const cpu = adapter.samples.filter((x) => x.ts >= t0 && x.ts <= t1).map((x) => x.cpu).filter((x): x is number => x != null)
      const run: Run = {
        phase, ttftMs: r.ttftMs, promptN: r.promptTokens, prefillTps: r.prefillTps, decodeTps: r.decodeTps, prefillMs: r.prefillMs, decodeMs: r.decodeMs,
        totalMs: r.totalMs, wallGapMs: r.prefillMs != null && r.decodeMs != null ? r.totalMs - r.prefillMs - r.decodeMs : null,
        peakDed: w.peakDed, sharedDelta: w.sharedDelta, ramMin: w.ramMin,
        gpuUtil: util.length ? util.reduce((a, c) => a + c, 0) / util.length : null, cpuAvg: cpu.length ? cpu.reduce((a, c) => a + c, 0) / cpu.length : null,
        error: r.error, timedOut: r.timedOut
      }
      row.runs.push(run)
      console.log(`${phase}: pp=${f1(run.prefillTps)} tg=${f1(run.decodeTps)} ttft=${f1(run.ttftMs)} n=${run.promptN} total=${f1(run.totalMs)} ded=${GiB(run.peakDed)} shrΔ=${GiB(run.sharedDelta)} ram=${GiB(run.ramMin)} gpu=${f1(run.gpuUtil)} ${run.error ?? ''}`)
      if (r.error) { row.status = r.timedOut ? 'timeout' : `error: ${r.error}`; break }
      if (w.ramMin != null && w.ramMin < RAM_FLOOR) { row.status = 'ram<4GB'; break }
    }
  } catch (e) {
    row.status = `fail: ${b.lastExit?.reason ?? 'load'} — ${(e as Error).message.slice(0, 160)}`
    row.tail = b.lastExit?.tail.slice(-8) ?? b.log.slice(-8)
    console.log(row.status)
  } finally {
    await b.unloadModel().catch((e) => console.log(`unload failed: ${(e as Error).message}`))
    await proc?.stop()
    await adapter.stop()
  }
  rows.push(row)
  write()
  return row
}

function write(): void {
  const L: string[] = ['# Calibration 2026-09-27 (RX 9070 XT 16 GB, Ryzen 7 9800X3D, 31 GB RAM, llama.cpp b11208 Vulkan)', '']
  L.push('Prompt = generateFiller(seed 1) tokenized to ≈0.75·ctx, n_predict 128, temp 0, seed 1, cache_prompt false. 1 warmup (same prompt) + 2 measured.')
  L.push(`Telemetry: typeperf 1 s; adapter luid ${luid}; shared Δ = peak adapter Shared Usage − idle baseline before that config; GPU util = mean over pid's 3D/Compute engine groups (max group).`, '')
  L.push('## Per config', '', '| model | ctx | ngl | status | load ms | layers | model MiB | KV MiB | compute MiB | idle ded GiB | idle shared GiB | idle RAM GiB | load peak ded GiB |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|')
  for (const r of rows) L.push(`| ${r.model} | ${r.ctx} | ${r.ngl} | ${r.status} | ${f1(r.loadMs)} | ${r.layers} | ${r.modelMiB} | ${r.kvMiB} | ${r.computeMiB} | ${GiB(r.base.ded)} | ${GiB(r.base.shr)} | ${GiB(r.base.ram)} | ${GiB(r.loadPeakDed)} |`)
  L.push('', '## Per request', '', '| model | ctx | ngl | phase | prompt_n | TTFT ms | prefill TPS | decode TPS | prefill ms | decode ms | total ms | total−(pp+tg) ms | peak ded GiB | shared Δ GiB | RAM min GiB | GPU util % | CPU avg % | error |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
  for (const r of rows) for (const x of r.runs) L.push(`| ${r.model} | ${r.ctx} | ${r.ngl} | ${x.phase} | ${x.promptN ?? 'n/a'} | ${f1(x.ttftMs)} | ${f1(x.prefillTps)} | ${f1(x.decodeTps)} | ${f1(x.prefillMs)} | ${f1(x.decodeMs)} | ${f1(x.totalMs)} | ${f1(x.wallGapMs)} | ${GiB(x.peakDed)} | ${GiB(x.sharedDelta)} | ${GiB(x.ramMin)} | ${f1(x.gpuUtil)} | ${f1(x.cpuAvg)} | ${x.error ?? ''} |`)
  const fails = rows.filter((r) => r.tail.length)
  if (fails.length) {
    L.push('', '## Failure tails', '')
    for (const r of fails) L.push(`### ${r.model} ctx ${r.ctx} ngl ${r.ngl}`, '```', ...r.tail, '```')
  }
  writeFileSync(out, L.join('\n') + '\n')
  writeFileSync(join(tmpdir(), 'lao-calibration.json'), JSON.stringify(rows, null, 2))
}

async function main(): Promise<void> {
  const b = new LlamaCppBackend(join('vendor', 'llama.cpp'))
  const dev = pickDiscreteDevice(await b.listDevices())
  if (!dev) throw new Error('no discrete device')
  console.log(`device ${dev.id} ${dev.name}`)
  for (const ctx of [2048, 8192, 32768]) await measure(QWEN, ctx, 99, dev.id)
  for (const ctx of [2048, 4096, 8192, 16384, 32768, 65536]) {
    const r = await measure(LLAMA, ctx, 99, dev.id)
    if (r.status !== 'ok') { console.log(`ladder stop at ${ctx}: ${r.status}`); break }
  }
  for (const ngl of [20, 0]) await measure(LLAMA, 8192, ngl, dev.id)
  console.log(`\nleftover llama-server processes: ${llamaServers()}`)
}

main().catch((e) => { console.error(e); process.exit(1) })
