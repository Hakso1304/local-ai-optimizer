// Real-machine calibration matrix (GPU).
// Usage: npx tsx scripts/calibrate.ts                       -> the 2026-09-27 1.5B/8B matrix (rewrites the md)
//        npx tsx scripts/calibrate.ts --model <gguf> --ctx 2048,8192 [--ngl 99] [--partial 30] [--label X]
//        -> one ladder (+ one partial-offload point at the first spilled/failed rung), fed through detectCliffs,
//           appended to the md as its own section.
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
import { detectCliffs } from '../src/core/scoring/cliff'
import { DEFAULT_SCORING_CONFIG } from '../src/core/scoring/workloads'
import type { BenchmarkRunResult, Metric } from '../src/shared/bench-types'

const MODELS = 'D:\\llm-models'
const QWEN = join(MODELS, 'qwen2.5-1.5b-instruct-q4_k_m.gguf')
const LLAMA = join(MODELS, 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
const N_PREDICT = 128
const REQ_TIMEOUT = 5 * 60_000
const RAM_FLOOR = 4 * 1024 ** 3
const arg = (k: string) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : undefined }
const out = join('docs', 'calibration-2026-09-27.md')
const pidFile = join(tmpdir(), 'lao-calibrate.pid')
const GiB = (b: number | null) => (b == null ? 'n/a' : (b / 1024 ** 3).toFixed(2))
const f1 = (x: number | null | undefined) => (x == null ? 'n/a' : x.toFixed(1))
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

// ---- telemetry (typeperf) ----
interface Sample { ts: number; ded: Record<string, number>; shr: Record<string, number>; ramB: number | null; cpu: number | null; gpuUtil: number | null; pDed: number | null; pShr: number | null; pWs: number | null }
const LUID = /luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+(?:_eng_\d+_engtype_(.+))?/i

function typeperf(pid: number | null, count?: number) {
  const ctrs = ['\\GPU Adapter Memory(*)\\Dedicated Usage', '\\GPU Adapter Memory(*)\\Shared Usage', '\\Memory\\Available MBytes', '\\Processor(_Total)\\% Processor Time']
  if (pid) ctrs.push(`\\GPU Engine(pid_${pid}_*)\\Utilization Percentage`, `\\GPU Process Memory(pid_${pid}_*)\\Dedicated Usage`,
    `\\GPU Process Memory(pid_${pid}_*)\\Shared Usage`, `\\Process V2(llama-server:${pid})\\Working Set - Private`)
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
    const s: Sample = { ts: Date.now(), ded: {}, shr: {}, ramB: null, cpu: null, gpuUtil: null, pDed: null, pShr: null, pWs: null }
    const add = (k: 'pDed' | 'pShr', v: number) => { s[k] = (s[k] ?? 0) + v }
    const groups = new Map<string, number>()
    cols.forEach((c, i) => {
      const v = Number(cells[i])
      if (!cells[i]?.trim() || !Number.isFinite(v)) return
      if (c.obj === 'GPU Adapter Memory' && c.luid) (c.ctr === 'Dedicated Usage' ? s.ded : s.shr)[c.luid] = v
      else if (c.obj === 'Memory') s.ramB = v * 1024 * 1024
      else if (c.obj === 'Processor') s.cpu = v
      else if (c.obj === 'GPU Engine' && c.eng && /^(3D|Compute \d+)$/.test(c.eng)) groups.set(c.eng, (groups.get(c.eng) ?? 0) + v)
      else if (c.obj === 'GPU Process Memory' && (!luid || c.luid === luid)) add(c.ctr === 'Dedicated Usage' ? 'pDed' : 'pShr', v)
      else if (c.obj === 'Process V2') s.pWs = v
    })
    const u = groups.size ? Math.max(...groups.values()) : null
    if (pid) s.gpuUtil = u !== null && u >= 0 && u <= 100 ? u : null // PDH glitch: 1e13 % seen 3x on 2026-09-27
    samples.push(s)
  })
  const done = new Promise<void>((r) => child.on('close', () => r()))
  return { samples, stop: async () => { child.kill(); await done }, done }
}

function llamaServers(): number {
  const o = execFileSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true })
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
interface Run { procDed: number | null; procShr: number | null; procWs: number | null; phase: string; ttftMs: number | null; promptN: number | null; prefillTps: number | null; decodeTps: number | null; prefillMs: number | null; decodeMs: number | null; totalMs: number; wallGapMs: number | null; peakDed: number | null; sharedDelta: number | null; ramMin: number | null; gpuUtil: number | null; cpuAvg: number | null; error: string | null; timedOut: boolean }
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
      const pmax = (k: 'pDed' | 'pShr' | 'pWs') => { const v = pw.map((x) => x[k]).filter((x): x is number => x != null); return v.length ? Math.max(...v) : null }
      const cpu = adapter.samples.filter((x) => x.ts >= t0 && x.ts <= t1).map((x) => x.cpu).filter((x): x is number => x != null)
      const run: Run = {
        procDed: pmax('pDed'), procShr: pmax('pShr'), procWs: pmax('pWs'), phase, ttftMs: r.ttftMs, promptN: r.promptTokens, prefillTps: r.prefillTps, decodeTps: r.decodeTps, prefillMs: r.prefillMs, decodeMs: r.decodeMs,
        totalMs: r.totalMs, wallGapMs: r.prefillMs != null && r.decodeMs != null ? r.totalMs - r.prefillMs - r.decodeMs : null,
        peakDed: w.peakDed, sharedDelta: w.sharedDelta, ramMin: w.ramMin,
        gpuUtil: util.length ? util.reduce((a, c) => a + c, 0) / util.length : null, cpuAvg: cpu.length ? cpu.reduce((a, c) => a + c, 0) / cpu.length : null,
        error: r.error, timedOut: r.timedOut
      }
      row.runs.push(run)
      console.log(`${phase}: pp=${f1(run.prefillTps)} tg=${f1(run.decodeTps)} ttft=${f1(run.ttftMs)} n=${run.promptN} total=${f1(run.totalMs)} ded=${GiB(run.peakDed)} shrΔ=${GiB(run.sharedDelta)} ram=${GiB(run.ramMin)} gpu=${f1(run.gpuUtil)} pid:ded=${GiB(run.procDed)} shr=${GiB(run.procShr)} ws=${GiB(run.procWs)} ${run.error ?? ''}`)
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
  if (!custom) write()
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

const custom = !!arg('--model')

/** Warm runs (run1/run2, medians) -> one BenchmarkRunResult per rung, per-PID telemetry, for detectCliffs. */
function toStep(r: Row): BenchmarkRunResult {
  const warm = r.runs.filter((x) => x.phase !== 'warmup' && !x.error)
  const med = (f: (x: Run) => number | null): Metric => {
    const v = warm.map(f).filter((x): x is number => x != null).sort((a, b) => a - b)
    if (!v.length) return { value: null, kind: 'unavailable', reason: 'no warm sample' }
    return { value: v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2, kind: 'measured' }
  }
  const ok = r.status === 'ok' && warm.length > 0
  // HTTP 400 exceed_context_size: llama-server capped n_ctx below the requested -c (seen: 14B -c 49152 -> 32768).
  const kind = ok ? null : /exceed_context_size|exceeds the available context/.test(r.status) ? 'config_drift' : /oom/.test(r.status) ? 'oom' : /device_lost/.test(r.status) ? 'device_lost'
    : r.status === 'timeout' ? 'req_timeout' : r.status === 'ram<4GB' ? 'guard_abort' : 'crash'
  return {
    configId: `${r.model}|ngl=${r.ngl}`, ctx: r.ctx, promptTokens: warm[0]?.promptN ?? null,
    status: ok ? 'pass' : r.status === 'timeout' ? 'timeout' : 'fail', failureKind: kind,
    loadTimeMs: r.loadMs == null ? { value: null, kind: 'unavailable', reason: 'load failed' } : { value: r.loadMs, kind: 'measured' },
    ttftMs: med((x) => x.ttftMs), prefillTps: med((x) => x.prefillTps), decodeTps: med((x) => x.decodeTps), totalMs: med((x) => x.totalMs),
    peakVramBytes: med((x) => x.procDed), peakSharedGpuBytes: med((x) => x.procShr), peakRamBytes: med((x) => x.procWs),
    avgGpuUtil: med((x) => x.gpuUtil), avgCpuUtil: med((x) => x.cpuAvg)
  }
}

function appendSection(label: string, ladder: Row[], partial: Row | null, vramTotal: number): void {
  const steps = ladder.map(toStep)
  const report = detectCliffs(steps, vramTotal)
  const all = [...ladder, ...(partial ? [partial] : [])]
  const L: string[] = ['', `## ${label}`, '',
    `Per-PID telemetry (GPU Process Memory, Process V2 private WS); warm = median of run1/run2. detectCliffs with DEFAULT_SCORING_CONFIG.cliff = \`${JSON.stringify(DEFAULT_SCORING_CONFIG.cliff)}\`, vramTotal ${vramTotal}.`, '',
    '| ctx | ngl | status | load ms | layers | KV MiB | prompt_n | TTFT ms | prefill TPS | decode TPS | pid ded GiB | pid shared GiB | pid private WS GiB | adapter ded GiB | adapter shared Δ GiB | RAM min GiB | GPU util % | CPU % |',
    '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
  for (const r of all) {
    const warm = r.runs.filter((x) => x.phase !== 'warmup')
    const w = warm.length ? warm : r.runs
    const vals = (f: (x: Run) => number | null) => w.map(f).filter((x): x is number => x != null)
    const mean = (f: (x: Run) => number | null) => { const v = vals(f); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null }
    const mx = (f: (x: Run) => number | null) => { const v = vals(f); return v.length ? Math.max(...v) : null }
    const mn = (f: (x: Run) => number | null) => { const v = vals(f); return v.length ? Math.min(...v) : null }
    L.push(`| ${r.ctx} | ${r.ngl} | ${r.status} | ${f1(r.loadMs)} | ${r.layers} | ${r.kvMiB} | ${w[0]?.promptN ?? 'n/a'} | ${f1(mean((x) => x.ttftMs))} | ${f1(mean((x) => x.prefillTps))} | ${f1(mean((x) => x.decodeTps))} | ${GiB(mx((x) => x.procDed))} | ${GiB(mx((x) => x.procShr))} | ${GiB(mx((x) => x.procWs))} | ${GiB(mx((x) => x.peakDed))} | ${GiB(mx((x) => x.sharedDelta))} | ${GiB(mn((x) => x.ramMin))} | ${f1(mean((x) => x.gpuUtil))} | ${f1(mean((x) => x.cpuAvg))} |`)
  }
  L.push('', '### detectCliffs on the ngl-99 ladder', '',
    `practicalContextCeiling ${report.practicalContextCeiling.value ?? 'n/a'}, degradedContextCeiling ${report.degradedContextCeiling.value ?? 'n/a'}, limitedBy ${report.limitedBy}, spillFreeUpTo ${report.spillFreeUpTo ?? 'n/a'}`, '',
    '| ctx | verdict | reasons |', '|---|---|---|')
  for (const st of report.steps) L.push(`| ${st.ctx} | ${st.verdict} | ${st.reasons.map((x) => `${x.code}: ${x.message}`).join('; ') || '—'} |`)
  for (const r of all.filter((x) => x.tail.length)) L.push('', `Failure tail ctx ${r.ctx} ngl ${r.ngl}:`, '```', ...r.tail, '```')
  const prev = readFileSync(out, 'utf8')
  const cut = prev.indexOf(`\n## ${label}\n`)
  writeFileSync(out, (cut < 0 ? prev.replace(/\n+$/, '\n') : prev.slice(0, cut + 1)) + L.join('\n').replace(/^\n/, '') + '\n')
  writeFileSync(join(tmpdir(), `lao-calibration-${label.replace(/\W+/g, '_')}.json`), JSON.stringify({ rows: all, steps, report }, null, 2))
  console.log(JSON.stringify(report.steps.map((st) => ({ ctx: st.ctx, verdict: st.verdict, reasons: st.reasons.map((x) => x.message) }))))
}

async function runCustom(dev: string, vramTotal: number): Promise<void> {
  const model = arg('--model')!
  const ctxs = (arg('--ctx') ?? '2048,8192').split(',').map(Number)
  const ngl = Number(arg('--ngl') ?? 99)
  const label = arg('--label') ?? model.split('\\').pop()!
  const ladder: Row[] = []
  for (const ctx of ctxs) {
    const r = await measure(model, ctx, ngl, dev)
    ladder.push(r)
    appendSection(label, ladder, null, vramTotal)
    if (r.status !== 'ok') { console.log(`ladder stop at ${ctx}: ${r.status}`); break }
  }
  if (!arg('--partial')) return
  // First rung that failed or spilled per PID (> the detector's shared threshold); else the largest rung.
  const spilled = ladder.find((r) => r.status !== 'ok' || r.runs.some((x) => (x.procShr ?? 0) > DEFAULT_SCORING_CONFIG.cliff.sharedSpillBytes))
  const partial = await measure(model, (spilled ?? ladder[ladder.length - 1]).ctx, Number(arg('--partial')), dev)
  appendSection(label, ladder, partial, vramTotal)
}

async function main(): Promise<void> {
  const b = new LlamaCppBackend(join('vendor', 'llama.cpp'))
  const dev = pickDiscreteDevice(await b.listDevices())
  if (!dev) throw new Error('no discrete device')
  console.log(`device ${dev.id} ${dev.name}`)
  if (custom) {
    await runCustom(dev.id, 17095983104) // qwMemorySize of the RX 9070 XT (scanner, registry)
    console.log(`\nleftover llama-server processes: ${llamaServers()}`)
    return
  }
  for (const ctx of [2048, 8192, 32768]) await measure(QWEN, ctx, 99, dev.id)
  for (const ctx of [2048, 4096, 8192, 16384, 32768, 65536]) {
    const r = await measure(LLAMA, ctx, 99, dev.id)
    if (r.status !== 'ok') { console.log(`ladder stop at ${ctx}: ${r.status}`); break }
  }
  for (const ngl of [20, 0]) await measure(LLAMA, 8192, ngl, dev.id)
  console.log(`\nleftover llama-server processes: ${llamaServers()}`)
}

main().catch((e) => { console.error(e); process.exit(1) })
