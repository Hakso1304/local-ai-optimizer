import { useEffect, useRef, useState } from 'react'
import type { QuantSuggestion, WorkloadId, WorkloadProfile } from '../../shared/bench-types'
import type { ModelInfo } from '../../shared/types'
import type { LiveState } from './benchState'
import { presetFor, type BenchPreset } from './LargeCodingCard'
import { fmtCtx, gib, num } from './ui'
import { Suggestions } from './Suggestions'

const v = (x: number | null | undefined, f: (n: number) => string) => (x == null ? '—' : f(x))
const LADDER = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

export function BenchmarkPage({ live, preset, onDownload }: { live: LiveState; preset?: BenchPreset; onDownload: () => void }) {
  const [workloads, setWorkloads] = useState<WorkloadProfile[]>([])
  const [workload, setWorkload] = useState<WorkloadId | null>(null)
  const [models, setModels] = useState<ModelInfo[] | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [msg, setMsg] = useState<string | null>(null)
  const [maxCtx, setMaxCtx] = useState<number>(0) // 0 = no cap (candidate rules decide)
  const [quality, setQuality] = useState(true)
  const [heavy, setHeavy] = useState(false)
  const [genSearch, setGenSearch] = useState(true)
  // Vulkan vs ROCm/HIP: offered (and on by default) only when both builds are installed.
  const [bothBackends, setBothBackends] = useState(false)
  const [compareBackends, setCompareBackends] = useState(true)
  useEffect(() => {
    window.api.installedBackends().then((bs) => setBothBackends(bs.filter((b) => b.status === 'available').length > 1), () => setBothBackends(false))
  }, [])
  const [qualityMode, setQualityMode] = useState<'thorough' | 'quick'>('thorough')
  const [reqCtx, setReqCtx] = useState(0) // 0 = Auto (workload default)
  const [minDec, setMinDec] = useState('') // blank = workload default
  // Choosing a workload with a preset applies it like the Dashboard button; leaving it restores what the user had.
  const beforePreset = useRef<{ heavy: boolean; reqCtx: number } | null>(null)
  const pickWorkload = (w: WorkloadId) => {
    const p = presetFor(w)
    if (p) {
      beforePreset.current ??= { heavy, reqCtx }
      setHeavy(p.heavyMode)
      setReqCtx(p.requiredContext ?? 0)
    } else if (beforePreset.current) {
      setHeavy(beforePreset.current.heavy)
      setReqCtx(beforePreset.current.reqCtx)
      beforePreset.current = null
    }
    setWorkload(w)
  }
  const [fit, setFit] = useState<Record<string, string | null>>({})
  const [suggestions, setSuggestions] = useState<QuantSuggestion[]>([])
  const [vram, setVram] = useState<{ inUse: number | null; total: number | null }>({ inUse: null, total: null })

  useEffect(() => {
    void Promise.all([window.api.listWorkloads(), window.api.getSettings()]).then(([ws, s]) => { setWorkloads(ws)
      // A preset (e.g. Dashboard "Benchmark for large-scale coding") wins over the saved choices.
      const w = preset?.workload ?? s.workload ?? ws[0]?.id ?? null
      const p = preset ?? presetFor(w) // a saved preset workload (e.g. large_coding) applies its preset too
      setWorkload(w)
      setReqCtx(p ? p.requiredContext ?? 0 : s.requiredContext ?? 0)
      if (p) { beforePreset.current = { heavy: false, reqCtx: s.requiredContext ?? 0 }; setHeavy(p.heavyMode) }
      if (preset?.qualityMode) setQualityMode(preset.qualityMode)
      if (preset?.genSearch !== undefined) setGenSearch(preset.genSearch)
    })
    window.api.listModels().then(setModels, (e: Error) => setMsg(e.message))
  }, [])

  const toggle = (id: string) => setPicked((p) => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const start = async () => {
    if (!workload) return
    setMsg(null)
    const r = await window.api.startBench({
      workload, modelIds: [...picked], runQuality: quality, heavyMode: heavy, genSearch, qualityMode, ...(bothBackends ? { compareBackends } : {}), ...(reqCtx ? { requiredContext: reqCtx } : {}), ...(minDec.trim() ? { minDecodeTps: Number(minDec) } : {}), ...(maxCtx ? { ladder: LADDER.filter((c) => c <= maxCtx) } : {})
    })
    if (!r.ok) setMsg(r.error)
  }
  const cancel = async () => { const r = await window.api.cancelBench(); if (!r.ok) setMsg(r.error ?? 'cancel failed') }
  const pause = async () => { const r = await window.api.pauseBench(); if (!r.ok) setMsg(r.error ?? 'pause failed') }
  useEffect(() => {
    if (workload) window.api.modelFit(workload).then((f) => { setFit(f.reasons); setSuggestions(f.suggestions ?? []); setVram({ inUse: f.vramInUseBytes, total: f.vramTotalBytes }) }, () => { setFit({}); setSuggestions([]) })
  }, [workload, models])
  const GiB = 1024 ** 3
  const busyGpu = vram.inUse != null && vram.inUse > 1.5 * GiB
  const freeTip = vram.inUse != null && vram.total != null ? `${((vram.total - vram.inUse) / GiB).toFixed(1)} GB free of ${(vram.total / GiB).toFixed(1)} GB` : undefined
  const heavyLabel = busyGpu ? `needs heavy mode while ${(vram.inUse! / GiB).toFixed(1)} GB VRAM is in use by other apps` : 'needs heavy mode'
  // configId = `${modelId}|ngl=<all|n>|…` (documented, deterministic): a numeric ngl > 0 is partial offload.
  const partial = /\|ngl=([1-9]\d*)\|/.test(live.configId ?? '')
  const t = live.telemetry
  const running = live.status === 'running'

  return (
    <section>
      <header className="bar">
        <h1>Benchmark</h1>
        <label>Workload{' '}
          <select value={workload ?? ''} onChange={(e) => pickWorkload(e.target.value as WorkloadId)} disabled={running}>
            {workloads.map((w) => <option key={w.id} value={w.id}>{w.label}</option>)}
          </select>
        </label>
        <label>Max ctx{' '}
          <select value={maxCtx} onChange={(e) => setMaxCtx(Number(e.target.value))} disabled={running}>
            <option value={0}>auto</option>
            {LADDER.map((c) => <option key={c} value={c}>{fmtCtx(c)}</option>)}
          </select>
        </label>
        <label>Required context{' '}
          <select value={reqCtx} disabled={running} onChange={(e) => { const v = Number(e.target.value); setReqCtx(v); void window.api.setRequiredContext(v || null) }}>
            <option value={0}>Auto (workload default)</option>
            {[32768, 65536, 131072].map((c) => <option key={c} value={c}>{fmtCtx(c)}</option>)}
          </select>
        </label>
        <label title="20–30 t/s is usable for large-scale work; blank = the workload's own gate">Min decode t/s{' '}
          <input type="number" min={0} max={1000} step={1} value={minDec} placeholder="default" disabled={running} style={{ width: 70 }} onChange={(e) => setMinDec(e.target.value)} />
        </label>
        <label><input type="checkbox" checked={quality} onChange={(e) => setQuality(e.target.checked)} disabled={running} /> quality suite</label>
        <label title="Models whose full GPU offload does not fit get a partial-offload ladder (slow, flagged degraded)">
          <input type="checkbox" checked={heavy} onChange={(e) => setHeavy(e.target.checked)} disabled={running} /> Include heavy models (partial GPU offload, degraded speed)
        </label>
        {bothBackends && (
          <label title="Plan every config on both llama.cpp builds (same release): the usable VRAM and speed differ per backend on AMD">
            <input type="checkbox" checked={compareBackends} onChange={(e) => setCompareBackends(e.target.checked)} disabled={running} /> Compare backends (Vulkan / HIP)
          </label>
        )}
        <label title="For thinking-capable models: also run the quality suite with thinking on (and each reasoning-effort level the template offers), using the model card's sampling when known">
          <input type="checkbox" checked={genSearch} onChange={(e) => setGenSearch(e.target.checked)} disabled={running || !quality} /> Search generation settings (thinking / effort / temperature)
        </label>
        <label title="Thorough (default): 60-item v2 suite, 3 seeded samples per test for sampled (T > 0) settings. Quick: 17-item v1 suite, 1 sample">Quality{' '}
          <select value={qualityMode} onChange={(e) => setQualityMode(e.target.value as 'thorough' | 'quick')} disabled={running || !quality}>
            <option value="thorough">thorough / qb-2.0.0 (≈3.5× longer than quick)</option><option value="quick">quick = 17-item v1</option>
          </select>
        </label>
        <button onClick={() => void start()} disabled={running || !picked.size}>Start</button>
        <button onClick={() => void pause()} disabled={!running} title="Stops after the current step; resume from Results">Pause</button>
        <button onClick={() => void cancel()} disabled={!running}>Cancel</button>
      </header>
      {msg && <p className="err">{msg}</p>}
      {busyGpu && vram.total != null && (
        <p className="warnline">GPU currently has {((vram.total - vram.inUse!) / GiB).toFixed(1)} GB free of {(vram.total / GiB).toFixed(1)} GB (other apps in use) — results will be affected.</p>
      )}
      <div className="bar actions"><button onClick={onDownload}>Download from Hugging Face</button></div>
      <Suggestions items={suggestions} disabled={running} onIncluded={(f) => {
        setSuggestions((xs) => xs.filter((x) => !f.toLowerCase().endsWith((x.sibling.path.split('/').pop() ?? '').toLowerCase())))
        window.api.listModels().then(setModels, () => {})
        setPicked((p) => new Set([...p, f]))
      }} />

      <table>
        <thead><tr><th /><th>Model</th><th>Source</th><th>Params</th><th>Quant</th><th>Ctx (train)</th><th>Size</th></tr></thead>
        <tbody>
          {models?.map((m) => (
            <tr key={m.id} className="click" onClick={() => !running && !m.meta?.incomplete && toggle(m.id)}>
              <td><input type="checkbox" checked={picked.has(m.id)} readOnly disabled={running} /></td>
              <td>{m.name}{m.meta?.genKnobs.supportsThinking && (
                  <span className="pill" title={`template kwargs: ${m.meta.templateKwNames.join(', ') || '—'}${m.meta.genKnobs.recommended ? ` · model card: ${JSON.stringify(m.meta.genKnobs.recommended)}` : ''}`}>
                    thinking{m.meta.genKnobs.effortValues ? `: ${m.meta.genKnobs.effortValues.join('/')}` : ''}{genSearch ? ' — searched' : ''}
                  </span>
                )}{fit[m.id] && <span className="pill warn-pill" title={[freeTip, fit[m.id]].filter(Boolean).join(' — ')}>{/incomplete/.test(fit[m.id]!) ? 'incomplete download' : /does not fit/.test(fit[m.id]!) ? heavyLabel : /runtime not installed/.test(fit[m.id]!) ? 'no runtime' : 'no config'}</span>}</td>
              <td className="muted">{m.runtime === 'llamacpp' ? 'folder' : m.runtime === 'lmstudio' ? 'LM Studio' : 'Ollama'}</td>
              <td>{v(m.meta?.parameterCount.value, (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(0)}M`))}</td>
              <td>{m.meta?.quantName ?? '—'}</td>
              <td>{v(m.meta?.contextLength, fmtCtx)}
                {reqCtx > 0 && m.meta?.contextLength != null && reqCtx > m.meta.contextLength && <span className="pill warn-pill">model declares {fmtCtx(m.meta.contextLength)}</span>}</td>
              <td>{gib(m.sizeBytes)}</td>
            </tr>
          ))}
          {models?.length === 0 && <tr><td colSpan={7} className="muted">No .gguf models found.</td></tr>}
        </tbody>
      </table>

      <h2>Live {live.sessionId && <span className="muted">session {live.sessionId} — {live.status}</span>}</h2>
      {live.error && <p className="err">{live.error}</p>}
      <div className="tiles">
        <div className="tile"><span className="muted">Model / config</span>{live.model ?? '—'}<span className="muted">{live.configId ?? ''}</span>{/\|hip$/.test(live.configId ?? '') && <span className="pill">ROCm (HIP)</span>}{partial && <span className="pill warn-pill">partial offload — degraded</span>}</div>
        <div className="tile"><span className="muted">Phase / ctx</span>{live.phase ?? '—'} {live.ctx != null && `@ ${fmtCtx(live.ctx)}`}</div>
        <div className="tile"><span className="muted">Progress</span>step {live.stepsDone}/{live.ctxSteps.length || '—'}{live.ctxSteps.length > 0 && ` (ladder up to ${fmtCtx(Math.max(...live.ctxSteps))})`}, config {live.candidatesDone}/{live.candidatesTotal ?? '—'}</div>
      </div>
      <div className="tiles">
        <div className="tile"><span className="muted">GPU util</span>{v(t?.gpuUtilPct, (n) => `${n.toFixed(0)}%`)}</div>
        <div className="tile"><span className="muted">VRAM ded / shared (proc)</span>{v(t?.procVramDedicatedBytes, gib)} / {v(t?.procVramSharedBytes, gib)}</div>
        <div className="tile"><span className="muted">CPU</span>{v(t?.cpuPct, (n) => `${n.toFixed(0)}%`)}</div>
        <div className="tile"><span className="muted">RAM available</span>{v(t?.ramAvailBytes, gib)}</div>
        <div className="tile"><span className="muted">Prefill / decode</span>{v(live.rate?.prefillTps, (n) => num(n, 0))} / {v(live.rate?.decodeTps, (n) => num(n))} t/s</div>
        <div className="tile"><span className="muted">TTFT</span>{v(live.rate?.ttftMs, (n) => `${num(n, 0)} ms`)}</div>
      </div>
      <pre className="log">{live.log.length ? live.log.join('\n') : 'No events yet.'}</pre>
    </section>
  )
}
