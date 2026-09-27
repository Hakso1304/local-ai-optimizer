import { useEffect, useReducer, useState } from 'react'
import type { WorkloadId, WorkloadProfile } from '../../shared/bench-types'
import type { ModelInfo } from '../../shared/types'
import { applyEvent, initialLive } from './benchState'
import { fmtCtx, gib, num } from './ui'

const v = (x: number | null | undefined, f: (n: number) => string) => (x == null ? '—' : f(x))

export function BenchmarkPage() {
  const [workloads, setWorkloads] = useState<WorkloadProfile[]>([])
  const [workload, setWorkload] = useState<WorkloadId | null>(null)
  const [models, setModels] = useState<ModelInfo[] | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [msg, setMsg] = useState<string | null>(null)
  const [live, dispatch] = useReducer(applyEvent, initialLive)

  useEffect(() => {
    void Promise.all([window.api.listWorkloads(), window.api.getSettings()]).then(([ws, s]) => { setWorkloads(ws); setWorkload(s.workload ?? ws[0]?.id ?? null) })
    window.api.listModels().then(setModels, (e: Error) => setMsg(e.message))
    return window.api.onBenchEvent(dispatch)
  }, [])

  const toggle = (id: string) => setPicked((p) => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n })
  const start = async () => {
    if (!workload) return
    setMsg(null)
    const r = await window.api.startBench({ workload, modelIds: [...picked] })
    if (!r.ok) setMsg(r.error)
  }
  const cancel = async () => { const r = await window.api.cancelBench(); if (!r.ok) setMsg(r.error ?? 'cancel failed') }
  const t = live.telemetry
  const running = live.status === 'running'

  return (
    <section>
      <header className="bar">
        <h1>Benchmark</h1>
        <label>Workload{' '}
          <select value={workload ?? ''} onChange={(e) => setWorkload(e.target.value as WorkloadId)} disabled={running}>
            {workloads.map((w) => <option key={w.id} value={w.id}>{w.label}</option>)}
          </select>
        </label>
        <button onClick={() => void start()} disabled={running || !picked.size}>Start</button>
        <button onClick={() => void cancel()} disabled={!running}>Cancel</button>
      </header>
      {msg && <p className="err">{msg}</p>}

      <table>
        <thead><tr><th /><th>Model</th><th>Params</th><th>Quant</th><th>Ctx (train)</th><th>Size</th></tr></thead>
        <tbody>
          {models?.map((m) => (
            <tr key={m.id} className="click" onClick={() => !running && toggle(m.id)}>
              <td><input type="checkbox" checked={picked.has(m.id)} readOnly disabled={running} /></td>
              <td>{m.name}</td>
              <td>{v(m.meta?.parameterCount.value, (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(0)}M`))}</td>
              <td>{m.meta?.quantName ?? '—'}</td>
              <td>{v(m.meta?.contextLength, fmtCtx)}</td>
              <td>{gib(m.sizeBytes)}</td>
            </tr>
          ))}
          {models?.length === 0 && <tr><td colSpan={6} className="muted">No .gguf models found.</td></tr>}
        </tbody>
      </table>

      <h2>Live {live.sessionId && <span className="muted">session {live.sessionId} — {live.status}</span>}</h2>
      {live.error && <p className="err">{live.error}</p>}
      <div className="tiles">
        <div className="tile"><span className="muted">Model / config</span>{live.model ?? '—'}<span className="muted">{live.configId ?? ''}</span></div>
        <div className="tile"><span className="muted">Phase / ctx</span>{live.phase ?? '—'} {live.ctx != null && `@ ${fmtCtx(live.ctx)}`}</div>
        <div className="tile"><span className="muted">Progress</span>step {live.stepsDone}/{live.ctxSteps.length || '—'}, candidate {live.candidatesDone}/{live.candidatesStarted || '—'}</div>
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
