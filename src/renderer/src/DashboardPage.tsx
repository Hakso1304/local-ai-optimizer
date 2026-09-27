import { useEffect, useState } from 'react'
import type { WorkloadId, WorkloadProfile } from '../../shared/bench-types'
import type { SessionDetail, SystemProfile } from '../../shared/types'
import { COMPONENT_LABEL, CtxPick, DemoBanner, M, ScoreBar, fmtCtx, gib } from './ui'

type Go = (section: 'Benchmark' | 'Results', sessionId?: number) => void

function SystemSummary() {
  const [p, setP] = useState<SystemProfile | null>(null)
  useEffect(() => { window.api.scanSystem().then(setP, () => {}) }, [])
  if (!p) return <p className="muted">Scanning system…</p>
  const gpu = p.gpus.value?.find((g) => !g.isIntegrated)
  const llama = p.runtimes.find((r) => r.id === 'llamacpp')
  return (
    <div className="tiles">
      <div className="tile"><span className="muted">GPU</span>{gpu?.name ?? '—'}</div>
      <div className="tile"><span className="muted">VRAM</span>{gpu?.dedicatedVramBytes.value != null ? gib(gpu.dedicatedVramBytes.value) : '—'}</div>
      <div className="tile"><span className="muted">RAM</span>{p.ram.value ? `${gib(p.ram.value.availableBytes)} free / ${gib(p.ram.value.totalBytes)}` : '—'}</div>
      <div className="tile"><span className="muted">CPU</span>{p.cpu.value ? `${p.cpu.value.model} (${p.cpu.value.physicalCores}C/${p.cpu.value.logicalCores}T)` : '—'}</div>
      <div className="tile"><span className="muted">Runtime</span>{llama?.status === 'available' ? `llama.cpp ${llama.version ?? ''}` : 'llama.cpp unavailable'}</div>
    </div>
  )
}

function RecommendedCard({ d, go }: { d: SessionDetail; go: Go }) {
  const rec = d.recommendation
  const c = d.candidates.find((x) => x.config.id === rec?.best?.configId)
  return (
    <div className={`card${d.session.demo ? ' demo-card' : ''}`}>
      {d.session.demo && <DemoBanner what={`DEMO DATA (workload ${d.session.workload})`} />}
      <h2>Recommended configuration</h2>
      {!rec?.best || !c ? (
        <><p>No candidate met this workload's requirements.</p><ul>{rec?.reasons.map((r) => <li key={r}>{r}</li>)}</ul></>
      ) : (
        <>
          <table className="kv">
            <tbody>
              <tr><td>Model</td><td>{c.model.name}</td></tr>
              <tr><td>Quant</td><td>{c.model.quant ?? '—'}</td></tr>
              <tr><td>Context</td><td><CtxPick recommended={rec.best.score.recommendedCtx} scored={rec.best.score.referenceCtx} /></td></tr>
              {d.session.requiredContext != null && <tr><td>Required context</td><td>{fmtCtx(d.session.requiredContext)} {(rec.best.practicalContext.value ?? 0) >= d.session.requiredContext ? <span className="pill">met</span> : <span className="pill warn-pill">not met</span>}</td></tr>}
              <tr><td>Practical ceiling</td><td><M m={rec.best.practicalContext} fmt={fmtCtx} /> <span className="muted">(model declares <M m={rec.best.declaredContext} fmt={fmtCtx} />)</span></td></tr>
              <tr><td>Backend</td><td>{c.config.device ? `llama.cpp Vulkan (${c.config.device})` : 'llama.cpp CPU'}</td></tr>
              <tr><td>GPU layers</td><td>{c.config.gpuLayersAll ? `all (${c.model.layers})` : c.config.gpuLayers}</td></tr>
              <tr><td>Threads</td><td>{c.config.threads}</td></tr>
              <tr><td>Score</td><td>{rec.best.score.total.toFixed(1)} / 100</td></tr>
            </tbody>
          </table>
          {rec.best.score.breakdown.map((b) => (
            <ScoreBar key={b.component} label={COMPONENT_LABEL[b.component]} score={b.score} kind={b.input.kind} weight={b.weight} />
          ))}
        </>
      )}
      <div className="bar actions">
        <button onClick={() => go('Benchmark')}>Run Benchmark</button>
        <button onClick={() => go('Results', d.session.id)}>View Details</button>
      </div>
    </div>
  )
}

export function DashboardPage({ go }: { go: Go }) {
  const [workloads, setWorkloads] = useState<WorkloadProfile[]>([])
  const [workload, setWorkload] = useState<WorkloadId | null>(null)
  const [real, setReal] = useState<SessionDetail | null | undefined>(undefined)
  const [demo, setDemo] = useState<SessionDetail | null>(null)

  useEffect(() => {
    void Promise.all([window.api.listWorkloads(), window.api.getSettings()]).then(([ws, s]) => {
      setWorkloads(ws)
      setWorkload(s.workload ?? ws[0]?.id ?? null)
    })
  }, [])
  useEffect(() => {
    if (!workload) return
    setReal(undefined)
    void window.api.latestRecommendation(workload).then(async (r) => setReal(r ? await window.api.getSession(r.sessionId) : null))
    // Demo is shown only as a separate, flagged preview when there is no real result.
    void window.api.listSessions().then(async (ss) => { const d = ss.find((s) => s.demo); setDemo(d ? await window.api.getSession(d.id) : null) })
  }, [workload])

  const pick = (w: WorkloadId) => { setWorkload(w); void window.api.setWorkload(w) }

  return (
    <section>
      <header className="bar">
        <h1>Dashboard</h1>
        <label>Workload{' '}
          <select value={workload ?? ''} onChange={(e) => pick(e.target.value as WorkloadId)}>
            {workloads.map((w) => <option key={w.id} value={w.id}>{w.label}</option>)}
          </select>
        </label>
      </header>
      <SystemSummary />
      {real === undefined ? <p className="muted">Loading…</p> : real ? <RecommendedCard d={real} go={go} /> : (
        <>
          <div className="card">
            <h2>Recommended configuration</h2>
            <p>No benchmark for this workload yet.</p>
            <div className="bar actions"><button onClick={() => go('Benchmark')}>Run Benchmark</button></div>
          </div>
          {demo && <RecommendedCard d={demo} go={go} />}
        </>
      )}
    </section>
  )
}
