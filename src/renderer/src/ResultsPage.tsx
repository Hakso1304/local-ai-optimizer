import { useEffect, useState } from 'react'
import type { Metric } from '../../shared/bench-types'
import type { SessionCandidate, SessionDetail, SessionSummary } from '../../shared/types'
import { LineChart, type Band } from './LineChart'
import { CtxPick, DemoBanner, M, Prov, fmtCtx, gib, num } from './ui'

const RESUMABLE = new Set(['cancelled', 'failed', 'paused', 'interrupted'])
const COLORS = ['#58a6ff', '#f0883e', '#a371f7', '#3fb950', '#db61a2']
const NA = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })

/** Largest measured value of a run metric across completed steps. */
function peak(c: SessionCandidate, k: 'peakVramBytes' | 'peakRamBytes'): Metric {
  const ms = c.runs.filter((r) => r.status === 'pass' || r.status === 'degraded').map((r) => r[k]).filter((m) => m.value != null)
  return ms.length ? ms.reduce((a, b) => (b.value! > a.value! ? b : a)) : NA('no completed step reported it')
}

const refRun = (c: SessionCandidate) => c.runs.find((r) => r.ctx === c.score?.referenceCtx)
const comp = (c: SessionCandidate, id: string) => c.score?.breakdown.find((b) => b.component === id)

/** llama-server CLI + JSON at the workload's recommendedCtx (falls back to the practical ceiling if unscored). */
function exportText(c: SessionCandidate): string {
  const ctx = c.score?.recommendedCtx ?? c.cliff.practicalContextCeiling.value
  const args = [
    '-m', `"${c.model.id}"`, '-c', String(ctx ?? 4096), '-ngl', c.config.gpuLayersAll ? '99' : String(c.config.gpuLayers),
    ...(c.config.device ? ['--device', c.config.device] : []), '-t', String(c.config.threads),
    '-ctk', c.config.kvType, '-ctv', c.config.kvType, '-fa', c.config.flashAttn ? 'on' : 'off', '-fit', 'off'
  ]
  const json = { configId: c.config.id, model: c.model.id, contextSize: ctx, gpuLayers: c.config.gpuLayers, device: c.config.device, threads: c.config.threads, kvType: c.config.kvType, flashAttn: c.config.flashAttn }
  return `llama-server ${args.join(' ')}\n\n${JSON.stringify(json, null, 2)}\n`
}

function Detail({ d, onRerun }: { d: SessionDetail; onRerun: (configId: string) => void }) {
  const [copied, setCopied] = useState<string | null>(null)
  const rec = d.recommendation
  const xs = [...new Set(d.candidates.flatMap((c) => c.runs.map((r) => r.ctx)))].sort((a, b) => a - b)
  const at = (c: SessionCandidate, f: (r: SessionCandidate['runs'][number]) => number | null) => xs.map((x) => { const r = c.runs.find((q) => q.ctx === x); return r ? f(r) : null })
  const bands: Band[] = d.candidates.flatMap((c, i) => {
    const s = c.cliff.steps.find((st) => st.verdict !== 'pass')
    const from = s?.reasons.find((r) => r.fromCtx != null)?.fromCtx
    return s && from != null ? [{ from, to: s.ctx, color: COLORS[i % COLORS.length], label: `${s.verdict} ${fmtCtx(s.ctx)}` }] : []
  })
  const copy = async (c: SessionCandidate) => {
    await navigator.clipboard.writeText(exportText(c))
    setCopied(c.config.id)
  }
  // Same model can appear with several configs, so always show the configId too.
  const name = (id: string | null) => (id ? `${d.candidates.find((c) => c.config.id === id)?.model.name ?? '?'} — ${id}` : '—')
  const tag = d.session.demo ? ' (DEMO DATA)' : ''

  return (
    <>
      {d.session.demo && <DemoBanner />}
      <h2>Comparison — {d.session.workload}{tag}</h2>
      <table>
        <thead><tr><th /><th>Model</th><th>Quant</th><th>Config</th><th>Recommended ctx</th><th>Practical ctx</th><th>Quality</th><th>Decode t/s</th><th>Prefill t/s</th><th>Peak VRAM</th><th>Peak RAM</th><th>Stability</th><th /></tr></thead>
        <tbody>
          {d.candidates.map((c, i) => {
            const q = comp(c, 'quality'), st = comp(c, 'stability'), r = refRun(c)
            return (
              <tr key={c.config.id}>
                <td><span className="swatch" style={{ background: COLORS[i % COLORS.length] }} /></td>
                <td>{c.model.name}{rec?.best?.configId === c.config.id && <span className="pill">best</span>}</td>
                <td>{c.model.quant ?? '—'} <Prov kind="declared" /></td>
                <td className="muted">ngl {c.config.gpuLayersAll ? 'all' : c.config.gpuLayers}, kv {c.config.kvType}, t {c.config.threads}</td>
                <td><CtxPick recommended={c.score?.recommendedCtx} scored={c.score?.referenceCtx} /></td>
                <td><M m={c.cliff.practicalContextCeiling} fmt={fmtCtx} /></td>
                <td>{q ? <M m={{ ...q.input, value: q.score }} fmt={(v) => v.toFixed(0)} /> : '—'}</td>
                <td><M m={r?.decodeTps} /> {r && <span className="muted">@{fmtCtx(r.ctx)}</span>}</td>
                <td><M m={r?.prefillTps} fmt={(v) => num(v, 0)} /></td>
                <td><M m={peak(c, 'peakVramBytes')} fmt={gib} /></td>
                <td><M m={peak(c, 'peakRamBytes')} fmt={gib} /></td>
                <td>{st ? <>{st.score.toFixed(0)} <Prov kind={st.input.kind} /></> : '—'}</td>
                <td className="bar">
                  <button onClick={() => void copy(c)}>{copied === c.config.id ? 'Copied' : 'Export config'}</button>
                  {!d.session.demo && <button onClick={() => onRerun(c.config.id)} title="Re-run every step of this configuration">Rerun</button>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <h2>Context scaling{tag}</h2>
      <div className="charts">
        <div><h3>Prefill t/s</h3><LineChart xs={xs} bands={bands} yLabel="prefill t/s" fmtY={(v) => num(v, 0)}
          series={d.candidates.map((c, i) => ({ label: c.config.id, color: COLORS[i % COLORS.length], ys: at(c, (r) => r.prefillTps.value) }))} /></div>
        <div><h3>Decode t/s</h3><LineChart xs={xs} bands={bands} yLabel="decode t/s" fmtY={(v) => num(v, 0)}
          series={d.candidates.map((c, i) => ({ label: c.config.id, color: COLORS[i % COLORS.length], ys: at(c, (r) => r.decodeTps.value) }))} /></div>
      </div>
      <h2>Memory scaling{tag} <span className="muted">(solid = VRAM, dashed = RAM, per process)</span></h2>
      <LineChart xs={xs} bands={bands} yLabel="GiB" fmtY={(v) => v.toFixed(1)}
        series={d.candidates.flatMap((c, i) => [
          { label: `${c.config.id} VRAM`, color: COLORS[i % COLORS.length], ys: at(c, (r) => (r.peakVramBytes.value == null ? null : r.peakVramBytes.value / 1024 ** 3)) },
          { label: `${c.config.id} RAM`, color: COLORS[i % COLORS.length], dashed: true, ys: at(c, (r) => (r.peakRamBytes.value == null ? null : r.peakRamBytes.value / 1024 ** 3)) }
        ])} />

      <h2>Cliff analysis{tag}</h2>
      {d.candidates.map((c) => (
        <div key={c.config.id} className="cliff">
          <b>{c.config.id}</b> — limited by <b>{c.cliff.limitedBy}</b>; practical <M m={c.cliff.practicalContextCeiling} fmt={fmtCtx} />, degraded <M m={c.cliff.degradedContextCeiling} fmt={fmtCtx} />
          <ul>
            {c.cliff.steps.flatMap((s) => s.reasons.map((r, k) => (
              <li key={`${s.ctx}-${k}`}><span className={`verdict ${s.verdict}`}>{s.verdict}</span> <code>{r.code}</code> {r.message}</li>
            )))}
            {c.cliff.steps.every((s) => !s.reasons.length) && <li className="muted">No cliff reasons: every step passed.</li>}
          </ul>
        </div>
      ))}

      <h2>Recommendation{tag}</h2>
      {rec ? (
        <div className="card">
          <p><b>{rec.best ? name(rec.best.configId) : 'No recommendation'}</b>{rec.best && <> — {rec.best.score.total.toFixed(1)}/100, context <CtxPick recommended={rec.best.score.recommendedCtx} scored={rec.best.score.referenceCtx} /></>}</p>
          <ul>{rec.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          <table>
            <tbody>
              <tr><td>Fastest</td><td>{name(rec.alternatives.fastest)}</td></tr>
              <tr><td>Best quality</td><td>{name(rec.alternatives.bestQuality)}</td></tr>
              <tr><td>Best long context</td><td>{name(rec.alternatives.bestLongContext)}</td></tr>
              <tr><td>Lowest memory</td><td>{name(rec.alternatives.lowestMemory)}</td></tr>
            </tbody>
          </table>
          {rec.excluded.map((e) => <p key={e.configId} className="err">Excluded {e.configId}: {e.reasons.join('; ')}</p>)}
        </div>
      ) : <p className="muted">No recommendation stored for this session.</p>}
    </>
  )
}

export function ResultsPage({ sessionId }: { sessionId?: number }) {
  const [list, setList] = useState<SessionSummary[] | null>(null)
  const [sel, setSel] = useState<number | undefined>(sessionId)
  const [detail, setDetail] = useState<SessionDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => { window.api.listSessions().then(setList, (e: Error) => setErr(e.message)) }, [tick])
  useEffect(() => {
    if (sel == null) return setDetail(null)
    window.api.getSession(sel).then(setDetail, (e: Error) => setErr(e.message))
  }, [sel, tick])
  // Refresh when a session ends or a step lands while this page is open (list status + detail charts).
  useEffect(() => window.api.onBenchEvent((e) => {
    if (e.type === 'step:done' || e.type.startsWith('session:')) setTick((t) => t + 1)
  }), [])
  const act = (p: Promise<{ ok: boolean; error?: string }>) => void p.then((r) => { if (!r.ok) setErr(r.error ?? 'failed'); else setTick((t) => t + 1) })

  return (
    <section>
      <header className="bar"><h1>Results</h1></header>
      {err && <p className="err">{err}</p>}
      <table>
        <thead><tr><th>#</th><th>Created</th><th>Workload</th><th>Status</th><th>Candidates</th><th>Best</th></tr></thead>
        <tbody>
          {list?.map((s) => (
            <tr key={s.id} className={`click${s.id === sel ? ' selected' : ''}`} onClick={() => setSel(s.id)}>
              <td>{s.id}</td><td>{s.createdAt}</td><td>{s.workload}</td>
              <td>
                {s.demo ? <span className="pill demo-pill">DEMO DATA</span> : s.status}
                {s.error && <span className="err" title={s.error}> ⚠</span>}
                {!s.demo && RESUMABLE.has(s.status) && (
                  <button className="mini" onClick={(e) => { e.stopPropagation(); act(window.api.resumeBench(s.id)) }}>Resume</button>
                )}
              </td>
              <td>{s.candidateCount}</td><td className="muted">{s.bestConfigId ?? '—'}</td>
            </tr>
          ))}
          {list?.length === 0 && <tr><td colSpan={6} className="muted">No benchmark sessions yet.</td></tr>}
        </tbody>
      </table>
      {detail && !detail.session.demo && (
        <div className="bar actions">
          <button onClick={() => act(window.api.resumeBench(detail.session.id, { retryFailed: true }))} title="Re-run steps that failed or timed out">Retry failed tests</button>
        </div>
      )}
      {detail && <Detail d={detail} onRerun={(configId) => act(window.api.resumeBench(detail.session.id, { rerunConfigIds: [configId] }))} />}
    </section>
  )
}
