import { useEffect, useMemo, useRef, useState } from 'react'
import type { SystemProfile } from '../../shared/types'
import type { WorkloadId } from '../../shared/bench-types'
import type { HfGgufFile, HfModel, HubAccount, HubApi, HubProgress } from '../../shared/hub-types'
import { recommend, type Fit } from '../../core/hub/fit'
import { PRISM_NOTE, rowSummary, seriesGroups } from './hubView'
import { WORKLOADS } from '../../core/scoring/workloads'
import type { HfGgufFile as GgufFileRow } from '../../shared/hub-types'

// Hub methods are spread into window.api by the preload (src/preload/hub.ts); listModels refreshes the Models list.
const api = () => window.api as unknown as HubApi & { listModels?: () => Promise<unknown> }

const gib = (b: number) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} GiB` : `${(b / 1024 ** 2).toFixed(0)} MiB`)
/** Immutable identity of one transfer: Resume and progress always refer to this, never to the repo being browsed. */
interface Transfer { readonly repoId: string; readonly path: string; readonly destDir: string }
const same = (a: Transfer | null, repoId: string, path: string) => !!a && a.repoId === repoId && a.path === path

const FIT_LABEL: Record<Fit, string> = { gpu: 'full GPU', shared: 'GPU (shared RAM)', offload: 'partial offload', cpu: 'CPU only' }
/** Repo file lists fetched for the recommendation, kept across page visits (one HF request per repo). */
const fileCache = new Map<string, GgufFileRow[]>()
const FILES_TO_FETCH = 40

const eta = (s: number | null) => (s === null ? '—' : s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`)

export function HubPage() {
  const [account, setAccount] = useState<HubAccount | null>(null)
  const [token, setToken] = useState('')
  const [query, setQuery] = useState('')
  const [models, setModels] = useState<HfModel[] | null>(null)
  const [repo, setRepo] = useState<string | null>(null)
  const [files, setFiles] = useState<HfGgufFile[] | null>(null)
  const [dirs, setDirs] = useState<string[]>([])
  const [dest, setDest] = useState('')
  const [target, setTarget] = useState<Transfer | null>(null)
  const opened = useRef<string | null>(null)
  const [progress, setProgress] = useState<HubProgress | null>(null)
  const [state, setState] = useState<'idle' | 'downloading' | 'paused' | 'done'>('idle')
  const [err, setErr] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  // Recommendation inputs: the scan + popular repos once, file lists as they arrive (fetched = progress tick), and the
  // use case chosen here (defaults to the Benchmark page's workload); the list is derived from them.
  const [raw, setRaw] = useState<{ models: HfModel[]; profile: SystemProfile; requiredContext: number | null; total: number } | { error: string } | null>(null)
  const [fetched, setFetched] = useState(0)
  const [useCase, setUseCase] = useState<WorkloadId>('general_chat')
  const fits = useMemo(() => {
    if (!raw || 'error' in raw) return raw
    const workload = WORKLOADS[useCase]
    const ctx = raw.requiredContext ?? workload.targetContext
    const out = recommend(raw.models, fileCache, raw.profile, workload, ctx)
    return { models: out.models.slice(0, 20), gpu: out.budget.gpu, shared: out.budget.shared, ram: out.budget.ram, ctx, workload: workload.label, fetched, total: raw.total }
  }, [raw, fetched, useCase])

  useEffect(() => {
    let alive = true
    void Promise.all([window.api.scanSystem(), api().hubPopular(), window.api.getSettings()]).then(async ([p, r, s]) => {
      if (!r.ok) { setRaw({ error: r.error }); return }
      if (s.workload) setUseCase(s.workload)
      // Name-based list first, then real file sizes for the most-downloaded repos as their file lists arrive.
      const targets = [...r.models].sort((a, b) => b.downloads - a.downloads).slice(0, FILES_TO_FETCH)
      setRaw({ models: r.models, profile: p, requiredContext: s.requiredContext ?? null, total: targets.length })
      setFetched(targets.filter((t) => fileCache.has(t.id)).length)
      for (let i = 0; i < targets.length && alive; i += 5) {
        await Promise.all(targets.slice(i, i + 5).filter((t) => !fileCache.has(t.id)).map(async (t) => {
          const fr = await api().hubFiles(t.id).catch(() => null)
          if (fr?.ok) fileCache.set(t.id, fr.files)
        }))
        if (alive) setFetched(Math.min(i + 5, targets.length))
      }
    }, (e: Error) => setRaw({ error: e.message }))
    void api().hubWhoami().then(setAccount)
    void api().hubDirs().then((d) => { setDirs(d); setDest((x) => x || d[0] || '') }) // [0] = the model store
    const off = api().onHubProgress(setProgress)
    return () => { alive = false; off() }
  }, [])

  const login = async () => {
    setErr(null)
    const r = await api().hubLogin(token)
    if (r.ok) { setAccount({ signedIn: true, name: r.name }); setToken('') } else setErr(r.error)
  }
  const logout = async () => { await api().hubLogout(); setAccount({ signedIn: false, name: null }) }
  const search = async () => {
    setErr(null); setRepo(null); setFiles(null)
    const r = await api().hubSearch(query)
    if (r.ok) setModels(r.models); else setErr(r.error)
  }
  const open = async (id: string) => {
    setErr(null); setRepo(id); setFiles(null)
    opened.current = id
    const r = await api().hubFiles(id)
    if (opened.current !== id) return // a later Files click superseded this reply
    if (r.ok) setFiles(r.files); else setErr(r.error)
  }
  const download = async (t: Transfer) => {
    setErr(null); setNote(null); setTarget(t); setState('downloading')
    const r = await api().hubDownload({ repoId: t.repoId, path: t.path, destDir: t.destDir })
    if (r.ok) {
      setState('done')
      setNote(`Model ready: ${r.filePath}${r.sha256Verified ? ' (sha256 verified)' : ''}`)
      void api().listModels?.()
    } else if (r.kind === 'cancelled') {
      setState((s) => (s === 'idle' ? 'idle' : 'paused'))
    } else {
      setState('paused') // any failure can be resumed from the .part
      setErr(r.error)
    }
  }
  const pause = async () => { setState('paused'); await api().hubCancel(false) }
  const cancel = async () => { setState('idle'); setProgress(null); await api().hubCancel(true) }

  const p = progress && same(target, progress.repoId, progress.path) ? progress : null
  return (
    <section>
      <header className="bar"><h1>Download models</h1></header>

      <div className="card">
        <h2>Hugging Face account</h2>
        {account?.signedIn ? (
          <div className="bar"><span>Signed in as <b>{account.name}</b></span><button onClick={() => void logout()}>Sign out</button></div>
        ) : (
          <div className="bar">
            <button onClick={() => void api().hubOpenTokenPage()}>Open token page</button>
            <input type="password" placeholder="hf_… (read token)" value={token} onChange={(e) => setToken(e.target.value)} style={{ width: 280 }} />
            <button disabled={!token.trim()} onClick={() => void login()}>Save</button>
            <span className="muted">Needed for gated models. Stored encrypted on this PC.</span>
          </div>
        )}
        {account?.error && <p className="muted">Saved token not valid: {account.error}</p>}
      </div>

      <div className="card">
        <div className="bar">
          <h2>Recommended for this PC <span className="muted">(estimated)</span></h2>
          <label>Use case <select value={useCase} onChange={(e) => setUseCase(e.target.value as WorkloadId)} title="Same use cases as the Benchmark page: context length, speed gate and coding boost follow it">
            {Object.values(WORKLOADS).map((w) => <option key={w.id} value={w.id}>{w.label}</option>)}
          </select></label>
        </div>
        {!fits ? <p className="muted">Scanning system and loading popular models…</p> : 'error' in fits ? <p className="muted">Unavailable: {fits.error}</p> : (
          <>
            <p className="muted">
              {fits.shared ? `Integrated GPU: ${gib(fits.gpu)} shared RAM` : fits.gpu ? `${gib(fits.gpu)} VRAM free for the model + ${gib(fits.ram)} RAM` : `no GPU: ${gib(fits.ram)} RAM`} · {fits.workload} at {fits.ctx.toLocaleString()} context (KV cache included) · same rules as the benchmark planner.
              {fits.fetched < fits.total ? ` Reading repo file lists ${fits.fetched}/${fits.total}…` : ' Sizes are the repos\' real files; speed is a bandwidth estimate. Benchmark to confirm.'}
            </p>
            <table>
              <thead><tr><th>Model</th><th>Released</th><th>File</th><th>Fit</th><th>Est. decode</th><th>Source</th><th>Downloads</th><th /></tr></thead>
              <tbody>
                {seriesGroups(fits.models).map((g) => [
                  <tr key={`s:${g.series.key}`} className="group"><td colSpan={8} title={g.series.profile}>
                    <b>{g.series.label}</b>{g.series.vendor && <span className="muted"> · {g.series.vendor}</span>}<span className="muted"> · {g.models.length} model{g.models.length > 1 ? 's' : ''} · hover for the series profile</span>
                  </td></tr>,
                  ...g.models.map((m) => (
                  <tr key={m.id} className={m.id === repo ? 'active' : ''} title={rowSummary(m)}>
                    <td>{m.id}{m.gated ? <span className="pill warn-pill">gated</span> : null}{m.file?.prism && <span className="pill warn-pill" title={PRISM_NOTE}>PrismML</span>}
                      {m.alsoIn.length > 0 && <span className="muted" title={m.alsoIn.join('\n')}> +{m.alsoIn.length} more repo{m.alsoIn.length > 1 ? 's' : ''}</span>}</td>
                    <td className="muted">{m.releasedAt ?? '—'}</td>
                    <td title={m.file?.path ?? 'estimated from the name (Q4_K_M)'}>{m.file ? `${m.file.quant}${m.file.shards > 1 ? ` ×${m.file.shards}` : ''} ${gib(m.weightsBytes)}` : `~${gib(m.weightsBytes)} (est.)`}<span className="muted"> + KV {gib(m.kvBytes)}</span></td>
                    <td><span className={`pill${m.fit === 'gpu' || m.fit === 'shared' ? '' : ' warn-pill'}`} title={m.fit === 'offload' ? `${Math.round(m.gpuShare * 100)} % of the weights on the GPU` : undefined}>{FIT_LABEL[m.fit]}</span></td>
                    <td title={`active ${m.activeB}B of ${m.paramsB}B per token`}>≈{m.estTps >= 10 ? Math.round(m.estTps) : m.estTps.toFixed(1)} t/s{!m.usable && <span className="pill warn-pill" title="below the workload's decode gate">slow</span>}</td>
                    <td><span className={`pill${m.trust === 'official' ? '' : ' warn-pill'}`}>{m.trust}</span></td>
                    <td>{m.downloads.toLocaleString()}</td>
                    <td><button className="mini" onClick={() => void open(m.id)}>Files</button></td>
                  </tr>
                  ))
                ])}
                {!fits.models.length && <tr><td colSpan={8} className="muted">No popular model fits the scanned memory.</td></tr>}
              </tbody>
            </table>
          </>
        )}
      </div>

      <div className="card">
        <h2>Search GGUF models</h2>
        <div className="bar">
          <input value={query} placeholder="e.g. llama 3.1 8b instruct" onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') void search() }} style={{ width: 360 }} />
          <button disabled={!query.trim()} onClick={() => void search()}>Search</button>
        </div>
        {models && (
          <table>
            <thead><tr><th>Repository</th><th>Downloads</th><th>Likes</th><th /></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} className={m.id === repo ? 'active' : ''}>
                  <td>{m.id}{m.gated ? <span className="pill warn-pill">gated</span> : null}</td>
                  <td>{m.downloads.toLocaleString()}</td><td>{m.likes.toLocaleString()}</td>
                  <td><button className="mini" onClick={() => void open(m.id)}>Files</button></td>
                </tr>
              ))}
              {!models.length && <tr><td colSpan={4} className="muted">No GGUF repositories found.</td></tr>}
            </tbody>
          </table>
        )}
      </div>

      {repo && (
        <div className="card">
          <h2>{repo}</h2>
          <div className="bar">
            <span className="muted">Save to</span>
            <select value={dest} onChange={(e) => setDest(e.target.value)} disabled={state === 'downloading'}>
              {dirs.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
            <button className="mini" disabled={state === 'downloading'} title="Add a folder on another drive; it is scanned for models and offered as a download target"
              onClick={() => void window.api.addModelDir().then(() => api().hubDirs()).then((d) => { setDirs(d); setDest(d[d.length - 1] ?? '') })}>Add folder…</button>
            {dest && dest !== dirs[0] && <button className="mini" disabled={state === 'downloading'} title="Forget this folder (files are not deleted)"
              onClick={() => void window.api.removeModelDir(dest).then(() => api().hubDirs()).then((d) => { setDirs(d); setDest(d[d.length - 1] ?? '') })}>Remove</button>}
          </div>
          {!files ? <p className="muted">Loading files…</p> : (
            <table>
              <thead><tr><th>File</th><th>Quant</th><th>Size</th><th /></tr></thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.path}>
                    <td>{f.path}{f.shard ? <span className="muted"> (part {f.shard.index}/{f.shard.count})</span> : null}</td>
                    <td>{f.quant ?? '—'}</td><td>{gib(f.sizeBytes)}</td>
                    <td><button className="mini" disabled={state === 'downloading' || !dest} onClick={() => void (state === 'paused' && same(target, repo, f.path) && target!.destDir === dest ? download(target!) : download({ repoId: repo, path: f.path, destDir: dest }))}>
                      {state === 'paused' && same(target, repo, f.path) && target!.destDir === dest ? 'Resume' : 'Download'}</button></td>
                  </tr>
                ))}
                {!files.length && <tr><td colSpan={4} className="muted">This repository has no .gguf files.</td></tr>}
              </tbody>
            </table>
          )}
        </div>
      )}

      {target && state !== 'idle' && (
        <div className="card">
          <h2>{target.repoId} / {target.path}</h2>
          <div className="scorebar" style={{ gridTemplateColumns: '1fr auto' }}>
            <span className="track"><span className="fill" style={{ width: `${p?.pct ?? (state === 'done' ? 100 : 0)}%` }} /></span>
            <span className="val">{p?.pct != null ? `${p.pct.toFixed(1)} %` : ''}</span>
          </div>
          <p className="muted">
            {p ? `${gib(p.bytes)} / ${p.total ? gib(p.total) : '?'} · ${(p.bytesPerSec / 1024 ** 2).toFixed(1)} MiB/s · ETA ${eta(p.etaSec)}` : state === 'paused' ? 'Paused' : ''}
          </p>
          <div className="bar">
            {state === 'downloading' && <button onClick={() => void pause()}>Pause</button>}
            {state === 'paused' && <button onClick={() => void download(target)}>Resume</button>}
            {(state === 'downloading' || state === 'paused') && <button onClick={() => void cancel()}>Cancel</button>}
          </div>
        </div>
      )}

      {err && <p className="err">{err}</p>}
      {note && <p>{note}</p>}
    </section>
  )
}
