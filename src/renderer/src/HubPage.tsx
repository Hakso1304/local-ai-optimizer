import { useEffect, useRef, useState } from 'react'
import type { HfGgufFile, HfModel, HubAccount, HubApi, HubProgress } from '../../shared/hub-types'
import { budget, fitModels, type Fit, type FitModel } from '../../core/hub/fit'

// Hub methods are spread into window.api by the preload (src/preload/hub.ts); listModels refreshes the Models list.
const api = () => window.api as unknown as HubApi & { listModels?: () => Promise<unknown> }

const gib = (b: number) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} GiB` : `${(b / 1024 ** 2).toFixed(0)} MiB`)
/** Immutable identity of one transfer: Resume and progress always refer to this, never to the repo being browsed. */
interface Transfer { readonly repoId: string; readonly path: string; readonly destDir: string }
const same = (a: Transfer | null, repoId: string, path: string) => !!a && a.repoId === repoId && a.path === path

const FIT_LABEL: Record<Fit, string> = { gpu: 'full GPU', offload: 'partial offload', cpu: 'CPU / shared RAM' }

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
  const [fits, setFits] = useState<{ models: FitModel[]; vram: number; ram: number } | { error: string } | null>(null)

  useEffect(() => {
    void Promise.all([window.api.scanSystem(), api().hubPopular()]).then(([p, r]) => {
      setFits(r.ok ? { models: fitModels(r.models, p).slice(0, 15), ...budget(p) } : { error: r.error })
    }, (e: Error) => setFits({ error: e.message }))
    void api().hubWhoami().then(setAccount)
    void api().hubDirs().then((d) => { setDirs(d); setDest((x) => x || d[d.length - 1] || '') })
    return api().onHubProgress(setProgress)
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
        <h2>Recommended for this PC <span className="muted">(estimated, Q4_K_M)</span></h2>
        {!fits ? <p className="muted">Scanning system and loading popular models…</p> : 'error' in fits ? <p className="muted">Unavailable: {fits.error}</p> : (
          <>
            <p className="muted">Budget: {fits.vram ? `${gib(fits.vram)} VRAM + ` : ''}{gib(fits.ram)} RAM. Sizes are guessed from the repo name; benchmark to confirm.</p>
            <table>
              <thead><tr><th>Repository</th><th>Params</th><th>Est. size</th><th>Fit</th><th>Downloads</th><th /></tr></thead>
              <tbody>
                {fits.models.map((m) => (
                  <tr key={m.id} className={m.id === repo ? 'active' : ''}>
                    <td>{m.id}{m.gated ? <span className="pill warn-pill">gated</span> : null}</td>
                    <td>{m.paramsB}B</td><td>~{gib(m.estBytes)}</td>
                    <td><span className={`pill${m.fit === 'gpu' ? '' : ' warn-pill'}`}>{FIT_LABEL[m.fit]}</span></td>
                    <td>{m.downloads.toLocaleString()}</td>
                    <td><button className="mini" onClick={() => void open(m.id)}>Files</button></td>
                  </tr>
                ))}
                {!fits.models.length && <tr><td colSpan={6} className="muted">No popular model fits the scanned memory.</td></tr>}
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
