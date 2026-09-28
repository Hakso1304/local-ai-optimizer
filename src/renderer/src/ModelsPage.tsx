import { useEffect, useState } from 'react'
import { exportConfigFrom, type ExportConfig } from '../../core/export/config'
import type { KvType } from '../../shared/bench-types'
import type { ModelInfo, SmokeResult } from '../../shared/types'
import { RunningModel, SERVE_IDLE, useServe } from './ui'

const SOURCE = { llamacpp: 'folder', lmstudio: 'LM Studio', ollama: 'Ollama' } as const
const gib = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
const params = (n: number | null) => (n == null ? '—' : n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(0)}M`)
const fmt = (v: unknown) => (v == null ? '—' : typeof v === 'number' ? String(Math.round(v * 100) / 100) : typeof v === 'object' ? JSON.stringify(v) : String(v))

/** The Run… form. base = the latest recommendation's export config when it picked this model (prefill), else null. */
type RunForm = { m: ModelInfo; ctx: number; all: boolean; ngl: number; kv: KvType; fa: boolean; threads: number; base: ExportConfig | null; note: string }

/** Flatten a result into dotted key/value rows (raw view until the Results page exists). */
function rows(o: object, prefix = ''): [string, unknown][] {
  return Object.entries(o).flatMap(([k, v]): [string, unknown][] =>
    v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length ? rows(v, `${prefix}${k}.`) : [[prefix + k, v]]
  )
}

export function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [smoke, setSmoke] = useState<{ model: string; r: SmokeResult } | null>(null)
  // "Link HF repo": fetch the model card's sampling defaults (generation_config.json) for a local file.
  const [linking, setLinking] = useState<{ id: string; repo: string; msg?: string } | null>(null)
  const link = async (m: ModelInfo) => {
    if (!linking?.repo.trim()) return
    const r = await window.api.linkModelRepo(m.path, linking.repo.trim())
    setLinking({ ...linking, msg: r.ok ? (r.generation ? `model card: ${JSON.stringify(r.generation)}` : 'linked; the repo has no generation_config.json') : r.error })
    if (r.ok) load()
  }

  const load = () => { window.api.listModels().then(setModels, (e: Error) => setErr(e.message)) }
  useEffect(load, [])

  // Run a model from the app: llama-server with the chosen launch config, chat in its built-in web UI.
  const [form, setForm] = useState<RunForm | null>(null)
  const { serve, setServe, stop } = useServe()
  const openRun = async (m: ModelInfo) => {
    let base: ExportConfig | null = null
    try { // prefill from the latest recommendation for the current workload when it picked this model
      const w = (await window.api.getSettings()).workload ?? 'fast_assistant'
      const latest = await window.api.latestRecommendation(w)
      const cid = latest?.recommendation.best?.configId
      if (latest && cid) {
        const cand = (await window.api.getSession(latest.sessionId))?.candidates.find((c) => c.config.id === cid)
        if (cand?.model.id === m.id) base = exportConfigFrom(latest.recommendation, cand.config, cand.model, String(latest.sessionId))
      }
    } catch { /* no usable recommendation: defaults below */ }
    setForm(base
      ? { m, ctx: base.ctx, all: base.gpuLayersAll, ngl: base.gpuLayers, kv: base.kvType, fa: base.flashAttn, threads: base.threads, base, note: `Recommended configuration for ${base.workload} (benchmark ${base.sessionId}). Edit if you like.` }
      : { m, ctx: Math.min(8192, m.meta?.contextLength ?? 8192), all: true, ngl: m.meta?.blockCount ?? 0, kv: 'f16', fa: true, threads: 0, base: null, note: 'No recommendation for this model yet (run a benchmark for one). Defaults: all layers on the GPU, threads = physical cores.' })
  }
  const start = async (f: RunForm) => {
    setServe({ ...SERVE_IDLE, busy: 'starting' })
    const cfg: ExportConfig = {
      ...(f.base ?? { sessionId: '', modelName: f.m.name, layers: f.m.meta?.blockCount ?? 0, batch: 2048, ubatch: 512, device: 'auto', kvOffload: true, mmap: true, backend: 'vulkan', workload: 'fast_assistant' }),
      configId: f.base?.configId ?? `${f.m.id}|manual`, modelPath: f.m.path, ctx: f.ctx, gpuLayersAll: f.all, gpuLayers: f.all ? (f.m.meta?.blockCount ?? 0) : f.ngl, kvType: f.kv, flashAttn: f.fa, threads: f.threads
    }
    const r = await window.api.serveStart(cfg)
    setServe(r.ok ? await window.api.serveStatus() : { ...SERVE_IDLE, error: r.error })
    if (r.ok) setForm(null)
  }

  const run = (m: ModelInfo) => {
    setBusy(m.id)
    setErr(null)
    window.api.benchSmoke(m.path).then((r) => setSmoke({ model: m.name, r }), (e: Error) => setErr(e.message)).finally(() => setBusy(null))
  }

  return (
    <section>
      <header className="bar">
        <h1>Models</h1>
        <button onClick={load}>Rescan</button>
      </header>
      {err && <p className="err">{err}</p>}
      <RunningModel s={serve} onStop={() => void stop()} />
      {serve.error && !serve.url && <p className="err">{serve.error}</p>}
      {form && (
        <div className="export">
          <h2>Run {form.m.name}</h2>
          <p className="muted">{form.note}</p>
          <div className="bar">
            <label>Context <input type="number" min={512} step={512} value={form.ctx} style={{ width: 90 }} onChange={(e) => setForm({ ...form, ctx: Number(e.target.value) })} /></label>
            <label><input type="checkbox" checked={form.all} onChange={(e) => setForm({ ...form, all: e.target.checked })} /> all layers on GPU</label>
            {!form.all && <label>GPU layers <input type="number" min={0} max={form.m.meta?.blockCount ?? 999} value={form.ngl} style={{ width: 70 }} onChange={(e) => setForm({ ...form, ngl: Number(e.target.value) })} /></label>}
            <label>KV cache <select value={form.kv} onChange={(e) => setForm({ ...form, kv: e.target.value as KvType })}><option value="f16">f16</option><option value="q8_0">q8_0</option></select></label>
            <label><input type="checkbox" checked={form.fa} onChange={(e) => setForm({ ...form, fa: e.target.checked })} /> flash attention</label>
            <label>Threads <input type="number" min={0} value={form.threads} title="0 = physical cores" style={{ width: 60 }} onChange={(e) => setForm({ ...form, threads: Number(e.target.value) })} /></label>
            <button disabled={!!serve.busy || !!serve.url} onClick={() => void start(form)}>{serve.busy === 'starting' ? 'Starting…' : 'Start'}</button>
            <button onClick={() => setForm(null)}>Cancel</button>
          </div>
          <p className="muted">Sampling (temperature, top-p, …) and the system prompt are set in the llama-server web UI that opens.</p>
        </div>
      )}
      <table>
        <thead><tr><th>Name</th><th>Source</th><th>Arch</th><th>Params</th><th>Quant</th><th>Ctx (train)</th><th>KV/token*</th><th>Size</th><th>Path</th><th /></tr></thead>
        <tbody>
          {models?.map((m) => (
            <tr key={m.id}>
              <td>{m.name}{m.meta?.incomplete && <span className="pill warn-pill" title={`${m.meta.fileSizeBytes} of ≥${m.meta.expectedMinBytes} bytes`}>incomplete download</span>}</td>
              <td className="muted" title={m.ollamaName}>{SOURCE[m.runtime]}</td>
              {m.meta ? (
                <>
                  <td>{m.meta.arch ?? '—'}</td>
                  <td title={m.meta.parameterCount.source}>{params(m.meta.parameterCount.value)}</td>
                  <td>{m.meta.quantName ?? '—'}</td>
                  <td>{m.meta.contextLength ?? '—'}</td>
                  <td className="muted">{m.meta.estimated.kvCacheBytesPerToken != null ? `${(m.meta.estimated.kvCacheBytesPerToken / 1024).toFixed(0)} KiB` : '—'}</td>
                </>
              ) : (
                <td colSpan={5} className="err">{m.metaError ?? 'no metadata'}</td>
              )}
              <td>{gib(m.sizeBytes)}</td>
              <td className="muted">{m.path}</td>
              <td className="bar">
                <button disabled={busy !== null || !!serve.url || !!serve.busy} title="Start llama-server with this model and chat with it" onClick={() => void openRun(m)}>Run…</button>
                <button disabled={busy !== null} onClick={() => run(m)}>{busy === m.id ? 'Running…' : 'Smoke test'}</button>
                {m.runtime === 'llamacpp' && (linking?.id === m.id ? (
                  <>
                    <input value={linking.repo} placeholder="owner/name (base model repo)" style={{ width: 220 }} onChange={(e) => setLinking({ id: m.id, repo: e.target.value })} onKeyDown={(e) => { if (e.key === 'Enter') void link(m) }} />
                    <button onClick={() => void link(m)} disabled={!linking.repo.trim()}>Link</button>
                    {linking.msg && <span className="muted">{linking.msg}</span>}
                  </>
                ) : (
                  <button title={m.meta?.genKnobs.recommended ? `model card: ${JSON.stringify(m.meta.genKnobs.recommended)}` : 'Link a Hugging Face repo to use its recommended sampling'} onClick={() => setLinking({ id: m.id, repo: '' })}>
                    {m.meta?.genKnobs.recommended ? 'HF repo ✓' : 'HF repo'}
                  </button>
                ))}
              </td>
            </tr>
          ))}
          {models?.length === 0 && <tr><td colSpan={10} className="muted">No .gguf files in the configured model directories.</td></tr>}
        </tbody>
      </table>
      <p className="muted">Declared values come from the GGUF header. *KV/token is ESTIMATED (f16 K+V), used for pruning only.</p>
      {smoke && (
        <>
          <h2>Smoke result — {smoke.model}</h2>
          <table>
            <tbody>
              {rows(smoke.r).map(([k, v]) => <tr key={k}><td>{k}</td><td>{fmt(v)}</td></tr>)}
            </tbody>
          </table>
        </>
      )}
    </section>
  )
}
