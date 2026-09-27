import { useEffect, useState } from 'react'
import type { ModelInfo, SmokeResult } from '../../shared/types'

const SOURCE = { llamacpp: 'folder', lmstudio: 'LM Studio', ollama: 'Ollama' } as const
const gib = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
const params = (n: number | null) => (n == null ? '—' : n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : `${(n / 1e6).toFixed(0)}M`)
const fmt = (v: unknown) => (v == null ? '—' : typeof v === 'number' ? String(Math.round(v * 100) / 100) : typeof v === 'object' ? JSON.stringify(v) : String(v))

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
