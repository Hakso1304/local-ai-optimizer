import { useEffect, useState } from 'react'
import type { ModelInfo, SmokeResult } from '../../shared/types'

const gib = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
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
        <thead><tr><th>Name</th><th>Size</th><th>Path</th><th /></tr></thead>
        <tbody>
          {models?.map((m) => (
            <tr key={m.id}>
              <td>{m.name}</td>
              <td>{gib(m.sizeBytes)}</td>
              <td className="muted">{m.path}</td>
              <td className="bar"><button disabled={busy !== null} onClick={() => run(m)}>{busy === m.id ? 'Running…' : 'Smoke test'}</button></td>
            </tr>
          ))}
          {models?.length === 0 && <tr><td colSpan={4} className="muted">No .gguf files in the configured model directories.</td></tr>}
        </tbody>
      </table>
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
