import { useState } from 'react'
import { exportConfigFrom, provenanceNote, toJson, toLlamaServerCommand, toLmStudioSettings, toOllamaModelfile } from '../../core/export/config'
import type { Recommendation } from '../../shared/bench-types'
import type { SessionCandidate } from '../../shared/types'

/** Export of the recommended config. The llama-server command reproduces the measured launch exactly; Ollama and
 *  LM Studio outputs are translations (LM Studio keys unverified). */
export function ExportMenu({ rec, cand, sessionId }: { rec: Recommendation; cand: SessionCandidate | undefined; sessionId: number }) {
  const [done, setDone] = useState<string | null>(null)
  const cfg = cand ? exportConfigFrom(rec, cand.config, cand.model, String(sessionId)) : null
  if (!cfg) return <p className="muted">Nothing to export: {rec.best ? 'no recommended context' : 'no recommendation'}.</p>
  const modelfile = () => toOllamaModelfile(cfg, { from: cfg.modelPath })
  const copy = async (label: string, text: string) => { await navigator.clipboard.writeText(text); setDone(`${label} copied`) }
  const save = async (label: string, name: string, text: string) => {
    const r = await window.api.saveFile(name, text)
    setDone(r.saved ? `${label} saved to ${r.saved}` : null)
  }
  return (
    <div className="export">
      <div className="bar">
        <button onClick={() => void copy('llama-server command', toLlamaServerCommand(cfg))}>Copy llama-server command</button>
        <button onClick={() => void copy('Ollama Modelfile', modelfile())}>Copy Ollama Modelfile</button>
        <button onClick={() => void copy('LM Studio settings', JSON.stringify(toLmStudioSettings(cfg), null, 2))} title="Key names not verified against a real LM Studio install">Copy LM Studio settings (unverified)</button>
        <button onClick={() => void copy('JSON', toJson(cfg))}>Copy JSON</button>
        <button onClick={() => void save('Modelfile', 'Modelfile', modelfile())}>Save Modelfile…</button>
        <button onClick={() => void save('JSON', `lao-config-${sessionId}.json`, toJson(cfg))}>Save JSON…</button>
      </div>
      {done && <p className="muted">{done}</p>}
      <pre className="log">{provenanceNote(rec)}</pre>
    </div>
  )
}
