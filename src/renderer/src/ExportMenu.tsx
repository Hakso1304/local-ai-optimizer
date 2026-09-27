import { useEffect, useState } from 'react'
import { exportConfigFrom, provenanceNote, toJson, toLlamaServerCommand, toLmStudioSettings, toOllamaModelfile } from '../../core/export/config'
import { templateKwargsFor } from '../../core/benchmark/gen'
import type { Recommendation } from '../../shared/bench-types'
import type { SessionCandidate } from '../../shared/types'

/** Export of the recommended config. The llama-server command reproduces the measured launch exactly; Ollama and
 *  LM Studio outputs are translations (LM Studio keys unverified). */
export function ExportMenu({ rec, cand, sessionId, ctx }: { rec: Recommendation; cand: SessionCandidate | undefined; sessionId: number; ctx?: number | null }) {
  const [done, setDone] = useState<string | null>(null)
  const [exes, setExes] = useState<Record<string, string | null>>({})
  useEffect(() => { window.api.installedBackends().then((bs) => setExes(Object.fromEntries(bs.map((b) => [b.kind, b.status === 'available' ? b.exePath : null]))), () => {}) }, [])
  const base = cand ? exportConfigFrom(rec, cand.config, cand.model, String(sessionId)) : null
  // An interpretation "use-context" action can pin -c (e.g. the last clean rung); otherwise recommendedCtx.
  const cfg = base && ctx ? { ...base, ctx } : base
  if (!cfg) return <p className="muted">Nothing to export: {rec.best ? 'no recommended context' : 'no recommendation'}.</p>
  const modelfile = () => toOllamaModelfile(cfg, { from: cfg.modelPath })
  // Chosen generation config → llama-server sampling defaults + chat-template kwargs (thinking / effort).
  const gen = rec.best?.gen?.config
  const kwargs = gen && cand ? templateKwargsFor(cand.model, gen) : undefined
  const genArgs = gen ? [
    `--temp ${gen.temperature}`, gen.topP !== undefined ? `--top-p ${gen.topP}` : '', gen.topK !== undefined ? `--top-k ${gen.topK}` : '',
    gen.minP !== undefined ? `--min-p ${gen.minP}` : '', kwargs ? `--chat-template-kwargs "${JSON.stringify(kwargs).replace(/"/g, '\\"')}"` : ''
  ].filter(Boolean).join(' ') : ''
  // The measured backend's own llama-server (a HIP config on the Vulkan exe would not find ROCm0).
  const exe = exes[cfg.backend] ?? null
  const missing = cfg.backend !== 'vulkan' && !exe
  const command = () => [
    ...(missing ? [`# measured on the llama.cpp ${cfg.backend === 'hip' ? 'ROCm (HIP)' : cfg.backend.toUpperCase()} build, which is not installed: install it on the System page (or use that build's llama-server)`] : []),
    [toLlamaServerCommand(cfg, exe ?? 'llama-server'), genArgs].filter(Boolean).join(' ')
  ].join('\n')
  const json = () => gen ? JSON.stringify({ ...JSON.parse(toJson(cfg)), generation: { ...gen, templateKwargs: kwargs ?? null } }, null, 2) : toJson(cfg)
  const copy = async (label: string, text: string) => { await navigator.clipboard.writeText(text); setDone(`${label} copied`) }
  const save = async (label: string, name: string, text: string) => {
    const r = await window.api.saveFile(name, text)
    setDone(r.saved ? `${label} saved to ${r.saved}` : null)
  }
  return (
    <div className="export">
      <div className="bar">
        <button onClick={() => void copy('llama-server command', command())}>Copy llama-server command</button>
        <button onClick={() => void copy('Ollama Modelfile', modelfile())}>Copy Ollama Modelfile</button>
        <button onClick={() => void copy('LM Studio settings', JSON.stringify(toLmStudioSettings(cfg), null, 2))} title="Key names not verified against a real LM Studio install">Copy LM Studio settings (unverified)</button>
        <button onClick={() => void copy('JSON', json())}>Copy JSON</button>
        <button onClick={() => void save('Modelfile', 'Modelfile', modelfile())}>Save Modelfile…</button>
        <button onClick={() => void save('JSON', `lao-config-${sessionId}.json`, json())}>Save JSON…</button>
      </div>
      {done && <p className="muted">{done}</p>}
      <pre className="log">{provenanceNote(rec)}</pre>
    </div>
  )
}
