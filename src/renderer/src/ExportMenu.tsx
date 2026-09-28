import { useEffect, useState } from 'react'
import { exportConfigFrom, provenanceNote, toJson, toLlamaServerCommand, toLmStudioSettings, toOllamaModelfile } from '../../core/export/config'
import { templateKwargsFor } from '../../core/benchmark/gen'
import type { Recommendation } from '../../shared/bench-types'
import type { ServeStatus, SessionCandidate } from '../../shared/types'
import { RunningModel } from './ui'

/** Export of the recommended config. The llama-server command reproduces the measured launch exactly; Ollama and
 *  LM Studio outputs are translations (LM Studio keys unverified). */
export function ExportMenu({ rec, cand, sessionId, ctx }: { rec: Recommendation; cand: SessionCandidate | undefined; sessionId: number; ctx?: number | null }) {
  const [done, setDone] = useState<string | null>(null)
  const [exes, setExes] = useState<Record<string, string | null>>({})
  const idle: ServeStatus = { url: null, configId: null, alias: null, ctx: null }
  const [serve, setServe] = useState<ServeStatus & { busy?: boolean; error?: string }>(idle)
  useEffect(() => { window.api.installedBackends().then((bs) => setExes(Object.fromEntries(bs.map((b) => [b.kind, b.status === 'available' ? b.exePath : null]))), () => {}) }, [])
  useEffect(() => { window.api.serveStatus().then(setServe, () => {}) }, [])
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
  const missing = !exe
  const command = () => [
    ...(missing ? [`# measured on the llama.cpp ${cfg.backend === 'hip' ? 'ROCm (HIP)' : cfg.backend.toUpperCase()} build, which is not installed: install it on the System page (or use that build's llama-server)`] : []),
    [toLlamaServerCommand(cfg, exe ?? 'llama-server'), genArgs].filter(Boolean).join(' ')
  ].join('\n')
  const json = () => gen ? JSON.stringify({ ...JSON.parse(toJson(cfg, exe)), generation: { ...gen, templateKwargs: kwargs ?? null } }, null, 2) : toJson(cfg, exe)
  const copy = async (label: string, text: string) => { await navigator.clipboard.writeText(text); setDone(`${label} copied`) }
  const save = async (label: string, name: string, text: string) => {
    const r = await window.api.saveFile(name, text)
    setDone(r.saved ? `${label} saved to ${r.saved}` : null)
  }
  const run = async () => {
    setServe({ ...idle, busy: true })
    const r = await window.api.serveStart(cfg)
    setServe(r.ok ? await window.api.serveStatus() : { ...idle, error: r.error })
  }
  const stop = async () => { await window.api.serveStop(); setServe(idle) }
  return (
    <div className="export">
      {serve.url
        ? <RunningModel s={serve} onStop={() => void stop()} />
        : <div className="bar"><button disabled={missing || serve.busy} onClick={() => void run()} title="Start llama-server with this exact config and open its chat UI in your browser">{serve.busy ? 'Starting…' : 'Run this model'}</button>{serve.error && <span className="err">{serve.error}</span>}</div>}
      <div className="bar">
        <button disabled={missing} onClick={() => void copy('llama-server command', command())}>Copy llama-server command</button>
        <button onClick={() => void copy('Ollama Modelfile', modelfile())}>Copy Ollama Modelfile</button>
        <button onClick={() => void copy('LM Studio settings', JSON.stringify(toLmStudioSettings(cfg), null, 2))} title="Key names not verified against a real LM Studio install">Copy LM Studio settings (unverified)</button>
        <button disabled={missing} onClick={() => void copy('JSON', json())}>Copy JSON</button>
        <button onClick={() => void save('Modelfile', 'Modelfile', modelfile())}>Save Modelfile…</button>
        <button disabled={missing} onClick={() => void save('JSON', `lao-config-${sessionId}.json`, json())}>Save JSON…</button>
      </div>
      {done && <p className="muted">{done}</p>}
      <pre className="log">{provenanceNote(rec)}</pre>
    </div>
  )
}
