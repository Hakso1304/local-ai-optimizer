import { useCallback, useEffect, useState } from 'react'
// Small shared view helpers. No logic beyond formatting.
import type { Metric, ProvenanceKind } from '../../shared/bench-types'

import { agentSetupText } from '../../core/export/config'
import type { ServeStatus } from '../../shared/types'

export type ServeView = ServeStatus & { busy?: 'starting' | 'stopping'; error?: string }
export const SERVE_IDLE: ServeView = { url: null, configId: null, alias: null, ctx: null, stopping: false }

/** Served-model state shared by the Models and Results pages: main is the source of truth (serve:status), stop
 *  shows progress and any teardown error instead of silently leaving the line on "Running". */
export function useServe() {
  const [serve, setServe] = useState<ServeView>(SERVE_IDLE)
  const refresh = useCallback(() => window.api.serveStatus().then((s) => setServe({ ...s, ...(s.stopping ? { busy: 'stopping' as const } : {}) }), () => {}), [])
  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => { // while served: pick up a GPU loss (main latches it from the server log)
    if (!serve.url) return
    const t = setInterval(() => void refresh(), 5000)
    return () => clearInterval(t)
  }, [serve.url, refresh])
  const stop = async () => {
    setServe((s) => ({ ...s, busy: 'stopping', error: undefined }))
    const r = await window.api.serveStop().catch((e: Error) => ({ ok: false, error: e.message }))
    const s = await window.api.serveStatus().catch(() => SERVE_IDLE)
    setServe(r.ok ? SERVE_IDLE : { ...s, error: r.error })
  }
  return { serve, setServe, stop, refresh }
}

/** Status line for the model served from the app: web UI link, API details, agent setup to copy, stop. */
export function RunningModel({ s, onStop }: { s: ServeView; onStop: () => void }) {
  const [copied, setCopied] = useState(false)
  if (!s.url) return null
  const stopping = s.busy === 'stopping'
  const copy = async () => { await navigator.clipboard.writeText(agentSetupText(s.url!, s.alias ?? 'local-model', s.ctx ?? 0)); setCopied(true); setTimeout(() => setCopied(false), 2000) }
  return (
    <p className="bar">
      <span>Running <b title={s.configId ?? ''}>{s.alias}</b> — chat: <a href={s.url} target="_blank" rel="noreferrer">{s.url}</a> · API: <code>{s.url}/v1</code> (OpenAI + Anthropic compatible, tool calling on) · model id <code>{s.alias}</code></span>
      <button disabled={stopping} onClick={() => void copy()} title="Base URL, model id and env vars for Claude Code, Cline, Continue, OpenCode and other agent tools">{copied ? 'Copied' : 'Copy agent setup'}</button>
      <button disabled={stopping} onClick={onStop} title="Kills llama-server and verifies its process tree is gone (a few seconds)">{stopping ? 'Stopping…' : 'Stop model'}</button>
      {s.gpuLost && <span className="err">The GPU was reset (Vulkan device lost): every request now fails. Stop the model and start it again.</span>}
      {s.error && <span className="err">Stop failed: {s.error} — retry Stop; nothing else can start a server until the cleanup is verified.</span>}
    </p>
  )
}

export const fmtCtx = (n: number) => (n % 1024 === 0 ? `${n / 1024}K` : String(n))
export const gib = (b: number) => `${(b / 1024 ** 3).toFixed(2)} GiB`
export const num = (v: number, d = 1) => v.toLocaleString(undefined, { maximumFractionDigits: d })

export function Prov({ kind, title }: { kind: ProvenanceKind; title?: string }) {
  return <span className={`prov ${kind}`} title={title}>{kind.toUpperCase()}</span>
}

/** Value + provenance badge; unavailable shows "—" with the reason as tooltip, never 0. */
export function M({ m, fmt = (v: number) => num(v) }: { m: Metric | null | undefined; fmt?: (v: number) => string }) {
  if (!m) return <span className="muted">—</span>
  const tip = m.kind === 'unavailable' ? m.reason : m.source
  return (
    <span className="metric">
      {m.value == null ? <span className="muted" title={tip}>—</span> : fmt(m.value)} <Prov kind={m.kind} title={tip} />
    </span>
  )
}

export function DemoBanner({ what = 'DEMO DATA' }: { what?: string }) {
  return <div className="demo">{what}: generated from test fixtures, not measured on this machine</div>
}

export function ScoreBar({ label, score, kind, weight }: { label: string; score: number; kind: ProvenanceKind; weight?: number }) {
  return (
    <div className="scorebar">
      <span className="lbl">{label}</span>
      <span className="track"><span className="fill" style={{ width: `${Math.max(0, Math.min(100, score))}%` }} /></span>
      <span className="val">{score.toFixed(0)}</span>
      <Prov kind={kind} />
      {weight != null && <span className="muted">w {(weight * 100).toFixed(0)}%</span>}
    </div>
  )
}

export const COMPONENT_LABEL: Record<string, string> = {
  quality: 'Quality', genSpeed: 'Speed', prefillSpeed: 'Prefill', latency: 'Latency', memory: 'Memory', stability: 'Stability', context: 'Context'
}

/** "32K (scored at 16K)" — recommendedCtx is what to configure; referenceCtx is where speed was scored. */
export function CtxPick({ recommended, scored }: { recommended: number | null | undefined; scored: number | null | undefined }) {
  if (recommended == null) return <span className="muted" title="no passing step fits the workload latency tolerance">—</span>
  return <span>{fmtCtx(recommended)} <Prov kind="measured" />{scored != null && scored !== recommended && <span className="muted"> (scored at {fmtCtx(scored)})</span>}</span>
}

/** Backend a config runs on, for labels: device null = CPU; absent backend = Vulkan (pre-HIP sessions). */
export function backendLabel(c: { backend?: string; device: string | null }): string {
  if (!c.device) return 'llama.cpp CPU'
  const kind = c.backend === 'hip' ? 'ROCm (HIP)' : c.backend === 'cuda' ? 'CUDA' : c.backend === 'prism' ? 'PrismML ternary (Vulkan)' : 'Vulkan'
  return `llama.cpp ${kind} (${c.device})`
}
