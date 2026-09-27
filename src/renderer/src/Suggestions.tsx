// Sibling-quantization suggestions (I-9.2) and the one-click download behind them: Hub download into the model's
// own folder, then link the repo (sidecar → baseModelId for the same-base comparison), then include the file.
import { useEffect, useState } from 'react'
import { parseDownloadAction } from '../../core/interpret/catalog'
import type { QuantSuggestion } from '../../shared/bench-types'
import type { HubApi } from '../../shared/hub-types'
import { fmtCtx, gib } from './ui'

const hub = () => window.api as unknown as HubApi // hub bridge is spread into window.api (preload/hub.ts)

/** Folder of a model file (Windows or POSIX separators). */
export const dirOf = (p: string): string => p.replace(/[\\/][^\\/]*$/, '')

/** Download one sibling file into destDir, then link its repo. Returns the new file path or throws a readable error. */
export async function downloadSibling(repoId: string, path: string, destDir: string): Promise<string> {
  const r = await hub().hubDownload({ repoId, path, destDir })
  if (!r.ok) throw new Error(r.error)
  await window.api.linkModelRepo(r.filePath, repoId).catch(() => null) // card fetch failure doesn't undo the download
  return r.filePath
}

/** Insight action "download <owner>/<repo>/<path>": confirm, then download into the folder of the model it is a
 *  sibling of. false = no known destination (the caller falls back to the Hub page). */
export async function downloadFromAction(action: string, modelPaths: string[]): Promise<boolean> {
  const a = parseDownloadAction(action.startsWith('download ') ? action : `download ${action}`)
  if (!a) return false
  const base = modelPaths.find(Boolean)
  if (!base) return false
  const dest = dirOf(base)
  if (!window.confirm(`Download ${a.path} from ${a.repoId} into ${dest}?`)) return true
  try {
    const f = await downloadSibling(a.repoId, a.path, dest)
    window.alert(`Downloaded ${f}. Include it in the next benchmark from the Benchmark page.`)
  } catch (e) {
    window.alert(`Download failed: ${(e as Error).message}`)
  }
  return true
}

export function Suggestions({ items, onIncluded, disabled }: { items: QuantSuggestion[]; onIncluded: (filePath: string) => void; disabled?: boolean }) {
  const [busy, setBusy] = useState<string | null>(null)
  const [pct, setPct] = useState<number | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  useEffect(() => hub().onHubProgress((p) => setPct(p.pct)), [])
  if (!items.length) return null
  const go = async (s: QuantSuggestion) => {
    const file = s.sibling.path.split('/').pop()!
    if (!window.confirm(`Download ${file} (${gib(s.sibling.sizeBytes)}) from ${s.sibling.repoId} into ${dirOf(s.modelId)} and include it?`)) return
    setBusy(s.sibling.path); setPct(null); setMsg(null)
    try {
      const f = await downloadSibling(s.sibling.repoId, s.sibling.path, dirOf(s.modelId))
      onIncluded(f)
      setMsg(`Downloaded and included ${f}`)
    } catch (e) {
      setMsg(`Download failed: ${(e as Error).message}`)
    } finally {
      setBusy(null)
    }
  }
  return (
    <div className="card">
      <h3>Suggested quantizations <span className="muted">(not benchmarked; estimated)</span></h3>
      <ul>
        {items.map((s) => (
          <li key={`${s.modelId}|${s.sibling.path}`}>
            <b>{s.sibling.quant}</b> {s.sibling.path.split('/').pop()} — {gib(s.sibling.sizeBytes)}: {s.gpuLayers}/{s.layers} layers on GPU at {fmtCtx(s.ctx)}{' '}
            <span className={`pill${s.speedClass === 'partial' ? ' warn-pill' : ''}`}>{s.speedClass === 'full-gpu' ? 'full GPU' : 'partial offload'}</span>
            <span className="muted"> (this file: {s.currentGpuLayers}/{s.layers})</span>{' '}
            <button className="mini" disabled={disabled || busy !== null} title={s.text} onClick={() => void go(s)}>
              {busy === s.sibling.path ? `Downloading…${pct != null ? ` ${pct}%` : ''}` : 'Download & include'}
            </button>
          </li>
        ))}
      </ul>
      {msg && <p className="muted">{msg}</p>}
    </div>
  )
}
