// Results → "Evidence": the decision trace and every run row it rests on (scored rows and the attempts they
// superseded), exportable as one JSON file. Read-only; values exactly as stored (Metric objects keep their provenance).
import { useState } from 'react'
import type { SessionDetail } from '../../shared/types'
import { M, fmtCtx, gib } from './ui'

/** The session's evidence bundle: what was decided, with which rules, from which stored rows. Pure. */
export function evidenceOf(d: SessionDetail) {
  const rec = d.recommendation as (SessionDetail['recommendation'] & { reinterpretedWith?: string; rulesVersion?: string }) | null
  return {
    session: { id: d.session.id, createdAt: d.session.createdAt, status: d.session.status, workload: d.session.workload, demo: d.session.demo },
    recommendation: rec && {
      rulesVersion: rec.rulesVersion ?? null, reinterpretedWith: rec.reinterpretedWith ?? null, best: rec.best?.configId ?? null,
      recommendedCtx: rec.best?.score.recommendedCtx ?? null, provisional: rec.provisional ?? null, provisionalBest: rec.provisionalBest ?? null, reasons: rec.reasons
    },
    decisionTrace: rec?.decisionTrace ?? null,
    candidates: d.candidates.map((c) => ({
      configId: c.config.id, model: c.model.name, backend: c.config.backend ?? 'vulkan', device: c.config.device,
      // rows the scoring used (latest attempt per ctx), then every attempt with what replaced it
      scoredRowIds: c.runIds,
      rows: c.history.map((h) => ({
        rowId: h.rowId, recordedAt: h.recordedAt, supersededBy: h.supersededBy, used: h.supersededBy === null,
        ctx: h.ctx, status: h.status, failureKind: h.failureKind, decodeTps: h.decodeTps, prefillTps: h.prefillTps, ttftMs: h.ttftMs,
        peakVramBytes: h.peakVramBytes, peakSharedGpuBytes: h.peakSharedGpuBytes, samplerErrors: h.samplerErrors
      })),
      quality: { rows: c.quality.length, genConfigs: c.genQuality?.map((g) => ({ gen: g.gen.id, qualityScore: g.qualityScore })) ?? [] }
    }))
  }
}

export function Evidence({ d }: { d: SessionDetail }) {
  const [saved, setSaved] = useState<string | null>(null)
  const save = async () => {
    const r = await window.api.saveFile(`lao-evidence-${d.session.id}.json`, JSON.stringify(evidenceOf(d), null, 2))
    setSaved(r.saved)
  }
  return (
    <details className="trace">
      <summary><b>Evidence</b> <span className="muted">— the run rows behind this result ({d.candidates.reduce((n, c) => n + c.history.length, 0)} rows)</span></summary>
      <div className="bar">
        <button onClick={() => void save()}>Export evidence JSON…</button>
        {saved && <span className="muted">saved to {saved}</span>}
      </div>
      {d.candidates.map((c) => (
        <div key={c.config.id}>
          <h3>{c.model.name} <span className="muted">{c.config.id}</span></h3>
          <table>
            <thead><tr><th>Row</th><th>Ctx</th><th>Status</th><th>Decode t/s</th><th>Prefill t/s</th><th>Peak VRAM</th><th>Shared</th><th>Used</th></tr></thead>
            <tbody>
              {c.history.map((h) => (
                <tr key={h.rowId} className={h.supersededBy === null ? '' : 'muted'}>
                  <td>#{h.rowId}</td><td>{fmtCtx(h.ctx)}</td><td>{h.status}{h.failureKind ? ` (${h.failureKind})` : ''}</td>
                  <td><M m={h.decodeTps} /></td><td><M m={h.prefillTps} /></td><td><M m={h.peakVramBytes} fmt={gib} /></td><td><M m={h.peakSharedGpuBytes} fmt={gib} /></td>
                  <td>{h.supersededBy === null ? 'scored' : `superseded by #${h.supersededBy}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </details>
  )
}
