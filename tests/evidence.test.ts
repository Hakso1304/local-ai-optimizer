import { describe, expect, it } from 'vitest'
import { evidenceOf } from '../src/renderer/src/Evidence'
import type { SessionDetail } from '../src/shared/types'

const m = (value: number | null) => (value === null ? { value, kind: 'unavailable', reason: 'x' } : { value, kind: 'measured', source: 's' })
const row = (rowId: number, ctx: number, supersededBy: number | null, status = 'pass') => ({
  rowId, recordedAt: 't', supersededBy, samplerErrors: [], ctx, status, failureKind: status === 'fail' ? 'oom' : null, configId: 'c|hip',
  decodeTps: m(50), prefillTps: m(900), ttftMs: m(100), peakVramBytes: m(8e9), peakSharedGpuBytes: m(null)
})

describe('Results → Evidence bundle', () => {
  it('keeps the trace, the rules versions and every row with what the scoring used vs what was superseded', () => {
    const d = {
      session: { id: 7, createdAt: 'c', status: 'done', workload: 'coding', demo: false },
      recommendation: { rulesVersion: 'pre', reinterpretedWith: 'interp-2', best: { configId: 'c|hip', score: { recommendedCtx: 16384 } }, provisional: false, reasons: ['[I-1.1] r'], decisionTrace: { winner: 'c|hip' } },
      candidates: [{
        config: { id: 'c|hip', backend: 'hip', device: 'ROCm0' }, model: { name: 'M' }, runIds: [3, 2], quality: [{}, {}], genQuality: [{ gen: { id: 'off' }, qualityScore: 71 }],
        history: [row(1, 8192, 3, 'fail'), row(2, 16384, null), row(3, 8192, null)]
      }]
    } as unknown as SessionDetail
    const e = evidenceOf(d)
    expect(e.session).toEqual({ id: 7, createdAt: 'c', status: 'done', workload: 'coding', demo: false })
    expect(e.recommendation).toMatchObject({ rulesVersion: 'pre', reinterpretedWith: 'interp-2', best: 'c|hip', recommendedCtx: 16384 })
    expect(e.decisionTrace).toEqual({ winner: 'c|hip' })
    const c = e.candidates[0]
    expect(c).toMatchObject({ backend: 'hip', device: 'ROCm0', scoredRowIds: [3, 2], quality: { rows: 2, genConfigs: [{ gen: 'off', qualityScore: 71 }] } })
    expect(c.rows.map((r) => [r.rowId, r.used, r.supersededBy])).toEqual([[1, false, 3], [2, true, null], [3, true, null]])
    expect(c.rows[0].peakSharedGpuBytes).toEqual(m(null)) // unavailable stays unavailable (never 0)
    expect(JSON.parse(JSON.stringify(e))).toEqual(e) // exportable as-is
  })
})
