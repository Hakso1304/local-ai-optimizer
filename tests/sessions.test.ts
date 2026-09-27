// Test scope: REAL node:sqlite on temp files and the real storage code; runs/quality rows are synthetic payloads (the
// runner's shape, not measured data). The demo seed uses tests/fixtures/scoring.
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { openDb } from '../src/core/storage/db'
import { sessionInputs, getSession, getSessionResume, latestRecommendation, listSessions, makeSessionStorage, markInterrupted, saveRecommendation, saveSession, seedDemoSession, telemetryForRun } from '../src/core/storage/sessions'
import { insertTelemetrySamples } from '../src/core/storage/db'
import type { BenchmarkRunResult, CandidateConfig, ModelMeta, Recommendation } from '../src/shared/bench-types'
import type { SystemProfile } from '../src/shared/types'

const dir = mkdtempSync(join(tmpdir(), 'lao-sess-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))
const fixtures = join(__dirname, 'fixtures', 'scoring')

describe('session storage', () => {
  it('seeds one DEMO session (idempotent) with a 16K→32K cliff and a recommendation', () => {
    const db = openDb(join(dir, 'a.db'))
    const id = seedDemoSession(db, fixtures)
    expect(seedDemoSession(db, fixtures)).toBe(id)
    const list = listSessions(db)
    expect(list).toHaveLength(1)
    expect(list[0]).toMatchObject({ id, demo: true, status: 'demo', candidateCount: 2 })
    const d = getSession(db, id)!
    expect(d.candidates).toHaveLength(2)
    const cliff = d.candidates.find((c) => c.config.kvType === 'f16')!.cliff
    expect(cliff.practicalContextCeiling.value).toBe(16384)
    expect(cliff.steps.find((s) => s.ctx === 32768)!.reasons.map((r) => r.code)).toContain('decode_drop')
    expect(d.recommendation?.reasons[0]).toMatch(/^DEMO DATA/)
    expect(d.candidates.every((c) => c.score !== null)).toBe(true)
    db.close()
  })

  it('latestRecommendation never returns demo data', () => {
    const db = openDb(join(dir, 'b.db'))
    seedDemoSession(db, fixtures)
    expect(latestRecommendation(db, 'long_context_coding')).toBeNull()
    const real = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'done')
    const rec = { workload: 'coding', best: null, reasons: ['x'] } as unknown as Recommendation
    saveRecommendation(db, real, rec, null)
    expect(latestRecommendation(db, 'coding')).toMatchObject({ sessionId: real, recommendation: { reasons: ['x'] } })
    expect(latestRecommendation(db, 'long_context_coding')).toBeNull()
    db.close()
  })

  it('telemetryForRun returns samples in insertion order', () => {
    const db = openDb(join(dir, 'c.db'))
    const sid = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'done')
    const rid = Number(db.prepare("INSERT INTO benchmark_run (session_id, status, payload) VALUES (?, 'pass', '{}')").run(sid).lastInsertRowid)
    insertTelemetrySamples(db, sid, rid, [{ ts: 1 }, { ts: 2 }])
    expect(telemetryForRun(db, rid).map((s) => s.ts)).toEqual([1, 2])
    db.close()
  })

  it('makeSessionStorage round-trips what the runner writes, and getSession reads it back', async () => {
    const db = openDb(join(dir, 'd.db'))
    const model = { id: 'E:/m/tiny.gguf', name: 'tiny', layers: 2, ctxTrain: 4096, quant: 'Q8_0' } as ModelMeta
    const config = { id: `${model.id}|ngl=all|kv=f16|t=8`, modelId: model.id, kvType: 'f16', gpuLayers: 99, gpuLayersAll: true, threads: 8 } as CandidateConfig
    const machine = { scannedAt: 'scan-1' } as SystemProfile
    const st = makeSessionStorage(db, () => ({ vramBytes: 16e9, candidates: [{ config, model }], machine }))
    const req = { workload: 'fast_assistant' as const, modelIds: [model.id] }
    const id = await st.createSession({ workload: req.workload, request: req, startedAt: 0 })
    const run = { configId: config.id, ctx: 2048, status: 'pass', decodeTps: { value: 300, kind: 'measured' } } as unknown as BenchmarkRunResult
    await st.saveRun(id, run, { samples: [{ ts: 5 } as never], reason: null, stderrTail: [], load: null, startedAt: 1, endedAt: 2 })
    expect(await st.listRuns(id)).toEqual([run]) // detail is stored but not leaked into BenchmarkRunResult
    await st.saveQuality(id, model.id, config.id, 2048, [{ testId: 't', category: 'coding', weight: 1, pass: true, score: 1, detail: '' }])
    expect((await st.listQuality(id, model.id)).map((q) => q.testId)).toEqual(['t'])
    await st.saveRecommendation(id, { workload: 'fast_assistant', best: { configId: config.id }, ranked: [], reasons: ['r'] } as unknown as Recommendation)
    await st.setSessionStatus(id, 'failed', 'boom')

    const d = getSession(db, Number(id))!
    expect(d.session).toMatchObject({ status: 'failed', error: 'boom', workload: 'fast_assistant', demo: false, bestConfigId: config.id })
    expect(d.candidates[0].runs).toHaveLength(1)
    expect(d.candidates[0].quality).toHaveLength(1)
    const runId = (db.prepare('SELECT id, model_id FROM benchmark_run').get() as { id: number; model_id: string })
    expect(runId.model_id).toBe(model.id)
    expect(telemetryForRun(db, runId.id)).toHaveLength(1)
    expect(getSessionResume(db, Number(id))).toEqual({ request: req, machine, plan: [config] }) // resume re-uses the stored plan + scan (d)
    expect(latestRecommendation(db, 'fast_assistant')).toBeNull() // failed session: not a real recommendation (F14)
    await st.setSessionStatus(id, 'done')
    expect(latestRecommendation(db, 'fast_assistant')?.sessionId).toBe(Number(id))
    db.close()
  })

  it('a retried/rerun step supersedes the old row in listRuns and getSession (a)', async () => {
    const db = openDb(join(dir, 'e.db'))
    const model = { id: 'E:/m/x.gguf', name: 'x' } as ModelMeta
    const config = { id: 'E:/m/x.gguf|ngl=all|kv=f16|t=8', modelId: model.id } as CandidateConfig
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [{ config, model }] }))
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: [model.id] }, startedAt: 0 })
    const detail = { samples: [], reason: null, stderrTail: [], load: null, startedAt: 0, endedAt: 0 }
    const run = (ctx: number, status: string) => ({ configId: config.id, ctx, status }) as unknown as BenchmarkRunResult
    await st.saveRun(id, run(2048, 'pass'), detail)
    await st.saveRun(id, run(4096, 'fail'), detail)
    await st.saveRun(id, run(4096, 'pass'), detail) // retry
    expect((await st.listRuns(id)).map((r) => `${r.ctx}:${r.status}`)).toEqual(['2048:pass', '4096:pass'])
    const cand = getSession(db, Number(id))!.candidates[0]
    expect(cand.runs.map((r) => `${r.ctx}:${r.status}`)).toEqual(['2048:pass', '4096:pass'])
    expect(cand.runIds).toEqual([1, 3]) // row ids of the surviving rows, for telemetryForRun
    // the superseded attempt stays in history, pointing at its replacement (W4f: a failed attempt still surfaces)
    expect(cand.history.map((r) => [r.rowId, r.status, r.supersededBy])).toEqual([[1, 'pass', null], [2, 'fail', 3], [3, 'pass', null]])
    expect(sessionInputs(db, Number(id))!.inputs[0]).toMatchObject({ history: expect.arrayContaining([expect.objectContaining({ rowId: 2, supersededBy: 3 })]) })
    // engine inputs (interp-2): every attempt with string ids, and why the session stopped
    const si = sessionInputs(db, Number(id))!
    expect(si.allRuns.map((r) => [r.runId, r.status, r.supersededBy ?? null])).toEqual([['1', 'pass', null], ['2', 'fail', '3'], ['3', 'pass', null]])
    expect(si.stopReason).toBeUndefined() // status still 'running'
    db.close()
  })

  it('markInterrupted flips sessions left running by a previous app run (b)', () => {
    const db = openDb(join(dir, 'f.db'))
    const a = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'running')
    const b = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'done')
    expect(markInterrupted(db)).toBe(1)
    expect(listSessions(db).map((s) => [s.id, s.status])).toEqual([[b, 'done'], [a, 'interrupted']])
    db.close()
  })

  it('sessionInputs re-scores stored measurements for another workload without touching the stored recommendation', async () => {
    const { recommend } = await import('../src/core/scoring/recommend')
    const { machineFromProfile } = await import('../src/core/benchmark/candidates')
    const db = openDb(join(dir, 'g.db'))
    const id = seedDemoSession(db, fixtures)
    const s = sessionInputs(db, id)!
    expect(s.inputs).toHaveLength(2)
    expect(s.inputs[0].runs.length).toBeGreaterThan(0)
    const machine = machineFromProfile({ gpus: { value: [], status: 'available', source: '' }, ram: { value: null, status: 'unavailable', source: '' }, cpu: { value: null, status: 'unavailable', source: '' } } as never, 'Vulkan0')
    const rec = recommend(s.inputs, machine, 'coding')
    expect(rec.workload).toBe('coding')
    expect(getSession(db, id)!.recommendation!.workload).toBe('long_context_coding') // stored one unchanged
    expect(sessionInputs(db, 999)).toBeNull()
    db.close()
  })

  it('latestRecommendation only comes from sessions that finished (W4 F14) and quality needs a complete current suite (F5)', async () => {
    const db = openDb(join(dir, 'h.db'))
    const sid = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'running')
    saveRecommendation(db, sid, { workload: 'coding', best: null, reasons: ['partial'] } as unknown as Recommendation, null)
    markInterrupted(db)
    expect(latestRecommendation(db, 'coding')).toBeNull()
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [] }))
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: ['m'] }, startedAt: 0 })
    const q = (t: string) => ({ testId: t, category: 'coding' as const, weight: 1, pass: true, score: 1, detail: '' })
    await st.saveQuality(id, 'm', 'c', 2048, [q('a'), q('b')])
    expect(await st.listQuality(id, 'm')).toHaveLength(2)
    db.prepare('DELETE FROM quality_result WHERE id = (SELECT max(id) FROM quality_result)').run() // now incomplete
    expect(await st.listQuality(id, 'm')).toEqual([]) // resume re-runs the suite
    db.close()
  })

  it('runner-filled run extras (repDecodeTps, minRamAvailBytes) survive the round trip', async () => {
    const db = openDb(join(dir, 'x.db'))
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [] }))
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: ['m'] }, startedAt: 0 })
    const run = { configId: 'm|ngl=all', ctx: 4096, status: 'pass', repDecodeTps: [301.5, 298.2], minRamAvailBytes: { value: 8e9, kind: 'measured' } } as unknown as BenchmarkRunResult
    await st.saveRun(id, run, { samples: [], reason: null, stderrTail: [], load: null, startedAt: 1, endedAt: 2 })
    expect((await st.listRuns(id))[0]).toMatchObject({ repDecodeTps: [301.5, 298.2], minRamAvailBytes: { value: 8e9 } })
    db.close()
  })
})
