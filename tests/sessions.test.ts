import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { openDb } from '../src/core/storage/db'
import { getSession, getSessionRequest, latestRecommendation, listSessions, makeSessionStorage, saveRecommendation, saveSession, seedDemoSession, telemetryForRun } from '../src/core/storage/sessions'
import { insertTelemetrySamples } from '../src/core/storage/db'
import type { BenchmarkRunResult, CandidateConfig, ModelMeta, Recommendation } from '../src/shared/bench-types'

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
    const st = makeSessionStorage(db, () => ({ vramBytes: 16e9, candidates: [{ config, model }] }))
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
    expect(getSessionRequest(db, Number(id))).toEqual(req)
    expect(latestRecommendation(db, 'fast_assistant')?.sessionId).toBe(Number(id))
    db.close()
  })
})
