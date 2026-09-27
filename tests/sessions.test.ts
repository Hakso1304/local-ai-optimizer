// Test scope: REAL node:sqlite on temp files and the real storage code; runs/quality rows are synthetic payloads (the
// runner's shape, not measured data). The demo seed uses tests/fixtures/scoring.
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { openDb, relabelV2Quality } from '../src/core/storage/db'
import { sessionInputs, getSession, getSessionResume, latestRecommendation, listSessions, makeSessionStorage, markInterrupted, saveRecommendation, saveSession, seedDemoSession, telemetryForRun } from '../src/core/storage/sessions'
import { insertTelemetrySamples } from '../src/core/storage/db'
import type { BenchmarkRunResult, CandidateConfig, ModelMeta, QualityResult, Recommendation } from '../src/shared/bench-types'
import type { SystemProfile } from '../src/shared/types'
import { suiteFor } from '../src/core/quality'
import { proofRowId } from '../src/core/benchmark/gen'

const dir = mkdtempSync(join(tmpdir(), 'lao-sess-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))
const fixtures = join(__dirname, 'fixtures', 'scoring')

describe('session storage', () => {
  it('annotates malformed, runtime-original and replay quality origins without rewriting stored payloads', async () => {
    const db = openDb(join(dir, 'quality-origin-read.db'))
    const model = { id: 'E:/m/origin.gguf', name: 'origin' } as ModelMeta
    const config = { id: `${model.id}|ngl=all|kv=f16|t=8`, modelId: model.id, backend: 'vulkan' } as CandidateConfig
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [{ config, model }] }))
    const hash = 'a'.repeat(64), alternate = 'b'.repeat(64)
    const mark = (row: unknown) => (row as { originValidation?: { ok: boolean; classification: string; reason: string | null } }).originValidation
    for (const [name, provenance, expected] of [
      ['runtime', { mode: 'runtime', status: 'original', originalPromptHashPresent: true,
        origin: { generationPromptHashPresent: true, firstReplayAt: null, lineage: [] } }, 'original'],
      ['replay', { mode: 'live-template-replay', status: 'original', originalPromptHashPresent: true,
        origin: { generationPromptHashPresent: true, firstReplayAt: '2026-09-28T00:00:00.000Z', lineage: [hash] } }, 'reconstructed'],
      ['malformed', { mode: 'runtime', status: 'original', originalPromptHashPresent: true,
        origin: { generationPromptHashPresent: true, firstReplayAt: 'not-an-iso-time', lineage: ['not-a-hash'] } }, 'incoherent']
    ] as const) {
      const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: [model.id], qualityMode: 'quick' }, startedAt: 0 })
      await st.saveRun(id, { configId: config.id, ctx: 2048, status: 'pass', versions: { runtime: 'vulkan:b1' } } as unknown as BenchmarkRunResult,
        { samples: [], reason: null, stderrTail: [], load: null, startedAt: 0, endedAt: 1 })
      const tagged = { configId: config.id, genId: 'off', testId: name, sample: 1 }
      const row = { ...tagged, suiteSeed: null, category: 'coding', weight: 1, pass: true, score: 1, detail: '', suite: suiteFor('quick', 0).suite,
        backend: 'vulkan', runtimeVersion: 'vulkan:b1', promptSha256: hash, proofProvenance: provenance,
        renderProof: { rowId: proofRowId(tagged), promptSha256: hash, renderedSha256: hash,
          counterfactualSha256: alternate, keys: ['enable_thinking'], status: 'proved' } } as unknown as QualityResult
      await st.saveQuality(id, model.id, config.id, 2048, [row])
      const stored = (db.prepare('SELECT payload FROM quality_result WHERE session_id = ?').get(Number(id)) as { payload: string }).payload
      const listed = await st.listQuality(id, model.id)
      expect(listed).toHaveLength(1)
      expect(mark(listed[0])).toMatchObject({ ok: expected !== 'incoherent', classification: expected })
      const displayed = getSession(db, Number(id))!.candidates[0].quality
      expect(displayed).toHaveLength(1)
      expect(mark(displayed[0])).toMatchObject({ classification: expected })
      expect(mark(sessionInputs(db, Number(id))!.inputs[0].quality[0])).toMatchObject({ classification: expected })
      expect((db.prepare('SELECT payload FROM quality_result WHERE session_id = ?').get(Number(id)) as { payload: string }).payload).toBe(stored)
    }
    db.close()
  })

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
    const req = { workload: 'fast_assistant' as const, modelIds: [model.id], qualityMode: 'quick' as const } // v1 rows below
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

  it.each(['b1-first', 'b2-first'] as const)('getSession and resume reject incomplete quality subsets from mixed runtime builds (%s)', async (order) => {
    const db = openDb(join(dir, `quality-build-${order}.db`))
    const model = { id: 'E:/m/quality.gguf', name: 'quality' } as ModelMeta
    const config = { id: `${model.id}|ngl=all|kv=f16|t=8`, modelId: model.id, backend: 'vulkan' } as CandidateConfig
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [{ config, model }] }))
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: [model.id], qualityMode: 'quick' }, startedAt: 0 })
    const run = { configId: config.id, ctx: 2048, status: 'pass', versions: { runtime: 'vulkan:b1' } } as unknown as BenchmarkRunResult
    await st.saveRun(id, run, { samples: [], reason: null, stderrTail: [], load: null, startedAt: 0, endedAt: 1 })
    const tests = suiteFor('quick', 0).tests
    const rows = tests.map((t, i) => ({ testId: t.id, category: t.category, weight: t.weight, pass: true, score: 1, detail: '',
      genId: 'off', sample: 1, suite: 'qb-1.1.0', suiteSeed: null, backend: 'vulkan', runtimeVersion: `vulkan:${(i + (order === 'b1-first' ? 0 : 1)) % 2 ? 'b2' : 'b1'}` }))
    await st.saveQuality(id, model.id, config.id, 2048, rows)
    expect(await st.listQuality(id, model.id)).toEqual([])
    expect(getSession(db, Number(id))!.candidates[0].quality).toEqual([])
    expect(sessionInputs(db, Number(id))!.inputs[0].quality).toEqual([])
    db.close()
  })

  it('getSession and resume retain a complete quick suite from one runtime build', async () => {
    const db = openDb(join(dir, 'quality-build-complete.db'))
    const model = { id: 'E:/m/quality-complete.gguf', name: 'quality' } as ModelMeta
    const config = { id: `${model.id}|ngl=all|kv=f16|t=8`, modelId: model.id, backend: 'vulkan' } as CandidateConfig
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [{ config, model }] }))
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: [model.id], qualityMode: 'quick' }, startedAt: 0 })
    await st.saveRun(id, { configId: config.id, ctx: 2048, status: 'pass', versions: { runtime: 'vulkan:b1' } } as unknown as BenchmarkRunResult,
      { samples: [], reason: null, stderrTail: [], load: null, startedAt: 0, endedAt: 1 })
    const rows = suiteFor('quick', 0).tests.map((t) => ({ testId: t.id, category: t.category, weight: t.weight, pass: true, score: 1, detail: '',
      genId: 'off', sample: 1, suite: 'qb-1.1.0', suiteSeed: null, backend: 'vulkan', runtimeVersion: 'vulkan:b1' }))
    await st.saveQuality(id, model.id, config.id, 2048, rows)
    expect(await st.listQuality(id, model.id)).toHaveLength(rows.length)
    expect(getSession(db, Number(id))!.candidates[0].quality).toHaveLength(rows.length)
    expect(sessionInputs(db, Number(id))!.inputs[0].quality).toHaveLength(rows.length)
    db.close()
  })

  it('stored T1 rows with duplicate item/sample do not stand in for samples 2 and 3', async () => {
    const db = openDb(join(dir, 'quality-duplicate-sample.db'))
    const model = { id: 'E:/m/t1.gguf', name: 't1', supportsThinking: true, genKnobs: { supportsThinking: true } } as ModelMeta
    const config = { id: `${model.id}|ngl=all|kv=f16|t=8`, modelId: model.id, backend: 'vulkan' } as CandidateConfig
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [{ config, model }] }))
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: [model.id], qualitySeed: 7 }, startedAt: 0 })
    await st.saveRun(id, { configId: config.id, ctx: 2048, status: 'pass', versions: { runtime: 'vulkan:b1' } } as unknown as BenchmarkRunResult,
      { samples: [], reason: null, stderrTail: [], load: null, startedAt: 0, endedAt: 1 })
    const suite = suiteFor(undefined, 7)
    const rows = suite.tests.flatMap((t) => [
      { testId: t.id, category: t.category, weight: t.weight, pass: true, score: 1, detail: '', genId: 'off', sample: 1,
        suite: suite.suite, suiteSeed: suite.suiteSeed, backend: 'vulkan', runtimeVersion: 'vulkan:b1' },
      ...[1, 2, 3].map((sample) => ({ testId: t.id, category: t.category, weight: t.weight, pass: true, score: 1, detail: '',
        genId: 'think-t1', sample: t.id === suite.tests[0].id ? 1 : sample, suite: suite.suite, suiteSeed: suite.suiteSeed,
        backend: 'vulkan', runtimeVersion: 'vulkan:b1' }))
    ])
    await st.saveQuality(id, model.id, config.id, 2048, rows)
    expect(await st.listQuality(id, model.id)).toEqual([])
    expect(getSession(db, Number(id))!.candidates[0].quality).toEqual([])
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
    const id = await st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: ['m'], qualityMode: 'quick' }, startedAt: 0 })
    const q = (t: string) => ({ testId: t, category: 'coding' as const, weight: 1, pass: true, score: 1, detail: '' })
    await st.saveQuality(id, 'm', 'c', 2048, [q('a'), q('b')])
    expect(await st.listQuality(id, 'm')).toHaveLength(2)
    db.prepare('DELETE FROM quality_result WHERE id = (SELECT max(id) FROM quality_result)').run() // now incomplete
    expect(await st.listQuality(id, 'm')).toEqual([]) // resume re-runs the suite
    db.close()
  })

  it('v2 quality: the runner\'s suite id is kept, and resume reuses it only for the session\'s own suite + seed', async () => {
    const db = openDb(join(dir, 'v2.db'))
    const st = makeSessionStorage(db, () => ({ vramBytes: null, candidates: [] }))
    const v2 = (t: string, suiteSeed: number) => ({ testId: t, category: 'coding' as const, weight: 1, pass: true, score: 1, detail: '', suite: 'qb-2.0.0', suiteSeed, instanceSeed: 1, generatorVersion: 'qbg-2.0.0' })
    const mk = (request: object) => st.createSession({ workload: 'coding', request: { workload: 'coding', modelIds: ['m'], ...request } as never, startedAt: 0 })
    const id = await mk({ qualitySeed: 7 }) // unset mode = thorough (v2); the runner records its seed
    await st.saveQuality(id, 'm', 'c', 2048, [v2('a', 7), v2('b', 7)])
    const rows = await st.listQuality(id, 'm')
    expect(rows.map((r) => (r as { suite?: string }).suite)).toEqual(['qb-2.0.0', 'qb-2.0.0']) // not overwritten with qb-1.1.0
    const other = await mk({ qualitySeed: 8 }) // same suite, different seed → different items → re-run
    await st.saveQuality(other, 'm', 'c', 2048, [v2('a', 7), v2('b', 7)])
    expect(await st.listQuality(other, 'm')).toEqual([])
    const quick = await mk({ qualityMode: 'quick' }) // v1 selected: v2 rows don't count
    await st.saveQuality(quick, 'm', 'c', 2048, [v2('a', 7)])
    expect(await st.listQuality(quick, 'm')).toEqual([])
    const noSeed = await mk({}) // v2 without a recorded seed can't be matched
    await st.saveQuality(noSeed, 'm', 'c', 2048, [v2('a', 7)])
    expect(await st.listQuality(noSeed, 'm')).toEqual([])
    db.close()
  })

  it('relabelV2Quality repairs v2 rows stored as qb-1.1.0 by older builds; v1 rows untouched; idempotent', () => {
    const db = openDb(join(dir, 'relabel.db'))
    const sid = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'running')
    const put = (p: object) => db.prepare('INSERT INTO quality_result (session_id, model_id, payload) VALUES (?, ?, ?)').run(sid, 'm', JSON.stringify(p))
    put({ testId: 'a', suite: 'qb-1.1.0', suiteSeed: 7, instanceSeed: 3, generatorVersion: 'qbg-2.0.0' }) // mislabelled v2
    put({ testId: 'b', suite: 'qb-1.1.0', suiteSeed: null, instanceSeed: 4, generatorVersion: 'qbg-2.0.0' }) // v2, seed only per item
    put({ testId: 'c', suite: 'qb-1.1.0' }) // genuine v1
    put({ testId: 'd', suite: 'qb-1.1.0', generatorVersion: 'qbg-2.0.0' }) // no seed at all: not provably v2 → left
    expect(relabelV2Quality(db)).toBe(2)
    const suites = (db.prepare('SELECT payload FROM quality_result ORDER BY id').all() as { payload: string }[]).map((r) => JSON.parse(r.payload).suite)
    expect(suites).toEqual(['qb-2.0.0', 'qb-2.0.0', 'qb-1.1.0', 'qb-1.1.0'])
    expect(relabelV2Quality(db)).toBe(0)
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
