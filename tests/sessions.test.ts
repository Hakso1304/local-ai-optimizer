import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { openDb } from '../src/core/storage/db'
import { getSession, latestRecommendation, listSessions, saveRecommendation, saveSession, seedDemoSession, telemetryForRun } from '../src/core/storage/sessions'
import { insertTelemetrySamples } from '../src/core/storage/db'
import type { Recommendation } from '../src/shared/bench-types'

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
})
