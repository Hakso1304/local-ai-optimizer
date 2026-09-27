import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { insertTelemetrySamples, openDb } from '../src/core/storage/db'

describe('storage', () => {
  it('open -> migrate -> insert -> reopen -> read', () => {
    const dir = mkdtempSync(join(tmpdir(), 'lao-db-'))
    const path = join(dir, 'optimizer.db')
    try {
      let db = openDb(path)
      const s = db.prepare("INSERT INTO benchmark_session (status, payload) VALUES ('done', '{}')").run()
      db.prepare('INSERT INTO benchmark_run (session_id, status, model_id, ctx_size, payload) VALUES (?, ?, ?, ?, ?)')
        .run(s.lastInsertRowid, 'ok', 'qwen', 2048, JSON.stringify({ decodeTps: 336.5 }))
      db.close()

      db = openDb(path) // second open must not re-run migration 1
      const row = db.prepare('SELECT model_id, ctx_size, payload FROM benchmark_run').get() as { model_id: string; ctx_size: number; payload: string }
      expect(row).toMatchObject({ model_id: 'qwen', ctx_size: 2048 })
      expect(JSON.parse(row.payload)).toEqual({ decodeTps: 336.5 })
      expect(db.prepare('SELECT count(*) AS n FROM schema_version').get()).toEqual({ n: 1 })
      db.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('bulk-inserts telemetry samples for a run and rolls back on failure', () => {
    const dir = mkdtempSync(join(tmpdir(), 'lao-db-'))
    try {
      const db = openDb(join(dir, 'optimizer.db'))
      const sid = Number(db.prepare("INSERT INTO benchmark_session (status, payload) VALUES ('running', '{}')").run().lastInsertRowid)
      const rid = Number(db.prepare("INSERT INTO benchmark_run (session_id, status, payload) VALUES (?, 'running', '{}')").run(sid).lastInsertRowid)
      const samples: { ts: number; cpuPct: number }[] = [{ ts: 1000, cpuPct: 5 }, { ts: 2000, cpuPct: 7 }]
      insertTelemetrySamples(db, sid, rid, samples)
      const rows = db.prepare('SELECT created_at, payload FROM telemetry_sample WHERE run_id = ? ORDER BY id').all(rid) as { created_at: string; payload: string }[]
      expect(rows.map((r) => JSON.parse(r.payload).cpuPct)).toEqual([5, 7])
      expect(rows[0].created_at).toBe(new Date(1000).toISOString())
      // FK violation (unknown run) -> throws, batch rolled back
      expect(() => insertTelemetrySamples(db, sid, 999, [{ ts: 3000 }])).toThrow()
      expect(db.prepare('SELECT count(*) AS n FROM telemetry_sample').get()).toEqual({ n: 2 })
      db.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('refuses a database written by a newer build instead of corrupting it', () => {
    const dir = mkdtempSync(join(tmpdir(), 'lao-db-'))
    try {
      const path = join(dir, 'optimizer.db')
      const db = openDb(path)
      db.prepare('INSERT INTO schema_version (version) VALUES (999)').run()
      db.close()
      expect(() => openDb(path)).toThrow(/newer than this build/)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
