import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { openDb } from '../src/core/storage/db'

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
})
