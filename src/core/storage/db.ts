import { DatabaseSync } from 'node:sqlite'

// node:sqlite verified usable in Electron 44 main (Node 24.21) and in system Node for tests — no native module.
// Tables are deliberately thin (json payload + a few indexed scalars) until the data design lands.

const MIGRATIONS: string[] = [
  `CREATE TABLE machine_profile (
     id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime('now')), payload TEXT NOT NULL);
   CREATE TABLE runtime (
     id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime('now')), runtime_id TEXT NOT NULL, version TEXT, payload TEXT NOT NULL);
   CREATE TABLE model (
     id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime('now')), path TEXT NOT NULL, size_bytes INTEGER, payload TEXT NOT NULL);
   CREATE TABLE benchmark_session (
     id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT (datetime('now')), status TEXT NOT NULL, payload TEXT NOT NULL);
   CREATE TABLE benchmark_run (
     id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES benchmark_session(id), created_at TEXT NOT NULL DEFAULT (datetime('now')),
     status TEXT NOT NULL, model_id TEXT, ctx_size INTEGER, payload TEXT NOT NULL);
   CREATE TABLE telemetry_sample (
     id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES benchmark_session(id), run_id INTEGER REFERENCES benchmark_run(id),
     created_at TEXT NOT NULL DEFAULT (datetime('now')), payload TEXT NOT NULL);
   CREATE TABLE quality_result (
     id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES benchmark_session(id), run_id INTEGER REFERENCES benchmark_run(id),
     created_at TEXT NOT NULL DEFAULT (datetime('now')), model_id TEXT, payload TEXT NOT NULL);
   CREATE TABLE recommendation (
     id INTEGER PRIMARY KEY, session_id INTEGER REFERENCES benchmark_session(id), created_at TEXT NOT NULL DEFAULT (datetime('now')),
     model_id TEXT, payload TEXT NOT NULL);
   CREATE INDEX benchmark_run_session ON benchmark_run(session_id);
   CREATE INDEX benchmark_run_model ON benchmark_run(model_id, ctx_size);
   CREATE INDEX telemetry_sample_run ON telemetry_sample(session_id, run_id);
   CREATE INDEX quality_result_run ON quality_result(session_id, run_id);
   CREATE INDEX recommendation_session ON recommendation(session_id);`
]

/** Open (creating if needed) and bring the schema up to date. Caller owns close(). */
export function openDb(path: string): DatabaseSync {
  const db = new DatabaseSync(path)
  try {
    db.exec('PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;')
    migrate(db)
    relabelV2Quality(db)
  } catch (e) {
    db.close() // don't leave the file locked when the schema is unusable
    throw e
  }
  return db
}

/** Bulk-insert telemetry samples for one run in a single transaction. Sample objects go in `payload` as JSON. */
export function insertTelemetrySamples(db: DatabaseSync, sessionId: number, runId: number, samples: { ts: number }[]): void {
  const stmt = db.prepare('INSERT INTO telemetry_sample (session_id, run_id, created_at, payload) VALUES (?, ?, ?, ?)')
  db.exec('BEGIN')
  try {
    for (const s of samples) stmt.run(sessionId, runId, new Date(s.ts).toISOString(), JSON.stringify(s))
    db.exec('COMMIT')
  } catch (e) {
    db.exec('ROLLBACK')
    throw e
  }
}

/** Data repair, idempotent (runs on every open, no schema bump so older builds can still open the file): builds before
 *  the fix stored every quality row as suite 'qb-1.1.0', so v2 rows (they carry generatorVersion qbg-2.0.0 and a seed)
 *  were mislabelled. Relabels them 'qb-2.0.0'; a process still running the old code is repaired on the next open.
 *  Returns the number of rows changed. */
export function relabelV2Quality(db: DatabaseSync): number {
  return Number(db.prepare(`UPDATE quality_result SET payload = json_set(payload, '$.suite', 'qb-2.0.0')
    WHERE json_valid(payload) AND json_extract(payload, '$.generatorVersion') = 'qbg-2.0.0'
      AND (json_extract(payload, '$.suiteSeed') IS NOT NULL OR json_extract(payload, '$.instanceSeed') IS NOT NULL)
      AND json_extract(payload, '$.suite') IS NOT 'qb-2.0.0'`).run().changes)
}

export function migrate(db: DatabaseSync): number {
  db.exec('CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)')
  const row = db.prepare('SELECT max(version) AS v FROM schema_version').get() as { v: number | null }
  let v = row.v ?? 0
  if (v > MIGRATIONS.length) throw new Error(`database schema v${v} is newer than this build (v${MIGRATIONS.length}); update the app`)
  for (; v < MIGRATIONS.length; v++) {
    db.exec('BEGIN')
    try {
      db.exec(MIGRATIONS[v])
      db.prepare('INSERT INTO schema_version (version) VALUES (?)').run(v + 1)
      db.exec('COMMIT')
    } catch (e) {
      db.exec('ROLLBACK')
      throw e
    }
  }
  return v
}
