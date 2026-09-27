// Read-only export of the persisted rows behind a run-session --db result.
// Usage: npx tsx scripts/export-session-evidence.ts --session 4 [--db path] [--head git-hash] [--out docs/file.json]
import { DatabaseSync } from 'node:sqlite'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

function flag(name: string): string | undefined {
  const i = process.argv.indexOf(name)
  return i < 0 ? undefined : process.argv[i + 1]
}

const sessionId = Number(flag('--session'))
if (!Number.isSafeInteger(sessionId) || sessionId < 1) throw new Error('--session requires a positive integer')
const dbPath = flag('--db') ?? join(process.env.APPDATA ?? '', 'local-ai-optimizer-dev', 'optimizer.db')
if (!existsSync(dbPath)) throw new Error(`database does not exist: ${dbPath}`)
const head = flag('--head') ?? (existsSync('HEAD.txt') ? readFileSync('HEAD.txt', 'utf8').trim() : null)
if (!head || !/^[0-9a-f]{40}$/i.test(head)) throw new Error('supply --head <full 40-character commit> or run from a snapshot with HEAD.txt')
const out = flag('--out') ?? join('docs', `session-evidence-${sessionId}.json`)

const db = new DatabaseSync(dbPath, { readOnly: true })
try {
  db.exec('PRAGMA query_only = ON')
  type Raw = Record<string, string | number | null>
  const session = db.prepare('SELECT id, created_at, status, payload FROM benchmark_session WHERE id = ?').get(sessionId) as Raw | undefined
  if (!session) throw new Error(`session ${sessionId} not found`)
  const rows = (sql: string) => (db.prepare(sql).all(sessionId) as Raw[]).map(({ payload, ...columns }) => ({ ...columns, payload: JSON.parse(String(payload)) }))
  // Keep every attempt, including superseded retries, rather than only the latest row per config/context.
  const benchmarkRuns = rows('SELECT id, session_id, created_at, status, model_id, ctx_size, payload FROM benchmark_run WHERE session_id = ? ORDER BY id')
  const qualityResults = rows('SELECT id, session_id, run_id, created_at, model_id, payload FROM quality_result WHERE session_id = ? ORDER BY id')
  const recommendations = rows('SELECT id, session_id, created_at, model_id, payload FROM recommendation WHERE session_id = ? ORDER BY id')
  const artifact = {
    kind: 'local-ai-optimizer/session-evidence-v1', exportedAt: new Date().toISOString(), snapshotHead: head, sessionId,
    session: { id: session.id, created_at: session.created_at, status: session.status, payload: JSON.parse(String(session.payload)) },
    counts: { benchmarkRuns: benchmarkRuns.length, qualityResults: qualityResults.length, recommendations: recommendations.length },
    benchmarkRuns, qualityResults, recommendations
  }
  // Exclusive creation preserves earlier evidence, even if another writer creates this path after the read.
  try {
    writeFileSync(out, JSON.stringify(artifact, null, 2), { flag: 'wx' })
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'EEXIST') throw new Error(`refusing to overwrite existing session evidence: ${out}`, { cause: error })
    throw error
  }
  console.log(`${out}: session ${sessionId}, ${benchmarkRuns.length} runs, ${qualityResults.length} quality rows, HEAD ${head}`)
} finally {
  db.close()
}
