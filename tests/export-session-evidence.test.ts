// CLI integration against a temporary SQLite file. Child runs Node/tsx only; no app, vendor runtime or device probe.
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { openDb } from '../src/core/storage/db'
import { saveQualityResults, saveRecommendation, saveRun, saveSession } from '../src/core/storage/sessions'
import type { BenchmarkRunResult, QualityResult, Recommendation } from '../src/shared/bench-types'

const dir = mkdtempSync(join(tmpdir(), 'lao-export-evidence-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))
const root = resolve(__dirname, '..')
const cli = join(root, 'node_modules', 'tsx', 'dist', 'cli.mjs')
const script = join(root, 'scripts', 'export-session-evidence.ts')
const head = 'a'.repeat(40)

function fixture() {
  const path = join(dir, 'session.db')
  const db = openDb(path)
  const id = saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'done')
  const one = saveRun(db, id, 'model', { configId: 'model|ngl=1', ctx: 2048, status: 'fail', failureKind: 'oom' } as BenchmarkRunResult)
  const two = saveRun(db, id, 'model', { configId: 'model|ngl=1', ctx: 2048, status: 'pass' } as BenchmarkRunResult)
  saveQualityResults(db, id, 'model', [{ testId: 'IF2-01', category: 'instruction', weight: 1, pass: true, score: 1, detail: '', skillId: 'format' } as QualityResult])
  saveRecommendation(db, id, { workload: 'coding', best: null, reasons: ['fixture'] } as Recommendation, 'model')
  db.close()
  return { path, id, runIds: [one, two] }
}

function exportCli(db: string, id: number, out: string) {
  return spawnSync(process.execPath, [cli, script, '--session', String(id), '--db', db, '--head', head, '--out', out],
    { cwd: root, encoding: 'utf8', windowsHide: true, timeout: 10_000 })
}

describe('persisted session evidence exporter (Node-only CLI)', () => {
  it('refuses an existing output and preserves every sentinel byte', () => {
    const { path, id } = fixture()
    const out = join(dir, 'sentinel.json')
    const sentinel = Buffer.from('KEEP ORIGINAL EVIDENCE\r\n\u0000\xff', 'latin1')
    writeFileSync(out, sentinel)
    const result = exportCli(path, id, out)
    expect(result.status).not.toBe(0)
    expect(`${result.stdout}\n${result.stderr}`).toMatch(/exist|overwrite|refus|already/i)
    expect(readFileSync(out)).toEqual(sentinel)
  })

  it('exports every attempted run plus quality and recommendation into a fresh file', () => {
    const { path, id, runIds } = fixture()
    const out = join(dir, 'fresh.json')
    const result = exportCli(path, id, out)
    expect(result.status).toBe(0)
    const artifact = JSON.parse(readFileSync(out, 'utf8'))
    expect(artifact).toMatchObject({ kind: 'local-ai-optimizer/session-evidence-v1', snapshotHead: head, sessionId: id,
      counts: { benchmarkRuns: 2, qualityResults: 1, recommendations: 1 } })
    expect(artifact.benchmarkRuns.map((r: { id: number }) => r.id)).toEqual(runIds)
    expect(artifact.benchmarkRuns.map((r: { payload: { status: string } }) => r.payload.status)).toEqual(['fail', 'pass'])
    expect(artifact.qualityResults[0].payload).toMatchObject({ testId: 'IF2-01', skillId: 'format' })
    expect(artifact.recommendations[0].payload.reasons).toEqual(['fixture'])
  })
})
