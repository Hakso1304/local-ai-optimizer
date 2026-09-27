import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { DatabaseSync } from 'node:sqlite'
import type {
  BenchmarkRunResult, CandidateConfig, CandidateInput, FailureKind, Metric, ModelMeta, QualityResult, Recommendation, RunStatus, WorkloadId
} from '../../shared/bench-types'
import type { SessionDetail, SessionPayload, SessionSummary } from '../../shared/types'
import type { TelemetrySample } from '../telemetry/sampler'
import { detectCliffs } from '../scoring/cliff'
import { recommend } from '../scoring/recommend'

// Read/write side for benchmark sessions. Rows keep whole objects in `payload` (see SessionPayload).

const json = <T>(s: string): T => JSON.parse(s) as T

export function saveSession(db: DatabaseSync, payload: SessionPayload, status: string): number {
  return Number(db.prepare('INSERT INTO benchmark_session (status, payload) VALUES (?, ?)').run(status, JSON.stringify(payload)).lastInsertRowid)
}

export function setSessionStatus(db: DatabaseSync, id: number, status: string): void {
  db.prepare('UPDATE benchmark_session SET status = ? WHERE id = ?').run(status, id)
}

export function saveRun(db: DatabaseSync, sessionId: number, modelId: string, run: BenchmarkRunResult): number {
  return Number(db.prepare('INSERT INTO benchmark_run (session_id, status, model_id, ctx_size, payload) VALUES (?, ?, ?, ?, ?)')
    .run(sessionId, run.status, modelId, run.ctx, JSON.stringify(run)).lastInsertRowid)
}

export function saveQualityResults(db: DatabaseSync, sessionId: number, modelId: string, results: QualityResult[]): void {
  const stmt = db.prepare('INSERT INTO quality_result (session_id, model_id, payload) VALUES (?, ?, ?)')
  for (const r of results) stmt.run(sessionId, modelId, JSON.stringify(r))
}

export function saveRecommendation(db: DatabaseSync, sessionId: number, rec: Recommendation, modelId: string | null): number {
  return Number(db.prepare('INSERT INTO recommendation (session_id, model_id, payload) VALUES (?, ?, ?)')
    .run(sessionId, modelId, JSON.stringify(rec)).lastInsertRowid)
}

interface SessionRow { id: number; created_at: string; status: string; payload: string }

function summary(db: DatabaseSync, row: SessionRow): SessionSummary {
  const p = json<SessionPayload>(row.payload)
  const rec = db.prepare('SELECT payload FROM recommendation WHERE session_id = ? ORDER BY id DESC LIMIT 1').get(row.id) as { payload: string } | undefined
  return {
    id: row.id,
    createdAt: row.created_at,
    status: row.status,
    workload: p.workload,
    demo: p.demo === true || row.status === 'demo',
    label: p.label ?? null,
    candidateCount: p.candidates.length,
    bestConfigId: rec ? json<Recommendation>(rec.payload).best?.configId ?? null : null
  }
}

export function listSessions(db: DatabaseSync): SessionSummary[] {
  return (db.prepare('SELECT id, created_at, status, payload FROM benchmark_session ORDER BY id DESC').all() as unknown as SessionRow[]).map((r) => summary(db, r))
}

export function getSession(db: DatabaseSync, id: number): SessionDetail | null {
  const row = db.prepare('SELECT id, created_at, status, payload FROM benchmark_session WHERE id = ?').get(id) as SessionRow | undefined
  if (!row) return null
  const p = json<SessionPayload>(row.payload)
  const runs = (db.prepare('SELECT payload FROM benchmark_run WHERE session_id = ? ORDER BY id').all(id) as { payload: string }[]).map((r) => json<BenchmarkRunResult>(r.payload))
  const quality = db.prepare('SELECT model_id, payload FROM quality_result WHERE session_id = ? ORDER BY id').all(id) as { model_id: string; payload: string }[]
  const recRow = db.prepare('SELECT payload FROM recommendation WHERE session_id = ? ORDER BY id DESC LIMIT 1').get(id) as { payload: string } | undefined
  const recommendation = recRow ? json<Recommendation>(recRow.payload) : null
  return {
    session: summary(db, row),
    recommendation,
    candidates: p.candidates.map(({ config, model }) => {
      const mine = runs.filter((r) => r.configId === config.id)
      return {
        config, model, runs: mine,
        cliff: detectCliffs(mine, p.vramBytes),
        score: recommendation?.ranked.find((s) => s.configId === config.id) ?? null,
        quality: quality.filter((q) => q.model_id === model.id).map((q) => json<QualityResult>(q.payload))
      }
    })
  }
}

/** Newest recommendation for a workload from a REAL session (demo sessions are excluded). */
export function latestRecommendation(db: DatabaseSync, workload: WorkloadId): { sessionId: number; recommendation: Recommendation } | null {
  const row = db.prepare(`
    SELECT r.session_id, r.payload FROM recommendation r JOIN benchmark_session s ON s.id = r.session_id
    WHERE s.status <> 'demo' AND json_extract(s.payload, '$.demo') IS NOT 1 AND json_extract(r.payload, '$.workload') = ?
    ORDER BY r.id DESC LIMIT 1`).get(workload) as { session_id: number; payload: string } | undefined
  return row ? { sessionId: row.session_id, recommendation: json<Recommendation>(row.payload) } : null
}

export function telemetryForRun(db: DatabaseSync, runId: number): TelemetrySample[] {
  return (db.prepare('SELECT payload FROM telemetry_sample WHERE run_id = ? ORDER BY id').all(runId) as { payload: string }[]).map((r) => json<TelemetrySample>(r.payload))
}

// ---- DEMO seed (dev only, LAO_SEED_DEMO=1). Built from tests/fixtures/scoring; every label says DEMO. ----

interface FixtureRun {
  configId: string; model: string; ctx: number; gpuLayers: number; threads: number; status: string; promptTokens: number | null
  loadMs: number | null; ttftMs: number | null; prefillTps: number | null; decodeTps: number | null; totalMs: number | null
  peakRamBytes: number | null; peakVramBytes: number | null; peakSharedGpuBytes: number | null; cpuAvgPct: number | null; gpuAvgPct: number | null
}
interface Fixture { vramBytes: number; models: ModelMeta[]; runs: FixtureRun[] }

const STATUS: Record<string, [RunStatus, FailureKind | null]> = {
  ok: ['pass', null], failed: ['fail', 'exit_1'], oom: ['fail', 'oom'], device_lost: ['fail', 'device_lost'],
  crashed: ['fail', 'crash'], timeout: ['timeout', 'req_timeout'], cancelled: ['cancelled', null]
}

function demoCandidate(f: Fixture, configId: string, kv: 'f16' | 'q8_0'): CandidateInput {
  const m = (v: number | null): Metric => (v === null ? { value: null, kind: 'unavailable', reason: 'DEMO fixture' } : { value: v, kind: 'measured', source: 'DEMO fixture' })
  const model = f.models[0]
  const rows = f.runs
  const config: CandidateConfig = {
    id: configId, modelId: model.id, device: 'Vulkan0', gpuLayers: rows[0].gpuLayers, gpuLayersAll: rows[0].gpuLayers >= model.layers,
    kvType: kv, flashAttn: true, threads: rows[0].threads, ctxSteps: rows.map((r) => r.ctx), skippedSteps: [],
    estVramBytes: { value: null, kind: 'unavailable', reason: 'DEMO' }, estRamBytes: { value: null, kind: 'unavailable', reason: 'DEMO' }, notes: ['DEMO DATA']
  }
  const runs = rows.map((r): BenchmarkRunResult => ({
    configId, ctx: r.ctx, promptTokens: r.promptTokens, status: STATUS[r.status][0], failureKind: STATUS[r.status][1],
    loadTimeMs: m(r.loadMs), ttftMs: m(r.ttftMs), prefillTps: m(r.prefillTps), decodeTps: m(r.decodeTps), totalMs: m(r.totalMs),
    peakVramBytes: m(r.peakVramBytes), peakSharedGpuBytes: m(r.peakSharedGpuBytes), peakRamBytes: m(r.peakRamBytes),
    avgGpuUtil: m(r.gpuAvgPct), avgCpuUtil: m(r.cpuAvgPct)
  }))
  return { config, model: { ...model, name: `DEMO ${model.name}` }, runs, quality: [] }
}

/** Insert one DEMO session (2 candidates, one with a 16K→32K cliff) unless one exists. Returns its id. */
export function seedDemoSession(db: DatabaseSync, fixturesDir: string): number {
  const existing = db.prepare("SELECT id FROM benchmark_session WHERE status = 'demo' LIMIT 1").get() as { id: number } | undefined
  if (existing) return existing.id
  const load = (n: string) => json<Fixture>(readFileSync(join(fixturesDir, n), 'utf8'))
  const cliffFx = load('sweep-cliff-16k-32k.json')
  const inputs = [
    demoCandidate(cliffFx, 'DEMO|llama31-8b-q4km|ngl=all|kv=f16', 'f16'),
    demoCandidate(load('sweep-smooth.json'), 'DEMO|llama31-8b-q4km|ngl=all|kv=q8_0', 'q8_0')
  ]
  const workload: WorkloadId = 'long_context_coding'
  const GiB = 1024 ** 3
  const rec = recommend(inputs, {
    vramBytes: { value: cliffFx.vramBytes, kind: 'declared', source: 'DEMO' }, vramInUseBytes: { value: 0, kind: 'measured', source: 'DEMO' },
    ramTotalBytes: { value: 32 * GiB, kind: 'declared', source: 'DEMO' }, ramAvailableBytes: { value: 20 * GiB, kind: 'measured', source: 'DEMO' },
    physicalCores: 8, gpuDevice: 'Vulkan0'
  }, workload)
  rec.reasons.unshift('DEMO DATA — generated from test fixtures, not measured on this machine')
  db.exec('BEGIN')
  try {
    const id = saveSession(db, {
      workload, demo: true, label: 'DEMO DATA (fixtures)', vramBytes: cliffFx.vramBytes,
      candidates: inputs.map(({ config, model }) => ({ config, model }))
    }, 'demo')
    for (const c of inputs) for (const r of c.runs) saveRun(db, id, c.model.id, r)
    saveRecommendation(db, id, rec, rec.best ? inputs.find((c) => c.config.id === rec.best!.configId)!.model.id : null)
    db.exec('COMMIT')
    return id
  } catch (e) {
    db.exec('ROLLBACK')
    throw e
  }
}
