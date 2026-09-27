import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { DatabaseSync } from 'node:sqlite'
import type {
  BenchmarkRunResult, CandidateConfig, CandidateInput, FailureKind, GenQuality, Metric, ModelMeta, QualityResult, Recommendation, RunStatus, WorkloadId
} from '../../shared/bench-types'
import type { SessionDetail, SessionPayload, SessionSummary } from '../../shared/types'
import type { SessionRequest } from '../../shared/bench-events'
import type { RunDetail, SessionStorage } from '../benchmark/session'
import type { TelemetrySample } from '../telemetry/sampler'
import { insertTelemetrySamples } from './db'
import { defaultTestSet, qualityScore } from '../quality'
import { BASELINE_GEN, genConfigsFor, summarizeGen, type GenRow } from '../benchmark/gen'
import type { StoredRun } from '../interpret/verdicts'
import { detectCliffs } from '../scoring/cliff'
import { recommend } from '../scoring/recommend'

// Read/write side for benchmark sessions. Rows keep whole objects in `payload` (see SessionPayload).

const json = <T>(s: string): T => JSON.parse(s) as T

export function saveSession(db: DatabaseSync, payload: SessionPayload, status: string): number {
  return Number(db.prepare('INSERT INTO benchmark_session (status, payload) VALUES (?, ?)').run(status, JSON.stringify(payload)).lastInsertRowid)
}

export function setSessionStatus(db: DatabaseSync, id: number, status: string, error?: string): void {
  if (error === undefined) db.prepare('UPDATE benchmark_session SET status = ? WHERE id = ?').run(status, id)
  else db.prepare("UPDATE benchmark_session SET status = ?, payload = json_set(payload, '$.error', ?) WHERE id = ?").run(status, error, id)
}

/** Stored quality rows (tagged genId/sample) → one GenQuality per generation config. GenConfig ids are deterministic
 *  (genConfigsFor(model, request)), so the configs are rebuilt rather than stored; unknown ids fall back to a
 *  minimal config. Rows without genId are the baseline ('off'). */
export function genQualityOf(model: ModelMeta, request: SessionRequest | null, rows: GenRow[]): (GenQuality & { qualityScore: number | null })[] {
  const known = new Map([BASELINE_GEN, ...genConfigsFor(model, request ?? {})].map((g) => [g.id, g] as const))
  const byGen = new Map<string, GenRow[]>()
  for (const r of rows) { const id = r.genId ?? BASELINE_GEN.id; byGen.set(id, [...(byGen.get(id) ?? []), r]) }
  return [...byGen].map(([id, rs]) => {
    const gen = known.get(id) ?? { id, thinking: id !== BASELINE_GEN.id, temperature: 0, source: 'default' as const }
    const samples = Math.max(1, ...rs.map((r) => r.sample ?? 1))
    return { ...summarizeGen(gen, rs, samples), qualityScore: qualityScore(rs) }
  })
}

/** Latest row per (configId, ctx): a retry/rerun inserts a new row that supersedes the old one. */
const LATEST_RUNS = `SELECT id, payload FROM benchmark_run WHERE id IN (
  SELECT max(id) FROM benchmark_run WHERE session_id = ? GROUP BY json_extract(payload, '$.configId'), ctx_size) ORDER BY id`

/** What resume needs: the original request and the scan the plan was made from (so configIds come out identical). */
export function getSessionResume(db: DatabaseSync, id: number): { request: SessionRequest; machine: SessionPayload['machine']; plan: CandidateConfig[] } | null {
  const row = db.prepare('SELECT payload FROM benchmark_session WHERE id = ?').get(id) as { payload: string } | undefined
  const p = row ? json<SessionPayload>(row.payload) : null
  return p?.request ? { request: p.request, machine: p.machine, plan: p.candidates.map((c) => c.config) } : null
}

/** App start: sessions still 'running' belong to a previous app run that quit mid-session. Returns how many. */
export function markInterrupted(db: DatabaseSync): number {
  return Number(db.prepare("UPDATE benchmark_session SET status = 'interrupted' WHERE status = 'running'").run().changes)
}

export function saveRun(db: DatabaseSync, sessionId: number, modelId: string, run: BenchmarkRunResult): number {
  return Number(db.prepare('INSERT INTO benchmark_run (session_id, status, model_id, ctx_size, payload) VALUES (?, ?, ?, ?, ?)')
    .run(sessionId, run.status, modelId, run.ctx, JSON.stringify(run)).lastInsertRowid)
}

/** One suite = one transaction: a failure midway leaves no partial suite behind (W4 F5). */
export function saveQualityResults(db: DatabaseSync, sessionId: number, modelId: string, results: QualityResult[]): void {
  const stmt = db.prepare('INSERT INTO quality_result (session_id, model_id, payload) VALUES (?, ?, ?)')
  db.exec('BEGIN')
  try {
    for (const r of results) stmt.run(sessionId, modelId, JSON.stringify(r))
    db.exec('COMMIT')
  } catch (e) {
    db.exec('ROLLBACK')
    throw e
  }
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
    error: p.error ?? null,
    requiredContext: p.request?.requiredContext ?? null,
    minDecodeTps: p.request?.minDecodeTps ?? null,
    heavyMode: p.request?.heavyMode === true,
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
  const rows = (db.prepare(LATEST_RUNS).all(id) as { id: number; payload: string }[]).map((r) => {
    const { detail: _d, ...run } = json<BenchmarkRunResult & { detail?: unknown }>(r.payload)
    return { rowId: r.id, run }
  })
  // Every persisted row, oldest first: a retried/rerun step supersedes its earlier row(s), which stay visible as
  // history (a device loss or guard abort in a superseded attempt must still surface; W4f / I-6.3).
  const all = (db.prepare('SELECT id, created_at, payload FROM benchmark_run WHERE session_id = ? ORDER BY id').all(id) as { id: number; created_at: string; payload: string }[])
    .map((r) => {
      const p = json<BenchmarkRunResult & { detail?: { samplerErrors?: string[]; startedAt?: number; endedAt?: number } }>(r.payload)
      const { detail, ...run } = p
      return { ...run, rowId: r.id, recordedAt: r.created_at, supersededBy: null as number | null, samplerErrors: detail?.samplerErrors ?? [], startedAt: detail?.startedAt ?? null, endedAt: detail?.endedAt ?? null }
    })
  const latest = new Map<string, number>()
  for (const r of all) latest.set(`${r.configId}@${r.ctx}`, r.rowId)
  for (const r of all) { const l = latest.get(`${r.configId}@${r.ctx}`)!; if (l !== r.rowId) r.supersededBy = l }
  const quality = db.prepare('SELECT model_id, payload FROM quality_result WHERE session_id = ? ORDER BY id').all(id) as { model_id: string; payload: string }[]
  const recRow = db.prepare('SELECT payload FROM recommendation WHERE session_id = ? ORDER BY id DESC LIMIT 1').get(id) as { payload: string } | undefined
  const recommendation = recRow ? json<Recommendation>(recRow.payload) : null
  return {
    session: summary(db, row),
    recommendation,
    candidates: p.candidates.map(({ config, model }) => {
      const mine = rows.filter((r) => r.run.configId === config.id)
      return {
        config, model, runs: mine.map((r) => r.run), runIds: mine.map((r) => r.rowId),
        history: all.filter((r) => r.configId === config.id),
        cliff: detectCliffs(mine.map((r) => r.run), p.vramBytes),
        score: recommendation?.ranked.find((s) => s.configId === config.id) ?? null,
        quality: quality.filter((q) => q.model_id === model.id).map((q) => json<QualityResult>(q.payload)),
        genQuality: genQualityOf(model, p.request ?? null, quality.filter((q) => q.model_id === model.id).map((q) => json<GenRow>(q.payload)))
      }
    })
  }
}

/** Scoring inputs of a stored session (latest row per step, quality per model) + the scan it was planned with,
 *  for re-scoring the same measurements under another workload. */
export function sessionInputs(db: DatabaseSync, id: number): {
  inputs: CandidateInput[]; machine: SessionPayload['machine']; request: SessionRequest | null
  /** Engine inputs (interp-2 InterpretData): every attempt, session planning values, why the session stopped. */
  allRuns: StoredRun[]
  planningSnapshot: { ramFloorBytes?: number; mmapCreditBytes?: number }
  stopReason?: 'done' | 'cancelled' | 'paused' | 'interrupted' | 'user-cap'
} | null {
  const d = getSession(db, id)
  const row = db.prepare('SELECT payload FROM benchmark_session WHERE id = ?').get(id) as { payload: string } | undefined
  if (!d || !row) return null
  const p = json<SessionPayload>(row.payload)
  const history = d.candidates.flatMap((c) => c.history)
  const allRuns: StoredRun[] = history.map(({ rowId, supersededBy, recordedAt: _r, samplerErrors: _s, startedAt, endedAt, ...run }) => ({
    ...run, runId: String(rowId), ...(supersededBy !== null ? { supersededBy: String(supersededBy) } : {}),
    ...(startedAt !== null ? { startedAt } : {}), ...(endedAt !== null ? { endedAt } : {})
  }))
  // Session-level planning values from the runs that recorded them (§12); absent = the engine's defaults.
  const nums = (k: 'ramFloorBytes' | 'mmapCreditBytes') => history.map((r) => (r as unknown as Record<string, unknown>)[k]).filter((v): v is number => typeof v === 'number')
  const floor = nums('ramFloorBytes'), mmap = nums('mmapCreditBytes')
  const planningSnapshot = { ...(floor.length ? { ramFloorBytes: Math.max(...floor) } : {}), ...(mmap.length ? { mmapCreditBytes: Math.min(...mmap) } : {}) }
  // A finished session whose ladder the user capped stopped by choice, not at a limit.
  const st = d.session.status
  const stopReason = st === 'done' ? (p.request?.ladder?.length ? 'user-cap' as const : 'done' as const)
    : st === 'cancelled' || st === 'paused' || st === 'interrupted' ? st : undefined
  return {
    // Baseline rows feed `quality`; every gen config (baseline included) feeds genQuality, like the runner does.
    inputs: d.candidates.map((c) => ({
      config: c.config, model: c.model, runs: c.runs, history: c.history,
      quality: c.quality.filter((q) => ((q as GenRow).genId ?? BASELINE_GEN.id) === BASELINE_GEN.id),
      ...(c.genQuality?.length ? { genQuality: c.genQuality } : {})
    })),
    machine: p.machine,
    request: p.request ?? null,
    allRuns, planningSnapshot, ...(stopReason ? { stopReason } : {})
  }
}

/** Newest recommendation for a workload from a REAL session (demo sessions are excluded). */
export function latestRecommendation(db: DatabaseSync, workload: WorkloadId): { sessionId: number; recommendation: Recommendation } | null {
  const row = db.prepare(`
    SELECT r.session_id, r.payload FROM recommendation r JOIN benchmark_session s ON s.id = r.session_id
    WHERE s.status = 'done' AND json_extract(s.payload, '$.demo') IS NOT 1 AND json_extract(r.payload, '$.workload') = ?
    ORDER BY r.id DESC LIMIT 1`).get(workload) as { session_id: number; payload: string } | undefined
  return row ? { sessionId: row.session_id, recommendation: json<Recommendation>(row.payload) } : null
}

export function telemetryForRun(db: DatabaseSync, runId: number): TelemetrySample[] {
  return (db.prepare('SELECT payload FROM telemetry_sample WHERE run_id = ? ORDER BY id').all(runId) as { payload: string }[]).map((r) => json<TelemetrySample>(r.payload))
}

/** Candidate plan for a request (same deterministic generateCandidates call the runner makes). */
export type PlanFor = (req: SessionRequest) => Pick<SessionPayload, 'vramBytes' | 'candidates' | 'machine'>

/** The runner's SessionStorage over these tables. Session ids are the integer row ids as strings.
 *  Runs keep RunDetail (minus samples) under payload.detail; samples go to telemetry_sample. */
export function makeSessionStorage(db: DatabaseSync, planFor: PlanFor): SessionStorage {
  const sid = (id: string) => {
    const n = Number(id)
    if (!Number.isInteger(n)) throw new Error(`bad session id ${id}`)
    return n
  }
  const modelOf = (id: string, configId: string) => {
    const row = db.prepare('SELECT payload FROM benchmark_session WHERE id = ?').get(sid(id)) as { payload: string } | undefined
    const c = row ? json<SessionPayload>(row.payload).candidates.find((x) => x.config.id === configId) : undefined
    return c?.model.id ?? configId.split('|')[0] // configId = `${modelId}|ngl=..|kv=..|t=..`
  }
  return {
    createSession: ({ workload, request }) => String(saveSession(db, { workload, request, ...planFor(request) }, 'running')),
    setSessionStatus: (id, status, error) => setSessionStatus(db, sid(id), status, error),
    listRunHistory: (id) => sessionInputs(db, sid(id))?.allRuns ?? [],
    listRuns: (id) =>
      (db.prepare(LATEST_RUNS).all(sid(id)) as { payload: string }[]).map((r) => {
        const { detail: _d, ...run } = json<BenchmarkRunResult & { detail?: unknown }>(r.payload)
        return run
      }),
    saveRun: (id, run, d: RunDetail) => {
      const { samples, ...detail } = d
      const runId = Number(db.prepare('INSERT INTO benchmark_run (session_id, status, model_id, ctx_size, payload) VALUES (?, ?, ?, ?, ?)')
        .run(sid(id), run.status, modelOf(id, run.configId), run.ctx, JSON.stringify({ ...run, detail })).lastInsertRowid)
      if (samples.length) insertTelemetrySamples(db, sid(id), runId, samples)
    },
    // Resume re-uses a stored suite only if it is this build's suite version and complete; otherwise it re-runs.
    listQuality: (id, modelId) => {
      const rows = (db.prepare('SELECT payload FROM quality_result WHERE session_id = ? AND model_id = ? ORDER BY id').all(sid(id), modelId) as { payload: string }[])
        .map((r) => json<QualityResult & { suite?: string; expected?: number }>(r.payload))
      const ok = rows.length > 0 && rows.every((r) => r.suite === defaultTestSet.suite && r.expected === rows.length)
      return ok ? rows : []
    },
    saveQuality: (id, modelId, configId, ctx, results) =>
      saveQualityResults(db, sid(id), modelId, results.map((r) => ({ ...r, configId, ctx, suite: defaultTestSet.suite, expected: results.length }))),
    saveRecommendation: (id, rec) => { saveRecommendation(db, sid(id), rec, rec.best ? modelOf(id, rec.best.configId) : null) }
  }
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
    avgGpuUtil: m(r.gpuAvgPct), avgCpuUtil: m(r.cpuAvgPct), warm: true // DEMO rows stand for warmed measurements (I-6.0)
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
