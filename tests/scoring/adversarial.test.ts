// Adversarial scoring suite (ACCEPTANCE §3: X3, X8, X10–X13, X15–X20) + cases from the 2026-09-27 real-machine
// calibration (docs/calibration-2026-09-27.md). #1's scoring.test.ts covers X1–X7, X9, X14.
// `it.fails` = known defect, reported to the orchestrator; remove `.fails` once fixed.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { BenchmarkRunResult, CandidateInput, WorkloadId } from '../../src/shared/bench-types'
import { detectCliffs, fmtCtx, fmtGiB } from '../../src/core/scoring/cliff'
import { componentScores } from '../../src/core/scoring/components'
import { recommend } from '../../src/core/scoring/recommend'
import { DEFAULT_SCORING_CONFIG, WORKLOADS } from '../../src/core/scoring/workloads'
import { estimateMemory, generateCandidates, machineFromProfile } from '../../src/core/benchmark/candidates'
import { scanSystem } from '../../src/core/system/scanner'
import { inputs, load, machine, toRun, type Fixture, type FixtureRun } from './helpers'

const GiB = 1024 ** 3
const calib = load('calib-rx9070-2026-09-27.json')
const ALL_WORKLOADS = Object.keys(WORKLOADS) as WorkloadId[]
const byId = (f: Fixture, id: string) => inputs(f).find((i) => i.config.id === id)!
const runsOf = (f: Fixture, id: string) => f.runs.filter((r) => r.configId === id)
const sub = (f: Fixture, ids: string[]): Fixture => ({ ...f, runs: f.runs.filter((r) => ids.includes(r.configId)) })
const M = machine(calib.vramBytes)
const llama = calib.models.find((m) => m.id === 'llama31-8b-q4km')!
const qwen = calib.models.find((m) => m.id === 'qwen25-1.5b-q4km')!

/** Deterministic Fisher–Yates with a seeded LCG (no Math.random in tests either). */
function shuffle<T>(xs: T[], seed: number): T[] {
  const a = [...xs]
  let s = seed >>> 0
  for (let i = a.length - 1; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    const j = s % (i + 1)
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

const step = (ctx: number, over: Partial<FixtureRun> = {}): BenchmarkRunResult => toRun({
  configId: 'x', model: 'llama31-8b-q4km', ctx, gpuLayers: 99, threads: 8, status: 'ok', promptTokens: ctx * 0.75, loadMs: 3000,
  ttftMs: 1000, prefillTps: 2500, decodeTps: 100, totalMs: 2000, peakRamBytes: 0.6 * GiB, peakVramBytes: 6 * GiB,
  peakSharedGpuBytes: 0.02 * GiB, cpuAvgPct: 20, gpuAvgPct: 90, ...over
})

// ---------------------------------------------------------------------------------------------------------------
describe('calibration (a): real full-offload 8B sweep, gradual decode decay', () => {
  const cliff = detectCliffs(runsOf(calib, 'llama8b|all').map(toRun), calib.vramBytes)

  it('−4/−5/−12/−18/−27% decode steps are not a cliff; every rung passes; ceiling = 64K (top of the ladder)', () => {
    expect(cliff.steps.map((s) => s.verdict)).toEqual(['pass', 'pass', 'pass', 'pass', 'pass', 'pass'])
    expect(cliff.steps.flatMap((s) => s.reasons)).toEqual([])
    expect(cliff.practicalContextCeiling).toMatchObject({ value: 65536, kind: 'measured' })
    expect(cliff.limitedBy).toBe('none')
    expect(cliff.spillFreeUpTo).toBe(65536)
  })

  it('shared-GPU Δ 0.12 GiB at 64K (first nonzero spill signal) stays under the 256 MiB spill rule', () => {
    expect(0.12 * GiB).toBeLessThan(DEFAULT_SCORING_CONFIG.cliff.sharedSpillBytes)
  })

  it('64K is memory-bound: memory component < 100 at 64K (83% of VRAM) but 100 at 8K', () => {
    const cands = byId(calib, 'llama8b|all')
    const at = (ctx: number) => componentScores({ ...cands, runs: cands.runs.filter((r) => r.ctx === ctx) }, M, WORKLOADS.document_analysis)
    expect(at(8192).components.memory.score).toBe(100)
    const m64 = at(65536).components.memory
    expect(m64.input.value!).toBeCloseTo(13.28 * GiB / calib.vramBytes, 3)
    expect(m64.score).toBeLessThan(100)
    expect(m64.score).toBeGreaterThan(80)
  })

  it('estimateMemory never UNDER-estimates measured per-PID VRAM by >5% (8B and 1.5B, every rung)', () => {
    for (const [model, id] of [[llama, 'llama8b|all'], [qwen, 'qwen|all']] as const) {
      for (const r of runsOf(calib, id)) {
        const ratio = estimateMemory(model, 99, r.ctx, 'f16').vramBytes / r.peakVramBytes!
        expect(ratio, `${model.id} @ ${fmtCtx(r.ctx)} est/measured`).toBeGreaterThan(0.95)
      }
    }
  })

  it('estimateMemory is within +25% for 8B at every rung', () => {
    for (const r of runsOf(calib, 'llama8b|all')) {
      expect(estimateMemory(llama, 99, r.ctx, 'f16').vramBytes / r.peakVramBytes!, fmtCtx(r.ctx)).toBeLessThan(1.25)
    }
  })

  // DEFECT (candidates.ts:46-48): fixed overhead 512 MiB compute + 256 MiB + ubatch·nVocab·4 ≈ 1.0 GiB, but the
  // measured compute buffer is 61–164 MiB. Qwen 1.5B is over-estimated 1.87× at 2K / 1.49× at 32K, so small models are
  // pruned on small GPUs that would run them.
  it.fails('estimateMemory is within +25% for a 1.5B model', () => {
    for (const r of runsOf(calib, 'qwen|all')) {
      expect(estimateMemory(qwen, 99, r.ctx, 'f16').vramBytes / r.peakVramBytes!, fmtCtx(r.ctx)).toBeLessThan(1.25)
    }
  })

  it('KV estimate is exact (llama-server log): 8B 0.125 MiB/token, 1.5B 0.02734 MiB/token', () => {
    expect(estimateMemory(llama, 99, 65536, 'f16').kvBytes).toBe(8192 * 1024 ** 2)
    expect(estimateMemory(qwen, 99, 32768, 'f16').kvBytes).toBe(896 * 1024 ** 2)
  })

  it('candidate generation keeps 64K for 8B on the 16 GB card with the measured 1.33 GiB idle VRAM', () => {
    const m = { ...M, vramInUseBytes: { value: 1.33 * GiB, kind: 'measured' as const } }
    const set = generateCandidates(m, llama, { backend: 'vulkan' }, WORKLOADS.long_context_coding)
    const full = set.candidates.find((c) => c.gpuLayersAll && c.kvType === 'f16')!
    expect(full.ctxSteps).toContain(65536)
    expect(full.skippedSteps.filter((s) => s.ctx <= 65536)).toEqual([])
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('calibration (b): TTFT 32 s at 64K vs workload latency tolerance', () => {
  const only64 = (): CandidateInput => {
    const c = byId(calib, 'llama8b|all')
    return { ...c, runs: c.runs.filter((r) => r.ctx === 65536) }
  }

  it('latency component: 0 for Fast Assistant (2 s) and General Chat (8 s), > 0 for Document Analysis (60 s)', () => {
    const lat = (w: WorkloadId) => componentScores(only64(), M, WORKLOADS[w]).components.latency.score
    expect(lat('fast_assistant')).toBe(0)
    expect(lat('general_chat')).toBe(0)
    expect(lat('document_analysis')).toBeGreaterThan(20)
  })

  it.each(ALL_WORKLOADS)('%s: on the full sweep the reference step (recommended ctx) meets the latency tolerance', (w) => {
    const c = byId(calib, 'llama8b|all')
    const cs = componentScores(c, M, WORKLOADS[w])
    const ref = runsOf(calib, 'llama8b|all').find((r) => r.ctx === cs.referenceCtx)!
    expect(ref.ttftMs!, `${w} @ ${fmtCtx(ref.ctx)}`).toBeLessThanOrEqual(WORKLOADS[w].latencyToleranceMs)
    if (w === 'fast_assistant' || w === 'general_chat') expect(cs.referenceCtx).toBeLessThan(65536)
  })

  // DEFECT: no latency gate exists (recommend.ts:528-533 gates only ctx, stability, quality). A config whose only
  // measured TTFT is 16× the Fast Assistant tolerance is still ELIGIBLE and can be the recommendation.
  it.fails('a config whose TTFT at its reference step exceeds the tolerance is ineligible for Fast Assistant/General Chat', () => {
    for (const w of ['fast_assistant', 'general_chat'] as const) {
      const r = recommend([only64()], M, w)
      expect(r.ranked[0].eligible, w).toBe(false)
      expect(r.ranked[0].gateFailures.join(' '), w).toMatch(/TTFT|latency/i)
    }
  })

  it('...but stays eligible for Document Analysis (32 s < 60 s tolerance)', () => {
    expect(recommend([only64()], M, 'document_analysis').ranked[0].eligible).toBe(true)
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('calibration (c): partial offload vs full offload (real numbers: decode 17.5 vs 99.9, CPU 71%)', () => {
  const f = sub(calib, ['llama8b|all', 'llama8b|ngl20', 'llama8b|ngl0'])

  it.each(ALL_WORKLOADS)('%s: never recommends partial/CPU offload when full offload passes', (w) => {
    const r = recommend(inputs(f), M, w)
    expect(r.best?.configId).toBe('llama8b|all')
  })

  const gap = (w: WorkloadId) => {
    const r = recommend(inputs(f), M, w)
    const t = (id: string) => r.ranked.find((s) => s.configId === id)!.total
    return { full: t('llama8b|all'), partial: t('llama8b|ngl20'), cpu: t('llama8b|ngl0') }
  }

  it('fast_assistant: partial offload scores ≥ 15 points below full, CPU-only lower still', () => {
    const g = gap('fast_assistant')
    expect(g.full - g.partial).toBeGreaterThanOrEqual(15)
    expect(g.partial).toBeGreaterThan(g.cpu)
  })

  // DEFECT (workloads.ts:602-605 genTargetTps 25–30): decode 17.5 t/s (5.7× slower than 99.9) still earns genSpeed ≈80,
  // and everything ≥30 t/s saturates at 100, so partial offload lands only 6–12 points below full.
  it.fails.each(['general_chat', 'coding', 'reasoning'] as const)('%s: partial offload scores ≥ 15 points below full', (w) => {
    const g = gap(w)
    expect(g.full - g.partial).toBeGreaterThanOrEqual(15)
  })

  it.each(['general_chat', 'coding', 'reasoning'] as const)('%s: partial still ranks above CPU-only', (w) => {
    const g = gap(w)
    expect(g.partial).toBeGreaterThan(g.cpu)
  })

  it('partial offload memory score is capped (60) even though it uses less VRAM', () => {
    const cs = componentScores(byId(f, 'llama8b|ngl20'), M, WORKLOADS.general_chat)
    expect(cs.components.memory.score).toBe(DEFAULT_SCORING_CONFIG.norm.memPartialOffloadCap)
    expect(cs.components.memory.note).toMatch(/partial GPU offload/)
  })

  it('generateCandidates offers no partial config for 8B when full offload fits 16 GB', () => {
    const set = generateCandidates(M, llama, { backend: 'vulkan' }, WORKLOADS.general_chat)
    expect(set.candidates.every((c) => c.gpuLayersAll || c.gpuLayers === 0)).toBe(true)
    expect(set.candidates.some((c) => c.gpuLayers === 0)).toBe(false) // 8B > cpuOnlyMaxParams
  })

  it('X10: a 100% CPU spike on the partial config never becomes a cliff/spill reason (CPU is not a cliff input)', () => {
    const r = detectCliffs([step(4096, { gpuLayers: 20, cpuAvgPct: 100 }), step(8192, { gpuLayers: 20, cpuAvgPct: 100, decodeTps: 95 })], calib.vramBytes)
    expect(r.steps.flatMap((s) => s.reasons)).toEqual([])
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('calibration (d) + X11: mmap RAM is not spill', () => {
  it('available-RAM drop ≈ model file size (5 GiB) with VRAM at 40% → no spill', () => {
    const r = detectCliffs([step(2048, { peakRamBytes: 0.6 * GiB }), step(4096, { peakRamBytes: 5.9 * GiB })], calib.vramBytes)
    expect(r.steps.flatMap((s) => s.reasons)).toEqual([])
    expect(r.spillFreeUpTo).toBe(4096)
  })

  it('real 64K rung (VRAM 83%) + a +5 GiB RAM jump → still no spill (saturation rule needs ≥95%)', () => {
    const r = detectCliffs([step(32768, { peakVramBytes: 8.57 * GiB }), step(65536, { peakVramBytes: 13.28 * GiB, peakRamBytes: 5.6 * GiB, decodeTps: 73 })], calib.vramBytes)
    expect(r.steps[1].verdict).toBe('pass')
  })

  it('contract: at ≥95% VRAM a ≥1 GiB growth IS spill — so the runner must feed private WS, never available-RAM deltas', () => {
    const r = detectCliffs([step(32768, { peakVramBytes: 14 * GiB }), step(65536, { peakVramBytes: 15.5 * GiB, peakRamBytes: 5.6 * GiB, decodeTps: 90 })], calib.vramBytes)
    expect(r.steps[1].reasons.map((x) => x.code)).toContain('vram_spill')
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('calibration (e) + X3/X16: determinism', () => {
  const f = calib

  it.each(ALL_WORKLOADS)('%s: same inputs twice → deep-equal Recommendation', (w) => {
    expect(recommend(inputs(f), M, w)).toEqual(recommend(inputs(f), M, w))
  })

  it.each(ALL_WORKLOADS)('%s: shuffled candidate order and shuffled runs → identical Recommendation', (w) => {
    const base = recommend(inputs(f), M, w)
    for (const seed of [1, 7, 42]) {
      const shuffled = shuffle(inputs(f), seed).map((c) => ({ ...c, runs: shuffle(c.runs, seed + 1) }))
      expect(recommend(shuffled, M, w)).toEqual(base)
    }
  })

  it('the full calibration set ranks sensibly: fast_assistant → qwen (366 t/s), max_quality → 8B', () => {
    expect(recommend(inputs(f), M, 'fast_assistant').best?.configId).toBe('qwen|all')
    expect(recommend(inputs(f), M, 'max_quality').best?.configId).toBe('llama8b|all')
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('X8: cliff edge cases', () => {
  it('tiny base: 3 → 1.5 t/s is −50% but under the 2 t/s absolute floor → no cliff', () => {
    const r = detectCliffs([step(2048, { decodeTps: 3 }), step(4096, { decodeTps: 1.5 })], null)
    expect(r.steps[1].reasons.filter((x) => x.code === 'decode_drop')).toEqual([])
  })

  it('compares with the previous step, not the 2K baseline: six −15% steps (−56% cumulative) → no cliff', () => {
    const r = detectCliffs([2048, 4096, 8192, 16384, 32768, 65536].map((c, i) => step(c, { decodeTps: 100 * 0.85 ** i })), null)
    expect(r.steps.every((s) => s.verdict === 'pass')).toBe(true)
  })

  it('exactly at the threshold (ratio 0.60) is a cliff; 0.61 is not', () => {
    expect(detectCliffs([step(2048, { decodeTps: 100 }), step(4096, { decodeTps: 60 })], null).steps[1].verdict).toBe('degraded')
    expect(detectCliffs([step(2048, { decodeTps: 100 }), step(4096, { decodeTps: 61 })], null).steps[1].verdict).toBe('pass')
  })

  it('OOM at 32K: failure limits the ceiling to 16K and says so; labelled failure, not cliff', () => {
    const r = detectCliffs([step(8192), step(16384), step(32768, { status: 'oom', decodeTps: null })], null)
    expect(r.practicalContextCeiling.value).toBe(16384)
    expect(r.limitedBy).toBe('failure')
    expect(r.steps[2].reasons[0].message).toBe('32K: run fail (oom)')
  })

  it('input order of steps does not matter', () => {
    const s = [step(2048), step(4096, { decodeTps: 50 }), step(8192, { decodeTps: 49 })]
    expect(detectCliffs(shuffle(s, 3), null)).toEqual(detectCliffs(s, null))
  })

  it('a missing middle rung (4K never run) compares 2K→8K directly and does not invent 4K', () => {
    const r = detectCliffs([step(2048, { decodeTps: 100 }), step(8192, { decodeTps: 55 })], null)
    expect(r.steps.map((s) => s.ctx)).toEqual([2048, 8192])
    expect(r.steps[1].reasons[0]).toMatchObject({ code: 'decode_drop', fromCtx: 2048, toCtx: 8192 })
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('X12: units', () => {
  it('formats bytes as GiB and ctx as K', () => {
    expect(fmtGiB(calib.vramBytes)).toBe('15.9 GiB')
    expect(fmtCtx(65536)).toBe('64K')
    expect(fmtCtx(49159)).toBe('49159')
  })

  it('TTFT is milliseconds against a millisecond tolerance: 1056 ms on Fast Assistant (2000 ms) scores ≈27.7', () => {
    const c = byId(calib, 'llama8b|all')
    const lat = componentScores(c, M, WORKLOADS.fast_assistant).components.latency
    expect(lat.input.value).toBe(1056)
    expect(lat.score).toBeCloseTo(100 * (1 - Math.log(1056 / 200) / Math.log(10)), 6)
  })

  it('decode and prefill scores use TPS from the reference step as-is (no ms/s mixups)', () => {
    const cs = componentScores(byId(calib, 'qwen|all'), M, WORKLOADS.fast_assistant)
    const ref = runsOf(calib, 'qwen|all').find((r) => r.ctx === cs.referenceCtx)!
    expect(cs.components.genSpeed.input.value).toBe(ref.decodeTps)
    expect(cs.components.genSpeed.score).toBe(100) // ≥ 60 t/s target
    expect(cs.components.prefillSpeed.input.value).toBe(ref.prefillTps)
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('X15: the benchmark device is the dGPU', () => {
  it('scan fixture (RX 9070 XT + iGPU + stale RTX 3080 key) → machine VRAM = 9070 XT 15.9 GiB', async () => {
    const raw = readFileSync(join(__dirname, '../fixtures/scan-rx9070.json'), 'utf8')
    const p = await scanSystem({ powershell: async () => raw, exec: async () => { throw new Error('no nvidia-smi') } })
    const m = machineFromProfile(p, 'Vulkan0')
    expect(m.vramBytes).toMatchObject({ value: 17095983104, kind: 'declared' })
    expect(m.gpuDevice).toBe('Vulkan0')
  })

  it('only an iGPU → no discrete GPU, gpuDevice null, VRAM unavailable (not 0)', async () => {
    const raw = JSON.parse(readFileSync(join(__dirname, '../fixtures/scan-rx9070.json'), 'utf8'))
    raw.video.data = raw.video.data.filter((v: { Name: string }) => v.Name.includes('(TM) Graphics'))
    const p = await scanSystem({ powershell: async () => JSON.stringify(raw), exec: async () => '' })
    const m = machineFromProfile(p, 'Vulkan1')
    expect(m.gpuDevice).toBeNull()
    expect(m.vramBytes).toMatchObject({ value: null, kind: 'unavailable' })
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('X18: aborted / partial runs never feed the score', () => {
  const withAbort = (status: string): CandidateInput => {
    const c = byId(calib, 'llama8b|all')
    const runs = c.runs.filter((r) => r.ctx <= 8192)
    // a cancelled 8K run that still carries partial (fast-looking) numbers
    runs[2] = toRun({ ...runsOf(calib, 'llama8b|all')[2], status, decodeTps: 500, ttftMs: 10 })
    return { ...c, runs }
  }

  it.each(['cancelled', 'timeout', 'crashed'])('%s 8K step: verdict fail, reference falls back to 4K, its numbers are ignored', (status) => {
    const cs = componentScores(withAbort(status), M, WORKLOADS.general_chat)
    expect(cs.cliff.steps[2].verdict).toBe('fail')
    expect(cs.referenceCtx).toBe(4096)
    expect(cs.components.genSpeed.input.value).toBe(105.4)
  })

  it('only aborted runs → excluded, not ranked', () => {
    const c = byId(calib, 'qwen|all')
    const r = recommend([{ ...c, runs: c.runs.map((x) => ({ ...x, status: 'cancelled' as const, failureKind: null })) }], M, 'general_chat')
    expect(r.best).toBeNull()
    expect(r.ranked).toEqual([])
    expect(r.excluded[0].configId).toBe('qwen|all')
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('X19: candidate explosion is bounded', () => {
  it.each(ALL_WORKLOADS)('%s: ≤ maxPerModel candidates per model, ids unique and deterministic', (w) => {
    for (const model of [llama, qwen]) {
      const a = generateCandidates(M, model, { backend: 'vulkan' }, WORKLOADS[w])
      expect(a.candidates.length).toBeLessThanOrEqual(4)
      expect(new Set(a.candidates.map((c) => c.id)).size).toBe(a.candidates.length)
      expect(generateCandidates(M, model, { backend: 'vulkan' }, WORKLOADS[w])).toEqual(a)
    }
  })

  it('a 70B-class model on 16 GB gives at most 2 partial configs + no full, never an empty set with no reason', () => {
    const big = { ...llama, id: 'big', fileBytes: 40 * GiB, paramCount: 70e9, layers: 80, headsKv: 8 }
    const s = generateCandidates(M, big, { backend: 'vulkan' }, WORKLOADS.general_chat)
    expect(s.candidates.filter((c) => c.gpuLayersAll)).toEqual([])
    expect(s.candidates.length + s.rejected.length).toBeGreaterThan(0)
    expect(s.candidates.length).toBeLessThanOrEqual(2)
  })
})

// ---------------------------------------------------------------------------------------------------------------
describe('X20: quality with no measured results is labelled ESTIMATED', () => {
  it('max_quality without a quality run: quality input kind estimated, note surfaces in the recommendation', () => {
    const r = recommend(inputs(sub(calib, ['llama8b|all'])), M, 'max_quality')
    const q = r.best!.score.breakdown.find((b) => b.component === 'quality')!
    expect(q.input.kind).toBe('estimated')
    expect(r.reasons.join('\n')).toMatch(/ESTIMATED/)
  })

  it('measured quality replaces the prior', () => {
    const c = byId(calib, 'llama8b|all')
    const measured = { ...c, quality: [{ testId: 'RS-01', category: 'reasoning' as const, weight: 1, pass: true, score: 1, detail: '' }] }
    const q = componentScores(measured, M, WORKLOADS.reasoning).components.quality
    expect(q.input.kind).toBe('measured')
    expect(q.score).toBe(100)
  })
})

// X13 (warm vs cold) and X17 (cross-session version drift): BenchmarkRunResult has no cold/warm flag and no
// runtime/driver version fields, so scoring cannot see either — covered only by the runner (warmup) and storage.
