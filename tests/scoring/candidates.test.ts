import { describe, expect, it } from 'vitest'
import type { ModelMeta } from '../../src/shared/bench-types'
import { budgetFor, estimateMemory, generateCandidates, machineFromProfile, vramBudgetKey } from '../../src/core/benchmark/candidates'
import type { SystemProfile } from '../../src/shared/types'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { load, machine } from './helpers'

const llama8b = load('sweep-smooth.json').models[0]
const model = (o: Partial<ModelMeta>): ModelMeta => ({ ...llama8b, ...o })
// ~14B Q4_K_M (Qwen2.5-14B-like): 48 layers, 8 KV heads, 128 head dim → 192 KiB/token f16 KV.
const q14b = model({ id: 'q14b', fileBytes: 8.99e9, paramCount: 14.8e9, layers: 48, nEmbd: 5120, heads: 40, headsKv: 8, keyLength: 128, valueLength: 128, nVocab: 152064 })
// ~70B Q4_K_M: 42 GB file, 80 layers.
const l70b = model({ id: 'l70b', fileBytes: 42.5e9, paramCount: 70.6e9, layers: 80, nEmbd: 8192, heads: 64, headsKv: 8 })
const vulkan = { backend: 'vulkan' as const }

describe('estimateMemory', () => {
  it('KV for Llama-3.1-8B f16 = 128 KiB/token', () => {
    expect(estimateMemory(llama8b, 32, 1, 'f16').kvBytes).toBe(131072)
  })
})

describe('generateCandidates (16 GB VRAM / 31 GB RAM)', () => {
  it('8B fits fully: full offload, all steps ≤ 64K, plus q8 KV for long-context', () => {
    const { candidates } = generateCandidates(machine(), llama8b, vulkan, WORKLOADS.long_context_coding)
    expect(candidates.map((c) => c.id)).toEqual(['llama31-8b-q4km|ngl=all|kv=f16|t=8', 'llama31-8b-q4km|ngl=all|kv=q8_0|t=8'])
    expect(candidates[0].ctxSteps).toEqual([2048, 4096, 8192, 16384, 32768, 65536])
    expect(candidates[0].estVramBytes.kind).toBe('estimated')
  })

  it('never exceeds the declared context (A13)', () => {
    const { candidates } = generateCandidates(machine(), model({ ctxTrain: 8192 }), vulkan, WORKLOADS.coding)
    expect(candidates[0].ctxSteps).toEqual([2048, 4096, 8192])
    expect(candidates[0].skippedSteps.map((s) => s.ctx)).toEqual([16384, 32768, 65536, 131072])
    expect(candidates[0].skippedSteps[0].reason).toMatch(/above declared context 8K/)
  })

  it('14B: keeps one step just above the VRAM estimate (≤ 1.15× budget), prunes far-over steps with a reason', () => {
    const m = { ...machine(), vramInUseBytes: { value: 1024 ** 3, kind: 'measured' as const } } // budget ≈ 13.9 GiB
    const [c] = generateCandidates(m, q14b, vulkan, WORKLOADS.coding).candidates
    expect(c.ctxSteps).toEqual([2048, 4096, 8192, 16384, 32768]) // 32K est ≈ 14.6 GiB: over, within margin → kept
    expect(c.notes).toContain('32K is above the VRAM estimate; kept to observe the cliff')
    expect(c.skippedSteps.map((s) => s.ctx)).toEqual([65536, 131072])
    expect(c.skippedSteps[0].reason).toMatch(/est\. VRAM .* > budget/)
  })

  it('8B on 16 GB: 64K fits, 128K is pruned as far over budget (memory-bound, calibration)', () => {
    const [c] = generateCandidates(machine(), llama8b, vulkan, WORKLOADS.document_analysis).candidates
    expect(c.ctxSteps.at(-1)).toBe(65536)
    expect(c.skippedSteps).toMatchObject([{ ctx: 131072, reason: expect.stringMatching(/est\. VRAM .* > budget/) }])
  })

  it('small model with full offload fitting gets no ngl=0 candidate (ngl 0 is not CPU-only on Vulkan)', () => {
    const tiny = model({ id: 'tiny', fileBytes: 1.1e9, paramCount: 1.5e9, layers: 28, nEmbd: 1536, heads: 12, headsKv: 2, keyLength: 128, valueLength: 128, ctxTrain: 32768 })
    const { candidates } = generateCandidates(machine(), tiny, vulkan, WORKLOADS.coding)
    expect(candidates.map((c) => c.gpuLayers)).toEqual([28])
  })

  it('70B: nothing fits; every combo rejected with a memory reason', () => {
    const { candidates, rejected } = generateCandidates(machine(), l70b, vulkan, WORKLOADS.coding)
    expect(candidates).toEqual([])
    expect(rejected.length).toBeGreaterThan(0)
    expect(rejected.every((r) => /est\. (VRAM|RAM)/.test(r.reason))).toBe(true)
  })

  it('no GPU → only ngl=0 (CPU)', () => {
    const { candidates } = generateCandidates({ ...machine(), gpuDevice: null }, llama8b, vulkan, WORKLOADS.coding)
    expect(candidates.map((c) => c.gpuLayers)).toEqual([0])
  })

  it('VRAM unavailable does not empty the list (X14)', () => {
    const m = { ...machine(), vramBytes: { value: null, kind: 'unavailable' as const, reason: 'test' } }
    const { candidates } = generateCandidates(m, q14b, vulkan, WORKLOADS.coding)
    expect(candidates.length).toBeGreaterThan(0)
    expect(candidates[0].notes).toContain('VRAM size unknown; not pruned by VRAM, runtime guards apply')
  })
})

describe('machineFromProfile: measured VRAM in use from the profile', () => {
  const GiB = 1024 ** 3
  const profile = (vramInUse?: SystemProfile['vramInUse']): SystemProfile => ({
    scannedAt: 't0',
    os: { value: { name: 'Windows 11', version: '10', build: '26200' }, status: 'available', source: 't' },
    cpu: { value: { model: 'Ryzen', physicalCores: 8, logicalCores: 16 }, status: 'available', source: 't' },
    ram: { value: { totalBytes: 31 * GiB, availableBytes: 20 * GiB }, status: 'available', source: 't' },
    gpus: { value: [{ name: 'RX 9070 XT', vendor: 'amd', pnpDeviceId: 'x', driverVersion: null, isIntegrated: false, dedicatedVramBytes: { value: 17095983104, status: 'available', source: 'registry' } }], status: 'available', source: 't' },
    cuda: { value: { available: false }, status: 'unsupported', source: 't' },
    disks: { value: [], status: 'available', source: 't' },
    runtimes: [],
    ...(vramInUse ? { vramInUse } : {})
  }) as SystemProfile
  const budgetOf = (p: SystemProfile, inUse?: number) => {
    const [c] = generateCandidates(machineFromProfile(p, 'Vulkan0', inUse), q14b, vulkan, WORKLOADS.coding).candidates
    return { reason: c.skippedSteps.find((s) => /budget/.test(s.reason))!.reason, notes: c.notes }
  }
  it('a measured 1.2 GiB in use lowers the VRAM budget by 1.2 GiB', () => {
    const m = machineFromProfile(profile({ value: 1.2 * GiB, status: 'available', source: 'GPU Adapter Memory\Dedicated Usage' }), 'Vulkan0')
    expect(m.vramInUseBytes).toMatchObject({ value: 1.2 * GiB, kind: 'measured' })
    const with12 = budgetOf(profile({ value: 1.2 * GiB, status: 'available', source: 'pdh' })).reason
    const none = budgetOf(profile(), 0).reason // explicit 0 in use
    const b = (r: string) => Number(/budget ([\d.]+) GiB/.exec(r)![1])
    expect(b(none) - b(with12)).toBeCloseTo(1.2, 1)
  })
  it('an unavailable reading budgets a conservative 1.5 GiB (never 0), says why in the notes; the explicit arg still wins', () => {
    const p = profile({ value: null, status: 'unavailable', source: 'pdh', error: 'counter read failed' })
    expect(machineFromProfile(p, 'Vulkan0').vramInUseBytes).toMatchObject({ value: null, kind: 'unavailable', reason: 'counter read failed' })
    expect(budgetOf(p).notes).toContain('VRAM in use by other apps unknown (counter read failed); budget assumes 1.5 GiB')
    const b = (r: string) => Number(/budget ([\d.]+) GiB/.exec(r)![1])
    expect(b(budgetOf(profile(), 0).reason) - b(budgetOf(p).reason)).toBeCloseTo(1.5, 1)
    expect(machineFromProfile(p, 'Vulkan0', 2 * GiB).vramInUseBytes.value).toBe(2 * GiB)
  })

describe('planning snapshot (data contract §12)', () => {
  it('every candidate carries the budget basis; unknown VRAM-in-use is the estimated default', () => {
    const f = load('calib-8b-rx9070.json')
    const m = machine(f.vramBytes)
    const c = generateCandidates(m, f.models[0], { backend: 'vulkan' }, WORKLOADS.coding).candidates[0]
    expect(c.planning).toMatchObject({ vramTotalBytes: f.vramBytes, planningVramBudgetBytes: f.vramBytes, planningReserveBytes: 1024 ** 3, candidateRulesVersion: 'cand-1.5', vramInUse: { kind: 'measured' }, effectiveBudget: { kind: 'estimated', source: expect.stringMatching(/^no measured budget on this machine yet; assuming 80 % of the adapter total \(some GPUs\/backends allow 100 %\)$/) } })
    const blind = generateCandidates({ ...m, vramInUseBytes: { value: null, kind: 'unavailable', reason: 'no counter' } }, f.models[0], { backend: 'vulkan' }, WORKLOADS.coding).candidates[0]
    expect(blind.planning!.vramInUse).toMatchObject({ value: 1.5 * 1024 ** 3, kind: 'estimated' })
  })
})
})

describe('effective per-process VRAM budget (observations per GPU/driver/backend)', () => {
  const G = 1024 ** 3
  const f = load('calib-8b-rx9070.json')
  // cal-2026-09-27, one card: 14B spill at 13.25 GiB dedicated, 8B f16 64K at 11.6 GiB (KV 8 GiB largest buffer)
  const obs14 = { ceilingBytes: 13.25 * G, modelId: 'qwen14b', ctx: 32768, kvType: 'f16' as const, kvBytes: 3 * G, largestBufferBytes: 8.4 * G, observedAt: 1 }
  const obs8 = { ceilingBytes: 11.6 * G, modelId: 'llama8b', ctx: 65536, kvType: 'f16' as const, kvBytes: 8 * G, largestBufferBytes: 8 * G, observedAt: 2 }
  const small = { ...obs14, ceilingBytes: 14 * G, largestBufferBytes: 2 * G }
  const mach = (o: typeof obs14[]) => ({ ...machine(f.vramBytes), vramBudgetObservations: o })
  it('min of comparable observations; else the most conservative, said so; else 80 % labelled estimated', () => {
    expect(budgetFor(mach([obs14, obs8]), 8 * G)).toMatchObject({ value: 11.6 * G, kind: 'measured', source: expect.stringMatching(/min of 2 observation\(s\) with a comparable largest buffer/) })
    expect(budgetFor(mach([small, obs14]), 2 * G)).toMatchObject({ value: 14 * G, kind: 'measured' })
    expect(budgetFor(mach([obs14, obs8]), 0.5 * G)).toMatchObject({ value: 11.6 * G, source: expect.stringMatching(/most conservative of 2 observation\(s\) — none with a comparable largest buffer/) })
    expect(budgetFor(mach([]), 8 * G)).toMatchObject({ value: f.vramBytes * 0.8, kind: 'estimated' })
    expect(machineFromProfile({ gpus: { value: [] }, ram: { value: null }, cpu: { value: null } } as unknown as SystemProfile, null).vramEffectiveBudgetBytes).toMatchObject({ value: null, kind: 'unavailable' })
  })
  it('a measured budget prunes planning (min with the adapter budget); the estimated fallback does not', () => {
    const plain = generateCandidates(machine(f.vramBytes), f.models[0], { backend: 'vulkan' }, WORKLOADS.long_context_coding).candidates
    const tight = generateCandidates(mach([{ ...obs8, ceilingBytes: 6 * G }]), f.models[0], { backend: 'vulkan' }, WORKLOADS.long_context_coding).candidates
    const last = (cs: typeof plain) => Math.max(...cs.flatMap((c) => c.ctxSteps))
    expect(last(tight)).toBeLessThan(last(plain))
    expect(tight.flatMap((c) => c.skippedSteps).find((s) => s.skip?.resource === 'vram')!.reason).toMatch(/> budget 6\.0 GiB \(per-process budget 6\.0 GiB, measured\)$/)
    expect(tight[0].planning!.effectiveBudget).toMatchObject({ kind: 'measured' })
  })
  it('the key names GPU, driver and backend build', () => {
    const p = { gpus: { value: [{ name: 'RX 9070 XT', driverVersion: '32.0.1', isIntegrated: false, dedicatedVramBytes: { value: 16 * G } }] } } as unknown as SystemProfile
    expect(vramBudgetKey(p, 'vulkan', 'b11208')).toBe('RX 9070 XT|32.0.1|vulkan:b11208')
    expect(vramBudgetKey(p, 'hip', 'b11208')).not.toBe(vramBudgetKey(p, 'vulkan', 'b11208'))
  })
})
