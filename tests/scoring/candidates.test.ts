import { describe, expect, it } from 'vitest'
import type { MachineLimits, ModelMeta, VramBudgetObservation } from '../../src/shared/bench-types'
import { applicableObservations, budgetFor, estimateMemory, generateCandidates, machineFromProfile, planCandidates, vramBudgetKey } from '../../src/core/benchmark/candidates'
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

describe('integrated-only shared-RAM planning', () => {
  const GiB = 1024 ** 3
  const profile = {
    gpus: { value: [{ name: 'Intel Iris Xe', pnpDeviceId: 'PCI\\IGPU', driverVersion: '1', isIntegrated: true,
      dedicatedVramBytes: { value: 16 * GiB, status: 'available', source: 'reported shared aperture' } }] },
    ram: { value: { totalBytes: 16 * GiB, availableBytes: 11 * GiB }, source: 'scan' },
    cpu: { value: { physicalCores: 4 } }
  } as unknown as SystemProfile
  it('fits a small model on Vulkan, never treats the advertised 16 GiB shared aperture as dedicated VRAM', () => {
    const m = machineFromProfile(profile, 'Vulkan0')
    const set = generateCandidates(m, llama8b, vulkan, WORKLOADS.coding)
    expect(vramBudgetKey(profile, 'vulkan', 'b1')).toBeNull()
    expect(m.vramBytes.value).toBeNull()
    expect(set.candidates[0]).toMatchObject({ device: 'Vulkan0', gpuLayersAll: true, mmap: false })
    expect(set.candidates[0].ctxSteps).toContain(2048)
    expect(set.candidates[0].notes).toEqual(expect.arrayContaining([expect.stringMatching(/device allocations share system RAM/)]))
    expect(set.candidates[0].planning?.planningVramBudgetBytes).toBeNull()
  })
  it('rejects when GPU buffers plus CPU buffers cross the available-RAM reserve, even if fake shared VRAM is large', () => {
    const m = machineFromProfile(profile, 'Vulkan0')
    const full = estimateMemory(q14b, q14b.layers, 2048, 'f16')
    expect(full.ramBytes).toBeLessThan(7 * GiB)
    expect(full.ramBytes + full.vramBytes).toBeGreaterThan(7 * GiB)
    const set = generateCandidates(m, q14b, vulkan, WORKLOADS.coding)
    expect(set.candidates).toEqual([])
    expect(set.rejected[0].reason).toMatch(/est\. RAM .* > available − reserve/)
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
    expect(c.planning).toMatchObject({ vramTotalBytes: f.vramBytes, planningVramBudgetBytes: f.vramBytes, planningReserveBytes: 1024 ** 3, candidateRulesVersion: 'cand-1.6', vramInUse: { kind: 'measured' }, effectiveBudget: { kind: 'estimated', source: expect.stringMatching(/^no measured budget on this machine yet; assuming 80 % of the adapter total \(some GPUs\/backends allow 100 %\)$/) } })
    const blind = generateCandidates({ ...m, vramInUseBytes: { value: null, kind: 'unavailable', reason: 'no counter' } }, f.models[0], { backend: 'vulkan' }, WORKLOADS.coding).candidates[0]
    expect(blind.planning!.vramInUse).toMatchObject({ value: 1.5 * 1024 ** 3, kind: 'estimated' })
  })
})
})

describe('effective per-process VRAM budget — advisory until qualified (w4m)', () => {
  const G = 1024 ** 3
  const f = load('calib-8b-rx9070.json')
  const origin = { sessionId: 's', configId: 'c', status: 'pass' as const, attempts: 2, firstPeakVramBytes: null, firstResidentSharedBytes: null }
  // cal-2026-09-27, one card: 14B spill at 13.25 GiB dedicated, 8B f16 64K at 11.6 GiB (KV 8 GiB largest buffer)
  const cap = (ceiling: number, largest: number | null, over: Partial<VramBudgetObservation> = {}): VramBudgetObservation =>
    ({ kind: 'capacity', qualified: true, ceilingBytes: ceiling * G, modelId: 'm', ctx: 65536, kvType: 'f16', gpuLayers: 33, kvBytes: null, largestBufferBytes: largest === null ? null : largest * G, origin, observedAt: 1, ...over })
  const mach = (o: VramBudgetObservation[]) => ({ ...machine(f.vramBytes), vramBudgetObservations: o })
  it('measured only from qualified capacity observations with a comparable load-log buffer (min of them)', () => {
    expect(budgetFor(mach([cap(13.25, 8.4), cap(11.6, 8)]), 8 * G)).toMatchObject({ value: 11.6 * G, kind: 'measured', source: expect.stringMatching(/min of 2 qualified observation\(s\) with a comparable largest buffer/) })
    expect(budgetFor(mach([]), 8 * G)).toMatchObject({ value: f.vramBytes * 0.8, kind: 'estimated', source: expect.stringMatching(/^no measured budget on this machine yet; assuming 80 %/) })
  })
  it('w4m-3: a non-comparable observation is advisory (estimated) and never prunes — 100 GiB buffer / 6 GiB ceiling', () => {
    const far = [cap(6, 100)]
    expect(budgetFor(mach(far), 8 * G)).toMatchObject({ value: 6 * G, kind: 'estimated', source: expect.stringMatching(/^advisory: 1 capacity observation\(s\) not comparable .* not applied to planning$/) })
    const last = (m: MachineLimits, w = WORKLOADS.long_context_coding) => Math.max(...generateCandidates(m, f.models[0], { backend: 'vulkan' }, w).candidates.flatMap((c) => c.ctxSteps))
    expect(last(mach(far))).toBe(last(machine(f.vramBytes)))
  })
  it('w4m-7: unqualified identity (or a buffer not from the load log) never prunes; pre-qualification rows are ignored', () => {
    expect(budgetFor(mach([cap(6, 8, { qualified: false })]), 8 * G)).toMatchObject({ kind: 'estimated', source: expect.stringMatching(/1 with unverified adapter\/driver\/backend identity/) })
    expect(budgetFor(mach([cap(6, null)]), 8 * G).kind).toBe('estimated')
    const legacy = { ceilingBytes: 6 * G, modelId: 'm', ctx: 65536, kvType: 'f16', kvBytes: null, largestBufferBytes: 8 * G, observedAt: 1 } as unknown as VramBudgetObservation
    expect(budgetFor(mach([legacy]), 8 * G).kind).toBe('estimated')
  })
  it('a comparable qualified ceiling prunes; an allocation observed clean there is protected', () => {
    const plain = generateCandidates(machine(f.vramBytes), f.models[0], { backend: 'vulkan' }, WORKLOADS.long_context_coding).candidates
    const largest = (ctx: number) => Math.max(...Object.values((({ vramWeightsBytes, vramKvBytes, vramOverheadBytes }) => ({ vramWeightsBytes, vramKvBytes, vramOverheadBytes }))(estimateMemory(f.models[0], f.models[0].layers, ctx, 'f16'))))
    const top = Math.max(...plain[0].ctxSteps)
    const tightObs = cap(6, largest(top) / G)
    const tight = generateCandidates(mach([tightObs]), f.models[0], { backend: 'vulkan' }, WORKLOADS.long_context_coding).candidates
    expect(Math.max(...tight[0].ctxSteps)).toBeLessThan(top)
    expect(tight[0].skippedSteps.find((x) => x.skip?.resource === 'vram')!.reason).toMatch(/\(per-process budget 6\.0 GiB, measured\)$/)
    const clean: VramBudgetObservation = { ...tightObs, kind: 'clean', ceilingBytes: 7 * G, modelId: f.models[0].id, ctx: top, gpuLayers: f.models[0].layers, residentSharedBytes: 0.02 * G }
    // w4n-N3: a clean record without a valid shared reading never protects
    expect(generateCandidates(mach([tightObs, { ...clean, residentSharedBytes: null }]), f.models[0], { backend: 'vulkan' }, WORKLOADS.long_context_coding).candidates[0].ctxSteps).not.toContain(top)
    expect(generateCandidates(mach([tightObs, clean]), f.models[0], { backend: 'vulkan' }, WORKLOADS.long_context_coding).candidates[0].ctxSteps).toContain(top)
  })
  it('w4m-7: the key is the adapter PNP id + driver + backend build; ambiguous or unknown → unverified', () => {
    const gpu = (pnp: string, drv: string | null) => ({ name: 'RX 9070 XT', pnpDeviceId: pnp, driverVersion: drv, isIntegrated: false, dedicatedVramBytes: { value: 16 * G } })
    const p = (...g: ReturnType<typeof gpu>[]) => ({ gpus: { value: g } }) as unknown as SystemProfile
    expect(vramBudgetKey(p(gpu('PCI\\VEN_1002&DEV_7550\\1', '32.0.1')), 'vulkan', 'b11208')).toEqual({ key: 'pnp:PCI\\VEN_1002&DEV_7550\\1|drv:32.0.1|vulkan:b11208', verified: true })
    expect(vramBudgetKey(p(gpu('PCI\\A', '32.0.1'), gpu('PCI\\B', '32.0.1')), 'vulkan', 'b11208')!.verified).toBe(false) // same name, two adapters
    expect(vramBudgetKey(p(gpu('PCI\\A', null)), 'vulkan', 'b11208')!.verified).toBe(false)
    expect(vramBudgetKey(p(gpu('PCI\\A', '32.0.1')), 'vulkan', null)!.verified).toBe(false)
    expect(vramBudgetKey(p(gpu('PCI\\A', '32.0.1')), 'hip', 'b11208')!.key).not.toBe(vramBudgetKey(p(gpu('PCI\\A', '32.0.1')), 'vulkan', 'b11208')!.key)
  })
})

describe('planCandidates: backend axis', () => {
  const f = load('calib-8b-rx9070.json')
  const p = { gpus: { value: [{ name: 'RX 9070 XT', pnpDeviceId: 'PCI\\A', driverVersion: '32', isIntegrated: false, dedicatedVramBytes: { value: f.vramBytes, status: 'available', source: 't' } }] },
    ram: { value: { totalBytes: 31 * 1024 ** 3, availableBytes: 20 * 1024 ** 3 }, source: 't' }, cpu: { value: { physicalCores: 8 } } } as unknown as SystemProfile
  it('Vulkan ids are unchanged; HIP gets the same configs with |hip, backend and its own device; CPU baselines once', () => {
    const vkOnly = planCandidates(p, f.models[0], [{ kind: 'vulkan', device: 'Vulkan0', runtimeVersion: 'b1' }], WORKLOADS.coding)
    const both = planCandidates(p, f.models[0], [{ kind: 'vulkan', device: 'Vulkan0', runtimeVersion: 'b1' }, { kind: 'hip', device: 'ROCm0', runtimeVersion: 'b1' }], WORKLOADS.coding)
    const gpu = vkOnly.candidates.filter((c) => c.gpuLayers > 0)
    expect(both.candidates.filter((c) => !c.backend).map((c) => c.id)).toEqual(vkOnly.candidates.map((c) => c.id))
    const hip = both.candidates.filter((c) => c.backend === 'hip')
    expect(hip.map((c) => c.id)).toEqual(gpu.map((c) => `${c.id}|hip`))
    expect(hip.every((c) => c.device === 'ROCm0' && c.gpuLayers > 0)).toBe(true)
    expect(hip.map((c) => c.ctxSteps)).toEqual(gpu.map((c) => c.ctxSteps))
  })
})

describe('w4n-N5: identity verified at apply time', () => {
  it('an unverified current key turns previously qualified records advisory', () => {
    const o = { kind: 'capacity', qualified: true, ceilingBytes: 6 * 1024 ** 3 } as VramBudgetObservation
    expect(applicableObservations({ verified: false }, [o])[0].qualified).toBe(false)
    expect(applicableObservations({ verified: true }, [o])[0].qualified).toBe(true)
  })
})
