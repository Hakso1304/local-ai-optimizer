// Candidate configs per model + memory estimates for pruning (DESIGN §2.7, §6). Pure, deterministic.
// Estimates only prune; they are never shown as facts (kind 'estimated').
import type {
  CandidateConfig, CandidateSet, KvType, MachineLimits, Metric, ModelMeta, RejectedCandidate, WorkloadProfile
} from '../../shared/bench-types'
import type { SystemProfile } from '../../shared/types'

const MiB = 1024 ** 2
const GiB = 1024 ** 3

export const DEFAULT_CANDIDATE_RULES = {
  /** 128K only runs where declared ctx and the VRAM estimate allow; otherwise it documents the memory bound. */
  ctxLadder: [2048, 4096, 8192, 16384, 32768, 65536, 131072],
  /** Ladder cap when the GGUF declares no context length. */
  unknownCtxMax: 8192,
  /** Covers the measured residual above model+KV+compute (0.2–0.9 GiB on 8B, largest at 64K). */
  vramMarginBytes: 1024 * MiB,
  /** The first step over the VRAM budget is kept (to observe the cliff) only if est ≤ budget × this. Calibration:
   *  estimate is 2–19% above measured per-PID VRAM on 8B (residual 0.2–0.9 GiB), so far-over steps would just spill. */
  keepOverVramMaxRatio: 1.15,
  /** A25: RAM kept free for the OS; configs whose RAM estimate exceeds available − reserve are skipped. */
  ramReserveBytes: 4 * GiB,
  partialFractions: [0.75, 0.5, 0.25],
  /** ngl=0 reference config for models this small even when full offload fits. 0 = off: calibration showed ngl 0 on
   *  the Vulkan build still offloads prefill matmuls (1.1K t/s, 1.8 GiB VRAM), so it is partial, not CPU-only.
   *  ngl=0 is still generated when there is no GPU or when no GPU config fits. */
  cpuOnlyMaxParams: 0,
  /** q8_0 KV variant only when both the workload target and the declared ctx reach this. */
  longContextMin: 32768,
  maxPerModel: 4,
  ubatch: 512
}
export type CandidateRules = typeof DEFAULT_CANDIDATE_RULES

// Sliding-window / hybrid / recurrent: KV formula is wrong for them, so KV is not used for pruning.
const LOW_CONFIDENCE_ARCHS = new Set(['gemma2', 'gemma3', 'gemma3n', 'qwen3next', 'mamba', 'rwkv6', 'rwkv7', 'jamba', 'granitehybrid', 'lfm2'])
const KV_BYTES: Record<KvType, number> = { f16: 2, q8_0: 34 / 32 }

const gib = (b: number) => `${(b / GiB).toFixed(1)} GiB`
const ctxK = (n: number) => (n % 1024 === 0 ? `${n / 1024}K` : String(n))
const num = (m: Metric) => (typeof m.value === 'number' && Number.isFinite(m.value) ? m.value : null)
const est = (value: number, source: string): Metric => ({ value, kind: 'estimated', source })

export const isLowConfidence = (m: ModelMeta) => m.slidingWindow !== null || LOW_CONFIDENCE_ARCHS.has(m.arch)

/** DESIGN §2.7. gpuLayers ≥ layers means full offload. RAM counts the whole mmap'd file (calibration: available RAM
 *  fell ≈ file size even at ngl 99), so the A25 pre-check is conservative; spill detection never uses it. */
export function estimateMemory(m: ModelMeta, gpuLayers: number, ctx: number, kv: KvType, ubatch = DEFAULT_CANDIDATE_RULES.ubatch) {
  const dk = m.keyLength ?? m.nEmbd / m.heads
  const dv = m.valueLength ?? dk
  const kvBytes = ctx * m.layers * m.headsKv * (dk + dv) * KV_BYTES[kv]
  const onGpu = Math.min(gpuLayers, m.layers)
  const wGpu = (m.fileBytes * onGpu) / m.layers
  // Calibration (b11208 Vulkan, fa on): compute buffers 61→91 MiB (1.5B, 2K→32K), 102→164 MiB (8B, 2K→64K) ≈
  // 32 MiB + ubatch·n_embd·32 B + 1 KiB/ctx token. The unexplained residual (0.2–0.9 GiB) is covered by
  // rules.vramMarginBytes in the budget, not here, so small models aren't over-estimated.
  const compute = 32 * MiB + ubatch * m.nEmbd * 32 + ctx * 1024
  return {
    vramBytes: onGpu > 0 ? wGpu + kvBytes + compute : 0,
    ramBytes: m.fileBytes + (onGpu === 0 ? kvBytes : 0) + 512 * MiB,
    kvBytes
  }
}

/** Largest non-integrated GPU with the most VRAM. gpuDevice comes from `llama-server --list-devices` mapping. */
export function machineFromProfile(p: SystemProfile, gpuDevice: string | null, vramInUseBytes?: number): MachineLimits {
  const NA = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
  const gpu = (p.gpus.value ?? []).filter((g) => !g.isIntegrated)
    .sort((a, b) => (b.dedicatedVramBytes.value ?? 0) - (a.dedicatedVramBytes.value ?? 0))[0]
  const v = gpu?.dedicatedVramBytes
  const ram = p.ram.value
  return {
    vramBytes: !gpu ? NA('no discrete GPU') : v?.status === 'available' && v.value ? { value: v.value, kind: 'declared', source: v.source } : NA(v?.error ?? 'VRAM size not reported'),
    vramInUseBytes: vramInUseBytes === undefined ? NA('not measured before launch') : { value: vramInUseBytes, kind: 'measured', source: 'GPU Adapter Memory\\Dedicated Usage' },
    ramTotalBytes: ram ? { value: ram.totalBytes, kind: 'declared', source: p.ram.source } : NA(p.ram.error ?? 'RAM not reported'),
    ramAvailableBytes: ram ? { value: ram.availableBytes, kind: 'measured', source: p.ram.source } : NA(p.ram.error ?? 'RAM not reported'),
    physicalCores: p.cpu.value?.physicalCores ?? 1,
    gpuDevice: gpu ? gpuDevice : null
  }
}

export function generateCandidates(
  machine: MachineLimits,
  model: ModelMeta,
  runtime: { backend: 'vulkan' | 'cuda' | 'cpu' },
  workload: WorkloadProfile,
  rules: CandidateRules = DEFAULT_CANDIDATE_RULES
): CandidateSet {
  const maxCtx = model.ctxTrain ?? rules.unknownCtxMax
  const ladder = rules.ctxLadder.filter((c) => c <= maxCtx)
  const aboveDeclared = rules.ctxLadder.filter((c) => c > maxCtx)
    .map((ctx) => ({ ctx, reason: model.ctxTrain ? `above declared context ${ctxK(model.ctxTrain)}` : `declared context unknown; capped at ${ctxK(maxCtx)}` }))
  const threads = machine.physicalCores
  const hasGpu = machine.gpuDevice !== null && runtime.backend !== 'cpu'
  const vram = num(machine.vramBytes)
  const gpuBudget = vram === null ? null : vram - (num(machine.vramInUseBytes) ?? 0) - rules.vramMarginBytes
  const ramBase = num(machine.ramAvailableBytes) ?? num(machine.ramTotalBytes)
  const ramBudget = ramBase === null ? null : ramBase - rules.ramReserveBytes
  const lowConf = isLowConfidence(model)
  const rejected: RejectedCandidate[] = []
  const candidates: CandidateConfig[] = []

  const idOf = (ngl: number, kv: KvType) => `${model.id}|ngl=${ngl >= model.layers ? 'all' : ngl}|kv=${kv}|t=${threads}`

  /** Returns the candidate, or null after recording the rejection. */
  const build = (ngl: number, kv: KvType): CandidateConfig | null => {
    const id = idOf(ngl, kv)
    if (!ladder.length) { rejected.push({ id, modelId: model.id, reason: `declared context ${maxCtx} is below the smallest step ${ctxK(rules.ctxLadder[0])}` }); return null }
    const steps: number[] = []
    const skipped = [...aboveDeclared]
    const notes: string[] = []
    let overVramKept = false
    for (const ctx of ladder) {
      const e = estimateMemory(model, ngl, ctx, kv, rules.ubatch)
      const vramNeed = lowConf ? e.vramBytes - e.kvBytes : e.vramBytes
      const ramNeed = lowConf && ngl === 0 ? e.ramBytes - e.kvBytes : e.ramBytes
      if (ramBudget !== null && ramNeed > ramBudget) {
        skipped.push({ ctx, reason: `skipped_memory: est. RAM ${gib(ramNeed)} > available − reserve ${gib(ramBudget)}` })
        continue
      }
      if (ngl > 0 && gpuBudget !== null && vramNeed > gpuBudget) {
        // Keep the first step over the VRAM estimate once: the cliff detector needs to see it, guards contain it.
        if (steps.length && !overVramKept && vramNeed <= gpuBudget * rules.keepOverVramMaxRatio) {
          overVramKept = true; steps.push(ctx); notes.push(`${ctxK(ctx)} is above the VRAM estimate; kept to observe the cliff`); continue
        }
        overVramKept = true // never skip a step and then run a larger one
        skipped.push({ ctx, reason: `est. VRAM ${gib(vramNeed)} > budget ${gib(gpuBudget)}` })
        continue
      }
      steps.push(ctx)
    }
    if (!steps.length) { rejected.push({ id, modelId: model.id, reason: skipped.find((s) => s.ctx === ladder[0])!.reason + ` at ${ctxK(ladder[0])}` }); return null }
    const e0 = estimateMemory(model, ngl, steps[0], kv, rules.ubatch)
    if (lowConf) notes.push('memory estimate is low-confidence for this architecture; KV not used for pruning')
    if (ngl > 0 && gpuBudget === null) notes.push('VRAM size unknown; not pruned by VRAM, runtime guards apply')
    if (ramBudget === null) notes.push('RAM size unknown; not pruned by RAM, runtime guards apply')
    const src = `DESIGN §2.7 at ${ctxK(steps[0])}`
    return {
      id, modelId: model.id, device: ngl > 0 ? machine.gpuDevice : null, gpuLayers: Math.min(ngl, model.layers), gpuLayersAll: ngl >= model.layers,
      kvType: kv, flashAttn: true, threads, ctxSteps: steps, skippedSteps: skipped.sort((a, b) => a.ctx - b.ctx),
      estVramBytes: est(e0.vramBytes, src), estRamBytes: est(e0.ramBytes, src), notes
    }
  }
  const add = (c: CandidateConfig | null) => { if (c) candidates.push(c) }

  if (hasGpu) {
    const full = build(model.layers, 'f16')
    add(full)
    if (full && workload.targetContext >= rules.longContextMin && maxCtx >= rules.longContextMin) add(build(model.layers, 'q8_0'))
    if (!full) {
      // Partial ladder: the largest fraction that fits, plus the next lower one.
      let kept = 0
      for (const f of rules.partialFractions) {
        if (kept === 2) break
        const c = build(Math.max(1, Math.floor(model.layers * f)), 'f16')
        if (c) { add(c); kept++ } else if (kept) break
      }
    }
  }
  if (!hasGpu || !candidates.length || (model.paramCount !== null && model.paramCount <= rules.cpuOnlyMaxParams)) {
    const c = build(0, 'f16')
    if (c && hasGpu) c.notes.push('ngl=0 on a GPU build still offloads prefill to the GPU (measured); treated as partial offload')
    add(c)
  }

  for (const c of candidates.splice(rules.maxPerModel)) rejected.push({ id: c.id, modelId: model.id, reason: `over the per-model cap of ${rules.maxPerModel}` })
  return { candidates, rejected }
}
