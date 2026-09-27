// Candidate configs per model + memory estimates for pruning (DESIGN §2.7, §6). Pure, deterministic.
// Estimates only prune; they are never shown as facts (kind 'estimated').
import type {
  CandidateConfig, CandidateSet, GpuBackendKind, KvType, MachineLimits, Metric, ModelMeta, PlannedSkip, PlanningSnapshot, RejectedCandidate, VramBudgetObservation, WorkloadProfile
} from '../../shared/bench-types'
import type { SystemProfile } from '../../shared/types'
import type { SessionRequest } from '../../shared/bench-events'

const MiB = 1024 ** 2
const GiB = 1024 ** 3

/** Bumped whenever planning (estimates, budgets, ordering) changes behaviour. */
export const CANDIDATE_RULES_VERSION = 'cand-1.5'

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
  /** Heavy-model mode (opt-in, SessionRequest.heavyMode): when full offload does not fit at all, benchmark a
   *  partial-offload ladder instead of rejecting the model. Off = full offload only (+ CPU-only without a GPU):
   *  calibration showed partial offload decodes -83 % (8B ngl 20), so it is never offered unasked. */
  heavyMode: false,
  /** Heavy mode checks RAM against the resident part (non-GPU weights + CPU KV + CPU compute) with this reserve
   *  (≥ 4 GiB — same floor as the runner; a 2 GiB reserve let a 27B CPU baseline starve the host). */
  heavyRamReserveBytes: 4 * GiB,
  /** Heavy mode skips the CPU baseline when the file exceeds this share of total RAM (it would fill RAM). */
  cpuBaselineMaxRamShare: 0.5,
  /** ngl=0 reference config for models this small even when full offload fits. 0 = off: calibration showed ngl 0 on
   *  the Vulkan build still offloads prefill matmuls (1.1K t/s, 1.8 GiB VRAM), so it is partial, not CPU-only.
   *  ngl=0 is still generated when there is no GPU. */
  cpuOnlyMaxParams: 0,
  /** q8_0 KV variant only when both the workload target and the declared ctx reach this. */
  longContextMin: 32768,
  maxPerModel: 4,
  /** VRAM used by other apps when the pre-launch reading is unavailable (measured idle here: 1.2–1.4 GiB). Never 0. */
  vramInUseUnknownBytes: 1.5 * GiB,
  /** Heavy mode adds the -nkvo rung only when putting the target ctx's KV on the GPU costs ≥ this many layers vs the
   *  smallest ctx. Measured at 2K: -nkvo lost to simply dropping ~5 layers (Qwen3.8 57-nkvo 7.4 vs 50 layers 10.7
   *  t/s; Gemma-4 26-nkvo 27.0 vs 21 layers 38.5 t/s) — useful only when the KV itself is what doesn't fit. */
  nkvoMinLayerGain: 8,
  ubatch: 512
}
export type CandidateRules = typeof DEFAULT_CANDIDATE_RULES

/** The rules a session request implies. The runner and anything re-planning a session (main's planFor) must use this
 *  so both produce identical configIds. */
export const rulesForRequest = (req: Pick<SessionRequest, 'candidateRules' | 'heavyMode'>): CandidateRules =>
  ({ ...DEFAULT_CANDIDATE_RULES, ...req.candidateRules, heavyMode: req.heavyMode ?? false })

// Sliding-window / hybrid / recurrent archs: without their layout keys the plain formula is an UPPER BOUND (all layers
// full attention, full ctx). We keep it for pruning — never treat unknown KV as 0 — and say so in the notes.
const LOW_CONFIDENCE_ARCHS = new Set(['gemma2', 'gemma3', 'gemma3n', 'gemma4', 'qwen3next', 'qwen35', 'mamba', 'rwkv6', 'rwkv7', 'jamba', 'granitehybrid', 'lfm2'])
const KV_BYTES: Record<KvType, number> = { f16: 2, q8_0: 34 / 32 }
/** q8_0 / f16 KV size ratio used by the estimator (shared with remedy feasibility, R5). */
export const KV_Q8_OVER_F16 = KV_BYTES.q8_0 / KV_BYTES.f16

const gib = (b: number) => `${(b / GiB).toFixed(1)} GiB`
const ctxK = (n: number) => (n % 1024 === 0 ? `${n / 1024}K` : String(n))
const num = (m: Metric) => (typeof m.value === 'number' && Number.isFinite(m.value) ? m.value : null)
const est = (value: number, source: string): Metric => ({ value, kind: 'estimated', source })

/** True when the KV estimate is only an upper bound: an SWA/hybrid arch whose layout keys are missing. */
export const isLowConfidence = (m: ModelMeta) =>
  (m.slidingWindow !== null && !m.slidingWindowPattern) || (LOW_CONFIDENCE_ARCHS.has(m.arch) && !m.slidingWindowPattern && !m.fullAttentionInterval)

/** KV bytes per sequence at ctx (f16 = 2 B/elem, q8_0 = 34/32), layer by layer when the layout is declared.
 *  source says which formula ran. unknown = the heads/head dim needed for any formula are missing. */
export function kvLayout(m: ModelMeta, ctx: number, kv: KvType, ubatch = DEFAULT_CANDIDATE_RULES.ubatch): { bytes: number; perLayer: number[]; source: string; unknown: boolean } {
  const dk = m.keyLength ?? m.nEmbd / m.heads
  const dv = m.valueLength ?? dk
  const heads = (i: number) => m.headsKvPerLayer?.[i] ?? m.headsKv
  if (!(m.headsKv > 0) || !(dk > 0) || !Number.isFinite(dk)) return { bytes: 0, perLayer: [], source: 'KV size unknown (no head count / head dim in GGUF)', unknown: true }
  const layout = !!(m.fullAttentionInterval || m.slidingWindowPattern || m.headsKvPerLayer)
  const perLayer: number[] = []
  let attnLayers = 0
  for (let i = 0; i < m.layers; i++) {
    if (m.fullAttentionInterval && (i + 1) % m.fullAttentionInterval !== 0) { perLayer.push(0); continue } // recurrent layer: no KV
    attnLayers++
    const swa = !!m.slidingWindowPattern?.[i] && m.slidingWindow !== null
    const tokens = swa ? Math.min(ctx, m.slidingWindow! + ubatch) : ctx
    const k = swa ? (m.keyLengthSwa ?? dk) : dk
    const v = swa ? (m.valueLengthSwa ?? dv) : dv
    perLayer.push(tokens * heads(i) * (k + v) * KV_BYTES[kv])
  }
  return {
    bytes: perLayer.reduce((a, b) => a + b, 0),
    perLayer,
    unknown: false,
    source: layout ? `declared layout: ${attnLayers}/${m.layers} attention layers${m.slidingWindowPattern ? `, sliding window ${m.slidingWindow}` : ''}`
      : isLowConfidence(m) ? `fallback upper bound: no per-arch KV layout (${m.arch} may use hybrid/linear-attention or sliding-window layers)`
        : 'standard: all layers full attention'
  }
}

/** DESIGN §2.7. gpuLayers ≥ layers means full offload. KV follows its layers (calibration, 8B ngl 20/33 @8K:
 *  KV CPU 416 + Vulkan0 608 MiB) unless kvOnGpu=false (-nkvo: all KV in RAM).
 *  ramBytes = ramResidentBytes = what must stay in RAM (non-GPU weights + CPU KV + 0.5 GiB). Checked against live
 *  available RAM after the previous server is unloaded (session.ts); the live floor guard is the safety net. */
export function estimateMemory(m: ModelMeta, gpuLayers: number, ctx: number, kv: KvType, ubatch = DEFAULT_CANDIDATE_RULES.ubatch, kvOnGpu = true) {
  const { bytes: kvBytes, perLayer, source: kvSource, unknown: kvUnknown } = kvLayout(m, ctx, kv, ubatch)
  const onGpu = Math.min(gpuLayers, m.layers)
  const wGpu = (m.fileBytes * onGpu) / m.layers
  // Calibration (b11208 Vulkan, fa on): compute buffers 61→91 MiB (1.5B, 2K→32K), 102→164 MiB (8B, 2K→64K) ≈
  // 32 MiB + ubatch·n_embd·32 B + 1 KiB/ctx token. The unexplained residual (0.2–0.9 GiB) is covered by
  // rules.vramMarginBytes in the budget, not here, so small models aren't over-estimated.
  const compute = 32 * MiB + ubatch * m.nEmbd * 32 + ctx * 1024
  // Partial offload (heavy run, Qwen3.8-27B ngl 62: 2.31 GiB shared spill where this estimate said it fits): the
  // output projection (n_vocab × n_embd at the file's bits/weight) stays GPU-resident, and the logits / graph-split
  // buffers scale with the vocab (248K here). Counted in full on top of the layer share — conservative; full offload
  // keeps the calibrated −2…+3 % formula.
  const partial = onGpu > 0 && onGpu < m.layers
  const bpw = m.paramCount ? (m.fileBytes * 8) / m.paramCount : 4.5
  const outputBytes = partial ? (m.nVocab * m.nEmbd * bpw) / 8 : 0
  const logits = partial ? ubatch * m.nVocab * 4 : 0
  // llama.cpp offloads the LAST onGpu layers: sum their own KV (per-layer heads / hybrid layers differ), not a share.
  const kvGpu = kvOnGpu ? perLayer.slice(m.layers - onGpu).reduce((a, b) => a + b, 0) : 0
  const kvCpu = kvBytes - kvGpu
  return {
    vramBytes: onGpu > 0 ? wGpu + kvGpu + compute + outputBytes + logits : 0,
    // Resident RAM: weights NOT on the GPU + CPU-side KV + 0.5 GiB. mmap'd pages of offloaded weights are clean and
    // reclaimable (they lower "available" while loaded — the in-step guard credits them), so they are not counted.
    // ngl 0: + 1.5 GiB CPU compute buffers / working set (a 27B CPU baseline exceeded file + KV + 0.5 GiB).
    ramBytes: m.fileBytes - wGpu + kvCpu + 512 * MiB + (onGpu === 0 ? 1.5 * GiB : 0),
    ramResidentBytes: m.fileBytes - wGpu + kvCpu + 512 * MiB + (onGpu === 0 ? 1.5 * GiB : 0),
    kvBytes,
    kvSource,
    kvUnknown,
    /** VRAM breakdown (F9: whether a KV-only remedy can fit): weights incl. output, KV on GPU, compute + logits. */
    vramWeightsBytes: onGpu > 0 ? wGpu + outputBytes : 0,
    vramKvBytes: kvGpu,
    vramOverheadBytes: onGpu > 0 ? compute + logits : 0
  }
}

const pickGpu = (p: SystemProfile) => (p.gpus.value ?? []).filter((g) => !g.isIntegrated)
  .sort((a, b) => (b.dedicatedVramBytes.value ?? 0) - (a.dedicatedVramBytes.value ?? 0))[0]

/** The per-process VRAM ceiling is specific to GPU, driver, backend build and allocation pattern (13.25 GiB for a
 *  14B vs 11.6 GiB for 8B f16 64K on one card; a ROCm/HIP backend used the full 16 GB on another) — observations
 *  are kept per this key, never generalised. */
export function vramBudgetKey(p: SystemProfile, backend: string, runtimeVersion: string | null | undefined): { key: string; verified: boolean } | null {
  const discrete = (p.gpus.value ?? []).filter((g) => !g.isIntegrated)
  const g = pickGpu(p)
  if (!g) return null
  // Stable identity = the adapter's PNP device id (not its display name); unambiguous only with one discrete GPU.
  const verified = discrete.length === 1 && !!g.pnpDeviceId && !!g.driverVersion && !!runtimeVersion
  return { key: `pnp:${g.pnpDeviceId || '?'}|drv:${g.driverVersion ?? '?'}|${backend}:${runtimeVersion ?? '?'}`, verified }
}

/** Identity is verified again when observations are APPLIED (review-w4n N5): with an ambiguous or unverified current
 *  identity, even previously qualified records are advisory. */
export function applicableObservations(key: { verified: boolean }, rows: VramBudgetObservation[]): VramBudgetObservation[] {
  return key.verified ? [...rows] : rows.map((o) => ({ ...o, qualified: false }))
}

/** Fallback share of the adapter total while nothing was measured (cal-2026-09-27: spill began at 83 % / 73 %). */
export const VRAM_BUDGET_FALLBACK_SHARE = 0.8
/** Observations whose largest single buffer is within this ratio of the planned one are "comparable". */
export const VRAM_BUDGET_COMPARABLE_RATIO = 1.25

/** Per-process budget for an allocation whose largest individual buffer is `largestBufferBytes` (planning estimate,
 *  matched against observations' load-log buffers). MEASURED only from qualified capacity observations with a
 *  comparable buffer (min of them); any other capacity observation is advisory → estimated, never prunes; none at
 *  all → 80 % of the total, estimated. Observations are already scoped to one adapter + driver + backend build. */
export function budgetFor(machine: MachineLimits, largestBufferBytes: number | null): Metric {
  const total = num(machine.vramBytes)
  const cap = (machine.vramBudgetObservations ?? []).filter((o) => o.kind === 'capacity') // pre-qualification rows have no kind
  const near = (o: VramBudgetObservation) => largestBufferBytes !== null && largestBufferBytes > 0 && o.largestBufferBytes != null && o.largestBufferBytes > 0
    && Math.max(o.largestBufferBytes / largestBufferBytes, largestBufferBytes / o.largestBufferBytes) <= VRAM_BUDGET_COMPARABLE_RATIO
  const min = (xs: VramBudgetObservation[]) => Math.min(...xs.map((o) => o.ceilingBytes))
  const comparable = cap.filter((o) => o.qualified && near(o))
  if (comparable.length) return { value: min(comparable), kind: 'measured', source: `per-process ceiling measured on this adapter/driver/backend: min of ${comparable.length} qualified observation(s) with a comparable largest buffer` }
  if (cap.length) {
    const unq = cap.filter((o) => !o.qualified).length
    return { value: min(cap), kind: 'estimated', source: `advisory: ${cap.length} capacity observation(s) ${unq ? `(${unq} with unverified adapter/driver/backend identity) ` : ''}not comparable to this allocation's largest buffer — not applied to planning` }
  }
  return total === null ? { value: null, kind: 'unavailable', reason: 'VRAM size unknown' }
    : { value: total * VRAM_BUDGET_FALLBACK_SHARE, kind: 'estimated', source: `no measured budget on this machine yet; assuming ${Math.round(VRAM_BUDGET_FALLBACK_SHARE * 100)} % of the adapter total (some GPUs/backends allow 100 %)` }
}

/** Largest non-integrated GPU with the most VRAM. gpuDevice comes from `llama-server --list-devices` mapping. */
export function machineFromProfile(p: SystemProfile, gpuDevice: string | null, vramInUseBytes?: number, budgetObservations: VramBudgetObservation[] = []): MachineLimits {
  const NA = (reason: string): Metric => ({ value: null, kind: 'unavailable', reason })
  const gpu = pickGpu(p)
  const v = gpu?.dedicatedVramBytes
  const ram = p.ram.value
  // Explicit arg wins; else the profile's pre-launch reading. Runner and main's planFor both come through here.
  const inUse = vramInUseBytes ?? (p.vramInUse?.status === 'available' ? p.vramInUse.value ?? undefined : undefined)
  const m: MachineLimits = {
    vramBytes: !gpu ? NA('no discrete GPU') : v?.status === 'available' && v.value ? { value: v.value, kind: 'declared', source: v.source } : NA(v?.error ?? 'VRAM size not reported'),
    vramInUseBytes: inUse === undefined ? NA(p.vramInUse?.error ?? 'not measured before launch') : { value: inUse, kind: 'measured', source: p.vramInUse?.source ?? 'GPU Adapter Memory\\Dedicated Usage' },
    ramTotalBytes: ram ? { value: ram.totalBytes, kind: 'declared', source: p.ram.source } : NA(p.ram.error ?? 'RAM not reported'),
    ramAvailableBytes: ram ? { value: ram.availableBytes, kind: 'measured', source: p.ram.source } : NA(p.ram.error ?? 'RAM not reported'),
    physicalCores: p.cpu.value?.physicalCores ?? 1,
    gpuDevice: gpu ? gpuDevice : null,
    vramBudgetObservations: budgetObservations
  }
  return { ...m, vramEffectiveBudgetBytes: budgetFor(m, null) }
}

export function generateCandidates(
  machine: MachineLimits,
  model: ModelMeta,
  runtime: { backend: GpuBackendKind | 'cpu' },
  workload: WorkloadProfile,
  rules: CandidateRules = DEFAULT_CANDIDATE_RULES
): CandidateSet {
  const maxCtx = model.ctxTrain ?? rules.unknownCtxMax
  const ladder = rules.ctxLadder.filter((c) => c <= maxCtx)
  const aboveDeclared = rules.ctxLadder.filter((c) => c > maxCtx)
    .map((ctx) => ({ ctx, reason: model.ctxTrain ? `above declared context ${ctxK(model.ctxTrain)}` : `declared context unknown; capped at ${ctxK(maxCtx)}`, skip: { resource: 'declared' as const, ruleId: 'I-2.3' } }))
  const threads = machine.physicalCores
  const hasGpu = machine.gpuDevice !== null && runtime.backend !== 'cpu'
  const vram = num(machine.vramBytes)
  const gpuBudget = vram === null ? null : vram - (num(machine.vramInUseBytes) ?? rules.vramInUseUnknownBytes) - rules.vramMarginBytes
  const ramBase = num(machine.ramAvailableBytes) ?? num(machine.ramTotalBytes)
  const ramBudget = ramBase === null ? null : ramBase - rules.ramReserveBytes
  const heavyRamBudget = ramBase === null ? null : ramBase - Math.max(4 * GiB, rules.heavyRamReserveBytes)
  const lowConf = isLowConfidence(model)
  const inUse = num(machine.vramInUseBytes)
  const planning: PlanningSnapshot = {
    vramTotalBytes: vram,
    vramInUse: inUse !== null ? machine.vramInUseBytes : { value: rules.vramInUseUnknownBytes, kind: 'estimated', source: `assumed default (${machine.vramInUseBytes.reason ?? 'reading unavailable'})` },
    planningVramBudgetBytes: vram === null ? null : vram - (inUse ?? rules.vramInUseUnknownBytes),
    planningReserveBytes: rules.vramMarginBytes,
    ramAvailableBytes: ramBase, ramReserveBytes: rules.ramReserveBytes, candidateRulesVersion: CANDIDATE_RULES_VERSION
  }
  const rejected: RejectedCandidate[] = []
  const candidates: CandidateConfig[] = []
  const largestOf = (e: ReturnType<typeof estimateMemory>) => Math.max(e.vramWeightsBytes, e.vramKvBytes, e.vramOverheadBytes) || null
  /** min(total − in use − reserve, per-process budget at this allocation's largest buffer). Only a MEASURED budget
   *  prunes: the 80 % fallback is reported, not applied (cal-2026-09-27: 27B ngl 55 @8K ran clean at 13.03 GiB
   *  dedicated, above 0.8 × 15.92 GiB — the fallback would have pruned a measured-clean config). */
  const vramBudgetAt = (e: ReturnType<typeof estimateMemory>, ngl: number, ctx: number, kv: KvType): number | null => {
    // An allocation that ran clean here is never pruned by a per-process ceiling (it demonstrably fits).
    const ranClean = (machine.vramBudgetObservations ?? []).some((o) => o.kind === 'clean' && o.qualified && typeof o.residentSharedBytes === 'number' && o.modelId === model.id && o.kvType === kv && o.gpuLayers === Math.min(ngl, model.layers) && o.ctx === ctx)
    const b = budgetFor(machine, largestOf(e)), eb = b.kind === 'measured' && !ranClean ? num(b) : null
    return gpuBudget === null ? null : eb === null ? gpuBudget : Math.min(gpuBudget, eb)
  }

  const idOf = (ngl: number, kv: KvType, kvOnGpu = true) => `${model.id}|ngl=${ngl >= model.layers ? 'all' : ngl}|kv=${kv}|t=${threads}${kvOnGpu ? '' : '|nkvo'}${runtime.backend === 'hip' ? '|hip' : ''}`

  /** Returns the candidate, or null after recording the rejection. heavy: resident-RAM check, no keep-over step. */
  const build = (ngl: number, kv: KvType, heavy?: { kvOnGpu: boolean }): CandidateConfig | null => {
    const kvOnGpu = heavy?.kvOnGpu ?? true
    const id = idOf(ngl, kv, kvOnGpu)
    if (!ladder.length) { rejected.push({ id, modelId: model.id, reason: `declared context ${maxCtx} is below the smallest step ${ctxK(rules.ctxLadder[0])}` }); return null }
    const steps: number[] = []
    const skipped: { ctx: number; reason: string; skip?: PlannedSkip }[] = [...aboveDeclared]
    const notes: string[] = []
    let overVramKept = false
    for (const ctx of ladder) {
      const e = estimateMemory(model, ngl, ctx, kv, rules.ubatch, kvOnGpu)
      const vramNeed = e.vramBytes
      const stepBudget = vramBudgetAt(e, ngl, ctx, kv)
      const ramNeed = heavy ? e.ramResidentBytes : e.ramBytes
      const rb = heavy ? heavyRamBudget : ramBudget
      if (rb !== null && ramNeed > rb) {
        skipped.push({ ctx, reason: `skipped_memory: est. RAM ${gib(ramNeed)} > available − reserve ${gib(rb)}`, skip: { resource: 'ram', estimateBytes: ramNeed, budgetBytes: rb, ruleId: 'I-2.3' } })
        continue
      }
      if (ngl > 0 && stepBudget !== null && vramNeed > stepBudget) {
        // Keep the first step over the VRAM estimate once: the cliff detector needs to see it, guards contain it.
        if (!heavy && steps.length && !overVramKept && vramNeed <= stepBudget * rules.keepOverVramMaxRatio) {
          overVramKept = true; steps.push(ctx); notes.push(`${ctxK(ctx)} is above the VRAM estimate; kept to observe the cliff`); continue
        }
        overVramKept = true // never skip a step and then run a larger one
        const eb = budgetFor(machine, largestOf(e))
        const why = stepBudget < gpuBudget! && eb.value !== null ? ` (per-process budget ${gib(eb.value)}, ${eb.kind})` : ''
        skipped.push({ ctx, reason: `est. VRAM ${gib(vramNeed)} > budget ${gib(stepBudget)}${why}`, skip: { resource: 'vram', estimateBytes: vramNeed, budgetBytes: stepBudget, ruleId: 'I-2.3', weightsBytes: e.vramWeightsBytes, kvBytes: e.vramKvBytes, overheadBytes: e.vramOverheadBytes } })
        continue
      }
      steps.push(ctx)
    }
    if (!steps.length) { rejected.push({ id, modelId: model.id, reason: skipped.find((s) => s.ctx === ladder[0])!.reason + ` at ${ctxK(ladder[0])}` }); return null }
    const e0 = estimateMemory(model, ngl, steps[0], kv, rules.ubatch, kvOnGpu)
    if (e0.kvUnknown) notes.push('KV size unknown: pruned on weights only — runtime guards apply')
    else if (lowConf || e0.kvSource.startsWith('declared')) notes.push(`KV estimate: ${e0.kvSource}`)
    if (ngl > 0 && gpuBudget === null) notes.push('VRAM size unknown; not pruned by VRAM, runtime guards apply')
    if (ngl > 0 && gpuBudget !== null && machine.vramInUseBytes.value === null) notes.push(`VRAM in use by other apps unknown (${machine.vramInUseBytes.reason ?? 'unavailable'}); budget assumes ${gib(rules.vramInUseUnknownBytes)}`)
    if (ramBudget === null) notes.push('RAM size unknown; not pruned by RAM, runtime guards apply')
    const src = `DESIGN §2.7 at ${ctxK(steps[0])}`
    return {
      id, ...(runtime.backend === 'hip' || runtime.backend === 'cuda' ? { backend: runtime.backend } : {}), modelId: model.id, device: ngl > 0 ? machine.gpuDevice : null, gpuLayers: Math.min(ngl, model.layers), gpuLayersAll: ngl >= model.layers,
      kvType: kv, flashAttn: true, threads, ctxSteps: steps, skippedSteps: skipped.sort((a, b) => a.ctx - b.ctx),
      estVramBytes: est(e0.vramBytes, src), estRamBytes: est(heavy ? e0.ramResidentBytes : e0.ramBytes, src), notes,
      ...(kvOnGpu ? {} : { kvOffload: false }), planning: ngl > 0 ? { ...planning, effectiveBudget: budgetFor(machine, largestOf(estimateMemory(model, ngl, steps[steps.length - 1], kv, rules.ubatch, kvOnGpu))) } : planning
    }
  }

  /** Heavy mode: the largest ngl whose est. VRAM (weights share + KV share + compute) fits the budget at ctx. */
  const maxNgl = (ctx: number, kvOnGpu: boolean) => {
    for (let n = model.layers - 1; n >= 1; n--) if (estimateMemory(model, n, ctx, 'f16', rules.ubatch, kvOnGpu).vramBytes <= gpuBudget!) return n
    return 0
  }
  const heavyLadder = () => {
    if (gpuBudget === null || !ladder.length) { rejected.push({ id: idOf(0, 'f16'), modelId: model.id, reason: 'heavy mode needs a known VRAM size' }); return }
    const target = ladder.filter((c) => c <= workload.targetContext).at(-1) ?? ladder[0]
    const why = model.fileBytes > gpuBudget ? `weights ${gib(model.fileBytes)} > VRAM budget ${gib(gpuBudget)}` : `weights + KV > VRAM budget ${gib(gpuBudget)}`
    // ponytail: 4 fixed probes (most layers at the smallest ctx, at the target ctx, KV in RAM, CPU baseline);
    // a finer ngl sweep only if measurements show it matters.
    const kvUnknown = estimateMemory(model, model.layers, target, 'f16', rules.ubatch).kvUnknown
    // KV size unknown → only the conservative probes: max ngl at the smallest ctx, KV in RAM, CPU baseline.
    const plans: [number, boolean][] = kvUnknown
      ? [[maxNgl(ladder[0], true), true], [maxNgl(ladder[0], false), false], [0, true]]
      // Two rungs at the target ctx (max ngl and max ngl − 4, so a spill at the edge still leaves a clean config the
      // scores can pick), KV-in-RAM, CPU baseline. The old max-ngl@2K probe is dropped to keep ≤ 4 configs.
      : [
          [maxNgl(target, true), true], [Math.max(1, maxNgl(target, true) - 4), true],
          ...(maxNgl(ladder[0], true) - maxNgl(target, true) >= rules.nkvoMinLayerGain ? [[maxNgl(target, false), false] as [number, boolean]] : []),
          [0, true]
        ]
    const seen = new Set<string>()
    const ramTotal = num(machine.ramTotalBytes)
    plans.forEach(([n, kvOnGpu], i) => {
      if (n === 0 && i < plans.length - 1) return // nothing fits on the GPU for this probe; the CPU baseline covers ngl 0
      if (n === 0 && ramTotal !== null && model.fileBytes > ramTotal * rules.cpuBaselineMaxRamShare) {
        rejected.push({ id: idOf(0, 'f16'), modelId: model.id, reason: `CPU baseline skipped: model is >${Math.round(rules.cpuBaselineMaxRamShare * 100)}% of system RAM` })
        return
      }
      const key = `${n}|${kvOnGpu}`
      if (seen.has(key)) return
      seen.add(key)
      const c = build(n, 'f16', { kvOnGpu })
      if (!c) return
      c.expectDegraded = true
      c.mmap = false
      c.notes.push('loaded without mmap (-lm none): host RAM ≈ CPU-side layers + KV')
      c.degradedReason = n === 0 ? `${why}; CPU-only baseline (0/${model.layers} layers on GPU)`
        : `${why}; ${n}/${model.layers} layers on GPU${kvOnGpu ? '' : ', KV cache in system RAM (-nkvo); KV on CPU: slower decode than dropping ~5 layers at short ctx; useful only for long context'}`
      if (kvUnknown) c.degradedReason += '; KV size unknown'
      c.notes.push(`heavy mode: ${c.degradedReason}`)
      if (model.expertCount) c.notes.push(`MoE (${model.expertCount} experts, ${model.expertUsedCount ?? '?'} active per token): active parameters are much smaller, so partial offload costs less decode speed than on a dense model (measured: Gemma-4-26B-A4B 38.5 t/s at 21/30 layers vs dense Qwen3.8-27B 10.7 t/s at 50/65)`)
      add(c)
    })
  }
  const add = (c: CandidateConfig | null) => { if (c) candidates.push(c) }

  if (hasGpu) {
    const full = build(model.layers, 'f16')
    add(full)
    // Long context: when the f16 full offload can't reach the target ctx within VRAM, add KV q8_0 (≈ half the KV) and
    // KV-in-RAM (-nkvo) full-offload variants, each noted. Partial offload stays heavy-mode only.
    const tctx = ladder.filter((c) => c <= workload.targetContext).at(-1)
    const shortOfTarget = !!full && tctx !== undefined && !full.ctxSteps.includes(tctx)
    if (full && ((workload.targetContext >= rules.longContextMin && maxCtx >= rules.longContextMin) || shortOfTarget)) {
      const q8 = build(model.layers, 'q8_0')
      q8?.notes.push('long-context variant: KV cache q8_0 (-ctk/-ctv q8_0, about half the KV size)')
      add(q8)
    }
    if (full && shortOfTarget) {
      const nk = build(model.layers, 'f16', { kvOnGpu: false })
      nk?.notes.push(`long-context variant: KV cache in system RAM (-nkvo) so ${ctxK(tctx!)} fits; decode is slower`)
      add(nk)
    }
    if (!full && rules.heavyMode) heavyLadder()
    else if (!full) {
      const r = rejected.find((x) => x.id === idOf(model.layers, 'f16'))
      if (r) r.reason += '; full GPU offload does not fit — enable heavy-model mode to benchmark partial offload'
    }
  }
  if (!hasGpu || (candidates.length > 0 && !rules.heavyMode && model.paramCount !== null && model.paramCount <= rules.cpuOnlyMaxParams)) add(build(0, 'f16'))

  for (const c of candidates.splice(rules.maxPerModel)) rejected.push({ id: c.id, modelId: model.id, reason: `over the per-model cap of ${rules.maxPerModel}` })
  return { candidates, rejected }
}

/** One installed backend as the planner sees it: its own device id (Vulkan0 / ROCm0), build and budget observations. */
export interface PlannedBackend { kind: GpuBackendKind | 'cpu'; device: string | null; runtimeVersion: string | null; observations?: VramBudgetObservation[] }

/** The same configs for every installed backend (runner and main's planFor both call this, so ids agree).
 *  backends[0] is the primary; CPU-only configs (ngl 0) are backend-independent and come from it alone. */
export function planCandidates(p: SystemProfile, model: ModelMeta, backends: PlannedBackend[], workload: WorkloadProfile, rules: CandidateRules = DEFAULT_CANDIDATE_RULES): CandidateSet {
  const out: CandidateSet = { candidates: [], rejected: [] }
  backends.forEach((b, i) => {
    const set = generateCandidates(machineFromProfile(p, b.device, undefined, b.observations ?? []), model, { backend: b.kind }, workload, rules)
    out.candidates.push(...(i === 0 ? set.candidates : set.candidates.filter((c) => c.gpuLayers > 0)))
    out.rejected.push(...(i === 0 ? set.rejected : set.rejected.filter((r) => !r.id.includes('|ngl=0|'))))
  })
  return out
}
