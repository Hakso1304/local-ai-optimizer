// "Recommended for this PC" on the Download page. Pure (no Node APIs) so the renderer can import it.
// Budget rules are the benchmark planner's (DEFAULT_CANDIDATE_RULES); sizes come from the repo's real GGUF files when
// the caller has fetched them, else from the repo name. Everything here is an estimate — the benchmark is the answer.
import type { SystemProfile } from '../../shared/types'
import type { WorkloadProfile } from '../../shared/bench-types'
import { DEFAULT_CANDIDATE_RULES, pickGpu } from '../benchmark/candidates'
import type { HfGgufFile, HfModel } from './hf'

const GB = 1e9

/** "Qwen3-30B-A3B-Instruct-GGUF" → 30, "Mixtral-8x7B" → 56, "LFM2.5-230M" → 0.23; null when the name has no size.
 *  Total params (not active), since every weight has to be resident. */
export function paramsB(repoId: string): number | null {
  const name = repoId.split('/').pop()!
  const moe = /(?:^|[-_.])(\d+)x(\d+(?:\.\d+)?)b(?=$|[-_.])/i.exec(name)
  if (moe) return Number(moe[1]) * Number(moe[2])
  const m = /(?:^|[-_.])(\d+(?:\.\d+)?)([bm])(?=$|[-_.])/i.exec(name)
  return m ? Number(m[1]) / (m[2].toLowerCase() === 'm' ? 1000 : 1) : null
}

/** Params read per token: "30B-A3B" → 3, "8x7B" → 2 experts ≈ 14; dense → total. Drives the speed estimate only. */
export function activeB(repoId: string, total: number): number {
  const name = repoId.split('/').pop()!
  const a = /(?:^|[-_.])a(\d+(?:\.\d+)?)b(?=$|[-_.])/i.exec(name)
  if (a) return Math.min(total, Number(a[1]))
  const moe = /(?:^|[-_.])(\d+)x(\d+(?:\.\d+)?)b(?=$|[-_.])/i.exec(name)
  return moe ? Math.min(total, 2 * Number(moe[2])) : total
}

// ponytail: √params fit to llama/qwen GQA layouts (8B → 113 KB/token vs 128 KB real, 32B → 226 vs 262 KB; f16 K+V).
// Read the GGUF header remotely (HF serves ranges) for exact layers/heads when this misleads on a family.
export const kvBytesPerToken = (b: number): number => 40_000 * Math.sqrt(b)

/** Q4_K_M weights from the name alone (≈ 4.9 bits/weight) when the repo's files were not fetched. */
export const q4WeightBytes = (b: number): number => b * GB * 0.61

export interface Budget {
  /** Bytes the GPU can hold: dedicated VRAM − in use − margin, or the shared-RAM budget on an integrated GPU. */
  gpu: number
  /** true = integrated GPU: its memory IS system RAM (gpu === ram), one pool, no separate VRAM ceiling. */
  shared: boolean
  /** System RAM minus the OS reserve. */
  ram: number
  ctx: number
}

/** Same rules as the benchmark planner (candidates.ts budgetFor / ramReserveBytes), so what is recommended here is
 *  what a benchmark would plan: VRAM in use comes from the scan when measured, else the planner's default. */
export function budget(p: SystemProfile, ctx: number): Budget {
  const R = DEFAULT_CANDIDATE_RULES
  const g = pickGpu(p)
  const ram = Math.max(0, (p.ram.value?.totalBytes ?? 0) - R.ramReserveBytes)
  if (g?.isIntegrated) return { gpu: ram, shared: true, ram, ctx }
  const v = g && g.dedicatedVramBytes.status === 'available' ? g.dedicatedVramBytes.value ?? 0 : 0
  const inUse = p.vramInUse?.status === 'available' && p.vramInUse.value !== null ? p.vramInUse.value : R.vramInUseUnknownBytes
  return { gpu: v ? Math.max(0, v - inUse - R.vramMarginBytes) : 0, shared: false, ram, ctx }
}

/** Quality order for picking a file: the best quant that fits wins. Only these are ever recommended: F16/BF16/F32
 *  are 2× Q8_0 for no practical gain, and sub-2-bit quants (TQ1_0, IQ1_*, "UD-…" 1.58-bit) degrade too far. */
export const QUANT_PREF = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 'Q5_0', 'Q4_K_M', 'Q4_K_S', 'IQ4_XS', 'IQ4_NL', 'Q4_0', 'MXFP4', 'PQ2_0', 'PTQ1_0', 'Q3_K_M', 'Q3_K_L', 'IQ3_M', 'Q3_K_S', 'IQ3_XS', 'Q2_K', 'IQ2_M']
const quantRank = (q: string) => { const i = QUANT_PREF.indexOf(q); return i < 0 ? QUANT_PREF.length : i }

export interface FileChoice {
  /** Path to download (the first shard of a split model). */
  path: string
  quant: string
  /** Whole model (shards summed). */
  sizeBytes: number
  shards: number
  /** PrismML ternary (PQ2_0 / PTQ1_0): loads only on the 'prism' backend. */
  prism: boolean
}

/** One entry per whole model in the repo: shards summed, mmproj and F16+ dropped, ternary flagged. */
export function fileOptions(files: HfGgufFile[]): FileChoice[] {
  const out = new Map<string, FileChoice>()
  for (const f of files) {
    if (/mmproj/i.test(f.path)) continue
    const ternary = /(?:^|[-_.])(PQ2_0|PTQ1_0)(?=[-_.]|$)/i.exec(f.path.split('/').pop()!)?.[1]?.toUpperCase()
    const quant = ternary ?? f.quant
    if (!quant || /^(F16|BF16|F32)$/.test(quant)) continue
    const key = f.shard ? f.path.replace(/-\d{5}-of-\d{5}\.gguf$/i, '') : f.path
    const cur = out.get(key)
    if (cur) { cur.sizeBytes += f.sizeBytes; cur.shards = Math.max(cur.shards, f.shard?.count ?? 1); if (f.shard?.index === 1) cur.path = f.path; continue }
    out.set(key, { path: f.path, quant, sizeBytes: f.sizeBytes, shards: f.shard?.count ?? 1, prism: !!ternary })
  }
  return [...out.values()].sort((a, b) => quantRank(a.quant) - quantRank(b.quant))
}

export type Fit = 'gpu' | 'shared' | 'offload' | 'cpu'
export const FIT_RANK: Record<Fit, number> = { gpu: 0, shared: 0, offload: 1, cpu: 2 }

/** Where weights + KV land. Partial offload keeps the GPU share so the speed estimate can say how bad it is. */
export function place(weightsBytes: number, kvBytes: number, b: Budget): { fit: Fit; gpuShare: number } | null {
  const need = weightsBytes + kvBytes
  if (b.shared) return need <= b.gpu ? { fit: 'shared', gpuShare: 1 } : null
  if (b.gpu > 0 && need <= b.gpu) return { fit: 'gpu', gpuShare: 1 }
  if (need <= b.gpu + b.ram) return b.gpu > 0 ? { fit: 'offload', gpuShare: b.gpu / need } : { fit: 'cpu', gpuShare: 0 }
  return null
}

// ponytail: bandwidth guesses (GB/s) — mid-range discrete VRAM, dual-channel DDR5 for shared/CPU; decode reaches ≈ 70 %.
// Replace with this machine's measured decode rates (benchmark sessions) once any exist.
export const BANDWIDTH = { vram: 400 * GB, ram: 60 * GB, efficiency: 0.7 }
/** Decode tokens/s ≈ effective bandwidth / bytes read per token (active weights + KV). Offload time is the sum of
 *  the GPU part and the RAM part, which is why 50 % offload is nearly CPU speed. */
export function estimateTps(activeBytes: number, kvBytes: number, gpuShare: number, shared: boolean): number {
  const bytes = activeBytes + kvBytes
  const vram = shared ? BANDWIDTH.ram : BANDWIDTH.vram
  const t = (bytes * gpuShare) / vram + (bytes * (1 - gpuShare)) / BANDWIDTH.ram
  return (BANDWIDTH.efficiency / t)
}

export type Trust = 'official' | 'community' | 'risky'
const OFFICIAL_ORG = /^(qwen|google|meta-llama|microsoft|mistralai|deepseek-ai|liquidai|openai|ggml-org|unsloth|bartowski|lmstudio-community|prism-ml|nvidia|ibm-granite|allenai|tiiuae|internlm|zai-org|moonshotai|ornith-ai|cohereforai|stabilityai)$/i
const RISKY_NAME = /abliterat|uncensor|heretic|obliterat|nsfw|jailbreak|erotic/i
export function trustOf(repoId: string): Trust {
  const [org, name] = repoId.split('/')
  if (RISKY_NAME.test(name ?? '')) return 'risky'
  return OFFICIAL_ORG.test(org) ? 'official' : 'community'
}

/** Repos of the same model collapse to one row: "Qwen/Qwen3-8B-GGUF" and "unsloth/Qwen3-8B-GGUF" → "qwen3-8b". */
export function familyOf(repoId: string): string {
  return repoId.split('/').pop()!.toLowerCase()
    .replace(/[-_.]?(gguf|imatrix|i1)(?=[-_.]|$)/g, '')
    .replace(/[-_.](q\d_[a-z0-9_]+|iq\d_[a-z0-9_]+|f16|bf16|mxfp4)(?=[-_.]|$)/g, '')
    .replace(/^[-_.]+|[-_.]+$/g, '')
}

export interface FitModel extends HfModel {
  family: string
  paramsB: number
  activeB: number
  /** The file the size came from; null = estimated from the name (files not fetched yet). */
  file: FileChoice | null
  weightsBytes: number
  kvBytes: number
  fit: Fit
  gpuShare: number
  estTps: number
  /** estTps ≥ the workload's decode gate. */
  usable: boolean
  trust: Trust
  /** Other repos of the same family that were folded into this row. */
  alsoIn: string[]
}

const TRUST_RANK: Record<Trust, number> = { official: 0, community: 1, risky: 2 }

/** Rank: what fits on the GPU first, then usable speed, then trusted sources, then the largest model (quality proxy;
 *  Coder models get 1.5× for coding workloads), then downloads. Risky repos sink to the bottom. */
export function recommend(models: HfModel[], files: ReadonlyMap<string, HfGgufFile[]>, p: SystemProfile, workload: WorkloadProfile, ctx = workload.targetContext): { budget: Budget; models: FitModel[] } {
  const b = budget(p, ctx)
  const kv = (params: number) => kvBytesPerToken(params) * ctx
  const coding = /coding/.test(workload.id)
  const rows: FitModel[] = []
  for (const m of models) {
    const total = paramsB(m.id)
    if (total === null) continue
    const active = activeB(m.id, total)
    const kvBytes = kv(total)
    const opts = files.get(m.id)
    let file: FileChoice | null = null
    let placed: ReturnType<typeof place> = null
    if (opts) {
      // Best quant that fits; a ternary file only when nothing else does (it needs the PrismML build). A file under a
      // quarter of the name-based Q4 size cannot be the whole model (draft/speculative or partial uploads): ignored,
      // and a repo with no plausible whole-model file is not recommended at all.
      const plausible = fileOptions(opts).filter((f) => QUANT_PREF.includes(f.quant) && f.sizeBytes >= q4WeightBytes(total) * 0.25)
        .sort((x, y) => Number(x.prism) - Number(y.prism) || quantRank(x.quant) - quantRank(y.quant))
      const fitting = plausible.map((f) => ({ f, pl: place(f.sizeBytes, kvBytes, b) })).filter((x): x is { f: FileChoice; pl: NonNullable<ReturnType<typeof place>> } => x.pl !== null)
      if (!fitting.length) continue
      const bestTier = Math.min(...fitting.map((x) => FIT_RANK[x.pl.fit]))
      const atTier = fitting.filter((x) => FIT_RANK[x.pl.fit] === bestTier)
      const tps = (x: typeof atTier[number]) => estimateTps(x.f.sizeBytes * (active / total), kvBytes, x.pl.gpuShare, b.shared)
      // Highest quality that clears the workload's decode gate; on a bandwidth-bound machine none may, and then the
      // best quant that is not outright too slow (≥ half the gate), else the smallest (fastest) file.
      const gate = workload.minDecodeTps ?? 0
      const pick = atTier.find((x) => tps(x) >= gate) ?? atTier.find((x) => tps(x) >= gate / 2) ?? atTier.reduce((a, x) => (x.f.sizeBytes < a.f.sizeBytes ? x : a))
      file = pick.f
      placed = pick.pl
    } else {
      placed = place(q4WeightBytes(total), kvBytes, b)
      if (!placed) continue
    }
    const weightsBytes = file ? file.sizeBytes : q4WeightBytes(total)
    const estTps = estimateTps(weightsBytes * (active / total), kvBytes, placed!.gpuShare, b.shared)
    rows.push({ ...m, family: familyOf(m.id), paramsB: total, activeB: active, file, weightsBytes, kvBytes, fit: placed!.fit, gpuShare: placed!.gpuShare, estTps, usable: estTps >= (workload.minDecodeTps ?? 0), trust: trustOf(m.id), alsoIn: [] })
  }
  const quality = (r: FitModel) => r.paramsB * (coding && /coder|code/i.test(r.id) ? 1.5 : 1)
  // The speed number is a bandwidth guess, so only picks under HALF the workload's decode gate sink below smaller
  // models; the rest keep their size rank and show a "slow" mark.
  const tooSlow = (r: FitModel) => r.estTps < (workload.minDecodeTps ?? 0) / 2
  rows.sort((x, y) => FIT_RANK[x.fit] - FIT_RANK[y.fit] || Number(tooSlow(x)) - Number(tooSlow(y)) || TRUST_RANK[x.trust] - TRUST_RANK[y.trust] || quality(y) - quality(x) || y.downloads - x.downloads)
  // Fold duplicates of a family into the best-ranked row.
  const byFamily = new Map<string, FitModel>()
  for (const r of rows) {
    const head = byFamily.get(r.family)
    if (head) head.alsoIn.push(r.id)
    else byFamily.set(r.family, r)
  }
  const out = [...byFamily.values()]
  // Risky repos are shown, but last, whatever their size or downloads.
  out.sort((x, y) => Number(x.trust === 'risky') - Number(y.trust === 'risky'))
  return { budget: b, models: out }
}
