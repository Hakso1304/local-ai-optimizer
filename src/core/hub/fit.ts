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
// Model vendors and the well-known quantizer accounts only; an org that cannot be vouched for stays 'community'.
const OFFICIAL_ORG = /^(qwen|google|meta-llama|microsoft|mistralai|deepseek-ai|liquidai|openai|ggml-org|unsloth|bartowski|lmstudio-community|prism-ml|nvidia|ibm-granite|allenai|tiiuae|internlm|zai-org|moonshotai|cohereforai|stabilityai)$/i
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

/** Model series (a vendor's line), for grouping the list and a hover profile. Profiles are static, deliberately
 *  general (vendor, positioning, distinguishing facts, license family) — no benchmark claims. Unknown → 'other'. */
export interface Series { key: string; label: string; vendor: string; profile: string }
const SERIES: (Series & { match: RegExp })[] = [
  { key: 'deepseek', label: 'DeepSeek', vendor: 'DeepSeek', match: /deepseek/i, profile: 'DeepSeek\'s open-weight line: V-series general MoE models and R-series reasoning models. Small "R1-Distill" variants are Qwen or Llama bases fine-tuned on R1 reasoning traces, so they inherit those bases\' behaviour. MIT license.' },
  { key: 'qwen', label: 'Qwen', vendor: 'Alibaba', match: /qwen|qwq/i, profile: 'Alibaba\'s open-weight family, dense (0.5B–32B) and MoE (e.g. 30B-A3B). Qwen3 onward has a switchable thinking mode; "Coder" variants are tuned for code and agentic tool use. Broad multilingual coverage. Apache-2.0 for most sizes.' },
  { key: 'llama', label: 'Llama', vendor: 'Meta', match: /llama(?!\.cpp)/i, profile: 'Meta\'s open-weight family (3.x: 1B–405B). Instruction-tuned chat models with a large ecosystem of fine-tunes and tooling. English-first with several supported languages. Llama community license (usage terms apply).' },
  { key: 'gemma', label: 'Gemma', vendor: 'Google', match: /gemma/i, profile: 'Google\'s open models derived from Gemini research (1B–27B). Larger Gemma 3 sizes are multimodal; long context. Gemma terms of use.' },
  { key: 'mistral', label: 'Mistral', vendor: 'Mistral AI', match: /mistral|mixtral|magistral|devstral|codestral|ministral/i, profile: 'Mistral AI\'s models: dense (7B, Ministral, Small 24B) and MoE (Mixtral 8x7B / 8x22B); Devstral and Codestral target code. Mostly Apache-2.0; some larger models under a research license.' },
  { key: 'phi', label: 'Phi', vendor: 'Microsoft', match: /(?:^|[-_/])phi[-_]?\d/i, profile: 'Microsoft\'s small models (3.8B–14B) trained on heavily filtered and synthetic "textbook-quality" data, aiming at reasoning ability above their size. MIT license.' },
  { key: 'lfm', label: 'LFM', vendor: 'Liquid AI', match: /lfm\d/i, profile: 'Liquid AI\'s LFM2 family for on-device use: a hybrid convolution + attention architecture built for fast, memory-light CPU and edge inference. Small sizes (hundreds of millions to a few billion parameters, plus a small MoE). LFM open license.' },
  { key: 'glm', label: 'GLM', vendor: 'Zhipu AI (Z.ai)', match: /glm-?\d|chatglm/i, profile: 'Zhipu AI\'s GLM line: general chat models plus coding/agent-oriented releases; bilingual Chinese/English focus. MIT license for recent releases.' },
  { key: 'gpt-oss', label: 'gpt-oss', vendor: 'OpenAI', match: /gpt-oss/i, profile: 'OpenAI\'s open-weight MoE models (20B, 120B) released in MXFP4 with adjustable reasoning effort and tool use. Apache-2.0.' },
  { key: 'bonsai', label: 'Bonsai', vendor: 'PrismML', match: /bonsai/i, profile: 'PrismML\'s ternary models: weights stored at about 2 bits (PQ2_0 / PTQ1_0), so a 27B fits in ~6–7 GB. They load only with the PrismML llama.cpp fork (install it on the System page); Vulkan and CPU backends work.' },
  { key: 'nemotron', label: 'Nemotron', vendor: 'NVIDIA', match: /nemotron/i, profile: 'NVIDIA\'s models, often built on Llama or Mistral bases and post-trained for reasoning and agentic use. NVIDIA open model license.' },
  { key: 'granite', label: 'Granite', vendor: 'IBM', match: /granite/i, profile: 'IBM\'s Granite family, positioned for enterprise use with documented training data; small dense and MoE sizes. Apache-2.0.' },
  { key: 'kimi', label: 'Kimi', vendor: 'Moonshot AI', match: /kimi/i, profile: 'Moonshot AI\'s Kimi line: very large MoE models oriented to agentic and coding tasks. Modified MIT license.' },
  { key: 'smollm', label: 'SmolLM', vendor: 'Hugging Face', match: /smollm/i, profile: 'Hugging Face\'s small open models (hundreds of millions to a few billion parameters) with fully open training data and recipes. Apache-2.0.' }
]
export const OTHER_SERIES: Series = { key: 'other', label: 'Other', vendor: '', profile: 'No series profile yet: check the repository card for the base model, license and intended use.' }
export function seriesOf(repoId: string): Series {
  const s = SERIES.find((x) => x.match.test(repoId.split('/').pop()!))
  return s ? { key: s.key, label: s.label, vendor: s.vendor, profile: s.profile } : OTHER_SERIES
}

/** Newer models do more per parameter. 1.0 up to 6 months after the repo was created, then linear to 0.5 at 30 months
 *  (a 2-year-old 14B ranks with a fresh 7B). Unknown date = 0.75 (neither rewarded nor buried). */
export const RECENCY = { fullMonths: 6, floorMonths: 30, floor: 0.5, unknown: 0.75 }
export function recencyFactor(createdAt: string | null | undefined, now: number): number {
  const t = createdAt ? Date.parse(createdAt) : NaN
  if (!Number.isFinite(t)) return RECENCY.unknown
  const months = Math.max(0, (now - t) / (30.44 * 86_400_000))
  if (months <= RECENCY.fullMonths) return 1
  if (months >= RECENCY.floorMonths) return RECENCY.floor
  return 1 - (1 - RECENCY.floor) * (months - RECENCY.fullMonths) / (RECENCY.floorMonths - RECENCY.fullMonths)
}

export interface FitModel extends HfModel {
  family: string
  series: Series
  /** Repo creation month "YYYY-MM" (release proxy); null when unknown. */
  releasedAt: string | null
  /** Size × recency: the quality proxy used for ranking (shown on hover). */
  quality: number
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

/** Rank: what fits on the GPU first, then not-too-slow, then trusted sources, then the quality proxy (size × recency;
 *  Coder models ×1.5 for coding workloads), then downloads. Risky repos sink to the bottom. */
export function recommend(models: HfModel[], files: ReadonlyMap<string, HfGgufFile[]>, p: SystemProfile, workload: WorkloadProfile, ctx = workload.targetContext, now = Date.now()): { budget: Budget; models: FitModel[] } {
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
    const quality = total * recencyFactor(m.createdAt, now) * (coding && /coder|code/i.test(m.id) ? 1.5 : 1)
    rows.push({ ...m, family: familyOf(m.id), series: seriesOf(m.id), releasedAt: m.createdAt ? m.createdAt.slice(0, 7) : null, quality, paramsB: total, activeB: active, file, weightsBytes, kvBytes, fit: placed!.fit, gpuShare: placed!.gpuShare, estTps, usable: estTps >= (workload.minDecodeTps ?? 0), trust: trustOf(m.id), alsoIn: [] })
  }
  const quality = (r: FitModel) => r.quality
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
