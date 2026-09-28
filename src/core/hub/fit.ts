// "Fits this PC" filter for Hub repos: parameter count from the repo name, rough Q4_K_M footprint vs the scanned
// VRAM/RAM. Pure (no Node APIs) so the renderer can import it. Estimates only — the benchmark is the real answer.
import type { SystemProfile } from '../../shared/types'
import { pickGpu } from '../benchmark/candidates'
import type { HfModel } from './hf'

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

// ponytail: fixed Q4_K_M ≈ 4.85 bits/weight + 1.5 GB for KV cache/buffers at a few K context; ignores arch and ctx.
export const q4Bytes = (b: number): number => b * GB * 0.61 + 1.5 * GB

export type Fit = 'gpu' | 'offload' | 'cpu'
export interface FitModel extends HfModel { paramsB: number; estBytes: number; fit: Fit }

/** Machine budget: dedicated VRAM of the discrete GPU (90 %) and half the system RAM (the OS and apps keep the rest). */
export function budget(p: SystemProfile): { vram: number; ram: number } {
  const g = pickGpu(p)
  const v = g && !g.isIntegrated && g.dedicatedVramBytes.status === 'available' ? g.dedicatedVramBytes.value ?? 0 : 0
  return { vram: v * 0.9, ram: (p.ram.value?.totalBytes ?? 0) * 0.5 }
}

/** Repos that fit, full-GPU first, then partial offload, then CPU/shared RAM; downloads order within a tier. */
export function fitModels(models: HfModel[], p: SystemProfile): FitModel[] {
  const { vram, ram } = budget(p)
  const rank: Record<Fit, number> = { gpu: 0, offload: 1, cpu: 2 }
  const out: FitModel[] = []
  for (const m of models) {
    const b = paramsB(m.id)
    if (b === null) continue
    const est = q4Bytes(b)
    const fit: Fit | null = vram && est <= vram ? 'gpu' : est <= vram + ram ? (vram ? 'offload' : 'cpu') : null
    if (fit) out.push({ ...m, paramsB: b, estBytes: est, fit })
  }
  return out.sort((a, b) => rank[a.fit] - rank[b.fit] || b.downloads - a.downloads)
}
