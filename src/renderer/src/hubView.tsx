// View helpers for the Download page's recommendation table (pure; testable without React).
import type { FitModel, Series } from '../../core/hub/fit'

export const PRISM_NOTE = 'PQ2_0 / PTQ1_0 ternary: loads only with the PrismML build (install it on the System page)'

const gib = (b: number) => (b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} GiB` : `${(b / 1024 ** 2).toFixed(0)} MiB`)

/** Rows grouped by series, groups in order of their best-ranked row; rank order kept inside a group. */
export function seriesGroups(models: FitModel[]): { series: Series; models: FitModel[] }[] {
  const out: { series: Series; models: FitModel[] }[] = []
  for (const m of models) {
    const g = out.find((x) => x.series.key === m.series.key)
    if (g) g.models.push(m)
    else out.push({ series: m.series, models: [m] })
  }
  return out
}

/** Hover text for one row: what the numbers rest on, so the rank is explainable. */
export function rowSummary(m: FitModel): string {
  const fit = m.fit === 'gpu' ? 'fits the GPU' : m.fit === 'shared' ? 'fits the shared GPU/RAM pool' : m.fit === 'offload' ? `${Math.round(m.gpuShare * 100)} % of the weights on the GPU, the rest in RAM` : 'CPU only'
  return [
    `${m.paramsB}B parameters${m.activeB !== m.paramsB ? ` (${m.activeB}B active per token, MoE)` : ''} · released ${m.releasedAt ?? 'unknown'}`,
    m.file ? `${m.file.quant} ${gib(m.weightsBytes)}${m.file.shards > 1 ? ` in ${m.file.shards} shards` : ''}: ${m.file.path}` : `size estimated from the name as Q4_K_M (${gib(m.weightsBytes)}); the repo's files were not read yet`,
    `+ ${gib(m.kvBytes)} KV cache at the workload's context → ${fit}`,
    `≈ ${m.estTps.toFixed(1)} tokens/s (bandwidth estimate)${m.usable ? '' : ' — below the workload\'s decode gate'}`,
    `rank: size × recency = ${m.quality.toFixed(1)} · source ${m.trust} · ${m.downloads.toLocaleString()} downloads${m.alsoIn.length ? ` · also in ${m.alsoIn.join(', ')}` : ''}`,
    ...(m.file?.prism ? [PRISM_NOTE] : [])
  ].join('\n')
}
