// Rank candidates for one workload and explain the pick in plain English. Pure and deterministic (A19).
import type {
  BreakdownRow, CandidateInput, ComponentId, MachineLimits, Metric, Recommendation, WorkloadId, WorkloadScore
} from '../../shared/bench-types'
import { componentScores } from './components'
import { fmtCtx, val } from './cliff'
import { DEFAULT_SCORING_CONFIG, type ScoringConfig } from './workloads'

const LABEL: Record<ComponentId, string> = {
  quality: 'quality', genSpeed: 'generation speed', prefillSpeed: 'prefill speed', latency: 'latency',
  memory: 'memory headroom', stability: 'stability', context: 'context capability'
}

interface Scored {
  input: CandidateInput
  score: WorkloadScore
  cs: ReturnType<typeof componentScores>
  decode: number | null
  vram: number | null
  ram: number | null
}

const byId = (a: Scored, b: Scored) => (a.input.config.id < b.input.config.id ? -1 : a.input.config.id > b.input.config.id ? 1 : 0)
/** Missing values sort last in both directions. */
const cmp = (a: number | null, b: number | null, desc: boolean) =>
  a === b ? 0 : a === null ? 1 : b === null ? -1 : desc ? b - a : a - b

export function recommend(
  inputs: CandidateInput[],
  machine: MachineLimits,
  workload: WorkloadId,
  cfg: ScoringConfig = DEFAULT_SCORING_CONFIG
): Recommendation {
  const profile = cfg.profiles[workload]
  const round = (x: number) => Number(x.toFixed(cfg.tieDecimals))
  const excluded: Recommendation['excluded'] = []
  const scored: Scored[] = []

  for (const input of inputs) {
    const cs = componentScores(input, machine, profile, cfg)
    if (!cs.usable) {
      const why = cs.cliff.steps.flatMap((s) => s.reasons.map((r) => r.message))
      excluded.push({ configId: input.config.id, reasons: why.length ? why : ['no runs recorded'] })
      continue
    }
    const breakdown: BreakdownRow[] = (Object.keys(profile.weights) as ComponentId[]).map((k) => {
      const c = cs.components[k]
      const weight = profile.weights[k]
      return { component: k, input: c.input, score: c.score, weight, contribution: weight * c.score }
    })
    const total = breakdown.reduce((s, r) => s + r.contribution, 0)
    const gateFailures: string[] = []
    const ceil = val(cs.cliff.practicalContextCeiling)
    if (ceil === null || ceil < profile.targetContext * cfg.gates.minCtxFraction) {
      gateFailures.push(`practical context ${ceil === null ? 'none' : fmtCtx(ceil)} is below ${fmtCtx(profile.targetContext * cfg.gates.minCtxFraction)}`)
    }
    if (cs.components.stability.score < cfg.gates.minStability) gateFailures.push(`stability ${cs.components.stability.score.toFixed(0)} < ${cfg.gates.minStability}`)
    const q = cs.components.quality
    if (q.score < profile.minQuality) gateFailures.push(`${q.input.kind === 'estimated' ? 'estimated ' : ''}quality ${q.score.toFixed(0)} < ${profile.minQuality}`)
    const ref = input.runs.find((r) => r.ctx === cs.referenceCtx)
    scored.push({
      input, cs,
      score: { configId: input.config.id, workload, total, eligible: gateFailures.length === 0, gateFailures, breakdown, referenceCtx: cs.referenceCtx },
      decode: val(ref?.decodeTps, true),
      vram: input.config.gpuLayers === 0 ? 0 : val(ref?.peakVramBytes),
      ram: val(ref?.peakRamBytes)
    })
  }

  // Tie-break chain (X3): eligible, total (rounded) desc, lower peak VRAM, lower peak RAM, configId asc.
  scored.sort((a, b) =>
    Number(b.score.eligible) - Number(a.score.eligible) ||
    round(b.score.total) - round(a.score.total) ||
    cmp(a.vram, b.vram, false) || cmp(a.ram, b.ram, false) || byId(a, b))
  excluded.sort((a, b) => (a.configId < b.configId ? -1 : a.configId > b.configId ? 1 : 0))

  const eligible = scored.filter((s) => s.score.eligible)
  const pick = (f: (a: Scored, b: Scored) => number) => [...eligible].sort((a, b) => f(a, b) || byId(a, b))[0]?.input.config.id ?? null
  const alternatives = {
    fastest: pick((a, b) => cmp(a.decode, b.decode, true)),
    bestQuality: pick((a, b) => b.cs.components.quality.score - a.cs.components.quality.score),
    bestLongContext: pick((a, b) => cmp(val(a.cs.cliff.practicalContextCeiling), val(b.cs.cliff.practicalContextCeiling), true) || cmp(a.decode, b.decode, true)),
    lowestMemory: pick((a, b) => cmp(a.vram, b.vram, false) || cmp(a.ram, b.ram, false))
  }

  const top = eligible[0]
  const reasons: string[] = []
  if (!inputs.length) reasons.push('No recommendation: no candidates were benchmarked')
  else if (!scored.length) reasons.push('No recommendation: no successful runs')
  else if (!top) {
    reasons.push(`No recommendation: no candidate meets the ${profile.label} requirements`)
    for (const s of scored) reasons.push(`${s.score.configId}: ${s.score.gateFailures.join('; ')}`)
  }

  let best: Recommendation['best'] = null
  if (top) {
    const { cliff, components } = top.cs
    const declared = top.input.model.ctxTrain
    const declaredContext: Metric = declared ? { value: declared, kind: 'declared', source: 'GGUF context_length' } : { value: null, kind: 'unavailable', reason: 'GGUF has no context_length' }
    best = { configId: top.score.configId, score: top.score, practicalContext: cliff.practicalContextCeiling, declaredContext, cliff }
    reasons.push(`Best for ${profile.label}: ${top.score.configId} scores ${top.score.total.toFixed(1)}/100`)
    if (scored.length === 1) reasons.push('Only one candidate; not compared')
    const leaders = [...top.score.breakdown].sort((a, b) => b.contribution - a.contribution || (a.component < b.component ? -1 : 1)).slice(0, 2)
    reasons.push(`Largest contributions: ${leaders.map((r) => `${LABEL[r.component]} ${r.contribution.toFixed(1)}`).join(', ')}`)
    const pc = val(cliff.practicalContextCeiling)!
    reasons.push(`Practical context ${fmtCtx(pc)} (measured)${declared ? `; model declares ${fmtCtx(declared)}` : ''}`)
    if (cliff.spillFreeUpTo !== null) reasons.push(`No VRAM spill up to ${fmtCtx(cliff.spillFreeUpTo)}`)
    for (const s of cliff.steps) for (const r of s.reasons) if (r.code !== 'beyond_limit') reasons.push(r.message)
    if (components.quality.note) reasons.push(components.quality.note)
  }

  return { workload, scoringVersion: cfg.version, best, alternatives, ranked: scored.map((s) => s.score), excluded, reasons }
}
