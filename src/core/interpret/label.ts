import type { CandidateInput } from '../../shared/bench-types'

/** "Qwen3.8-27B Q4_K_M (54/65 layers)": model, quant (unless the name has it), and what differs from a full offload. */
export function label(i: CandidateInput): string {
  const { model: m, config: c } = i
  const name = m.quant && !m.name.includes(m.quant) ? `${m.name} ${m.quant}` : m.name
  const extra = [
    c.gpuLayersAll ? null : c.gpuLayers === 0 ? 'CPU only' : `${c.gpuLayers}/${m.layers} layers`,
    c.kvType === 'q8_0' ? 'KV q8_0' : null,
    c.kvOffload === false ? 'KV in RAM' : null
  ].filter((x): x is string => x !== null)
  return extra.length ? `${name} (${extra.join(', ')})` : name
}
