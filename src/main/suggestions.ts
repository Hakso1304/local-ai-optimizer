// Sibling-quantization suggestions for the Benchmark page (I-9.2): for each listed model whose repo offers a smaller
// quantization that would put more layers on the GPU. ESTIMATED (DESIGN §2.7) unless a measured per-process budget
// applies; quantSuggestions() says which in each text. Pure given its inputs (main passes the same machine as models:fit).
import { basename } from 'node:path'
import { quantSuggestions, rulesForRequest } from '../core/benchmark/candidates'
import { toModelMeta } from '../core/models/gguf'
import type { MachineLimits, QuantSuggestion, WorkloadProfile } from '../shared/bench-types'
import type { ModelInfo } from '../shared/types'

export function modelSuggestions(machine: MachineLimits, infos: ModelInfo[], workload: WorkloadProfile): QuantSuggestion[] {
  const local = infos.map((i) => basename(i.path)) // a sibling already on disk is listed as a model, not suggested
  return infos.flatMap((info) => {
    const mm = toModelMeta(info)
    return mm.meta ? quantSuggestions(machine, mm.meta, workload, rulesForRequest({ heavyMode: false }), local) : []
  })
}
