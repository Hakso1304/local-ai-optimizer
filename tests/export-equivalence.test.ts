// T06: the exported llama-server command launches exactly what the session runner measured.
// Runner path: loadConfigFor → LlamaCppBackend.loadModel (spawn captured); export path: exportConfigFrom → toLlamaServerArgs.
import { describe, expect, it } from 'vitest'
import { loadConfigFor } from '../src/core/benchmark/session'
import { exportConfigFrom, toLlamaServerArgs } from '../src/core/export/config'
import { LlamaCppBackend } from '../src/core/runtimes/llamacpp'
import type { CandidateConfig, ModelMeta, Recommendation } from '../src/shared/bench-types'

const NOT_INFERENCE = new Set(['--host', '--port', '-lv', '--metrics'])
const ALIAS: Record<string, string> = { '--device': '-dev' }
/** flag → value ('' for bare flags), host/port/logging dropped. */
function flags(args: string[]): Record<string, string> {
  const out: Record<string, string> = {}
  for (let i = 0; i < args.length; i++) {
    const f = ALIAS[args[i]] ?? args[i]
    const val = args[i + 1] !== undefined && !args[i + 1].startsWith('-') ? args[++i] : ''
    if (!NOT_INFERENCE.has(f)) out[f] = val
  }
  return out
}

async function runnerArgs(cand: CandidateConfig, model: ModelMeta, ctx: number): Promise<string[]> {
  let captured: string[] = []
  const b = new LlamaCppBackend('unused', { spawnFn: (_c, args) => { captured = args; throw new Error('captured') } })
  await expect(b.loadModel({ ...loadConfigFor(cand, model, ctx), port: 1 })).rejects.toThrow('captured')
  return captured
}

const model = { id: 'D:\\llm-models\\m.gguf', name: 'm', layers: 36 } as ModelMeta
const base = { id: 'c', gpuLayers: 36, gpuLayersAll: true, threads: 8, flashAttn: true, kvType: 'f16', device: 'Vulkan0' } as CandidateConfig
const variants: [string, CandidateConfig][] = [
  ['full offload f16', base],
  ['partial q8_0 KV, fa off', { ...base, gpuLayersAll: false, gpuLayers: 20, kvType: 'q8_0', flashAttn: false }],
  ['heavy: -nkvo, -lm none', { ...base, gpuLayersAll: false, gpuLayers: 55, kvOffload: false, mmap: false }],
  ['CPU only', { ...base, device: null, gpuLayersAll: false, gpuLayers: 0, kvType: 'q8_0' }]
]

describe('T06 export equivalence', () => {
  it.each(variants)('%s', async (_n, cand) => {
    const ctx = 16384
    const rec = { workload: 'coding', best: { configId: cand.id, score: { recommendedCtx: ctx, referenceCtx: ctx } } } as unknown as Recommendation
    const exported = flags(toLlamaServerArgs(exportConfigFrom(rec, cand, model, '1')!))
    const ran = flags(await runnerArgs(cand, model, ctx))
    expect(exported).toEqual(ran)
    // the flags that matter are actually present (not both-missing)
    for (const f of ['-m', '-c', '-ngl', '-t', '-b', '-ub', '-fa', '-dev', '--parallel', '-fit', '--cache-ram']) expect(ran).toHaveProperty(f)
  })
})
