// T06: the exported llama-server command launches exactly what the session runner measured.
// Runner path: loadConfigFor → LlamaCppBackend.loadModel (spawn captured); export path: exportConfigFrom → toLlamaServerArgs.
import { describe, expect, it } from 'vitest'
import { loadConfigFor } from '../src/core/benchmark/session'
import { exportConfigFrom, toJson, toLlamaServerArgs, toLlamaServerCommand } from '../src/core/export/config'
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
  ['integrated GPU: -ub 128, -lm none', { ...base, mmap: false, ubatch: 128 }],
  ['CPU only', { ...base, device: null, gpuLayersAll: false, gpuLayers: 0, kvType: 'q8_0' }],
  ['ROCm/HIP build: -dev ROCm0', { ...base, id: 'c|hip', backend: 'hip', device: 'ROCm0' }]
]

describe('T06 export equivalence', () => {
  it.each(variants)('%s', async (_n, cand) => {
    const ctx = 16384
    const rec = { workload: 'coding', best: { configId: cand.id, score: { recommendedCtx: ctx, referenceCtx: ctx } } } as unknown as Recommendation
    const exported = flags(toLlamaServerArgs(exportConfigFrom(rec, cand, model, '1')!))
    const ran = flags(await runnerArgs(cand, model, ctx))
    expect(exported).toEqual(ran)
    expect(ran['-ub']).toBe(String(cand.ubatch ?? 512))
    // the flags that matter are actually present (not both-missing)
    for (const f of ['-m', '-c', '-ngl', '-t', '-b', '-ub', '-fa', '-dev', '--parallel', '-fit', '--cache-ram']) expect(ran).toHaveProperty(f)
  })
})

it('export carries the measured backend (the app picks that build\'s llama-server) and HIP devices as -dev ROCm0', () => {
  const cand = variants.at(-1)![1]
  const rec = { workload: 'coding', best: { configId: cand.id, score: { recommendedCtx: 8192, referenceCtx: 8192 } } } as unknown as Recommendation
  const c = exportConfigFrom(rec, cand, model, '1')!
  expect(c.backend).toBe('hip')
  const exe = String.raw`C:\rt\llama.cpp-hip\llama-server.exe`
  const cmd = toLlamaServerCommand(c, exe)
  expect(cmd.startsWith(`${exe} -m `)).toBe(true)
  expect(cmd).toContain(' -dev ROCm0 ')
  expect(exportConfigFrom(rec, { ...cand, backend: undefined }, model, '1')!.backend).toBe('vulkan') // pre-HIP sessions
})

it.each([
  ['hip' as const, 'ROCm0', String.raw`C:\rt\llama.cpp-hip\llama-server.exe`],
  ['cuda' as const, 'CUDA0', String.raw`C:\rt\llama.cpp-cuda\llama-server.exe`]
])('O5: JSON for %s carries the actual executable, device and safe environment', (backend, device, exe) => {
  const cand = { ...base, id: `c-${backend}`, backend, device }
  const rec = { workload: 'coding', best: { configId: cand.id, score: { recommendedCtx: 8192, referenceCtx: 8192 } } } as unknown as Recommendation
  const c = exportConfigFrom(rec, cand, model, '1')!
  expect(() => toJson(c)).toThrow(/executable unavailable/)
  const json = JSON.parse(toJson(c, exe))
  expect(json.backend).toBe(backend)
  expect(json.llamaServer).toMatchObject({ executable: exe, command: toLlamaServerCommand(c, exe), environment: { GGML_CUDA_ENABLE_UNIFIED_MEMORY: null } })
  expect(json.llamaServer.args).toEqual(toLlamaServerArgs(c))
  expect(json.llamaServer.args).toContain(device)
  expect(json.llamaServer.environmentPolicy).toMatch(/case-insensitively/)
})
