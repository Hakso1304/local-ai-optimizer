import { describe, expect, it } from 'vitest'
import { sanitizeRequest } from '../../src/main/validate'
import { estimateMemory, generateCandidates } from '../../src/core/benchmark/candidates'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { needle } from '../../src/core/quality/checkers'
import { classifyExit } from '../../src/core/runtimes/llamacpp/parse'
import { peaks } from '../../src/core/telemetry/sampler'
import { openDb } from '../../src/core/storage/db'
import { saveQualityResults, saveSession } from '../../src/core/storage/sessions'
import type { MachineLimits, ModelMeta } from '../../src/shared/bench-types'

const GiB = 1024 ** 3
const metric = (value: number | null) => value === null
  ? { value: null, kind: 'unavailable' as const, reason: 'probe failed' }
  : { value, kind: 'measured' as const, source: 'fixture' }

const model = (patch: Partial<ModelMeta> = {}): ModelMeta => ({
  id: 'model.gguf', name: 'model', arch: 'gemma4', fileBytes: 6 * GiB,
  paramCount: 10e9, quant: 'Q4_K_M', ctxTrain: 8192, layers: 4,
  nEmbd: 1024, heads: 8, headsKv: 8, keyLength: 128, valueLength: 128,
  nVocab: 32000, slidingWindow: null, headsKvPerLayer: [1, 1, 8, 8],
  fullAttentionInterval: null, slidingWindowPattern: null,
  keyLengthSwa: null, valueLengthSwa: null, supportsThinking: false,
  ...patch
})

const machine = (inUse: number | null): MachineLimits => ({
  vramBytes: metric(8 * GiB), vramInUseBytes: metric(inUse),
  ramTotalBytes: metric(32 * GiB), ramAvailableBytes: metric(20 * GiB),
  physicalCores: 8, gpuDevice: 'Vulkan0'
})

describe('W4 adversarial review', () => {
  it('rejects inherited names as workload IDs at the renderer boundary', () => {
    const got = sanitizeRequest({ workload: '__proto__', modelIds: ['C:\\models\\m.gguf'] }, ['C:\\models'])
    expect(got.ok).toBe(false)
  })

  it.fails('does not budget unobserved VRAM use as zero', () => {
    const got = generateCandidates(machine(null), model(), { backend: 'vulkan' }, WORKLOADS.general_chat)
    expect(got.candidates).toHaveLength(0)
  })

  it.fails('allocates partial GPU KV by the offloaded layers actual per-layer head counts', () => {
    const m = model()
    const withKv = estimateMemory(m, 2, 8192, 'f16', 512, true)
    const withoutKv = estimateMemory(m, 2, 8192, 'f16', 512, false)
    // llama.cpp offloads the final two transformer layers: 8+8 of 1+1+8+8 heads.
    expect(withKv.vramBytes - withoutKv.vramBytes).toBeCloseTo(withKv.kvBytes * 16 / 18, 0)
  })

  it('recognizes Vulkan VK_ERROR_DEVICE_LOST as device loss', () => {
    expect(classifyExit(['ggml_vulkan: VK_ERROR_DEVICE_LOST'])).toBe('device_lost')
  })

  it('rejects a negated needle instead of crediting retrieval', () => {
    expect(needle('The project name is not HELIOTROPE-5.', 'HELIOTROPE-5').pass).toBe(false)
  })

  it('stores a quality suite atomically if an insert fails midway', () => {
    const db = openDb(':memory:')
    try {
      const id = saveSession(db, { workload: 'general_chat', candidates: [] } as any, 'running')
      const good = { testId: 'IF-01', category: 'instruction' as const, weight: 1, pass: true, score: 1, detail: 'ok' }
      const bad = { ...good, testId: 'IF-02' }
      ;(bad as any).cycle = bad
      expect(() => saveQualityResults(db, id, 'model', [good, bad])).toThrow()
      const rows = db.prepare('SELECT count(*) AS n FROM quality_result WHERE session_id = ?').get(id) as { n: number }
      expect(rows.n).toBe(0)
    } finally {
      db.close()
    }
  })

  it('keeps every aggregate unavailable when the sampler has no rows', () => {
    const p = peaks([])
    expect(p.n).toBe(0)
    expect(p.max.procVramSharedBytes).toBeNull()
    expect(p.max.ramAvailBytes).toBeNull()
    expect(p.meanGpuUtilPct).toBeNull()
  })
})
