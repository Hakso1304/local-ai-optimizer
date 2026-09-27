import { describe, expect, it } from 'vitest'
import { DEFAULT_CANDIDATE_RULES as D } from '../src/core/benchmark/candidates'
import { isInside, sanitizeRequest } from '../src/main/validate'

const roots = ['D:\\llm-models', 'E:\\Product_1\\models']
const ok = (raw: unknown) => {
  const r = sanitizeRequest(raw, roots)
  if (!r.ok) throw new Error(r.error)
  return r.req
}

describe('request validation (renderer → main trust boundary)', () => {
  it('path containment uses path.relative, not prefix matching', () => {
    expect(isInside('D:\\llm-models', 'D:\\llm-models\\sub\\m.gguf')).toBe(true)
    expect(isInside('D:\\llm-models', 'D:\\llm-models-evil\\m.gguf')).toBe(false) // startsWith would accept this
    expect(isInside('D:\\llm-models', 'D:\\llm-models\\..\\secret.gguf')).toBe(false)
    expect(isInside('D:\\llm-models', 'D:\\llm-models')).toBe(false)
    expect(isInside('D:\\llm-models', 'C:\\llm-models\\m.gguf')).toBe(false)
  })

  it('rejects unknown workloads and models outside the configured dirs', () => {
    expect(sanitizeRequest({ workload: 'nope', modelIds: ['D:\\llm-models\\a.gguf'] }, roots)).toMatchObject({ ok: false })
    expect(sanitizeRequest({ workload: 'coding', modelIds: ['D:\\llm-models-evil\\a.gguf'] }, roots)).toMatchObject({ ok: false, error: expect.stringMatching(/not in a configured/) })
    expect(sanitizeRequest({ workload: 'coding', modelIds: [] }, roots)).toMatchObject({ ok: false })
  })

  it('clamps reps, keeps only known ladder rungs, drops resumeSessionId and unknown fields', () => {
    const req = ok({ workload: 'coding', modelIds: ['D:\\llm-models\\a.gguf'], reps: 99, ladder: [4096, 1234, 2048, 4096], resumeSessionId: '7', evil: 1 })
    expect(req).toEqual({ workload: 'coding', modelIds: ['D:\\llm-models\\a.gguf'], reps: 5, ladder: [2048, 4096] })
    expect(ok({ workload: 'coding', modelIds: ['D:\\llm-models\\a.gguf'], reps: 0 }).reps).toBe(1)
  })

  it('candidateRules: whitelist only, safety margins can only get stricter', () => {
    const req = ok({
      workload: 'coding', modelIds: ['D:\\llm-models\\a.gguf'],
      candidateRules: { ramReserveBytes: 0, keepOverVramMaxRatio: 10, vramMarginBytes: 8 * D.vramMarginBytes, maxPerModel: 50, ctxLadder: [999999], cpuOnlyMaxParams: 1e12 }
    })
    expect(req.candidateRules).toEqual({ ramReserveBytes: D.ramReserveBytes, keepOverVramMaxRatio: D.keepOverVramMaxRatio, vramMarginBytes: 8 * D.vramMarginBytes, maxPerModel: D.maxPerModel })
  })

  it('passes heavyMode / retryFailed / rerunConfigIds / runQuality through when well-typed', () => {
    const req = ok({ workload: 'coding', modelIds: ['E:\\Product_1\\models\\m.gguf'], heavyMode: true, retryFailed: true, rerunConfigIds: ['c1'], runQuality: false })
    expect(req).toMatchObject({ heavyMode: true, retryFailed: true, rerunConfigIds: ['c1'], runQuality: false })
    expect(ok({ workload: 'coding', modelIds: ['E:\\Product_1\\models\\m.gguf'], heavyMode: 'yes', rerunConfigIds: [1] })).not.toHaveProperty('heavyMode')
  })
})
