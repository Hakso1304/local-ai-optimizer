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

  it('requiredContext: only 32K/64K/128K; absent/null = Auto', () => {
    const base = { workload: 'coding', modelIds: ['D:\\llm-models\\a.gguf'] }
    expect(ok({ ...base, requiredContext: 131072 })).toMatchObject({ requiredContext: 131072 })
    expect(ok({ ...base, requiredContext: null })).not.toHaveProperty('requiredContext')
    expect(sanitizeRequest({ ...base, requiredContext: 50000 }, roots)).toMatchObject({ ok: false, error: expect.stringMatching(/32K, 64K or 128K/) })
  })

  it('minDecodeTps: 0–1000 t/s, blank/null = workload default', () => {
    const base = { workload: 'coding', modelIds: ['E:/Product_1/models/m.gguf'] }
    expect(ok({ ...base, minDecodeTps: 25 })).toMatchObject({ minDecodeTps: 25 })
    expect(ok({ ...base, minDecodeTps: null })).not.toHaveProperty('minDecodeTps')
    expect(sanitizeRequest({ ...base, minDecodeTps: 5000 }, roots)).toMatchObject({ ok: false })
    expect(sanitizeRequest({ ...base, minDecodeTps: -1 }, roots)).toMatchObject({ ok: false })
  })

  // Windows integration (T13): NTFS junctions; the lexical checks above run everywhere.
  it.skipIf(process.platform !== 'win32')('an existing model reached through a junction that points outside the root is rejected (W4 F11) [Windows: junctions]', async () => {
    const { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, rmSync } = await import('node:fs')
    const { tmpdir } = await import('node:os')
    const { join } = await import('node:path')
    const base = mkdtempSync(join(tmpdir(), 'lao-root-'))
    try {
      const root = join(base, 'models'), outside = join(base, 'secret')
      mkdirSync(root); mkdirSync(outside)
      writeFileSync(join(outside, 'x.gguf'), 'x')
      writeFileSync(join(root, 'ok.gguf'), 'x')
      symlinkSync(outside, join(root, 'link'), 'junction') // no admin needed on Windows
      expect(sanitizeRequest({ workload: 'coding', modelIds: [join(root, 'link', 'x.gguf')] }, [root])).toMatchObject({ ok: false })
      expect(sanitizeRequest({ workload: 'coding', modelIds: [join(root, 'ok.gguf')] }, [root])).toMatchObject({ ok: true })
      expect(sanitizeRequest({ workload: 'toString', modelIds: [join(root, 'ok.gguf')] }, [root])).toMatchObject({ ok: false }) // F8
    } finally {
      rmSync(base, { recursive: true, force: true })
    }
  })
})
