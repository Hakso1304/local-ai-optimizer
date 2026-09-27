import { describe, expect, it } from 'vitest'
import { HubError } from '../src/core/hub/hf'
import { diskCheck, isAllowedDest, nearestExisting, progressInfo, userMessage } from '../src/main/hub-logic'

const GiB = 1024 ** 3

describe('hub IPC pure logic', () => {
  it('progress: pct and ETA from bytes/total/speed; unknowns stay null', () => {
    expect(progressInfo('r', 'f', 2 * GiB, 4 * GiB, 100 * 1024 ** 2)).toMatchObject({ pct: 50, etaSec: 20.48 })
    expect(progressInfo('r', 'f', 5, null, 10)).toMatchObject({ pct: null, etaSec: null })
    expect(progressInfo('r', 'f', 5, 10, 0)).toMatchObject({ pct: 50, etaSec: null })
  })

  it('error mapping: gated keeps the model-page URL; auth/404/cancel are actionable', () => {
    const gated = new HubError('gated_accept_license', 'Access denied: this model is gated — open https://huggingface.co/meta-llama/X, accept its license', 403)
    expect(userMessage(gated)).toEqual({ kind: 'gated_accept_license', error: gated.message })
    expect(userMessage(new HubError('auth_required', 'x', 401)).error).toMatch(/access token/)
    expect(userMessage(new HubError('not_found', 'x', 404)).error).toMatch(/does not exist/)
    expect(userMessage(new HubError('cancelled', 'x')).error).toMatch(/^Paused/)
    expect(userMessage(new Error('boom'))).toEqual({ kind: 'network', error: 'boom' })
  })

  it('destination must be a configured model dir or inside one', () => {
    const dirs = ['D:\\llm-models', 'E:\\Product_1\\models']
    expect(isAllowedDest('D:\\llm-models', dirs)).toBe(true)
    expect(isAllowedDest('D:\\llm-models\\sub', dirs)).toBe(true)
    expect(isAllowedDest('D:\\llm-models\\..\\Windows', dirs)).toBe(false)
    expect(isAllowedDest('C:\\Windows', dirs)).toBe(false)
  })

  it('free-space probe walks up to the nearest existing dir; a missing drive gives null', () => {
    const have = new Set(['D:\\', 'D:\\llm-models'])
    expect(nearestExisting('D:\\llm-models\\new\\sub', (p) => have.has(p))).toBe('D:\\llm-models')
    expect(nearestExisting('Q:\\nope\\x', (p) => have.has(p))).toBeNull()
  })

  it('disk check: remaining bytes (resume-aware) + 1 GiB margin', () => {
    expect(diskCheck(20 * GiB, 16 * GiB, 0)).toEqual({ ok: true })
    expect(diskCheck(10 * GiB, 16 * GiB, 8 * GiB)).toEqual({ ok: true }) // 8 GiB left + 1 GiB ≤ 10 GiB
    const r = diskCheck(5 * GiB, 16 * GiB, 0)
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.error).toBe('Not enough disk space: need 17.0 GiB (file 16.0 GiB + 1 GiB margin), 5.0 GiB free')
  })
})
