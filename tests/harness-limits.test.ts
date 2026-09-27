import { describe, expect, it } from 'vitest'
import { validateHarnessLimits } from '../scripts/harness-limits'

describe('hardware harness CLI limits', () => {
  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY, 300_001, null, undefined])('rejects request cap %s before launch', (requestCapMs) => {
    expect(() => validateHarnessLimits({ requestCapMs, ramAbortGib: 4 })).toThrow(/request-cap-ms/)
  })

  it.each([0, 3.99, -1, Number.NaN, Number.POSITIVE_INFINITY, null, undefined])('rejects RAM floor %s before launch', (ramAbortGib) => {
    expect(() => validateHarnessLimits({ requestCapMs: 300_000, ramAbortGib })).toThrow(/ram-abort-gib/)
  })

  it('accepts the production edge bounds and preserves the validated values', () => {
    expect(validateHarnessLimits({ requestCapMs: 300_000, ramAbortGib: 4 })).toEqual({ requestCapMs: 300_000, ramAbortGib: 4 })
    expect(validateHarnessLimits({ requestCapMs: 1, ramAbortGib: 8 })).toEqual({ requestCapMs: 1, ramAbortGib: 8 })
  })
})
