import { describe, expect, it } from 'vitest'
import { WORKLOADS } from '../src/core/scoring/workloads'
import { sloDefaults } from '../src/renderer/src/SloFilter'

describe('required context', () => {
  it("SLO min practical ctx defaults to the session's requiredContext, else half the workload target", () => {
    expect(sloDefaults(WORKLOADS.coding, 131072).minPracticalCtx).toBe(131072)
    expect(sloDefaults(WORKLOADS.coding, null).minPracticalCtx).toBe(WORKLOADS.coding.targetContext / 2)
  })
})

describe('workload presets (W4c D10)', () => {
  it('large_coding implies heavy mode + 64K required context; other workloads have no preset', async () => {
    const { presetFor } = await import('../src/renderer/src/LargeCodingCard')
    expect(presetFor('large_coding')).toEqual({ workload: 'large_coding', heavyMode: true, requiredContext: 65536 })
    expect(presetFor('coding')).toBeNull()
    expect(presetFor(null)).toBeNull()
  })
})
