import { describe, expect, it } from 'vitest'
import { WORKLOADS } from '../src/core/scoring/workloads'
import { sloDefaults } from '../src/renderer/src/SloFilter'

describe('required context', () => {
  it("SLO min practical ctx defaults to the session's requiredContext, else half the workload target", () => {
    expect(sloDefaults(WORKLOADS.coding, 131072).minPracticalCtx).toBe(131072)
    expect(sloDefaults(WORKLOADS.coding, null).minPracticalCtx).toBe(WORKLOADS.coding.targetContext / 2)
  })
})
