// L1 (cal-longctx-2026-09-27): with ≈ 2.5 GiB of VRAM held by other processes, two collapsing rungs sat at 67–73 %
// dedicated with 2.0–2.1 GiB raw shared and an adjusted spill of 0.00 — the raw-growth rule must catch them.
import { describe, expect, it } from 'vitest'
import { detectCliffs } from '../../src/core/scoring/cliff'
import { recommendForWorkload } from '../../src/core/scoring/recommend'
import { inputs, load, machine, toRun, withQuality } from './helpers'

const f = load('calib-longctx-contention-rx9070.json') as ReturnType<typeof load> & { vramInUseBytes: number }
const runsOf = (id: string) => f.runs.filter((r) => r.configId === id).map(toRun)

describe('spill under other-process VRAM use', () => {
  it('f16 64K (25.0 t/s, 2.08 GiB raw) and q8_0 128K (24.2 t/s, 11.59 GiB ded + 2.00 GiB raw) are spills, not only decode drops', () => {
    for (const [id, ctx] of [['llama8b|f16', 65536], ['llama8b|q8', 131072]] as const) {
      const step = detectCliffs(runsOf(id), f.vramBytes).steps.find((s) => s.ctx === ctx)!
      expect(step.verdict, id).toBe('degraded')
      expect(step.reasons.map((r) => r.code), id).toEqual(expect.arrayContaining(['decode_drop', 'shared_spill']))
      expect(step.reasons.find((r) => r.code === 'shared_spill')!.message).toMatch(/shared GPU memory grew \+[12]\.\d GiB between \d+K and \d+K \(raw, host-pinned excluded\)/)
    }
  })
  it('clean rungs (raw growth ≤ 0.05 GiB) stay clean; q8_0 KV is the 64K pick with its clean ceiling', () => {
    const q8 = detectCliffs(runsOf('llama8b|q8'), f.vramBytes)
    expect(q8.practicalContextCeiling.value).toBe(65536)
    expect(detectCliffs(runsOf('llama8b|f16'), f.vramBytes).practicalContextCeiling.value).toBe(32768)
    const M = { ...machine(f.vramBytes), vramInUseBytes: { value: f.vramInUseBytes, kind: 'measured' as const } }
    const rec = recommendForWorkload({ candidates: withQuality(inputs(f)), machine: M }, 'large_coding', { requiredContext: 65536 })
    expect(rec.best?.configId).toBe('llama8b|q8')
  })
})
