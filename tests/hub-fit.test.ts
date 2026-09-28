import { describe, expect, it } from 'vitest'
import { fitModels, paramsB } from '../src/core/hub/fit'
import type { SystemProfile } from '../src/shared/types'

const GiB = 1024 ** 3
const prof = (vram: number | null, ramGiB: number): SystemProfile => ({
  ram: { value: { totalBytes: ramGiB * GiB, availableBytes: 0 }, status: 'available', source: 't' },
  gpus: { value: vram === null ? [] : [{ name: 'g', vendor: 'nvidia', pnpDeviceId: 'x', driverVersion: '1', isIntegrated: false, dedicatedVramBytes: { value: vram * GiB, status: 'available', source: 't' } }], status: 'available', source: 't' }
}) as unknown as SystemProfile
const m = (id: string, downloads = 0) => ({ id, downloads, likes: 0, gated: false, lastModified: null })

describe('hub fit', () => {
  it('parses sizes from repo names', () => {
    expect(paramsB('unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF')).toBe(30)
    expect(paramsB('x/Mixtral-8x7B-v0.1-GGUF')).toBe(56)
    expect(paramsB('x/Llama-3.2-3B-Instruct-GGUF')).toBe(3)
    expect(paramsB('LiquidAI/LFM2.5-230M-GGUF')).toBe(0.23)
    expect(paramsB('CMSManhattan/JiRackUltra_14b')).toBe(14)
    expect(paramsB('unsloth/gpt-oss-20b-GGUF')).toBe(20)
    expect(paramsB('TheBloke/phi-GGUF')).toBeNull()
  })
  it('tiers by VRAM, then RAM; drops what does not fit or has no size', () => {
    const r = fitModels([m('a/Big-70B', 9), m('a/Mid-14B', 5), m('a/Small-7B', 1), m('a/Tiny-3B', 3), m('a/NoSize')], prof(8, 32))
    expect(r.map((x) => [x.id, x.fit])).toEqual([['a/Tiny-3B', 'gpu'], ['a/Small-7B', 'gpu'], ['a/Mid-14B', 'offload']])
    expect(fitModels([m('a/Small-7B')], prof(null, 16))[0].fit).toBe('cpu')
  })
})
