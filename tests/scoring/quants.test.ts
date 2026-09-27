// Sibling quantization suggestions (planner) with the real Qwen3.8-27B repo file list (unsloth, 2026-09-28).
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import type { ModelMeta, VramBudgetObservation } from '../../src/shared/bench-types'
import { generateCandidates, quantSuggestions } from '../../src/core/benchmark/candidates'
import { quantFromName } from '../../src/core/hub/hf'
import { quantStem, refreshSiblings, siblingsFromFiles } from '../../src/core/hub/quants'
import { parseDownloadAction } from '../../src/core/interpret/catalog'
import { interpret, verdicts } from '../../src/core/interpret'
import { WORKLOADS } from '../../src/core/scoring/workloads'
import { machine } from './helpers'

const GiB = 1024 ** 3
const REPO = 'unsloth/Qwen3.8-27B-GGUF'
const LIST: [string, number][] = [
  ['Qwen3.8-27B-UD-IQ4_XS.gguf', 14.25e9], ['Qwen3.8-27B-UD-Q4_K_S.gguf', 15.36e9], ['Qwen3.8-27B-UD-Q3_K_XL.gguf', 13.15e9],
  ['Qwen3.8-27B-UD-Q4_K_M.gguf', 16464440224], ['mmproj-F16.gguf', 0.9e9],
  ['BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf', 27e9], ['BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf', 27e9]
]
const files = LIST.map(([path, sizeBytes]) => {
  const s = /-(\d{5})-of-(\d{5})\.gguf$/.exec(path)
  return { path, sizeBytes, sha256: null, quant: quantFromName(path), shard: s ? { index: Number(s[1]), count: Number(s[2]) } : null }
})
const SELF = 'D:/llm-models/Qwen3.8-27B-UD-Q4_K_M.gguf'
const siblings = siblingsFromFiles(REPO, files, SELF)
const q27: ModelMeta = {
  id: SELF, name: 'Qwen3.8-27B', fileBytes: 16464440224, paramCount: 27e9, quant: 'Q4_K_M', arch: 'qwen35', ctxTrain: 262144, layers: 65,
  nEmbd: 5120, heads: 24, headsKv: 4, keyLength: 256, valueLength: 256, nVocab: 248320, slidingWindow: null, fullAttentionInterval: 4,
  baseModelId: `${REPO}#qwen3.8-27b`, siblingQuants: siblings
}
const M = () => machine(17095983104)

describe('sibling quantizations from the linked repo', () => {
  it('stem ignores quant, UD- marker and shard suffix; siblings exclude self, mmproj; shards are summed', () => {
    expect(quantStem('Qwen3.8-27B-UD-IQ4_XS.gguf')).toBe('qwen3.8-27b')
    expect(quantStem('BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf')).toBe('qwen3.8-27b')
    expect(siblings.map((s) => [s.quant, s.sizeBytes])).toEqual([['Q3_K_XL', 13.15e9], ['IQ4_XS', 14.25e9], ['Q4_K_S', 15.36e9], ['BF16', 54e9]])
    expect(siblings.find((s) => s.quant === 'BF16')).toMatchObject({ path: 'BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf', shards: 2 })
  })
  it('refreshSiblings caches the list in <model>.meta.json', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'lao-q-')), p = join(dir, 'Qwen3.8-27B-UD-Q4_K_M.gguf')
    writeFileSync(p, '')
    const got = await refreshSiblings(p, REPO, async () => files)
    expect(got).toHaveLength(4)
    expect(JSON.parse(readFileSync(`${p}.meta.json`, 'utf8'))).toMatchObject({ siblings: got, siblingsFetchedAt: expect.any(String) })
  })
})

describe('quantSuggestions (estimated, not benchmarked)', () => {
  it('Q4_K_M 27B fits 55/65 at 8K: IQ4_XS and Q3_K_XL are estimated to fully offload, Q4_K_S partially; BF16 never', () => {
    const s = quantSuggestions(M(), q27, WORKLOADS.max_quality)
    expect(s.map((x) => [x.sibling.quant, x.gpuLayers, x.speedClass])).toEqual([['Q3_K_XL', 65, 'full-gpu'], ['IQ4_XS', 65, 'full-gpu'], ['Q4_K_S', 59, 'partial']])
    expect(s[1]).toMatchObject({ ctx: 8192, currentGpuLayers: 55, layers: 65, estVramBytes: { kind: 'estimated' } })
    expect(s[1].text).toBe('not benchmarked: sibling quantization Qwen3.8-27B-UD-IQ4_XS.gguf (13.3 GiB) would fit with 65/65 layers on GPU at 8K (estimated vs the adapter budget; no measured per-process ceiling on this backend yet — it may be lower; this file: 55/65) — full offload — download to include')
  })
  it('a measured per-process ceiling tightens the estimate (and is named)', () => {
    const obs: VramBudgetObservation = { kind: 'capacity', qualified: true, ceilingBytes: 13.25 * GiB, modelId: 'x', ctx: 8192, kvType: 'f16', gpuLayers: 65, kvBytes: null, largestBufferBytes: 13 * GiB, observedAt: 0,
      origin: { sessionId: 's', configId: 'c', status: 'pass', attempts: 2, firstPeakVramBytes: null, firstResidentSharedBytes: null } }
    const s = quantSuggestions({ ...M(), vramBudgetObservations: [obs] }, q27, WORKLOADS.max_quality)
    const iq = s.find((x) => x.sibling.quant === 'IQ4_XS')!
    expect(iq.text).toMatch(/estimated vs the measured per-process budget/)
  })
  it('none when the file already fully offloads, when there is no linked repo, or when the sibling is local', () => {
    expect(quantSuggestions(M(), { ...q27, siblingQuants: undefined }, WORKLOADS.max_quality)).toEqual([])
    expect(quantSuggestions(M(), { ...q27, fileBytes: 8 * GiB, layers: 65 }, WORKLOADS.max_quality)).toEqual([])
    expect(quantSuggestions(M(), q27, WORKLOADS.max_quality, undefined, ['Qwen3.8-27B-UD-IQ4_XS.gguf']).map((x) => x.sibling.quant)).not.toContain('IQ4_XS')
  })
  it('generateCandidates carries them as structured suggestions', () => {
    expect(generateCandidates(M(), q27, { backend: 'vulkan' }, WORKLOADS.max_quality).suggestions?.map((x) => x.sibling.quant)).toEqual(['Q3_K_XL', 'IQ4_XS', 'Q4_K_S'])
  })
})

describe('I-9.2 insight + download action', () => {
  it('carries repoId/path so the Hub page can start it in one click', () => {
    const cand = { model: q27, config: { id: `${SELF}|ngl=55|kv=f16|t=8`, modelId: SELF, device: 'Vulkan0', gpuLayers: 55, gpuLayersAll: false, kvType: 'f16' as const, flashAttn: true, threads: 8, ctxSteps: [8192], skippedSteps: [], estVramBytes: { value: 14 * GiB, kind: 'estimated' as const }, estRamBytes: { value: 2 * GiB, kind: 'estimated' as const }, notes: [] }, runs: [], quality: [] }
    const ins = interpret(verdicts({ candidates: [cand], machine: M() }, 'max_quality', {})).filter((i) => i.ruleId === 'I-9.2')
    const iq = ins.find((i) => /IQ4_XS/.test(i.text))!
    expect(iq.action).toBe(`download ${REPO}/Qwen3.8-27B-UD-IQ4_XS.gguf`)
    expect(parseDownloadAction(iq.action)).toEqual({ repoId: REPO, path: 'Qwen3.8-27B-UD-IQ4_XS.gguf' })
    expect(parseDownloadAction(`download ${REPO}/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf`)).toEqual({ repoId: REPO, path: 'BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf' })
    expect(parseDownloadAction('use-context 16K')).toBeNull()
  })
})
