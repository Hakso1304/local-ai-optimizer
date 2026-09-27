// Sibling quantizations of a local GGUF in its linked HF repo (same base, other quant). Pure except refreshSiblings.
import { basename } from 'node:path'
import type { SiblingQuant } from '../../shared/bench-types'
import { listGgufFiles, quantFromName, type HfGgufFile } from './hf'
import { writeSidecar } from './modelcard'

const SHARD = /-\d{5}-of-\d{5}$/i
const QUANT = /[-_.](?:UD[-_])?(I?Q\d_[A-Z0-9]+(?:_[A-Z0-9]+)?|BF16|F16|F32|MXFP4)$/i

/** "Qwen3.8-27B-UD-Q4_K_XL-00001-of-00002.gguf" → "qwen3.8-27b" (quant, UD- marker, shard suffix removed). */
export function quantStem(path: string): string {
  return basename(path).replace(/\.gguf$/i, '').replace(SHARD, '').replace(QUANT, '').toLowerCase()
}

/** Same-stem files of the repo other than `selfPath`'s quant, shards summed into one entry (first shard's path). */
export function siblingsFromFiles(repoId: string, files: HfGgufFile[], selfPath: string): SiblingQuant[] {
  const stem = quantStem(selfPath), self = quantFromName(basename(selfPath))
  const by = new Map<string, SiblingQuant>()
  for (const f of files) {
    if (/mmproj/i.test(f.path) || quantStem(f.path) !== stem || !f.quant || f.quant === self) continue
    const k = `${f.quant}|${f.path.replace(/-\d{5}-of-\d{5}\.gguf$/i, '')}`
    const e = by.get(k)
    if (e) { e.sizeBytes += f.sizeBytes; e.shards = (e.shards ?? 1) + 1; if (f.shard?.index === 1) e.path = f.path }
    else by.set(k, { repoId, path: f.path, quant: f.quant, sizeBytes: f.sizeBytes, ...(f.shard ? { shards: 1 } : {}) })
  }
  return [...by.values()].sort((a, b) => a.sizeBytes - b.sizeBytes || (a.path < b.path ? -1 : 1))
}

/** Fetch the repo's file list once and cache the siblings in <model>.meta.json (only for a linked repo). */
export async function refreshSiblings(modelPath: string, repoId: string, list: typeof listGgufFiles = listGgufFiles, opts: { token?: string } = {}): Promise<SiblingQuant[]> {
  const siblings = siblingsFromFiles(repoId, await list(repoId, opts), modelPath)
  writeSidecar(modelPath, { siblings, siblingsFetchedAt: new Date().toISOString() })
  return siblings
}
