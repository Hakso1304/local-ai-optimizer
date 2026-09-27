import { proofRowId } from '../benchmark/gen'

export interface QualityOriginRow {
  configId?: string; genId?: string; testId: string; sample?: number; suiteSeed?: number; generatorSeed?: number | string
  promptSha256?: unknown
  renderProof?: unknown
  proofProvenance?: unknown
}

export interface OriginContext { expectedLineageTail?: string; replay?: boolean }
export interface OriginValidation {
  ok: boolean
  classification: 'original' | 'reconstructed' | 'incoherent'
  reason: string | null
}

const hex = (x: unknown): x is string => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x)
const object = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && !Array.isArray(x)
const iso = (x: unknown): x is string => typeof x === 'string' && !Number.isNaN(Date.parse(x)) && new Date(x).toISOString() === x
const incoherent = (reason: string): OriginValidation => ({ ok: false, classification: 'incoherent', reason })
const reconstructed = (reason: string): OriginValidation => ({ ok: true, classification: 'reconstructed', reason })

/** One read-time check for runtime, replay and imported quality rows. Never mutates a row. */
export function validateQualityOrigin(row: QualityOriginRow, ctx: OriginContext = {}): OriginValidation {
  const prior = row.proofProvenance
  if (prior === undefined || prior === null) return reconstructed('generation origin provenance not recorded')
  if (!object(prior) || !object(prior.origin)) return incoherent('missing or partial origin provenance')
  const origin = prior.origin
  const generation = origin.generationPromptHashPresent
  if (typeof generation !== 'boolean' || typeof prior.originalPromptHashPresent !== 'boolean' ||
      prior.originalPromptHashPresent !== generation ||
      prior.status !== (generation ? 'original' : 'reconstructed')) return incoherent('generation origin flags or status disagree')
  if (!Array.isArray(origin.lineage) || !origin.lineage.every(hex)) return incoherent('origin lineage is not lowercase SHA-256 hashes')
  if (generation && !hex(row.promptSha256)) return incoherent('original generation prompt hash is missing or malformed')
  if (!generation && row.promptSha256 !== undefined && row.promptSha256 !== null) return incoherent('reconstructed prompt hash claims generation origin')
  if (prior.mode === 'runtime') {
    if (!generation || origin.firstReplayAt !== null || origin.lineage.length !== 0 || ctx.expectedLineageTail !== undefined) {
      return incoherent('runtime origin contains replay metadata or lacks its generation prompt hash')
    }
  } else if (prior.mode === 'live-template-replay') {
    if (!iso(origin.firstReplayAt) || origin.lineage.length === 0) return incoherent('replay origin timestamp or lineage is malformed')
    if (ctx.expectedLineageTail !== undefined && (!hex(ctx.expectedLineageTail) || origin.lineage.at(-1) !== ctx.expectedLineageTail)) {
      return incoherent('replay lineage does not match the source artifact')
    }
  } else return incoherent('unknown origin mode')
  const proof = row.renderProof
  if (proof !== undefined && proof !== null) {
    if (!object(proof) || proof.rowId !== proofRowId(row) || proof.promptSha256 !== row.promptSha256 ||
        proof.renderedSha256 !== row.promptSha256 || !Array.isArray(proof.keys) ||
        !proof.keys.every((k) => typeof k === 'string') ||
        !['proved', 'unproved', 'contradicted'].includes(String(proof.status))) {
      return incoherent('render proof is not bound to this row and prompt hash')
    }
    if (proof.status === 'contradicted') return incoherent('render proof contradicts the recorded prompt')
  }
  if (prior.mode === 'runtime' && generation && object(proof) && proof.status === 'proved') {
    return { ok: true, classification: 'original', reason: null }
  }
  return reconstructed(prior.mode === 'live-template-replay' ? 'template replay does not prove the generation request' : 'row-bound runtime render proof not verified')
}
