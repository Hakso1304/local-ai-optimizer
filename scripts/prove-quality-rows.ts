// Re-render stored quality rows against an already running local llama-server.
// Usage: npx tsx scripts/prove-quality-rows.ts --in docs/session-evidence.json --out docs/session-evidence-proved.json --port 8080 [--session 5]
// This reads an immutable export, writes a new copy, and never opens a DB or launches a model.
import { createHash } from 'node:crypto'
import { readFileSync, statSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { buildQualityPrompts, suiteFor } from '../src/core/quality'
import { proofRowId, templateAlternate, templateKwargsFor, type GenRow } from '../src/core/benchmark/gen'
import { DEFAULT_SESSION_CONFIG } from '../src/core/benchmark/session'
import { TEMPLATE_DATE } from '../src/core/runtimes/llamacpp'
import type { CandidateConfig, GenConfig, ModelMeta } from '../src/shared/bench-types'

const flag = (name: string): string | undefined => {
  const i = process.argv.indexOf(name)
  return i < 0 ? undefined : process.argv[i + 1]
}
const input = flag('--in'), output = flag('--out'), port = Number(flag('--port'))
if (!input || !output || !Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error('required: --in <immutable session evidence JSON> --out <new JSON> --port <already running local llama-server>')
}
if (resolve(input) === resolve(output)) throw new Error('input and output must differ')
const sha256 = (s: string | Buffer) => createHash('sha256').update(s).digest('hex')
const raw = readFileSync(input)
const replayId = sha256(raw)
const artifact = JSON.parse(raw.toString('utf8')) as {
  sessionId: number
  session: { payload: { request: { qualityMode?: 'quick' | 'thorough'; qualitySeed?: number; genConfigs?: GenConfig[] }; candidates: { config: CandidateConfig; model: ModelMeta }[] } }
  qualityResults: { id: number; model_id: string; payload: GenRow & { ctx?: number; requestedTemplateKwargs?: Record<string, unknown>; appliedTemplateKwargs?: Record<string, unknown>; checkerVersion?: string; templateHash?: string | null; modelFingerprint?: string | null; acceptedSampling?: Record<string, unknown> | null } }[]
  [key: string]: unknown
}
if (!Array.isArray(artifact.qualityResults) || !artifact.session?.payload?.request) throw new Error('input is not a session-evidence export')
const previousReplay = artifact.rowProofReplay as { runId?: string } | undefined
if (previousReplay && (typeof previousReplay.runId !== 'string' || !/^[0-9a-f]{64}$/.test(previousReplay.runId))) {
  throw new Error('input replay artifact has no valid run lineage id')
}
// The root generation record is immutable. Validate it before any template
// request: a partial import must never be repaired into apparent proof.
for (const q of artifact.qualityResults) {
  const row = q.payload, prior = row.proofProvenance, origin = prior?.origin
  if (previousReplay && (!origin || prior?.mode !== 'live-template-replay' ||
      !Array.isArray(origin.lineage) || origin.lineage.at(-1) !== previousReplay.runId)) {
    throw new Error(`row ${q.id}: incoherent replay lineage or origin provenance`)
  }
  if (!origin) {
    if (prior?.mode === 'live-template-replay') throw new Error(`row ${q.id}: replay provenance without origin`)
    continue // legacy: missing root record can only be reconstructed
  }
  const validTime = (x: unknown) => typeof x === 'string' && !Number.isNaN(Date.parse(x)) && new Date(x).toISOString() === x
  const coherent = typeof origin.generationPromptHashPresent === 'boolean' &&
    prior?.originalPromptHashPresent === origin.generationPromptHashPresent &&
    prior.status === (origin.generationPromptHashPresent ? 'original' : 'reconstructed') &&
    Array.isArray(origin.lineage) && origin.lineage.every((x) => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x)) &&
    (prior.mode === 'runtime'
      ? origin.generationPromptHashPresent && origin.firstReplayAt === null && origin.lineage.length === 0 && !previousReplay
      : prior.mode === 'live-template-replay' && validTime(origin.firstReplayAt) && origin.lineage.length > 0 && !!previousReplay) &&
    (!origin.generationPromptHashPresent || typeof row.promptSha256 === 'string' && /^[0-9a-f]{64}$/.test(row.promptSha256))
  if (!coherent) throw new Error(`row ${q.id}: incoherent or partial origin provenance`)
}
const expectedSession = flag('--session')
if (expectedSession && Number(expectedSession) !== artifact.sessionId) throw new Error(`session mismatch: export is ${artifact.sessionId}`)
const request = artifact.session.payload.request
if (!Number.isInteger(request.qualitySeed)) throw new Error('persisted qualitySeed is required to reconstruct quality prompts')
const suite = suiteFor(request.qualityMode, request.qualitySeed!)
const qualityFillerMax = Number(flag('--quality-filler-max') ?? DEFAULT_SESSION_CONFIG.qualityFillerMax)
if (!Number.isInteger(qualityFillerMax) || qualityFillerMax < 0) throw new Error('invalid --quality-filler-max')
const propsResponse = await fetch(`http://127.0.0.1:${port}/props`, { signal: AbortSignal.timeout(10_000) })
if (!propsResponse.ok) throw new Error(`/props returned HTTP ${propsResponse.status}`)
const props = await propsResponse.json() as { model_path?: string; chat_template?: string }
if (!props.model_path || !props.chat_template) throw new Error('/props must disclose model_path and chat_template')
const serverTemplateHash = sha256(props.chat_template)
const samePath = (a: string, b: string) => resolve(a).toLowerCase() === resolve(b).toLowerCase()
const modelCandidates = artifact.session.payload.candidates.filter((c) => samePath(c.model.id, props.model_path!))
if (!modelCandidates.length) throw new Error('loaded /props model_path does not match an exported candidate')
const selectedModel = modelCandidates[0].model
if (statSync(props.model_path).size !== selectedModel.fileBytes) throw new Error('loaded model file size differs from export')
const modelRows = artifact.qualityResults.filter((q) => samePath(q.model_id, selectedModel.id))
if (!modelRows.length) throw new Error('no quality rows for the loaded model')
if (artifact.qualityResults.length !== modelRows.length) throw new Error('export has other models; prove each on its own loaded server/export')
for (const q of modelRows) {
  if (q.payload.templateHash !== serverTemplateHash) throw new Error(`row ${q.id}: template hash differs from loaded server`)
  if (q.payload.modelFingerprint !== `${selectedModel.id}#${selectedModel.fileBytes}`) throw new Error(`row ${q.id}: model fingerprint differs`)
  if (q.payload.checkerVersion !== suite.suite || (suite.suiteSeed !== null && q.payload.suiteSeed !== suite.suiteSeed)) {
    throw new Error(`row ${q.id}: suite/version/seed differs from reconstructed suite`)
  }
}
const promptMaps = new Map<number, Map<string, { messages: { role: 'user'; content: string }[] }>>()
const promptFor = (ctx: number, testId: string) => {
  let map = promptMaps.get(ctx)
  if (!map) {
    map = new Map(buildQualityPrompts(suite, { fillerTokens: Math.min(qualityFillerMax, Math.floor(ctx * 0.6)), thinking: false })
      .map((p) => [p.testId, p]))
    promptMaps.set(ctx, map)
  }
  return map.get(testId)
}
const applyTemplate = async (messages: { role: string; content: string }[], kwargs: Record<string, unknown>): Promise<string> => {
  const response = await fetch(`http://127.0.0.1:${port}/apply-template`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages, chat_template_kwargs: { date_string: TEMPLATE_DATE, ...kwargs } }),
    signal: AbortSignal.timeout(15_000)
  })
  if (!response.ok) throw new Error(`POST /apply-template returned HTTP ${response.status}`)
  const body = await response.json() as { prompt?: unknown }
  if (typeof body.prompt !== 'string') throw new Error('/apply-template response has no prompt')
  return body.prompt
}
let proved = 0, unproved = 0, contradicted = 0, reconstructed = 0
for (const q of modelRows) {
  const row = q.payload
  const gen = request.genConfigs?.find((g) => g.id === row.genId)
  const candidate = modelCandidates.find((c) => c.config.id === row.configId)
  if (!gen || !candidate || !Number.isInteger(row.ctx) || !promptFor(row.ctx!, row.testId)) throw new Error(`row ${q.id}: config/gen/context/test cannot be reconstructed`)
  const messages = promptFor(row.ctx!, row.testId)!.messages
  const expectedKwargs = templateKwargsFor(selectedModel, gen)
  const kwargs = row.requestedTemplateKwargs ?? {}
  if (JSON.stringify(kwargs) !== JSON.stringify(expectedKwargs ?? {})) throw new Error(`row ${q.id}: recorded kwargs differ from model/gen contract`)
  const sample = row.sample
  if (!Number.isInteger(sample) || sample! < 1) throw new Error(`row ${q.id}: sample is missing`)
  const requestedSampling = { temperature: gen.temperature, topP: gen.topP ?? null, topK: gen.topK ?? null, minP: gen.minP ?? null, seed: sample! }
  const prior = row.proofProvenance
  if (previousReplay && (!prior?.origin || !Array.isArray(prior.origin.lineage) ||
      prior.origin.lineage.length === 0 || previousReplay.runId !== prior.origin.lineage.at(-1))) {
    throw new Error(`row ${q.id}: replay lineage is missing or does not match the input artifact`)
  }
  if (!previousReplay && prior?.mode === 'live-template-replay') throw new Error(`row ${q.id}: replay provenance without artifact lineage`)
  const origin = prior?.origin ?? {
    generationPromptHashPresent: false,
    firstReplayAt: null, lineage: [] as string[]
  }
  if (typeof origin.generationPromptHashPresent !== 'boolean' || !Array.isArray(origin.lineage) ||
      (previousReplay && origin.firstReplayAt === null) ||
      (origin.generationPromptHashPresent && prior?.status === 'reconstructed')) {
    throw new Error(`row ${q.id}: incoherent origin provenance`)
  }
  const originalPromptHash = origin.generationPromptHashPresent ? row.promptSha256 : null
  Object.assign(row, { original: (row as typeof row & { original?: unknown }).original ?? {
    promptSha256: row.promptSha256 ?? null, renderProof: row.renderProof ?? null, proofProvenance: prior ?? null,
    appliedTemplateKwargs: row.appliedTemplateKwargs ?? null, templateKwargProof: row.templateKwargProof ?? null,
    requestedSampling: row.requestedSampling ?? null
  } })
  const requestedRender = await applyTemplate(messages, kwargs)
  const renderedSha256 = sha256(requestedRender)
  if (!originalPromptHash) reconstructed++
  const promptSha256 = originalPromptHash ?? renderedSha256
  const proof: NonNullable<GenRow['templateKwargProof']> = {}
  const effortKey = selectedModel.genKnobs?.effortKw ?? 'reasoning_effort'
  for (const [key, requested] of Object.entries(kwargs)) {
    const alternate = templateAlternate(key, requested, effortKey, selectedModel.genKnobs?.effortValues ?? [])
    if (alternate === undefined) {
      proof[key] = { requested, counterfactual: null, requestedSha256: renderedSha256, counterfactualSha256: null, status: 'unavailable' }
      continue
    }
    try {
      const counterfactualSha256 = sha256(await applyTemplate(messages, { ...kwargs, [key]: alternate }))
      proof[key] = { requested, counterfactual: alternate, requestedSha256: renderedSha256, counterfactualSha256,
        status: counterfactualSha256 === renderedSha256 ? 'unchanged' : 'proved' }
    } catch {
      proof[key] = { requested, counterfactual: alternate, requestedSha256: renderedSha256, counterfactualSha256: null, status: 'unavailable' }
    }
  }
  const keys = Object.keys(kwargs).filter((key) => proof[key]?.status === 'proved')
  const allProved = keys.length === Object.keys(kwargs).length
  const status: NonNullable<GenRow['renderProof']>['status'] = promptSha256 !== renderedSha256 ? 'contradicted'
    : !originalPromptHash ? 'reconstructed' : allProved ? 'proved' : 'unproved'
  row.promptSha256 = promptSha256
  row.renderProof = { rowId: proofRowId(row), promptSha256, renderedSha256,
    counterfactualSha256: Object.keys(kwargs).length ? proof[Object.keys(kwargs)[0]]?.counterfactualSha256 ?? null : null,
    counterfactuals: Object.fromEntries(Object.keys(kwargs).map((key) => [key, proof[key]?.counterfactualSha256 ?? null])),
    keys, status }
  row.templateKwargProof = proof
  row.requestedSampling = requestedSampling
  if (status === 'proved' && Object.keys(kwargs).length) row.appliedTemplateKwargs = kwargs
  else delete row.appliedTemplateKwargs
  Object.assign(row, { proofProvenance: { mode: 'live-template-replay', sourceRowId: q.id,
    originalPromptHashPresent: !!originalPromptHash, status: originalPromptHash ? 'original' : 'reconstructed',
    originalGenerationPromptMatch: originalPromptHash ? promptSha256 === renderedSha256 : null,
    origin: { generationPromptHashPresent: origin.generationPromptHashPresent,
      firstReplayAt: origin.firstReplayAt ?? new Date().toISOString(), lineage: [...origin.lineage, replayId] } } })
  if (status === 'proved') proved++
  else if (status === 'contradicted') contradicted++
  else unproved++
}
Object.assign(artifact, { rowProofReplay: {
  runId: replayId,
  sourceSha256: sha256(raw), sourcePath: resolve(input), sessionId: artifact.sessionId,
  serverModelPath: props.model_path, serverTemplateSha256: serverTemplateHash,
  promptReconstruction: 'Rows without an original prompt hash use deterministic suite re-render; original generation prompt equality cannot be independently checked.',
  qualityFillerMax, counts: { proved, unproved, contradicted, reconstructed }
} })
writeFileSync(output, JSON.stringify(artifact, null, 2), { flag: 'wx' })
console.log(`${output}: session ${artifact.sessionId}, proved ${proved}, unproved ${unproved}, contradicted ${contradicted}; ${reconstructed} original prompt hashes reconstructed`)
