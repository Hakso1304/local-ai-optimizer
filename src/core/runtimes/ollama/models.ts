import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, relative, sep } from 'node:path'
import type { ModelInfo } from '../../../shared/types'
import { readGgufMetadata } from '../../models/gguf'

// Ollama / LM Studio model stores on disk. Neither app is installed on the dev machine, so the layout is taken from
// upstream, not observed:
//   Ollama (github.com/ollama/ollama, server/modelpath.go + manifest.go; docs/faq.md "Where are models stored?"):
//     <root>/manifests/<registry>/<namespace>/<model>/<tag>   JSON { layers: [{ mediaType, digest: "sha256:<hex>", size }] }
//     <root>/blobs/sha256-<hex>                                content-addressed; the image.model layer is a plain GGUF
//     root = %OLLAMA_MODELS% or %USERPROFILE%\.ollama\models. Default registry/namespace: registry.ollama.ai/library.
//   LM Studio (lmstudio.ai docs "Model directory"): %USERPROFILE%\.lmstudio\models\<publisher>\<repo>\*.gguf,
//     legacy %USERPROFILE%\.cache\lm-studio\models. Plain GGUF files → findGgufModels() lists them as-is.

const MODEL_LAYER = 'application/vnd.ollama.image.model'
const PARAMS_LAYER = 'application/vnd.ollama.image.params'

export interface OllamaModel {
  /** As `ollama list` shows it: "llama3.1:8b"; other namespace "ns/name:tag"; other registry "host/ns/name:tag". */
  name: string
  manifestPath: string
  /** GGUF blob; llama-server can load it directly with -m (our MVP benchmark path, read-only). */
  blobPath: string
  sizeBytes: number
  paramsBlobPath?: string
  /** false when the manifest points at a blob that is not on disk (partial pull / pruned). */
  exists: boolean
}

export const defaultOllamaRoot = (): string => process.env.OLLAMA_MODELS || join(homedir(), '.ollama', 'models')

export const defaultLmStudioDirs = (): string[] => [join(homedir(), '.lmstudio', 'models'), join(homedir(), '.cache', 'lm-studio', 'models')]

const blobFile = (root: string, digest: string) => join(root, 'blobs', digest.replace(':', '-'))

function displayName(parts: string[]): string {
  // parts = [registry, namespace, model, tag]; deeper paths (unusual) keep every segment.
  const tag = parts.at(-1)!
  let path = parts.slice(0, -1)
  if (path[0] === 'registry.ollama.ai') path = path.slice(1)
  if (path.length === 2 && path[0] === 'library') path = path.slice(1)
  return `${path.join('/')}:${tag}`
}

/** Every manifest with a model layer. Missing root → []; unreadable/odd manifests are skipped. */
export async function listOllamaModels(root = defaultOllamaRoot()): Promise<OllamaModel[]> {
  const manifests = join(root, 'manifests')
  if (!existsSync(manifests)) return []
  const out: OllamaModel[] = []
  for (const rel of readdirSync(manifests, { recursive: true, encoding: 'utf8' }).sort()) {
    const manifestPath = join(manifests, rel)
    if (!statSync(manifestPath).isFile()) continue
    let layers: { mediaType?: string; digest?: string; size?: number }[]
    try {
      layers = (JSON.parse(readFileSync(manifestPath, 'utf8')) as { layers?: typeof layers }).layers ?? []
    } catch {
      continue
    }
    const model = layers.find((l) => l.mediaType === MODEL_LAYER && typeof l.digest === 'string')
    if (!model) continue
    const blobPath = blobFile(root, model.digest!)
    const exists = existsSync(blobPath)
    const params = layers.find((l) => l.mediaType === PARAMS_LAYER && typeof l.digest === 'string')
    out.push({
      name: displayName(relative(manifests, manifestPath).split(sep)),
      manifestPath, blobPath, exists,
      sizeBytes: exists ? statSync(blobPath).size : (model.size ?? 0),
      ...(params ? { paramsBlobPath: blobFile(root, params.digest!) } : {})
    })
  }
  return out
}

/** ModelInfo-shaped entry for the model list. Benchmarking still goes through llama-server on blobPath.
 *  ponytail: ModelInfo.runtime is 'llamacpp' only in shared/types.ts; #2 widens it to include 'ollama'. */
export type OllamaModelInfo = Omit<ModelInfo, 'runtime'> & { runtime: 'ollama'; ollamaName: string }

export async function toModelInfo(m: OllamaModel, read = readGgufMetadata): Promise<OllamaModelInfo> {
  const base = { id: m.blobPath, name: m.name, path: m.blobPath, sizeBytes: m.sizeBytes, runtime: 'ollama' as const, ollamaName: m.name }
  if (!m.exists) return { ...base, meta: null, metaError: `blob missing: ${m.blobPath}` }
  try {
    return { ...base, meta: await read(m.blobPath) }
  } catch (e) {
    return { ...base, meta: null, metaError: (e as Error).message }
  }
}
