import type { RuntimeDetection } from '../../shared/types'
import { LlamaCppBackend } from './llamacpp'
import { LmStudioBackend, OllamaBackend } from './others'

/** Detect all known runtimes in parallel. Each detect() never throws. */
export function detectRuntimes(llamaVendorDir: string): Promise<RuntimeDetection[]> {
  return Promise.all([new LlamaCppBackend(llamaVendorDir), new OllamaBackend(), new LmStudioBackend()].map((b) => b.detect()))
}
