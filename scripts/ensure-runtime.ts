// Downloads the latest official llama.cpp Windows Vulkan build into vendor/llama.cpp (idempotent).
import { join } from 'node:path'
import { LlamaCppBackend } from '../src/core/runtimes/llamacpp'

const backend = new LlamaCppBackend(join(process.cwd(), 'vendor', 'llama.cpp'))
backend
  .ensureRuntime((m) => console.log(m))
  .then((d) => console.log(JSON.stringify(d, null, 2)))
  .catch((e) => { console.error(`ensureRuntime failed: ${(e as Error).message}`); process.exit(1) })
