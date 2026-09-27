# Local AI Optimizer

Windows desktop app (Electron + React + TypeScript) that scans the PC, benchmarks locally runnable
LLM configurations with real inference (llama.cpp), and will recommend model + config per workload
from measured data.

## Prerequisites

- Windows 10/11, PowerShell 5.1 (built in), Node 24+, npm
- Internet access for first-time runtime/model download
- No admin rights, compilers or native Node modules required

## Run

```sh
npm install              # if electron.exe is missing afterwards: node node_modules/electron/install.js
npm run setup:runtime    # downloads newest official llama.cpp Windows Vulkan build -> vendor/llama.cpp
npm run dev              # dev window with hot reload
npm run build            # typecheck + production build to out/
npm run preview          # run the production build
npm test                 # vitest unit tests (no real machine access needed)
```

Test model (optional, for manual smoke tests):

```sh
curl -L -o models/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

## Layout

- `src/main` Electron main (window + IPC only) · `src/preload` contextBridge API · `src/renderer` React UI
- `src/core/system` system scanner (PowerShell/CIM/registry, every value carries status + source)
- `src/core/runtimes` `InferenceBackend` interface; llama.cpp adapter; Ollama / LM Studio detection stubs
- `src/shared` types shared by main and renderer · `tests` vitest

## Status

- Done: scaffold, System page (OS/CPU/RAM/GPU with true VRAM/CUDA/disks/runtimes), llama.cpp download,
  detect, launch + health check.
- Not yet: benchmarking, context stress tests, scoring/recommendations, persistence. Dashboard,
  Benchmark, Models and Results pages are placeholders.
