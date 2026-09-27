# Local AI Optimizer

Windows desktop app that finds the best way to run local LLMs on **this** PC. It scans the hardware, benchmarks
real inference with llama.cpp across model / GPU-offload / context-size configurations, runs a small deterministic
quality suite, and recommends a configuration per workload (chat, coding, long-context coding, reasoning, document
analysis, fast assistant, max quality). Every number is labelled MEASURED, DECLARED, ESTIMATED or UNAVAILABLE —
nothing is guessed silently.

## Features

- Hardware scan with provenance labels (GPU/VRAM, CPU, RAM, disks); llama.cpp runtime install (Vulkan, or CUDA on NVIDIA)
- Models from your folders, LM Studio and Ollama; optional **download from Hugging Face** (search, gated-model login, resumable, sha256-verified)
- Benchmark sessions per workload: context ladder with cliff/spill detection, quality suite, live telemetry, pause/resume, retry failed steps, rerun a selected configuration
- **Heavy models**: models larger than VRAM can be benchmarked with partial GPU offload (flagged as degraded speed)
- Recommendation with a score breakdown, recommended context size and plain-English reasons; per-run telemetry chart; export to llama-server / Ollama Modelfile / LM Studio / JSON

## Usage

1. System page: *Install llama.cpp runtime* (first run).
2. Models page: check your models. **Download from Hugging Face**: Download page → search → pick a GGUF file → Download (sign in with a read token for gated models). *(Download page wiring pending.)*
3. Benchmark page: pick a workload and models. **Include heavy models** to also test models whose full GPU offload does not fit (partial offload, slow). Then Start; Pause/Resume and Cancel work mid-run.
4. Results page: recommendation, charts, per-run telemetry, and the *Export* menu.

## Prerequisites

- Windows 11 (the scanner and telemetry use PowerShell 5.1, WMI/CIM, the registry and `typeperf`); no admin rights
- A GPU with a Vulkan driver (AMD / NVIDIA / Intel). CPU-only works but is slow
- Disk: ~30 MB for the runtime, plus the GGUF models you want to test (0.6–10+ GB each)
- Internet on first run (llama.cpp runtime download from github.com/ggml-org/llama.cpp)
- For development: Node 24+ and npm. No compilers and no native Node modules (storage is the built-in `node:sqlite`)

## Runtimes

- **llama.cpp (`llama-server`, official Windows Vulkan build)** is the only benchmark runtime. The app downloads
  it on first run (System page → *Install llama.cpp runtime*); in dev, `npm run setup:runtime` puts it in `vendor/llama.cpp`.
- **Ollama / LM Studio**: detected only (is the API up, where are the models). Their GGUF files can be benchmarked
  through llama-server; the export can emit an Ollama Modelfile / LM Studio settings (unverified translations).

## Develop, build, package

```sh
npm install
npm run setup:runtime    # dev only: newest official llama.cpp Vulkan build -> vendor/llama.cpp
npm run dev              # dev window with hot reload
npm test                 # vitest unit tests (no GPU needed; fake llama-server for process tests)
npm run build            # typecheck + production build to out/
npm run preview          # run the production build
npm run package          # build + electron-builder -> dist/ (portable .exe and NSIS installer, unsigned)
LAO_SEED_DEMO=1 npm run dev   # UI demo with fixture data in a separate optimizer-demo.db, flagged "DEMO DATA"
```

Models are read from `<project>/models` (dev) or `%APPDATA%\local-ai-optimizer\models` (packaged), plus any
`modelDirs` listed in `settings.json` in the same folder (none by default), plus LM Studio's model dirs and Ollama's
blob store. Results live in `optimizer.db` there. Dev runs use `%APPDATA%\local-ai-optimizer-dev` so they never touch
the installed app's data. Uninstalling keeps `%APPDATA%\local-ai-optimizer` (database, settings, downloaded runtime);
delete it by hand for a clean slate. Small test model:

```sh
curl -L -o models/qwen2.5-0.5b-instruct-q8_0.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
```

## How a benchmark works

For each selected model the app reads the GGUF header (architecture, layers, heads, context length, quantization)
and generates candidate configurations (full or partial GPU offload, KV-cache type, threads), pruning the ones a
memory estimate says cannot fit. Each candidate climbs a context ladder (2K, 4K, 8K … up to what fits): at every
rung llama-server is restarted with that `-c` and fixed flags (`-fit off`, explicit device), warmed up with a
same-size prompt, and measured twice (median). The app records TTFT (wall clock), prefill and decode tokens/s
(from llama-server's own timings) and per-process VRAM / shared-GPU memory / RAM from Windows performance
counters. A rung is flagged degraded or failed on crashes, OOM, sharp speed drops or VRAM spilling into shared
memory; the last clean rung is the *practical context ceiling*.

After the ladder, the quality suite (17 deterministic tests: instruction following, reasoning, code, structured
output, extraction, long-context recall) runs once per model at a context that passed. Scores for quality, speed,
prefill, latency, memory headroom, stability and context are weighted per workload; candidates failing a
workload's gates (quality floor, minimum context, stability) are not recommended. The Results page shows the
comparison, context/memory scaling charts with the cliff marked, the reasons verbatim, and an export of the exact
llama-server command.

## Current limitations

The full, code-checked list is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Highlights:

- AMD GPU temperature, power and clocks are UNAVAILABLE (they need a native vendor SDK); NVIDIA telemetry via
  nvidia-smi is fixture-tested only
- Telemetry uses English performance-counter names; on localized Windows those fields show unavailable
- 1 s telemetry granularity: very short steps may have no samples (shown as unavailable, never 0)
- Calibrated on one GPU (RX 9070 XT 16 GB) and two model families; the CUDA runtime path is untested on NVIDIA hardware
- The quality suite is small (17 tests): a relative signal, not a leaderboard
- Windows only; packaged builds are unsigned (SmartScreen will warn)

## Layout

- `src/main` Electron main (IPC, session runner wiring) · `src/preload` contextBridge API · `src/renderer` React UI
- `src/core/system` scanner · `src/core/runtimes` llama.cpp adapter, Ollama/LM Studio detection · `src/core/models` GGUF reader
- `src/core/telemetry` typeperf / nvidia-smi samplers · `src/core/benchmark` candidates + session runner
- `src/core/quality` test suite + sandbox · `src/core/scoring` cliffs, scores, recommendation · `src/core/storage` sqlite
- `docs/` design, benchmark method, architecture, limitations · `tests/` vitest
