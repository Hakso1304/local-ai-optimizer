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
2. Models page: check your models. **Download from Hugging Face** (Download page, or the button on the Benchmark
   page): search → pick a repository → pick a GGUF file → choose one of your configured model folders → Download.
   Installed builds create a writable default folder at `%APPDATA%\local-ai-optimizer\models` (the install directory
   and app.asar are not writable). Portable builds create it on first launch; downloaded GGUFs remain there after uninstall.
   Signing in is optional; public models work without it. Gated models need a read token (*Open token page*, paste
   it; it is stored encrypted on this PC) and the model's license accepted on huggingface.co. Downloads show
   progress, speed and ETA, can be paused and resumed (partial file kept) or cancelled, are checked against the
   repository's sha256 when it publishes one, and appear on the Models page when finished.
3. Benchmark page: pick a workload and models. **Include heavy models** to also test models whose full GPU offload does not fit (partial offload, slow). Then Start; Pause/Resume and Cancel work mid-run.
   - **Required context** (Auto / 32K / 64K / 128K): Auto uses the workload's default. A fixed value extends the
     context ladder to that size and only recommends configurations that reach it; the workload's TTFT tolerance
     then becomes advisory (a long prompt is allowed to take long).
   - **Min decode t/s**: your floor for generation speed (blank = the workload's gate). It is a hard constraint and is
     never overridden: with a fixed Required context, configurations that reach it but are slower are listed as
     unmet-constraint alternatives, not recommended. 20–30 t/s is a common preference for large-scale work (a product
     choice, not a measured threshold).
   - Any selection can be benchmarked regardless of these gates; they only decide what gets recommended.
4. Results page: recommendation, charts, per-run telemetry, and the *Export* menu.

## Prerequisites

- Windows 11, x64 (developed and tested there; Windows 10 is untested). The scanner and telemetry use PowerShell 5.1,
  WMI/CIM, the registry and `typeperf`; no admin rights are needed
- A GPU with a Vulkan driver (AMD / NVIDIA / Intel). On integrated-only laptops the Vulkan iGPU is used and
  allocations are planned against available system RAM minus 4 GiB; the live floor is at least 4 GiB or 8% of
  total RAM, whichever is higher. The shared-memory aperture is not dedicated VRAM. NVIDIA gets the CUDA build
  when supported (untested on real NVIDIA hardware). CPU-only works but is slow.
- Disk: ~30 MB for the runtime (a CUDA build adds its runtime DLLs), plus the GGUF models you test (0.4–17+ GB each)
- Internet for the runtime download (github.com/ggml-org/llama.cpp) and optional Hugging Face downloads
- For development only: Node 24+ and npm. No compilers and no native Node modules (storage is the built-in `node:sqlite`)

## Runtimes

- **llama.cpp (`llama-server`)**, official Windows build (Vulkan, or CUDA + cudart on NVIDIA with a Vulkan fallback),
  is the only benchmark runtime. The app installs it on first run (System page → *Install llama.cpp runtime*); in dev,
  `npm run setup:runtime` puts the Vulkan build in `vendor/llama.cpp`.
- **Ollama / LM Studio**: detected (is the API up) and their GGUF files are listed as model sources. They are
  benchmarked through our own llama-server, never through Ollama/LM Studio. The export can emit an Ollama Modelfile /
  LM Studio settings (translations; LM Studio keys unverified).

## Backends (AMD: Vulkan vs ROCm/HIP)

- **Vulkan** is the default install. On a discrete AMD GPU the System page also offers *Install ROCm (HIP) runtime*:
  the official llama.cpp ROCm build (~245 MB) of the same release as the Vulkan one, with the HIP runtime bundled (no
  HIP SDK needed). It covers RDNA1–RDNA4, RX 9070 XT included. It lives in its own folder next to the Vulkan build.
- With both installed, the benchmark plans every config on both backends (toggle *Compare backends*, on by default).
  They are separate candidates (HIP ids end in `|hip`, device `ROCm0`), and the recommendation names the backend it
  chose. The export uses that backend's llama-server.
- Managed-memory oversubscription (`GGML_CUDA_ENABLE_UNIFIED_MEMORY`) is never enabled, so "fits in VRAM" stays measurable.
- **The HIP path is untested on real hardware** until the first calibration run. Whether the driver exposes the GPU to
  HIP is only known then; if it does not, the session runs on Vulkan and says so. See [docs/HIP-BACKEND.md](docs/HIP-BACKEND.md).

## Where things live

| What | Installed / portable app | Dev (`npm run dev`) |
|---|---|---|
| App data folder (below: *userData*) | `%APPDATA%\local-ai-optimizer` | `%APPDATA%\local-ai-optimizer-dev` (never touches the installed app's data) |
| Results database | *userData*`\optimizer.db` | same; `LAO_SEED_DEMO=1` uses a separate `optimizer-demo.db` |
| Settings (workload, required context, model folders) | *userData*`\settings.json` | same |
| llama.cpp runtime | *userData*`\runtime\llama.cpp` (downloaded on first run) | `<project>\vendor\llama.cpp` |
| Models scanned | *userData*`\models` + every folder in `settings.json` → `modelDirs` + LM Studio's model dirs + Ollama's blob store | `<project>\models` + the same extra sources |
| Hugging Face token | *userData*`\hf-token.bin`, encrypted with Windows DPAPI (never in plain text) | same |
| Running server record | *userData*`\llama-server.pid` (pid, exe path, start time; used to clean up after a crash) | same |
| Model-card cache | `<model>.gguf.meta.json` next to a downloaded or linked model | same |

There is no default model folder beyond *userData*`\models`: a folder like `D:\llm-models` is used only when it is
listed in `settings.json` → `modelDirs` (or chosen as a download target that is already configured). Uninstalling keeps
*userData* (database, settings, runtime, token); delete the folder by hand for a clean slate. Only one copy of the app
runs at a time per *userData* (the portable and installed builds share it); a second launch focuses the first window.

## Develop, build, package

```sh
npm install
npm run setup:runtime    # dev only: newest official llama.cpp Vulkan build -> vendor/llama.cpp
npm run dev              # dev window with hot reload
npm test                 # vitest unit tests (no GPU needed; fake llama-server for process tests)
npm run build            # typecheck + production build to out/
npm run preview          # run the production build
npm run package          # build + electron-builder -> dist/ (portable .exe and NSIS installer, unsigned)
                         # + dist/BUILD-INFO.txt: commit (DIRTY if uncommitted edits), rules version, test results, HIP flag
                         #   (LAO_BUILD_LABEL="nightly …" labels it; package from a clean worktree for a clean build)
LAO_SEED_DEMO=1 npm run dev   # UI demo with fixture data in a separate optimizer-demo.db, flagged "DEMO DATA"
```

Storage locations are listed in *Where things live* above. Small test model:

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
output, extraction, long-context recall) runs once per model on a context rung that passed (a CPU / KV-in-RAM config only if nothing else passed); thinking-capable models are also run with thinking on at low and medium effort. Scores for quality, speed,
prefill, latency, memory headroom, stability and context are weighted per workload; candidates failing a
workload's gates (measured quality for quality-weighted workloads, quality floor, minimum context, latency, stability) are not recommended; every reason cites its interpretation rule (docs/INTERPRETATION.md). The Results page shows the
comparison, context/memory scaling charts with the cliff marked, the reasons verbatim, and an export of the
benchmarked llama-server parameters.

## Reading the results

- **Interpretation panel** (top of each Results session, and the short lines on the Dashboard cards): the app's
  reading of the numbers, grouped by the sections of [docs/INTERPRETATION.md](docs/INTERPRETATION.md) — the practical
  context ceiling and quality first, then speed, memory, stability, the comparison and generation settings. Each
  item can be expanded to show its evidence (metric, value, provenance, context step).
- **Every reason cites a rule.** Recommendation reasons and insights start with a tag such as `[I-2.1]`; hover it
  for the rule text. The guide explains each rule and its thresholds. Recommendations made under older rules are
  shown as recorded and labelled "reinterpreted with rules …" when the current rules are applied for display.
- **Provenance badges**: MEASURED (observed on this machine in this session), DECLARED (read from the GGUF, driver,
  OS or runtime log), ESTIMATED (computed by the app's formulas, e.g. the quality prior or memory estimates),
  UNAVAILABLE (could not be obtained — shown as "—", never as 0).
- **Quality "Q ± u (n)"**: the score with a band and the number of graded items. With the current small suite the
  band is a heuristic, not a statistical guarantee; thorough mode (repeated samples) and the larger v2 suite
  tighten it. Two configs whose bands overlap are not meaningfully different in quality.
- **Provisional vs confirmed**: a recommendation is *provisional* when some candidate's quality is only an estimated
  prior (no quality run); run the quality suite to confirm it. Quality-weighted workloads may give no
  recommendation at all without measured quality, and the panel says why.
- **Action buttons** (next steps suggested by a rule): *Enable heavy mode* and *Run thorough quality* / *Search
  generation settings* open the Benchmark page preset accordingly; *Download* opens the Hugging Face page; *Use N K
  in export* pins that context in the Export menu. Other suggestions (e.g. try a smaller quant, re-run on an idle
  GPU) are shown as hints.

## Current limitations

The full, code-checked list is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Highlights:

- AMD GPU temperature, power and clocks are UNAVAILABLE (they need a native vendor SDK); NVIDIA telemetry via
  nvidia-smi is fixture-tested only
- Telemetry uses English performance-counter names; on localized Windows those fields show unavailable
- 1 s telemetry granularity: very short steps may have no samples (shown as unavailable, never 0)
- Calibrated on one GPU (RX 9070 XT 16 GB): Llama-3.1-8B, Qwen2.5-1.5B/14B full offload; Qwen3.8-27B and Gemma-4-26B-A4B partial offload. The CUDA runtime path is untested on NVIDIA hardware
- The quality suite is small (17 tests): a relative signal, not a leaderboard
- Windows only; packaged builds are unsigned (SmartScreen will warn)

## Layout

- `src/main` Electron main (IPC, session runner wiring) · `src/preload` contextBridge API · `src/renderer` React UI
- `src/core/system` scanner · `src/core/runtimes` llama.cpp adapter, Ollama/LM Studio detection · `src/core/models` GGUF reader
- `src/core/telemetry` typeperf / nvidia-smi samplers · `src/core/benchmark` candidates + session runner
- `src/core/quality` test suite + sandbox · `src/core/scoring` cliffs, scores, recommendation · `src/core/storage` sqlite
- `docs/` design, benchmark method, architecture, limitations · `tests/` vitest
