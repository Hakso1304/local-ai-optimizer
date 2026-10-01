# Changelog

## Unreleased
- Models page: "Browse…" picks the model store (default download target and first scanned folder; created if missing), "Reset to default" returns to the app's folder. Files are not moved; the old folder stays reachable via "Add folder…".
- Download page: a Use case dropdown (same use cases as the Benchmark page, defaulting to its workload) drives the recommendation's context length, speed gate and coding boost.
- Download page recommendations are grouped by model series (Qwen, Llama, Gemma, …) with a hover profile per series and per row (what the size, file, speed and rank are based on); repo creation date acts as a release proxy so newer models outrank older ones of the same size (1.0 → 0.5 over 6–30 months).
- Download page recommendations rebuilt: budgets follow the benchmark planner (VRAM minus in-use and margin, RAM minus reserve, an integrated GPU's shared pool as one budget), sizes come from each repo's real GGUF files (best quant that fits, shards summed, ternary flagged) with the workload's context KV cache included, a bandwidth-based decode estimate marks slow picks, duplicate quantizer repos fold into one row, official sources rank first and abliterated/uncensored repos last.
- Ternary models (PrismML Bonsai, PQ2_0 / PTQ1_0): the Models page flags files whose tensor types mainline llama.cpp rejects, and the System page can install the PrismML llama.cpp fork (Vulkan build) as an opt-in third backend. Such models run and benchmark on that backend only; verified on an Intel Arc iGPU (Bonsai 2 27B fully offloaded, ~4 tok/s).
- Run a model from the app: "Run this model" (Results) and "Run…" (Models page, prefilled from the recommendation or defaults) start llama-server and open its chat UI; the running-model line shows the API base URL and model id, with a copy-paste setup for Claude Code, Cline, Continue and other agent tools.
- Download page: "Add folder…" registers a model folder on another drive as a scan and download target (C: is often the small drive).
- Classify the bare "Intel(R) Arc(TM) Graphics" adapter (Core Ultra iGPU) as integrated, so its shared-memory use is no longer aborted as VRAM spill.
- Benchmark integrated-only laptops on Vulkan instead of forcing CPU-only: account for GPU allocations in shared system RAM, retain the live RAM floor, and avoid interpreting ordinary shared-GPU usage as dedicated-VRAM spill.
- Create the writable `%APPDATA%\local-ai-optimizer\models` download folder during NSIS installation and on app startup (including portable builds); preserve downloads on uninstall.

- Use CIM creation time for inspection and a tolerant creation-time match for kill, restoring session completion after `0a20d21`..`bbe12dc` (`127d96f`).
- Keep standalone quality MEASURED and disclose an unverified thinking state (`127d96f`).
- Disclose paired quality and answer-time ratio when a thinking option exceeds tolerance; never accept estimated answer time as within tolerance, leaving that choice provisional (`340f3ee`).
- Accept an optional AbortSignal in the typeperf VRAM probe (`7e122fe`); pass the harness signal in session and A/B preflights (`f0fda83`, `3a52076`).
- Supervise session preflight and reap owned process chains (`0d74210`).
- Supervise A/B case teardown and separate host buffer evidence (`147a5a8`).
- Verify measurement dependencies before and after launch (`9e34367`).
- Bind exported session evidence to the runner log identity (`b8328e2`).
- Bound process reaping checks and hash probes (`3ad0641`).
- Reject ambiguous port ownership and abort when the server is lost (`3a52076`).
- Align selected HIP device metadata with ROCm0 (`f0fda83`).
- After the server exits, adopt and reap a descendant whose parent is a captured process identity, leave a child of the exited server PID born after the exit, and fail closed on any other unverified descendant (`251e497`, `d6db229`).
- Kill owned processes on app quit with the same µs-tolerant creation-time match as the verified kill (`54e7a71`).
- Reserve and checkpoint session dumps atomically (`b647ed9`).

## 0.1.0 — 2026-09-27 / 2026-09-28 (first release, Windows 11)

Local AI Optimizer benchmarks local LLM configurations on your own machine with llama.cpp. From measured data it
recommends which model and settings fit a workload, and every reason cites a rule of the interpretation guide.

### System scan & telemetry
- System page: OS, CPU, RAM, GPUs (vendor, dedicated VRAM, driver), disks and installed runtimes. Every value is marked
  measured, declared, estimated or unavailable, never a silent 0.
- Live telemetry per benchmark step via `typeperf` (per-process dedicated and shared GPU memory, RAM, CPU, GPU load).
  The sampler restarts itself when Windows changes the counter set mid-run, and reports dropped rows instead of hiding them.
- NVIDIA temperature and power via `nvidia-smi` when available. VRAM already used by other apps is measured before
  planning, and a busy GPU is flagged.
- No console windows: every helper process (typeperf, PowerShell, llama-server, tasklist) runs hidden.

### Runtimes & backends
- One-click install of the official llama.cpp Windows build (Vulkan; CUDA + cudart on NVIDIA, with a Vulkan fallback).
  Downloads are size- and sha256-checked.
- **ROCm/HIP opt-in on AMD**: the System page offers the llama.cpp ROCm build of the same release, with the HIP runtime
  bundled (RDNA1–RDNA4, RX 9070 XT included). With both installed, the benchmark compares the two backends as
  separate candidates, and the recommendation and export name the backend.
- Managed-memory oversubscription (`GGML_CUDA_ENABLE_UNIFIED_MEMORY`) is never enabled, so "fits in VRAM" stays measurable.
- Ollama and LM Studio are detected and their GGUF files are listed as model sources, but benchmarking always goes
  through the app's own llama-server.

### Models & Hugging Face download
- GGUF header parser: architecture, layers, heads, per-layer KV layout (hybrid, sliding window), MoE experts, chat
  template capabilities (thinking, reasoning effort). Partially downloaded files are flagged and never planned.
- Hugging Face page: search, file list, resumable download with a disk-space check and link-safe destinations, and
  optional login (token stored encrypted).
- Link a local model to its repo to read the model card's recommended sampling. Smaller **sibling quantizations**
  that would fit more layers on the GPU are suggested (estimated) with *Download & include*.

### Benchmark engine & safety guards
- A benchmark session plans candidate configs per model (GPU layers, KV type, flash attention, threads) and runs them
  step by step. Sessions can be paused, cancelled, resumed, and failed steps retried or re-run individually.
- Safety: RAM floor with an independent guard, shared-memory spill abort, a pre-load RAM check, stuck-server
  detection, one app instance, and a pid-verified cleanup of leftover servers (also by the uninstaller).
- **Heavy-model mode**: partial GPU offload (layers on the CPU, KV cache in RAM, no mmap) for models that don't fit,
  clearly flagged as degraded.
- Required context and minimum decode speed can be set per session. Every attempt is stored and superseded rows are
  kept as history.

### Context ladder & spill detection
- Each config climbs a context ladder (2K … 128K) with a prompt that fills about 75 % of the context, so speed and
  memory are measured at realistic fill.
- The practical context ceiling is where the model first spills out of dedicated VRAM, runs out of memory or slows
  sharply ("cliff"), not the context length the model declares.
- Spill is measured per process with host-pinned buffers subtracted. Per-process VRAM ceilings are learned per GPU +
  driver + backend build, and they stay advisory until verified.

### Quality suite v1 → v2 + uncertainty
- v1 (qb-1.1.0, 17 deterministic items) is now the *quick* option. The default *thorough* suite v2 (qb-2.0.0, 60 items
  from seeded generators) runs in a sandboxed checker.
- Quality is shown as a score with an uncertainty band and item count. Two configs whose bands overlap are treated as
  equal in quality, and speed and memory decide instead.
- Resume re-uses stored quality only for the same suite and seed. Mixed-build and truncated rows are quarantined, and
  infrastructure failures never count as wrong answers.

### Generation-config search
- For thinking-capable models the quality suite also runs with thinking on and each reasoning-effort level, using the
  model card's sampling when known.
- Each setting counts only when its application is proven: the rendered template shows the setting took effect, the
  sampling the runtime reports matches, and the "thinking off" baseline is verified.
- The chosen generation setting is exported with the config.

### Interpretation guide & rule engine (interp-1 → interp-2)
- `docs/INTERPRETATION.md` is the normative rulebook. The rule engine turns measurements into verdicts, and every
  recommendation reason and insight cites a rule `[I-x.y]`.
- interp-2 adds a full decision trace (eligibility, comparisons with the totals actually compared, tie-breaks,
  thresholds used), safety before ranking, and provisional picks when quality isn't measured. It also covers backend
  comparison and the learned per-process VRAM budget.
- Older recommendations are shown as recorded and labelled "reinterpreted with rules …" when viewed under current rules.

### UI
- **Dashboard**: recommended configuration per workload with the practical context ceiling, quality ± band, decode
  band and alerts; a Large-scale coding card; a provisional pick is never shown as the recommendation.
- **Results**:
  - comparison table, ladder and memory charts, per-step telemetry, Pareto chart with an SLO filter, view-as-workload;
  - Interpretation panel (criticals first, evidence per item, action buttons);
  - "How this was decided" trace, and an **Evidence** section listing every run row, exportable as JSON.
- **Export**: llama-server command (exactly the benchmarked inference flags, the right backend's executable), Ollama
  Modelfile, LM Studio settings (unverified keys), JSON.

### Packaging & builds
- electron-builder portable `.exe` and per-user NSIS installer. `npm run package` also writes `dist/BUILD-INFO.txt`
  (commit, rules version, test results, HIP status).
- Builds made on 2026-09-27/28: 2957026, 3bf6a09, 90738ce, and the nightly **3ec8278** ("nightly 2026-09-28
  pre-verdict").

### Reviews & calibration evidence
- An independent reviewer audited the core, security, documentation, tests and conformance with the interpretation
  guide over many passes (`docs/review-*.md`); every finding was fixed with a regression test or recorded as open.
- Calibration on an RX 9070 XT (8B–27B models, contexts to 128K, heavy mode, long-context variants, overnight stages).
  `docs/calibration-ledger.md` and `docs/EVIDENCE.md` map every cited number to a run, and say which runs are valid.

### Known limitations (see docs/LIMITATIONS.md)
- **Unsigned binaries**: Windows SmartScreen warns on first launch. Windows 11 only.
- **NVIDIA/CUDA and ROCm/HIP paths are untested on real hardware**: CUDA is fixture-tested only, and HIP is untested
  until its first calibration run (if the driver doesn't expose the GPU to HIP, the session runs on Vulkan).
- **Heuristic bands and thresholds**: the quality band is a heuristic, not a validated 95 % interval, and with small
  suites most comparisons are "insufficient evidence". Guard thresholds and latency/speed targets are heuristics too.
- **Historical rows are not validated**: early 2026-09-27 runs predate several fixes (prompt sizing, spill adjustment,
  thinking handling). Their speed timings are historical observations, and some quality results are invalid; see the ledger.
- The cause of the per-process VRAM ceiling on Vulkan is not yet established (the A/B contradicted the placement explanation).
- Ollama Modelfile GPU/thread/batch parameters and LM Studio keys are unverified. There is no runtime update check.
