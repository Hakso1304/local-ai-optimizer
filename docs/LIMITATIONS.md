# Limitations — Local AI Optimizer (state at 2026-09-27)

This is the single consolidated list, and every item is checked against the code. Each item has one status:
- **DONE-WITH-CAVEAT**: works and is tested, with the stated limit.
- **PARTIAL**: implemented in core, or only part of it; not complete end to end.
- **BLOCKED (env)**: cannot be verified on this machine (AMD RX 9070 XT, Windows 11, no NVIDIA/Ollama/LM Studio).

## Telemetry
- **DONE-WITH-CAVEAT: AMD temperature, power and clocks are UNAVAILABLE.** They need the ADLX/ADL native SDK. The app shows "—", never 0 (DESIGN §1.3).
- **DONE-WITH-CAVEAT: English PDH counter names only.** On localized Windows the typeperf fields report unavailable. The WMI class fallback (DESIGN §1.2) is not implemented (`telemetry/sampler.ts`).
- **DONE-WITH-CAVEAT: 1 s granularity.** typeperf's minimum is 1 s and its first row takes ~2 s. The sampler starts during load and waits up to 3 s (`firstSampleWaitMs`). Steps with no row at all report telemetry as unavailable, never as a made-up value.
- **DONE-WITH-CAVEAT: PDH glitch rows are dropped.** Rows with an impossible percentage (1.3e13 % GPU util, and garbage RAM cells) are discarded whole, which can leave a step with fewer or zero samples.
- **DONE-WITH-CAVEAT: GPU util is the max over the PID's 3D/Compute engine groups.** It is display-only; scoring and cliff rules don't use it.
- **DONE-WITH-CAVEAT: per-PID VRAM comes from WDDM counters.** On this GPU the spill starts at ≈83 % dedicated, and other drivers may differ.
- **BLOCKED (env): NVIDIA telemetry is wired but fixture-tested only.** `withNvidia` merges nvidia-smi temp/power into samples when `probeNvidiaSmi()` works (acbd169). nvidia-smi here fails with "insufficient permissions" (stale driver, exit 4).

## Runtimes and models
- **DONE-WITH-CAVEAT: llama.cpp (llama-server, Vulkan build b11208) is the only benchmark runtime.**
- **DONE-WITH-CAVEAT: Ollama / LM Studio models** are listed in `models:list` (acbd169) and benchmarked through our llama-server, not through Ollama or LM Studio themselves. Runtime detection is HTTP-only.
- **BLOCKED (env): the Ollama / LM Studio on-disk layouts come from upstream code and docs.** Neither app is installed, so they have never been observed here.
- **BLOCKED (env): the CUDA path is wired but untested on real NVIDIA hardware.** `runtime:install` picks the CUDA build + cudart via `pickReleaseAsset` from the driver's CUDA major, and falls back to Vulkan if `--version` fails (acbd169).
- **PARTIAL: no ROCm / SYCL / CPU-only build selection.** A CPU-only machine gets ngl=0 candidates on the Vulkan build.
- **DONE-WITH-CAVEAT: GGUF parsing is header-only.** Split shards and mmproj files are listed as-is, without grouping (`models/gguf.ts`).
- **DONE-WITH-CAVEAT: loadModel waits a hardcoded 120 s for `/health`**, not configurable per phase. A served n_ctx ≠ requested is reported as `config_drift`.

## Benchmark method
- **DONE-WITH-CAVEAT: calibrated on one GPU (RX 9070 XT 16 GB, WDDM) and two model families.**
  - Models: Llama-3.1-8B Q4_K_M and Qwen2.5-1.5B/14B Q4_K_M.
  - Thresholds: 0.60 decode drop, 256 MiB shared spill, 0.80 VRAM saturation.
  - Profile latency tolerances and speed targets are still heuristics elsewhere (`docs/BENCHMARK.md` §6–7).
- **DONE-WITH-CAVEAT: one real cliff has been observed and detected** (14B at 32K spill). No other failure shapes have been seen on real data: gradual thermal throttling, driver resets, or NVIDIA spill behaviour.
- **DONE-WITH-CAVEAT: sticky degraded verdict.** A real ≥ 40 % dip that recovers at the next step still ends the practical ceiling. The defence is only the median of 2 reps.
- **DONE-WITH-CAVEAT: reps = 2 with a median.** There are no CV-based extra reps (DESIGN §3.4 not implemented).
- **DONE-WITH-CAVEAT: no `ignore_eos`.** Short generations make decode TPS noisier.
- **DONE-WITH-CAVEAT: warm/cold is only a flag (`warm`).** One size-matched warmup precedes the measured reps. Cold-start performance is not measured or scored.
- **DONE-WITH-CAVEAT: the warmup roughly doubles the time of large steps**, because it uses the same full-context prompt. For example, 8B at 64K takes ≈ 32 s per request.
- **DONE-WITH-CAVEAT: ladder prompts assume ≈ 4 chars/token** at 0.75·ctx fill. This was measured as ≈ 0.75·ctx on the calibration models, and other tokenizers may differ.
- **DONE-WITH-CAVEAT: cross-session comparison is not done.** Each recommendation uses one session; mixed `versions` within a session only produce a warning.
- **PARTIAL: the ladder can include 128K.** It only runs where the declared ctx and the VRAM estimate allow; none of the calibrated models did (8B: memory-bound at 64K).

## Quality
- **DONE-WITH-CAVEAT: suite `qb-1.1.0` is small and strict.** It has 17 tests in 6 categories, with deterministic checkers and one rep at temperature 0. It is a relative signal, not a leaderboard; with few tests per category, one flaky answer moves Q by 5–33 points.
- **DONE-WITH-CAVEAT: without a quality run, Q is an ESTIMATED prior** from parameter count and quantization. It is labelled in the breakdown and reasons, and it can decide close calls (e.g. 8B vs 14B for Document Analysis).
- **DONE-WITH-CAVEAT: thinking models are scored with thinking OFF.** The quality suite passes `enable_thinking=false`, and the reason says so. A model's with-thinking quality is not measured; the ×4 token-boost path exists but is unused.
- **DONE-WITH-CAVEAT: model-written JS runs in a child-process sandbox** (`--permission`, memory cap, vm context, timeout). Network is blocked by the vm context, not by `--permission`.
  - The sandbox spawns `process.execPath` with `ELECTRON_RUN_AS_NODE=1` (`quality/sandbox.ts`), so it depends on Electron's **RunAsNode fuse** staying enabled (the electron-builder default).
  - If the fuses are hardened (`electronFuses.runAsNode: false`), the coding-quality tests (CD-01..03) fail as "request failed" instead of running. Keep the fuse on, or ship a separate Node binary for the sandbox.

## Scoring and recommendation
- **DONE-WITH-CAVEAT: absolute normalization.** Scores are comparable across sessions only while `scoring-1.0.0` constants are unchanged, and the version is stored.
- **DONE-WITH-CAVEAT: unavailable inputs score a neutral 50.** They are flagged "unknown", never counted as 0 or as a pass.
- **DONE-WITH-CAVEAT: scoring vs recommended ctx.** Scores are taken at the workload's target ctx, while the recommended `-c` is the largest passing step within the TTFT tolerance. The reasons show both.
- **DONE-WITH-CAVEAT: partial offload is ineligible** whenever the same model's full offload has a usable step (calibration: −83 % decode; a spilled full offload still beat ngl 30 by 4.7×).
- **PARTIAL: heavy-model mode** is calibrated on **one real dense model** (Qwen3.8-27B at 55/65 layers: 12–13 t/s decode, no spill). The **MoE** model (Gemma-4-26B-A4B) and the `-nkvo` / CPU-baseline paths are not yet measured.
  - The first real run showed a 27B CPU baseline driving RAM to 1.0 GiB free. The fixes: floor max(4 GiB, 8 %) in all modes, +1.5 GiB at ngl 0, the CPU baseline skipped when the file is > 50 % of RAM, the guard active from load with 250 ms polling for heavy configs, and heavy configs ordered most-offloaded first.
  - The `minDecodeTps` gate keeps such models out of Fast Assistant, Chat and Coding.
- **PARTIAL: KV layout for hybrid / sliding-window archs.**
  - `kvLayout` handles per-layer KV heads, `full_attention_interval` (qwen35) and `sliding_window_pattern` (gemma4). gguf.ts reads those keys since a456679.
  - Unknown archs without layout keys get the all-layers **upper bound**: conservative, never OOM from an under-estimate.
  - Recurrent state (e.g. DeltaNet layers, about 150 MB) is not modelled; the 1 GiB VRAM margin covers it.

## Hugging Face download
- **PARTIAL: HF search/list/resumable download works in core** (`core/hub/hf.ts`, tested against local servers). The Electron parts (`main/hub.ts`: safeStorage token, token-page window, IPC; `HubPage.tsx`) are **untested live until #2 wires them**.
- **BLOCKED (env): no real huggingface.co call is in the test suite.** The API field names and the Link pagination follow the HF docs.

## Export
- **DONE-WITH-CAVEAT: export menu** (e31dc64): llama-server command, Ollama Modelfile, LM Studio settings, JSON and the provenance note, saved to a file.
- **BLOCKED (env): the Ollama Modelfile `num_gpu/num_thread/num_batch` and the LM Studio setting keys are unverified.** The llama-server export is exact, because it mirrors the measured launch.

## Safety
- **DONE-WITH-CAVEAT: guard thresholds are heuristics:**
  - RAM floor max(4 GiB, 8 % RAM), in heavy mode too
  - 4 GiB candidate RAM reserve
  - 2 GiB shared-spill abort
  - 1 GiB VRAM margin, keep-over ≤ 1.15×
  - The RAM estimate is resident-only (non-GPU weights + CPU KV + 0.5 GiB), checked after the previous server is unloaded. The in-step floor credits reclaimable mmap pages.
- **DONE-WITH-CAVEAT: the guard reacts at 1 s telemetry granularity.** A very fast allocation can still reach OOM, which is then recorded as `oom`.
- **DONE-WITH-CAVEAT: process cleanup** is kill → `taskkill /T /F`, a pid file and a stale-server kill at start, plus the NSIS uninstall killing the pid-file server. A hard kill of the Electron main process can still leave one llama-server until the next launch. Its session is shown as `interrupted` and is resumable.
- **DONE-WITH-CAVEAT: one benchmark at a time per app instance.** Two app instances are not prevented from running at once.

## Platform and packaging
- **DONE-WITH-CAVEAT: Windows 11 only** (PowerShell/WMI scanner, typeperf, taskkill). macOS and Linux are not supported.
- **DONE-WITH-CAVEAT: packaging** (a3dc31e): electron-builder portable exe and a per-user NSIS installer.
  - The binaries are **unsigned**, so Windows SmartScreen warns on first launch.
  - The packaged app installs the llama.cpp runtime to `userData/runtime/llama.cpp` on first run, via the System page.
  - Packaged data lives in `%APPDATA%\local-ai-optimizer`; dev uses `…-dev`. A DB newer than the build is refused. Uninstall keeps userData.
- **DONE-WITH-CAVEAT: the runtime download takes the newest `bNNNNN` prerelease that has a win-vulkan asset, and keeps it** (`release-tag.txt`); there is no update check. Upstream flag churn in a newer build can break the argv, and flags are not re-validated against `--help` at install.
