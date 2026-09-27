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
- **BLOCKED (env): NVIDIA telemetry (`telemetry/nvidia.ts`) is fixture-tested only**, and it is not wired into the runner. nvidia-smi here fails with "insufficient permissions" (stale driver, exit 4).

## Runtimes and models
- **DONE-WITH-CAVEAT: llama.cpp (llama-server, Vulkan build b11208) is the only benchmark runtime.**
- **PARTIAL: Ollama / LM Studio.**
  - Done: runtime detection over HTTP (`runtimes/others.ts`), the Ollama manifest→blob enumeration (`runtimes/ollama/models.ts`) and the LM Studio default dirs.
  - Not yet listed in the Models page (`models:list` scans GGUF folders only).
  - The Ollama blobs are benchmarked through our llama-server, not through Ollama itself.
- **BLOCKED (env): the Ollama / LM Studio on-disk layouts come from upstream code and docs.** Neither app is installed, so they have never been observed here.
- **PARTIAL + BLOCKED (env): the CUDA path.** `pickReleaseAsset` (`runtimes/llamacpp/assets.ts`) selects the CUDA build plus cudart. `ensureRuntime` still downloads Vulkan only (`pickVulkanAsset`), so this is untested on real NVIDIA hardware.
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
- **PARTIAL: thinking-model token boost (×4) is off**, because `ModelMeta` has no `supportsThinking`.
- **DONE-WITH-CAVEAT: model-written JS runs in a child-process sandbox** (`--permission`, memory cap, vm context, timeout). Network is blocked by the vm context, not by `--permission`.

## Scoring and recommendation
- **DONE-WITH-CAVEAT: absolute normalization.** Scores are comparable across sessions only while `scoring-1.0.0` constants are unchanged, and the version is stored.
- **DONE-WITH-CAVEAT: unavailable inputs score a neutral 50.** They are flagged "unknown", never counted as 0 or as a pass.
- **DONE-WITH-CAVEAT: scoring vs recommended ctx.** Scores are taken at the workload's target ctx, while the recommended `-c` is the largest passing step within the TTFT tolerance. The reasons show both.
- **DONE-WITH-CAVEAT: partial offload is ineligible** whenever the same model's full offload has a usable step (calibration: −83 % decode; a spilled full offload still beat ngl 30 by 4.7×).
- **PARTIAL: heavy-model mode** (partial offload for models that don't fit) is opt-in. It is tested with a synthetic 27B only; no real >16 GB model has been benchmarked. `-nkvo` is verified in `--help` only. A decode gate (`minDecodeTps`) keeps such models out of Fast Assistant, Chat and Coding.

## Export
- **PARTIAL: the core generators exist** (`export/config.ts`: llama-server, Ollama Modelfile, LM Studio, JSON, provenance note). The Results page still uses its own simpler llama-server text.
- **BLOCKED (env): the Ollama Modelfile `num_gpu/num_thread/num_batch` and the LM Studio setting keys are unverified.** The llama-server export is exact, because it mirrors the measured launch.

## Safety
- **DONE-WITH-CAVEAT: guard thresholds are heuristics:**
  - RAM floor max(2 GiB, 8 % RAM)
  - 4 GiB candidate RAM reserve
  - 2 GiB shared-spill abort
  - 1 GiB VRAM margin, keep-over ≤ 1.15×
  - The RAM estimate counts the whole mmap'd file, which makes it conservative: it may skip configs that would page fine.
- **DONE-WITH-CAVEAT: the guard reacts at 1 s telemetry granularity.** A very fast allocation can still reach OOM, which is then recorded as `oom`.
- **DONE-WITH-CAVEAT: process cleanup** is kill → `taskkill /T /F`, a pid file and a stale-server kill at start. A hard kill of the Electron main process can still leave one llama-server until the next launch.
- **DONE-WITH-CAVEAT: one benchmark at a time per app instance.** Two app instances are not prevented from running at once.

## Platform and packaging
- **DONE-WITH-CAVEAT: Windows 11 only** (PowerShell/WMI scanner, typeperf, taskkill). macOS and Linux are not supported.
- **PARTIAL: packaging is not verified yet.** `electron-builder` is a devDependency, but `vendor/` and `models/` are located via `app.getAppPath()`, which is correct for dev/preview only. Update this item after #2's packaging task.
- **DONE-WITH-CAVEAT: the runtime download takes the newest `bNNNNN` prerelease that has a win-vulkan asset, and keeps it** (`release-tag.txt`); there is no update check. Upstream flag churn in a newer build can break the argv, and flags are not re-validated against `--help` at install.
