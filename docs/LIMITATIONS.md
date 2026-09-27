# Limitations — Local AI Optimizer (state at 2026-09-27)

This is the single consolidated list, and every item is checked against the code. Each item has one status:
- **DONE**: a former limitation that is now fixed and tested; kept one release for the record.
- **DONE-WITH-CAVEAT**: works and is tested, with the stated limit.
- **PARTIAL**: implemented in core, or only part of it; not complete end to end.
- **BLOCKED (env)**: cannot be verified on this machine (AMD RX 9070 XT, Windows 11, no NVIDIA/Ollama/LM Studio).

## Telemetry
- **DONE-WITH-CAVEAT: AMD temperature, power and clocks are UNAVAILABLE.** They need the ADLX/ADL native SDK. The app shows "—", never 0 (DESIGN §1.3).
- **DONE-WITH-CAVEAT: English PDH counter names only.** On localized Windows the typeperf fields report unavailable. The WMI class fallback (DESIGN §1.2) is not implemented (`telemetry/sampler.ts`).
- **DONE-WITH-CAVEAT: 1 s granularity.** typeperf's minimum is 1 s and its first row takes ~2 s. The sampler starts during load and waits up to 3 s (`firstSampleWaitMs`). Steps with no row at all report telemetry as unavailable, never as a made-up value.
- **DONE: util overshoot no longer drops rows (H7 root cause).** Heavy steps of 29–62 s had 0 samples with no sampler error: every row with util slightly > 100 % was being discarded as a glitch. Percentages in (100, 1000] are now clamped to 100. Only negative or > 1000 values mark a glitch row (e.g. 1.3e13 % with garbage RAM cells), which is still dropped whole. `stop()` reports "typeperf rows dropped: N misaligned, M impossible" in `RunDetail.samplerErrors`, so an empty step says why.
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
- **DONE-WITH-CAVEAT: loadModel waits a hardcoded 120 s for `/health`**, not configurable per phase. Only a served n_ctx **smaller** than requested is reported as `config_drift` (D17); a larger or missing n_ctx, and the effective GPU layers/device, are not compared. Every load (ladder and quality) gets the session signal, so a cancel during load kills the server at once.

## Benchmark method
- **DONE-WITH-CAVEAT: calibrated on one GPU (RX 9070 XT 16 GB, WDDM).** Full offload: Llama-3.1-8B and Qwen2.5-1.5B/14B (2026-09-27 calibration). Heavy partial offload: Qwen3.8-27B and Gemma-4-26B-A4B (session-run-H files of 2026-09-27; see Heavy mode below). Earlier aborted runs (`calibration-heavy-2026-09-27.md`, incl. a failed CPU-baseline attempt) are history, not calibration (D20).
  - Models: Llama-3.1-8B Q4_K_M and Qwen2.5-1.5B/14B Q4_K_M.
  - Thresholds: 0.60 decode drop, 256 MiB shared spill, 0.80 VRAM saturation.
  - Profile latency tolerances and speed targets are still heuristics elsewhere (`docs/BENCHMARK.md` §6–7).
- **DONE-WITH-CAVEAT: one real cliff has been observed and detected** (14B at 32K spill). No other failure shapes have been seen on real data: gradual thermal throttling, driver resets, or NVIDIA spill behaviour.
- **DONE-WITH-CAVEAT: recovery-aware cliffs.** A ≥ 40 % dip that the next rung recovers from is no longer a cliff. A drop on the last rung can't be confirmed, so it still counts.
- **DONE-WITH-CAVEAT: reps = 2 with a median.** There are no CV-based extra reps (DESIGN §3.4 not implemented).
- **DONE-WITH-CAVEAT: no `ignore_eos`.** Short generations make decode TPS noisier.
- **DONE-WITH-CAVEAT: warm/cold is only a flag (`warm`).** One size-matched warmup precedes the measured reps. Cold-start performance is not measured or scored.
- **DONE-WITH-CAVEAT: the warmup adds roughly 50 % request time to large steps** at the default 2 reps (1 warmup + 2 measured prompts, excluding load), because it uses the same full-context prompt. For example, 8B at 64K takes ≈ 32 s per request.
- **DONE-WITH-CAVEAT: ladder prompts are sized by characters (≈ 4 chars/token), not tokenized** (D08). The app's real fill is lower than the calibration script's tokenized 0.75·ctx: Run A measured **1167 prompt tokens at ctx 2048 (57 %)**. TTFT per rung is therefore for a ~57 %-full prompt on Llama-family tokenizers; the recorded `promptTokens` says the actual number.
- **DONE-WITH-CAVEAT: cross-session comparison is not done.** Each recommendation uses one session; mixed `versions` within a session only produce a warning.
- **PARTIAL: the ladder can include 128K.** It only runs where the declared ctx and the VRAM estimate allow; none of the calibrated models did (8B: memory-bound at 64K).

## Quality
- **DONE-WITH-CAVEAT: suite `qb-1.1.0` is small and strict.** It has 17 tests in 6 categories, with deterministic checkers and one rep at temperature 0. It is a relative signal, not a leaderboard; with few tests per category, one flaky answer moves Q by 5–33 points.
- **DONE-WITH-CAVEAT: without a quality run, Q is an ESTIMATED prior** from parameter count and quantization. It is labelled in the breakdown and reasons, and it can decide close calls (e.g. 8B vs 14B for Document Analysis).
- **DONE: the quality phase is guarded like a ladder step**: previous server unloaded, live RAM pre-check, abortable load, and the RAM-floor guard (with fail-safe) during the suite. A trip discards the suite.
- **DONE: stored quality is transactional.** One suite is saved in one transaction with its suite version and expected test count. Resume reuses only a complete suite of the current version; anything else re-runs.
- **DONE-WITH-CAVEAT: thinking models are scored with thinking OFF.** The quality suite passes `enable_thinking=false`, and the reason says so. A model's with-thinking quality is not measured; the ×4 token-boost path exists but is unused.
- **DONE-WITH-CAVEAT: model-written JS runs in a child-process sandbox** (`--permission`, memory cap, vm context, timeout). Network is blocked by the vm context, not by `--permission`.
  - The sandbox spawns `process.execPath` with `ELECTRON_RUN_AS_NODE=1` (`quality/sandbox.ts`), so it depends on Electron's **RunAsNode fuse** staying enabled (the electron-builder default).
  - If the fuses are hardened (`electronFuses.runAsNode: false`), the coding-quality tests (CD-01..03) fail as "execution failed" instead of running. Keep the fuse on, or ship a separate Node binary for the sandbox.
  - Memory cap ceiling. `--max-old-space-size` does not cover ArrayBuffer backing stores. Two checks close most of that gap:
    - the child checks its own `arrayBuffers`/RSS after the run (cap, and 1.5× cap + 48 MiB RSS);
    - the parent polls the child's working set every 250 ms and kills it above 1.5× cap + 48 MiB.
    An allocation burst shorter than one poll can briefly exceed the cap before the kill. A hard OS limit (Job Object) would need native code.
  - The VM context is not a security boundary by itself. The child's host code is strict (so V8 callsites can't hand a host function to a model-installed `Error.prepareStackTrace`), and `Error` is frozen in the context. Thrown values are never read through getters or `toString`. The `--permission` barrier (no fs / child_process / worker) is the backstop if a context escape is found.

## Scoring and recommendation
- **DONE-WITH-CAVEAT: absolute normalization.** Scores are comparable across sessions only while the stored `scoringVersion` (now `scoring-1.1.0`) and `rulesVersion` (`interp-1`) match. A version string does not prove an unchanged procedure; runs mixing runtime/benchmark/quality-suite versions get an [I-6.2] warning (D23).
- **DONE: every ranking decision is a rule** (`src/core/interpret`, rules from `docs/INTERPRETATION.md`); every reason cites its rule id, and the insight panel lists evidence with provenance.
- **DONE (D07): unknown inputs never pass a gate.** Unavailable component inputs still score a neutral 50 inside the total (flagged "unknown"), but: no measured quality ⇒ ineligible for the quality-weighted workloads (Coding, Large-scale Coding, Reasoning, Document Analysis, Maximum Quality) and provisional elsewhere; an unknown TTFT fails the latency gate unless latency is advisory; an estimated quality is capped at the lowest measured one. Unknown spill gives no penalty but an [I-1.2] note.
- **DONE-WITH-CAVEAT (D06): scoring vs recommended ctx.** Scores are taken at the largest passing rung ≤ the workload target (else the smallest passing rung). The recommended `-c` is always a measured passing rung: the largest ≤ maxContext with a measured TTFT within tolerance (advisory latency: the largest ≤ maxContext); if none is within tolerance, the smallest passing rung is reported and the TTFT gate makes the candidate ineligible. With a sparse ladder the scoring rung can be well below the target.
- **DONE-WITH-CAVEAT: quality uncertainty.** Q carries a 95 % band (Agresti–Coull per category, category-weighted; repeated samples count as items). With 17 tests the band is ±15–20 points. The band is a heuristic over this suite, not a statistical guarantee. For quality-weighted workloads, a candidate whose band lies entirely above the leader's wins regardless of speed (I-5.2).
- **DONE-WITH-CAVEAT: partial offload is ineligible** whenever the same model's full offload has a usable step (calibration: −83 % decode; a spilled full offload still beat ngl 30 by 4.7×).
- **DONE-WITH-CAVEAT: heavy-model mode** is calibrated on one GPU and two models (RX 9070 XT; runs of 2026-09-27, the Coding run still in progress):
  - Qwen3.8-27B (dense): 55/65 layers 12–13 t/s at 2K–8K; in the Coding run 54/65 layers decode 12.6 → 10.8 t/s from 2K to 16K with no spill.
  - Gemma-4-26B-A4B (MoE): 25 layers ≈ 46–50 t/s with a marginal spill (recorded, not aborted).
  - `-nkvo` was dominated at short ctx on both (slower than the KV-on-GPU rungs), hence it runs last and only when the KV costs ≥ 8 layers. Its long-context benefit is not yet observed.
  - The CPU baseline was never measured on these models: the > 50 %-of-RAM rule skips it.
  - The first real run showed a 27B CPU baseline driving RAM to 1.0 GiB free. The fixes: floor max(4 GiB, 8 %) in all modes, +1.5 GiB at ngl 0, the CPU baseline skipped when the file is > 50 % of RAM, the guard active from load with 250 ms polling for heavy configs, and heavy configs ordered most-offloaded first.
  - The workload decode gates keep such models out of Fast Assistant (30 t/s) only; Chat and Coding gate at 10 t/s, which the measured 12–13 t/s passes (D03). A user **preferred speed** (Min decode t/s) gates too, except for the explicit required-context fallback: if nothing meets it, the fastest config reaching the required context is returned, marked "meets required context; below preferred speed" [I-2.10].
- **PARTIAL: KV layout for hybrid / sliding-window archs.**
  - `kvLayout` handles per-layer KV heads, `full_attention_interval` (qwen35) and `sliding_window_pattern` (gemma4). gguf.ts reads those keys since a456679.
  - GPU KV = the sum over the **last ngl layers** (llama.cpp offloads the tail), so hybrid/SWA layouts split per layer, not by layer share.
  - Unknown archs without layout keys get the all-layers upper bound. If the head count or head dimensions are missing, the KV estimate is **0 and flagged unknown** — no bound at all (D02).
  - Recurrent state (e.g. DeltaNet layers, about 150 MB on Qwen3.8) is not modelled; the 1 GiB VRAM margin absorbed it there, which is not a guarantee for other hybrids. The estimate is heuristic; the live guards, not the estimate, prevent OOM.

- **DONE-WITH-CAVEAT (D05): a provisional recommendation is saved after each model's ladder + quality**, so a later cancel or abort still leaves one (marked provisional). Runs are saved per step; quality per model in one transaction.
- **DONE-WITH-CAVEAT (D09): quality runs on a passed rung** (min(target, ceiling) snapped down to a passing rung) of the best-offload config that passed one; a CPU or `-nkvo` config only when it is the only one that passed (said in the reasons). A model with no passing rung gets no quality run.
- **DONE-WITH-CAVEAT: generation-config search** (thinking on/off, effort, temperature) runs the suite once per config on thinking-capable models: baseline off/T=0, thinking at the lowest and middle effort at the model card's sampling (else T=1.0), 3 seeded samples when T > 0 in Thorough mode. The choice per workload is the best quality whose time-to-answer (TTFT + reasoning tokens / decode) is within tolerance. Reasoning tokens come from the runtime's thinking-region count, else a text-length split (estimated). Seeded sampling is not reproducible across builds/hardware.

## Hugging Face download
- **DONE-WITH-CAVEAT: HF search/list/resumable download** (`core/hub/hf.ts`, wired in fa491b0: `main/hub.ts`, `HubPage.tsx`), tested against local servers and mocked fetch/fs.
  - The token goes only to an exact origin allowlist (https `huggingface.co` / `*.huggingface.co` on the default port, or the configured base), checked on every page and every redirect hop.
  - Repo paths are refused per segment for `..`, NTFS ADS `:`, `<>"|?*\`, control chars, trailing dot/space and device names (CON, NUL, COMn, LPTn…).
  - A `.part` resumes only with a matching `<part>.json` sidecar (repo, revision, path, size, sha256) and a 206 whose Content-Range starts at the offset. Bytes past the expected size stop the download. Disk errors reject as `disk_error` instead of crashing main.
  - Links and junctions from destDir to the `.part` are refused; main also checks destDir by the realpath of its nearest existing ancestor. **Caveat: this is check-then-open** (NTFS has no O_NOFOLLOW from Node): a link swapped in between the check and the open is not caught. A fresh `.part` uses `wx`, so that case fails closed.
  - One download at a time (op lock claimed synchronously); Cancel deletes only that op's own `.part` after it has settled.
- **BLOCKED (env): no real huggingface.co call is in the test suite.** The API field names and the Link pagination follow the HF docs.

## Export
- **DONE-WITH-CAVEAT: export menu** (e31dc64): llama-server command, Ollama Modelfile, LM Studio settings, JSON and the provenance note, saved to a file.
- **BLOCKED (env): the Ollama Modelfile `num_gpu/num_thread/num_batch` and the LM Studio setting keys are unverified.** The llama-server export reproduces the benchmarked **inference parameters**, not the exact argv/executable (the host/port/log/metrics flags and the runtime path/version are not exported) (D16).

## Safety
- **DONE-WITH-CAVEAT: guard thresholds are heuristics:**
  - RAM floor max(4 GiB, 8 % RAM), in heavy mode too
  - 4 GiB candidate RAM reserve
  - 2 GiB shared-spill abort
  - 1 GiB VRAM margin, keep-over ≤ 1.15×
  - The RAM estimate is resident-only (non-GPU weights + CPU KV + 0.5 GiB), checked after the previous server is unloaded. The in-step floor credits reclaimable mmap pages.
- **DONE-WITH-CAVEAT: the RAM guard reads OS free RAM itself** on every poll (1 s, 250 ms for heavy configs), independent of typeperf, and fails safe: 3 polls with neither an OS reading nor a telemetry row → `guard_abort`. An allocation faster than one poll can still reach OOM, which is then recorded as `oom`.
- **DONE-WITH-CAVEAT: process cleanup** is kill → `taskkill /T /F`, a pid file and a stale-server kill at start, plus the NSIS uninstall killing the pid-file server. The stale-server and uninstall cleanup skip a process they cannot verify (exe path + start time within 30 s), so a crashed server from a pre-D11 build must be closed by hand. Unload keeps the handle and pid file until the exit is confirmed; a server that survives `taskkill /F` raises `ServerStuckError`, a hard stop (no further candidates, no cleanup that would drop its pid file). **No Job Object**: the OS does not tie llama-server to the app, so a hard kill of the Electron main process can still leave one llama-server until the next launch. Its session is shown as `interrupted` and is resumable.
- **DONE: one app instance.** `requestSingleInstanceLock` runs before any DB or pid-file cleanup; a second instance focuses the first and exits (1f44374). Within the instance, benchmark, smoke, runtime install and hub download each claim an op lock synchronously, and late IPC replies/events from a previous session or repo are ignored (stale-reply guards).

## Platform and packaging
- **DONE-WITH-CAVEAT: Windows 11 only** (PowerShell/WMI scanner, typeperf, taskkill). macOS and Linux are not supported.
- **DONE-WITH-CAVEAT: packaging** (a3dc31e): electron-builder portable exe and a per-user NSIS installer.
  - The binaries are **unsigned**, so Windows SmartScreen warns on first launch.
  - The packaged app installs the llama.cpp runtime to `userData/runtime/llama.cpp` on first run, via the System page.
  - Packaged data lives in `%APPDATA%\local-ai-optimizer`; dev uses `…-dev`. A DB newer than the build is refused. Uninstall keeps userData.
- **DONE-WITH-CAVEAT: the runtime download takes the newest `bNNNNN` prerelease that has a win-vulkan asset, and keeps it** (`release-tag.txt`); there is no update check. Upstream flag churn in a newer build can break the argv, and flags are not re-validated against `--help` at install.
