# Limitations — Local AI Optimizer (state at 2026-09-27)

This is the single consolidated list, and every item is checked against the code. Each item has one status:
- **DONE**: a former limitation that is now fixed and tested; kept one release for the record.
- **DONE-WITH-CAVEAT**: works and is tested, with the stated limit.
- **PARTIAL**: implemented in core, or only part of it; not complete end to end.
- **BLOCKED (env)**: cannot be verified on this machine (AMD RX 9070 XT, Windows 11, no NVIDIA/Ollama/LM Studio).

## Telemetry
- **DONE-WITH-CAVEAT: spill under other-process VRAM use (L1).** Adjusted spill is ungated: raw per-PID shared − host-pinned − benign baseline. Other-process use (the planning-time scan, not a per-rung reading) changes only the disclosed saturation evidence. Raw shared growth ≥ 1 GiB vs the previous rung is a spill (cal-longctx-2026-09-27). Consequence: transient placement or unlisted pinned host memory now counts as spill until the I-2.8 retry clears it.
- **DONE-WITH-CAVEAT: AMD temperature, power and clocks are UNAVAILABLE.** They need the ADLX/ADL native SDK. The app shows "—", never 0 (DESIGN §1.3).
- **DONE-WITH-CAVEAT: English PDH counter names only.** On localized Windows the typeperf fields report unavailable. The WMI class fallback (DESIGN §1.2) is not implemented (`telemetry/sampler.ts`).
- **DONE-WITH-CAVEAT: 1 s granularity.** typeperf's minimum is 1 s and its first row takes ~2 s. The sampler starts during load and waits up to 3 s (`firstSampleWaitMs`). Steps with no row at all report telemetry as unavailable, never as a made-up value.
- **DONE: util overshoot no longer drops rows (H7 root cause).** Heavy steps of 29–62 s had 0 samples with no sampler error: every row with util slightly > 100 % was being discarded as a glitch. Percentages in (100, 1000] are now clamped to 100. Only negative or > 1000 values mark a glitch row (e.g. 1.3e13 % with garbage RAM cells), which is still dropped whole. `stop()` reports "typeperf rows dropped: N misaligned, M impossible" in `RunDetail.samplerErrors`, so an empty step says why.
- **DONE-WITH-CAVEAT: GPU util is the max over the PID's 3D/Compute engine groups.** It is display-only; scoring and cliff rules don't use it.
- **DONE-WITH-CAVEAT: per-PID VRAM comes from WDDM counters.** One scoped observation on this GPU (CAL-14, 14B at 32K, raw per-PID shared): spill began at ≈83 % dedicated. Under contention (H-LONG) spill-like growth began at 67–73 %. Neither is a Windows constant; other drivers may differ.
- **DONE-WITH-CAVEAT: effective per-process VRAM budget.** The usable VRAM for one model is the per-process ceiling of the GPU + driver + backend, not the card total.
  - Here (llama.cpp Vulkan) it was ≈ 11.6–13.25 GiB of 15.92, about 73–83 %, so a 16 GB card behaved like ≈ 12–13 GB. Ollama's ROCm/HIP backend on an RX 6800 used the full 16 GB. HIP allocates device memory directly, while Vulkan on this driver keeps a per-process ceiling.
  - Ceilings are learned per adapter (PNP id) + driver + backend build, only from residency that persists after a fresh restart. They stay advisory until qualified (identity verified, load-log buffer comparable to the planned allocation). Only qualified, comparable ceilings prune, and never an allocation observed clean. Until one exists, 80 % is shown as an estimate. I-2.8 (placement) is heuristic until the held A/B confirms.
  - The app measures only the backend it runs: a ROCm/HIP backend (LM Studio or Ollama on AMD) may allocate differently from Vulkan, and its ceiling is not inferred from ours.
- **DONE-WITH-CAVEAT: process ownership is identity-verified, not OS-enforced.**
  - Owned `llama-server`/`typeperf` trees are bound to pid + creation time, re-read at kill. A timing window or a name is never proof.
  - A failure to enumerate or verify the tree is fatal (`ServerStuckError`) and blocks backend switching and new loads.
  - FOLLOW-UP: a Windows Job Object (kill-on-close) would remove the enumeration race, but needs native code (ARCHITECTURE §5).
- **Replay never upgrades evidence:** a partial or incoherent proof-origin record stays unverified/`reconstructed`. It is never repaired into proof.
- **OPEN (review-w4q): harness safety blockers. No new GPU stage runs until these are fixed and re-reviewed by Astra:**
  - **Q1** — cancellation can be lost between preprocessing (applyTemplate) and generation.
  - **Q2** — PID-chain teardown kills the parent first and uses `taskkill /T` only if the parent survives, so children can outlive it.
  - **Q3** — the 4 GiB watchdog does not cover probes/teardown (untimed synchronous tasklist/git; probes before the watchdog exists).
  - **Q5** — session dumps are not no-clobber/atomic.
  - **Q6** — a ZIP snapshot does not freeze the environment (junctioned node_modules/vendor are mutable).
  - **Q7** — A/B `largestBufferMiB` includes host buffers.
  - **Q8** — A/B health readiness is not bound to the owned server/model (/props not verified).
- **DONE-WITH-CAVEAT: generation-config evidence before 2026-09-28 is non-evaluable** for effort/thinking claims. It has no original prompt hash, so there is no row-bound render proof (I-8.0), and this includes session 5's Qwen off/low/medium. A replay can only label such rows *reconstructed*.
- **UNKNOWN ORIGIN: the Vulkan per-process ceiling.**
  - Repaired stage-2 A/B (8B f16 64K): dedicated stops at 11.60 GiB with 1.36 GiB raw shared from the first sample. This is independent of prompt fill (36K vs 49K), of a prior large load (B1b) and of `-ub 256` (B2).
  - It is not driver placement (EVIDENCE E-12 contradicted) and not a proven capacity limit.
  - Likely an undeclared host-visible or other buffer: the A/B lacked `-lv 4`, now added, so buffer declarations will be logged next time.
  - HIP stays **untested** on hardware.
- **FOLLOW-UP: AMD backend choice.** llama.cpp also ships HIP (ROCm) Windows builds for supported gfx targets, and a HIP backend may use more VRAM than Vulkan on the same card. It is not integrated yet; whether the HIP SDK supports the RX 9070 XT (gfx1201) is still to be checked.
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
- **DONE-WITH-CAVEAT: real cliffs observed and detected:** 14B at 32K (CAL-14 spill), Gemma-4 ngl18 at 16K (H-1021 decode −44 %), and the 8B f16 64K / q8_0 128K collapses under contention (H-LONG; caught by decode_drop, and now also by the raw-growth spill rule). Not seen on real data: gradual thermal throttling, driver resets, NVIDIA spill behaviour.
- **DONE-WITH-CAVEAT: recovery-aware cliffs.** A ≥ 40 % dip that the next rung recovers from is no longer a cliff. A drop on the last rung can't be confirmed, so it still counts.
- **DONE-WITH-CAVEAT: reps = 2 with a median.** There are no CV-based extra reps (DESIGN §3.4 not implemented).
- **DONE-WITH-CAVEAT: no `ignore_eos`.** Short generations make decode TPS noisier.
- **DONE-WITH-CAVEAT: warm/cold is only a flag (`warm`).** One size-matched warmup precedes the measured reps. Cold-start performance is not measured or scored.
- **DONE-WITH-CAVEAT: the warmup adds roughly 50 % request time to large steps** at the default 2 reps (1 warmup + 2 measured prompts, excluding load), because it uses the same full-context prompt. For example, CAL-S 8B at 64K: TTFT 32.4 s, ≈ 34.8 s per request.
- **DONE (D08, `ladder-2`): ladder prompts are tokenized to 0.75·ctx** with the loaded model's tokenizer (±3 %). Earlier app runs (`ladder-1`) were character-sized at ≈ 0.56·ctx (Run A: Llama 1168 tokens at 2K; H-LONG: 36,572 at 64K) and are not speed-comparable with ladder-2 rows (different `versions.prompts`, excluded by I-6.0). Every insight states the actual `promptTokens`.
- **DONE-WITH-CAVEAT: cross-session comparison is not done.** Each recommendation uses one session; mixed `versions` within a session only produce a warning.
- **PARTIAL: 128K.** It only runs where the declared ctx and the VRAM estimate allow. The 8B reached 128K only with q8_0 KV and degraded (H-LONG, 24.24 t/s under contention); no clean 128K has been observed, and no idle 128K run exists.

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
- **DONE-WITH-CAVEAT: absolute normalization.** Scores are comparable across sessions only while the stored `scoringVersion` (now `scoring-1.2.0`) and `rulesVersion` (now `interp-2`) match; stored interp-1 recommendations keep their reasons and are shown as reinterpreted. A version string does not prove an unchanged procedure; runs mixing runtime/benchmark/quality-suite versions get an [I-6.2] warning (D23).
- **DONE: every ranking decision is a rule** (`src/core/interpret`, rules from `docs/INTERPRETATION.md`); every reason cites its rule id, and the insight panel lists evidence with provenance.
- **DONE (D07): unknown inputs never pass a gate.** Unavailable component inputs still score a neutral 50 inside the total (flagged "unknown"), but: no measured quality ⇒ ineligible for the quality-weighted workloads (Coding, Large-scale Coding, Reasoning, Document Analysis, Maximum Quality) and provisional elsewhere; an unknown TTFT fails the latency gate unless latency is advisory; an estimated quality is capped at the lowest measured one. Unknown spill gives no penalty but an [I-1.2] note.
- **DONE-WITH-CAVEAT (D06): scoring vs recommended ctx.** Scores are taken at the largest passing rung ≤ the workload target (else the smallest passing rung). The recommended `-c` is always a measured passing rung: the largest ≤ maxContext with a measured TTFT within tolerance (advisory latency: the largest ≤ maxContext); if none is within tolerance, the smallest passing rung is reported and the TTFT gate makes the candidate ineligible. With a sparse ladder the scoring rung can be well below the target.
- **DONE-WITH-CAVEAT: quality uncertainty** (`uncertainty.ts`, unc-1). The band is a **heuristic**, not a validated 95 % interval: unique items or skill clusters are the unit, repeats never narrow it, correlated items and suite selection violate its assumptions. With 17 items the computed band is ±15–20 points (a single-run computation on the old suite, not repeated-run coverage), so most real comparisons are "insufficient evidence to distinguish" and quality is neutralized (I-5.2); only a paired difference that excludes 0 lets quality decide.
- **DONE: confirmed vs provisional** (I-1.2). Without a quality run no candidate is a confirmed winner (for every workload); the result carries `provisionalBest` with the estimated term named (rendered as provisional by the UI).
- **DONE-WITH-CAVEAT: thinking picks are provisional wherever latency has weight.** The time-to-answer of a thinking config combines the ladder's TTFT with the quality suite's reasoning time from another context, so it is ESTIMATED (G08); only workloads without a latency weight (e.g. Maximum Quality) can confirm a thinking config.
- **DONE: measured safety before ranking.** A measured RAM minimum below the recorded floor disqualifies a pick; unmeasured spill or an estimated VRAM peak leaves it provisional, never confirmed.
- **DONE-WITH-CAVEAT: generation comparisons need verified template kwargs** (I-8.0): a thinking config counts only when its kwargs changed the rendered prompt; the runtime's own acceptance is not reported by llama-server's `/apply-template`.
- **DONE-WITH-CAVEAT: partial offload is vetoed only by a same-model full offload that meets the same hard constraints** (I-7.6). Scoped evidence: CAL-S ngl 20 −83 % decode; CAL-14 spilled full offload beat ngl 30 by 4.7× at 32K.
- **DONE-WITH-CAVEAT: heavy-model mode** has scoped observations on one GPU and two models (RX 9070 XT, 2026-09-27; see `docs/calibration-ledger.md` for run validity):
  - Qwen3.8-27B (dense): 55/65 layers 12–13 t/s at 2K–8K (H-CONSOLE: console-only, reaped run, default mmap, quality lost); 54/65 layers 12.6 → 10.8 t/s from 2K to 16K (H-0953 ladder; its quality is INVALID).
  - Gemma-4-26B-A4B (MoE): 25 layers ≈ 46–50 t/s (H-0953 ladder). H-1021 is the re-run with valid baseline quality (15–16/17).
  - `-nkvo` was dominated at short ctx on both (slower than the KV-on-GPU rungs), hence it runs last and only when the KV costs ≥ 8 layers. Its long-context benefit is not yet observed.
  - The CPU baseline was never measured on these models: the > 50 %-of-RAM rule skips it.
  - The first real run showed a 27B CPU baseline driving RAM to 1.0 GiB free. The fixes: floor max(4 GiB, 8 %) in all modes, +1.5 GiB at ngl 0, the CPU baseline skipped when the file is > 50 % of RAM, the guard active from load with 250 ms polling for heavy configs, and heavy configs ordered most-offloaded first.
  - The workload decode gates keep such models out of Fast Assistant (30 t/s) only; Chat and Coding gate at 10 t/s, which the measured 12–13 t/s passes (D03). A user decode floor (Min decode t/s) is a hard constraint: it is never overridden — with a required context, configs that reach it but miss the floor are listed as unmet-constraint alternatives, not recommended (I-2.5).
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
