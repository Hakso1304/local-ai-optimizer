# ACCEPTANCE — Local AI Optimizer MVP

Status: draft v1 (2026-09-27). Owner: Review/Validation. Each line has an ID so a test or a manual
step can point at it. **Auto** means a vitest test under `tests/` covers it. **Manual** means section 4 covers it.

Provenance labels used below:
`MEASURED` means observed on this machine in this session. `ESTIMATED` means computed from measured data or a heuristic, and the formula is shown.
`DECLARED` means taken from metadata (GGUF header, vendor spec, registry). `UNAVAILABLE` means we could not get it, and the reason is shown.
> Code today uses `Sourced.status = available|unavailable|unsupported` (src/shared/types.ts). The
> mapping to the four labels must be explicit (see R1 in the risks list at the end).

## 1. MVP acceptance criteria

### Launch & hardware
- **A1** `npm run dev` opens the main window in under 10 s with no uncaught errors in the main or renderer console. (Manual)
- **A2** The hardware panel shows the OS, CPU model, core and thread counts, total and available RAM, every *present* GPU with its vendor and VRAM, and the disks. Each field shows a provenance label and a `source`. (Auto: scanner fixture. Manual)
- **A3** A GPU whose VRAM is read from `qwMemorySize` shows 16 GB for the RX 9070 XT. It never shows the 4 GB that `AdapterRAM` reports (uint32 cap). (Auto)
- **A4** Stale registry adapters that have no present device, such as the "RTX 3080" in `scan-rx9070.json`, are not listed. (Auto)
- **A5** When a scan section fails, only that field shows `UNAVAILABLE` with an error. The other fields still render. (Auto)

### Runtimes & models
- **A6** At least one runtime is detected (the bundled llama.cpp Vulkan build) and shows its version and path. Runtimes that are missing show `UNAVAILABLE`, not an error dialog. (Auto + Manual)
- **A7** At least one local GGUF model is listed with its size on disk (MEASURED) and its declared context, quant and parameter count (DECLARED, from the GGUF header). (Auto: header parse fixture. Manual)
- **A8** A model that cannot be parsed is listed with an `UNAVAILABLE` reason. It does not crash the listing. (Auto)

### Workload & benchmark
- **A9** The user can select exactly one of the 7 workload profiles. The selected profile's weights are visible. (Manual)
- **A10** "Run benchmark" starts a real llama-server process and sends real prompts. There are no fake or sleep-based paths in production code. (Auto: a grep test fails if `Math.random`/mock imports appear under `src/core/benchmark` outside `*.test.ts`. Manual)
- **A11** Each run reports these metrics: load time, TTFT, prefill TPS, decode TPS, total time, peak RAM, peak VRAM, average and peak CPU %, average and peak GPU %, and status ∈ {ok, failed, timeout, cancelled, oom, device_lost, crashed}. A metric that could not be observed is `UNAVAILABLE`, never 0. (Auto)
- **A12** Prefill and decode TPS come from llama-server `timings`, which is MEASURED. If `timings` is absent, TPS is derived from wall clock and token count and labeled ESTIMATED. (Auto)
- **A13** The context sweep runs 2K, 4K, 8K, 16K, 32K, and 64K where the model permits. Sizes above the model's declared context are shown as skipped with a reason. They are not run. (Auto)
- **A14** A resource-utilization timeline (RAM, VRAM, CPU, GPU over time) is shown for each run, drawn from the samples. (Manual)
- **A15** Cliff and spill detection flags the first context size where decode TPS drops by ≥40% from the previous step, or where VRAM stays at its limit while RAM rises (spill). The flag names the metric and the threshold. (Auto)

### Recommendation
- **A16** The recommendation lists the model, quant, context, GPU layers, and threads, and includes a score breakdown: each weighted term shows its raw value, normalized value, weight, and contribution. The contributions sum to the total. (Auto)
- **A17** Only runs with status=ok feed the score. Failed configs appear in an "excluded" list with the reason. (Auto)
- **A18** When no run is ok, the app says "No recommendation: no successful runs" and does not pick a winner. (Auto)
- **A19** The recommendation is deterministic: the same stored results give byte-identical output. (Auto)

### Persistence
- **A20** After you quit and relaunch, previous sessions, runs, samples, and recommendations are listed and viewable. (Auto: storage round-trip with a temp SQLite file. Manual)
- **A21** If a run is interrupted by an app kill, the next launch shows it as `crashed/incomplete`. It is not shown as running. (Auto + Manual)

### Integrity & safety
- **A22** No fabricated data. Every number displayed traces to a sample row, a runtime response, a scan field, or a documented formula. (Review + A10 grep)
- **A23** Cancel stops the active run in under 3 s, kills the llama-server process tree, and records `cancelled` with its partial samples. (Auto with a fake process. Manual)
- **A24** Timeouts apply at each stage: load, first token, and total. A timeout records `timeout`, kills the process, and moves on to the next config. (Auto)
- **A25** Memory safeguard: before each load, estimated need = model file size + KV(ctx) + margin. If that exceeds free VRAM + free RAM − reserve (default 4 GB), the config is skipped as `skipped_memory`. During a run, if system available RAM falls below the reserve, the run is aborted as `oom`. (Auto)
- **A26** Process cleanup: after any outcome (ok, fail, timeout, cancel, app quit), no `llama-server.exe` spawned by the app is still alive, and its port is free. (Auto with a fake process. Manual)

## 2. Core-logic test plan & fixtures

All fixtures go under `tests/fixtures/`. They are plain JSON, loaded with `JSON.parse(readFileSync(...))`.
Timestamps are ms offsets from `t0` so the fixtures are deterministic. Tests never spawn real inference.

### 2.1 Machine profiles (`machine/*.json`, shape = `SystemProfile`)
| File | Content | Tests |
|---|---|---|
| `scan-rx9070.json` (exists, raw scan) | RX 9070 XT + iGPU + stale RTX 3080 reg key | A2–A4 |
| `machine/rx9070.json` | parsed profile: 31 GB RAM, 1 dGPU 16 GB `available`, iGPU flagged | memory guard, candidates |
| `machine/cpu-only.json` | `gpus.value=[]` | candidates have `gpuLayers=0` only |
| `machine/vram-unavailable.json` | GPU present, `dedicatedVramBytes.status='unavailable'` | guard must not assume 0 or ∞ (see X14) |
| `machine/low-ram.json` | 8 GB RAM, 2 GB available | most configs `skipped_memory` |
| `machine/scan-partial.json` (raw) | `cpu.ok=false`, `disks` missing | A5 |

### 2.2 Telemetry sample series (`telemetry/*.json`)
Shape: `{ "t0": 0, "intervalMs": 250, "samples": [{ "t": 0, "ramUsedBytes": n, "vramUsedBytes": n|null, "cpuPct": n|null, "gpuPct": n|null }] }`
- `steady.json` has 40 samples with flat values. Expected peak and average are given in a sibling `expected` block.
- `gaps.json` includes `null` values and one duplicated `t`. Averages must skip the nulls and not treat them as 0, and the duplicate must be kept or dropped deterministically.
- `spill.json`: VRAM rises to 15.8 GB of 16 GB and plateaus, then RAM climbs by 6 GB while GPU % falls from 95 to 40 and CPU % rises from 10 to 70. This must flag `memory_spill`.
- `partial-offload-spike.json`: CPU is at 100% for 2 samples during load and then 20%. The peak must include the spike, but it must not trigger spill or throttling (see X10).
- `empty.json` has `samples: []`. Every derived metric must be `UNAVAILABLE`, with no NaN and no throw.

### 2.3 Benchmark results (`bench/*.json`)
One run has this shape: `{ configId, model, quant, ctx, gpuLayers, threads, status, loadMs, ttftMs, prefillTps, decodeTps, totalMs, peakRamBytes, peakVramBytes, cpuAvgPct, gpuAvgPct, error? }`
- `sweep-cliff-16k-32k.json`: one model with ctx 2K/4K/8K/16K/32K/64K. decodeTps is 92, 90, 87, 82, **31**, 18. VRAM peaks at 9.1, 9.6, 10.5, 12.4, **15.9**, and 15.9 GB, while RAM jumps at 32K. Expected: the cliff is at 32K with `prev=16K`, the practical max context is 16K, and 64K is also marked degraded.
- `sweep-smooth.json` degrades gradually by about 8% per step. Expected: no cliff. This guards against false positives.
- `sweep-noisy.json` has a single −35% dip at 8K that recovers at 16K. Expected: no cliff at a 40% threshold, and a test that pins the chosen rule (see X8).
- `session-multi.json` has 4 configs across 2 models × 2 quants, all ok, and contains a known expected ranking for each of the 7 profiles.
- `session-single.json` contains one ok config (see X1).
- `session-all-failed.json` has 3 configs: oom, timeout, crashed (A18).
- `session-tie.json` has 2 configs with identical metrics and different `configId` values (X3).
- `session-zero-tps.json` has one ok run with `decodeTps: 0` and `ttftMs: 0` (X4).

### 2.4 Runtime failure cases (`runtime/*.json`, used by a fake `InferenceBackend` / fake child process)
Shape: `{ "script": [{ "atMs": n, "event": "stdout"|"stderr"|"exit"|"http", "data": ... }], "expectStatus": "..." }`
| File | Scripted behavior | Expected status / cleanup |
|---|---|---|
| `oom-load.json` | stderr `failed to allocate ... ErrorOutOfDeviceMemory`, exit 1 during load | `oom`, no retry at same config, next config runs |
| `device-lost.json` | mid-decode stderr `vk::DeviceLostError` / `VK_ERROR_DEVICE_LOST`, exit | `device_lost`, **abort remaining GPU configs** in session, surface warning |
| `timeout-load.json` | `/health` returns 503 forever | `timeout` at load limit, process killed |
| `timeout-first-token.json` | health ok, completion stream never emits | `timeout` (ttft stage) |
| `crash-mid-run.json` | 30 tokens streamed then exit code 0xC0000005 | `crashed`, partial tokens/samples persisted, metrics marked partial |
| `cancel.json` | normal stream; test calls `cancel()` at 500 ms | `cancelled` < 3 s, kill called on process tree, port released |
| `port-in-use.json` | stderr `couldn't bind` | retry on next port once, then `failed` |
| `no-timings.json` | completion response without `timings` | TPS labeled `ESTIMATED` (A12) |

Every case asserts: (a) the final status, (b) that `kill` was called exactly once or the process had already exited, (c) that there are no dangling timers (`vi.useFakeTimers` + `vi.getTimerCount()===0`), and (d) that the DB row matches the in-memory result.

### 2.5 Unit targets
- **Scoring**: normalization, weights, breakdown sum, exclusion of non-ok runs, determinism. Run each case twice and do a deep-equal on the outputs.
- **Candidate selection**: the configs generated for each machine fixture. Expect no ctx above the declared context, `gpuLayers=0` when there is no GPU, and memory-guard skips.
- **Cliff detection**: the sweeps in 2.3.
- **Storage**: write session → close → reopen the same temp file → read back an equal object. Also, the migration from an empty DB is idempotent.

## 3. Adversarial checklist

Each item gives the failure mode and the test that catches it.

| # | Failure mode | Test |
|---|---|---|
| X1 | **Single candidate.** Min-max normalization gives (x−min)/(max−min) = 0/0 = NaN, or rank-based normalization makes the single candidate score 0. | `session-single`: the score is finite, the breakdown has no NaN, and the rec is that config. The note says "only one candidate; not compared". |
| X2 | **All failed.** The code picks the "least bad" config, or divides by 0. | `session-all-failed`: rec is null, the reason is shown, and all 3 configs are in the excluded list. |
| X3 | **Ties.** The winner depends on insertion order or `Array.sort` stability. | `session-tie` with input order shuffled in both directions gives the same winner. The tie-break chain is documented (e.g. lower peak VRAM, then smaller model, then `configId` lexicographic). |
| X4 | **Zero or missing TPS.** `1/decodeTps` gives Infinity, `0/0` gives NaN, and NaN poisons the sort because comparisons are false. | `session-zero-tps`: the run is excluded or its metric is UNAVAILABLE. Assert `Number.isFinite` on every score. A property test feeds random runs that include 0, null, and NaN and checks that every output is finite or null. |
| X5 | **Rank-based normalization instability.** Adding an irrelevant bad config changes the order of the top two (IIA violation). | Score A and B, then add a much worse C. The order of A and B must not flip. Prefer min-max against fixed per-metric references, or document the chosen approach and pin it. |
| X6 | **Weights don't sum to 1**, or a weight is negative, so profile totals can't be compared. | Every one of the 7 profiles: \|Σw−1\|<1e-9 and all w ≥ 0. This is a static test over the profile table. |
| X7 | **Lower-is-better metrics are not inverted** (TTFT, load, peak memory), so slow configs get rewarded. | Two configs that are identical except TTFT: the faster one scores higher on every profile where the TTFT weight is above 0. |
| X8 | **Cliff false positives and negatives.** Noise, a single dip, a relative drop measured from a tiny base, or a comparison against the 2K baseline instead of the previous step. | `sweep-smooth` gives no cliff, `sweep-noisy` gives no cliff, and `sweep-cliff-16k-32k` gives the cliff at 32K exactly. Also check a sweep where 32K failed (oom): the practical max is 16K and it is labeled "limit: failure", not "cliff". |
| X9 | **Declared vs practical context.** The recommendation uses the GGUF-declared 128K when only 16K was measured as stable. For Long-context profiles, an unmeasured context must never be credited. | The Long-context Coding rec on the cliff fixture reports practical=16K (MEASURED) and declared=128K (DECLARED) separately. Its score uses the practical value. |
| X10 | **Partial-offload CPU spikes.** The load-phase spike counts as inference CPU, or a spike is read as spill or throttling. | `partial-offload-spike`: the inference-phase average excludes the load window, the peak includes it, and there is no spill flag. |
| X11 | **Spill is inferred from RAM alone.** The RAM rise is really OS file cache from mmap, not spill. | A fixture where RAM rises but VRAM is well below its limit and TPS is steady: there is no spill flag. |
| X12 | **Units.** GiB vs GB, ms vs s, and tokens/s computed on a total that includes prompt tokens. | Unit test: 1.0 s with 100 predicted tokens gives decode 100 tps. Use `predicted_per_second` from timings if present. The fixture values in GiB must render to known strings. |
| X13 | **Warm vs cold.** The first run includes shader/pipeline compile (Vulkan) and file cache, which biases load time and TTFT. | The runner does one discarded warmup or records `cold:true`. A test asserts that the flag is present and that the scoring policy for cold runs is applied consistently. |
| X14 | **VRAM UNAVAILABLE treated as 0**, which skips all GPU configs, or as ∞, which leads to OOM. | `vram-unavailable` machine: the guard falls back to DECLARED or ESTIMATED with a label, or it asks the user. The candidate list must not be empty solely because of this. |
| X15 | **Stale or duplicate GPUs** (registry ghosts, the iGPU) are picked as the benchmark device. | `rx9070` fixture: the device chosen for Vulkan is the dGPU, and the iGPU is not a candidate by default. |
| X16 | **Non-determinism from the clock or randomness.** `Date.now()` sneaks into the score, or the prompt differs per run. | Inject the clock. Pin `seed` and `temperature=0`. Assert that the prompt text hash is stored with each run. Run the score twice and compare with deep-equal. |
| X17 | **Comparing across sessions**, where a driver or runtime version changed or the thermal state differs. | Each run stores the runtime version, driver version, and model file hash. The rec only compares runs from the same session, or it warns about mixed versions. |
| X18 | **Aborted or partial runs enter the score.** | `crash-mid-run` and `cancel` results are absent from the scoring input. |
| X19 | **Candidate explosion.** The quant × ctx × layers × threads product exceeds the time budget. | For the rx9070 fixture, candidate count ≤ the documented cap. The estimated total duration is shown before the benchmark starts. |
| X20 | **Profile with no applicable metric.** For example, Max Quality depends on quant/params (DECLARED) only. | The rec explains that the quality term is DECLARED, not measured. The breakdown labels it. |

## 4. Manual verification script (real machine — run LATER, GPU allowed)

Machine: Ryzen 7 9800X3D, RX 9070 XT 16 GB, iGPU, 31 GB RAM, Win 11 26200. Before starting, close GPU-heavy apps and
open Task Manager → Performance, and keep it visible.

1. `npm test` shows all green. `npm run typecheck` is clean.
2. `npm run setup:runtime` must confirm the Vulkan llama-server exists under `vendor/llama.cpp`. `llama-server --version` must print a version.
3. `npm run dev`. Check A1 and A2: the CPU is 9800X3D 8C/16T, RAM is about 31 GB, the RX 9070 XT shows 16 GB, and the iGPU is marked integrated. There is **no RTX 3080** (A3, A4).
4. Check the runtimes panel: llama.cpp shows `available` with a version, and Ollama and LM Studio show a status and do not crash (A6).
5. Check the models panel: `qwen2.5-0.5b-instruct-q8_0.gguf` shows its size, quant Q8_0, and declared context (A7).
6. Select **Fast Assistant** and run the benchmark on the 0.5B model at 2K. Record the following, then compare against Task Manager's GPU "Dedicated memory" and "3D/Compute" graphs: load ms, TTFT, prefill/decode TPS, peak VRAM, and GPU %. VRAM should agree within about 10% (A11, A12, A14).
7. Run the context sweep 2K→32K on the same model. Each size gets a row. Check the 64K behavior against the declared context: it is either skipped with a reason or it runs (A13).
8. Use a larger model (7–14B Q4) that fits at low context. Sweep up to 64K and look for the VRAM plateau near 16 GB followed by a RAM rise and a TPS drop. The app should flag a cliff or spill at the right step (A15). If there is no cliff, record the measured values, because the thresholds may need tuning.
9. Click **Cancel** mid-decode. The UI should show `cancelled` within 3 s. `Get-Process llama-server -ErrorAction SilentlyContinue` should return nothing, and `netstat -ano | findstr <port>` should be empty (A23, A26).
10. Set a tiny total timeout in dev settings (e.g. 2 s) and run. Expect `timeout` and no leftover process (A24).
11. Force OOM by picking a config whose estimate exceeds VRAM + RAM. Expect `skipped_memory` before launch. Then override the guard, if a dev flag exists, and expect `oom` rather than an app crash (A25).
12. Open the recommendation for each of the 3 profiles you ran. Check that the breakdown contributions sum to the total and that each metric carries a MEASURED, ESTIMATED, or DECLARED label (A16, A22).
13. Kill the app mid-run from Task Manager (end the Electron process tree). Relaunch. The interrupted run should show `crashed/incomplete`, no orphan llama-server should remain, and earlier sessions should all be present (A20, A21, A26).
14. Quit normally and relaunch. The history and the last recommendation should be identical (A19, A20).
15. Record the driver version, the llama.cpp build, and the results in `docs/VERIFICATION-LOG.md`, which a later task will create.

## 5. Report

**STATUS**: Done (draft v1).
**RESULT**: ACCEPTANCE.md has 26 acceptance criteria, a fixture test plan (machine, telemetry, bench, runtime-failure), a 20-item adversarial checklist, and a 15-step manual script.
**CHANGES**: I created only `docs/ACCEPTANCE.md`.
**VERIFICATION**: I read `src/shared/types.ts`, `src/core/runtimes/types.ts`, `scanner.ts`, and `tests/fixtures/scan-rx9070.json`, which confirmed the AdapterRAM 4 GB cap, the stale RTX 3080 registry key, and the iGPU. I ran no inference and no tests.
**RISKS**:
- R1: `Sourced.status` (available/unavailable/unsupported) doesn't carry MEASURED/ESTIMATED/DECLARED yet. It needs a `kind` field.
- R2: the 40% cliff threshold and the 4 GB reserve are guesses and need tuning on real data (step 8).
- R3: Vulkan per-process VRAM/GPU% telemetry on AMD/Windows may be UNAVAILABLE. The source (PDH GPU Engine/Adapter Memory counters) is unverified.
- R4: `llamacpp.cancel()` is still NotImplemented.
**ASSUMPTIONS**:
- The status enum values are as listed in A11.
- The fixture filenames are proposals.
- DESIGN.md may rename fields. If it does, align this document to it, not the other way round.
**NEXT ACTION**: reconcile with DESIGN.md (labels, status enum, thresholds). Then have the implementer create the fixtures in section 2 and the tests for X1–X8 first.
