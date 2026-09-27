# Benchmark methodology — Local AI Optimizer

This document describes what the code measures and how it scores, as implemented. Constants are quoted from code, and the file paths are authoritative when this document and the code disagree.
- Scoring config: `DEFAULT_SCORING_CONFIG` (`src/core/scoring/workloads.ts`, version `scoring-1.2.0`). Every decision is a rule in `src/core/interpret/rules.v2.json` (`interp-2`, `docs/INTERPRETATION.md` v2) made in `verdicts()` and recorded in `Recommendation.decisionTrace` (I-7.2); reasons, why-not lines and generation lines are rendered from it and cite the rule id. `rules.v1.json` is kept only as the record of what stored interp-1 recommendations cited.
- Session config: `DEFAULT_SESSION_CONFIG` (`src/core/benchmark/session.ts`).
- Candidate rules: `DEFAULT_CANDIDATE_RULES` (`src/core/benchmark/candidates.ts`).

## 1. Provenance

Every measured/estimated metric field is a `Metric { value, kind, source?, reason? }` (`src/shared/bench-types.ts`); plain numbers such as `ctx`, `promptTokens`, scores and weights are not (D26). `value === null` iff `kind === 'unavailable'`, and unknown is never 0.

| kind | Meaning | Examples |
|---|---|---|
| `measured` | Observed on this machine in this session | load time, TTFT, TPS from llama-server `timings`, telemetry peaks, quality pass rates, practical context ceiling |
| `declared` | Read from metadata | GGUF header (ctx, layers, params, quant) → `ModelMeta`; registry VRAM total; scan RAM total |
| `estimated` | Our formula, named in `source` | memory estimates (pruning only); TPS derived from wall clock when `timings` are missing; the quality prior |
| `unavailable` | Could not get it; `reason` says why | telemetry with 0 samples, a failed request, a missing counter |

## 2. Metrics per step (`BenchmarkRunResult`)

A step is one candidate config at one context size. The server is started with `-c ctx`. Reps are reduced to the **median** of the successful reps (`session.ts` `median`).

| Field | Source | kind |
|---|---|---|
| `loadTimeMs` | spawn → `/health` 200 (`LlamaCppBackend.loadModel`) | measured |
| `ttftMs` | client wall clock, request → first streamed content | measured |
| `prefillTps`, `decodeTps` | `timings` `prompt_n/prompt_ms`, `predicted_n/predicted_ms` (`parse.ts` `toPromptResult`) | measured. Without timings: `promptTokens/TTFT` and `decodeTokens/(total−TTFT)`, estimated (A12) |
| `totalMs` | client wall clock | measured |
| `peakVramBytes` | max per-PID `GPU Process Memory\Dedicated Usage` | measured |
| `peakSharedGpuBytes` | **adjusted spill**: max per-PID `GPU Process Memory\Shared Usage` − host-pinned buffers − the config's unsaturated first-step level, counted only while per-PID dedicated ≥ 80 % of the effective budget = min(total − other-process use, per-process budget) (never adapter totals). The in-step guard uses this same definition; a guard_abort row persists the tripping value (raw separately) | measured |
| `peakSharedGpuRawBytes`, `hostPinnedBytes` | raw per-PID shared peak; host-side buffers from the load log | measured / declared |
| `repDecodeTps`, `minRamAvailBytes` | decode of each rep (rep-variance rule I-6.1); lowest RAM available seen by the guard (I-4.3) | measured |
| `peakRamBytes` | max `Process V2(llama-server:<pid>)\Working Set - Private` (excludes the mmap file cache) | measured |
| `avgGpuUtil`, `avgCpuUtil` | mean over samples: GPU = max over the PID's 3D/Compute engine groups; CPU = `Processor(_Total)` | measured |
| `status`, `failureKind` | see §4 | — |
| `warm` | true when the size-matched warmup succeeded before the measured reps (X13) | — |
| `versions` | `{benchmark: BENCHMARK_VERSION ('bench-1.0.0'), prompts: PROMPT_VERSION ('ladder-1'), quality: the session's selected suite id ('qb-2.0.0' default, 'qb-1.1.0' when qualityMode='quick'), runtime: SessionDeps.runtimeVersion}` (X17) | declared |

- Telemetry: `startSampler({pid})` (`src/core/telemetry/sampler.ts`) runs `typeperf -si 1`.
  - It starts **during load**, as soon as the new server pid exists, and runs through warmup and reps.
  - Peaks (`peak*Bytes`) use every sample; memory stays allocated after load.
  - Averages (`avgGpuUtil`, `avgCpuUtil`) use only the warmup+measure window, so the load CPU spike is excluded (X10).
- Short steps: typeperf needs ~2 s for its first row, and a 0.5B step can finish in ~1.5 s. After the reps the runner waits up to `firstSampleWaitMs` = 3 s (from sampler start) for one real row. If there are still **0 samples**, every telemetry field is `unavailable` rather than 0.
- Percentages in (100, 1000] are clamped to 100 (real overshoot under full load). Only negative or > 1000 values mark a PDH glitch row, which is dropped whole; the counts are reported in `samplerErrors` ("typeperf rows dropped: N misaligned, M impossible"). Before this fix every overshooting row was dropped, which left 29–62 s heavy steps with 0 samples (H7).

## 3. Context ladder, warmup, reps

- Ladder: `[2048, 4096, 8192, 16384, 32768, 65536, 131072]` ∩ ≤ declared `ctxTrain` (unknown → cap 8192). Steps above that are listed in `skippedSteps` with a reason. `SessionRequest.ladder` can restrict the list further.
- Prompt (`ladder-2`): `ladderPrompt(ctx, fill)` (`src/core/benchmark/prompts.ts`) = `generateFiller(floor(fill·ctx), seed = ctx)` + "Continue the story in the same style:". After load the runner resizes `fill` with the model's tokenizer (`POST /tokenize`, up to 3 rounds) until the prompt is 0.75·ctx tokens ±3 %, so it is deterministic per ctx and model. Without a tokenizer it stays character-sized. History: `ladder-1` was character-sized only and measured ≈ 0.56·ctx (Run A: Llama 1168 tokens at 2K; H-LONG: 36,572 at 64K); the calibration script (CAL-S, CAL-14) was tokenized to 0.75·ctx. Rows from the two procedures differ in `versions.prompts` and are not speed-comparable (I-6.0). The request uses `n_predict = 128`, `temperature 0`, `seed 1` and `cache_prompt false`. There is no `ignore_eos`, so `decodeTokens` may be < 128.
- Warmup: one discarded request with the **same prompt** (`backend.warmup(prompt)`, 8 tokens). This compiles the Vulkan pipelines for that batch shape (DESIGN F7). A warmup failure fails the step.
- Reps: `reps = 2` measured prompts. The first failing rep fails the step.
- Timeouts: prompt `60 s + 10 ms × ctx`; load 120 s (inside the backend); quality 180 s per test.
- Pre-check before each step: est. RAM (`estimateMemory`) > live available − floor, where floor = **max(4 GiB, 8 % of RAM)** in every mode → `fail / skipped_memory`, and the model is not loaded. The guard polls from the start of **load** (every 250 ms for heavy configs). A trip during load kills the server, and the step is recorded as `guard_abort` before any request.
- In-step guard (1 s poll, 250 ms for heavy configs): RAM available **+ mmap credit** < floor, or adjusted spill (the `peakSharedGpuBytes` definition) > 2 GiB → `backend.cancel()` (or kill during load) → `fail / guard_abort`. The guard reason is recorded, not the "cancelled" error its own cancel caused.
  - RAM available is read from the OS (`readRamAvailableBytes` = `os.freemem`) on every poll, independent of typeperf; typeperf rows are the second source. **Fail-safe:** `guardBlindPollsMax` = 3 polls with neither → `guard_abort` "RAM guard inputs unreadable".
  - Heavy configs record a > 2 GiB spill (degraded + reason) and move to the next config instead of aborting; the RAM floor still aborts.
  - mmap credit = file × GPU layers / all layers. The weights already uploaded to the GPU are clean file pages the OS can drop. Calibration showed available RAM falls by ≈ the file size, so without the credit a 16 GiB model would falsely trip the floor.
  - Heavy mode uses the same floor; its pre-check counts only the resident RAM (`ramResidentBytes`).

### Stop rules (per candidate)
1. The first **FAIL** verdict (§6): oom, device_lost, crash, load_fail, load_timeout, config_drift, req_timeout, request_error, guard_abort, skipped_memory, cancelled.
2. **2 consecutive DEGRADED** steps (`maxConsecutiveDegraded`). The step right after a cliff therefore still runs.
3. Cancel, or a lost device. `device_lost` also skips every later GPU candidate in the session.
4. `ServerStuckError` (unload could not confirm the server exited, even after `taskkill /F`): a hard stop of the whole session. No further candidates or quality run, and no final cleanup that would drop the pid file.

## 4. Status vocabulary

`RunStatus = pass | degraded | fail | timeout | cancelled`, plus `FailureKind`:

| failureKind | Set when |
|---|---|
| `oom` / `device_lost` / `crash` | from `lastExit.reason` (`classifyExit` on the stderr tail: allocation failure → oom; DeviceLost → device_lost; otherwise crash) |
| `load_fail` / `load_timeout` | loadModel threw without an exit, or with "did not become healthy within 120s" |
| `config_drift` | loadModel threw `ConfigDriftError`: `/props` n_ctx ≠ the requested `-c` (e.g. 48K requested, 32K served = ctx_train) |
| `req_timeout` | `PromptResult.timedOut` |
| `request_error` | a request error with the server still alive |
| `guard_abort` / `skipped_memory` | the safety guards in §3 |

ACCEPTANCE A11 mapping: ok → pass; failed / oom / device_lost / crashed → fail + kind.

## 5. Quality suite (`qb-1.1.0`, file `tests.v1.json`)

- **Suite selection (since ccc0a4c).**
  - The default, unset or `qualityMode: 'thorough'`, is **qb-2.0.0** (`tests.v2.json` + `generators.v2.ts`, see docs/quality-v2.md): 60 items, of which 13 are generated from a per-session `qualitySeed` persisted with the request. It uses 3 seeded samples only for stochastic (T > 0) gen configs.
  - `qualityMode: 'quick'` is an explicit opt-in to qb-1.1.0 below: 17 items, 1 sample.
  - Runtime: the thorough suite is ≈3.5× the quick one. It runs once per model on the best-offload config, not once per configuration.
  - Every candidate in a session gets the same resolved items. Stored quality is reused only for the same suite id and seed.

- Files: `src/core/quality/tests.v1.json` (17 tests), `checkers.ts`, `index.ts`.
- Categories: instruction IF-01..03, reasoning RS-01..04 (since qb-1.1.0: step-by-step reasoning allowed, ending in a final `Answer: X` line; the `finalAnswer` checker takes the last Answer line, tolerating case, bold and backticks, and applies the inner exact/number check), coding CD-01..03 (`jsCode` cases `{expr, expected}`, compared as the JSON string of the value — order-sensitive for object keys, D18), structured SO-01..02, extraction EX-01..02, context CR-10/50/90 (needle at 10/50/90 % depth in seeded filler).
- Runner (`session.ts` `runQuality`): runs once per **model**, after all ladders, on the model's **best-offload** usable candidate: most GPU layers, then fastest measured decode. It is never run on a CPU baseline or `-nkvo` probe just because that ran first.
  - It loads at ctx = min(profile.targetContext, practical ceiling) with filler `min(3000, 0.6·ctx)` tokens.
  - The load is guarded like a ladder step (`guardedLoad`): previous server unloaded, live RAM pre-check, abortable load, the RAM-floor guard with fail-safe, checked between prompts. A trip discards the suite.
  - Each prompt goes `applyTemplate(messages)` → `runPrompt` (temp 0, seed 1) → `evaluateAsync` (`jsCode` runs in the child-process sandbox).
- **Generation configs** (`src/core/benchmark/gen.ts`): the baseline is thinking off, T=0 (controlled settings: fixed seed, temperature and token limits; not a proof of deterministic, fast or complete answers — a template may ignore the flag, D19). Thinking-capable models (`ModelMeta.genKnobs`, else `supportsThinking`) also run thinking at the lowest and the middle effort level at the model card's sampling (else T=1.0), max 3 configs, all on ONE guarded load; thinking prompts get maxTokens ×4 (≥ 1024) and 2× timeout. Stochastic configs (T > 0) run 3 seeded samples per test in Thorough mode (`qualityMode`), 1 in Quick. Rows carry genId, sample, answer/reasoning tokens and totalMs. Configs are considered in a fixed order (baseline, then ascending effort); a higher effort replaces the current choice only when its paired quality difference excludes 0, and only with non-empty applied template kwargs (I-8.0, I-8.1). Thinking configs are priced with the same-request effective answer rate of the suite (answer tokens / total s) and a time-to-answer = TTFT at the scoring rung + the suite's reasoning time. That time-to-answer mixes two contexts and is **ESTIMATED**, so wherever latency has weight a thinking pick is provisional (I-1.2) [I-3.2].
- A failed request counts as `pass:false` with the error in `detail`. An incomplete suite (cancel/crash/guard) is **discarded**, not stored as partial. A suite is saved in one transaction with its version and expected test count; on resume only a complete suite of the current version is reused.
- Q = 100 · Σ_c W_c · passRate_c / Σ_c W_c over the categories that have results, where passRate_c = Σ weight·pass / Σ weight. **Uncertainty** (`core/scoring/uncertainty.ts`, `unc-1`, a heuristic band): the unit is the unique item (testId + seed) or skill cluster; repeated samples collapse to an item mean and never narrow the band; truncated answers count as failures; any `infra_error` row quarantines the result (I-5.7). The report states Q, the band, method/version and coverage counts (items / skills / samples) [I-5.1].
- **Comparisons use the paired difference** on shared items (`pairedDifference`, I-5.2). If its interval excludes 0, quality decides for the quality-weighted workloads (`qualityFirst`: coding, large_coding, reasoning, document_analysis, max_quality) regardless of speed. Otherwise the quality delta is **neutralized**: its contribution is removed from both totals, the rest decides, and the neutralization is recorded in the decision trace. W = instruction .2, reasoning .25, coding .25, structured .1, extraction .1, context .1. Scoring restricts c to the profile's `promptSetIds`.
- With no results the prior is used, **immutable** (never adjusted from other candidates). A candidate whose decisive component is estimated or unavailable is **provisional** (I-1.2): it is ranked only among provisional candidates and returned as `provisionalBest`, never as the confirmed `best`. The prior is `min(90, 35 + 15·log2(params/1e9)) × {bpw ≥ 6: 1, ≥ 4.5: .97, ≥ 3.5: .9, else .75}`, labelled **estimated** in the breakdown and the reasons.

## 6. Cliff detection (`src/core/scoring/cliff.ts` `detectCliffs(steps, vramTotalBytes)`)

Steps are sorted by ctx. A step is **usable** iff its status is `pass|degraded` and `decodeTps` is finite and > 0. Relative rules compare with the **previous step only**.

| Constant (`cliff.*`) | Value | Rule → reason code |
|---|---|---|
| `decodeDropRatio`, `minDecodeDropTps` | 0.60, 2 t/s | dec_b/dec_a ≤ 0.60 **and** dec_a − dec_b ≥ 2 → `decode_drop` |
| `prefillDropPerDoubling` | 0.5 | pp_b/pp_a < 0.5^log2(ctx_b/ctx_a) → `prefill_drop` |
| `sharedSpillBytes` | 256 MiB | **spill** > 256 MiB → `shared_spill`, checked at every step. Spill = per-PID shared − host-pinned buffers (load log: `Vulkan_Host`/`CPU`, plus `CPU_Mapped` without mmap) − the config's first-step level when that step was unsaturated; counted only while per-PID dedicated ≥ `vramSaturation`. Under `-lm none`, WDDM reports the CPU layers' pinned buffers as shared (Qwen3.8 ngl 50: 3.82 GiB shared with 4 GiB of VRAM free), so raw shared alone was a false positive. `peakSharedGpuRawBytes` and `hostPinnedBytes` are recorded alongside. |
| `rawSharedGrowthBytes` | 1 GiB | raw per-PID shared − host-pinned grows ≥ 1 GiB vs the previous rung → `shared_spill`, whatever the dedicated share (cal-longctx-2026-09-27: with ≈ 2.5 GiB held by other processes, collapsing rungs sat at 67–73 % dedicated with 2.0–2.1 GiB raw shared and adjusted spill 0.00). The runner's adjusted-spill saturation share is also taken against the effective budget (VRAM total − measured other-process use). |
| `vramSaturation`, `ramGrowthBytes` | 0.80, 1 GiB | per-PID dedicated ≥ 80 % of the VRAM total **and** per-PID private RAM ≥ +1 GiB vs the previous step → `vram_spill`. This corroborates `shared_spill`. RAM growth alone never flags, and available-RAM deltas (mmap) must never be fed in. |

- Verdicts:
  - **FAIL** = not usable (`run_failed` / `invalid_metrics`).
  - **DEGRADED** = any rule fired, or `beyond_limit` (sticky: every step after the first cliff or failure is at least degraded). A `decode_drop` the **next rung recovers from** (back above 0.60 × the pre-drop rate) is treated as a transient dip, not a cliff; the last rung has nothing to confirm with, so its drop counts.
  - **PASS** = otherwise.
- Outputs:
  - `practicalContextCeiling`: the last step of the all-PASS prefix, measured.
  - `degradedContextCeiling`: the last non-FAIL step before the first FAIL.
  - `limitedBy`: none | cliff | failure | untested.
  - `spillFreeUpTo`.
  - Reasons: `{code, metric, fromCtx, toCtx, from, to, ratio, threshold, message}`, e.g. "decode TPS fell 62% between 16K and 32K (82.0 → 31.0 t/s)".
- CPU % is never a cliff or spill signal.

### Calibration status (`docs/calibration-2026-09-27.md`, RX 9070 XT, b11208 Vulkan)
- **Llama-3.1-8B Q4_K_M, full offload, 2K→64K:**
  - Decode was 109.2 / 104.6 / 99.9 / 88.0 / 72.0 / 52.3 t/s (first measured rep; the fixture `calib-8b-rx9070.json` holds the mean of two, 109.45 … 52.35). The per-step ratios are .96 / .96 / .88 / .82 / .73, all above 0.60, so there is correctly no cliff.
  - Prefill was 2943 → 1514 t/s. The worst per-doubling ratio is .74, above 0.5.
  - CAL-S used **adapter** counters (not per-PID): the adapter shared delta was ≤ 0.12 GiB (< 256 MiB) and adapter dedicated peaked at 14.6 GiB. These are not comparable with CAL-14's per-PID values.
  - Result: no false positives on a real smooth sweep.
  - Do not raise `decodeDropRatio` into ~0.70–0.75: the healthy 32K→64K ratio is 0.73, so gradual decay would be flagged as a cliff (D26).
- **Real cliff observed** on Qwen2.5-14B Q4_K_M, full offload (`calib-14b-rx9070.json`):
  - The run went 2K/8K/16K/32K: decode 61.6 / 57.1 / 51.2 / **26.4** t/s, per-PID shared 0.02 → **1.05 GiB** at 32K, TTFT 22.6 s.
  - `decode_drop` fired (ratio 0.516 ≤ 0.60), and so did `shared_spill` (1.05 GiB > 256 MiB).
  - Verdict: practical 16K, degraded 32K, `limitedBy: cliff`, which matches a human reading.
  - 48K failed (the server served only n_ctx 32768 = ctx_train, HTTP 400).
- **WDDM saturation:** in CAL-14 (raw per-PID shared, before the adjusted-v1 spill metric) per-PID dedicated stalled at 13.25 GiB = **83 %** of 15.92 GiB when the spill began, so the old 0.95 saturation rule could never fire. It is now 0.80: it fires at 14B 32K (private WS +1.02 GiB). For the 8B at 64K the 81 % is an adapter-minus-idle proxy and its private RAM was not captured. This is one scoped observation on this GPU, not a Windows constant: under contention (H-LONG) spill-like shared growth began at 67–73 % per-PID dedicated, which the raw-growth rule now catches.
- **Spill vs partial:** in CAL-14 at 32K the spilled 14B full offload decodes 26.4 t/s, while ngl 30 decodes 5.6 t/s (4.7× slower). This is scoped evidence for the same-model veto (I-7.6), which applies only when the full offload meets the same hard constraints (§8).
- The 0.60 decode and 256 MiB shared thresholds are **policy** consistent with the two sweeps (the 14B cliff fires, the smooth 8B sweep stays quiet). Two sweeps do not validate their sensitivity or specificity. Other GPUs are still [A].
- **Effective per-process VRAM budget (cand-1.5, I-4.1/I-4.5/I-2.8):** the usable VRAM for one model is the per-process ceiling of this GPU + driver + backend, not the card total. With llama.cpp Vulkan on this card, dedicated never exceeded ≈ 11.6–13.25 GiB of 15.92 (14B: 13.25 GiB = 83 %; 8B f16 64K: 11.6 GiB = 73 %), so a 16 GB card behaved like ≈ 12–13 GB. Ollama (ROCm/HIP) on an RX 6800 16 GB used the full 16 GB. The ceiling is therefore backend-specific, and even on one card it depends on the allocation pattern.
  - **Observations:** a run whose per-PID shared − pinned − baseline exceeds 256 MiB records its per-PID dedicated peak as a ceiling observation `{ceiling, model, ctx, kv type, KV bytes, largest single device buffer}` (buffers from the load log). This is taken *after* the I-2.8 placement retry, so a spill that a fresh server clears is not recorded. Observations are stored per `GPU name | driver version | backend:build` (`vram_budget_observation`) and never generalised across keys.
  - **Budget:** the min of the observations whose largest buffer is within 1.25× of the planned one; else the most conservative observation (and it says so); else 80 % of the total, labelled **estimated**: "no measured budget on this machine yet; assuming 80 % (some GPUs/backends allow 100 %)".
  - **Planning:** the budget is min(total − in use − reserve, measured budget). The estimated fallback is disclosed but does **not** prune, because the 27B ngl 55 @8K ran clean at 13.03 GiB, above 0.8 × 15.92 = 12.74. I-4.1 names whichever budget is the basis.
  - **I-2.8:** room = budget − dedicated peak (capped by same-window adapter free). Spill with ≥ 1 GiB room means placement: restart and re-measure once. Spill at the budget is a genuine capacity limit.
- **Memory-bound, not cliff-bound:** on 16 GB the 8B ceiling is 64K because 128K is estimated far over the VRAM budget and pruned. Reasons say "memory-bound at 64K (128K: est. VRAM … > budget …)" instead of "no cliff".
- **Partial offload** (ngl 20/32 @ 8K) decodes at 17.5 t/s vs 99.9 with full offload, and ngl 0 at 7.1 t/s. That is a config difference, not a cliff. It is handled by linear genSpeed, the partial-offload memory cap and the partial-offload gate (§7, §8).
- Tests: `tests/scoring/calibration.test.ts` (fixture `calib-8b-rx9070.json`, built from the calibration table) and `tests/scoring/adversarial.test.ts` (#3, `calib-rx9070-2026-09-27.json`).

## 7. Component scores (`src/core/scoring/components.ts`)

All components are 0–100 with **absolute** normalization: fixed floors and targets per profile, no min-max or rank across candidates. A single candidate therefore scores normally, and components never depend on other candidates. Eligibility can (D15): a usable full offload makes the same model's partial configs ineligible, and measured quality caps other candidates' priors. An **unavailable input scores `norm.unknownScore` = 50 (neutral)** and its note starts with "unknown (…)". It is never 0 (that would punish missing telemetry) and never 100, and weights are not renormalized. Calibration: requests under ~1 s get 0 typeperf samples. A step with no usable run still scores 0 (it is excluded anyway).

Two steps are picked per candidate (`referenceStep`):
- **Scoring step** (`referenceCtx`): the largest PASS step ≤ targetContext, else the smallest PASS step, else the smallest usable step. Speed, latency and memory are read here, so candidates are compared at the same workload need.
- **Recommended context** (`recommendedCtx`): the largest PASS step ≤ `maxContext` whose full-prompt TTFT ≤ `latencyToleranceMs`. Steps with unknown TTFT qualify only up to targetContext. This is the `-c` to configure, and it is reported with its TTFT in the reasons. Scoring deliberately does *not* use this step: at the tolerance edge the latency score would be ≈0 by construction.

| Component | Formula (`norm.*`) |
|---|---|
| quality | §5 Q (measured) or the prior (estimated) |
| genSpeed | `linScore(decodeTps, 2, genTargetTps) = 100·clamp((x−2)/(t−2), 0, 1)`. It is linear because the log version left a 5.7× slower partial offload only 6–12 points behind (calibration). |
| prefillSpeed | `logScore(prefillTps, 20, prefillTargetTps)`, where `logScore(x,f,t) = 100·clamp(ln(x/f)/ln(t/f), 0, 1)` (prefill spans orders of magnitude) |
| latency | 100 at TTFT ≤ tol/10, falling log-linearly to 0 at tol = `latencyToleranceMs`. The TTFT is for a prompt filling the reference step. |
| memory | u = peak/total (VRAM; RAM only for ngl=0 on a machine with no GPU): 100 at u ≤ .80, then linear to 40 at .97, then to 0 at 1.0. Shared > 256 MiB caps it at 30; partial offload (including ngl=0 on a GPU machine) caps it at 60. |
| stability | 100 · usable/attempted over steps ≤ targetContext, − 20 if any crash/device_lost |
| context | `100·clamp(log2(ceiling/2048) / log2(targetContext/2048), 0, 1)`, using the measured practical ceiling (never the declared ctx) |

Total = Σ wᵢ · scoreᵢ. Breakdown rows `{component, input: Metric, score, weight, contribution}` sum exactly to the total.

### Workload profiles (`WORKLOADS`)
| Profile | Q | G | P | L | M | S | C | targetCtx | maxContext | genTarget | ppTarget | TTFT tol | minQ | quality categories |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| general_chat | .30 | .25 | .05 | .15 | .10 | .10 | .05 | 8K | 16K | 60 | 1000 | 8 s | 40 | instr, reason, struct, extract |
| coding | .40 | .20 | .10 | .10 | .05 | .10 | .05 | 16K | 64K | 60 | 1500 | 15 s | 50 | coding, instr, struct |
| long_context_coding | .30 | .10 | .20 | .05 | .05 | .10 | .20 | **64K** | 128K | 40 | 2000 | **90 s** | 50 | coding, context |
| reasoning | .45 | .20 | 0 | .05 | .05 | .15 | .10 | 8K | 16K | 50 | 1000 | 10 s | 55 | reason, instr |
| document_analysis | .30 | .05 | .25 | .05 | .05 | .10 | .20 | **64K** | 128K | 30 | 2000 | **120 s** | 45 | extract, context, struct |
| fast_assistant | .15 | .35 | .10 | .25 | .05 | .10 | 0 | 4K | 8K | 100 | 1000 | 2 s | 30 | instr, extract |
| max_quality | .70 | .05 | 0 | 0 | .05 | .15 | .05 | 8K | 16K | 15 | 500 | 20 s | 0 | all |
| **large_coding** | .40 | .10 | .10 | 0 | .05 | .15 | .20 | 64K | 128K | 30 | 1000 | 180 s (advisory) | 50 | coding, instr, struct, context |

**Long-context tolerances:** a full 64K prefill costs ~21–32 s on the 8B, but it is a one-off per session (follow-ups reuse the prompt cache), hence 90 s / 120 s — a rationale for a deployed chat with prompt caching; this benchmark always sends `cache_prompt:false` and measures a full prefill each time (D26).

**`large_coding`** ("bigger model, slower is fine"):
- Latency is advisory: TTFT is reported but never gates, and its weight is 0. The decode gate is 8 t/s.
- The Dashboard "large-scale coding" preset applies heavyMode + requiredContext 64K; choosing the workload in the Benchmark dropdown does not (D10).
- Calibrated (`required.test.ts`): a 27B partial offload at 12 t/s with measured Q 100 beats the 8B at ~100 t/s with Q 60 when both reach 64K. Plain Coding still picks the 8B.

Calibrated on the 8B full offload (TTFT 0.5 s @ 2K, 1.1 s @ 4K, 2.2 s @ 8K, 4.9 s @ 16K, 12.0 s @ 32K, 32.4 s @ 64K). The recommended contexts are asserted in `calibration.test.ts`: fast_assistant 4K; general_chat, reasoning and max_quality 16K; coding 32K; long_context_coding and document_analysis 64K. genTarget values are near full-offload speed on this GPU class (52–109 t/s), so partial offload is clearly behind. These are calibrated on one machine and one model, so they are still [A] elsewhere.

## 8. Recommendation (`src/core/scoring/recommend.ts` `recommend(inputs, machine, workload)`)

- **Excluded**: candidates with no usable step, listed with their step reasons (A17).
- **Gates** (the candidate is still ranked, but `eligible:false`): practical ceiling < 0.5 · targetContext; stability < 50; quality < minQuality (an estimated quality also gates, and the reason says "estimated"); **TTFT at the scoring step > latencyToleranceMs**; **decode at the scoring step < `minDecodeTps`** (fast 30, chat/coding 10, reasoning/long-ctx 5, doc 3, max quality 2 t/s).
- **Partial-offload veto** (I-7.6): on a GPU machine, a partial config (ngl < all, including ngl 0) is ineligible only when a full-offload config of the **same model** meets the same hard constraints (required context, decode floor, latency); the comparator is named in the failure. Scoped evidence: CAL-S ngl 20 decodes −83 % (8B), CAL-14 spilled 14B full offload beat ngl 30 by 4.7× at 32K.
- **Scoring rung** (I-7.1): all candidates of a workload are scored at one rung = min(target, largest clean context of any candidate). A candidate that does not reach it is read at its largest passing rung below and scores latency 0; one whose ladder merely skips that rung is read at the nearest passing rung below without penalty.
- **Measured safety before ranking** (I-1.1, I-4.3): a measured RAM minimum below the recorded floor on any rung the pick relies on is a hard failure; absent safety evidence is "not verified". Only MEASURED decisive components confirm a pick: an estimated or unavailable term (e.g. an estimated VRAM peak, an unmeasured spill — never treated as 0) makes it provisional. A candidate read at another rung than the common one (its ladder skipped it) is provisional too. Paired quality comparisons require the same generation config, checker version and token budget on shared items.
- **Eligibility** (I-6.0, I-1.1): only completed rows with `warm === true`, a measured decode and the session's benchmark/prompt versions (rows without versions are rejected once the session version is known) enter speed scoring; hard constraints (required context, decode floor, TTFT) are evaluated first and an unknown value counts as not met. The partial-offload veto applies only against a same-model full offload that meets the same hard constraints (I-7.6).
- **Ranking / tie-break**: confirmed eligible → provisional eligible → ineligible; within each, total rounded to 1e-6, descending → higher decode → lower peak VRAM (CPU-only counts as 0) → `configId` ascending (S2). `decisionTrace.comparisons` records every pairwise comparison as it happened, with the totals actually compared (with or without the quality contribution, or the paired quality difference); `tieBreakChain` is the chain of the comparison that decided, on those totals; a revisited leader is reported as `cycle` (not a proven total order). Excluded candidates, per-candidate component basis/safety/recommended ctx, the quality-vs-speed basis and the executing cliff/norm parameters (`thresholdsUsed`) are in the trace too; reasons are rendered from it. Request overrides (required context, decode floor) are applied inside `verdicts()` for every caller. Unknown values sort last. The winner and ranking are independent of input order; auxiliary reason lists (unplanned models, RAM-skip messages) keep the caller's order (D15).
- **Best** = the first eligible candidate, with `practicalContext` (measured) and `declaredContext` (declared) reported separately.
- **Alternatives**, among eligible candidates, each with a configId tie-break:
  - `fastest`: decode TPS at the reference step.
  - `bestQuality`.
  - `bestLongContext`: practical ceiling, then decode.
  - `lowestMemory`.
- **No winner**, with the message:
  - no inputs → "No recommendation: no candidates were benchmarked"
  - nothing usable → "No recommendation: no successful runs"
  - all gated → "No recommendation: no candidate meets the <profile> requirements", plus each candidate's gate failures.
- **Headline** (`best.headline`): "`<model> <quant> [(n/L layers, KV q8_0, KV in RAM)]` @ `<recommendedCtx>` — `<decode>` t/s, quality `<Q>` [(estimated)], no spill up to `<ctx>`". The decode is at the recommended ctx. Example: "Qwen3.8-27B-UD-Q4_K_M (55/65 layers) @ 8K — 13.0 t/s, quality 100, no spill up to 8K".
- **Why not** (`whyNot: [{configId, model, summary}]`): the top 2 non-winners by rank, plus the best-ranked config of each other model with higher **measured** quality than the winner. Each summary is one sentence built from the facts the score used, never prose without a number:
  - ineligible → "`<label>`: ineligible — `<gate failures>`";
  - otherwise "`<label>`: `<pros>` but `<cons>`; `<Profile>` total a vs b". The pros and cons come from four facts: the quality delta (rounded points); decode at each candidate's scoring step, when they differ by ≥ 10 % ("slower (13.0 t/s at 8K vs 93.2 at 16K)"); the first `decode_drop` with any shared spill ("decode fell 51.2 → 26.4 t/s after 16K (shared-VRAM spill 1.05 GiB at 32K)"); and the practical context vs the winner's.
  - Empty when there is no winner. Asserted on the 8B/14B and heavy fixtures in `tests/scoring/whynot.test.ts`.
- **Reasons** (deterministic English): the score, "Only one candidate; not compared", the top-2 contributions, "Recommended context 32K: TTFT 12.0 s for a full prompt (tolerance 15 s), decode 72.0 t/s", practical vs declared context plus what bounded it (cliff / failure / "memory-bound at 64K (128K: est. VRAM …)" / declared ctx), a warning when runs mix runtime/benchmark versions, "No VRAM spill up to …", every cliff message, and the estimated-quality notice.

## 9. Candidate generation (`src/core/benchmark/candidates.ts`)

Estimates are `kind:'estimated'`, prune only, and never rank.
- **`estimateMemory`:**
  - KV = ctx·L·Hkv·(dk+dv)·bytes. This is exact vs the llama-server log: 8B 0.125 MiB/token, Qwen2.5-1.5B 0.0273 MiB/token.
  - VRAM = file·min(ngl,L)/L + KV + compute, where compute = 32 MiB + ubatch·n_embd·32 B + 1 KiB·ctx (fit to the measured 61–164 MiB compute buffers).
  - RAM = **resident part**: weights not on the GPU + CPU-side KV + 512 MiB. It is checked after the previous step's server is unloaded (D1: its mmap had made a 14B look out of RAM). Offloaded weights' mmap pages are reclaimable and are credited by the in-step floor guard. Spill detection never uses RAM available: it uses the private working set and per-PID shared memory.
  - Estimated vs measured on 8B: −2 % … +3 % per rung.
- **Budgets:**
  - VRAM = total − in-use − **1 GiB**. The margin absorbs the measured residual of 0.2–0.9 GiB, largest at 64K. When the in-use reading is unavailable, 1.5 GiB is assumed (`vramInUseUnknownBytes`, measured idle here 1.2–1.4 GiB), never 0.
  - RAM = available (else total) − 4 GiB.
  - RAM over budget → the step is skipped (never kept).
  - VRAM over budget → the first such step is kept once, to observe the cliff, but only if est ≤ 1.15 × budget (`keepOverVramMaxRatio`). Every later step is skipped.
- **Order:**
  1. ngl=all f16.
  2. ngl=all q8_0 KV, when targetContext ≥ 32K and declared ≥ 32K.
  3. If (1) is rejected: **normal mode generates nothing more.** The rejection says "full GPU offload does not fit — enable heavy-model mode". **Heavy mode** (`SessionRequest.heavyMode`) generates up to 4 partial configs, all `expectDegraded` with a `degradedReason`:
     - the max ngl that fits at the target ctx, and that ngl − 4 (so an edge spill still leaves a clean config);
     - the KV-in-RAM (`-nkvo`) rung is generated **only when the target ctx's KV costs ≥ 8 layers** vs the smallest ctx (`nkvoMinLayerGain`), and it runs after the KV-on-GPU rungs. Measured at 2K: 57-nkvo 7.4 t/s vs 50 layers 10.7 (Qwen3.8); 26-nkvo 27.0 vs 21 layers 38.5 (Gemma-4);
     - MoE models (`expertCount` > 0) get the note that active parameters are much smaller, so partial offload is cheaper (Gemma-4-26B-A4B 38.5 t/s at 21/30 layers vs dense Qwen3.8-27B 10.7 t/s at 50/65). There is no formula change;
     - the max ngl with KV in RAM (`kvOffload:false` → `-nkvo`, verified in b11208 `--help`), when the rule above allows it;
     - a CPU baseline (ngl 0, `-dev none`).
     Heavy configs load **without mmap** (`-lm none`): with mmap the whole GGUF was reported to stay resident at 55/65 layers (narrative only: no before/during/after-unload RAM observation is archived, so no cache/release claim is made; the file is 16,464,440,224 bytes ≈ 15.33 GiB). Their VRAM estimate adds the output projection (n_vocab × n_embd at the file's bits/weight) and vocab-sized logits, after a reported ngl 62 spill of Qwen3.8 that the old estimate had called a fit (no archived ngl 62 row; the archived ngl 62 -nkvo row was cancelled at 0.84 GiB raw shared). A shared spill > 2 GiB on a heavy config is recorded (degraded + spill reason) and the ladder moves to the next config instead of aborting. The RAM floor still aborts.
     The RAM check uses `ramResidentBytes` (non-GPU weights + CPU KV + 0.5 GiB, **+1.5 GiB more at ngl 0**) vs available − 4 GiB. There is no keep-over step.
     The **CPU baseline is skipped when the file is > 50 % of total RAM** ("CPU baseline skipped: model is >50% of system RAM"). In the real 27B run it drove available RAM to 1.0 GiB.
  4. ngl=0 only when there is no GPU (`cpuOnlyMaxParams` = 0 = off).
  - `estimateMemory` splits KV per layer: GPU KV = the sum over the **last ngl layers** (llama.cpp offloads the tail; with a hybrid/SWA layout those layers' KV differs from the average). For a uniform model it equals the layer share (8B ngl 20/33: KV CPU 416 + Vulkan 608 MiB). `rulesForRequest(req)` gives the runner and any re-planner the same rules → the same configIds.
- **Other rules:**
  - Max 4 per model.
  - Threads = physical cores.
  - fa = on.
  - VRAM total unavailable → no VRAM pruning (with a note).
  - SWA/hybrid archs prune with the per-layer KV layout (`kvLayout`); missing head metadata gives KV 0 flagged unknown (D02, D13).

### Required context (`SessionRequest.requiredContext`: 32K / 64K / 128K; Auto = the workload default)
- **Effective profile** (`effectiveProfile`): targetContext = required, maxContext ≥ required, latency **advisory**. The runner, the quality ctx and `recommend()` all use it; `recommendForWorkload(data, workload, request)` recomputes any workload from stored data.
- **Ladder:** every candidate runs all rungs ≤ min(required, declared) — the UI ladder cap is ignored, nothing above is run — and the "2 consecutive degraded" stop is suspended below the required ctx.
- **Gate:** practical ceiling ≥ required, else "practical context 64K < required 128K (limited by spill|memory|cliff|failure|declared context)". TTFT above the tolerance does not gate; the reason says "(above the profile's N s tolerance; accepted because you required X)".
- **User decode gate:** `SessionRequest.minDecodeTps` replaces the profile gate when set.
- **Fallback:** if nothing passes but some confirmed config reaches the required ctx and fails only the **workload's** speed gates, the fastest such config is returned with `best.fallback = 'meets required context; below preferred speed'`. A **user** decode floor is never overridden: such configs are listed as `unmetAlternatives`, and `best` stays null (I-2.5, I-1.1).
- **Ladder with a required context (L2):** an explicit `req.ladder` that reaches the required rung is a rung selection and is honoured; a lower ladder (the UI's size cap) is overridden by the required context.
- **Long-context calibration** (`docs/calibration-longctx-2026-09-27.md`, 8B Q4_K_M, ≈ 2.5 GiB of VRAM held by other processes): q8_0 KV is clean to 64K at 68.5 t/s (TTFT 21.8 s); 128K is reachable only with q8_0 and degraded under contention (24.24 t/s, TTFT ≈ 65.4 s, 2.00 GiB raw shared — adjusted 0.00 under the v1 share-of-total gate, so this is shared growth plus collapse, not a recorded adjusted-v2 spill); f16 KV is clean to 32K (76.5 t/s) and collapsed at 64K (24.97 t/s, 2.08 GiB raw shared); `-nkvo` reaches 32K at 7.25 t/s. Loaded host (≈ 2.4–2.5 GiB adapter-minus-PID), `ladder-1` fill ≈ 0.56·ctx, default mmap. No idle 128K run exists; the idle-budget prediction in that report is arithmetic, not an observation.
- **Long-context variants:** when f16 full offload cannot reach the target ctx within VRAM, a KV q8_0 variant and a KV-in-RAM (`-nkvo`) full-offload variant are added, each noted. For example, the 8B on 16 GB at 128K: q8_0 plans 128K; the `-nkvo` variant is RAM-limited at 128K (16 GiB KV). Partial offload stays heavy-mode only.
- **CR-04-long:** when required ≥ 32K, one extra needle test at 50 % depth of a ~0.75 × required prompt runs on a config that reached the required ctx (category context). Otherwise it is skipped, with a reason in the recommendation.
- **Quality over speed:** when the pick decodes ≥ 1.5× slower than the fastest eligible alternative, the reasons say "Chosen for quality over speed: decode X t/s (Y× slower than …)".

### Heavy-model mode (summary)
- **What it is:** opt-in (`SessionRequest.heavyMode`, the "Include heavy models" checkbox). A model whose full GPU offload does not fit gets up to 4 partial-offload probes (§9, all `expectDegraded` with a reason) instead of being rejected.
- **RAM:** the check uses the resident part with a 4 GiB reserve (the same floor as normal mode). The in-step floor credits the mmap pages of GPU-offloaded weights. The guard polls every 250 ms from the start of load for heavy configs.
- **Order:** full-offload configs run first, then heavy configs most-offloaded first (KV-on-GPU rungs, then -nkvo), CPU baselines last. Each model's quality runs as soon as its last config finished, followed by a saved provisional recommendation, so a later cancel/abort still leaves one (D05). A fatal ServerStuckError stops before any further save.
- **Unplanned models:** a model with no candidates is named in the recommendation reasons, e.g. "Not benchmarked: X — … full GPU offload does not fit — enable heavy-model mode".
- **Gates:**
  - `minDecodeTps` per workload (fast 30, chat/coding 10, reasoning/long-ctx 5, doc 3, max quality 2 t/s) keeps slow partial configs out of interactive workloads.
  - A partial config is never eligible when the same model's full offload ran.
  - Across models, the scores decide: a ~27B at 8 t/s can win Maximum Quality.
- **Reasons:** they say "Partial GPU offload (N/M layers) — degraded speed expected: decode X t/s (…)".
- **KV:** it uses the per-layer layout (hybrid `full_attention_interval`, SWA pattern, per-layer heads) when the GGUF declares it; otherwise an all-layers upper bound, never 0.
- **Calibration** (`docs/calibration-heavy-2026-09-27.md`, fixture `calib-heavy-qwen38-rx9070.json`):
  - Qwen3.8-27B (dense hybrid, 16.5 GB) at 55/65 layers, 2K/4K/8K (H-CONSOLE: console observations of a reaped run, default mmap load, quality lost; no recorded prompt token counts): prefill 611/661/689 t/s, decode 12.4/13.0/13.0 t/s, TTFT 1.9/3.5/6.7 s, VRAM 12.7–13.0 GiB, shared 0.04 GiB.
  - For comparison, the 8B full offload decodes 110 → 79 t/s over 2K → 32K.
  - Outcomes asserted in `heavy.test.ts`: the 27B wins Maximum Quality when its quality is higher; it passes Coding's 10 t/s gate but loses to the 8B on speed; Fast Assistant rejects it.
  - H-0953 ladder data (`session-run-H-coding-heavy-…09-53-34…INVALID-quality.json`; its big-model quality is INVALID — thought blocks open): Qwen3.8-27B at 54/65 layers decodes 12.6 → 10.8 t/s from 2K to 16K; Gemma-4-26B-A4B (MoE) at 25 layers ≈ 46–50 t/s. H-0939 ladder rates at 2K: -nkvo 7.44 vs ngl 50 10.74 t/s (dense), 27.04 vs ngl 21 38.50 t/s (MoE); the 3.6× MoE/dense figure is 38.50/10.74 across different families and is not a general law.
  - H-1021 re-run (valid baseline quality with closed thought templates, qb-1.1.0): Qwen3.8 ngl48/44 and Gemma-4 ngl22/18 ladder timings; quality 15–16/17 per model.
  - The CPU baseline was skipped for both (file > 50 % of RAM), so it is unmeasured.

## 10. Export (`src/core/export/config.ts`)
- `exportConfigFrom(rec, cand, model, sessionId)` takes the winner at `recommendedCtx`.
- `toLlamaServerArgs/Command` reproduce the benchmarked inference parameters (not the operational host/port/log/metrics flags or the executable path, D16): `-dev`, `-fit off`, `-c`, `-ngl 999|n`, `-t`, `-b 2048 -ub 512`, `-fa`, `-ctk/-ctv` for q8_0, `--parallel 1`, `--cache-ram 0`. The last flag turns off the host-RAM prompt cache (default 8 GiB): it saves each previous slot state when a new task starts, so the quality phase's 60+ tasks on one server held ≈ 8.6 GiB more RAM than the ladder's three-task launches (Qwen3.8 ngl 49 @8K). Benchmark and export both run without it, so the measured RAM is the deployed RAM.
- `toOllamaModelfile(c, {from})` emits FROM plus num_ctx / num_gpu / num_thread / num_batch, with flash attention and KV type as env-var comments.
- `toLmStudioSettings` uses lmstudio-js keys.
- `toJson` bundles every format, and `provenanceNote(rec)` says which inputs were measured, estimated or unavailable.
- The Ollama and LM Studio outputs are unverified, because neither app is installed.

## 11. Known limitations
See `docs/LIMITATIONS.md`.
