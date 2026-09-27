# Benchmark methodology — Local AI Optimizer

This document describes what the code measures and how it scores, as implemented. Constants are quoted from code, and the file paths are authoritative when this document and the code disagree.
- Scoring config: `DEFAULT_SCORING_CONFIG` (`src/core/scoring/workloads.ts`, version `scoring-1.0.0`).
- Session config: `DEFAULT_SESSION_CONFIG` (`src/core/benchmark/session.ts`).
- Candidate rules: `DEFAULT_CANDIDATE_RULES` (`src/core/benchmark/candidates.ts`).

## 1. Provenance

Every reported number is a `Metric { value, kind, source?, reason? }` (`src/shared/bench-types.ts`). `value === null` iff `kind === 'unavailable'`, and unknown is never 0.

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
| `peakSharedGpuBytes` | max per-PID `GPU Process Memory\Shared Usage` (spill signal, never adapter totals) | measured |
| `peakRamBytes` | max `Process V2(llama-server:<pid>)\Working Set - Private` (excludes the mmap file cache) | measured |
| `avgGpuUtil`, `avgCpuUtil` | mean over samples: GPU = max over the PID's 3D/Compute engine groups; CPU = `Processor(_Total)` | measured |
| `status`, `failureKind` | see §4 | — |
| `warm` | true when the size-matched warmup succeeded before the measured reps (X13) | — |
| `versions` | `{benchmark: BENCHMARK_VERSION ('bench-1.0.0'), prompts: PROMPT_VERSION ('ladder-1'), quality: 'qb-1.0.0', runtime: SessionDeps.runtimeVersion}` (X17) | declared |

- Telemetry: `startSampler({pid})` (`src/core/telemetry/sampler.ts`) runs `typeperf -si 1`.
  - It starts **during load**, as soon as the new server pid exists, and runs through warmup and reps.
  - Peaks (`peak*Bytes`) use every sample; memory stays allocated after load.
  - Averages (`avgGpuUtil`, `avgCpuUtil`) use only the warmup+measure window, so the load CPU spike is excluded (X10).
- Short steps: typeperf needs ~2 s for its first row, and a 0.5B step can finish in ~1.5 s. After the reps the runner waits up to `firstSampleWaitMs` = 3 s (from sampler start) for one real row. If there are still **0 samples**, every telemetry field is `unavailable` rather than 0.
- PDH glitch rows (an impossible percentage, with garbage in the other cells) are dropped whole by the sampler.

## 3. Context ladder, warmup, reps

- Ladder: `[2048, 4096, 8192, 16384, 32768, 65536, 131072]` ∩ ≤ declared `ctxTrain` (unknown → cap 8192). Steps above that are listed in `skippedSteps` with a reason. `SessionRequest.ladder` can restrict the list further.
- Prompt: `ladderPrompt(ctx)` (`src/core/benchmark/prompts.ts`) = `generateFiller(floor(0.75·ctx), seed = ctx)` + "Continue the story in the same style:". It is deterministic per ctx. The request uses `n_predict = 128`, `temperature 0`, `seed 1` and `cache_prompt false`. There is no `ignore_eos`, so `decodeTokens` may be < 128.
- Warmup: one discarded request with the **same prompt** (`backend.warmup(prompt)`, 8 tokens). This compiles the Vulkan pipelines for that batch shape (DESIGN F7). A warmup failure fails the step.
- Reps: `reps = 2` measured prompts. The first failing rep fails the step.
- Timeouts: prompt `60 s + 10 ms × ctx`; load 120 s (inside the backend); quality 180 s per test.
- Pre-check before each step: est. RAM (`estimateMemory`) > live available − floor, where floor = max(2 GiB, 8 % of RAM) → `fail / skipped_memory`, and the model is not loaded.
- In-step guard (1 s poll of new samples): RAM available < floor, or per-PID shared > 2 GiB → `backend.cancel()` → `fail / guard_abort`. The guard reason is recorded, not the "cancelled" error its own cancel caused.

### Stop rules (per candidate)
1. The first **FAIL** verdict (§6): oom, device_lost, crash, load_fail, load_timeout, config_drift, req_timeout, request_error, guard_abort, skipped_memory, cancelled.
2. **2 consecutive DEGRADED** steps (`maxConsecutiveDegraded`). The step right after a cliff therefore still runs.
3. Cancel, or a lost device. `device_lost` also skips every later GPU candidate in the session.

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

## 5. Quality suite v1 (`qb-1.0.0`)

- Files: `src/core/quality/tests.v1.json` (17 tests), `checkers.ts`, `index.ts`.
- Categories: instruction IF-01..03, reasoning RS-01..04, coding CD-01..03 (`jsCode` cases `{expr, expected}`, compared as canonical JSON), structured SO-01..02, extraction EX-01..02, context CR-10/50/90 (needle at 10/50/90 % depth in seeded filler).
- Runner (`session.ts` `runQuality`): runs once per **model**, on the first candidate with a usable step. It loads at ctx = min(profile.targetContext, practical ceiling) with filler `min(3000, 0.6·ctx)` tokens. Each prompt goes `applyTemplate(messages)` → `runPrompt` (temp 0, seed 1) → `evaluateAsync` (`jsCode` runs in the child-process sandbox).
- A failed request counts as `pass:false` with the error in `detail`. An incomplete suite (cancel/crash) is **discarded**, not stored as partial.
- Q = 100 · Σ_c W_c · passRate_c / Σ_c W_c over the categories that have results, where passRate_c = Σ weight·pass / Σ weight. W = instruction .2, reasoning .25, coding .25, structured .1, extraction .1, context .1. Scoring restricts c to the profile's `promptSetIds`.
- With no results, the scorer uses the prior `min(90, 35 + 15·log2(params/1e9)) × {bpw ≥ 6: 1, ≥ 4.5: .97, ≥ 3.5: .9, else .75}`, labelled **estimated** in the breakdown and the reasons.

## 6. Cliff detection (`src/core/scoring/cliff.ts` `detectCliffs(steps, vramTotalBytes)`)

Steps are sorted by ctx. A step is **usable** iff its status is `pass|degraded` and `decodeTps` is finite and > 0. Relative rules compare with the **previous step only**.

| Constant (`cliff.*`) | Value | Rule → reason code |
|---|---|---|
| `decodeDropRatio`, `minDecodeDropTps` | 0.60, 2 t/s | dec_b/dec_a ≤ 0.60 **and** dec_a − dec_b ≥ 2 → `decode_drop` |
| `prefillDropPerDoubling` | 0.5 | pp_b/pp_a < 0.5^log2(ctx_b/ctx_a) → `prefill_drop` |
| `sharedSpillBytes` | 256 MiB | per-PID shared > 256 MiB, checked at every step including the first → `shared_spill` |
| `vramSaturation`, `ramGrowthBytes` | 0.80, 1 GiB | per-PID dedicated ≥ 80 % of the VRAM total **and** per-PID private RAM ≥ +1 GiB vs the previous step → `vram_spill`. This corroborates `shared_spill`. RAM growth alone never flags, and available-RAM deltas (mmap) must never be fed in. |

- Verdicts:
  - **FAIL** = not usable (`run_failed` / `invalid_metrics`).
  - **DEGRADED** = any rule fired, or `beyond_limit` (sticky: every step after the first cliff or failure is at least degraded).
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
  - Decode was 109.2 / 104.6 / 99.9 / 88.0 / 72.0 / 52.3 t/s. The per-step ratios are .96 / .96 / .88 / .82 / .73, all above 0.60, so there is correctly no cliff.
  - Prefill was 2943 → 1514 t/s. The worst per-doubling ratio is .74, above 0.5.
  - Per-PID shared Δ was ≤ 0.12 GiB (< 256 MiB), and dedicated peaked at 14.6 GiB (≈ 92 % < 95 %).
  - Result: no false positives on a real smooth sweep.
  - Do not lower `decodeDropRatio` below ~0.65: a 25–30 % threshold would fire at 64K.
- **Real cliff observed** on Qwen2.5-14B Q4_K_M, full offload (`calib-14b-rx9070.json`):
  - The run went 2K/8K/16K/32K: decode 61.6 / 57.1 / 51.2 / **26.4** t/s, per-PID shared 0.02 → **1.05 GiB** at 32K, TTFT 22.6 s.
  - `decode_drop` fired (ratio 0.516 ≤ 0.60), and so did `shared_spill` (1.05 GiB > 256 MiB).
  - Verdict: practical 16K, degraded 32K, `limitedBy: cliff`, which matches a human reading.
  - 48K failed (the server served only n_ctx 32768 = ctx_train, HTTP 400).
- **WDDM saturation:** per-PID dedicated stalled at 13.25 GiB = **83 %** of 15.92 GiB when the spill began, so the old 0.95 saturation rule could never fire. It is now 0.80: it fires at 14B 32K (private WS +1.02 GiB) and stays quiet on 8B 64K (81 %, flat RAM).
- **Spill vs partial:** at 32K the spilled full offload decodes 26.4 t/s, while partial offload with ngl 30 decodes 5.6 t/s (4.7× slower). Partial offload is therefore never preferred because of a spill (§8).
- The 0.60 decode and 256 MiB shared thresholds are confirmed in both directions on this GPU. Other GPUs are still [A].
- **Memory-bound, not cliff-bound:** on 16 GB the 8B ceiling is 64K because 128K is estimated far over the VRAM budget and pruned. Reasons say "memory-bound at 64K (128K: est. VRAM … > budget …)" instead of "no cliff".
- **Partial offload** (ngl 20/32 @ 8K) decodes at 17.5 t/s vs 99.9 with full offload, and ngl 0 at 7.1 t/s. That is a config difference, not a cliff. It is handled by linear genSpeed, the partial-offload memory cap and the partial-offload gate (§7, §8).
- Tests: `tests/scoring/calibration.test.ts` (fixture `calib-8b-rx9070.json`, built from the calibration table) and `tests/scoring/adversarial.test.ts` (#3, `calib-rx9070-2026-09-27.json`).

## 7. Component scores (`src/core/scoring/components.ts`)

All components are 0–100 with **absolute** normalization: fixed floors and targets per profile, no min-max or rank across candidates. A single candidate therefore scores normally, and adding a candidate never reorders the others. An **unavailable input scores `norm.unknownScore` = 50 (neutral)** and its note starts with "unknown (…)". It is never 0 (that would punish missing telemetry) and never 100, and weights are not renormalized. Calibration: requests under ~1 s get 0 typeperf samples. A step with no usable run still scores 0 (it is excluded anyway).

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
| coding | .40 | .20 | .10 | .10 | .05 | .10 | .05 | 16K | 32K | 60 | 1500 | 15 s | 50 | coding, instr, struct |
| long_context_coding | .30 | .10 | .20 | .05 | .05 | .10 | .20 | 32K | 128K | 40 | 2000 | 40 s | 50 | coding, context |
| reasoning | .45 | .20 | 0 | .05 | .05 | .15 | .10 | 8K | 16K | 50 | 1000 | 10 s | 55 | reason, instr |
| document_analysis | .30 | .05 | .25 | .05 | .05 | .10 | .20 | 32K | 128K | 30 | 2000 | 60 s | 45 | extract, context, struct |
| fast_assistant | .15 | .35 | .10 | .25 | .05 | .10 | 0 | 4K | 8K | 100 | 1000 | 2 s | 30 | instr, extract |
| max_quality | .70 | .05 | 0 | 0 | .05 | .15 | .05 | 8K | 16K | 15 | 500 | 20 s | 0 | all |

Calibrated on the 8B full offload (TTFT 0.5 s @ 2K, 1.1 s @ 4K, 2.2 s @ 8K, 4.9 s @ 16K, 12.0 s @ 32K, 32.4 s @ 64K). The recommended contexts are asserted in `calibration.test.ts`: fast_assistant 4K; general_chat, reasoning and max_quality 16K; coding 32K; long_context_coding and document_analysis 64K. genTarget values are near full-offload speed on this GPU class (52–109 t/s), so partial offload is clearly behind. These are calibrated on one machine and one model, so they are still [A] elsewhere.

## 8. Recommendation (`src/core/scoring/recommend.ts` `recommend(inputs, machine, workload)`)

- **Excluded**: candidates with no usable step, listed with their step reasons (A17).
- **Gates** (the candidate is still ranked, but `eligible:false`): practical ceiling < 0.5 · targetContext; stability < 50; quality < minQuality (an estimated quality also gates, and the reason says "estimated"); **TTFT at the scoring step > latencyToleranceMs**.
- **Partial-offload gate**: on a GPU machine, a partial config (ngl < all, including ngl 0) is ineligible whenever a full-offload config of the **same model** has any usable step, even one that is gated or degraded by a spill. Calibration: ngl 20 decodes −83 % (8B), a spilled 14B full offload beat ngl 30 by 4.7×, and ngl 0 still uses the GPU.
- **Ranking / tie-break**: eligible first → total rounded to 1e-6, descending → lower peak VRAM (CPU-only counts as 0) → lower peak RAM → `configId` ascending (code-unit order). Unknown values sort last. Output is deep-equal for any input order.
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
- **Reasons** (deterministic English): the score, "Only one candidate; not compared", the top-2 contributions, "Recommended context 32K: TTFT 12.0 s for a full prompt (tolerance 15 s), decode 72.0 t/s", practical vs declared context plus what bounded it (cliff / failure / "memory-bound at 64K (128K: est. VRAM …)" / declared ctx), a warning when runs mix runtime/benchmark versions, "No VRAM spill up to …", every cliff message, and the estimated-quality notice.

## 9. Candidate generation (`src/core/benchmark/candidates.ts`)

Estimates are `kind:'estimated'`, prune only, and never rank.
- **`estimateMemory`:**
  - KV = ctx·L·Hkv·(dk+dv)·bytes. This is exact vs the llama-server log: 8B 0.125 MiB/token, Qwen2.5-1.5B 0.0273 MiB/token.
  - VRAM = file·min(ngl,L)/L + KV + compute, where compute = 32 MiB + ubatch·n_embd·32 B + 1 KiB·ctx (fit to the measured 61–164 MiB compute buffers).
  - RAM = **whole file (mmap resident)** + KV if ngl=0 + 512 MiB. Available RAM fell by ≈ file size even at ngl 99. This makes the pre-check conservative, and spill detection never uses it: spill uses the private working set and per-PID shared memory.
  - Estimated vs measured on 8B: −2 % … +3 % per rung.
- **Budgets:**
  - VRAM = total − in-use − **1 GiB**. The margin absorbs the measured residual of 0.2–0.9 GiB, largest at 64K.
  - RAM = available (else total) − 4 GiB.
  - RAM over budget → the step is skipped (never kept).
  - VRAM over budget → the first such step is kept once, to observe the cliff, but only if est ≤ 1.15 × budget (`keepOverVramMaxRatio`). Every later step is skipped.
- **Order:**
  1. ngl=all f16.
  2. ngl=all q8_0 KV, when targetContext ≥ 32K and declared ≥ 32K.
  3. If (1) is rejected: partial ngl = ⌊L·f⌋ for f ∈ .75/.5/.25, keeping the largest that fits plus the next lower one.
  4. ngl=0 only when there is no GPU, or no GPU config fits (`cpuOnlyMaxParams` = 0 = off). On a GPU machine it carries the note "treated as partial offload".
- **Other rules:**
  - Max 4 per model.
  - Threads = physical cores.
  - fa = on.
  - VRAM total unavailable → no VRAM pruning (with a note).
  - SWA/hybrid/recurrent archs prune on weights only.

## 10. Export (`src/core/export/config.ts`)
- `exportConfigFrom(rec, cand, model, sessionId)` takes the winner at `recommendedCtx`.
- `toLlamaServerArgs/Command` reproduce the measured launch: `-dev`, `-fit off`, `-c`, `-ngl 999|n`, `-t`, `-b 2048 -ub 512`, `-fa`, `-ctk/-ctv` for q8_0, `--parallel 1`.
- `toOllamaModelfile(c, {from})` emits FROM plus num_ctx / num_gpu / num_thread / num_batch, with flash attention and KV type as env-var comments.
- `toLmStudioSettings` uses lmstudio-js keys.
- `toJson` bundles every format, and `provenanceNote(rec)` says which inputs were measured, estimated or unavailable.
- The Ollama and LM Studio outputs are unverified, because neither app is installed.

## 11. Known limitations
See `docs/LIMITATIONS.md`.
