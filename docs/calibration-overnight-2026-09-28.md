# Overnight calibration — 2026-09-28

Host: Ryzen 9800X3D, 31 GB RAM, RX 9070 XT 16 GB, Windows 11 26200, llama.cpp b11208 Vulkan. Harness: scripts/run-session.ts
(runSession end-to-end, persisted to the app DB `%APPDATA%\local-ai-optimizer-dev\optimizer.db`), RAM watchdog 4 GiB.
Every stage runs from a git-archive snapshot; HEAD is given per stage.

## Stage 1 — session 3 resume (coding, heavy; 14B / Qwen3.8-27B / Gemma-4-26B-A4B), HEAD 3e7183f
Conditions:
- 01:13–01:28 local; vramInUse at planning 1.13 GiB.
- Resume reused every persisted step. Measured anew: Qwen3.8 + Gemma quality (qb-2.0.0, thinking off, T=0) and the Gemma ngl 19 ladder.

Historical stored recommendation (not a validated final ranking while O1-affected alternatives remain unmeasured): **Qwen2.5-14B ngl all f16 @16K**: 54.4 t/s decode, TTFT 6.4 s (9,152 prompt tokens), Q 86 [70, 94].
- 32K: `guard_abort`. Session 3 run 18's guard reason reports a 3.0 GiB residual and its raw per-PID shared peak is 3.05 GiB, but persisted `peakSharedGpuBytes.value` is **0**. The guard reason and stored adjusted metric disagree; do not use this row as a qualified spill or budget observation.
- Historical stored `-c` 16K verdict (I-3.7); remeasure affected alternatives before treating it as the final choice.

Quality, qb-2.0.0 (n = 32 scored items for the Coding weights; 1 sample):

| Model | Q | Band |
|---|---|---|
| Qwen3.8-27B | 96 | [83, 99] |
| Gemma-4 | 93 | [79, 98] |
| 14B | 86 | [70, 94] |

- Qwen3.8 ngl 49 is only provisional under the stored verdict: run 23 reports 0.50 GiB adjusted shared usage at 16K, but O1 leaves the spill attribution unvalidated. Its 16K speed is therefore not used as a confirmed comparable rung. ngl 45 has a pass status and zero reported adjusted reading through 32K; it decodes at 8.3 t/s at 16K.

Gemma-4 ngl 19, new:

| Context | Decode | TTFT | VRAM |
|---|---|---|---|
| 4K | 31.0 t/s | 2.1 s | 10.52 GiB |
| 8K | 29.5 t/s | 5.3 s | 10.57 GiB |
| 16K | 26.9 t/s | 15.9 s | 10.72 GiB |
| 32K | 17.9 t/s | 54.7 s | 10.99 GiB |

- Pass status and zero reported adjusted reading at each context. At 32K the rep spread is 44 % (13.9 / 21.8 t/s, I-6.1).

Gemma-4 ngl 23 run 19 stores 0.70 GiB adjusted shared usage at 4K; the first-rung spill attribution is unvalidated under O1.

Host RAM, `--cache-ram 0` check:
- Minimum RAM available across the quality phases and the Gemma ladder: **8.01 GiB**. Before b3e671f, the same quality phase tripped the 4 GiB and 5 GiB watchdogs, with the server at ≈13.6 GiB.
- `prompt cache is enabled` was not seen in the server log tail.
- Caveat: the per-phase split is not available for this run. The phase flag was sticky; fixed in run-session.

### Validation scope (RECHECK4, docs/review-w4o-2026-09-28.md O1–O5)
- **O1 (benign first-rung baseline ordering) affects stage 1's spill verdicts. The source fix has landed, but these old rows remain UNVALIDATED until remeasured with the O1 fix:**
  - Gemma-4 ngl 23: "spilled 0.70 GiB at 4K" is its first rung, with a residual below the 1 GiB benign bound.
  - Qwen3.8 ngl 49: "0.50 GiB at 16K".
  - The 14B 32K guard reason (3.0 GiB) is above that benign bound, but run 18 persists adjusted spill as zero despite 3.05 GiB raw shared usage. Keep the guard outcome and the metric mismatch separate; neither establishes a qualified capacity budget.
- **O2/O3/O5 (backend fallback, needle routing, CUDA identity) are not in this path.** The harness runs the runner with one Vulkan backend: deps.backendKind 'vulkan', no `backends` list. All stored configs are Vulkan ids, so no HIP/CUDA config exists to fall back from.
- **O4 (unified-memory env casing): no variable matching /unified/i exists** in the process, User or Machine environment (checked 01:40). The inherited env passed to llama-server cannot carry it.
- **Pinned configs (`--pin`, from stage 3 on) keep the planner's estimates for the cloned base config**, and say so in their notes. Their measured rows are experiment evidence. Their planning snapshot (I-2.3/I-4.1 texts) is not planner calibration.

## Stage 2 — VRAM A/B partial (A1/A2 only), HEAD b868e3a68d123849ab2cdacd32a2a034695fb75b

Status: **INCOMPLETE**. The safety harness stopped during the 120-second idle before B1a because `llama-server.exe` appeared. Its exact error line was `Error: llama-server appeared during idle; stop this experiment` at `scripts/ab-spill.ts:233`. The partial artifact was written at **2026-09-28 01:55:57.840 KST**, immediately after detection; the harness did not separately timestamp the detection event. An immediate process check found no server or harness process; the transient process's PID and owner were not captured. A1/A2 server PIDs were 26660/22572, and the harness tree was 28676 → 37640 → 23336 → 35168. B1a/B1b/B2 did not run. Do not infer placement causality, an allocation-pattern ceiling, a qualified budget [I-4.0], or an O1/I-2.8 validation from this partial A/B. The untouched partial result is `docs/ab-spill-2026-09-28.json` (SHA-256 `E55E7ABB13A3CD5A3A91AB2672DAFAE3C50861D55ECEA3AA0678410D37972B0E`).

Subsequent timing review found that S2's full suite was running an Electron E2E runtime scan from 01:55:13–01:56:12 KST; that path invokes `llama-server.exe --version`. It plausibly explains the brief name collision at 01:55:57, but no captured PID/path ties that invocation to the guard event. Treat the owner as unproven. Future full/E2E/live telemetry tests require the GPU lane to be idle; scoped fake tests can run during a lease.

Conditions: fresh `git archive` of the full HEAD above; `scripts/ab-spill.ts` SHA-256 `A0CFA25122D5D333E0AEBC31F38FC03908F99B22CE82BF84705E29B989CC6281`; RX 9070 XT Vulkan0, 15.92 GiB adapter total; initial measured adapter in-use 1.12 GiB; initial available host RAM 17.39 GiB. No `llama-server.exe`, `typeperf.exe`, or measurement harness was present at launch, and no spelling of `GGML_CUDA_ENABLE_UNIFIED_MEMORY` existed in Process/User/Machine environments. The harness used a 4 GiB live RAM floor, 300,000 ms request cap, bounded 120-second load, safe child environment and verified child exit. Model: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, 4,920,739,232 bytes. Common argv: `-c 65536 -ngl 999 -dev Vulkan0 -t 8 -b 2048 -ub 512 -fa on -fit off --parallel 1`; `-lm none` was **not** set, so mmap mode was not constrained like the app runner. Do not compare these peaks directly with app learned budgets.

| Case / server PID | Prompt requested / actual | Decode reps (t/s) | Prefill reps (ms) | Peak PID dedicated / raw shared (GiB) | Peak adapter dedicated (GiB) | Min RAM available (GiB) | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 / 26660 | 36,572 / 36,559 | 36.89 / 37.44 | 21,071 / 20,975 | 11.60 / 1.08 | 12.76 | 11.56 | 73 |
| A2 / 22572 | 49,152 / 49,169 | 30.05 / 30.03 | 32,798 / 33,207 | 11.60 / 1.08 | 12.75 | 10.61 | 110 |

Both completed with `error=null` and `ramAbort=null`. The JSON property named `ttftMs` is the runtime's `timings.prompt_ms` (prefill duration), **not** measured client TTFT; client TTFT is unavailable. Per-PID raw shared was 1.08 GiB at the first recorded sample and at peak in both cases; no >256 MiB growth onset was detected. This is raw residency only, not a validated adjusted spill or benign baseline. `buffersMiB` and `offloadLines` are empty, `largestBufferMiB: 0` is a harness placeholder meaning **UNAVAILABLE**, and `vulkaninfoHeaps` is empty; no heap/largest-buffer evidence was collected. The snapshot listened to stderr but did not retain raw child logs or drain stdout, so the missing buffer evidence cannot be reconstructed from this run.

## Repaired stage 2 setup stop, HEAD 3d12db19b91e32c588464d1c12ef06bc9081f7d0

The source archive passed the software gate (TypeScript and 16 fake tests), but `git archive` omitted the ignored `vendor/llama.cpp` runtime. After a clean process/env preflight and the initial 120-second idle check, A1's bounded `--list-devices` probe failed with `spawnSync vendor/llama.cpp/llama-server.exe ENOENT`. No server or model launched, and no A/B measurement was made. The harness stopped with exit code 1 and verified zero server, typeperf, and harness processes afterward. The exclusive output was preserved as `docs/ab-spill-repaired-setup-failure-2026-09-28.json` (SHA-256 `35242F0D02364A445AAB914D7EB4E60D6058CCD24A25AB9A3FECE2A4CEA77E1F`); its one error row has 0 samples, `promptTokensActual=null`, and minimum available RAM 17.28 GiB. The clean preflight at 02:11:16 KST observed 17.22 GiB RAM and 1.12 GiB RX 9070 XT dedicated use; no unified-memory environment key was present. The source ZIP SHA-256 was `11DEFC9632A752430F106CBC8ACE250E580DB24BB3FABE7A5A24EFCD4462E650` and script SHA-256 was `2FFA2B5385D4BF40393F1DA3FA1D1CDC8ED03883B292D5040FBDD0E5BB02A5A8`. This is setup failure evidence, not a case result. GPU remains on hold pending a complete pinned runtime bundle and a new release.

## Repaired stage 2 VRAM A/B, source HEAD 3d12db19b91e32c588464d1c12ef06bc9081f7d0

Status: **five cases complete as standalone raw residency and timing evidence**. Astra authorized one retry with the same git-archive source ZIP (SHA-256 `11DEFC9632A752430F106CBC8ACE250E580DB24BB3FABE7A5A24EFCD4462E650`) and a validated junction to the ignored `vendor/llama.cpp` runtime. The script SHA-256 remained `2FFA2B5385D4BF40393F1DA3FA1D1CDC8ED03883B292D5040FBDD0E5BB02A5A8`. The runtime release tag was `b11208`, executable SHA-256 `352D52FBCCDF88CEB094421B9A4A0B8C56B9354CF9D10F9B82AB21828F0876FA`; hashes of all 53 vendor files, junction targets, model path/size, and preflight are in `docs/ab-spill-repaired-runtime-2026-09-28.json` (committed SHA-256 `8BCEB3B10AF79E7BD59487EB37E23A1D56873AFBA213B2ECC4CA9C1B43A6C1AF`). The immutable result is `docs/ab-spill-repaired-2026-09-28.json` (49,311 bytes; SHA-256 `EBC5CAC32421EE8ADF1EB65AF99FD936D3E087BB66AC08CC81116AFABED3DB8B`). The original A1/A2 partial and the ENOENT setup artifact remain separate.

At 02:15:30 KST, preflight found zero server/typeperf/harness processes and zero case-insensitive `GGML_CUDA_ENABLE_UNIFIED_MEMORY` keys in Process/User/Machine environments; RAM available was 17.34 GiB and RX 9070 XT dedicated use was 1.11 GiB. The model was 4,920,739,232 bytes. Every child used the safe environment and hidden window; requests were capped at 300,000 ms, RAM was guarded at 4 GiB during load/request/idle, and owned children were reaped before the next case. The full run exited 0 with `stopReason=null`, no case errors or RAM aborts, and zero server/typeperf/harness processes at 02:30:20 KST (RAM 17.35 GiB). Common server flags included `-lm none --cache-ram 0`; B1a added q8_0 K/V and 128K context, while B2 used `-ub 256` instead of 512.

| Case | Requested / actual prompt tokens | Decode reps (t/s) | Server prefill reps (ms) | Request wall reps (ms) | PID dedicated / raw shared baseline=peak (GiB) | Adapter dedicated peak (GiB) | Min RAM (GiB) | Samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 36,572 / 36,559 | 37.13 / 37.29 | 20,842 / 20,774 | 24,276 / 24,190 | 11.60 / 1.36 | 12.75 | 15.44 | 73 |
| A2 | 49,152 / 49,169 | 30.06 / 29.99 | 32,482 / 32,246 | 36,722 / 36,515 | 11.60 / 1.36 | 12.75 | 15.52 | 108 |
| B1a load only | none | none | none | none | 11.27 / 2.28 | 12.38 | 14.74 | 3 |
| B1b after B1a | 36,572 / 36,559 | 36.97 / 37.00 | 20,930 / 20,659 | 24,375 / 24,105 | 11.60 / 1.36 | 12.75 | 15.57 | 72 |
| B2 `-ub 256` | 36,572 / 36,559 | 37.44 / 37.42 | 21,335 / 21,396 | 24,764 / 24,802 | 11.51 / 1.32 | 12.66 | 15.62 | 74 |

`prefillMs` comes from llama.cpp `timings.prompt_ms`, and `requestWallMs` is the client's non-streaming request duration; `clientTtftMs` is null for every repetition. No prompt-cache event was observed in the retained logs; the request payload sets `cache_prompt:false`, and the logs warn that `--cache-idle-slots` was disabled with `--cache-ram 0`. The raw shared baseline equalled peak in each case, so the >256 MiB growth heuristic reported `spillOnset=null`; this does **not** validate adjusted spill attribution or a clean budget baseline [RECHECK4 O1/I-2.8].

Both streams were captured without truncation. Stdout had 0 bytes in all five cases; stderr had 9,347 / 12,328 / 792 / 9,199 / 9,498 UTF-8 bytes for A1/A2/B1a/B1b/B2. Representative retained load lines include `verbosity = 3`, `load_model: loading model`, and `llama_server: model loaded`. No stream contains a buffer allocation declaration or offload line, so `buffersMiB` is empty and `largestBufferMiB=null` (unavailable) for all cases; `vulkaninfoHeaps` is also empty. The app runner explicitly passes `-lv 4` to emit device/offload detail, while this A/B base argv omitted it. That instrumentation difference is a likely explanation, not retroactive allocation evidence. There is no supported allocation-pattern comparison, learned capacity budget, final placement ranking, or O1 validation from this batch. Do not substitute a separate load's declarations for these attempts.
