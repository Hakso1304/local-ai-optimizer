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
