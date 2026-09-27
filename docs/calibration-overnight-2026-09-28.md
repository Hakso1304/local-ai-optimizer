# Overnight calibration — 2026-09-28

Host: Ryzen 9800X3D, 31 GB RAM, RX 9070 XT 16 GB, Windows 11 26200, llama.cpp b11208 Vulkan. Harness: scripts/run-session.ts
(runSession end-to-end, persisted to the app DB `%APPDATA%\local-ai-optimizer-dev\optimizer.db`), RAM watchdog 4 GiB.
Every stage runs from a git-archive snapshot; HEAD is given per stage.

## Stage 1 — session 3 resume (coding, heavy; 14B / Qwen3.8-27B / Gemma-4-26B-A4B), HEAD 3e7183f
Conditions:
- 01:13–01:28 local; vramInUse at planning 1.13 GiB.
- Resume reused every persisted step. Measured anew: Qwen3.8 + Gemma quality (qb-2.0.0, thinking off, T=0) and the Gemma ngl 19 ladder.

Recommendation: **Qwen2.5-14B ngl all f16 @16K**: 54.4 t/s decode, TTFT 6.4 s (9,152 prompt tokens), Q 86 [70, 94].
- 32K: guard_abort; the row now stores the tripping value (3.0 GiB).
- Recommended `-c` 16K (I-3.7).

Quality, qb-2.0.0 (n = 32 scored items for the Coding weights; 1 sample):

| Model | Q | Band |
|---|---|---|
| Qwen3.8-27B | 96 | [83, 99] |
| Gemma-4 | 93 | [79, 98] |
| 14B | 86 | [70, 94] |

- Qwen3.8 ngl 49 is only provisional. Its speed is unmatched at the 16K scoring rung because it spilled 0.50 GiB at 16K. ngl 45 reaches 32K clean but decodes at 8.3 t/s at 16K.

Gemma-4 ngl 19, new:

| Context | Decode | TTFT | VRAM |
|---|---|---|---|
| 4K | 31.0 t/s | 2.1 s | 10.52 GiB |
| 8K | 29.5 t/s | 5.3 s | 10.57 GiB |
| 16K | 26.9 t/s | 15.9 s | 10.72 GiB |
| 32K | 17.9 t/s | 54.7 s | 10.99 GiB |

- No spill at any context. At 32K the rep spread is 44 % (13.9 / 21.8 t/s, I-6.1).

Gemma-4 ngl 23 spills at 4K (0.70 GiB), as before.

Host RAM, `--cache-ram 0` check:
- Minimum RAM available across the quality phases and the Gemma ladder: **8.01 GiB**. Before b3e671f, the same quality phase tripped the 4 GiB and 5 GiB watchdogs, with the server at ≈13.6 GiB.
- `prompt cache is enabled` was not seen in the server log tail.
- Caveat: the per-phase split is not available for this run. The phase flag was sticky; fixed in run-session.
