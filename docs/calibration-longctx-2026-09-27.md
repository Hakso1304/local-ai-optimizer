# Long-context calibration 2026-09-27 (Llama-3.1-8B Q4_K_M, RX 9070 XT 16 GB, llama.cpp b11208 Vulkan)

Run: `npx tsx scripts/run-session.ts H --workload large_coding --required-ctx 131072 --models Meta-Llama-3.1-8B-Instruct-Q4_K_M --ladder 32768,65536,131072 --ram-abort-gib 3`. This is runSession with requiredContext = 131072; the planner produced the variants.

- The runner also measured the lower rungs 2K–16K; `--ladder` did not restrict it.
- Prompt fill was ≈ 0.56 · ctx (e.g. 36,572 tokens at 64K).
- Dump: `session-run-H-large_coding-2026-09-27T13-32-49-071Z.json`, incremental. All 18 ladder steps are in it.
- The quality phase and the recommendation are missing. Claude Code's memory-pressure reaper stopped the job after the last ladder step: the host had three other agents and Electron builds running.
- 14B is not included: it declares 32K, so it cannot test 64K/128K.
- VRAM budget this run was 13.4 GiB (≈ 2.5 GiB held by other processes), not the 14.9 GiB of earlier runs.

## Variants and rungs (warm median of 2 reps)

| variant | ctx | prompt_n | prefill TPS | decode TPS | TTFT s | per-PID ded GiB | shared raw GiB | shared (corrected) | KV buffer MiB | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| q8_0 KV | 16K | 9153 | 2529 | 89.7 | 3.6 | 5.64 | 0.03 | 0.00 | 1088 | pass |
| q8_0 KV | 32K | 18248 | 2166 | 84.8 | 8.4 | 6.75 | 0.05 | 0.00 | 2176 | pass |
| q8_0 KV | 64K | 36572 | 1678 | **68.5** | 21.8 | 8.98 | 0.08 | 0.00 | 4352 | pass |
| q8_0 KV | 128K | 73252 | 1120 | **24.2** | 65.4 | 11.59 | **2.00** | 0.00 | 8704 | degraded (decode −65%) |
| f16 KV | 16K | 9153 | 2560 | 91.6 | 3.6 | 6.55 | 0.03 | 0.00 | 2048 | pass |
| f16 KV | 32K | 18248 | 2205 | **76.5** | 8.3 | 8.56 | 0.05 | 0.00 | 4096 | pass |
| f16 KV | 64K | 36572 | 1599 | **25.0** | 22.9 | 10.60 | **2.08** | 0.00 | 8192 | degraded (decode −67%) |
| f16 KV | 128K | — | — | — | — | — | — | — | — | pruned: est. VRAM 20.8 GiB > 13.4 GiB |
| -nkvo (KV in RAM) | 2K | 1168 | 2486 | 30.9 | 0.5 | 4.51 | — | 0.00 | CPU 256 | pass |
| -nkvo | 8K | 4593 | 2458 | 18.2 | 1.9 | 4.51 | — | 0.00 | CPU 1024 | pass |
| -nkvo | 16K | 9153 | 2185 | 12.0 | 4.2 | 4.52 | 0.03 | 0.00 | CPU 2048 | pass |
| -nkvo | 32K | 18248 | 1782 | **7.3** | 10.2 | 4.58 | 0.05 | 0.00 | CPU 4096 | pass |
| -nkvo | 64K / 128K | — | — | — | — | — | — | — | — | skipped_memory (est. RAM 8.5 / 16.5 GiB > available − reserve 8.3 GiB) |

For reference, 2K–8K per variant: q8_0 106 / 103 / 101 t/s; f16 109 / 106 / 100 t/s.

## Findings

- **Largest clean rung per variant.** q8_0 KV reaches 64K at 68.5 t/s (TTFT 21.8 s). f16 KV reaches 32K at 76.5 t/s. -nkvo reaches 32K but at 7.3 t/s, so it is unusable for coding.
- **128K.** Only q8_0 KV reaches it, and it degrades: decode 24.2 t/s, TTFT 65 s for 73K prompt tokens, and a 2.0 GiB spill into shared GPU memory. That is usable within large_coding's 180 s latency tolerance, but at a third of the 64K speed. With an idle GPU (budget ≈ 14.7 GiB) q8_0 at 128K should fit without spill: ded 11.6 + 2.0 spilled ≈ 13.6 GiB.
- **f16 at 64K collapsed here but not before.** Earlier today it ran 60 t/s at 12.6 GiB. This time other processes held ≈ 2.5 GiB of VRAM, so the same config spilled 2.08 GiB and fell to 25 t/s.
- **DEFECT (spill gate).** The corrected spill metric (2d51cc0) counts spill only when per-PID dedicated ≥ 80% of total VRAM. Both collapsing rungs sat at 67–73% dedicated because of other processes, and showed 2.0–2.1 GiB raw shared yet 0.00 corrected. The cliff was caught only by the decode_drop rule. Fix: gate against the effective budget (VRAM − other-process usage measured at load), or count raw-shared growth ≥ 1 GiB vs the previous rung regardless of the dedicated share.
- **-nkvo decode.** It roughly halves per context doubling (30.9 → 25.1 → 18.2 → 12.0 → 7.3), and it needs system RAM ≈ the KV size (4 GiB at 32K). The RAM guard correctly skipped 64K and 128K on this loaded host.
- **Recommendation for 128K coding on 16 GB.** Use q8_0 KV. Plan on 64K as the fast ceiling (68 t/s) and treat 128K as possible but slow (24 t/s, ~1 min to first token), and only with a mostly free GPU.
