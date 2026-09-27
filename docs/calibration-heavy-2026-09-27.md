# Heavy-mode real run 2026-09-27 (RX 9070 XT 16 GB, Ryzen 7 9800X3D, 31 GB RAM, llama.cpp b11208 Vulkan)

Command: `npx tsx scripts/run-session.ts H --heavy --workload max_quality --models Qwen3.8-27B-UD-Q4_K_M,gemma-4-26B-A4B-it-UD-Q4_K_M,Meta-Llama-3.1-8B-Instruct-Q4_K_M --ladder 2048,4096,8192,16384,32768`.
Ladder capped at 32K by choice (uncapped CPU / -nkvo ladders to 64K–128K on 27B would take hours).
**Run incomplete:** at 422 s Claude Code's memory-pressure reaper stopped the job (system RAM critically low); the orphaned runner/servers were killed by hand, so the final JSON dump never ran. Numbers below are the runner's own `step:done` events from the console log (`calibration-heavy-2026-09-27.console.txt`). Gemma-4 and the Coding workload were not reached.

## GGUF metadata (readGgufMetadata, all DECLARED)

| model | arch | layers | params | ctx_train | heads / kv | head dim k/v | vocab | SWA | KV/token (f16, all layers) | file |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-UD-Q4_K_M | qwen35 | 65 | 27.32B | 262144 | 24 / 4 | 256 / 256 | 248320 | — | 260 KiB (formula; hybrid linear-attn arch → over-estimate) | 16,464,440,224 B |
| gemma-4-26B-A4B-it-UD-Q4_K_M | gemma4 (MoE 26B/4B active) | 30 | 25.23B | 262144 | 16 / 16 | 512 / 512 | 262144 | 1024 | 960 KiB (formula; SWA → low-confidence, not used for pruning) | 16,947,541,728 B |

Both load with b11208 (`-ngl 0 --device none -c 2048`: model loaded + listening in 9 s / 11 s); no runtime upgrade needed.
llama-server log for Qwen3.8: "chat template supports preserving reasoning, it is enabled by default" (thinking model).

## Candidates (generateCandidates, this machine, max_quality)

- heavyMode=false: 0 candidates for both big models; rejected "est. VRAM 16.0/16.8 GiB > budget 14.9 GiB at 2K; full GPU offload does not fit — enable heavy-model mode". The stored Recommendation says only "No recommendation: no candidates were benchmarked" (hint is log-only).
- heavyMode=true (session plan, 9 total): Llama 8B full; Qwen3.8 ngl 0 (CPU), 55, 60, 62-nkvo; Gemma-4 ngl 27, 28-nkvo, 0 (CPU). Plan order (est. VRAM+RAM asc.) put Qwen3.8's CPU baseline first.

## Steps measured (warm median of 2 reps; VRAM/shared = per-PID peak)

| config | ctx | status | prefill TPS | decode TPS | TTFT ms | pid VRAM GiB | pid shared GiB |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B ngl all | 2048 | pass | 2932 | 110.5 | 401 | 4.78 | 0.02 |
| Llama-3.1-8B ngl all | 4096 | pass | 2974 | 107.7 | 780 | 5.03 | 0.02 |
| Llama-3.1-8B ngl all | 8192 | pass | 2857 | 96.5 | 1617 | 5.54 | 0.02 |
| Llama-3.1-8B ngl all | 16384 | pass | 2593 | 93.2 | 3534 | 6.55 | 0.03 |
| Llama-3.1-8B ngl all | 32768 | pass | 2230 | 78.9 | 8189 | 8.56 | 0.05 |
| Qwen3.8-27B ngl 0 (CPU) | 2048 | fail / guard_abort | — | — | — | 0.01 | 0.00 |
| Qwen3.8-27B ngl 55/65 | 2048 | pass | 611 | 12.4 | 1916 | 12.67 | 0.04 |
| Qwen3.8-27B ngl 55/65 | 4096 | pass | 661 | 13.0 | 3479 | 12.81 | 0.04 |
| Qwen3.8-27B ngl 55/65 | 8192 | pass | 689 | 13.0 | 6678 | 13.03 | 0.04 |

Quality: 8B suite ran (75.7 → 94.5 s); Qwen3.8 ngl 55 suite ran 235.5 → 422.4 s (~3 min at 13 t/s with thinking). Results were in the in-memory store only and are lost with the killed job.

## Notes

- RAM guard on the 16 GB mmap: the Qwen3.8 CPU baseline passed the pre-check (resident estimate ≈ 16 GiB vs live available ≈ 18 GiB − heavy floor 2 GiB) and then drove system RAM available to **1.0 GiB** before the in-step guard (1 s poll) aborted it. That episode is what triggered the memory-pressure stop. The partial-offload configs stayed at 12.7–13.0 GiB VRAM with ~0.04 GiB shared — no spill.
- Partial offload 55/65 layers of a dense 27B: decode 12.4–13.0 t/s (vs 8B full offload 97–110), prefill 611–689 t/s; flat across 2K→8K, so the CPU-side layers set the rate, not context.
- MoE (Gemma-4-26B-A4B) behaviour was not observed (not reached).
