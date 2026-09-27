# Calibration 2026-09-27 (RX 9070 XT 16 GB, Ryzen 7 9800X3D, 31 GB RAM, llama.cpp b11208 Vulkan)

Prompt = generateFiller(seed 1) tokenized to ≈0.75·ctx, n_predict 128, temp 0, seed 1, cache_prompt false. 1 warmup (same prompt) + 2 measured.
Telemetry: typeperf 1 s; adapter luid 0x00000000_0x00016058; shared Δ = peak adapter Shared Usage − idle baseline before that config; GPU util = mean over pid's 3D/Compute engine groups (max group).

## Per config

| model | ctx | ngl | status | load ms | layers | model MiB | KV MiB | compute MiB | idle ded GiB | idle shared GiB | idle RAM GiB | load peak ded GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct-q4_k_m | 2048 | 99 | ok | 1404.7 | 29/29 | CPU_Mapped:125 Vulkan0:935 | Vulkan0:56 | Vulkan0:61 Vulkan_Host:8 | 1.15 | 0.80 | 17.35 | n/a |
| qwen2.5-1.5b-instruct-q4_k_m | 8192 | 99 | ok | 964.4 | 29/29 | CPU_Mapped:125 Vulkan0:935 | Vulkan0:224 | Vulkan0:67 Vulkan_Host:14 | 1.16 | 0.80 | 17.31 | n/a |
| qwen2.5-1.5b-instruct-q4_k_m | 32768 | 99 | ok | 1069.4 | 29/29 | CPU_Mapped:125 Vulkan0:935 | Vulkan0:896 | Vulkan0:91 Vulkan_Host:38 | 1.18 | 0.79 | 17.38 | n/a |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 2048 | 99 | ok | 3443.3 | 33/33 | CPU_Mapped:282 Vulkan0:4403 | Vulkan0:256 | Vulkan0:102 Vulkan_Host:18 | 1.21 | 0.80 | 17.00 | 5.73 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 4096 | 99 | ok | 2561.3 | 33/33 | CPU_Mapped:282 Vulkan0:4403 | Vulkan0:512 | Vulkan0:104 Vulkan_Host:20 | 1.42 | 0.80 | 16.07 | n/a |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 99 | ok | 2664.0 | 33/33 | CPU_Mapped:282 Vulkan0:4403 | Vulkan0:1024 | Vulkan0:108 Vulkan_Host:24 | 1.16 | 0.80 | 17.21 | n/a |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 16384 | 99 | ok | 2779.6 | 33/33 | CPU_Mapped:282 Vulkan0:4403 | Vulkan0:2048 | Vulkan0:116 Vulkan_Host:32 | 1.16 | 0.80 | 17.38 | 5.55 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 32768 | 99 | ok | 3095.4 | 33/33 | CPU_Mapped:282 Vulkan0:4403 | Vulkan0:4096 | Vulkan0:132 Vulkan_Host:48 | 1.15 | 0.80 | 17.35 | 5.53 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 65536 | 99 | ok | 4594.1 | 33/33 | CPU_Mapped:282 Vulkan0:4403 | Vulkan0:8192 | Vulkan0:164 Vulkan_Host:80 | 1.33 | 0.80 | 17.15 | 6.03 |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 20 | ok | 2460.9 | 20/33 | CPU_Mapped:2816 Vulkan0:2789 | CPU:416 Vulkan0:608 | Vulkan0:136 Vulkan_Host:24 | 1.28 | 0.81 | 16.84 | n/a |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 0 | ok | 1814.4 | 0/33 | CPU_Mapped:4685 | CPU:1024 | Vulkan0:136 Vulkan_Host:24 | 1.47 | 0.82 | 16.33 | n/a |

## Per request

| model | ctx | ngl | phase | prompt_n | TTFT ms | prefill TPS | decode TPS | prefill ms | decode ms | total ms | total−(pp+tg) ms | peak ded GiB | shared Δ GiB | RAM min GiB | GPU util % | CPU avg % | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct-q4_k_m | 2048 | 99 | warmup | 1537 | 685.8 | 2300.1 | 361.8 | 668.2 | 351.0 | 1036.7 | 17.5 | 2.25 | 0.01 | 16.22 | n/a | 25.3 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 2048 | 99 | run1 | 1537 | 121.2 | 12991.4 | 365.1 | 118.3 | 347.9 | 469.3 | 3.1 | n/a | n/a | n/a | 0.0 | n/a |  |
| qwen2.5-1.5b-instruct-q4_k_m | 2048 | 99 | run2 | 1537 | 124.1 | 12735.4 | 367.3 | 120.7 | 345.8 | 470.0 | 3.6 | 2.29 | 0.01 | 16.19 | n/a | 13.2 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 8192 | 99 | warmup | 6151 | 1037.7 | 6049.7 | 328.3 | 1016.7 | 386.8 | 1425.0 | 21.4 | 2.45 | 0.01 | 16.17 | 8.0 | 18.7 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 8192 | 99 | run1 | 6151 | 532.0 | 11719.6 | 332.9 | 524.8 | 381.5 | 913.8 | 7.5 | 2.45 | 0.01 | 16.14 | 49.2 | 13.4 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 8192 | 99 | run2 | 6151 | 529.2 | 11768.3 | 331.7 | 522.7 | 382.9 | 912.3 | 6.7 | 2.47 | 0.01 | 16.21 | 92.4 | 25.2 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 32768 | 99 | warmup | 24588 | 3487.9 | 7122.8 | 256.2 | 3452.0 | 495.7 | 3984.6 | 36.8 | 3.16 | 0.04 | 15.95 | 70.4 | 33.3 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 32768 | 99 | run1 | 24588 | 3192.9 | 7753.3 | 257.6 | 3171.3 | 493.0 | 3686.8 | 22.5 | 3.16 | 0.04 | 16.15 | 89.7 | 19.3 |  |
| qwen2.5-1.5b-instruct-q4_k_m | 32768 | 99 | run2 | 24588 | 3198.5 | 7740.1 | 256.7 | 3176.7 | 494.8 | 3694.3 | 22.8 | 3.16 | 0.04 | 16.10 | 96.5 | 23.3 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 2048 | 99 | warmup | 1538 | 531.4 | 2984.4 | 109.5 | 515.3 | 1159.5 | 1691.1 | 16.2 | 6.25 | 0.02 | 11.69 | 16.7 | 28.8 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 2048 | 99 | run1 | 1538 | 525.2 | 2943.1 | 109.2 | 522.6 | 1162.9 | 1688.3 | 2.8 | 6.20 | 0.02 | 11.64 | 92.3 | 28.9 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 2048 | 99 | run2 | 1538 | 520.0 | 2971.0 | 109.7 | 517.7 | 1157.5 | 1677.8 | 2.7 | 6.26 | 0.02 | 11.66 | 91.6 | 28.9 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 4096 | 99 | warmup | 3067 | 1066.6 | 2921.3 | 105.2 | 1049.9 | 1207.6 | 2274.5 | 17.0 | 6.53 | 0.02 | 11.57 | 56.2 | 27.5 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 4096 | 99 | run1 | 3067 | 1054.9 | 2916.0 | 104.6 | 1051.8 | 1214.1 | 2269.4 | 3.6 | 6.50 | 0.02 | 11.54 | 90.3 | 27.5 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 4096 | 99 | run2 | 3067 | 1056.7 | 2911.0 | 106.2 | 1053.6 | 1196.1 | 2252.9 | 3.2 | 6.50 | 0.02 | 11.55 | 91.5 | 29.1 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 99 | warmup | 6152 | 2773.4 | 2232.8 | 99.7 | 2755.2 | 1273.8 | 4047.7 | 18.7 | 6.69 | 0.02 | 12.73 | 66.7 | 16.2 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 99 | run1 | 6152 | 2246.8 | 2742.3 | 99.9 | 2243.4 | 1270.7 | 3517.8 | 3.7 | 6.70 | 0.02 | 12.79 | 80.2 | 16.2 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 99 | run2 | 6152 | 2222.1 | 2772.4 | 99.8 | 2219.0 | 1272.9 | 3495.3 | 3.4 | 6.70 | 0.02 | 12.81 | 95.7 | 14.5 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 16384 | 99 | warmup | 12289 | 4982.5 | 2475.6 | 88.2 | 4964.0 | 1439.8 | 6422.9 | 19.1 | 7.67 | 0.03 | 12.58 | 81.2 | 23.6 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 16384 | 99 | run1 | 12289 | 4925.3 | 2497.5 | 88.0 | 4920.5 | 1443.6 | 6369.5 | 5.4 | 7.70 | 0.03 | 12.76 | 94.3 | 14.6 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 16384 | 99 | run2 | 12289 | 4874.8 | 2523.3 | 88.2 | 4870.2 | 1439.7 | 6315.1 | 5.1 | 7.70 | 0.03 | 12.72 | 95.7 | 14.0 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 32768 | 99 | warmup | 24589 | 12063.3 | 2042.0 | 71.9 | 12041.4 | 1766.0 | 13830.7 | 23.4 | 9.75 | 0.05 | 12.68 | 87.1 | 16.8 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 32768 | 99 | run1 | 24589 | 12070.4 | 2038.5 | 72.0 | 12062.3 | 1764.7 | 13836.3 | 9.3 | 9.72 | 0.05 | 13.16 | 13099175457793.5 | 16.3 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 32768 | 99 | run2 | 24589 | 11990.2 | 2052.1 | 71.9 | 11982.1 | 1767.1 | 13758.5 | 9.4 | 9.72 | 0.05 | 13.54 | 94.1 | 13.6 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 65536 | 99 | warmup | 49159 | 32641.6 | 1507.7 | 51.6 | 32605.1 | 2460.0 | 35104.0 | 38.9 | 14.68 | 0.12 | 11.59 | 91.6 | 18.5 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 65536 | 99 | run1 | 49159 | 32482.2 | 1514.2 | 52.3 | 32466.0 | 2426.7 | 34911.5 | 18.9 | 14.61 | 0.12 | 11.90 | 5421200939511.0 | 18.5 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 65536 | 99 | run2 | 49159 | 32253.1 | 1524.9 | 52.4 | 32237.5 | 2423.8 | 34679.7 | 18.3 | 13.88 | 0.09 | 11.98 | 91.3 | 33.3 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 20 | warmup | 6152 | 3923.7 | 1575.1 | 16.2 | 3905.9 | 7830.6 | 11754.8 | 18.3 | 4.86 | 0.03 | 10.95 | 23.9 | 75.1 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 20 | run1 | 6152 | 3423.6 | 1799.6 | 18.5 | 3418.6 | 6854.0 | 10277.9 | 5.3 | 4.87 | 0.02 | 11.51 | 27.3 | 69.5 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 20 | run2 | 6152 | 3378.5 | 1823.1 | 16.5 | 3374.6 | 7674.1 | 11053.1 | 4.4 | 4.97 | 0.04 | 11.30 | 25.4 | 72.3 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 0 | warmup | 6152 | 5475.5 | 1127.1 | 7.2 | 5458.2 | 17697.2 | 23173.1 | 17.7 | 1.82 | 0.02 | 10.33 | 9.1 | 71.4 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 0 | run1 | 6152 | 5358.7 | 1149.0 | 6.5 | 5354.2 | 19593.9 | 24952.8 | 4.7 | 1.75 | 0.03 | 10.06 | 7345251759841.5 | 81.2 |  |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 8192 | 0 | run2 | 6152 | 5562.7 | 1106.9 | 7.6 | 5557.7 | 16670.5 | 22233.9 | 5.6 | 1.73 | 0.03 | 9.80 | 9.5 | 76.8 |  |

## Notes

- 8B ladder at ngl 99: every rung 2K→64K was ok. No fail and no OOM; 128K not tested. Practical ceiling on 16 GB = 64K: peak dedicated 14.68 GiB, shared Δ 0.12 GiB. That Δ is the first nonzero spill signal; all other rungs are ≤ 0.05.
- 8B decode TPS per rung (warm): 109.5, 105.4 (−4%), 99.9 (−5%), 88.1 (−12%), 72.0 (−18%), 52.4 (−27%). This is healthy KV growth, not a cliff. A 40% step threshold stays quiet here; a 25–30% threshold would false-positive at 64K.
- Qwen 1.5B decode TPS: 366, 332 (−9%), 257 (−23%).
- KV buffer is exactly linear in ctx: 8B = 0.125 MiB/token, Qwen 1.5B = 0.0273 MiB/token. The model buffer is constant and ≈ file size (8B Vulkan0 4403 + CPU_Mapped 282 MiB).
- Peak dedicated − idle − (model + KV + compute) = residual (8B): 2K 0.38, 16K 0.48, 32K 0.17, 64K 0.89 GiB.
- RAM: available RAM drops ≈ 5 GiB for 8B even at ngl 99 (idle 17.0 → min 11.6). This is the mmap'd GGUF resident. A RAM estimate should count ≈ file size, not the CPU_Mapped buffer.
- llama-server timings vs wall clock:
  - total − (prompt_ms + predicted_ms) is 3–10 ms warm and 16–39 ms on warmup. TTFT ≈ prompt_ms + 3–30 ms. They match.
- Warmup matters: the first request's prefill was 5.6× slower on Qwen 2K (2300 vs 12900 TPS) and 19% slower on 8B 8K. Decode was unaffected.
- Partial offload (8B 8K), ngl 20 vs 99:
  - decode −83% (16–18.5 vs 99.8)
  - prefill −34%
  - CPU avg 70–75% vs ~15%
  - GPU util 25% vs 90%+
  This is the decode-collapse signature.
- ngl 0: decode 6.5–7.6 TPS, but prefill is still 1107–1149 TPS and dedicated is 1.8 GiB. The Vulkan build still offloads large-batch matmuls to the GPU with 0 layers, so ngl 0 is not CPU-only for prefill.
- Unavailable metrics:
  - Telemetry for requests shorter than ~1 s (typeperf 1 s granularity; Qwen 2K run1 got 0 samples → n/a).
  - Load peak for loads under ~2 s.
  - GPU util returned garbage 3× (1.3e13, 5.4e12, 7.3e12 %). This is a PDH GPU Engine counter glitch; the values are left raw in the table on purpose.
## Qwen2.5-14B Q4_K_M spill/cliff observation

Per-PID telemetry (GPU Process Memory, Process V2 private WS); warm = median of run1/run2. detectCliffs with DEFAULT_SCORING_CONFIG.cliff = `{"decodeDropRatio":0.6,"minDecodeDropTps":2,"prefillDropPerDoubling":0.5,"sharedSpillBytes":268435456,"vramSaturation":0.95,"ramGrowthBytes":1073741824}`, vramTotal 17095983104.

| ctx | ngl | status | load ms | layers | KV MiB | prompt_n | TTFT ms | prefill TPS | decode TPS | pid ded GiB | pid shared GiB | pid private WS GiB | adapter ded GiB | adapter shared Δ GiB | RAM min GiB | GPU util % | CPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 99 | ok | 4398.1 | 49/49 | Vulkan0:384 | 1537 | 966.9 | 1596.2 | 61.6 | 8.58 | 0.02 | 0.12 | 10.35 | 0.02 | 6.41 | 95.8 | 17.0 |
| 8192 | 99 | ok | 4566.0 | 49/49 | Vulkan0:1536 | 6151 | 4098.4 | 1503.6 | 57.1 | 9.71 | 0.03 | 0.13 | 11.35 | -0.01 | 6.51 | 97.1 | 19.6 |
| 16384 | 99 | ok | 4685.8 | 49/49 | Vulkan0:3072 | 12288 | 9005.0 | 1366.8 | 51.2 | 11.22 | 0.04 | 0.14 | 12.74 | 0.04 | 5.97 | 97.3 | 24.3 |
| 32768 | 99 | ok | 5449.3 | 49/49 | Vulkan0:6144 | 24588 | 22584.8 | 1089.8 | 26.4 | 13.25 | 1.05 | 1.16 | 14.76 | 1.05 | 5.59 | 95.4 | 17.6 |
| 49152 | 99 | error: POST /completion -> HTTP 400: {"error":{"code":400,"message":"request (36879 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":36879,"n_ctx":32768}} | 6821.9 | 49/49 | Vulkan0:9216 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 32768 | 30 | ok | 4920.6 | 30/49 | CPU:2432 Vulkan0:3712 | 24588 | 31272.4 | 787.2 | 5.6 | 9.04 | 0.05 | 2.54 | 10.61 | 0.05 | 4.05 | 40.5 | 45.4 |

### detectCliffs on the ngl-99 ladder

practicalContextCeiling 16384, degradedContextCeiling 32768, limitedBy cliff, spillFreeUpTo 16384

| ctx | verdict | reasons |
|---|---|---|
| 2048 | pass | — |
| 8192 | pass | — |
| 16384 | pass | — |
| 32768 | degraded | decode_drop: decode TPS fell 48% between 16K and 32K (51.2 → 26.4 t/s); shared_spill: spilled 1.1 GiB into shared GPU memory at 32K |
| 49152 | fail | run_failed: 48K: run fail (config_drift) — recomputed after fixing the script mapping; the live run labelled it `crash` |

Notes (14B):
- The rules that fired at 32K on real data are `decode_drop` (ratio 0.516 ≤ 0.60; 51.2 → 26.4 t/s) and `shared_spill` (per-PID shared 1.05 GiB > 256 MiB).
- `vram_spill` did not fire. Per-PID dedicated plateaued at 13.25 GiB = 83% of 15.92 GiB, while the adapter showed 14.76 GiB. WDDM starts spilling this process at ≈13.2 GiB, well below the 95% saturation rule. The shared-memory rule caught it; the saturation rule would never fire here.
- The detector verdict is: practical ceiling 16K, degraded ceiling 32K, limitedBy cliff. That matches a human reading: 16K is the last rung without spill; 32K works at half the decode speed.
- 48K is not a real OOM. llama-server allocated the 48K KV (9216 MiB) but served only n_ctx 32768 (= Qwen2.5 ctx_train) and rejected the 36879-token prompt with HTTP 400 exceed_context_size_error. loadModel accepted the server without checking its effective n_ctx.
- Partial offload at 32K: ngl 30 (30/49 layers) decodes 5.6 t/s vs 26.4 for the spilled full offload, which is 4.7× slower. Prefill 787 vs 1090. RAM min 4.05 GiB, at the guard edge. So a ~1 GiB spill costs less than moving 19 layers to the CPU; do not prefer partial over spilled-full on decode.
- RAM available min was 5.6–6.5 GiB at every ngl-99 rung (9 GB GGUF mmap'd); idle baseline 14.1–15.6 GiB this run, because other workers were active.
