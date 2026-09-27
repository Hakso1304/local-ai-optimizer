# Morning report — 2026-09-28 (skeleton; fill from the app DB, docs/calibration-overnight-2026-09-28.md, docs/EVIDENCE.md)

## 1. What ran overnight (per stage: session id, HEAD, conditions, wall time, watchdog events)
- Stage 1 — session 3 resume (coding heavy): DONE — see docs/calibration-overnight-2026-09-28.md
- Stage 2 — VRAM A/B (A1/A2/B1/B2): 
- Stage 3 — gen-config sweep (Gemma-4 off/think; Qwen3.8 off/low/medium/xhigh): 
- Stage 4 — HIP --list-devices: 
- Stage 5 — Vulkan vs HIP A/B: 
- Stage 6 — iGPU/UMA split: 
- Stage 7 — Q3_K_XL fully-on-GPU vs Q4_K_M ngl54: 
- E1–E8 evidence batch: 

## 2. Recommendations per workload (interp-2, valid quality) with decision-trace summary
| Workload | Winner (model/quant/backend/ngl/ctx) | decode t/s | Q [lo, hi] | why (rule ids) | provisional? |
|---|---|---|---|---|---|

## 3. Answers to the user's questions
- Why only ~13 of 16 GB (Vulkan per-process ceiling vs HIP; measured): 
- Does HIP use more VRAM / run faster on the RX 9070 XT: 
- Smaller quant fully on GPU vs Q4 partial: 
- iGPU/UMA as extra VRAM: 
- Thinking on/off/effort/temperature best config per model: 

## 4. Engine conformance status (Astra verdicts, last: docs/review-w4o) and what changed overnight (S1 commits)

## 5. Evidence status: which INTERPRETATION §11 thresholds are now measured-calibration (docs/EVIDENCE.md rows flipped)

## 6. Open defects / risks / what needs the user (GPU go, downloads, decisions)

## 7. Builds: nightly pre-verdict (3ec8278) + any later build (dist/BUILD-INFO.txt)
