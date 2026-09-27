# EVIDENCE — claim → artifact index

Purpose: every threshold (INTERPRETATION §11), rule `calibration` string (`rules.v2.json`) and planner/runner constant that cites an observation maps here to the exact artifact id, file, line, row or commit that supports it. Reports cite **E-ids** (`E-12`), not prose.

Artifact ids are the ones used in [calibration-ledger.md](calibration-ledger.md): CAL-S, CAL-14, A, B, C, Q-0853, H-*, FX-*. The ledger records their provenance and validity; this file does not repeat it. Ids added after the ledger: **DEV-DB-S1** (the app DB `%APPDATA%/local-ai-optimizer/optimizer.db`, session 1; not in the repo, so row ids are cited and the values are copied here), **BIN-HIP** (static inspection of a release binary) and **EXT-*** (a user report with no artifact).

Status:
- **measured-scoped** — safe to cite for that model/config/rung and machine only (ledger SAFE-TO-CITE).
- **narrative** — stated in a doc; no raw dump.
- **derived** — a fixture, not an observation.
- **estimated** — planner arithmetic.
- **unconfirmed** — a hypothesis with a mechanism but no artifact.
- **external** — a user report.
- **policy** — a chosen value; the evidence only informs it.

Hardware for every measured row: RX 9070 XT (15.92 GiB), Ryzen 7 9800X3D, ~31 GiB RAM, Windows/WDDM, llama.cpp Vulkan b11208 (`85ca3b52c`).

## §11 thresholds

| E | Threshold (value) | Used by | Evidence | Status |
|---|---|---|---|---|
| E-1 | adjusted spill warn 256 MiB | I-2.2 `ctx.spill`, `sharedSpillBytes` | CAL-14 `calibration-2026-09-27.md:91` (14B@32K 1.05 GiB raw per-PID shared, decode 26.4); CAL-S `:66` (8B adapter Δ ≤ 0.12 GiB) | policy informed by measured-scoped; **not validated for the ungated w4n definition** |
| E-2 | decode cliff ratio ≤ 0.60 and ≥ 2 t/s | I-2.1, cliff `decodeDropRatio` | CAL-14 `:104` (16K→32K 51.2→26.4 = 0.516); CAL-S smooth 8B ≥ 0.73 | policy consistent with two sweeps; sensitivity not validated (ledger :78) |
| E-3 | prefill severe 0.5× per doubling | I-3.4 | CAL-S smooth worst 0.74× | heuristic |
| E-4 | thinking effective 50 % | I-3.2 | none | heuristic |
| E-5 | budget headroom warn 0.5 GiB | I-4.1 | CAL-S `:66` 8B residual 2K 0.38 / 16K 0.48 / 32K 0.17 / 64K 0.89 GiB | policy informed by measured-scoped |
| E-6 | in-use at plan 1.5 GiB | I-4.2, `vramInUseUnknownBytes` | CAL-S idle adapter ≈ 1.18 GiB (`:12`, idle ded column) | policy informed by measured-scoped |
| E-7 | rep spread 15 % of median, ≥ 2 reps | I-6.1 | none | heuristic |
| E-8 | limited coverage < 30 items / < 3 per category | I-5.x | none | policy |
| E-9 | category weak ≤ 1/3 (≤ 2/3 coding) | I-5.x | none | policy |

## Rule calibration strings and code constants

| E | Claim | Used by | Evidence | Status |
|---|---|---|---|---|
| E-10 | 14B spill began at 13.25 GiB per-PID dedicated = 83 % of 15.92 | I-4.5 text, `vramSaturation` 0.80, `VRAM_BUDGET_FALLBACK_SHARE` 0.8 | CAL-14 `calibration-2026-09-27.md:91,109` | measured-scoped (raw per-PID, pre-adjusted metric) |
| E-11 | under contention, spill-like growth at 67–73 % dedicated; 2.0–2.1 GiB raw shared while the gated metric read 0.00 | `rawSharedGrowthBytes` 1 GiB, the w4n ungating | H-LONG `calibration-longctx-2026-09-27.md:10,22,36–37` | measured-scoped under measured contention (~2.5 GiB other-process VRAM) |
| E-12 | 8B f16 64K: 1.08 GiB per-PID shared at 11.6 GiB dedicated while 3.1 GiB adapter dedicated was free; Run A had identical load buffers (4403.49 / 8192 / 164.01 + 80.01 MiB), argv and prompt (36,572 tokens), all in dedicated | I-2.8 calibration string (heuristic), the runner retry | DEV-DB-S1 run row 14 (with its telemetry_sample rows) vs A (`session-run-A-…08-45-32-573Z.json`, L8 64K row) | measured-scoped for the observation; the **cause (driver placement) is unconfirmed** until #3's A/B |
| E-13 | "11.6 GiB = a per-process ceiling" (I-4.5 text) | I-4.5 calibration string | the same row as E-12 | **contradicted as a ceiling**: an identical allocation fit in dedicated (A). Kept only as a residency observation. |
| E-14 | 27B Q4_K_M ngl 55 @8K ran clean at 13.03 GiB dedicated (> 0.8 × 15.92 = 12.74) | the estimated per-process budget does **not** prune (31ce4f7) | H-CONSOLE, heavy narrative `calibration-heavy-2026-09-27.md:32–34` (12.67 / 12.81 / 13.03 GiB, shared 0.04) | measured-scoped (console era, pre-no-mmap fix; ledger H-CONSOLE caveats) |
| E-15 | ROCm/HIP (Ollama) on an RX 6800 16 GB used the full 16 GB | I-4.5 text, LIMITATIONS, HIP-BACKEND.md | none (user report relayed by the orchestrator, 2026-09-28) | external. Possibly managed-memory oversubscription (E-22) |
| E-16 | spilled full offload beats partial: 14B@32K 26.4 vs ngl 30 5.6 t/s (4.7×) | I-7.6 `cmp.partial-veto` | CAL-14 `:91` + ngl30 row | measured-scoped (ledger :78: scope it, no unconditional veto) |
| E-17 | partial offload −83 % decode (8B ngl 20 at 8K) | I-7.6 motivation, heavyMode off by default | CAL-S `:72` (16–18.5 vs 99.8 t/s) | measured-scoped |
| E-18 | MoE vs dense 3.6× | I-3.6 | H-0939 G26 ngl21 38.50 vs Q27 ngl50 10.74 t/s @2K (`BENCHMARK.md:280`; ledger :78) | measured-scoped, **cross-family**: not a law |
| E-19 | `-lm none`: CPU layers' pinned buffers show as shared (Qwen3.8 ngl 50: 3.82 GiB shared with 4 GiB VRAM free) | host-pinned subtraction (`hostPinnedBytes`) | H-0939 raw shared 2.52 / 3.82 / 4.57 GiB (ledger :101) | measured-scoped |
| E-20 | VRAM margin 1 GiB; keep-over ≤ 1.15× | `vramMarginBytes`, `keepOverVramMaxRatio` | CAL-S residual `:66` (0.17–0.89 GiB); estimate 2–19 % over measured | policy informed by measured-scoped |
| E-21 | quality-phase RAM +8.6 GiB over the ladder (Qwen3.8 ngl 49 @8K: ≈ 13.6 vs ≈ 5 GiB), attributed to llama-server's `--cache-ram` (default 8 GiB) prompt cache | `--cache-ram 0` at launch and in the export (b3e671f) | the observation: #3's run, relayed (no persisted artifact found). The flag and default: b11208 binary strings (`llama-common.dll`, `llama-server-impl.dll`) | **unconfirmed** cause: confirm by the server log no longer printing "prompt cache is enabled" and quality RAM ≈ ladder RAM |
| E-22 | the HIP build embeds gfx1010–1012, 1030–1036, 1100–1103, 1150–1153, 1200/1201 (RX 9070 XT gfx1201, RX 6800 gfx1030) and bundles the HIP runtime; `GGML_CUDA_ENABLE_UNIFIED_MEMORY` and `hipMallocManaged` present | HIP-BACKEND.md; `serverEnv()` strips the variable | BIN-HIP: `llama-b11208-bin-win-rocm-10.0-x64.zip` sha256 `769c6476…034a`, `ggml-hip.dll` strings; `--version` build 11208 `85ca3b52c` (1375c9f) | static inspection; **device enumeration not run** (GPU hold) |
| E-23 | Qwen3.8-27B siblings: IQ4_XS (14.25 GB) and Q3_K_XL (13.15 GB) would fully offload @8K; Q4_K_S 59/65; this Q4_K_M 55/65 | I-9.2 `plan.sibling-quant` | the planner estimate (`tests/scoring/quants.test.ts`, c234465); 55/65 agrees with E-14 | estimated. The observed Vulkan ceiling (E-10) may make a full offload spill |
| E-24 | per-process budget observations (qualified capacity/clean) | `budgetFor`, planning prune | none yet. The `vram_budget_observation` table is empty on this machine; the first rows come from post-eb30f93 runs | — |
| E-25 | ladder prompt sizing: DEV-DB-S1 rows are stamped `ladder-2` but were character-sized (run-session wrap() did not forward tokenize/templateHash) | the prompt-version stamp fix (684d44b) | DEV-DB-S1 row 14 promptTokens 36,572 = A's character-sized prompt | measured (stamp defect) |

## Pending: #3's overnight sessions
Append one row per new session: `E-n | claim | rule/constant | <session JSON or DB session/run ids> | status`. Priority confirmations:
- **E-12 cause:** A1/A2/B1 placement A/B. It decides whether I-2.8 goes heuristic → measured-calibration.
- **E-21:** quality-phase RAM with `--cache-ram 0`.
- **E-24:** first qualified per-process ceilings.
- **E-1 / E-2** under the ungated spill definition (ledger RE-MEASURE 4).
- **E-22 → measured:** HIP `--list-devices` and a Vulkan-vs-HIP ladder, after the GPU go.
