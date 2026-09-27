# Status — evidence matrix (2026-09-27)

This is the MVP Definition of Done, the later user requirements and the cross-cutting spec requirements, each with evidence.

**Status legend:**
- **DONE**: verified.
- **DONE-WITH-CAVEAT**: works, with the stated limit.
- **PARTIAL**: works in part.
- **BLOCKED(env)**: cannot be verified on this machine (AMD RX 9070 XT 16 GB, Ryzen 7 9800X3D, 31 GB RAM, Windows 11, no NVIDIA / Ollama / LM Studio).

**Where the evidence lives:**
- Commits are in `git log`.
- Tests: `npm test` = 28 files / 355 tests green at 5dacad4.
- E2E runs are `docs/session-run-*.json` (A = coding session, B = cancel, C = RAM-guard skip path, H = heavy).
- Calibration is in `docs/calibration-2026-09-27.md` and `docs/calibration-heavy-2026-09-27.md`.
- Criteria IDs (A1–A26, X1–X20) refer to `docs/ACCEPTANCE.md`.
- The full caveat list is in `docs/LIMITATIONS.md`.

## MVP Definition of Done

| # | Item | Status | Evidence | Caveat |
|---|---|---|---|---|
| 1 | App launches | DONE-WITH-CAVEAT | e60e25d scaffold; e315e7b wired app; the UI was verified on LAO_SEED_DEMO=1 and on a real coding session (184eb33 notes) (A1) | Unsigned builds, so SmartScreen warns |
| 2 | Real hardware scan | DONE | e60e25d `scanner.ts` + `tests/scanner.test.ts` on a captured real scan (A2–A5: 16 GB from qwMemorySize, no stale RTX 3080, iGPU flagged) | AMD temperature/power UNAVAILABLE (no native SDK) |
| 3 | ≥ 1 runtime detected | DONE | llama.cpp b11208 Vulkan detected/installed (e60e25d, a3dc31e System-page install); Ollama/LM Studio HTTP detection (`runtimes/others.ts`) (A6) | Ollama/LM Studio not installed here: their detection shows unavailable |
| 4 | ≥ 1 local model identified | DONE | `gguf.ts` header reader (cb0cfac, a456679 layout keys, b0f2fca incomplete-file flag) + `tests/gguf.test.ts`; 5 real GGUFs in D:\llm-models (A7, A8) | Ollama/LM Studio stores are fixture-tested only (70552a9, acbd169) |
| 5 | Select a workload | DONE | 7 profiles in `scoring/workloads.ts` (527f86c); workload picker (78c616d); weights sum to 1 in `scoring.test.ts` (A9, X6) | — |
| 6 | Run a real benchmark | DONE | Run A: coding session, 1.5B + 8B, 183 s, real llama-server (2d46405, e315e7b) (A10) | — |
| 7 | Real metrics | DONE-WITH-CAVEAT | TTFT (wall clock), prefill/decode from llama-server `timings`, load time, per-PID VRAM/shared/private WS; calibration tables (05b153e) (A11, A12) | 1 s typeperf granularity: sub-second steps may have 0 samples, shown as unavailable |
| 8 | Multiple context sizes | DONE | Ladder 2K→64K on the 8B, 2K→32K + a real cliff on the 14B (5a871ce); `detectCliffs` fixtures (A13, A15) | 128K rung never reached on this machine (memory-bound) |
| 9 | Resource utilization | DONE-WITH-CAVEAT | typeperf sampler (cb0cfac), glitch-row drop, request-window means (1027926); per-run TelemetryChart (089e204, wired 184eb33) (A14) | English PDH counter names only; GPU util is display-only |
| 10 | Recommendation from measured data | DONE | `recommend.ts` (527f86c, c5e8a56); run A: "Best for Coding: …8B… 96.5/100"; `calibration.test.ts` per-workload picks (A16–A19) | Profiles calibrated on one GPU and 2–3 model families |
| 11 | Inspect why | DONE | Breakdown + plain-English reasons, cliff messages, recommended vs scoring ctx, "Not benchmarked: … — reason" (9e014c2, 5dacad4); Results page (78c616d) | — |
| 12 | Reopen without losing history | DONE | node:sqlite storage + `tests/storage.test.ts` / `sessions.test.ts`; interrupted sessions marked on start (07dcd84) (A20, A21) | Newer-than-build DB is refused (2fa68f4): upgrade-only |

## Later user requirements

| Item | Status | Evidence | Caveat |
|---|---|---|---|
| Heavy-model mode | PARTIAL | e38ce73, 4ad9884 (UI toggle), 1027926, 5dacad4 (RAM safety, ordering); `tests/scoring/heavy.test.ts`, `kv.test.ts`; `calibration-heavy-2026-09-27.md` (Qwen3.8-27B 55/65 layers: 12–13 t/s, no spill) | One real dense model; MoE, -nkvo and CPU-baseline paths not yet measured |
| Heavy-model live re-run | DONE-WITH-CAVEAT | d24730e (docs/calibration-heavy-2026-09-27.md "Re-run with fixes"): Qwen3.8-27B ngl48/44 2K–16K 9.4→7.3 t/s, Gemma-4-26B-A4B ngl22 36–40 t/s to 16K, ngl18 cliff at 16K (−44%, no spill), shared 0.00 after the spill fix; quality (thinking off) Qwen 16/17, Gemma 16/17, 8B 15/17; Max Quality → Gemma-4 ngl18 93.6 over 8B 91.3; Coding → 8B (Qwen below the 10 t/s gate) | 17-item suite gives ±15 bands (I-5.2 will call it indistinguishable until suite v2); telemetry H7 recurrence on one rung |
| Pareto chart + SLO filter | DONE-WITH-CAVEAT | 184eb33 `ParetoChart.tsx`, `SloFilter.tsx`, `tests/results/pareto-slo.test.ts` | Hidden configs don't show why (review finding 1) |
| Required context 32K / 64K / 128K | DONE-WITH-CAVEAT | Core 0a99f35 (effective profile, ladder to the required ctx, gate "practical < required (limited by …)", advisory TTFT, user min-decode, fallback, KV q8_0 / -nkvo variants, CR-04-long); UI 91ec981; `tests/scoring/required.test.ts` + session tests | Long-context behaviour is not yet calibrated on real 64K–128K runs |
| Long-context calibration (64K–128K) | pending: #3 | — | — |
| Large-scale Coding profile + Dashboard card / view-as | PARTIAL | Profile + `recommendForWorkload` 0a99f35 (27B at 12 t/s with higher quality beats 8B; Coding still picks 8B); card + view-as b7baf98 | #2 is swapping the card over to `recommendForWorkload` |
| Spill false-positive fix (`-lm none` pinned host memory) | DONE-WITH-CAVEAT | 2d51cc0: spill = shared − host-pinned − unsaturated baseline, only while dedicated ≥ 80 %; session tests | Checked against #3's figures (Qwen3.8 ngl 50: 3.82 GiB shared, 4 GiB free); not yet re-run live |
| VRAM in use before planning | DONE | 960166c (measured before planning) + 4de83cc (`machineFromProfile` reads it; the note says when unavailable); `candidates.test.ts` | — |
| Sampler pid-column restart | DONE-WITH-CAVEAT | 82670a9 (sampler `hasPidColumns`/`restart`) + 5f8f669 (runner restarts once; `samplerErrors` persisted); session test | Fake sampler only; live behaviour depends on typeperf timing |
| Generation-config search (thinking / effort / temperature) | DONE-WITH-CAVEAT | 9636681 runner + CI stats; 2d99358 knobs from chat templates (Qwen3.8 reasoning_effort low…xhigh; Gemma-4 thinking), HF generation_config.json defaults, sampling params, per-model gen table, export flags | Scoring-side selection lands with the rule engine; live sweep on Qwen3.8/Gemma-4 pending (#3) |
| Interpretation guide + rule engine | DONE-WITH-CAVEAT | Guide v2 (64927c6, a7845dc) reviewed by Worker #4; engine interp-2 (8b5f090): rules.v2.json with v2 ids + origin tags, verdicts() with hard-constraints-first, measured-eligibility predicate (I-6.0), common scoring rung (I-7.1), scoped partial veto (I-7.6), immutable priors, confirmed vs provisional lists, paired quality difference via uncertainty.ts, persisted decision trace (I-7.2); all 24 acceptance tests from the conformance audit (tests/review-w4g) pass; Interpretation panel + Dashboard insights in UI | Spill gate vs effective budget (L1) and req.ladder/prompt fill (L2) in progress; Astra conformance re-review on interp-2 pending |
| Quality suite v2 (qb-2.0.0) | PARTIAL | 66e5fc2 by Worker #4: ~60 items, seeded generators, good/bad proofs, 69 tests; docs/quality-v2.md hook spec | Integration into the runner/quality mode 'thorough' pending (#3); uncertainty module pending (#4) |
| Long-context calibration (64K/128K) | DONE-WITH-CAVEAT | 655063c docs/calibration-longctx-2026-09-27.md: 8B via runSession(requiredContext=131072): KV q8_0 clean to 64K (68.5 t/s, TTFT 21.8 s) and the only variant reaching 128K (24.2 t/s degraded under GPU contention); f16 max 32K; -nkvo 32K at 7.3 t/s, 64K+ skipped by the RAM guard | Quality/recommendation phase lost to host memory pressure; spill gate defect L1 (share-of-total) routed to #1 |
| Responsive layout | DONE | aea9c42: fluid main, auto-fit grids, container-sized charts, min window 960×640; CDP-verified at 1280×720 / 1920×1080 / 2560×1440 | — |
| GPT reviewer pass (Worker #4, GPT-6-Astra) | DONE-WITH-CAVEAT | Pass 1 cf05710 (14 findings) fixed in ba019cf / 6389ff7 / c71cfc8; pass 2 3781e5e (20 findings) fixed in 4f6ea4c / 1f44374 / 09063eb; all it.fails markers resolved (438/438); pass 3 (docs accuracy + test quality) in progress | Sandbox memory cap is RSS-polled (no Job Object); NTFS link checks are check-then-open |
| HF download with login | DONE-WITH-CAVEAT | Core 282b3ec + `tests/hub.test.ts` (local servers: resume, redirect/no token leak, sha256, 401/403/404); IPC/page 2a349e7, 09fbf8b; wired fa491b0, 486f958 | No real huggingface.co call in the test suite; not yet verified live |

## Cross-cutting requirements

| Item | Status | Evidence | Caveat |
|---|---|---|---|
| MEASURED / ESTIMATED / DECLARED / UNAVAILABLE labels | DONE | `Metric` type (527f86c); unavailable scores a neutral 50 (c5e8a56); ESTIMATED quality never plotted (184eb33) (X20) | — |
| No fabricated data | DONE | Unavailable ≠ 0 everywhere; peaks unavailable when n = 0 (d71eb57); demo data only in a separate DB, flagged (78c616d) (A22) | — |
| Cancel | DONE | Run B: abort → cancelled in 783 ms, 0 leftover llama-server; `session.test.ts` cancel/pause tests (A23) | — |
| Timeouts | DONE-WITH-CAVEAT | Per-prompt timeout (60 s + 10 ms × ctx), quality 180 s; `req_timeout` test (A24) | Load wait is a fixed 120 s |
| Memory guard | DONE-WITH-CAVEAT | Pre-check + in-step floor (54eb829, 1027926, 5dacad4); run C: RAM-guard skip path; `session.test.ts` guard tests (A25) | Thresholds are heuristics; the first heavy run hit 1.0 GiB free before the 5dacad4 fix |
| Process cleanup | DONE-WITH-CAVEAT | kill → taskkill /T /F, pid file + stale kill, samplers stopped on quit (06c8140), NSIS pid kill (2fa68f4); `llamacpp-lifecycle.test.ts` (A26) | A hard kill of the Electron main process leaves the server until the next launch |
| Pause / resume / retry / rerun | DONE | 7442817, 07dcd84 (IPC), 38bb53c + 3d5968c (stored plan); `session.test.ts` pause/retryFailed/rerunConfigIds/plan tests | — |
| Export / apply params | DONE-WITH-CAVEAT | 01e5c9e generators + `tests/export.test.ts`; export menu e31dc64 | Ollama Modelfile / LM Studio keys unverified (neither installed) |
| Tests | DONE | 355 tests: scanner, GGUF, telemetry, llama.cpp fake-process lifecycle, quality/sandbox, scoring + adversarial (c217b85) + calibration fixtures, runner, storage, hub, export | No real-GPU test in CI; real runs are scripted (`scripts/calibrate.ts`, `run-session.ts`) |
| Packaging | DONE-WITH-CAVEAT | a3dc31e electron-builder portable + NSIS; 2fa68f4 review fixes (dev `-dev` userData, DB refusal, NSIS cleanup) | Unsigned |
| Packaged build verification | DONE-WITH-CAVEAT | Shipped build at 04b4cef (+67fdc2c): portable 100,276,724 B, NSIS setup 100,507,524 B (unsigned), 609 tests; verified: real hardware, runtime, models with incomplete-file detection, heavy-mode / required-context / min-decode / gen-search controls, both Dashboard cards, View-as, Interpretation panel with real [I-x.y] insights, Download page, single-instance, responsive 1280×720; earlier builds ran real persisted sessions | Unsigned (SmartScreen); next re-package after rules interp-2 |
| NVIDIA / CUDA path | BLOCKED(env) | 56c1bbb, acbd169 wired; `tests/nvidia.test.ts` fixtures | No NVIDIA GPU here (nvidia-smi: insufficient permissions) |
