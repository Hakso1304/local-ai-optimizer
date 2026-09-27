# Status — evidence matrix (2026-09-27)

This is the MVP Definition of Done, the later user requirements and the cross-cutting spec requirements, each with evidence.

**Status legend:**
- **DONE**: verified.
- **DONE-WITH-CAVEAT**: works, with the stated limit.
- **PARTIAL**: works in part.
- **BLOCKED(env)**: cannot be verified on this machine (AMD RX 9070 XT 16 GB, Ryzen 7 9800X3D, 31 GB RAM, Windows 11, no NVIDIA / Ollama / LM Studio).

**Where the evidence lives:**
- Commits are in `git log`.
- Tests: full software suite 59 files / 786 passed at f70b832 (S1/S2); independent coordinator follow-up 179 passed. These checks do not supply hardware calibration.
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
| 8 | Multiple context sizes | DONE | Ladder 2K→64K on the 8B, 2K→32K + a real cliff on the 14B (5a871ce); `detectCliffs` fixtures (A13, A15) | 128K reached only by q8_0 KV on the 8B, degraded under contention (H-LONG); no clean 128K observed |
| 9 | Resource utilization | DONE-WITH-CAVEAT | typeperf sampler (cb0cfac), glitch-row drop, request-window means (1027926); per-run TelemetryChart (089e204, wired 184eb33) (A14) | English PDH counter names only; GPU util is display-only |
| 10 | Recommendation from measured data | DONE | `recommend.ts` (527f86c, c5e8a56); run A: "Best for Coding: …8B… 96.5/100"; `calibration.test.ts` per-workload picks (A16–A19) | Profiles calibrated on one GPU and 2–3 model families |
| 11 | Inspect why | DONE | Breakdown + plain-English reasons, cliff messages, recommended vs scoring ctx, "Not benchmarked: … — reason" (9e014c2, 5dacad4); Results page (78c616d) | — |
| 12 | Reopen without losing history | DONE | node:sqlite storage + `tests/storage.test.ts` / `sessions.test.ts`; interrupted sessions marked on start (07dcd84) (A20, A21) | Newer-than-build DB is refused (2fa68f4): upgrade-only |

## Later user requirements

| Item | Status | Evidence | Caveat |
|---|---|---|---|
| Heavy-model mode | PARTIAL | e38ce73, 4ad9884 (UI toggle), 1027926, 5dacad4 (RAM safety, ordering); `tests/scoring/heavy.test.ts`, `kv.test.ts`; `calibration-heavy-2026-09-27.md` (Qwen3.8-27B 55/65 layers: 12–13 t/s, no spill) | Scoped runs H-0939/0953/1021/1039 cover MoE and -nkvo ladders (0939/0953 quality INVALID); CPU baseline skipped by rule; see calibration-ledger.md |
| Heavy-model live re-run | DONE-WITH-CAVEAT | d24730e (docs/calibration-heavy-2026-09-27.md "Re-run with fixes"): Qwen3.8-27B ngl48/44 2K–16K 9.4→7.3 t/s, Gemma-4-26B-A4B ngl22 36–40 t/s to 16K, ngl18 cliff at 16K (−44%, no spill), shared 0.00 after the spill fix; quality (thinking off) Qwen 16/17, Gemma 16/17, 8B 15/17; Max Quality → Gemma-4 ngl18 93.6 over 8B 91.3 and Coding → 8B were interp-1 scorer outputs (historical, not validated rankings; the 1039 Max-Q rerun's Q27 quality is contaminated by a guard-cancelled item) | 17-item suite gives ±15 bands (I-5.2 will call it indistinguishable until suite v2); telemetry H7 recurrence on one rung |
| Pareto chart + SLO filter | DONE-WITH-CAVEAT | 184eb33 `ParetoChart.tsx`, `SloFilter.tsx`, `tests/results/pareto-slo.test.ts` | Hidden configs don't show why (review finding 1) |
| Required context 32K / 64K / 128K | DONE-WITH-CAVEAT | Core 0a99f35 (effective profile, ladder to the required ctx, gate "practical < required (limited by …)", advisory TTFT, user min-decode, fallback, KV q8_0 / -nkvo variants, CR-04-long); UI 91ec981; `tests/scoring/required.test.ts` + session tests | Long-context: see the H-LONG row (loaded host, ladder-1 fill) |
| Large-scale Coding profile + Dashboard card / view-as | PARTIAL | Profile + `recommendForWorkload` 0a99f35 (27B at 12 t/s with higher quality beats 8B; Coding still picks 8B); card + view-as b7baf98 | #2 is swapping the card over to `recommendForWorkload` |
| Spill false-positive fix (`-lm none` pinned host memory) | DONE-WITH-CAVEAT | 3d7a852 ungates adjusted spill from budgets; b02096f applies benign first-rung baseline before classification/learning; e1ea388 handles incremental pinned declarations and fresh consecutive guard samples | Software tests pass; historical stage1 sub-1 GiB spill attributions remain UNVALIDATED (O1). Stage2 standalone raw A/B does not retroactively validate them. |
| VRAM in use before planning | DONE | 960166c (measured before planning) + 4de83cc (`machineFromProfile` reads it; the note says when unavailable); `candidates.test.ts` | — |
| Sampler pid-column restart | DONE-WITH-CAVEAT | 82670a9 (sampler `hasPidColumns`/`restart`) + 5f8f669 (runner restarts once; `samplerErrors` persisted); session test | Fake sampler only; live behaviour depends on typeperf timing |
| Generation-config search (thinking / effort / temperature) | DONE-WITH-CAVEAT | 9636681 runner + CI stats; 2d99358 knobs from chat templates (Qwen3.8 reasoning_effort low…xhigh; Gemma-4 thinking), HF generation_config.json defaults, sampling params, per-model gen table, export flags | interp-2 generation selection is wired; dd760fa enforces runtime/config scope and skill/seed provenance. Gemma session4 off/think complete:360rows,paired+3[-3,+11] includes0,offkept; historical token-split caveat. Qwen off/low/medium released on842d396; xhigh pending. |
| Interpretation guide + rule engine | PARTIAL | b02096f + e1ea388 close RECHECK4 O1-O9 synthetic scenarios; d67da70/f70b832 harder backend scope/TTFT tests; independent five-file run 179 passed; full suite reported 59 files / 786 passed | dd760fa closes mixed-build/duplicate reuse and v2 skill/seed propagation; bb006ad fixes long-needle infrastructure classification. Independent follow-up 147 passed. Stage2 raw A/B complete; Gemma stage3 released on6bac783. 7166792 closes off-comparator proof/fallback gap (39 residual tests pass); 70c5c9e preserves future token provenance and explicit T1 labels (independent176 scoped tests pass). Current6bac783 Gemma run retains historical split caveat. 842d396 closes effort-proof gap with per-key controlled renders/read-time gate (183 scoped tests); Gemma audit accepted; Qwen main released on842d396 with per-key evidence required. Thresholds and hardware conformance remain unvalidated. |
| Quality suite v2 (qb-2.0.0) | DONE-WITH-CAVEAT | 66e5fc2 suite (Worker #4: 47 static + 13 generated items, seeded) + ccc0a4c integration (#3): selectable per session, thorough = default (60 items), quick = v1 opt-in; seeds persisted for resume; uncertainty module afdb754 wired into scoring (8b5f090) | OVN-1 stores 60 thorough rows per model, one sample per item; skill labels were not propagated in those historical rows. dd760fa fixes future skill/seed provenance; new T=1 generation sweep pending. |
| Long-context calibration (64K/128K) | DONE-WITH-CAVEAT | 655063c docs/calibration-longctx-2026-09-27.md: 8B via runSession(requiredContext=131072): KV q8_0 clean to 64K (68.5 t/s, TTFT 21.8 s) and the only variant reaching 128K (24.2 t/s degraded under GPU contention); f16 max 32K; -nkvo 32K at 7.3 t/s, 64K+ skipped by the RAM guard | Quality/recommendation phase lost to host memory pressure; spill gate defect L1 (share-of-total) routed to #1 |
| Calibration ledger | DONE | docs/calibration-ledger.md (Worker #4): every real-hardware run with commit, contention, spill-algorithm version, thinking handling, quality validity; citation audit; SAFE-TO-CITE vs RE-MEASURE protocol | Historical numbers stay cited with their scope; a clean idle-host re-measurement per the RE-MEASURE protocol is scheduled last |
| Clean idle-host calibration (final engine) | PARTIAL | Session 3 raw timings/quality and run ids are indexed in EVIDENCE.md E26-E31 and calibration-ledger.md; original stage1 results preserved | 14B run18 guard reason 3.0 GiB/raw 3.05 GiB conflicts with stored adjusted=0. Gemma/Qwen sub-1 GiB spill attribution UNVALIDATED. Dedicated peaks are allocations, not capacity budgets. Repaired stage2 five-case A/B complete (b45065d): raw residency and prefill/decode only, client TTFT and allocation buffers unavailable; no capacity or placement-origin promotion. Original partial and setup failure preserved separately. Gemma off/think complete with no established quality gain; Qwen sweep underway. |
| AMD backend choice (Vulkan vs ROCm/HIP) | PARTIAL | b02096f fixes absent-backend fallback, needle switch, CUDA metadata/export and environment casing; e1ea388 scopes quality by backend/build and compares measured same-rung TTFT/decode | HIP hardware unverified. Stage2 Vulkan A/B complete on 3d12db1 archive with recorded b11208 runtime hashes; subsequent HIP experiment requires its own gate. |
| GPU harness safety (Astra review w4q) | PARTIAL | docs/review-w4q-2026-09-28.md: Q1 abort can be lost between apply-template and generation; Q2 PID-chain teardown not guaranteed; Q3 watchdog gaps around OS probes/teardown; Q5 session-dump atomicity; Q6 environment freeze; Q7 host/device buffer separation; Q8 A/B port/props binding; snapshot hashes verified | No new GPU stage until Q1–Q3 (+Q8 for A/B) are fixed (S1 runner/runtime, S3 scripts, S2 negatives) and re-reviewed |
| Responsive layout | DONE | aea9c42: fluid main, auto-fit grids, container-sized charts, min window 960×640; CDP-verified at 1280×720 / 1920×1080 / 2560×1440 | — |
| GPT reviewer pass (Worker #4, GPT-6-Astra) | PARTIAL | Reviews w4…w4p: G/F/R/A-B-C/budget-safety/memory rounds closed by executed probes; RECHECK5 (docs/review-w4p): O1–O9 closed, G08 = fixture repair; remaining P1 HIGH (per-key template proof broadcast across template branches → thinking/effort attributions unproved), P2 seed acceptance, P3 malformed proof | S1/S2 fixing; session 5 effort rows need per-row re-proof (offline tool) before any effort claim; ranking/memory conformant-with-caveats, generation comparisons not yet |
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
| Packaged build verification | DONE-WITH-CAVEAT | Nightly 2026-09-28 pre-verdict build at 3ec8278 (clean worktree): dist/Local AI Optimizer-0.1.0-nightly-2026-09-28-pre-verdict-portable.exe 100,311,388 B + -setup.exe 100,542,095 B, dist/BUILD-INFO.txt; packaged check on 5 sessions (interp-2 reinterpretation, trace, evidence section); console windows hidden (3e7183f); HIP opt-in install present | Unsigned; Astra's verdict pending; HIP untested |
| NVIDIA / CUDA path | BLOCKED(env) | 56c1bbb, acbd169 wired; `tests/nvidia.test.ts` fixtures | No NVIDIA GPU here (nvidia-smi: insufficient permissions) |

## 2026-09-28 postfix nightly checkpoint

Clean detached build `5814270`: `dist/nightly-2026-09-28-postfix-unvalidated/` contains portable and NSIS artifacts plus BUILD-INFO. Typecheck/build passed; validation 803 passed, 1 skipped. This is an unsigned, hardware-unvalidated nightly, separate from the pre-verdict build. See orchestrator-log.md for hashes and calibration gate status.
