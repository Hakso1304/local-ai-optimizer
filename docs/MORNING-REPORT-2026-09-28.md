# Morning report — 2026-09-28 (in progress; filled by Fable as stages land)

## 1. What ran overnight
| Stage | Session / artefact | Conditions | Result |
|---|---|---|---|
| 1. Coding heavy resume | app DB session 3; HEAD 3e7183f; vramInUse 1.13 GiB; 876 s | 14B / Gemma-4 / Qwen3.8, thinking off, qb-2.0.0 thorough | DONE — see §2; `--cache-ram 0` confirmed (min RAM 8.0 GiB vs 4–5 GiB before) |
| 2. VRAM ceiling A/B | scripts/ab-spill (S3 repaired), 5 cases | 8B f16 -c 64K: A1 36.6K prompt, A2 49.2K, B1b after a q8_0 128K load, B2 -ub 256 | DONE (scoped) — per-PID dedicated peak **11.60 GiB in every case** (11.51 with -ub 256) while the adapter still had ≈3 GiB free; raw shared flat 1.32–1.36 GiB from load. The ceiling does not depend on prompt fill, prior placement or ubatch → per-process/allocation-pattern limit of this driver+Vulkan backend, not adapter exhaustion. No capacity/spill claim (pinned-vs-spill undecidable until the baseline definition is settled) |
| 3a. Gemma-4 gen sweep | session 4; 360 quality rows (off/think × 60 items × 3 samples); seed 424242; ngl19 @8K; T1/top_k 64 | thinking off vs on | DONE — paired difference think − off = **+3 [−3, +11]** (includes 0) → no established benefit; "off" kept. Caveat: min_p unset (runtime ≈0.05); reasoning-token split unusable |
| 3b. Qwen3.8 gen sweep | session 5 (running under S3 lease); off/low/medium; ngl49 @8K; T1/top_k 20/min_p 0; seed 424242; 540 rows expected | 8K ladder passed ≈10 t/s, TTFT 8.6 s | RUNNING |
| 3c. Qwen3.8 xhigh | — | held | PENDING |
| 4–5. HIP device check + Vulkan-vs-HIP A/B | vendor/llama.cpp-hip b11208 | held | PENDING (decisive for the 11.6 GiB ceiling question) |
| 6. iGPU/UMA split | — | held | PENDING |
| 7. Q3_K_XL fully-on-GPU vs Q4_K_M ngl54 | file complete (13,146,393,504 B) | held | PENDING |
| E1–E8 evidence batch | E1 software replay only (a44a567): sessions 1/3 deterministic, candidate permutations invariant | measured repeat pending | PENDING |

## 2. Recommendations per workload (interp-2, measured quality)
| Workload | Winner | decode t/s | Q [lo, hi] | why | provisional? |
|---|---|---|---|---|---|
| large_coding @128K (session 1, 8B) | **none** | — | 42/60 | [I-2.5] largest clean context 64K < required 128K (spill); q8_0 KV reaches 128K only degraded (26 t/s, TTFT 62 s) | n/a (no eligible) |
| coding, heavy (session 3) | **Qwen2.5-14B f16 @16K** | 54.4 | 86 [70, 94] | full offload, clean to 16K; 32K guard_abort (3.0 GiB spill stored) | no |
| — runner-up | Qwen3.8-27B ngl49 | 10.5 @8K | 96 [83, 99] | provisional: unmatched rung (spilled 0.50 GiB at 16K); ngl45 clean to 32K at 8.3 t/s @16K | yes |
| — runner-up | Gemma-4-26B-A4B ngl19 | 31.0 @4K … 17.9 @32K | 93 [79, 98] | clean, 10.5–11.0 GiB, no spill; 32K rep spread 44 % | no (Q band overlaps 14B/Qwen) |

Quality bands overlap between the three (14B 86, Gemma 93, Qwen 96, all ±≈10) → per I-5.2 quality is not decisive on a 60-item suite; speed and context decided Coding.

## 3. Answers to the user's questions
- **Why only ~13 of 16 GB?** Measured: with llama.cpp **Vulkan** on this driver a single process keeps ≈11.6–13.3 GiB in dedicated VRAM (stage 2: 11.60 GiB in all four variants with 3 GiB free on the adapter); it is an allocation/per-process limit, not the card. The planner now learns this per machine (advisory until qualified). Whether **HIP** lifts it is stage 4–5 (pending).
- **Smaller quant fully on GPU vs Q4 partial?** Stage 7 pending (Q3_K_XL file ready); planner already suggests sibling quants with estimated fit.
- **iGPU / UMA as extra VRAM?** Stage 6 pending; expectation (I-3.8): same DDR5 bandwidth as CPU offload, 2 CUs → no gain expected; will be measured.
- **Thinking / effort / temperature?** Gemma-4: no measurable benefit from thinking (+3 [−3, +11]). Qwen3.8: off/low/medium running (session 5); xhigh pending.

## 4. Engine conformance
- Rules `interp-2` through the overnight fixes (O1–O9 closed: b02096f, e1ea388; comparator/token-provenance/per-key template proof: 7166792, 70c5c9e, 842d396) with S2 negative tests; Astra's last verdict (docs/review-w4o) preceded these — a fresh re-check is due once the lane is idle.
- One regression found at 03:40 (interp2-rereview G08: fixture lacked the new per-key template proof) → fixed in 3dd8a48 (fixture proof added, rule unchanged); Astra RECHECK5 pending.
- Release gate: docs/RELEASE-GATE.md (9252667) — every item must cite a commit/output; release notes: CHANGELOG.md (0209597).

## 5. Evidence status
docs/EVIDENCE.md E-1…E-31 (S2 owns updates); E-12/E-13 now supported by stage 2 (ceiling independent of fill/placement/ubatch); E-21 (`--cache-ram 0`) supported; E-22 (HIP) and E-23 (Q3 vs Q4) pending.

## 6. Open items / needs the user
- HIP: driver enumeration unknown until `--list-devices` (stage 4).
- Historical rows before O1 (sub-1 GiB spill verdicts) remain "unvalidated" in the ledger.
- Console-window popups fixed (3e7183f).

## 7. Builds
- Nightly pre-verdict at 3ec8278: `dist/Local AI Optimizer-0.1.0-nightly-2026-09-28-pre-verdict-*.exe` (+ BUILD-INFO.txt).
- Post-fix nightly at 5814270 (`dist/nightly-2026-09-28-postfix-unvalidated`) — predates 7166792/70c5c9e/842d396; a final build follows the G08 fix + re-check on an idle lane.
