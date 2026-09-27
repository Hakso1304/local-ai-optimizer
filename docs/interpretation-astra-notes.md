# Interpretation guide review — Astra / W4D

Reviewed `docs/INTERPRETATION.md` v1 as the normative specification, including the concurrently arriving implementation in `src/core/interpret/index.ts`, `rules.v1.json`, scoring, generation search and storage. Observed HEAD during review: `1225b60`; other workers were editing these files. This is a proposed guide revision, not authorization to replace its rules silently. No existing file was changed or real inference run. Code locations identify the reads made during this review.

## HIGH — Rules to clarify or add before ranking depends on them

### 1. Define the sampling unit and interval method before using “95%” to choose a winner

**Rules:** I-5.1, I-5.2, I-5.6, I-7.3, I-8.2. `INTERPRETATION.md:149-164,187-203`.

`qualityStats` (`src/core/scoring/components.ts:67-89`) currently counts repeated rows as additional items and combines category variances assuming independence. Repeating the same item three times does not provide three new skills; v2's two or three variants within each skill are also related. Its adjusted proportion supplies the variance, but the displayed symmetric band is centered at the unadjusted Q. This is not the adjusted interval described in its comment. A one-item, all-pass category illustrates the issue: the displayed lower endpoint is about 56%, although observing one success should not exclude an underlying 10% pass probability at a claimed 95% coverage level. No calibration artifact establishes confidence coverage.

**Add `quality.coverage-unit`:** publish unique skill count, unique resolved item count, problem-seed count, completion count and samples per item separately. Call the current band an *uncertainty heuristic* until the method and coverage assumptions are specified/tested. For paired comparisons use the same items/seeds and estimate the **difference** with skill/item clusters; report [lower, upper] within [0,100], not an unqualified symmetric ± near 0/100. Save `intervalMethod`, `intervalVersion`, `unit`, `confidenceLevel`, `lower`, `upper` and the resolved item/skill/sample identities.

**Threshold change:** replace `n < 30 graded rows` with `<30 unique completed items OR incomplete required skill/category coverage`, explicitly a policy warning, not a confidence guarantee. Three repeats of 17 tasks still trigger it. V2 offers 60 items but only 24 skill groups, so 60 must not imply 60 independent abilities.

### 2. Overlap is insufficient evidence, not proof of equal quality or of what decided the ranking

**Rules:** I-5.2, I-7.3, I-7.4.

Two overlapping marginal bands do not prove the difference lies within a paired-difference interval, nor that the models are equivalent. “Speed and memory decided” is false if the weighted score still used the point quality difference. `interpret/index.ts:219` starts a band-based policy, while `scoreOf` at `:112-118` still sums point contributions.

**Add `cmp.decision-trace`:** store the actual applied gate, eligible set, comparison basis, quality-difference interval and tie-break chain. Say “insufficient evidence to distinguish quality on this suite” unless a paired comparison supports more. If speed/memory should decide inside a quality uncertainty region, the algorithm must explicitly remove/neutralize the quality delta there. For a smaller quantization, prefer it only if it meets required context/speed/safety and no predeclared quality non-inferiority margin is violated; do not turn non-significance into equivalence.

**Missing fields:** paired `problemInstanceId`, `sampleSeed`, `genConfigHash`, decision trace and a declared equivalence/non-inferiority margin. No observed run calibrates such a margin; do not invent an empirical one from Q100/Q60 synthetic tests.

### 3. Resolve the provenance ranking contradiction with an explicit eligibility policy

**Rules:** I-1 table versus I-1.1 (`:35,39-40`), I-5.5 and I-7.3.

“ESTIMATED never decides a ranking against MEASURED” conflicts with permitting it to decide a provisional winner. Unknown as neutral also does not specify whether an unverified hard constraint passes. The new engine caps estimated quality at the **lowest measured quality among all candidates** (`interpret/index.ts:161-174`), not only comparable eligible candidates. A newly added failed-gate model with Q0 can change other models' priors/gates. Other estimated components can still affect totals. This is a data-dependent policy, not simply attaching provenance.

**Proposed rule:** evaluate hard user constraints first; unknown hard-constraint evidence is “not verified,” never a measured pass. Define a confirmed winner using comparable measured inputs; keep estimate-only candidates in a separately labelled provisional exploration list. If the product instead allows provisional winners, explicitly revoke “never decides” and show precisely which estimated term changed the decision via a recorded counterfactual. Do not adjust one model's prior using an unrelated candidate's measured minimum.

**Missing fields:** decisive-component/decision-trace metadata; metric `kind` alone shows inclusion in a score, not causal decisiveness.

### 4. Unknown memory must not become zero headroom cost or a fabricated cause

**Rules:** I-1.2, I-4.1, I-4.2.

The guide's I-1.2 assumes every missing memory observation came from a short step. There are missing/localized counters, dropped rows and sampler failures; the later H measurements also motivated an overshoot/drop fix. Preserve the recorded reason. `interpret/index.ts:429` currently computes `total - (inUse ?? 0)`, whereas the planner (`candidates.ts:167`) subtracts measured/assumed in-use **and the 1 GiB reserve**. This overstates the guide's “budget minus peak” headroom by at least the reserve and, if in-use is unknown, by more. The engine's unavailable-memory loop at `:449` covers VRAM/shared but not the guide's peak RAM case.

**Add `mem.budget-basis`:** distinguish (a) planning budget remaining after its reserve, (b) measured whole-adapter free VRAM, and (c) per-PID dedicated usage. Show unavailable when the inputs to a measured headroom claim are absent; a reconstructed assumed budget is ESTIMATED and must name the assumptions. Never sum/max counters from different times as if simultaneous. Replace “other apps will push this into shared” with “less room for other apps; spill risk increases.”

**Missing fields:** immutable `planningVramBudgetBytes`, `planningVramReserveBytes`, budget source/assumption, GPU identity and timestamp, and simultaneous adapter-free/per-PID observations for a measured free-space claim. Existing scan `vramInUse` and request overrides can help reconstruct some budgets, but are not the old rule version or contemporaneous free VRAM.

### 5. Treat an untested ceiling as censored coverage, not the machine's proven limit

**Rules:** I-2.1, I-2.3, I-2.4, I-2.5.

The guide's limit enum includes memory/spill/declared but omits “largest tested,” “user cap,” “cancelled/partial,” and “unknown.” `detectCliffs` (`scoring/cliff.ts:108-124`) returns a passing prefix and `none|cliff|failure|untested`; planning skips require additional evidence. I-2.4 would label 8K as the machine's usable limit merely because a 128K model was tested only to 8K. I-2.3 hardcodes VRAM even when the skip was RAM. `interpret/index.ts:307-309` currently attaches a reason **string** under an `estVramBytes` evidence field; that is not a numeric byte observation.

**Add `ctx.coverage`:** report `largestCleanTested`, `firstObservedFailure`, `firstPlannedSkip`, and `limitKind` independently. An unattempted higher rung gives a lower bound on capability, not a demonstrated maximum. Required context can be “not reached,” “tested and failed,” or “not tested,” with different actions. Keep first failure/cliff sticky unless a documented recovery rule applies; do not silently interpret arbitrary later passes as a continuous usable ceiling.

**Threshold change:** retain `<25% declared` only as a **coverage note** unless a measured failure establishes a limit. Change “this machine's usable limit” to “largest clean context measured in this run.”

**Missing fields:** structured skip `{resource, estimateBytes, budgetBytes, ruleId}`, stop reason/user cap, completion status, and planner rule-version/budget snapshot; current reason strings are insufficient for robust numeric rules.

### 6. Do not equate adjusted spill with a speed collapse or universal Windows ceiling

**Rules:** I-2.2, I-3.5, I-4.5.

Keep 256 MiB as a provisional spill-warning threshold: the smooth 8B record has ≤0.12 GiB shared delta, while the 14B at 32K has 1.05 GiB and 51.2→26.4 t/s (`docs/calibration-2026-09-27.md:62-66,86-113`). That does **not** prove all spill causes a decode fall, or that all WDDM systems cap at 83%. That same 8B calibration has adapter dedicated about 14.68 GiB; the 14B figure is per-PID dedicated 13.25 GiB / 15.92 GiB. Baselines/scopes differ. Do not generalize the measured 83% observation into “95% is not reachable on Windows.”

**Proposed rules:** say “adjusted shared usage exceeded threshold at this rung”; append a measured decode delta only if both comparable adjacent rows exist. A one-sample excursion is provisional, not a precise onset; corroborate with sustained samples or replay before claiming a plateau. Keep runtime safety aborts independent of this presentation-confidence rule. First-rung spill has no last clean rung: do not offer `use-context` at the same spilling rung.

Use the host-pinned/baseline adjustments and actual GPU saturation evidence, with source/version; do not compare old raw-shared runs to new adjusted-spill rows without that metadata. `interpret/index.ts:299-304` already makes the decode-drop clause conditional, an improvement the guide should retain.

### 7. Exclude invalid evaluation infrastructure from model quality

**Rules to add:** `quality.harness-invalid` (critical), `quality.truncated` (warn), `quality.incomplete` (warn).

The artifact now explicitly named `docs/session-run-H-coding-heavy-2026-09-27T09-53-34-020Z.INVALID-quality.json` must not calibrate quality thresholds or winner claims. An evaluator startup/sandbox failure is not an incorrect model answer. The runner still has generic `pass:false` request/error rows (`session.ts`, quality loop) and persisted `QualityResult` provides only pass/score/detail; consumers cannot reliably separate operational invalidity from semantic failure by this shape.

**Policy:** retain inference performance from independently valid phases but quarantine affected quality/gen comparisons; count only executed valid evaluations in completed coverage, never silently renormalize missing required categories into a final score. A budget-truncated answer can be a failure under a declared generation-budget policy, but must be labelled separately so increasing maxTokens is an actionable diagnosis.

**Missing fields:** `evaluationStatus: valid|infra_error|unrun`, `stopReason`, `outputTruncated`, per-item `maxTokens`, `checkerVersion`, infrastructure error code, and artifact/session invalidation reason. Filename annotations are not sufficient machine-readable provenance.

### 8. Thinking token estimates must stay estimated; compare matching contexts and configs

**Rules:** I-3.2, I-5.4, I-8.1–I-8.3.

New `GenRow`/`GenQuality` fields are a useful start, but `session.ts:713-714` can apportion total tokens by reasoning/answer **character length**, and `benchmark/gen.ts:summarizeGen` marks answerTokens/effectiveTps measured while reasoningTokens is estimated. Derived answer TPS cannot have stronger provenance than its estimated numerator. No stored baseline H run supplies a verified answer/reasoning split or a complete matched thinking-effort experiment.

**Add `gen.comparable`:** persist actual accepted template/sampling knobs, model/template/runtime hashes, prompt instance, context, token budget, sample seed, token-count source and per-request timing. A requested thinking flag does not prove the template honored it. Compare effective versus raw generation on the same request/task/context, not suite medians against a different ladder prompt. Use the 50% effective/raw rule only when both quantities are comparable and positive, and call it a chosen diagnostic threshold, not calibrated harm.

**Missing fields:** verified thinking/effort support, per-count provenance, per-item promptTokens/context and raw decode/TTFT alongside answer-time data; exact applied GenConfig snapshot rather than reconstructing it later from changing defaults (`storage/sessions.ts:30-39`).

### 9. Cold rows and failed-only candidates need real enforcement, not just an insight

**Rules:** I-6.3, I-6.4.

`scoring/cliff.ts:15-16` accepts pass/degraded plus positive decode regardless of `warm`; `interpret/index.ts:473` merely emits “excluded from speed scoring.” Thus a manually restored/legacy `warm:false` positive row can still influence components. Unknown `warm` in older rows is also not warm=true. The insight loops run over usable verdict candidates (`interpret/index.ts:134-139,461-474`); a model whose **only** rows crashed can be absent from I-6.3's failure list despite the “any failure in the session” rule.

**Fix to encode in guide:** one shared speed-eligibility predicate must enforce warm/provenance/finite-positive/context rules before scoring and explaining. Audit failures from **all original persisted runs**, including excluded models, but distinguish superseded retry rows from active evidence. Label old warm-unknown data unverified rather than inventing successful warmup.

### 10. Confidence cannot identify identical models or eliminate context/generation confounding

**Rules:** I-7.1, I-7.4, plus partial-offload preference.

`interpret/index.ts:487-488` recognizes “same model, different quant” by arch, equal parameter count and different quant. Distinct fine-tunes can satisfy all three. A shared machine scan also does not make rows comparable if one has different context occupancy, KV type, template, active adapter, CPU load or generation mode. The unconditional partial-offload gate (`interpret/index.ts:208-211`) can exclude a partial config reaching required context solely because a full-offload config has one usable **short** rung.

**Add `cmp.identity-and-fit`:** match model family/base revision and tuning identity, tokenizer/template and benchmark procedure; require comparable actual prompt tokens and context. Only call one config dominated if its alternative satisfies the same hard constraints. Use the real 14B 26.4 versus 5.6 t/s at 32K as evidence for that observed comparison, not a universal veto against a longer-context partial candidate.

**Missing fields:** base-model/revision and fine-tune identity, GGUF content fingerprint, tokenizer/template hashes and exact device identity. Param count and architecture are insufficient.

## MED — Threshold/wording proposals grounded in current measurements

| Rule | Proposed change | Evidence and limit |
|---|---|---|
| I-3.1 decode bands | Use explicit half-open intervals `<3`, `[3,8)`, `[8,20)`, `[20,40)`, `[40,100)`, `>=100`; replace “instant” with “very fast streaming,” never use a band as an independent gate. | 8B decodes 52–110 t/s but its 64K first-token wait is ~32 s. A high decode rate is not instant response. These boundaries are preferences, not measured human usability. |
| I-3.3 TTFT | Keep 1/5/20/60 s as descriptive bands only; let the effective workload/user tolerance determine a gate, including advisory-latency profiles. | Long-context profiles accept 90/120 s and Large-scale Coding is advisory; a blanket >60 s rejection without explicit required context conflicts with those policies. 8B at 64K ~32 s is expected work, not proof of failure. |
| I-3.4 prefill scaling | Keep 0.5 per doubling as a **severe** slowdown heuristic; use actual prompt-token ratios when occupancy differs. Remove implication that only <0.5 is super-linear. | Even throughput dropping 0.8× while tokens double makes prefill time grow ~2.5×. Baseline worst per-doubling throughput ratio ~0.74 is smooth, not a cliff (`calibration-2026-09-27.md`). |
| I-2.6 transient dip | Keep the 0.60 ratio + 2 t/s absolute threshold/recovery convention aligned with the cliff engine; say “recovered on the next tested rung; cause unverified,” not “likely contention.” | 8B's steepest smooth decode change is only ~27%; the 14B actual spill falls ~48%. No recorded thermal/contender intervention proves causation. Missing rungs must be disclosed. |
| I-4.1 margin | Retain 0.5 GiB as an explicitly chosen low-reserve warning after defining budget; also expose the separate 1 GiB planner reserve. Do not claim 0.5 guarantees safety. | The measured residual in initial 8B estimates ranged ~0.17–0.89 GiB (`calibration-2026-09-27.md:66`), exceeding 0.5 GiB at some contexts. That residual is not itself measured free memory. |
| I-4.3 RAM near floor | Keep +1 GiB as advisory; report signed raw minimum minus floor, actual floor and mmap credit separately. Never clamp a below-floor distance to zero. | The original heavy run hit 1.0 GiB free (`calibration-heavy-2026-09-27.md:35`). Current `interpret/index.ts:447` clamps negative distance, hiding how far below it went. Heuristic reclaimable credit is not physically free RAM. |
| I-5.3 weak categories | Specify inclusive fractions `<=1/3` and `<=2/3` (if those are intended), and an explicit minimum item count; separate coverage from measured weakness. | Decimal .33/.66 and `<` differ from 1/3 and 2/3. With v1 coding 2/3, the intended boundary is easy to miss; current engine uses `<` at `:349-350`. No real v2 quality calibration yet supports sharper thresholds. |
| I-2.3 / I-9 q8 action | Say “about 47% less KV storage than f16 for the modeled q8_0 blocks,” not exactly half, and never halve total model memory. Offer only when KV is the limiting resource and the runtime/layout supports it. | `candidates.ts:58` uses f16=2, q8_0=34/32 bytes: ratio 0.53125. Weight/compute/recurrent memory remains. RAM-bound -nkvo may worsen the RAM limit. |
| I-3.5 / I-3.6 partial/MoE | Show measured alternatives only when available at comparable context; otherwise omit numeric comparisons. Do not infer active parameters from total × activeExperts/experts. | Real 2K rows: Qwen -nkvo 7.44 vs 10.74, Gemma -nkvo 27.04 vs 38.50 (`session-run-H-max_quality-heavy-2026-09-27T09-39-07-979Z.json:431,521,611,791`). Shared dense layers/embeddings remain active; 3.6× is one cross-family observation, not a general MoE causal rule. |
| I-4.4 mmap | Change “is reclaimable, not a leak” to “consistent with mmap cache” only with measured before/after/after-unload evidence; otherwise explain mmap as a possibility. | Heavy run's large drop motivated no-mmap mode; `interpret/index.ts:435` currently emits the diagnosis for every full-offload mmap config from declared file size alone. |
| I-6.1 rep variance | Define `(max-min)/median > .15` or retain `(max-min)/max > .15`, but name it consistently; require >=2 valid reps and show them. Use this to request a rerun, not assert thermal cause. | Current engine uses /max (`:466-467`); 85 and 100 gives 15% by max but 16.2% by median. Two reps cannot estimate a stable variance distribution. No artifact validates 15% as an optimal cutoff. |

## MED — Rule completeness, ordering and executable actions

1. **Add `data.scope` / `data.invalidated` before interpretation.** Every numeric evidence entry needs value, unit, kind, source, sample count, timestamp/window, config/rung and artifact/rule versions. “MEASURED = full trust” (`:34`) should instead mean observed with a stated scope; dropped/glitched/censored observations still require validation. Evidence may be DECLARED/ESTIMATED for conditional diagnostics; the preamble's “never without measured evidence” otherwise contradicts metadata and missing-data insights.
2. **Resolve panel priority.** I-0.1 says ceiling/quality lead, but I-2.5 says unmet required context leads. Use critical invalid-data/safety/user-requirement failures first, then coverage/quality, then speed. Do not hide a critical guard/device failure behind a two-warning Dashboard cap.
3. **Make actions conditional and typed.** I-9.1 requires an action for every warning; a version mismatch may need `rerun-comparable`, and missing telemetry needs `retry-telemetry`/`inspect-diagnostics`, neither currently listed. Never suggest `raise-min-decode` as a way to speed hardware, or `use-context` without a known clean context. Download suggestions require a verified identity and a clear label that their performance was not measured here.
4. **No silent fallback from recommended to reference context.** Label each number's actual rung and prompt token count; if recommendedCtx lacks the measurement, render unavailable rather than substituting a smaller-context rate. The newly added `referenceWhy/recommendedWhy` metadata is useful; preserve it through storage/rendering.
5. **Version rules alongside decisions.** Existing sessions can be reinterpreted under new rules, but display “reinterpreted with rules X” versus historical recommendation version. Store thresholds actually used, not merely the present `rules.v1.json` path.
6. **Separate profiles from universal truths.** Guide intro `:5` calls all numbers calibration-derived, while `:236-239` correctly calls speed/quality bands qualitative judgment. Attach `origin: measured-calibration|policy|heuristic` and calibration IDs to each configurable threshold.

## Data feasibility matrix — do not fabricate fields for old sessions

| Rules | Can current saved data evaluate it? | Required missing/conditional fields |
|---|---|---|
| I-2.1/2.3/2.4 ceiling cause | Partly: runs, declared ctx and skipped reason strings exist | Structured skip/budget/stop reason and old planning-rule version; distinguish absence from a measured limit. |
| I-2.2 adjusted spill/onset | New adjusted metrics partly; old runs have different semantics | Spill algorithm version, raw/pinned/baseline scope and synchronized samples; no baseline observation means unknown, not zero confidence. |
| I-2.6 recovered dip | Three comparable runs permit the pattern | Contention/thermal causation remains unavailable; never synthesize it. |
| I-3.2/5.4/8.x thinking comparisons | **New code is adding** GenRow/GenQuality, not retroactive evidence | Exact applied GenConfig, verified token split/source, problem/sample seeds, per-item context and decode. Existing historical JSON generally lacks these fields. |
| I-3.6 active parameters | expertCount/expertUsedCount may exist | `activeParameterCount` with source; total parameter count is not sufficient. |
| I-4.1 headroom | Per-PID peaks and optional planning in-use available | Immutable actual budget/reserve, in-use provenance and adapter/time alignment. |
| I-4.3 minimum RAM | **New** `BenchmarkRunResult.minRamAvailBytes` added at `bench-types.ts:224`; old telemetry samples can supply sampled minima | Old zero-row runs have no minimum; live OS guard minima, mmap credit, actual floor and coverage must be persisted to claim the true guard-observed minimum. |
| I-4.4 reclaimable cache | Cannot diagnose from file size/private WS alone | `ramAvailBeforeLoad`, system file-cache/mapped working-set counters, `ramAvailAfterUnload`, corresponding times and interference evidence. |
| I-4.5 stall | Peak alone is insufficient | Time series/plateau definition and device/driver/backend identity; current samples can sometimes support a local observation. |
| I-5.1/5.2/5.6 quality interval | Pass rows support a descriptive score; not the advertised rigorous band by themselves | Unique skill/item/instance/sample identities, complete suite manifest, interval method/version, valid-evaluation status and paired comparison records. |
| I-6.1 rep spread | **New** `repDecodeTps` exists at `bench-types.ts:222`; older rows store medians | Missing old individual reps cannot be recovered from medians; new list also needs per-rep status/source/warm/prompt/seed when comparing. |
| I-6.3 diagnostics | failureKind/reason exists; RunDetail may hold stderr | Consumer API currently drops some RunDetail (`storage/sessions.ts` listRuns); expose classified stderr/timestamps and all excluded failures explicitly. |
| I-6.4 warm filter | `warm` exists for newer rows | Missing old warm state is unknown, not true; enforce exclusion/qualification in the scorer itself. |
| I-7.1/7.4 comparable identity | One saved machine snapshot is available in newer sessions | Content/base-model/fine-tune/tokenizer/template hashes and exact GPU/runtime/settings; same arch/params is not identity. |

## Evidence that should remain in v2 of the guide

- The 14B 32K spill and 8B smooth 64K sweep justify retaining a conservative relative cliff threshold and adjusted-spill warning as starting heuristics, with clear scope and versions.
- Speed should be reported at the intended measured context and with actual prompt tokens; the original scripts tokenized filler to ~75%, while application Run A has 1167 tokens at 2048 (`session-run-A-2026-09-27T08-45-32-573Z.json:423-424`). Context allocation alone is not a comparable prompt workload.
- Run B's 783 ms cancellation and no leftover server, Run C's precheck skip, and the early heavy-run 1 GiB free-RAM incident are valid **individual observations**. None is universal proof of current guard timing or isolation.
- The guide's relative-signal/not-a-leaderboard language, measured/declared distinction, explicit required-context reporting, and insistence on explainable decisions are sound. Make the newly delivered qb-2.0.0 coverage/seed manifest part of the evidence contract rather than treating a larger raw item count as automatic credibility.
