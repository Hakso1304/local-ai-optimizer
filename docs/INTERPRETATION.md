# Benchmark Result Interpretation Guide (v2)

Authors: Fable (orchestrator) with review by Worker #4 (GPT-6-Astra,
`docs/interpretation-astra-notes.md`). Status: **normative** for `src/core/interpret/`
and for every ranking decision in `src/core/scoring/recommend.ts`. Every rule has an
id; every recommendation reason cites one. Rules are data (`rules.v2.json`); thresholds
carry an `origin` tag: `measured-calibration` (with the calibration id), `policy`
(a product choice), or `heuristic` (a starting point awaiting evidence).

## Changes from v1 (what Astra's review corrected)

- The quality band is an **uncertainty heuristic**, not a proven 95 % interval; the
  unit of evidence is the unique skill/item, not the graded row (§5).
- Comparisons use a **decision trace**: the app records which gate, evidence basis and
  tie-break actually decided; "within band" neutralizes the quality delta explicitly
  instead of pretending speed decided (§7).
- **Unknown never passes a hard constraint** and an estimate-only candidate is never a
  confirmed winner; it lives in a separate provisional list (§1).
- A ceiling is **coverage, not proof**: `largestCleanTested` ≠ the machine's limit
  unless a failure or spill was observed above it (§2).
- Memory headroom names its **budget basis**; distances to floors are signed (§4).
- Spill is reported at the rung where it exceeded the threshold, with a decode delta
  only when adjacent comparable rungs exist; the WDDM 83 % observation is machine-local
  evidence, not a Windows law (§2, §4).
- Quality results carry an **evaluation status** (valid / infra_error / unrun /
  truncated); infrastructure failures never count as wrong answers (§5).
- Thinking-token splits are ESTIMATED unless the runtime reported them; comparisons
  are only made between comparable requests (§8).
- One shared **speed-eligibility predicate** enforces warm/provenance/finite rules
  before scoring; failure lists are built from all persisted runs (§6).
- "Same model, different quantization" requires identity beyond arch + parameter
  count; the partial-offload veto only applies against an alternative that meets the
  same hard constraints (§7).
- Actions are typed and conditional; panel priority puts critical safety/data/user-
  requirement failures first (§9, §10).
- New data contract (§12): fields the runner/storage must persist for a rule to be
  evaluable; a rule whose inputs are absent renders "not evaluable", never a guess.

---

## 0. Reading order

1. Critical: invalid data, safety aborts, device loss, unmet **required** context.
2. Coverage and ceiling: what was tested, where it stopped, and why (§2).
3. Quality with its uncertainty and coverage counts (§5).
4. Speed at the context you will use, with actual prompt tokens (§3).
5. Memory margin with its budget basis (§4).
6. Comparisons, always within one workload and one machine snapshot (§7).

Rule I-0.1 (`panel.priority`, policy): insight panels are ordered critical → coverage/
quality → speed → memory → comparisons. The Dashboard shows all critical insights even
if that exceeds its two-warning cap.
Rule I-0.2 (`data.scope`, policy): every evidence entry carries value, unit, kind,
source, sample count, window/timestamp, configId, ctx and rule/algorithm versions.
"MEASURED" means *observed with a stated scope*, not "beyond question".

---

## 1. Provenance and eligibility policy

| Kind | Meaning | Use in decisions |
|---|---|---|
| MEASURED | observed by the runtime/OS in a stated window | may confirm a winner |
| ESTIMATED | formula or fallback (memory estimates, wall-clock TPS, quality prior, character-split token counts) | may rank only among provisional candidates; never confirms |
| DECLARED | from the model file / runtime | describes the file, not this machine |
| UNAVAILABLE | not observed; reason recorded (`unavailableReason`) | "not verified"; never a pass, never 0 |

Rule I-1.1 (`prov.hard-constraints-first`, policy): user hard constraints
(requiredContext, minDecodeTps, safety) are evaluated before ranking. Evidence for a
hard constraint must be MEASURED; unknown ⇒ "not verified" ⇒ the constraint is not met.
Rule I-1.2 (`prov.confirmed-vs-provisional`, policy): a **confirmed** winner is
chosen only from candidates whose decisive components are MEASURED and comparable.
Candidates that depend on any ESTIMATED decisive component form a separate
**provisional** list, shown as such, with the estimated term named. No candidate's prior
is adjusted using another candidate's measurements.
Rule I-1.3 (`prov.decisive-trace`, policy): when a decision used an estimated or
unavailable input, the recorded decision trace names the term and the counterfactual
("without the quality prior, X and Y are indistinguishable").
Rule I-1.4 (`prov.unavailable-reason`, note): an unavailable metric is rendered with
its recorded reason (short step, dropped rows, localized counters, sampler failure,
not attempted) — never "the step was too short" by default.

---

## 2. Context coverage and the practical ceiling

Definitions. **Declared** context: the file's value. **largestCleanTested**: largest
rung with status pass. **firstObservedFailure**: lowest rung with fail/degraded, with
its structured reason. **firstPlannedSkip**: lowest rung the planner did not attempt,
with `{resource, estimateBytes, budgetBytes, ruleId}`. **limitKind** ∈ {failure, spill,
cliff, planned-skip:memory, planned-skip:ram, user-cap, cancelled, largest-tested,
unknown}. The **practical ceiling** is `largestCleanTested`; it is a lower bound on
capability unless `limitKind` is failure/spill/cliff.

Rule I-2.1 (`ctx.coverage`, info, always shown): "Largest clean context measured in
this run: 16K (declared 131K). Stopped because <limitKind>: <structured reason with
numbers>." When `limitKind` is largest-tested/user-cap/cancelled: "higher contexts were
not attempted".
Rule I-2.2 (`ctx.spill`, warn, threshold 256 MiB adjusted spill, origin
measured-calibration cal-2026-09-27 14B@32K): "adjusted shared-GPU usage exceeded
<t> at <ctx> (<spill> GiB)". Append "decode fell <a> → <b> t/s" only if the previous
rung is a comparable measured row. A single-sample excursion is provisional. Action:
`use-context <last clean rung>` only if one exists (never at a first-rung spill);
`enable-kv-q8`; `try-smaller-quant`.
Rule I-2.3 (`ctx.planned-skip`, note): planner skip → name the resource (VRAM or RAM),
the estimate and the budget from the persisted planning snapshot. Actions:
`enable-kv-q8` (≈47 % less KV than f16 for q8_0 blocks; nothing else shrinks),
`enable-heavy-mode` (KV or layers to RAM; slower), `lower-required-context`.
Rule I-2.4 (`ctx.declared-vs-tested`, note): largestCleanTested < 25 % of declared →
coverage note "only <x> of the declared <declared> was tested/usable in this run" —
it becomes a warn only when `limitKind` is failure/spill/cliff.
Rule I-2.5 (`ctx.required`, critical/info): requiredContext set → state per model one
of: reached (rung passed), tested-and-failed (reason), not-tested (why). Unmet ⇒ lead
insight with actions.
Rule I-2.6 (`ctx.recovered-dip`, note, thresholds = cliff engine's 0.60 ratio and 2 t/s
absolute, origin policy): "decode dipped at <ctx> and recovered on the next tested rung;
cause unverified". Missing rungs between are disclosed.

---

## 3. Speed

Definitions: decode t/s; prefill t/s; TTFT (wait to first token, ≈ prompt tokens ÷
prefill t/s at that rung); **effective answer t/s** for thinking configs (answer
tokens ÷ total seconds). Every speed number states its rung and actual prompt tokens.
No silent fallback: if recommendedCtx has no measurement, the value is UNAVAILABLE
with the reason; the reference rung used for scoring is named separately.

### 3.1 Decode bands (descriptive only; never a gate by themselves; origin policy)

| Band | decode t/s (half-open) | Reads as |
|---|---|---|
| unusable | < 3 | slower than reading; batch-only |
| patient | [3, 8) | one-off high-quality answers |
| usable | [8, 20) | large-scale coding/agent work where quality dominates |
| comfortable | [20, 40) | everyday chat/coding |
| snappy | [40, 100) | interactive assistants |
| very fast streaming | ≥ 100 | streaming speed no longer differentiates (TTFT may still) |

Rule I-3.1 (`speed.decode-band`, info): band + the **effective gate** ("Coding gate
10 t/s; your floor 20 t/s"). The gate is the workload's `minDecodeTps` or the user's
value when set; bands never gate.
Rule I-3.2 (`speed.thinking-effective`, warn, threshold 50 %, origin heuristic):
only when raw decode and effective answer t/s come from the same request/context and
both are positive: "reasoning consumed <r> tokens; effective <e> t/s vs <d> t/s raw".
Token split provenance is shown (runtime-reported = measured; character-split =
estimated).

### 3.2 TTFT bands (descriptive; the gate is the effective tolerance)

| Band | TTFT | Reads as |
|---|---|---|
| immediate | < 1 s | chat feel |
| short wait | [1, 5) s | coding turns |
| noticeable | [5, 20) s | long-context work, expected |
| long | [20, 60) s | batch; fine when the context was required |
| very long | ≥ 60 s | only with an explicit requirement |

Rule I-3.3 (`speed.ttft-band`, info/warn): band at the stated rung and prompt tokens.
Warn when above the **effective** tolerance (workload, or advisory when the user
required the context/large_coding — then phrased "accepted because required").
Rule I-3.4 (`speed.prefill-scaling`, note, threshold 0.5× per context doubling =
severe, origin heuristic; smooth calibration worst was 0.74×): reported with the actual
token ratio when occupancy differs between rungs; TTFT growth is stated from measured
TTFT, not inferred.
Rule I-3.5 (`speed.partial-offload`, warn, on expectDegraded configs): "<n>/<m> layers
on GPU: <decode> t/s" plus, only when a comparable measured full-offload or other
partial rung exists at the same context, the comparison ("-nkvo 7.4 vs 10.7 t/s at 2K"
— measured 2026-09-27, dense; 27.0 vs 38.5 MoE). No universal claim.
Rule I-3.6 (`speed.moe-note`, info): expertCount > 0 → "mixture-of-experts:
<expertUsedCount>/<expertCount> experts per token; partial offload tends to cost less
than for a dense model (one cross-family observation: 3.6× on 2026-09-27)". Active
parameter count is shown only if the file declares it.

---

## 4. Memory

Rule I-4.1 (`mem.budget-basis`, info): headroom is always stated with its basis:
(a) planning budget remaining = planningVramBudgetBytes − planningReserveBytes − peak
dedicated (from the persisted planning snapshot), (b) adapter free VRAM measured in the
same window, or (c) per-PID dedicated vs adapter total. Missing inputs ⇒ UNAVAILABLE.
Warn (threshold 0.5 GiB, origin policy): "less than <x> GiB of the planning budget
remains at <ctx>; less room for other apps, spill risk increases".
Rule I-4.2 (`mem.in-use-at-plan`, note, threshold 1.5 GiB): VRAM in use by other apps
at planning > 1.5 GiB → "planned while <x> GiB was already in use (measured at
<time>); rerun on an idle GPU for the full budget". Unknown in-use ⇒ "assumed
<default> GiB (estimated)".
Rule I-4.3 (`mem.ram-floor`, warn/critical): signed distance `minRamAvailBytes − floor`
(never clamped), the floor value, and the mmap credit separately; guard_abort ⇒
critical with the recorded reason.
Rule I-4.4 (`mem.mmap-note`, info, conditional): only when ramAvailBeforeLoad,
during-load minimum and after-unload values exist: "RAM drop of <x> GiB during load is
consistent with the file cache (mmap) and was released after unload". Otherwise omit.
Rule I-4.5 (`mem.saturation-observed`, note): per-PID dedicated plateau observed at
<p> % of VRAM across ≥ 3 samples before spill → "on this machine spill began at
≈<p> % (measured <date>)". No generalization to other machines.

---

## 5. Quality

Q = 100 · Σ_c W_c · passRate_c / Σ W_c over categories with valid results.
Coverage counts are published with every Q: unique skills, unique items, generator
seeds, completions, samples per item, and the suite manifest id.

Rule I-5.1 (`quality.report`, info): "Q <value> (uncertainty ± <u>, heuristic;
n=<items> items / <skills> skills / <samples> samples) — categories: …". The band is
labelled a heuristic until `intervalMethod`/`intervalVersion` with a documented
coverage assumption is stored; then it is labelled by that method.
Rule I-5.2 (`quality.difference`, info/warn): two candidates are compared on the same
items/seeds; the **difference** and its uncertainty are reported ("+9 [+2, +16]"). When
the difference interval includes 0: "insufficient evidence to distinguish quality on
this suite" and the decision trace shows the quality delta neutralized (not "speed
decided"). Action: `run-thorough-quality`.
Rule I-5.3 (`quality.category`, note): category pass rate ≤ 1/3 (inclusive) is named;
for coding workloads ≤ 2/3 in `coding` is a warn; both only when the category has ≥ 3
valid items, else "insufficient coverage".
Rule I-5.4 (`quality.thinking`, note): the gen config used is stated; if a thinking
config was measured, both Q values are shown with their effective speeds (§8).
Rule I-5.5 (`quality.estimated`, warn): quality is a prior → the candidate is
provisional (§1); the prior's basis is named.
Rule I-5.6 (`quality.coverage`, note, policy): < 30 unique completed items OR any
required category with < 3 valid items → "limited coverage" (a policy warning, not a
confidence statement).
Rule I-5.7 (`quality.harness-invalid`, critical): any item with evaluationStatus
`infra_error` (sandbox/evaluator/template failure) → the affected quality and gen
comparisons are quarantined (excluded, shown as invalid); performance rows from valid
phases are kept.
Rule I-5.8 (`quality.truncated`, warn): items with `outputTruncated` are failures under
the declared token budget and are listed separately with the budget; action:
`try-thinking-config` (lower effort) or a larger budget note.
Rule I-5.9 (`quality.not-a-leaderboard`, info, once): "relative signal for these
candidates on this suite, not a leaderboard".

---

## 6. Speed eligibility, stability and failures

Rule I-6.0 (`speed.eligible`, policy): one shared predicate decides whether a row may
enter speed scoring or speed insights: status pass/degraded, `warm === true` (unknown ⇒
not eligible, labelled "warmup not recorded"), decode finite and > 0, MEASURED
provenance, same benchmark/prompt versions as the session.
Rule I-6.1 (`stab.rep-spread`, warn, threshold (max−min)/median > 0.15 with ≥ 2 valid
reps, origin heuristic): show the reps; action `rerun-idle`. No thermal/contention
cause is asserted.
Rule I-6.2 (`stab.versions`, warn): mixed runtime/suite/prompt/rules versions → name
them; comparisons across versions are labelled `rerun-comparable`.
Rule I-6.3 (`stab.failures`, warn/critical): built from **all persisted runs** of the
session (including models with no usable row), superseded retry rows marked; device_lost
⇒ critical "results after <time> are suspect".
Rule I-6.4 (`stab.cold-rows`, note): rows excluded by I-6.0 are listed with the reason.

---

## 7. Comparing candidates

Rule I-7.1 (`cmp.scope`, policy): comparisons are valid only within one workload, one
machine snapshot, comparable prompt tokens/context, KV type, template and gen config.
Rule I-7.2 (`cmp.decision-trace`, policy): every recommendation stores {hard
constraints evaluated, eligible set with failing rule ids, comparison basis per
component (rung, provenance), quality-difference interval, neutralizations applied,
tie-break chain, rules version, thresholds used}. Reasons are rendered from the trace.
Rule I-7.3 (`cmp.why-not`, info): each why-not sentence carries, in order: quality
difference with uncertainty, speed at the reference rung, ceiling/coverage, failed gate.
Rule I-7.4 (`cmp.quality-vs-speed`, info): winner slower but higher quality → "chosen
for quality: +<d> [lo, hi] … <k>× slower". Winner faster with a within-band quality
difference → "quality indistinguishable on this suite; speed/memory decided".
Rule I-7.5 (`cmp.identity`, policy): "same model, different quantization" requires
matching base-model identity (repo/base + fine-tune) or a content fingerprint, not just
arch + parameter count; otherwise "related models" wording. Preferring the smaller
quant requires it to meet the same hard constraints and a declared non-inferiority
margin (none is asserted yet).
Rule I-7.6 (`cmp.partial-veto`, policy): a partial-offload config is dominated only by
an alternative of the same model that satisfies the **same** hard constraints (incl.
required context); the 14B 32K observation (26.4 vs 5.6 t/s) is evidence for that
comparison, not a universal veto.

---

## 8. Generation configuration

Rule I-8.0 (`gen.comparable`, policy): a gen comparison stores the applied template
kwargs and sampling values as accepted by the runtime, template/runtime/model hashes,
prompt instance, context, token budget, sample seeds, token-count source and timings.
A requested flag is not proof it was honored (`appliedTemplateKwargs` is required).
Rule I-8.1 (`gen.best-config`, info): per model: "thinking on (effort low, T=1.0):
Q +17 [lo, hi] vs off; answers <k>× slower (effective <e> vs <d> t/s)".
Rule I-8.2 (`gen.stochastic`, note): T > 0 → "sampled (seeded), n=<s> per item".
Rule I-8.3 (`gen.effort-saturation`, note): a higher effort adds reasoning tokens with
a quality difference whose interval includes 0 → recommend the lower effort.

---

## 9. Actions (typed, conditional)

`use-context <ctx>` (only with a known clean rung) · `enable-kv-q8` (only when KV is the
limiting resource) · `enable-heavy-mode` · `lower-required-context` ·
`try-smaller-quant <file>` (verified identity; "not measured here") ·
`try-thinking-config` · `run-thorough-quality` · `rerun-idle` · `rerun-comparable`
(version mismatch) · `retry-telemetry` / `inspect-diagnostics` (missing telemetry,
sampler errors) · `download <model>` (labelled unmeasured).
Rule I-9.1: warn/critical insights carry ≥ 1 action whose precondition holds; never
suggest `raise-min-decode` as a remedy for slow hardware.

---

## 10. Panel layout

Dashboard card: headline · coverage line (I-2.1) · quality with coverage counts (I-5.1)
· decode band with effective gate (I-3.1) · all critical insights, then ≤ 2 warns.
Results → Interpretation: sections 1–8 in priority order; evidence expandable with
scope; decision trace viewable; actions as buttons where executable. Historical
recommendations show "recommended with rules <v>; reinterpreted with rules <w>".

---

## 10a. Gate rules added by the engine (origin policy; adopted into the guide)

Rule I-2.7 (`gate.context-floor`, warn): largest clean context below half the
workload's target → ineligible; action `lower-required-context`.
Rule I-3.7 (`ctx.recommended`, info): "Recommended -c <ctx>: <why>" — the rung the
export uses and the reason it was chosen (from the decision trace).
Rule I-5.10 (`gate.quality-min`, warn): measured quality below the workload minimum →
ineligible; action `run-thorough-quality`.
Rule I-6.5 (`gate.stability`, warn, min 50): stability component below 50 →
ineligible; action `rerun-idle`.

## 11. Threshold provenance

| Threshold | Value | Origin |
|---|---|---|
| adjusted spill warn | 256 MiB | policy informed by cal-2026-09-27 (CAL-14 14B@32K 1.05 GiB raw per-PID shared; CAL-S 8B ≤ 0.12 GiB adapter delta) — not validated for adjusted-v2 |
| decode cliff | ratio ≤ 0.60 and ≥ 2 t/s | policy, consistent with 14B (0.516) vs 8B smooth (≥ 0.73) |
| decode/TTFT bands | §3 | policy |
| prefill severe scaling | 0.5× per doubling | heuristic (smooth worst 0.74×) |
| thinking effective | 50 % | heuristic |
| budget headroom warn | 0.5 GiB | policy (8B residual 0.17–0.89 GiB observed) |
| in-use at plan | 1.5 GiB | policy (idle other-app usage ≈ 1.2 GiB observed in CAL-S idle adapter readings) |
| rep spread | 15 % of median, ≥ 2 reps | heuristic |
| limited coverage | < 30 unique items or < 3 per category | policy |
| category weak | ≤ 1/3 (≤ 2/3 coding) | policy |

---

## 12. Data contract (fields a rule needs; absent ⇒ "not evaluable")

Per run row: status, failureKind, structured `skip {resource, estimateBytes,
budgetBytes, ruleId}`, `warm`, versions {benchmark, prompts, quality, runtime, rules},
promptTokens (actual), repDecodeTps[], minRamAvailBytes (signed vs floor), peak per-PID
dedicated/shared raw/adjusted + hostPinnedBytes + spill algorithm version, sampler
errors, timestamps.
Per session: planning snapshot {vramTotal, vramInUse (kind), planningVramBudgetBytes,
planningReserveBytes, ramFloorBytes, candidateRulesVersion}, machine snapshot with GPU
identity, stop reason (done/cancelled/paused/interrupted/user-cap), rules version.
Per quality item: testId, skillId, generatorSeed, sample, genId, evaluationStatus
(valid | infra_error | unrun | truncated), outputTruncated, maxTokens, checkerVersion,
answerTokens/reasoningTokens with source (runtime | estimated), promptTokens, ctx,
timings.
Per recommendation: decision trace (I-7.2), insights, provisional flag with named terms,
rulesVersion, thresholds used.
Model identity: arch, parameter count, quant, file fingerprint, base-model/repo id
when known, template hash, expertCount/expertUsedCount.
