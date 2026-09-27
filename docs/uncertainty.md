# Quality uncertainty (`unc-1`)

These are **uncertainty heuristics until validated**, with nominal level 0.95.
They describe evidence on this suite, not general model ability or leaderboard rank.
The module is pure TypeScript and does not run inference or access storage.

## Evidence and coverage

Pass/fail is binary (`score` is not substituted for `pass`). Identity is
`[testId, generatorSeed]`; missing seed differs from seed 0 and numeric seeds differ
from strings. Repeats collapse to their item's mean, never increasing independent n.
Supply one row per actual completion; sample IDs are metadata, not deduplication keys.
Missing skillId defaults to testId; missing evaluationStatus defaults to valid.
Truncated output counts as a failure even if pass=true; unrun rows are excluded.
Any infra_error quarantines inference: scoring, pairing and flags throw an error.
`coverage` remains available to explain this invalid state.

`coverage` returns uniqueItems/uniqueSkills across all supplied rows, distinct known
seeds, validItems with at least one evaluated completion, and categoriesCovered with
such items. Completions counts valid/truncated graded samples; infraErrors and
truncated count rows. samplesPerItem maps JSON item keys to those completion counts,
including zero for entirely unrun/invalid items. Truncated items are valid evaluated
evidence under their declared budget. These counts include zero-weight evidence;
statistical n and flags exclude zero-weight items/categories.

## I-5.1: one candidate

`qualityUncertainty(rows, categoryWeights)` returns Q and lower/upper in [0,100],
method, version, unit, level and n. Q weights item means within each category by
item weight, then weights the observed categories by supplied category weights.
Absent/zero category weights exclude that category; missing categories are not zero.
Empty positive-weight evidence throws rather than returning invented zero quality.

Independent items use Wilson with Kish effective item count for unequal weights;
reported n is the actual unique-item count. Multi-item skills use an 8192-draw,
fixed-seed skill bootstrap and report skill n. Original category/item weight masses
are preserved inside resampled blocks; category composition may vary across draws.
One multi-item skill gets [0,100]. Bootstrap all-equal outcomes can yield a point
band: this is a limitation, not proof of certainty. Full formulas are in the module
header. Independence across items/skills, never across repeats, is assumed.

UI example: `Q 80 [58, 93] — uncertainty heuristic; 30 items / 12 skills / 60 samples`.
Prefer endpoints because bands are asymmetric; if ± is required, use the larger
distance to Q and retain exact endpoints in details. Display method/version and
category coverage. `categoryFlags` returns all six categories with rate (0..1 or
null), validItems and flags: inclusive weak <=1/3, coding warning <=2/3, and only
“insufficient coverage” below three items. Apply the coding warning to coding
workloads. The caller adds I-5.6's <30-item/required-category coverage policy.

## I-5.2: comparing candidates

`pairedDifference(A,B,weights)` matches only testId+seed, collapses repeats, and
bootstraps per-item A-B differences by skill. It returns diff/lower/upper in percentage
points [-100,100], sharedItems and method. Below five positively weighted matches,
or below two shared skills, it returns `{interval:null, reason, sharedItems}` instead
of a numeric result. Check `'interval' in result` before using interval helpers.
Identical paired outcomes yield [0,0]; this does not establish general equivalence.

`includesZero` is inclusive: show “insufficient evidence to distinguish quality on
this suite” and neutralize the quality delta in the decision trace. `nonInferior`
checks lower >= -margin; callers must declare the margin in percentage points.
Neither helper decides the recommendation. No default noninferiority margin exists.

Callers must partition by suite version, candidate, context, generation config and
measurement scope, then enforce §7 comparability before comparing. Mixed genId
values within either input are rejected; different genIds across A/B are permitted
for explicit generation-config comparisons. Conflicting matched item skill/category/
weight metadata is rejected. These helpers cannot validate machine snapshots,
templates or prompt occupancy from quality rows. Persist method/version and those
comparison inputs alongside the decision trace. A null interval or thrown validation
error must be shown as unavailable/invalid, never converted to zero or MEASURED.
