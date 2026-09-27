# Quality suite v2 integration contract

`src/core/quality/tests.v2.json` is the **qb-2.0.0 manifest**, not a drop-in `QualityTestSet` cast: 47 static `QualityTest` objects plus 13 `{generator, params}` references. Total: instruction 12, reasoning 12, coding 12, structured 8, extraction 8, context 8. Category weights remain .20/.25/.25/.10/.10/.10. Every skill has 2–3 variants of equal weight; `difficulty: 1|2|3`, `skill` and `variant` are additive metadata. Tiers are author-assigned, not measured model difficulty.

## Exact hook for #3

Before `buildQualityPrompts` **and** `evaluate/evaluateAsync`, resolve the selected manifest once:

```ts
import suiteV2 from './tests.v2.json'
import { resolveSuiteV2, type V2Manifest } from './generators.v2'

const resolved = resolveSuiteV2(suiteV2 as V2Manifest, suiteSeed)
const prompts = buildQualityPrompts(resolved, buildOptions)
const testsById = new Map(resolved.tests.map(t => [t.id, t]))
// For each returned prompt/output, use testsById.get(prompt.testId) in evaluateAsync.
// Never look up a v2 prompt's checker in defaultTestSet (v1) or the raw manifest.
```

Extend the runner's selected-suite path rather than calling a generator only while expanding a prompt: **the generated checker must come from the same instance as the prompt**. Static entries already work with the existing `QualityTest` shape. Generator entries are a separate discriminated raw type (`GeneratorEntry`); `resolveSuiteV2` returns a real `QualityTestSet` with concrete prompts and the existing checker vocabulary. No change to v1 is made here. If exposing this through `index.ts`, import the resolver normally; the resolver's imports of index/checker types are type-only, avoiding a runtime import cycle.

The registered names are `arithmetic`, `wordProblem`, `extraction`, `schemaRecord`, `codeConstants`. Each exported function accepts `(seed: uint32, params?: {variant?: 1|2|3})` and returns a `QualityTest` plus metadata. Expectations are computed from the generated operands/entities/dates/field names/constants. `seedForItem(suiteSeed, item.id)` makes generation order-independent; never include model ID, generation-config ID, or current time in this derivation. All competing models/settings must receive the same resolved items.

Persist **suite id, generator version `qbg-2.0.0`, suiteSeed, item id, instanceSeed, skill, variant, difficulty, resolved prompt/checker (or their hash plus immutable generator source), actual promptTokens/context and generation parameters**. Retain raw output, stop reason, evaluation result, and sample index/seed. Resume only the exact manifest/seed/generator/config combination. Prompt seed and sampling seed are different: repeated samples use the same problem; fresh problem variants use a new recorded suite seed. Do not mistake the current `QualityPrompt.seed:1` for a problem seed.

Use `evaluateAsync`/`runCheckerAsync` for real model-written JS. Synchronous checker calls in the tests evaluate only our small trusted canned programs. The manifest never includes canned solutions; they live exclusively under `tests/quality-v2`.

## Credibility and scoring

- Run all 60 items for the v2 score; never mix v1/v2 or silently renormalize an interrupted subset into a complete result. Report completed/expected counts and category/skill coverage. A single passing item is at most 1/12 or 1/8 of its category.
- Reasoning permits explanations and requires the final `Answer: X` line. Generated numeric answers use anchored inner regexes, avoiding `number`'s last-number acceptance of several guessed answers. Static text answers use `finalAnswer` with exact matching.
- Code cases cover empty inputs, negatives, duplicates, boundaries, types and input preservation where relevant. JSON extraction/conversion checks content, not just schema. Generated schema tasks use supported type/required/enum constraints to verify named values, and explicitly allow additional properties because the current validator cannot forbid them.
- Eight context documents test lookup, revised facts, two-hop retrieval and resistance to irrelevant instructions. They contain intervening archive records and distinct answers. They are **fixed documents, not evidence of 32K–128K recall**. Tokenize the rendered prompt before loading; choose an actually measured context with space for output, or mark the item unrun. Never truncate away the target or silently shorten the document. Long-context testing needs separately recorded scaled-document instances and actual occupancy; `fillerTokens` does not resize these concrete prompts.
- Forty-seven static items are public and can be memorized; the 13 generated slots reduce dependence on fixed operands/answers, not all contamination. Use a newly chosen, persisted suite seed for a fresh assessment and share it across candidates. Reusing public seed 1234 is a regression fixture, not an unseen evaluation.
- Variants of a skill and repeated samples are correlated. Report unique skills/items/seeds and samples separately; do not call 3 samples of 60 items 180 independent tasks. Do not let a tiny or incomplete category give an overconfident total band. No model-quality calibration or empirical difficulty claim is made by passing these authoring tests.
- The retained checker vocabulary has limits: finite JS probes cannot establish arbitrary-program correctness, and code runs in the same JS realm as its harness and can tamper with built-ins. V2 improves task coverage; a separately hardened harness remains necessary against intentionally adversarial code. Existing `finalAnswer` tolerates the last Answer line even if later prose exists; the runner must not claim stricter output validation than that checker implements.

## Validation

`npx vitest run tests/quality-v2` runs every static item through `runChecker` with a canned good output, a plausible wrong output and blank output. Generator tests independently solve question text across seven seeds, check manually verified seed-1234 witnesses, realistic coding mutations, async sandbox execution, replay metadata, seed/order determinism and manifest immutability. **Final targeted run: 2 files / 69 tests passed (19:38:43 local).** `npx tsc --noEmit --pretty false` initially passed; the later whole-repository run reported only `tests/scoring/required.test.ts(3,31): TS6196: QualityResult is declared but never used` after concurrent worker edits. That existing file was left untouched under the new-files-only rule. No GPU/model benchmark was performed.
