# Origin validator contract — `validateQualityOrigin(row, ctx?)`

**One function, three callers:** the replay tool (`scripts/prove-quality-rows.ts`), interpretation (`src/core/interpret/verdicts.ts`, the I-8.0 contract check) and the storage read path (`src/core/storage/sessions.ts`, rows loaded for display or resume). No caller re-implements any rule below. A rule change is made here and in the function only, with a negative test for each caller.

## Input
From the quality row (`GenRow`):
- `proofProvenance`:
  - `mode`: `'runtime' | 'live-template-replay'`
  - `status`: `'original' | 'reconstructed'`
  - `originalPromptHashPresent`: boolean
  - `origin`:
    - `generationPromptHashPresent`: boolean
    - `firstReplayAt`: `string | null`
    - `lineage`: `string[]`
- `promptSha256?: string` — SHA-256 of the exact string sent to runPrompt.
- `renderProof?`: `{ rowId, promptSha256, renderedSha256, counterfactualSha256, keys, status }`.
- Row identity fields for `proofRowId(row)`: configId, genId, testId, sample, suiteSeed, generatorSeed.

`ctx` is optional:
- `ctx.expectedLineageTail?: string` — the runId of the artifact being replayed. The replay tool always passes it.
- `ctx.replay?: boolean` — true while the tool is producing a replay.

## Result
```ts
{ ok: boolean; classification: 'original' | 'reconstructed' | 'incoherent'; reason: string | null }
```
- **original** (`ok: true`) means every rule holds and the row is row-bound to its generation request. Only these rows can prove an effort/thinking comparison.
- **reconstructed** (`ok: true`) means the record is coherent but not bound to the original generation prompt. Examples: a legacy row with no provenance, `generationPromptHashPresent: false`, or a template re-render. It is a valid measurement, but **not evaluable** for I-8.0 claims.
- **incoherent** (`ok: false`) means some field contradicts another or is malformed. `reason` names the first failed rule.

## Coherence rules (all must hold; the first failure ⇒ incoherent)
- **No provenance:**
  - `proofProvenance` absent ⇒ **reconstructed** (a legacy row), never original.
  - `mode === 'live-template-replay'` with `origin` absent ⇒ incoherent.
- **Mode ↔ firstReplayAt:**
  - `runtime` ⇒ `origin.firstReplayAt === null`.
  - `live-template-replay` ⇒ `firstReplayAt` is a string with `new Date(x).toISOString() === x` (exact ISO round-trip; any other format ⇒ incoherent).
- **Mode ↔ lineage:**
  - `lineage` is an array, and every element is 64 lowercase hex characters.
  - `runtime` ⇒ `lineage.length === 0`.
  - `live-template-replay` ⇒ `lineage.length ≥ 1`, and when `ctx.expectedLineageTail` is given, `lineage.at(-1) === ctx.expectedLineageTail`.
- **Flag agreement:**
  - `originalPromptHashPresent === origin.generationPromptHashPresent`.
  - `status === 'original'` ⇔ `originalPromptHashPresent === true`.
  - `status === 'reconstructed'` ⇔ it is false.
- **Prompt hash:**
  - `promptSha256` present and 64-hex ⇔ `generationPromptHashPresent === true`.
  - `generationPromptHashPresent === false` with a `promptSha256` that claims generation origin ⇒ incoherent.
- **renderProof binding** (when present; absent ⇒ at most reconstructed):
  - `renderProof.rowId === proofRowId(row)`.
  - `renderProof.promptSha256 === row.promptSha256`.
  - `renderProof.renderedSha256 === row.promptSha256`.
  - `keys` is a string array.
  - `status ∈ {'proved','unproved','contradicted'}`, and `contradicted` ⇒ incoherent.
- **Classification:** all rules hold, `mode === 'runtime'`, `status === 'original'`, `generationPromptHashPresent`, and `renderProof.status === 'proved'` ⇒ **original**. Every other coherent case ⇒ **reconstructed**.
  - A live-template replay is never classified original, even when the rendered hash equals the stored prompt hash. It confirms the template, not the generation request.

## Consequences (every caller)
- **Incoherent ⇒ non-comparable everywhere.**
  - verdicts: the gen config is not comparable (I-8.0 missing proof), and the reason is disclosed.
  - storage read: the row is shown with its reason, is excluded from comparisons, and is never silently dropped.
  - replay tool: it rejects the artifact before any `/apply-template` request.
- **Never repaired.** No caller fills in, re-derives or normalises a missing or malformed field (a timestamp, lineage, flag or hash) to make a row pass. Replay output only appends lineage and a new `firstReplayAt`; it never edits the root generation record.
- Reconstructed rows remain valid measurements (quality, speed), but cannot prove effort/thinking deltas.

## Negative example — RECHECK8 T3 (docs/review-w4t-2026-09-28.md)
Before this contract, `verdicts.ts` checked only array presence and truthiness. An otherwise valid thinking-low row carried:
```json
{ "mode": "runtime", "status": "original", "originalPromptHashPresent": true,
  "origin": { "generationPromptHashPresent": true, "firstReplayAt": "not-an-iso-time", "lineage": ["not-a-hash"] } }
```
It was accepted as comparable. Under this contract it is **incoherent**, twice over: `runtime` requires `firstReplayAt: null` and an empty lineage, and `not-a-hash` fails the 64-hex rule. The same record with `mode: 'live-template-replay'` is also incoherent: the ISO round-trip fails, and so does the lineage syntax.

Tests: every caller gets this probe as a negative, plus a valid runtime-original control and a valid replay-reconstructed control.
