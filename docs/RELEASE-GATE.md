# RELEASE GATE — must be green before the final package

Every box is checked by the orchestrator, citing a commit, file or output, before `npm run package` produces the final (non-nightly) build. A box that cannot be checked is a **stop**, not a waiver. Nightly/pre-verdict builds are exempt and say so in `LAO_BUILD_LABEL`.

## Code and review
- [ ] **G08 fixed:** `tests/scoring/interp2-rereview.test.ts` "equal quality: the lower effort is kept whatever the input order" passes. Its fixtures carry per-key `templateKwargProof` (842d396). The rule was not relaxed.
- [ ] **Astra RECHECK5** (`docs/review-w4p-*.md` or successor) verdict is at least **"conformant for ranking/generation, memory advisory"**, and every HIGH finding is closed or explicitly downgraded by Astra in writing.
- [ ] Every RECHECK4 O1–O5 follow-up routed on 2026-09-28 has landed:
  - the legacy `backendKind:'cpu'` `stateOf` match;
  - the O1 absorbed first-rung baseline disclosed on runs and in I-2.2;
  - the I-2.8 calibration string, with no "after a previous large load" cause (E-12 contradicted).

## Tests and build (idle GPU lane only: no S3 lease, no llama-server/typeperf running)
- [ ] `npx tsc --noEmit -p tsconfig.json` → exit 0.
- [ ] Full `npm test`, run once on the idle lane, passes with 0 failures. Known flakes (`tests/inference.test.ts` fake-server timing) re-run in isolation and pass. A flake is recorded in STATUS, never skipped.
- [ ] **Windows-hide scan:** `tests/windows-hide.test.ts` passes. Every spawn/exec/execFile in `src/` and `scripts/` passes `windowsHide: true`.
- [ ] `npm run build` → success.
- [ ] `npm run package` → `dist/BUILD-INFO.txt` shows:
  - the commit, **not DIRTY**;
  - the rules version `interp-2`;
  - the test line reading `N/N passed`, or `P/N passed, K skipped` with P + K = N, with no FAILED;
  - every skipped test is named on its own `skipped:` line (file :: test, with its reason); an unnamed skip is a stop;
  - an `origin:` line with origin/master at build time; if origin is ahead of the built commit, the delta must be
    docs-only (BUILD-INFO says so), otherwise stop.
- [ ] `LAO_HIP_VERIFIED=1` is set **only if** S3 stage 4 (`llama-server --list-devices` shows ROCm0) **and** stage 5 (the Vulkan-vs-HIP A/B) both passed, with their rows in `calibration-overnight-2026-09-28.md`. Otherwise BUILD-INFO must say HIP unverified, and the HIP install stays opt-in.

## Docs and evidence
- [ ] `docs/STATUS.md`: every row is current to the release commit (status, evidence commit, caveat). No row cites a superseded fix.
- [ ] `docs/EVIDENCE.md` is reconciled with `docs/calibration-ledger.md` and `docs/calibration-overnight-2026-09-28.md`:
  - E-12 is marked contradicted;
  - E-21 is supported per phase (session 4);
  - E-29/E-31 are UNVALIDATED pending the O1 rerun;
  - session 4 and 5 rows are present;
  - every row cited in MORNING-REPORT resolves to an E-id.
- [ ] **LIMITATIONS lists the three known caveats:**
  1. Spill/placement attribution is heuristic. I-2.8 is not calibrated, and first-rung residuals under 1 GiB are absorbed as the baseline.
  2. The learned per-process VRAM budget is advisory until qualified observations exist. Estimates never prune, so plans on this card may still include rungs that spill.
  3. The HIP backend is unverified on hardware unless `LAO_HIP_VERIFIED`. Other backends (ROCm via Ollama/LM Studio) may allocate differently, and the app measures only what it runs.
- [ ] README/ACCEPTANCE mention nothing the gate did not verify.

## Tree
- [ ] `git status --porcelain` is empty: no uncommitted or untracked files, and no stray `docs/review-*` drafts.
- [ ] The orchestrator has pushed. `git log origin/master -1` equals the packaged commit, or is a descendant of it whose
  delta is docs-only (recorded in BUILD-INFO `origin:`; decided 2026-09-28).
