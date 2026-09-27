# Stage 3 generation sweep preflight (2026-09-28)

Status: prepared, not run. GPU owner: Opus #3 until explicit handoff. The stage 2 A/B and its cleanup must finish before these commands start.

## Conditions

- Use a fresh `git archive` of a committed HEAD that includes `b868e3a` and the required S1 source fixes. Record the full HEAD hash in each result. Never run from the shared mutable worktree.
- Confirm zero `llama-server.exe` and zero `run-session` / `ab-spill` harness processes, no other GPU owner, at least 4 GiB RAM available, and idle adapter use near the clean-host baseline (about 1.1–1.2 GiB). Record the observed value, including a higher value if the host is busy.
- Reject every casing of `GGML_CUDA_ENABLE_UNIFIED_MEMORY` in the parent environment. The child environment must also omit every casing. All child processes use `windowsHide: true`.
- Use `--request-cap-ms 300000 --ram-abort-gib 4 --quality-mode thorough`. Keep the agreed explicit `--quality-seed 424242` in all three sessions so the 13 generated qb-2.0.0 items are identical. Each T=1 config has three samples per item.
- Run Gemma first. Run Qwen off/low/medium next, then off/xhigh. `genConfigsFor` caps a session at three configs. The second off is the within-session template comparator required to establish `appliedTemplateKwargs` for xhigh [I-8.0]. Compare Qwen sessions only after checking suite, seed, template hash, model fingerprint, runtime, context, token budget and accepted sampling.
- Pin the known clean 8K placements: Gemma-4 `ngl19` and Qwen3.8 `ngl49`. `--max-per-model 1` without pins could select a degraded full-GPU candidate and leave quality unrun. The pin's cloned planning estimates are experiment metadata, not calibrated planner limits [I-4.0].

## Commands from the immutable snapshot

```powershell
$db = Join-Path $env:APPDATA 'local-ai-optimizer-dev\optimizer.db'
$cap = @('--request-cap-ms','300000','--ram-abort-gib','4','--quality-mode','thorough','--quality-seed','424242','--db',$db)

npx tsx scripts/run-session.ts H --workload coding --heavy --ladder 8192 --models gemma-4-26B-A4B-it-UD-Q4_K_M --pin '[{"model":"gemma","ngl":19}]' @cap --gen-configs '[{"id":"off","thinking":false,"temperature":1,"topP":0.95,"topK":64,"source":"model-card"},{"id":"think","thinking":true,"temperature":1,"topP":0.95,"topK":64,"source":"model-card"}]'

npx tsx scripts/run-session.ts H --workload coding --heavy --ladder 8192 --models Qwen3.8-27B-UD-Q4_K_M --pin '[{"model":"Qwen3.8","ngl":49}]' @cap --gen-configs '[{"id":"off","thinking":false,"temperature":1,"topP":0.95,"topK":20,"minP":0,"source":"model-card"},{"id":"low","thinking":true,"effort":"low","temperature":1,"topP":0.95,"topK":20,"minP":0,"source":"model-card"},{"id":"medium","thinking":true,"effort":"medium","temperature":1,"topP":0.95,"topK":20,"minP":0,"source":"model-card"}]'

npx tsx scripts/run-session.ts H --workload coding --heavy --ladder 8192 --models Qwen3.8-27B-UD-Q4_K_M --pin '[{"model":"Qwen3.8","ngl":49}]' @cap --gen-configs '[{"id":"off","thinking":false,"temperature":1,"topP":0.95,"topK":20,"minP":0,"source":"model-card"},{"id":"xhigh","thinking":true,"effort":"xhigh","temperature":1,"topP":0.95,"topK":20,"minP":0,"source":"model-card"}]'
```

Leave at least 60 seconds idle between sessions. Record every session ID and dump path. If a RAM or request abort occurs, keep the partial dump and identify the incomplete config; do not turn missing items into failures [I-5.7].

After each session finishes, export its persisted evidence from the app DB into a distinct owned JSON file:

```powershell
npx tsx scripts/export-session-evidence.ts --session <session-id> --db $db --out docs/session-evidence-stage3-<gemma|qwen-main|qwen-xhigh>-2026-09-28.json
```

Run this inside the same snapshot so `HEAD.txt` supplies the full source hash. `run-session.ts --db` writes to app storage, so the dump's local `db.runs` and `db.quality` arrays remain empty. The export reads `benchmark_session`, **all** `benchmark_run` attempts, `quality_result`, and recommendations read-only. Use those persisted payloads to verify `acceptedSampling`, `templateHash`, requested/applied kwargs and token-count provenance; never infer them from the summary or transcript.

## Result record for `docs/calibration-overnight-2026-09-28.md`

For each session record: full snapshot HEAD, start/end time, workload, model file and fingerprint, backend/device/runtime build, VRAM in use at planning, env-key check, request cap, RAM floor, suite ID and seed, context rung and actual prompt tokens, quality item/skill/sample counts, min RAM by ladder/quality phase, `promptCacheSeen`, and abort/cleanup status.

For each gen config record: requested thinking/effort and sampling; runtime `acceptedSampling`; requested and `appliedTemplateKwargs`; template hash; valid / `infra_error` / truncated / unrun counts; Q and its heuristic interval; measured raw decode and effective answer t/s; reasoning/answer token split with its provenance; max token budget; and the paired quality-difference interval against that session's off config. Mark a comparison not evaluable when required application or identity fields are absent [I-8.0]. Do not attribute a quality difference to an effort setting when its interval includes zero [I-8.3].
