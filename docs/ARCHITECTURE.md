# Architecture — Local AI Optimizer

This document describes what the code does today. The design record and its rationale are in `docs/DESIGN.md`. Metric and scoring definitions are in `docs/BENCHMARK.md`.

## 1. Components

```
renderer (React, sandboxed)            main (Electron, Node 24)                     core (plain TS, no Electron)
────────────────────────────           ────────────────────────────                 ──────────────────────────────
Dashboard / Benchmark / Models   ──►   preload: window.api (contextBridge)   ──►   system/scanner.ts      hardware scan
Results / System pages                 ipcMain.handle(...) in main/index.ts        runtimes/*             llama.cpp, Ollama, LM Studio
benchState.ts (event fold)      ◄──   'bench:event' channel (sendBenchEvent) ◄──   benchmark/session.ts   session runner
                                       settings.json (userData)                     benchmark/candidates.ts  config generation
                                       optimizer.db (node:sqlite, userData)         telemetry/sampler.ts   typeperf sampler
                                                                                    quality/*              quality suite + sandbox
                                                                                    scoring/*              cliffs, scores, recommendation
                                                                                    models/gguf.ts         GGUF header reader
                                                                                    storage/db.ts, sessions.ts  SQLite schema + queries
shared/ (types only): types.ts (scan, IPC API, stored payloads), bench-types.ts (benchmark/scoring), bench-events.ts (live events)
```

| Layer | Files | Rule |
|---|---|---|
| Renderer | `src/renderer/src/*.tsx`, `benchState.ts` | Can only reach main through `window.api`. Uses `sandbox: true` and `contextIsolation: true`. |
| Preload | `src/preload/index.ts` | Maps each `RendererApi` method to one `ipcRenderer.invoke` channel. `onBenchEvent` subscribes to `bench:event`. |
| Main | `src/main/index.ts` | Wires IPC to core, owns the single `LlamaCppBackend`, the DB handle and settings. Runs no benchmark logic of its own. |
| Core | `src/core/**` | Node-only modules (child_process, fs, sqlite). Scoring (`core/scoring`) and candidate generation are **pure**: no I/O, no clock, no randomness. |
| Shared | `src/shared/*.ts` | Types only, no runtime code. Safe to import from the renderer. |

## 2. Data flow

**Scan.** `system:scan` calls `scanSystem()` (`core/system/scanner.ts`): one PowerShell call that emits JSON. VRAM comes from registry `qwMemorySize`, and adapters are joined to present PnP devices so stale registry GPUs are dropped. The call also runs `detectRuntimes()`. It returns a `SystemProfile` in which every section is `Sourced<T>`, and a failing section only marks itself `unavailable`.

**Models.** `models:list` calls `LlamaCppBackend.enumerateModels(dirs)`, which calls `findGgufModels` (`core/models/gguf.ts`). The dirs are `<app>/models` plus `settings.modelDirs` (default `D:\llm-models`). Each `ModelInfo` carries `meta` (GGUF header, streamed in 1 MiB windows) or `metaError`.

**Benchmark (live).** The renderer calls `startBench(SessionRequest)`, which goes to `bench:start`. Main then calls `runSession(req, deps, emit)` (`core/benchmark/session.ts`). Each emitted `SessionEvent` goes through `sendBenchEvent` to every window on `bench:event`, and the renderer folds them in `benchState.ts`. **Status:** `bench:start` / `bench:cancel` still return `runner not wired yet` (`main/index.ts`). The runner and the event contract exist and are tested, but the main-side wiring does not exist yet.

**Results.** `sessions:list`, `sessions:get` and `recommendation:latest` read `core/storage/sessions.ts`. `getSession` recomputes `detectCliffs` from the stored runs, so cliff verdicts always reflect the current code.

**Smoke.** `bench:smoke` runs one tiny real request: ctx 2048, 32 tokens, one warmup, on the discrete Vulkan device chosen by `pickDiscreteDevice(listDevices())`.

## 3. InferenceBackend abstraction

`src/core/runtimes/types.ts`:
```ts
interface InferenceBackend {
  readonly id: 'llamacpp' | 'ollama' | 'lmstudio'
  detect(): Promise<RuntimeDetection>          // never throws
  enumerateModels(dirs): Promise<ModelInfo[]>
  loadModel(cfg: LoadConfig): Promise<LoadResult>   // resolves when the model answers; throws on failure
  unloadModel(): Promise<void>
  runPrompt(req: PromptRequest, onToken?): Promise<PromptResult>  // request failures go in .error, never thrown
  getRuntimeStats(), configure(opts), cancel(), healthCheck()
}
```
- **llama.cpp** (`runtimes/llamacpp/index.ts`, parsing in `parse.ts`) is the only backend that can benchmark. `loadModel` spawns `llama-server` on a free loopback port with `-c`, `-ngl`, `--device <id>`, `-fit off`, `--parallel 1`, `-lv 4` and `--metrics`. It polls `/health` for up to 120 s, then checks that `/props.model_path` names our model (so a port squatter is detected). Startup log lines are parsed into `LoadResult.declared` (buffer MiB per device, layers offloaded).
- `runPrompt` streams `POST /completion` with `cache_prompt:false`. TTFT is client wall clock to the first content chunk; prefill and decode TPS come from the final `timings`. The request's own timeout aborts it.
- `applyTemplate(messages)` calls `POST /apply-template`, so the chat-format quality prompts can use `runPrompt`.
- The exit is recorded from the process `close` event: `lastExit = {code, reason: classifyExit(tail), tail}`, where reason is `oom` (allocation failure patterns), `device_lost` or `crash`.
- `unloadModel` sends `kill()`, escalates to `taskkill /T /F` after 5 s, and throws if the process is still alive. The pid is persisted to `userData/llama-server.pid`. On startup, `killStaleServer` kills a leftover server, but only if that pid is still `llama-server.exe`. `before-quit` and `exit` call `killSync()`.
- **Ollama / LM Studio** (`runtimes/others.ts`) implement `detect()` only: an HTTP probe with a 1.5 s timeout plus a model-dir hint. Every other method throws `NotImplementedError`.

### Adding a backend
1. Implement `InferenceBackend` in `src/core/runtimes/<name>/index.ts`. `detect()` must never throw, and `runPrompt` must report errors in the result rather than throwing.
2. To be benchmarkable, the backend must also satisfy `SessionBackend` (`core/benchmark/session.ts`): expose `pid` (for telemetry), `lastExit`, `warmup(prompt)`, `applyTemplate(messages)` and `cancel()`. Map its fatal errors onto `ExitInfo.reason` (`oom|device_lost|crash`).
3. Add it to `detectRuntimes()` (`runtimes/index.ts`) and add its id to `RuntimeDetection['id']` (`shared/types.ts`).
4. If it has its own device naming, provide the `gpuDevice` string for `SessionDeps`. Candidate generation (`candidates.ts`) takes `runtime.backend: 'vulkan'|'cuda'|'cpu'`; `cpu` means only `ngl=0` candidates.
5. Add a fake-process test like `tests/runtime/llamacpp-lifecycle.test.ts` (uses `tests/fixtures/fake-llama-server.cjs`). Do not use real inference in tests.

### Vendor paths
| Vendor | Runtime build | Telemetry | Status |
|---|---|---|---|
| AMD | Vulkan (`llama-b<N>-bin-win-vulkan-x64.zip`) | PDH counters via typeperf (`telemetry/sampler.ts`): per-PID VRAM/shared, GPU engine util, CPU, RAM. Temperature, power and clocks are **UNAVAILABLE** (they need the ADLX native SDK). | Tested (RX 9070 XT) |
| NVIDIA | CUDA build + `cudart-llama-bin-win-cuda-<ver>-x64.zip` extracted into the **same dir**, chosen by `pickReleaseAsset` (`runtimes/llamacpp/assets.ts`) from the driver's CUDA major, with Vulkan as the fallback | PDH (same as AMD) plus `startNvidiaSampler` (`telemetry/nvidia.ts`, `nvidia-smi --query-gpu … -lms`): util, VRAM used, temperature, power. `probeNvidiaSmi()` treats failure text or a non-zero exit as unavailable. | **UNTESTED on real hardware.** Fixtures only. The dev box's stale driver reports "insufficient permissions" (exit 4). |
| Intel | Vulkan | PDH (same as AMD) | Untested |

Not integrated yet: `assets.ts` and `nvidia.ts` are standalone. `ensureRuntime` still downloads Vulkan only, and the session runner uses only the PDH sampler.

## 4. Session runner (`src/core/benchmark/session.ts`)

`runSession(req, deps, emit): Promise<Recommendation | null>` is one async function with closures and no classes. Its dependencies are injected (`SessionDeps`):

| dep | What it is |
|---|---|
| `backend()` | factory, called once per session. `LlamaCppBackend` fits. |
| `startSampler(pid)` | `startSampler({pid})` from `core/telemetry/sampler.ts`, started once per step, after load. |
| `storage` | `SessionStorage` (below) |
| `machine`, `gpuDevice`, `backendKind` | the scan profile, the runtime device id (`Vulkan0`) and the backend kind |
| `models: ModelMeta[]` | `id` = absolute GGUF path, which is passed as `modelPath` |
| `clock`, `readRamAvailableBytes?`, `evaluate?`, `signal?`, `config?` | time, live RAM, the quality checker (default `evaluateAsync`), cancellation, overrides of `DEFAULT_SESSION_CONFIG` |

Flow:
1. `createSession`, or reuse `req.resumeSessionId` → status `running` → `session:started`.
2. For each requested model, `generateCandidates(machineFromProfile(...), model, ...)`. Rejections are logged. All candidates across models are sorted by estimated VRAM+RAM ascending, then by id.
3. For each candidate (`candidate:started`):
   - If the GPU was lost earlier in the session, a GPU candidate is skipped.
   - For each ctx in `cand.ctxSteps` (∩ `req.ladder`): a step already stored for this session is reused. Otherwise `runStep`:
     - Live RAM pre-check: an estimate above available − floor records `fail/skipped_memory` without loading.
     - `loadModel` with `-c ctx`, `-ngl 999|n`, `--device`, `-t`, `-b 2048`, `-ub 512`, `-fa on`, plus `-ctk/-ctv q8_0` for q8 candidates. The KV cache is allocated at load, so each step restarts the server.
     - Start the sampler on the new pid. A 1 s guard timer emits `telemetry` and trips `guard_abort` (calling `backend.cancel()`) if RAM available < floor or per-PID shared GPU memory > 2 GiB.
     - One size-matched warmup, then `reps` measured prompts (median). Each rep emits `token-rate`.
     - Stop the sampler, compute peaks and build the `BenchmarkRunResult` (`saveRun`, `step:done` with its cliff verdict).
   - Stop the ladder on the first FAIL verdict or after 2 consecutive DEGRADED steps.
   - Quality: once per model, on the first candidate of that model with a usable step, at `min(profile.targetContext, practical ceiling)`. Skipped when `runQuality === false`, and reused on resume via `listQuality`.
   - `unloadModel` → `candidate:done {status: done|failed|cancelled|skipped, reason}`.
4. `recommend(inputs, machine, workload)` → `saveRecommendation` → status `done` → `session:done`.
5. Cancel: `signal` abort calls `backend.cancel()`; loops check `signal.aborted`; status `cancelled` → `session:cancelled`, with partial runs already persisted. Any thrown error is caught and becomes `session:failed` + status `failed`, and `finally` always unloads.

### SessionStorage (to be implemented over `storage/db.ts` / `sessions.ts`)
```ts
createSession({workload, request, startedAt}): string
setSessionStatus(id, 'running'|'done'|'cancelled'|'failed', error?)
listRuns(id): BenchmarkRunResult[]                 // for resume
saveRun(id, run, {samples, reason, stderrTail, load, startedAt, endedAt})
listQuality(id, modelId): QualityResult[]
saveQuality(id, modelId, configId, ctx, results)
saveRecommendation(id, rec)
```
Every method may be sync or async. For the event list see `src/shared/bench-events.ts`: `session:started`, `candidate:started`, `phase`, `step:started`, `step:done`, `telemetry`, `token-rate`, `log`, `candidate:done`, `session:done`, `session:cancelled`, `session:failed`. All events carry `sessionId`.

## 5. Safety guards (as implemented)

| Guard | Where | Rule |
|---|---|---|
| Explicit config | `llamacpp.loadModel` | `-fit off`, explicit `-c`, `-ngl` and `--device`, and `--parallel 1`, so llama-server never silently changes the config. |
| Discrete device only | `pickDiscreteDevice`, `machineFromProfile` | iGPU names are excluded, and the largest non-integrated GPU supplies the VRAM total. |
| Memory pre-pruning | `candidates.ts` | RAM est > available − 4 GiB → step skipped (never kept). VRAM est > total − in-use − 512 MiB → the first such step is kept once, the rest are skipped. |
| Live RAM floor | `session.ts` | Before each step: est RAM > live available − max(2 GiB, 8 % RAM) → `skipped_memory`. During a step: sample RAM available < floor → cancel, `guard_abort`. |
| Spill abort | `session.ts` | Per-PID shared GPU memory > 2 GiB → cancel, `guard_abort`. |
| Timeouts | `llamacpp`, `session.ts` | load: 120 s health wait (hardcoded). prompt: 60 s + 10 ms × ctx. quality: 180 s per test. |
| Process cleanup | `llamacpp` | kill → `taskkill /T /F` → throws if still alive. Pid file + stale-server kill at startup. `killSync` on quit. The runner unloads in `finally`. |
| Device loss | `session.ts` | `device_lost` fails the step and skips all later GPU candidates in the session. |
| Model code | `quality/sandbox.ts` | Runs in a child process (`ELECTRON_RUN_AS_NODE`, `--permission`, `--max-old-space-size`, fresh vm context, timeout, 64 KB stdout cap). Not in a worker, because a heap blow-up there aborts the host. |
| Demo data isolation | `main/index.ts` | `LAO_SEED_DEMO=1` uses a separate `optimizer-demo.db`, and demo sessions are excluded from `latestRecommendation`. |

## 6. Persistence (`src/core/storage/db.ts`)

`node:sqlite` `DatabaseSync` runs in main (WAL, foreign keys on). Migrations are an ordered array of SQL strings, and `schema_version` records the applied count. Rows are thin: a JSON `payload` column plus a few indexed scalars.

| Table | Scalars | payload |
|---|---|---|
| `machine_profile` | — | SystemProfile |
| `runtime` | runtime_id, version | detection |
| `model` | id, path, size_bytes | model info |
| `benchmark_session` | status | `SessionPayload` {workload, candidates[{config, model}], vramBytes, demo?, label?} |
| `benchmark_run` | session_id, status, model_id, ctx_size | `BenchmarkRunResult` |
| `telemetry_sample` | session_id, run_id | `TelemetrySample` (bulk insert: `insertTelemetrySamples`) |
| `quality_result` | session_id, run_id, model_id | `QualityResult` |
| `recommendation` | session_id, model_id | `Recommendation` (workload is read with `json_extract`) |

`DESIGN.md §7` describes a fully columnar schema. The code deliberately stores payload JSON instead, so the types in `shared/` are the schema.

## 7. Tests

`npm test` (vitest, `tests/**/*.test.ts`) needs no GPU, network or real inference. It covers:
- **Scanner:** a real captured PowerShell JSON.
- **llama.cpp:** lifecycle tests with a fake child process and `parse.ts`.
- **GGUF:** synthetic buffers.
- **Telemetry:** typeperf CSV fixtures.
- **Quality:** checkers and the sandbox.
- **Scoring:** the fixtures in `tests/fixtures/scoring`.
- **Session runner:** a fake backend, fake sampler and in-memory storage.
- **Storage:** a temp SQLite file.

`npm run build` = `tsc --noEmit` + electron-vite build. Real-GPU calibration is `scripts/calibrate.ts`, and its results are in `docs/calibration-2026-09-27.md`.

## 8. Known limitations
- `bench:start` / `bench:cancel` are not wired to `runSession` yet, and no `SessionStorage` adapter over `sessions.ts` exists yet.
- The load timeout (120 s) is hardcoded in `LlamaCppBackend.loadModel` and is not configurable per phase.
- Ollama / LM Studio are detection-only. There is no CUDA/ROCm build selection, and only the Vulkan asset is downloaded (`pickVulkanAsset`).
- The telemetry sampler uses English PDH counter names only. On localized Windows its fields report unavailable, because the WMI fallback in DESIGN §1.2 is not implemented.
- The runner compares runs within one session only, and there is no warning for a changed driver or runtime between sessions (X17).
- `app.getAppPath()` locates `vendor/` and `models/`, which is correct for dev/preview only. Packaged-build paths are not handled.
