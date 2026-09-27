# Architecture — Local AI Optimizer

This document describes what the code does today. The design record and its rationale are in `docs/DESIGN.md`. Metric and scoring definitions are in `docs/BENCHMARK.md`.

## 1. Components

```
renderer (React, sandboxed)            main (Electron, Node 24)                     core (plain TS, no Electron)
────────────────────────────           ────────────────────────────                 ──────────────────────────────
Dashboard / Benchmark / Models   ──►   preload: window.api (contextBridge)   ──►   system/scanner.ts      hardware scan
Results (ExportMenu, TelemetryChart)   ipcMain.handle(...) in main/index.ts        runtimes/*             llama.cpp, Ollama, LM Studio
System / HubPage (HF download)         main/validate.ts (request sanitizing)        hub/hf.ts              Hugging Face client
                                       main/hub.ts (HF IPC + op lock)     
benchState.ts (event fold)      ◄──   'bench:event' channel (sendBenchEvent) ◄──   benchmark/session.ts   session runner
                                       settings.json (userData)                     benchmark/candidates.ts  config generation
                                       optimizer.db (node:sqlite, userData)         telemetry/sampler.ts   typeperf sampler
                                                                                    quality/*              quality suite + sandbox
                                                                                    scoring/*              cliffs, scores, recommendation
                                                                                    models/gguf.ts         GGUF header reader
                                                                                    storage/db.ts, sessions.ts  SQLite schema + queries
                                                                                    runtimes/ollama/models.ts   Ollama/LM Studio model stores
                                                                                    telemetry/nvidia.ts    nvidia-smi sampler (merged via withNvidia)
                                                                                    export/config.ts       llama-server/Ollama/LM Studio export
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

**Models.** `models:list` (`listAllModels`, `main/index.ts`) merges three sources, deduplicated by path:
- `findGgufModels` over the model dirs: `userData/models` (packaged) or `<project>/models` (dev), plus `settings.modelDirs`;
- LM Studio's `defaultLmStudioDirs()`;
- Ollama blobs via `listOllamaModels()` → `toModelInfo` (runtime `ollama`). These are benchmarked through our llama-server.

Each `ModelInfo` carries `meta` (GGUF header, streamed in 1 MiB windows, including per-layer KV / hybrid / SWA layout keys since a456679) or `metaError`. `models:fit` tells the UI which models need heavy mode.

**Benchmark (live).** The renderer calls `startBench(SessionRequest)`, which goes to `bench:start` → `startSession` (`main/index.ts`):
1. One app instance (`requestSingleInstanceLock`, taken before any DB/pid cleanup; a second instance focuses the first and exits). Only one session or smoke run at a time, and none while a runtime install holds its lock. `sanitizeRequest` (`main/validate.ts`) whitelists fields, clamps reps and the ladder, and checks that model paths are inside a model root (`isInside`, resolve-based).
2. The system scan is cached per app run.
3. `findGgufModels` + `toModelMeta`, then `listDevices()` → `pickDiscreteDevice`, and `detect()` for the runtime version.
4. `runSession(req, {backend: () => llama, startSampler: (pid) => startSampler({pid}), storage: makeSessionStorage(db, planFor), readRamAvailableBytes: os.freemem, evaluate: evaluateAsync, runtimeVersion, signal}, sendBenchEvent)`.

Events go to every window on `bench:event`, and the renderer folds them in `benchState.ts`.
- `bench:cancel` aborts the session.
- `bench:pause` sets `pauseSignal`: the current step finishes, then the session is marked `paused`.
- `bench:resume(id, {retryFailed?, rerunConfigIds?})` re-runs the stored request with `resumeSessionId`, using the stored machine scan so the plan is identical.
- At startup, `markInterrupted` turns sessions left `running` by a killed app into `interrupted`; those are resumable too.

**Results.** `sessions:list`, `sessions:get` and `recommendation:latest` read `core/storage/sessions.ts`. `getSession` recomputes `detectCliffs` from the stored runs, so cliff verdicts always reflect the current code.
- **Export** (wired in e31dc64): `ExportMenu.tsx` uses `core/export/config.ts`. It offers the llama-server command (the benchmarked inference parameters), an Ollama Modelfile, LM Studio settings (unverified keys), JSON and the provenance note, saved via `file:save`.
- **Per-run telemetry:** `telemetry:run(runId)` → `TelemetryChart.tsx`, showing GPU/CPU % and per-PID VRAM/shared/private GiB over time, with gaps for null readings.

**Hugging Face download** (`core/hub/hf.ts`, `main/hub.ts`, `preload/hub.ts`, `HubPage.tsx`, wired in fa491b0):
- Channels: `hub:whoami/login/logout/openTokenPage/dirs/search/files/download/cancel` and the `hub:progress` event.
- The token is stored encrypted with safeStorage (`userData/hf-token.bin`). The token page opens in a sandboxed window with its own `persist:huggingface` partition. Logout invalidates an in-flight login; whoami times out after 15 s.
- The token is sent only to an exact origin allowlist (`tokenAllowed`), checked on every page and redirect hop; redirects are followed by hand, and CDN URLs get no token.
- Downloads: per-segment path checks (`safeSegment`: ADS, device names, trailing dot/space), a `<part>.json` identity sidecar for resume, the 206 Content-Range must start at the offset, and the stream runs through `stream.pipeline` (disk errors → `disk_error`, bytes past the expected size stop it). Links/junctions on the way to the `.part` are refused (check-then-open); main checks destDir by the realpath of its nearest existing ancestor.
- One download at a time: the op and its AbortController are claimed synchronously. HubPage keys a transfer by {repoId, path, destDir}, so Resume and progress never follow the repo being browsed.

**IPC channels** (main/index.ts): `system:scan`, `runtimes:detect`, `runtime:install`, `models:list`, `models:fit`, `settings:get`, `settings:setWorkload`, `workloads:list`, `sessions:list`, `sessions:get`, `recommendation:latest`, `telemetry:run`, `file:save`, `bench:start|pause|cancel|resume|smoke`, plus the `hub:*` channels above; events `bench:event`, `runtime:progress`, `hub:progress`. Ids are checked as positive safe integers and workload ids with `Object.hasOwn`. The renderer ignores late replies and events of a previous session (the watched id is adopted only on `session:started`).

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
- `unloadModel` sends `kill()`, escalates to `taskkill /T /F` after 5 s, and keeps the handle and pid file until the exit is confirmed; if the process is still alive it throws `ServerStuckError` (`.fatal`), which the runner treats as a hard stop. The pid is persisted to `userData/llama-server.pid`. On startup, `killStaleServer` kills a leftover server, but only if that pid is still `llama-server.exe`. `before-quit` and `exit` call `killSync()`.
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

Integrated in acbd169:
- `runtime:install` → `ensureRuntime(log, {vendor, cudaMajor})` picks the build with `pickReleaseAsset`, falling back to Vulkan if the CUDA build fails `--version`.
- When `probeNvidiaSmi()` is available, the runner's sampler is `withNvidia(startSampler({pid}), startNvidiaSampler())`, which merges tempC/powerW into the samples.

## 4. Session runner (`src/core/benchmark/session.ts`)

`runSession(req, deps, emit): Promise<Recommendation | null>` is one async function with closures and no classes. Its dependencies are injected (`SessionDeps`):

| dep | What it is |
|---|---|
| `backend()` | factory, called once per session. `LlamaCppBackend` fits. |
| `startSampler(pid)` | `startSampler({pid})` from `core/telemetry/sampler.ts`. One per step, because each step restarts the server with a new pid and typeperf's per-PID counter set is fixed at start. It is started **during load**, as soon as the new pid exists. |
| `storage` | `SessionStorage` (below) |
| `machine`, `gpuDevice`, `backendKind` | the scan profile, the runtime device id (`Vulkan0`) and the backend kind |
| `models: ModelMeta[]` | `id` = absolute GGUF path, which is passed as `modelPath` |
| `clock`, `readRamAvailableBytes?`, `evaluate?`, `signal?`, `pauseSignal?`, `runtimeVersion?`, `config?` | time, live RAM, the quality checker (default `evaluateAsync`), cancel, pause (between steps only), the version stamped on runs, overrides of `DEFAULT_SESSION_CONFIG` |

Flow:
1. `createSession`, or reuse `req.resumeSessionId` → status `running` → `session:started`.
2. For each requested model, `generateCandidates(machineFromProfile(...), model, ...)`. Rejections are logged. Order: full-offload configs first, then heavy configs (KV-on-GPU by offload share, then -nkvo, then CPU baselines), then estimated VRAM+RAM ascending, then id (D26).
3. For each candidate (`candidate:started`):
   - If the GPU was lost earlier in the session, a GPU candidate is skipped.
   - For each ctx in `cand.ctxSteps` (2K…128K ∩ declared ctx ∩ VRAM estimate, ∩ `req.ladder`): a step already stored for this session is reused. Otherwise `runStep`:
     - Unload the previous step's server (and wait for exit), then the live RAM pre-check: resident estimate > available − floor → `fail/skipped_memory` without loading.
     - `loadModel` (with the session `signal`, so a cancel kills a loading server at once) with `-c ctx`, `-ngl 999|n`, `--device`, `-t`, `-b 2048`, `-ub 512`, `-fa on`, plus `-ctk/-ctv q8_0` for q8 candidates. The KV cache is allocated at load, so each step restarts the server.
       - A 50 ms poll starts the sampler once `backend.pid` changes, so typeperf's ~2 s start-up overlaps the load.
       - Load errors map to failure kinds: `ConfigDriftError` (`/props` n_ctx < requested; larger/absent n_ctx and GPU layers/device are not compared, D17) → `config_drift`; exit reasons → `oom|device_lost|crash`; "healthy within" → `load_timeout`; otherwise `load_fail`.
     - A guard timer (1 s; 250 ms for heavy configs), running from the start of load, emits `telemetry` and trips `guard_abort` (kill during load, else `backend.cancel()`) if RAM available + mmap credit < floor or per-PID shared GPU memory > 2 GiB (heavy configs: spill recorded, next config). RAM is read from the OS on every poll, independent of typeperf; 3 polls with no reading at all trip it too (fail-safe).
       - The request error that this cancel causes is attributed to the guard: the guard reason wins over "cancelled".
     - One size-matched warmup (sets `warm`), then `reps` measured prompts (median). Each rep emits `token-rate`.
     - If there are 0 samples after the reps, the runner waits for one real row until `firstSampleWaitMs` (3 s) after sampler start, then stops the sampler.
       - Peaks use all samples; averages use only the warmup+measure window.
       - Still 0 samples → unavailable.
     - `BenchmarkRunResult` carries `versions` {benchmark, prompts, quality, runtime} → `saveRun`, then `step:done` with its cliff verdict.
   - Stop the ladder on the first FAIL verdict or after 2 consecutive DEGRADED steps.
   - `unloadModel` → `candidate:done {status: done|failed|cancelled|paused|skipped, reason}`.
4. Quality runs after all ladders, once per model, on its **best-offload** usable candidate: most GPU layers, then fastest decode. It is never run on a CPU/-nkvo probe. The ctx is `min(targetContext, practical ceiling)`.
   - The load goes through `guardedLoad` (unload, RAM pre-check, abortable load, RAM guard), checked between prompts; a trip discards the suite.
   - Thinking models (`supportsThinking`) run with `enable_thinking=false`.
   - Skipped when `runQuality === false`; reused on resume via `listQuality` only when it is the complete current-version suite.
5. `recommend(inputs, machine, workload)` → `saveRecommendation` → status `done` → `session:done`. A pause at any point → status `paused` → `session:paused`.
   - Resume rules: cancelled and `skipped_memory` steps always re-run; `retryFailed` re-runs fail/timeout steps (not `config_drift`); `rerunConfigIds` re-runs every step of those configs. Reads keep the last row per (configId, ctx).
6. Cancel: `signal` abort calls `backend.cancel()`; loops check `signal.aborted`; status `cancelled` → `session:cancelled`, with partial runs already persisted. Any thrown error is caught and becomes `session:failed` + status `failed`, and `finally` unloads — except after a `ServerStuckError`, which stops the session at the next boundary and skips the final unload so the pid file survives for the stale-server kill.

### SessionStorage (implemented by `makeSessionStorage(db, planFor)` in `storage/sessions.ts`)
```ts
createSession({workload, request, startedAt}): string
setSessionStatus(id, 'running'|'done'|'cancelled'|'failed', error?)
listRuns(id): BenchmarkRunResult[]                 // for resume
saveRun(id, run, {samples, reason, stderrTail, load, startedAt, endedAt})
listQuality(id, modelId): QualityResult[]
saveQuality(id, modelId, configId, ctx, results)   // one transaction, with suite version + expected count
saveRecommendation(id, rec)
```
Every method may be sync or async. For the event list see `src/shared/bench-events.ts`: `session:started`, `candidate:started`, `phase`, `step:started`, `step:done`, `telemetry`, `token-rate`, `log`, `candidate:done`, `session:done`, `session:cancelled`, `session:failed`. All events carry `sessionId`.

## 5. Safety guards (as implemented)

| Guard | Where | Rule |
|---|---|---|
| Explicit config | `llamacpp.loadModel` | `-fit off`, explicit `-c`, `-ngl` and `--device`, and `--parallel 1`, so llama-server does not fit-adjust the config. `/props` n_ctx smaller than requested → `ConfigDriftError` (e.g. Qwen2.5 served 32K when 48K was requested). |
| Discrete device only | `pickDiscreteDevice`, `machineFromProfile` | iGPU names are excluded, and the largest non-integrated GPU supplies the VRAM total. |
| Memory pre-pruning | `candidates.ts` | RAM est (resident part: non-GPU weights + CPU KV + 0.5 GiB, +1.5 GiB at ngl 0) > available − 4 GiB → step skipped (never kept). VRAM est > total − in-use − 1 GiB → the first such step is kept once if ≤ 1.15× the budget; the rest are skipped. |
| Live RAM floor | `session.ts` | Before each step and the quality load: est RAM > live available − max(4 GiB, 8 % RAM) → `skipped_memory`. From the start of load (250 ms poll for heavy configs): OS RAM available (else the telemetry row) + mmap credit < floor → kill (during load) or cancel → `guard_abort`. 3 blind polls → `guard_abort` (fail-safe). |
| Spill abort | `session.ts` | Per-PID shared GPU memory > 2 GiB → cancel, `guard_abort` (the guard reason takes precedence over the resulting "cancelled"). Heavy configs record it and move on. |
| Telemetry glitches | `sampler.ts` | Percentages in (100, 1000] are clamped to 100; rows with a negative or > 1000 value (e.g. 1.3e13 % GPU util) are dropped whole, and the drop counts go to `samplerErrors`. |
| Timeouts | `llamacpp`, `session.ts` | load: 120 s health wait (hardcoded). prompt: 60 s + 10 ms × ctx. quality: 180 s per test. |
| Process cleanup | `llamacpp` | kill → `taskkill /T /F` → `ServerStuckError` if still alive (session hard stop). Pid file + stale-server kill at startup. `killSync` on quit. The runner unloads in `finally`. No Job Object. |
| Single instance / op locks | `main/index.ts`, `main/hub.ts` | `requestSingleInstanceLock` before any cleanup. Benchmark/smoke, runtime install and hub download each claim their lock synchronously. |
| Hub download | `core/hub/hf.ts` | Token origin allowlist per request/hop, per-segment path checks, link refusal (check-then-open), sidecar identity + Content-Range offset check, size cap, `stream.pipeline` error handling. |
| Device loss | `session.ts` | `device_lost` fails the step and skips all later GPU candidates in the session. |
| Model code | `quality/sandbox.ts` | Runs in a child process (`ELECTRON_RUN_AS_NODE`, `--permission`, `--max-old-space-size`, fresh vm context, timeout, 64 KB stdout cap). Not in a worker, because a heap blow-up there aborts the host. Strict host code, `Error` frozen in the context, thrown values never read through getters/`toString`. RSS cap: the child checks its arrayBuffers/RSS after the run and the parent polls its working set every 250 ms (kill > 1.5× cap + 48 MiB); a burst shorter than one poll can briefly exceed it (no Job Object). |
| Demo data isolation | `main/index.ts` | `LAO_SEED_DEMO=1` uses a separate `optimizer-demo.db`, and demo sessions are excluded from `latestRecommendation`. |
| Request validation | `main/validate.ts` | Renderer requests are sanitized: whitelisted rule keys, reps and ladder clamped, model paths resolved and checked to be inside a root. |
| Uninstall | `build/installer.nsh` | The NSIS uninstall kills the llama-server in the app's pid file only after verifying its exe path and start time (a process it cannot verify is left alone). userData is kept on purpose. |

## 6. Persistence (`src/core/storage/db.ts`)

`node:sqlite` `DatabaseSync` runs in main (WAL, foreign keys on). Migrations are an ordered array of SQL strings, and `schema_version` records the applied count.
- A DB newer than the build is **refused** ("database schema vN is newer than this build").
- userData is `%APPDATA%\local-ai-optimizer` when packaged and `…-dev` in dev, so dev migrations never touch the installed app's data.
- Rows are thin: a JSON `payload` column plus a few indexed scalars.

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

`npm test` (vitest, `tests/**/*.test.ts`) needs no GPU, internet or real model inference; some tests bind local loopback sockets (hub, llama-server lifecycle). It covers:
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
See `docs/LIMITATIONS.md`, the single consolidated list.
