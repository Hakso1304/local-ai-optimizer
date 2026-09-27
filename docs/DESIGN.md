# LOCAL AI OPTIMIZER — Implementation Design (v1)

Status legend: **[V]** = verified on the dev machine / fetched source on 2026-09-27; **[S]** = from upstream source/docs, not exercised locally; **[A]** = assumption/heuristic, to be calibrated.
Provenance of data: see §5.0 (`ProvenanceKind` = measured | declared | estimated | unavailable).

---

## 0. Key facts discovered (read these first)

| # | Fact | Consequence |
|---|------|-------------|
| F1 [V] | GitHub `releases/latest` for ggml-org/llama.cpp returns tag `v0.5.0` with **no binaries** (only `nightly-tag.txt`). Real builds are tags `bNNNNN` marked **prerelease=true** (latest `b11207`). | Use `GET /repos/ggml-org/llama.cpp/releases?per_page=10`, pick first release that has the wanted asset. Never `/releases/latest`. Pin a known-good tag in config (default `b11207`), allow "update". |
| F2 [V] | `llama-server --list-devices` shows `Vulkan0: AMD Radeon RX 9070 XT (16304 MiB, 15416 MiB free)` **and** `Vulkan1: AMD Radeon(TM) Graphics (16187 MiB …)` (iGPU, shared RAM). | Always pass `-dev Vulkan0` (the discrete GPU). Default `-sm layer` would otherwise split across the iGPU. |
| F3 [V] | `--fit` defaults to **on**: silently adjusts unset args (ngl, ctx) to fit memory. `-ngl` default is `auto`. `-c 0` = model's train ctx. | Always pass `-fit off` and explicit `-c`, `-ngl`, else measurements are not of the config we think. |
| F4 [V] | Registry `…\Class\{4d36e968-…}\00xx` still contains a **stale NVIDIA RTX 3080** (qwMemorySize 10 GiB). 9070 XT = 17095983104 B. | Join registry subkeys to *present* devices by `MatchingDeviceId` prefix of `Win32_VideoController.PNPDeviceID`. Ignore unmatched keys. |
| F5 [V] | `nvidia-smi` prints "NVIDIA-SMI has failed because you do not have sufficient permissions" and **exits 0**. | Parse stdout (CSV must match regex), never trust exit code. |
| F6 [V] | `node:sqlite` works in Electron 44.4.5 main (Node 24.21, SQLite 3.53.4) via `ELECTRON_RUN_AS_NODE`. | Use `DatabaseSync` from `node:sqlite`. No sql.js. Main process only; renderer via IPC. |
| F7 [V] | With `--no-warmup`, first request of a *new batch shape* is slow (8-token prompt: 575 ms = 13.9 t/s; after that normal). | Keep built-in warmup ON and add our own shape warmup (§3.4). |
| F8 [V] | KV over-allocation: `ErrorOutOfDeviceMemory` → process exits **code 1** after ~6 s; one run took >60 s before failing (allocation retries under WDDM). | Load timeout + stderr pattern match; kill on first fatal pattern, don't wait for exit (§2.5). |
| F9 [V] | Idle adapter-level shared usage on the dGPU is ~1.1–1.5 GB (desktop/other apps); per-PID shared for llama-server at 8K ctx = 13 MB. | Spill detection must use **per-PID** `GPU Process Memory` shared-usage delta, never adapter totals. |
| F10 [V] | `Memory\Available MBytes` fluctuated 5.2 GB → 19 GB during the session (other apps). | RAM floor is re-checked before every step, not just at session start. |

---

## 1. Telemetry (Windows, no native modules, no admin)

### 1.1 Verified counters (English counter names worked on this machine)
| Counter | Result [V] | Instance format |
|---|---|---|
| `\GPU Adapter Memory(*)\Dedicated Usage` | 5884596224 (bytes) for dGPU | `luid_0x00000000_0x00016058_phys_0` |
| `\GPU Adapter Memory(*)\Shared Usage` | 1484787712 | same |
| `\GPU Process Memory(pid_<PID>*)\Dedicated Usage` | llama-server PID 16704: 773464064 (0.5B Q8 @8K) | `pid_16704_luid_0x00000000_0x00016058_phys_0` |
| `\GPU Process Memory(pid_<PID>*)\Shared Usage` | 13123584 | same |
| `\GPU Engine(pid_<PID>*)\Utilization Percentage` | 458 instances total; e.g. 37.9 | `pid_26128_luid_…_phys_0_eng_2_engtype_Compute 0` (engtype: 3D, Copy, Compute N, Video …) |
| `\Processor(_Total)\% Processor Time` | 74.79 | — |
| `\Memory\Available MBytes` | 5263 | — |
| `\Process(llama-server*)\Working Set - Private` | **fails** "No valid counters" if the process doesn't exist yet; instance names are `name#N` (unstable) | — |
| `\Process V2(llama-server:<PID>)\Working Set - Private` | 133181440 [V] | `name:PID` (stable). Use this. |
| WMI `Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory` / `…GPUProcessMemory` / `…GPUEngine` | same values [V] | `Name` = same instance strings |

Also available: `GPU Local Adapter Memory\Local Usage`, `GPU Process Memory\{Local,Non Local,Total Committed}`. There is **no** "Dedicated Limit" counter. Total VRAM comes from the registry (§1.4).

Instance parsing regex (TS):
```
/pid_(\d+)_luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)(?:_eng_(\d+)_engtype_(.+))?/i
```
Key = `luidHigh_luidLow`. **LUID → adapter mapping:** no counter carries the adapter name. Rule: the benchmark GPU's LUID = the LUID where the llama-server PID holds the most Dedicated Usage after load (MEASURED). Before any run, guess it as the adapter LUID with the largest Dedicated Usage. The iGPU has near-zero dedicated usage [V].

GPU util per PID = **max over engtypes** of sum(`Utilization Percentage`) per engtype for that PID+LUID. Vulkan compute shows up as `3D` or `Compute N` depending on driver, so take the max of those two groups.

### 1.2 Sampling approach (decision)
- **Use one `typeperf` child process per benchmark run**, started *after* llama-server's PID is known: `typeperf <counters…> -si 1 -f CSV` (stdout streaming, one row/sec, first row = header) [V]. The counter set is **fixed at start** (instances created later are not picked up), which is why it starts after the PID is known and uses PID-scoped wildcards:
  - `\GPU Process Memory(pid_<PID>*)\Dedicated Usage`, `…\Shared Usage`
  - `\GPU Engine(pid_<PID>*)\Utilization Percentage`
  - `\GPU Adapter Memory(*)\Dedicated Usage`, `…\Shared Usage`
  - `\Process V2(llama-server:<PID>)\Working Set - Private`, `\Process V2(llama-server:<PID>)\% Processor Time`
  - `\Processor(_Total)\% Processor Time`, `\Memory\Available MBytes`
- Interval **1 s** (typeperf minimum). Decode runs are ≥2 s by design (n_predict=128), so each run gets ≥2 samples. Parse the CSV header to column-index map. Values are `"%f"` strings. Kill typeperf with `child.kill()` at run end.
- **Localization fallback:** counter *names* are localized on non-English Windows. On startup, probe `typeperf "\Memory\Available MBytes" -sc 1`. If it fails, switch to polling WMI classes (`Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_*`), whose class names are locale-invariant, from a long-lived `powershell -NoProfile -Command -` child reading commands on stdin, polled every 1 s, emitting JSON lines. (The OS here uses cp949 console output. Set `chcp 65001` / `[Console]::OutputEncoding=UTF8`.)
- Per run we store raw samples (TelemetrySample) plus peaks/means: peak dedicated (PID), peak shared (PID), delta shared vs pre-load baseline, mean GPU util during decode window, peak private WS, min Available MBytes.

### 1.3 Temperature / power / clocks
| Metric | AMD (no native) | NVIDIA | Intel |
|---|---|---|---|
| VRAM used (per-PID, adapter) | AVAILABLE (PDH) | AVAILABLE (PDH + nvidia-smi) | AVAILABLE (PDH) |
| GPU utilization | AVAILABLE (PDH GPU Engine) | AVAILABLE | AVAILABLE |
| GPU temperature | **UNAVAILABLE** (needs ADLX/ADL native SDK. `MSAcpi_ThermalZoneTemperature` needs admin and isn't the GPU) | AVAILABLE via nvidia-smi if it works, else UNAVAILABLE | UNAVAILABLE |
| GPU power | **UNAVAILABLE** | AVAILABLE via nvidia-smi | UNAVAILABLE |
| GPU clocks | UNAVAILABLE | AVAILABLE via nvidia-smi | UNAVAILABLE |
| CPU temp | UNAVAILABLE (no admin) | — | — |
| CPU util / RAM | AVAILABLE (PDH) | AVAILABLE | AVAILABLE |

`UNSUPPORTED` = the runtime/vendor can't provide it by design (e.g. per-PID GPU memory on Windows < 10 1709). `UNAVAILABLE` = could exist but not without admin/native code. The UI shows "—" with a tooltip, never 0.

NVIDIA [S]: `nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,driver_version --format=csv,noheader,nounits -lms 1000` (streaming). Valid only if every line matches `^\d+, .+`. On this machine it fails (F5), so the vendor-neutral PDH path is primary for all vendors.

### 1.4 Static scan sources [V unless noted]
- GPU list: `Win32_VideoController` (Name, PNPDeviceID, DriverVersion, AdapterRAM *ignored*: it overflows at 4 GiB, e.g. 4293918720 for the 16 GB card).
- True VRAM: `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\00NN` → `HardwareInformation.qwMemorySize` (QWORD, bytes), `DriverDesc`, `MatchingDeviceId`. Read with `reg query` or `Get-ItemProperty` (no admin). Join to present devices (F4). Some drivers expose `HardwareInformation.MemorySize` (REG_BINARY/DWORD) instead [S]: fall back to it.
- dGPU vs iGPU: VEN_10DE/VEN_1002/VEN_8086 + qwMemorySize ≥ 2 GiB and name not matching `/Radeon\(TM\) Graphics|UHD|Iris/` → discrete.
- Vulkan: `C:\Windows\System32\vulkan-1.dll` exists [V]. Authoritative device list = `llama-server --list-devices` (no model load, <1 s, parse `^\s*(Vulkan\d+|CUDA\d+): (.+) \((\d+) MiB, (\d+) MiB free\)`) [V].
- CUDA: `System32\nvcuda.dll` exists [V: True here, stale driver] **and** nvidia-smi CSV parses. Both must hold to offer the CUDA build.
- CPU: `Win32_Processor` Name, NumberOfCores (8), NumberOfLogicalProcessors (16) [V]. RAM: `Win32_ComputerSystem.TotalPhysicalMemory` (33410048000) [V].
- Disk: `Win32_LogicalDisk -Filter DriveType=3` DeviceID/Size/FreeSpace [V]. Check free space on the models drive before downloads.
- OS: `Win32_OperatingSystem` Caption/BuildNumber.
- Run everything in ONE `powershell -NoProfile -NonInteractive -Command` that emits `ConvertTo-Json -Depth 4` (≈1–2 s total).

---

## 2. llama-server integration

### 2.1 Release assets [V, tag b11207]
- Vulkan: `llama-b<N>-bin-win-vulkan-x64.zip` (33 MB). URL: `https://github.com/ggml-org/llama.cpp/releases/download/b<N>/llama-b<N>-bin-win-vulkan-x64.zip`.
- CUDA: `llama-b<N>-bin-win-cuda-12.4-x64.zip` (263 MB) + runtime `cudart-llama-bin-win-cuda-12.4-x64.zip` (391 MB); or `…cuda-13.4-x64.zip` (152 MB) + `cudart-…-13.4-x64.zip` (424 MB). Choose 12.4 unless the driver supports CUDA 13 (nvidia-smi header "CUDA Version: 13.x"). Unzip the cudart DLLs into the same folder.
- Others present: cpu-x64, rocm-10.0-x64 (257 MB), sycl-x64, openvino. Not MVP.
- Zip layout: **flat**, no subfolder. Required for server: `llama-server.exe` (9 KB stub), `llama-server-impl.dll`, `llama.dll`, `llama-common.dll`, `ggml.dll`, `ggml-base.dll`, `ggml-vulkan.dll` (45 MB), `ggml-cpu-*.dll` (dynamic CPU variants, e.g. `ggml-cpu-zen4.dll`), `libomp.dll`, `mtmd.dll`. Extract the whole zip. Don't cherry-pick.
- Version: `llama-server --version` → `version: 0.5.0-dev (build 11207, commit 7ac59a6e3)`. Also `/props.build_info` = `b11207-7ac59a6e3`.

### 2.2 Launch command (decision)
```
llama-server.exe -m <gguf> -dev Vulkan0 -fit off -ngl <all|N|0> -c <ctx>
  -np 1 -b 2048 -ub 512 -t <physCores> -tb <physCores> -fa <on|off>
  -ctk <f16|q8_0> -ctv <f16|q8_0> --cache-ram 0 --metrics --no-webui
  --host 127.0.0.1 --port <free port> --seed 1 -lv 3
```
| Flag | Why [V = accepted by b11207] |
|---|---|
| `-dev Vulkan0` | F2. CUDA build: `-dev CUDA0`. For ngl=0 use `-dev none`. |
| `-fit off` | F3. Isolation. |
| `-ngl all\|N` | `all` literal accepted. |
| `-c` | Explicit. Required because `0` = train ctx. |
| `-np 1` | One slot. Default `-1` = auto (multiple slots split KV). |
| `--cache-ram 0` | Disables host prompt cache (default 8192 MiB of RAM!). Also send `cache_prompt:false` per request. |
| `-fa on` | Explicit (default `auto` hides what ran). Quantized V cache requires FA. |
| `-ctk/-ctv` | Allowed: f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1. We use f16 / q8_0. |
| `--metrics` | `/metrics` Prometheus endpoint (secondary cross-check). |
| `-t/-tb` | Physical cores (8), not 16 logical. |
| `-lv 3` | Default. Includes load/buffer lines and print_timing. `-lv 4` adds tensor buffer sizes (useful on failure; enable on retry). |
| **Do NOT use** `--no-warmup` | F7. `--no-mmap` no longer exists; the replacement is `-lm/--load-mode {auto,none,mmap,mlock,dio}`. Keep auto. |

Load-log lines worth parsing [V at -lv 4]: `load_tensors: Vulkan0 model buffer size = 500.79 MiB`, `load_tensors: CPU_Mapped model buffer size = 137.94 MiB`, `n_ctx_seq (X) > n_ctx_train (Y)` warning, `using device Vulkan0 (...) - 15416 MiB free`. Store them as DECLARED-by-runtime.

### 2.3 Endpoints and response fields [V against a live server]
- `GET /health` → 200 `{"status":"ok"}`; 503 `{"error":{"code":503,"message":"Loading model",…}}` while loading [S]. Poll every 500 ms.
- `GET /props` → keys: `default_generation_settings.n_ctx` (=8192 [V]), `total_slots`, `model_path`, `model_ftype`, `build_info`, `chat_template`, `chat_template_caps`, `is_sleeping` [V]. Assert `n_ctx == -c` and `total_slots == 1` after load (guards against silent fit).
- `GET /slots` → `[{id, n_ctx, is_processing, n_prompt_tokens, …}]` [V].
- `GET /metrics` → `llamacpp:prompt_tokens_total`, `prompt_seconds_total`, `tokens_predicted_total`, `tokens_predicted_seconds_total`, `prompt_tokens_seconds`, `predicted_tokens_seconds`, `n_tokens_max`, … [V].
- `POST /tokenize {"content": "..."}` → `{"tokens":[…]}` [S]. Used to build prompts of exact token length.
- **Speed runs: `POST /completion`** (raw prompt, no chat template, so token counts are exact):
```json
// request
{"prompt":"<filler>","n_predict":128,"temperature":0,"seed":1,"cache_prompt":false,
 "ignore_eos":true,"stream":true,"return_progress":true}
// stream chunks [V]: SSE lines "data: {...}"
{"content":"","tokens":[0],"stop":false,"tokens_evaluated":8,"prompt_progress":{"total":8,"cache":0,"processed":8,"time_ms":574}}
{"content":" ","tokens":[220],"stop":false,"tokens_predicted":1,...}     // <- first token
// final chunk (stop:true) [V]:
{"stop":true,"stop_type":"limit","tokens_predicted":64,"tokens_evaluated":2005,"truncated":false,
 "timings":{"cache_n":0,"prompt_n":2005,"prompt_ms":519.441,"prompt_per_second":3859.9,
            "predicted_n":64,"predicted_ms":180.961,"predicted_per_second":348.1}}
```
  - **PrefillTPS** = `prompt_n / prompt_ms * 1000`. Valid only if `cache_n == 0`.
  - **DecodeTPS** = `predicted_n / predicted_ms * 1000`. Require `predicted_n == n_predict` (ignore_eos makes it deterministic).
  - **TTFT (client)** = wall time from request send to first chunk with `tokens_predicted >= 1` (MEASURED, includes HTTP). **TTFT (server)** = `prompt_ms + predicted_ms/predicted_n`. Store both. Score uses client.
  - `prompt_progress` chunks give a stall detector (no progress change in 60 s → abort).
- **Quality runs: `POST /v1/chat/completions`** (applies the model's jinja template [V]). Response has OpenAI `usage` + llama `timings` [V]. `reasoning_format: deepseek` is the default [V], so thinking models put their reasoning in `message.reasoning_content` and checkers read `message.content` only.

### 2.4 Process lifecycle
- `spawn(exe, args, {windowsHide:true, cwd: runtimeDir})`. Pipe stderr+stdout to a ring buffer (last 2000 lines) plus a per-run log file.
- Port: bind `net.createServer().listen(0)` to get a free port, close it, then pass it.
- Kill: `child.kill()`, then after 3 s `taskkill /PID <pid> /T /F`. On app start, clean up orphans: `tasklist /FI "IMAGENAME eq llama-server.exe" /FO CSV` and kill any PIDs recorded in DB as ours (F8: a timed-out parent left an orphan during probing).
- Exactly **one** llama-server at a time, globally (mutex in the benchmark service).

### 2.5 Failure detection
Classify in this order. The first match wins and we kill immediately.
| Class | Signal |
|---|---|
| `OOM_VRAM` | stderr `/ErrorOutOfDeviceMemory\|Device memory allocation of size \d+ failed\|failed to allocate \w+ buffer of size\|failed to allocate buffer for kv cache/` [V all four]; CUDA [S]: `/CUDA error: out of memory\|cudaMalloc failed/` |
| `LOAD_FAIL` | `/failed to create context with model\|exiting due to model loading error/` [V]; `/error loading model\|failed to load model\|invalid magic\|unknown model architecture/` [S] |
| `DEVICE_LOST` | `/ErrorDeviceLost\|DeviceLostError\|VK_ERROR_DEVICE_LOST/` [S]; CUDA `/unspecified launch failure\|illegal memory access/` [S] |
| `CRASH` | exit code ∉ {0,1} while running. Windows NTSTATUS: 3221225477 (0xC0000005 AV), 3221226505 (0xC0000409), 3221225725 (0xC00000FD stack overflow) [S] |
| `EXIT_1` | exit code 1 with no matching pattern [V: load errors exit 1] |
| `LOAD_TIMEOUT` | /health not 200 within `30 s + 10 s/GB of model` |
| `REQ_TIMEOUT` | prefill stall 60 s (no prompt_progress change), or decode > 120 s, or hard cap 900 s/request |
| `GUARD_ABORT` | our safety guard killed it (§3.3) |
| `CONFIG_DRIFT` | `/props` n_ctx or total_slots differ from requested |

### 2.6 GGUF metadata parser (no model load) [V against gguf.h + a real file]
Little-endian. Layout:
```
u32 magic = 0x46554747 ("GGUF")   u32 version (3; accept 2)   u64 n_tensors   u64 n_kv
n_kv × { gguf_string key; u32 value_type; value }
n_tensors × { gguf_string name; u32 n_dims; u64 dims[n_dims]; u32 ggml_type; u64 offset }
gguf_string = u64 len + UTF-8 bytes (no NUL)
value types: 0 u8,1 i8,2 u16,3 i16,4 u32,5 i32,6 f32,7 bool(u8),8 string,
             9 array{u32 elem_type; u64 n; n×elem},10 u64,11 i64,12 f64
```
- **Stream-read.** The header can be large: for Qwen2.5-0.5B it was **5,931,189 bytes** (tokenizer arrays, 151k strings) [V]. Read in 1 MB chunks with a cursor. Skip array contents for `tokenizer.ggml.*` without allocating strings.
- Keys (verified on qwen2 file): `general.architecture`=qwen2, `general.name`, `general.size_label`="630M", `general.file_type`=7, `general.quantization_version`, `<arch>.context_length`=32768, `<arch>.block_count`=24, `<arch>.embedding_length`=896, `<arch>.attention.head_count`=14, `<arch>.attention.head_count_kv`=2, optional `<arch>.attention.key_length`/`value_length`, `<arch>.attention.sliding_window`, `<arch>.expert_count`, `tokenizer.chat_template`.
- `general.parameter_count` is often **absent** (absent here). Compute it from tensor infos: Σ Π dims → `param_count` (DECLARED-derived, exact). Split files (`-00001-of-0000N.gguf`): sum over shards.
- `general.file_type` → name (llama.h `LLAMA_FTYPE_MOSTLY_*`) [V]: 0 F32, 1 F16, 2 Q4_0, 3 Q4_1, 7 Q8_0, 8 Q5_0, 9 Q5_1, 10 Q2_K, 11 Q3_K_S, 12 Q3_K_M, 13 Q3_K_L, 14 Q4_K_S, 15 Q4_K_M, 16 Q5_K_S, 17 Q5_K_M, 18 Q6_K, 19 IQ2_XXS, 20 IQ2_XS, 21 Q2_K_S, 22 IQ3_XS, 23 IQ3_XXS, 24 IQ1_S, 25 IQ4_NL, 26 IQ3_S, 27 IQ3_M, 28 IQ2_S, 29 IQ2_M, 30 IQ4_XS, 31 IQ1_M, 32 BF16, 36 TQ1_0, 37 TQ2_0, 38 MXFP4_MOE, 39 NVFP4, 40 Q1_0, 41 Q2_0; anything else → `ftype_<n>`. Fallback: parse from filename `/(I?Q\d_[A-Z0-9_]+\|F16\|BF16\|Q8_0)/i`.
- Provenance: architecture/ctx/layers/heads/ftype = DECLARED. param_count = DECLARED (derived). head_dim (if key_length absent) = `embedding_length/head_count` = ESTIMATED. `supportsThinking` = ESTIMATED from chat_template containing `<think>` or `enable_thinking`.
- Unit check: the parser must reproduce the values above on `qwen2.5-0.5b-instruct-q8_0.gguf`, plus a synthetic buffer test (vitest).

### 2.7 Memory estimation (ESTIMATED, pruning only)
```
L = block_count; Hkv = head_count_kv; dk = key_length ?? n_embd/head_count; dv = value_length ?? dk
bytesPer(t) = {f32:4, f16:2, bf16:2, q8_0:34/32, q4_0:18/32, q4_1:20/32, q5_0:22/32, q5_1:24/32, iq4_nl:18/32}
KV(ctx)   = ctx * L * Hkv * (dk*bytesPer(ctk) + dv*bytesPer(ctv))
W_gpu     = fileBytes * min(ngl, L)/L       // token_embd stays host-side [V: CPU_Mapped 138 MiB]
Compute   = 512 MiB + ub * n_vocab * 4 + (fa ? 0 : ub * ctx * head_count * 4)   // [A]
VRAM_est  = W_gpu + (ngl>0 ? KV : 0) + Compute + 256 MiB
RAM_est   = (fileBytes - W_gpu) + (ngl==0 ? KV : 0) + 512 MiB
```
Check: qwen2.5-0.5B, f16 KV → 24·2·64·(2+2) = **12288 B/token**, consistent with the failed 4M-ctx allocation (≈49 GB) [V]. Sliding-window / hybrid / recurrent archs (key `…sliding_window` present, or arch ∈ {gemma2, gemma3, gemma3n, qwen3next, mamba, rwkv*, jamba, granitehybrid, lfm2}) → mark estimate `LOW_CONFIDENCE` and don't prune by KV. Calibrate after each successful load: `ratio = measuredPeakDedicated / VRAM_est`, stored per (arch, runtime) and applied to later pruning (clamped 0.7–1.5).

### 2.8 Ollama / LM Studio (detection + listing only; neither installed here [V])
- Ollama: `GET http://127.0.0.1:11434/api/tags` (1 s timeout) → `{models:[{name, size, details:{family, parameter_size, quantization_level}}]}` [S]. `GET /api/version`. Model store: `%OLLAMA_MODELS%` or `%USERPROFILE%\.ollama\models` (`manifests/`, `blobs/sha256-*`, and the blobs are GGUF, parseable by §2.6) [S]. Executable: `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` [S].
- LM Studio: `GET http://127.0.0.1:1234/v1/models` [S]. Model dirs: `%USERPROFILE%\.lmstudio\models\<publisher>\<repo>\*.gguf` (current), `%USERPROFILE%\.cache\lm-studio\models` (legacy) [S]. The `.gguf` files found there are offered as benchmark inputs to our llama-server (read-only, never copied).

### 2.9 Test models (URLs HEAD-verified 200 [V])
| Use | URL | Size | Declared ctx |
|---|---|---|---|
| Smoke/E2E (tiny) | `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf` | 675,710,816 B | 32768 [V parsed] |
| Small realistic | `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1,117,320,736 B | 32768 [S] |
| 8B Q4_K_M | `https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4,920,739,232 B | 131072 [S] |
| 8B alt (thinking model) | `https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf` | 5,027,783,488 B | 40960 [S] |

Recommend the 1.5B plus Llama-3.1-8B for E2E (the 8B's 131K declared ctx exercises the full 2K→64K ladder and the spill region on 16 GB). The 0.5B is already downloaded in the #1 scratchpad and can be reused for CI-like smoke tests.

---

## 3. Benchmark architecture

### 3.1 Session state machine
```
pending → running ⇄ paused
running → cancelled | failed | done
paused  → cancelled | running
(app crash while running) → on next start: status=interrupted → user may Resume (→running) or Discard (→cancelled)
```
- Run-level states: `queued, running, ok, degraded, failed, skipped`. Each run records `fail_class` from §2.5.
- **Pause** = finish the current request, stop the server, persist. **Cancel** = abort the request (AbortController), kill, and mark remaining runs `skipped`.
- **Resume** = rerun every run not in {ok, degraded, failed, skipped}. Runs are idempotent units (a run = server launch + its requests). A hardware snapshot hash mismatch on resume (driver/runtime changed) → warn and offer a new session.

### 3.2 Run ordering
Per model (smallest file first, so early results arrive fast), per candidate config (§6):
1. `probe`: launch at ctx=4096 (or the lowest rung), verify /props, warmup, one speed rep. Fail → skip all configs of this model with the same or higher ngl.
2. `speed`: 3 reps at a 512-token prompt, 128 decode tokens.
3. `ctx_ladder`: rungs ascending (§3.5), 1 server launch per rung.
4. `quality`: only for the **best config per model** (quality is model+quant dependent, not ngl dependent). Run at ctx 8192 (or the ceiling if lower).
Server restarts between phases only when -c changes.

### 3.3 Safety guards (checked before each launch and on every telemetry sample)
| Guard | Rule | Action |
|---|---|---|
| RAM floor | `Available MBytes < max(2048, 8% of total RAM)` | kill run → `GUARD_ABORT(ram_floor)`. Pre-launch: skip if `RAM_est > Available - floor`. |
| VRAM saturation | per-PID dedicated ≥ 0.97 × qwMemorySize **or** per-PID shared delta > 2 GiB | kill → `GUARD_ABORT(vram_spill)`, rung status FAIL (spill is reportable data, not a crash) |
| Timeouts | §2.5 | kill |
| Disk | models drive free < model size + 2 GB | block download |
| Thermal | UNAVAILABLE on AMD | cooldown instead: 5 s idle between runs, 20 s after any run > 60 s |
| Cleanup | `finally{ kill server; kill typeperf }` on every path, orphan sweep at startup | — |
| Foreground load | if `Processor(_Total)` > 60 % for 5 s before a run while our server is idle | wait up to 60 s, then run and flag `noisy_env` |

### 3.4 Warmup and repetitions
- Keep built-in warmup. After /health, send (a) a 32-token prompt + 8 predicted tokens, (b) a prompt of the run's exact length + 8 tokens. Both discarded (compiles the Vulkan pipelines for every ubatch shape, see F7).
- Speed: 3 measured reps, median reported, CV stored. If CV of decode TPS > 15 %, run 2 more and take the median of 5; still > 15 % → flag `unstable`.
- Ladder rungs: 2 reps (prefill dominates), median.
- Quality: 1 rep (temperature 0, seed 1 → deterministic per build).

### 3.5 Context stress ladder
- Rungs: `[2048, 4096, 8192, 16384, 32768, 65536]` filtered by `≤ declared context_length` and `VRAM_est(rung) ≤ VRAM_budget` (for ngl=all) or `RAM_est ≤ Available - floor` (partial/0). Budget = `qwMemorySize - (adapter dedicated usage before launch) - 512 MiB`. The rung just above the estimated limit is **kept once** on purpose (cliff detection needs to see it, and the guards contain it).
- Each rung: server `-c rung`, prompt = deterministic filler of `rung - 128 - 64` tokens (built with /tokenize, then trimmed), with a needle sentence at 50 % depth ("The vault code is 7291-QX."), ending with the question "What is the vault code?". `n_predict=128`, `ignore_eos`. Record `needle_ok = /7291-QX/.test(first 64 chars of content)`.
- Filler: a seeded PRNG (mulberry32, seed=rung) choosing words from a fixed 2,000-word list in sentences. No repetition loops, same content for every model.
- Stop rules: stop after the first FAIL; after 2 consecutive DEGRADED; if rung time > 600 s; or on user cancel.

### 3.6 Recorded per run
`runtime_version` (build_info), exe hash, `hardware_snapshot_id`, the full argv, model sha256 (first-load hash, cached by path+size+mtime), config JSON, phase, rung, request params, raw timings JSON, derived TPS/TTFT, telemetry peaks/means, fail_class, stderr tail (≤ 200 lines on failure), load time (spawn → health 200), runtime-declared buffer sizes (log), started/ended timestamps.

---

## 4. Quality benchmark (suite `qb-1.0.0`, implemented)

Source of truth: `src/core/quality/tests.v1.json` (17 tests), checkers in `checkers.ts`, runner/scoring in `index.ts`. Not duplicated here.
- Categories: IF-01..03 instruction, RS-01..04 reasoning (RS-03: 2^20 mod 7 = **4**), CD-01..03 coding, SO-01..02 structured, EX-01..02 extraction, CR-10/50/90 context (needle at 10/50/90 % depth, seeded filler).
- All requests `temperature:0, seed:1`, no system prompt. Thinking models: reasoning/coding `max_tokens` ×4.
- `jsCode` cases are `{expr, expected}` compared via canonical JSON. Model code runs **out of process** (`sandbox.ts`: `process.execPath` + `ELECTRON_RUN_AS_NODE=1`, `--permission`, `--max-old-space-size`, fresh vm context, timeout, 64 KB stdout cap). Workers with `resourceLimits` were rejected: a heap blow-up aborts the host [V per sandbox.ts].
- `QualityResult {testId, category, weight, pass, score, detail}` (exported by `core/quality`, re-exported by `bench-types.ts`).
- Q = 100 · Σ_c W_c · passRate_c / Σ_c W_c over categories with results; passRate_c = Σ weight·pass / Σ weight; W = instruction .2, reasoning .25, coding .25, structured .1, extraction .1, context .1. Null when there are no results. Scoring restricts c to the profile's `promptSetIds` (§5.3).

---

## 5. Scoring and cliff detection (implemented: `src/core/scoring/*`, types in `src/shared/bench-types.ts`)

> **Calibrated 2026-09-27.** Current constants and rules are in `docs/BENCHMARK.md` and in code. Changes vs the text below: genSpeed is linear; unavailable inputs score a neutral 50; the scoring step and recommended ctx are separate (`maxContext`, TTFT tolerance); there is a TTFT gate and a partial-offload gate; the ladder includes 128K; the VRAM margin is 1 GiB with keep-over ≤ 1.15×; compute = 32 MiB + ub·n_embd·32 + 1 KiB·ctx; RAM counts the whole mmap'd file; ngl=0 is generated only without a GPU or as a fallback.

All thresholds and weights live in ONE plain object, `DEFAULT_SCORING_CONFIG` (`workloads.ts`, version `scoring-1.0.0`, stored per recommendation). Every function takes it as an optional last argument. Pure TS, no I/O, no clock, no randomness.

### 5.0 Provenance (ACCEPTANCE R1)
`type ProvenanceKind = 'measured' | 'declared' | 'estimated' | 'unavailable'`; `interface Metric<T=number> { value: T|null; kind; source?; reason? }`, where `value === null` iff `kind === 'unavailable'` (then `reason` says why). Unknown is never 0.
- **measured**: observed this session (timings, telemetry, sweep results, quality pass rates).
- **declared**: GGUF header, registry, driver, runtime log (VRAM total, ctxTrain, quant). `ModelMeta` fields are all declared.
- **estimated**: our formula, named in `source` (memory estimates; the quality prior). Estimates prune; they never rank except the quality prior, which is labelled in the breakdown and reasons.
- Mapping from `Sourced.status` (types.ts): `available` → `declared` or `measured` depending on the source; `unavailable|unsupported` → `unavailable`.
- Run status (A11 mapping): `RunStatus = pass|degraded|fail|timeout|cancelled` + `failureKind ∈ {oom, load_fail, device_lost, crash, exit_1, load_timeout, req_timeout, guard_abort, config_drift, skipped_memory}`. ok→pass; failed/oom/device_lost/crashed→fail+kind.

### 5.1 Usable steps
A step feeds scoring only if its status is `pass|degraded` **and** `decodeTps` is finite and > 0 (X4, X18). Cancelled, timed-out, crashed and zero-TPS runs become FAIL steps with a reason. TPS/TTFT ≤ 0 or non-finite are treated as unavailable, never as a value.

### 5.2 Component scores (0–100, absolute)
Fixed floors/targets per profile. There is no min-max or rank normalization across candidates, so a single candidate scores (X1) and an added candidate can't reorder others (X5). Unavailable input → score 0 with a note. There is no reweighting, so missing data never helps.
**Reference step** = the largest PASS step ≤ `targetContext`, else the smallest PASS step, else the smallest usable step. Speed, latency and memory are read there. It is never beyond the practical ceiling (X9).

| Component | Formula (constants in `norm`) |
|---|---|
| quality | §4 Q over the profile's categories (measured). No results → prior `min(90, 35 + 15·log2(params/1e9))` × {bpw≥6: 1, ≥4.5: .97, ≥3.5: .9, else .75}, **estimated** (X20) |
| genSpeed | `logScore(decodeTps, 2, genTarget)`; `logScore(x,f,t)=100·clamp(ln(x/f)/ln(t/f),0,1)` |
| prefillSpeed | `logScore(prefillTps, 20, prefillTarget)` |
| latency | 100 at TTFT ≤ tol/10, log-linear to 0 at tol. tol = `latencyToleranceMs` for a prompt filling the reference step (≈ctx−192 tok). Lower-is-better, inverted (X7) |
| memory | u = peak/total (VRAM, or RAM when ngl=0): 100 at u≤.80, linear to 40 at .97, linear to 0 at 1.0. Shared spill > 256 MiB → ≤30. Partial offload → ≤60 |
| stability | 100·usable/attempted over steps ≤ targetContext (ladder failures above target are expected, not instability), −20 if any crash/device_lost |
| context | `100·clamp(log2(ceiling/2048)/log2(target/2048),0,1)`. ceiling = measured practical ceiling (X9). All targets > 2048, so there is no 0/0 |

Total = Σ wᵢ·scoreᵢ (weights sum to 1, checked by a test, X6). Breakdown rows carry `{input: Metric, score, weight, contribution}` and sum exactly to the total (A16).
**Gates** (the candidate is ranked but ineligible): practical ceiling < 0.5·targetContext; stability < 50; quality < minQuality (estimated quality gates too, and the reason says "estimated").
**Ranking / tie-break (X3)**: eligible first → total rounded to 1e-6, desc → lower peak VRAM (CPU-only = 0) → lower peak RAM → `configId` ascending (code-unit order). Unknowns sort last. Output is deep-equal for shuffled input (A19).
**No-winner cases**: no inputs → "No recommendation: no candidates were benchmarked"; no usable step anywhere → "No recommendation: no successful runs" (A18), with every config in `excluded` and its step reasons; usable but all gated → "No recommendation: no candidate meets the <profile> requirements" + per-config gate failures. The app never picks the "least bad" config.
**Alternatives** (among eligible, same tie chain): fastest (decode TPS at ref), bestQuality, bestLongContext (practical ceiling, then decode), lowestMemory.
**Reasons** (plain English, deterministic): the score, "Only one candidate; not compared", the top-2 contributions, "Practical context 16K (measured); model declares 128K", "No VRAM spill up to 16K", every cliff message, and the estimated-quality notice.

### 5.3 Workload profiles (`WORKLOADS`)
| Profile | Q | G | P | L | M | S | C | targetCtx | genTarget | ppTarget | TTFT tol | minQ | quality categories |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| general_chat | .30 | .25 | .05 | .15 | .10 | .10 | .05 | 8K | 30 | 1000 | 8 s | 40 | instr, reason, struct, extract |
| coding | .40 | .20 | .10 | .10 | .05 | .10 | .05 | 16K | 30 | 1500 | 15 s | 50 | coding, instr, struct |
| long_context_coding | .30 | .10 | .20 | .05 | .05 | .10 | .20 | 32K | 20 | 2000 | 40 s | 50 | coding, context |
| reasoning | .45 | .20 | 0 | .05 | .05 | .15 | .10 | 8K | 25 | 1000 | 10 s | 55 | reason, instr |
| document_analysis | .30 | .05 | .25 | .05 | .05 | .10 | .20 | 32K | 15 | 2000 | 60 s | 45 | extract, context, struct |
| fast_assistant | .15 | .35 | .10 | .25 | .05 | .10 | 0 | 4K | 60 | 1000 | 2 s | 30 | instr, extract |
| max_quality | .70 | .05 | 0 | 0 | .05 | .15 | .05 | 8K | 8 | 500 | 20 s | 0 | all |

TTFT tolerances changed from v1 (1–30 s at a short prompt) because the ladder measures TTFT with a full-context prompt. All values are [A] until calibrated on real sweeps.

### 5.4 Cliff detection (`detectCliffs(steps, vramTotalBytes)`)
Steps are sorted by ctx. The relative rules compare each step with the **previous step only**, never the 2K baseline (X8).

| Constant (`cliff.*`) | Value | Rule → code | Rationale |
|---|---|---|---|
| `decodeDropRatio` | 0.60 | dec_b/dec_a ≤ .60 → `decode_drop` | A15: a ≥40 % drop in one doubling. A steady attention slowdown is ~5–15 %/step, so a single −35 % noise dip passes |
| `minDecodeDropTps` | 2 t/s | ...and the absolute drop is ≥ 2 t/s | no cliffs from tiny bases (X8) |
| `prefillDropPerDoubling` | 0.5 | pp_b/pp_a < 0.5^log2(ctx_b/ctx_a) → `prefill_drop` | worse than attention cost explains |
| `sharedSpillBytes` | 256 MiB | per-PID shared GPU mem above baseline → `shared_spill` (every step, including the first) | F9: per-PID, never adapter totals |
| `vramSaturation`, `ramGrowthBytes` | .80 (calibrated; WDDM spills at ≈83 %), 1 GiB | per-PID dedicated ≥ 80 % of VRAM **and** private RAM +1 GiB vs the previous step → `vram_spill` | A15 spill. RAM growth alone never flags (X11). Private working set excludes mmap file cache |

CPU % is never a cliff or spill signal. avgCpuUtil is decode-window only; the load spike is excluded from averages and kept in peaks (X10).
Verdicts: **FAIL** = unusable step (§5.1). **DEGRADED** = any rule fired, or the step is `beyond_limit` (sticky: every step after the first cliff/failure is at least degraded). **PASS** = otherwise.
`practicalContextCeiling` = the last step of the all-PASS prefix (measured). `degradedContextCeiling` = the last non-FAIL step before the first FAIL. `limitedBy` = none | cliff | failure | untested (X8: an OOM at 32K reads "failure", not "cliff"). `spillFreeUpTo` feeds the "no VRAM spill up to …" reason. Every reason carries `{code, metric, fromCtx, toCtx, from, to, ratio, threshold, message}`.
Known ceiling: stickiness means a real ≥40 % transient dip ends the ceiling. The defence is the median of reps upstream (§3.4). Add a "confirm on next step" rule if real sweeps show it.

---

## 6. Candidate generation (implemented: `src/core/benchmark/candidates.ts`)
`generateCandidates(machine, model, runtime, workload, rules = DEFAULT_CANDIDATE_RULES)` returns `{candidates, rejected}`. Every rejection and every skipped step carries a reason. Estimates are `kind:'estimated'` (§2.7 formula).
- Ladder `[2K…64K]` ∩ ≤ declared ctx (unknown → cap 8K). Steps above that are listed in `skippedSteps` with a reason (A13).
- VRAM budget = total − in-use-before-launch − 512 MiB. RAM budget = available (else total) − 4 GiB reserve (A25).
- Per step, RAM over budget → `skipped_memory`. This is never kept, because host OOM isn't contained. VRAM over budget → the **first** such step is kept once (to observe the cliff; guards contain it), and the rest are skipped.
- Order: (1) ngl=all f16. (2) ngl=all q8_0 KV if targetContext ≥ 32K and declared ≥ 32K. (3) if (1) is rejected: partial ngl=⌊L·f⌋, f ∈ .75/.5/.25, keep the largest that fits + the next lower one. (4) ngl=0 if there is no GPU / backend cpu, or params ≤ 3B. Max 4 per model (the rest are rejected "over cap"). Threads = physical cores. fa=on (the fa=off retry is runner policy).
- The VRAM total being unavailable (X14) doesn't empty the list: no VRAM pruning, and a note says the runtime guards apply. Low-confidence archs (SWA/hybrid/recurrent) prune on weights only.
- `machineFromProfile(profile, gpuDevice, vramInUse?)` picks the largest non-integrated GPU (X15). The iGPU is never the device. The `Vulkan0` mapping comes from `--list-devices`.
- Deferred (not in MVP): thread/ubatch sweeps, the duration estimate (X19; the UI can multiply steps × per-step time), and cross-session comparison (X17: compare within one session only).

---

## 7. Data model (node:sqlite, `PRAGMA journal_mode=WAL; foreign_keys=ON`)
Provenance convention: columns end in `_decl` (DECLARED), `_est` (ESTIMATED), `_meas` (MEASURED); unsuffixed = identity/bookkeeping. JSON columns are TEXT.
```sql
CREATE TABLE machine_profile(id INTEGER PRIMARY KEY, captured_at TEXT, snapshot_hash TEXT UNIQUE,
  os_caption_decl TEXT, os_build_decl TEXT, cpu_name_decl TEXT, cpu_cores_decl INT, cpu_threads_decl INT,
  ram_total_bytes_decl INT, disks_json_decl TEXT, raw_json TEXT);
CREATE TABLE gpu_profile(id INTEGER PRIMARY KEY, machine_id INT REFERENCES machine_profile,
  name_decl TEXT, vendor TEXT, pnp_id_decl TEXT, driver_version_decl TEXT, vram_bytes_decl INT,
  vram_source TEXT /*registry_qw|registry_bin|nvidia_smi|wmi_capped*/, is_discrete INT,
  pdh_luid_meas TEXT, vulkan_device TEXT, cuda_ok INT, telemetry_caps_json TEXT /*metric→AVAILABLE|UNAVAILABLE|UNSUPPORTED*/);
CREATE TABLE runtime(id INTEGER PRIMARY KEY, kind TEXT /*llamacpp|ollama|lmstudio*/, backend TEXT /*vulkan|cuda|cpu*/,
  version TEXT, build_info TEXT, path TEXT, exe_sha256 TEXT, installed_at TEXT, devices_json_decl TEXT);
CREATE TABLE model(id INTEGER PRIMARY KEY, path TEXT UNIQUE, file_bytes INT, sha256 TEXT, source TEXT /*download|ollama|lmstudio|user*/,
  arch_decl TEXT, name_decl TEXT, size_label_decl TEXT, param_count_decl INT, ftype_decl INT, quant_decl TEXT,
  ctx_train_decl INT, layers_decl INT, n_embd_decl INT, heads_decl INT, heads_kv_decl INT, head_dim_est INT,
  n_vocab_decl INT, swa_decl INT, experts_decl INT, supports_thinking_est INT, gguf_meta_json TEXT);
CREATE TABLE model_configuration(id INTEGER PRIMARY KEY, model_id INT REFERENCES model, runtime_id INT REFERENCES runtime,
  device TEXT, ngl TEXT, ctk TEXT, ctv TEXT, fa TEXT, threads INT, batch INT, ubatch INT, extra_args_json TEXT,
  vram_est_bytes_at_4k INT, kv_bytes_per_token_est INT, est_confidence TEXT, UNIQUE(model_id,runtime_id,device,ngl,ctk,ctv,fa,threads,batch,ubatch));
CREATE TABLE benchmark_session(id INTEGER PRIMARY KEY, machine_id INT, gpu_id INT, runtime_id INT,
  status TEXT /*pending|running|paused|interrupted|cancelled|failed|done*/, profile_set_json TEXT,
  plan_json TEXT, suite_version TEXT, scoring_version TEXT, created_at TEXT, started_at TEXT, ended_at TEXT, error TEXT);
CREATE TABLE benchmark_run(id INTEGER PRIMARY KEY, session_id INT REFERENCES benchmark_session, config_id INT REFERENCES model_configuration,
  seq INT, phase TEXT /*probe|speed|ctx_ladder|quality*/, ctx INT, rep INT, prompt_tokens INT, n_predict INT,
  status TEXT /*queued|running|ok|degraded|failed|skipped*/, fail_class TEXT, argv_json TEXT, pid INT,
  load_ms_meas REAL, prompt_n_meas INT, prompt_ms_meas REAL, predicted_n_meas INT, predicted_ms_meas REAL,
  prefill_tps_meas REAL, decode_tps_meas REAL, ttft_client_ms_meas REAL, ttft_server_ms_meas REAL,
  peak_vram_dedicated_meas INT, peak_vram_shared_delta_meas INT, peak_private_ws_meas INT, min_ram_avail_mb_meas INT,
  gpu_util_decode_mean_meas REAL, cpu_util_mean_meas REAL, needle_ok_meas INT, runtime_buffers_json_decl TEXT,
  timings_json TEXT, cliff_json TEXT /*reasons + rung status*/, stderr_tail TEXT, started_at TEXT, ended_at TEXT);
CREATE TABLE telemetry_sample(run_id INT REFERENCES benchmark_run, t_ms INT, metric TEXT, value REAL, PRIMARY KEY(run_id,t_ms,metric)) WITHOUT ROWID;
CREATE TABLE quality_test(id TEXT, version INT, suite_version TEXT, category TEXT, weight REAL, spec_json TEXT, PRIMARY KEY(id,version));
CREATE TABLE quality_result(id INTEGER PRIMARY KEY, run_id INT REFERENCES benchmark_run, test_id TEXT, test_version INT,
  passed_meas INT, output TEXT, reasoning_len INT, checker_detail TEXT, latency_ms_meas REAL, tokens_meas INT);
CREATE TABLE recommendation(id INTEGER PRIMARY KEY, session_id INT REFERENCES benchmark_session, profile TEXT,
  config_id INT REFERENCES model_configuration, rank INT, total_score REAL, components_json TEXT /*Q,G,P,L,M,S,C + inputs*/,
  practical_ctx_meas INT, degraded_ctx_meas INT, gates_json TEXT, rationale TEXT, scoring_version TEXT, created_at TEXT);
CREATE INDEX ix_run_session ON benchmark_run(session_id, seq);
```
Schema version in `PRAGMA user_version`. Migrations are an ordered array of SQL strings in core.

---

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Upstream flag churn (e.g. `--no-mmap` → `-lm`, `--fit` default on, `-ngl auto`) breaks argv | Pin the tag. On runtime install, run `--help` and assert every flag we use appears; fail with a clear message. |
| `releases/latest` points to a binary-less tag (F1) | List releases and match the asset regex `^llama-b(\d+)-bin-win-vulkan-x64\.zip$`. |
| WDDM oversubscription: Vulkan allocations may silently spill to shared memory instead of failing → slow, not OOM | Per-PID shared-delta guard and cliff rule. This is exactly what we report. |
| Other apps using VRAM (5.8 GB here at idle) skew results / budget | Baseline adapter dedicated usage before each launch, stored in the run. Flag `noisy_env` if it changes > 1 GiB during a run. |
| Localized PDH counter names | typeperf probe, then WMI class fallback (§1.2). |
| Orphan llama-server after crash/timeout (observed during probing) | PID registry in DB + startup sweep, `taskkill /T /F`. |
| iGPU picked as device / layers split to iGPU | Always explicit `-dev`. Map by `--list-devices` name to the discrete GPU. |
| Vulkan first-shape compile inflates TTFT | Shape warmup (§3.4). |
| AMD temps/power unavailable | Documented UNAVAILABLE. Cooldown pacing instead. |
| Memory estimate wrong for SWA/MoE/hybrid | Confidence flag, calibration ratio, and the guards (the estimate only prunes; it never decides). |
| Quality suite too small / contaminated | Version it (`qb-1.0.0`), keep checkers strict, show per-test results. It's a relative signal, not a leaderboard. |
| Model-generated JS execution | Permission model + vm context without globals + timeout + memory cap (network is not blocked by `--permission` in Node 24, hence vm). |
| Long ladders take hours for 8B@64K on partial offload | Stop rules, 600 s rung cap, time estimate before start, resume. |
| Antivirus/SmartScreen quarantining downloaded exe | Verify files after unzip. Surface "blocked" errors from spawn (EPERM/ENOENT). |
| Stale registry keys (F4), `nvidia-smi` exit 0 on failure (F5) | Join to present devices. Parse output. |
