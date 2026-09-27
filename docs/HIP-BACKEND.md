# HIP (ROCm) backend: investigation and design

Status: **investigation only, not wired.** The build was downloaded and `--version` was run. No `--list-devices` and no model load were done, because GPU testing is on hold. 2026-09-28, Opus #1.

## Why this matters
- Per-process VRAM ceiling on this card with llama.cpp Vulkan: ≈ 11.6–13.25 GiB of 15.92 (cal-2026-09-27).
- Ollama on an RX 6800 16 GB used the full 16 GB. Ollama uses the ROCm/HIP backend there.
- So the usable VRAM is backend-specific. A HIP build may fit larger models or contexts on the same card. That is unproven until we measure it (LIMITATIONS: FOLLOW-UP "AMD backend choice").

## 1. Release assets (verified 2026-09-28 against the GitHub releases API)
- **Asset.** Every Windows release since at least b11208 ships `llama-b<N>-bin-win-rocm-10.0-x64.zip`.
  - b11208: 257,364,326 B, sha256 `769c6476e709f890ac68b3e3c265fc7ab88d0e866ab3aacb67c182dd81db034a`.
  - b11218 (latest): 245 MiB.
  - There is also `…-ubuntu-rocm-10.0-x64.tar.gz`. There is no `-hip-`, `-radeon` or per-gfx split: one zip covers every supported target.
- **gfx targets.** They are embedded as code objects in `ggml-hip.dll` (0.99 GB; each target appears 278×):
  - `gfx1010 gfx1011 gfx1012` (RDNA1)
  - `gfx1030–gfx1036` (RDNA2, incl. **RX 6800 = gfx1030**)
  - `gfx1100–gfx1103` (RDNA3)
  - `gfx1150–gfx1153` (RDNA3.5)
  - `gfx1200 gfx1201` (RDNA4, incl. **RX 9070 XT = gfx1201**)
  - No gfx9 (Vega/MI) code objects.
- **Prerequisites.**
  - The zip bundles the HIP runtime: `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`. There are no separate rocBLAS/hipBLASLt DLLs (the kernels are compiled in).
  - So no HIP SDK install is needed to run it.
  - It still depends on the installed Adrenalin kernel driver. Whether the current driver exposes gfx1201 to HIP runtime 7 is **unverified** until `--list-devices` is allowed.
- `--version` output (the Vulkan build prints the same):
  - `version: 0.5.0-dev (build 11208, commit 85ca3b52c)`
  - `built with Clang 20.1.8 for Windows x86_64`

## 2. Side-by-side install
- The HIP build is extracted to `vendor/llama.cpp-hip/` (1.2 GB unpacked; `vendor/` is git-ignored). The Vulkan build stays in `vendor/llama.cpp/`.
- Both are the same build and commit, so an A/B isolates the backend.
- Each dir is self-contained (own `ggml.dll`, `llama.dll`, `ggml-*` backend DLLs), and the DLLs load from the exe's dir. No PATH changes are needed, and the two builds don't collide.
- The app already supports another exe via `LlamaCppBackend.configure({ exePath | vendorDir })`. A second backend instance can point at the HIP dir.

## 3. Device naming and flags
- **Device name.** HIP devices are named **`ROCm<N>`**: the `ROCm` string appears 5,562× in `ggml-hip.dll`, along with `ROCm_Host` for the pinned host buffer.
  - `session.ts` `GPU_DEVICE` already accepts `ROCm\d+`.
  - `hostPinnedBytes` must treat `ROCm_Host` as host memory. It already does, since only `^(Vulkan|CUDA|ROCm|…)\d+$` counts as a device.
- **`-dev`.** It is `-dev ROCm0` instead of `Vulkan0`. `pickDiscreteDevice` must map the scan's GPU to the right `ROCm<N>` via `--list-devices`, which is not run yet.
- **Unified memory.** `GGML_CUDA_ENABLE_UNIFIED_MEMORY` (env) is present, and HIP can allocate managed memory (`hipMallocManaged`). This may be how Ollama "uses 16 GB": oversubscription into system RAM would look like full VRAM.
  - The benchmark must **not** set it by default.
  - Its telemetry must keep per-PID dedicated and shared separate, as it does now.
- **Other flags.** `-fa`, `-ctk/-ctv q8_0`, `-nkvo`, `--cache-ram 0`, `-fit off` and `--parallel 1` are shared flags and apply unchanged.

## 4. Backend as a candidate axis (design)
- **Types.** Add `CandidateConfig.backend: 'vulkan' | 'hip' | 'cuda'`.
  - It currently comes from `SessionDeps.backendKind`.
  - `configId` gets a suffix only for non-default backends: `…|t=8|hip`. Existing Vulkan ids stay stable and stored sessions still resolve.
- **Planner.** `generateCandidates` runs once per installed backend with the same rules, then candidates are concatenated.
  - Per-backend memory estimates are identical, since the weights and KV math don't change.
  - The per-process budget differs: observations are already keyed `pnp|drv|<backend>:<build>` (eb30f93). HIP observations never prune Vulkan plans and vice versa.
- **Runner.** Today it has one `backend()` factory. It becomes `backendFor(cand.backend)`, which returns the Vulkan or HIP `LlamaCppBackend` (same class, different `vendorDir`). The sampler is backend-agnostic (per-PID counters).
- **Versions.** Rows record `versions.runtime` = `<backend>:<build>`, so the identity proof (I-6.0) and the comparability checks separate backends.
- **Export.**
  - `ExportConfig.backend` makes `toLlamaServerCommand` emit the matching exe (the `vendor/llama.cpp-hip/llama-server.exe` path, or a note to install the ROCm build) and `-dev ROCm0`.
  - The export-equivalence test compares against the HIP backend's argv too.
- **Installer.** `pickReleaseAsset` for `vendor: 'amd'` also returns the `win-rocm-*-x64.zip` as an optional second install. It is opt-in, because it is 245 MiB vs 32 MiB. Only gfx targets present in `ggml-hip.dll` qualify; the scan's PNP `DEV_` id would need mapping to gfx, which is not implemented.
- **Interpretation.** Backend is a comparable axis, like KV type. I-2.8 and I-4.x wording already names "GPU/driver/backend". Cross-backend speed comparisons need the same `versions.benchmark` and prompts, which already applies.

## 5. Effort estimate
| Piece | Effort |
|---|---|
| Types + configId suffix + planner per backend | 0.5 day |
| Runner `backendFor`, versions.runtime prefix, tests | 0.5 day |
| Device mapping via `--list-devices` (needs the GPU) | 0.25 day + a GPU session |
| Installer opt-in asset + UI toggle | 0.5 day |
| Export (exe path, `-dev ROCm0`) + equivalence tests | 0.25 day |
| First calibration A/B: Vulkan vs HIP, same model/ctx ladder | 1 GPU session |

Total ≈ 2 dev-days plus two GPU sessions. The GPU gate is the risk: whether HIP runtime 7 on the current Adrenalin driver enumerates gfx1201 at all, and whether it is stable under `-fa on`.

## Next step when the GPU hold lifts
`vendor/llama.cpp-hip/llama-server.exe --list-devices` should show `ROCm0: AMD Radeon RX 9070 XT (… MiB)`. Then run the Vulkan-vs-HIP ladder for 8B f16 at 32K/64K with identical argv apart from `-dev`, and compare per-PID dedicated and shared.
