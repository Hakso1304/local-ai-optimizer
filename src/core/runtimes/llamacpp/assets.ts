// Pick the llama.cpp Windows release asset for the GPU vendor. Pure. Naming verified against the b11208 release JSON
// (tests/fixtures/llamacpp-release-b11208.json):
//   llama-b<N>-bin-win-vulkan-x64.zip
//   llama-b<N>-bin-win-cuda-<maj.min>-x64.zip   + cudart-llama-bin-win-cuda-<maj.min>-x64.zip (no build number!)
// The cudart zip holds the CUDA runtime DLLs and must be extracted into the SAME directory as the CUDA build.
// UNTESTED on real NVIDIA hardware.

export interface ReleaseAsset {
  name: string
  size: number
  browser_download_url: string
  /** GitHub releases API "sha256:<hex>" (newer releases); checked after download when present. */
  digest?: string
}

const ROCM = /^llama-b\d+-bin-win-rocm-[\d.]+-x64\.zip$/
/** Opt-in AMD ROCm/HIP build (docs/HIP-BACKEND.md §1): one zip carries every supported gfx target (RDNA1–RDNA4);
 *  null if the release has none. */
export function pickRocmAsset(assets: ReleaseAsset[]): ReleaseAsset | null {
  return assets.find((a) => ROCM.test(a.name)) ?? null
}

/** sha256 of assets verified by hand (docs/HIP-BACKEND.md §1), for releases whose API entry has no digest. */
export const KNOWN_SHA256: Record<string, string> = {
  'llama-b11208-bin-win-rocm-10.0-x64.zip': '769c6476e709f890ac68b3e3c265fc7ab88d0e866ab3aacb67c182dd81db034a'
}
/** Expected sha256 (hex) of an asset, or null when neither the API nor the table knows it. */
export const expectedSha256 = (a: ReleaseAsset): string | null =>
  (a.digest?.startsWith('sha256:') ? a.digest.slice(7).toLowerCase() : null) ?? KNOWN_SHA256[a.name] ?? null

export interface AssetPick {
  /** What to install; null only if the release has neither a usable CUDA nor a Vulkan build. */
  main: ReleaseAsset | null
  /** cudart runtime zip for a CUDA main; null otherwise. */
  extra: ReleaseAsset | null
  /** Always the Vulkan build (if present): install it too / fall back to it when CUDA fails to load. */
  fallback: ReleaseAsset | null
  reason: string
}

const VULKAN = /^llama-b\d+-bin-win-vulkan-x64\.zip$/
const CUDA = /^llama-b\d+-bin-win-cuda-(\d+)\.(\d+)-x64\.zip$/

/** cudaMajor = highest CUDA major the driver supports (nvidia-smi banner "CUDA Version: X.Y"). Unknown → the oldest
 *  major offered (widest driver compatibility). */
export function pickReleaseAsset(
  assets: ReleaseAsset[],
  pref: { vendor: 'nvidia' | 'amd' | 'intel' | 'other'; cudaMajor?: number }
): AssetPick {
  const vulkan = assets.find((a) => VULKAN.test(a.name)) ?? null
  const vk = (reason: string): AssetPick => ({ main: vulkan, extra: null, fallback: vulkan, reason: vulkan ? reason : `${reason}; but the release has no win-vulkan-x64 asset` })
  if (pref.vendor !== 'nvidia') return vk(`${pref.vendor} GPU: Vulkan build`)

  const cuda = assets
    .map((a) => ({ a, m: CUDA.exec(a.name) }))
    .filter((x): x is { a: ReleaseAsset; m: RegExpExecArray } => x.m !== null)
    .map(({ a, m }) => ({ a, major: Number(m[1]), minor: Number(m[2]), ver: `${m[1]}.${m[2]}` }))
    .sort((x, y) => x.major - y.major || x.minor - y.minor)
  if (!cuda.length) return vk('NVIDIA GPU but the release has no win-cuda-x64 build: Vulkan build')
  const ok = pref.cudaMajor === undefined ? cuda.filter((c) => c.major === cuda[0].major) : cuda.filter((c) => c.major <= pref.cudaMajor!)
  if (!ok.length) return vk(`NVIDIA driver supports CUDA ${pref.cudaMajor}, oldest build needs ${cuda[0].ver}: Vulkan build`)
  const pick = ok[ok.length - 1]
  const extra = assets.find((a) => a.name === `cudart-llama-bin-win-cuda-${pick.ver}-x64.zip`) ?? null
  if (!extra) return vk(`CUDA ${pick.ver} build has no matching cudart-llama-bin-win-cuda-${pick.ver}-x64.zip: Vulkan build`)
  return {
    main: pick.a, extra, fallback: vulkan,
    reason: `NVIDIA GPU: CUDA ${pick.ver} build${pref.cudaMajor === undefined ? ' (driver CUDA version unknown; oldest major chosen)' : ` (driver supports CUDA ${pref.cudaMajor})`}; extract ${extra.name} into the same directory`
  }
}
