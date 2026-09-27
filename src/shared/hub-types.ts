// Renderer-facing contract for the Hugging Face hub feature (src/main/hub.ts ↔ src/preload/hub.ts ↔ HubPage). Types only.
import type { HfGgufFile, HfModel, HubErrorKind } from '../core/hub/hf'

export type { HfGgufFile, HfModel }

export interface HubProgress {
  repoId: string
  path: string
  bytes: number
  total: number | null
  bytesPerSec: number
  /** Seconds remaining; null when the total or the speed is unknown. */
  etaSec: number | null
  /** 0–100; null when the total is unknown. */
  pct: number | null
}

export type HubFailKind = HubErrorKind | 'disk_full' | 'busy' | 'bad_dest' | 'no_secure_storage'
export type HubResult<T> = ({ ok: true } & T) | { ok: false; kind: HubFailKind; error: string }

export interface HubAccount { signedIn: boolean; name: string | null; error?: string }

export interface HubApi {
  hubWhoami(): Promise<HubAccount>
  /** Validates the token with /api/whoami-v2, then stores it encrypted (Electron safeStorage). */
  hubLogin(token: string): Promise<HubResult<{ name: string }>>
  hubLogout(): Promise<void>
  /** Opens huggingface.co/settings/tokens in an app window (separate session partition). */
  hubOpenTokenPage(): Promise<void>
  hubDirs(): Promise<string[]>
  hubSearch(query: string): Promise<HubResult<{ models: HfModel[] }>>
  hubFiles(repoId: string): Promise<HubResult<{ files: HfGgufFile[] }>>
  /** One download at a time. Calling it again for the same file resumes from the .part. */
  hubDownload(req: { repoId: string; path: string; destDir: string }): Promise<HubResult<{ filePath: string; sha256Verified: boolean | null }>>
  /** Stops the running download. discard=false keeps the .part (pause); true deletes it (cancel). */
  hubCancel(discard?: boolean): Promise<void>
  onHubProgress(cb: (p: HubProgress) => void): () => void
}
