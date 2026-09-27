// Hugging Face login + download IPC. Register once from main: registerHubIpc(ipcMain, () => mainWindow, deps).
// The token lives only in memory and in userData/hf-token.bin encrypted with Electron safeStorage (DPAPI on
// Windows); it is never written in plaintext and never sent to the renderer.
import { BrowserWindow, safeStorage, type IpcMain } from 'electron'
import { existsSync, readFileSync, rmSync, statSync, statfsSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { downloadFile, listGgufFiles, searchModels, whoami, type HfGgufFile } from '../core/hub/hf'
import type { HubAccount, HubResult } from '../shared/hub-types'
import { diskCheck, isAllowedDest, progressInfo, resolvedProbe, userMessage } from './hub-logic'

export interface HubDeps { userDataDir: string; modelDirs: () => string[] }

export function registerHubIpc(ipcMain: IpcMain, getWindow: () => BrowserWindow | null, deps: HubDeps): void {
  const tokenFile = join(deps.userDataDir, 'hf-token.bin')
  let token: string | null = null
  // One download at a time. The operation is claimed synchronously (before any await) and owns its .part file (W4b F6/F8).
  let active: { ctl: AbortController; part: string | null; discard: boolean } | null = null
  // Bumped by logout: a whoami still in flight must not restore a token after the user signed out (W4b F20).
  let loginGen = 0
  const timeout = <T>(p: Promise<T>, ms = 15_000): Promise<T> =>
    Promise.race([p, new Promise<T>((_, rej) => setTimeout(() => rej(new Error(`no answer from Hugging Face within ${ms / 1000} s`)), ms))])

  const loadToken = (): string | null => {
    if (token) return token
    if (!existsSync(tokenFile) || !safeStorage.isEncryptionAvailable()) return null
    try { token = safeStorage.decryptString(readFileSync(tokenFile)) } catch { token = null }
    return token
  }
  const fail = (e: unknown) => ({ ok: false as const, ...userMessage(e) })

  ipcMain.handle('hub:whoami', async (): Promise<HubAccount> => {
    const t = loadToken()
    if (!t) return { signedIn: false, name: null }
    const r = await timeout(whoami(t)).catch((e: Error) => ({ error: e.message }))
    return 'error' in r ? { signedIn: false, name: null, error: r.error } : { signedIn: true, name: r.name }
  })

  ipcMain.handle('hub:login', async (_e, t: unknown): Promise<HubResult<{ name: string }>> => {
    if (typeof t !== 'string' || !/^hf_[A-Za-z0-9]{20,}$/.test(t.trim())) return { ok: false, kind: 'auth_required', error: 'That does not look like a Hugging Face token (hf_…).' }
    if (!safeStorage.isEncryptionAvailable()) return { ok: false, kind: 'no_secure_storage', error: 'Secure storage is unavailable, so the token cannot be saved.' }
    const gen = ++loginGen
    const r = await timeout(whoami(t.trim())).catch((e: Error) => ({ error: e.message }))
    if (gen !== loginGen) return { ok: false, kind: 'auth_required', error: 'Signed out while the token was being checked.' }
    if ('error' in r) return { ok: false, kind: 'auth_required', error: `Token rejected: ${r.error}` }
    token = t.trim()
    writeFileSync(tokenFile, safeStorage.encryptString(token), { mode: 0o600 })
    return { ok: true, name: r.name }
  })

  ipcMain.handle('hub:logout', () => { loginGen++; token = null; rmSync(tokenFile, { force: true }) })

  ipcMain.handle('hub:openTokenPage', () => {
    // Separate persistent partition: the HF login cookies stay in their own session, never read by the app.
    const w = new BrowserWindow({
      width: 1000, height: 800, parent: getWindow() ?? undefined, title: 'Hugging Face — Access Tokens',
      webPreferences: { partition: 'persist:huggingface', sandbox: true, contextIsolation: true, nodeIntegration: false }
    })
    w.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    void w.loadURL('https://huggingface.co/settings/tokens')
  })

  ipcMain.handle('hub:dirs', () => deps.modelDirs())

  ipcMain.handle('hub:search', async (_e, q: unknown) => {
    if (typeof q !== 'string' || !q.trim()) return { ok: true, models: [] }
    try { return { ok: true, models: await searchModels(q.trim(), { token: loadToken() ?? undefined, limit: 30 }) } } catch (e) { return fail(e) }
  })

  ipcMain.handle('hub:files', async (_e, repoId: unknown) => {
    if (typeof repoId !== 'string' || !/^[\w.-]+\/[\w.-]+$/.test(repoId)) return { ok: false, kind: 'not_found', error: 'bad repository id' }
    try { return { ok: true, files: await listGgufFiles(repoId, { token: loadToken() ?? undefined }) } } catch (e) { return fail(e) }
  })

  ipcMain.handle('hub:download', async (_e, req: { repoId?: unknown; path?: unknown; destDir?: unknown }) => {
    if (active) return { ok: false, kind: 'busy', error: 'Another download is running.' }
    const op = { ctl: new AbortController(), part: null as string | null, discard: false }
    active = op // claimed before the first await: a second click can't start a parallel download of the same file
    try {
      const { repoId, path, destDir } = req ?? {}
      if (typeof repoId !== 'string' || typeof path !== 'string' || typeof destDir !== 'string') return { ok: false, kind: 'bad_path', error: 'bad request' }
      if (!isAllowedDest(destDir, deps.modelDirs())) return { ok: false, kind: 'bad_dest', error: 'Destination must be one of the configured model folders.' }
      const t = loadToken() ?? undefined
      let file: HfGgufFile | undefined
      try {
        // Size and sha256 come from the Hub listing, not from the renderer.
        file = (await listGgufFiles(repoId, { token: t })).find((f) => f.path === path)
      } catch (e) { return fail(e) }
      if (op.ctl.signal.aborted) return { ok: false, kind: 'cancelled', error: 'Cancelled.' } // cancel during the listing
      if (!file) return { ok: false, kind: 'not_found', error: `${path} is not a GGUF file in ${repoId}` }
      const part = `${resolve(destDir, ...path.split('/'))}.part`
      op.part = part
      const partBytes = existsSync(part) ? statSync(part).size : 0
      // Free space of the volume the file really lands on (a junction may point to another drive; W4b F19).
      const probe = resolvedProbe(destDir)
      if (!probe) return { ok: false, kind: 'bad_dest', error: `The destination drive for ${destDir} does not exist.` }
      let free: number
      try { const fs = statfsSync(probe); free = fs.bavail * fs.bsize } catch (e) { return { ok: false, kind: 'bad_dest', error: `Cannot read free space of ${probe}: ${(e as Error).message}` } }
      const disk = diskCheck(free, file.sizeBytes, partBytes)
      if (!disk.ok) return { ok: false, kind: 'disk_full', error: disk.error }
      try {
        const r = await downloadFile({
          repoId, path, destDir, token: t, sha256: file.sha256, sizeBytes: file.sizeBytes, signal: op.ctl.signal,
          onProgress: (bytes, total, bps) => getWindow()?.webContents.send('hub:progress', progressInfo(repoId, path, bytes, total, bps))
        })
        return { ok: true, filePath: r.filePath, sha256Verified: r.sha256Verified }
      } catch (e) {
        return fail(e)
      }
    } finally {
      // downloadFile has settled, so its stream is closed: only now drop THIS operation's partial on Cancel.
      if (op.discard && op.part) rmSync(op.part, { force: true })
      if (active === op) active = null
    }
  })

  ipcMain.handle('hub:cancel', (_e, discard?: unknown) => {
    if (!active) return
    if (discard === true) active.discard = true // Cancel deletes the partial; Pause keeps it for resume
    active.ctl.abort()
  })
}
