import { expect, it, vi } from 'vitest'
vi.mock('electron', () => ({ BrowserWindow: vi.fn(), safeStorage: { isEncryptionAvailable: () => false } }))
vi.mock('node:fs', () => ({
  existsSync: (p: string) => !p.endsWith('.part') && !p.endsWith('.bin'),
  readFileSync: vi.fn(), writeFileSync: vi.fn(), rmSync: vi.fn(), statSync: () => ({ size: 0 }),
  realpathSync: Object.assign((p: string) => p, { native: (p: string) => p }), // hub-logic resolves links (F4)
  statfsSync: () => ({ bavail: 100 * 1024 ** 3, bsize: 1 })
}))
vi.mock('../../src/core/hub/hf', () => ({
  HubError: class extends Error {}, downloadFile: vi.fn(), listGgufFiles: vi.fn(), searchModels: vi.fn(), whoami: vi.fn()
}))
import { registerHubIpc } from '../../src/main/hub'
import { downloadFile, listGgufFiles } from '../../src/core/hub/hf'

it('claims the single-download lock before awaiting the Hub listing', async () => {
  const handlers = new Map<string, (...args: any[]) => any>()
  registerHubIpc({ handle: (name: string, fn: any) => handlers.set(name, fn) } as any, () => null,
    { userDataDir: 'C:\\user', modelDirs: () => ['C:\\models'] })
  let release!: (v: any) => void
  const listing = new Promise<any>((r) => { release = r })
  vi.mocked(listGgufFiles).mockReturnValue(listing)
  const downloads: ((v: any) => void)[] = []
  vi.mocked(downloadFile).mockImplementation(() => new Promise((r) => { downloads.push(r) }))
  const request = { repoId: 'owner/model', path: 'same.gguf', destDir: 'C:\\models' }
  const a = handlers.get('hub:download')!({}, request)
  const b = handlers.get('hub:download')!({}, request)
  release([{ path: 'same.gguf', sizeBytes: 3, sha256: null }])
  await Promise.resolve(); await Promise.resolve()
  for (const done of downloads) done({ filePath: 'same.gguf', sha256Verified: null })
  await Promise.all([a, b])
  expect(downloadFile).toHaveBeenCalledTimes(1)
})
