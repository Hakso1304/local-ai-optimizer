// Preload side of the hub feature. Spread into the bridge: `const api = { ...existing, ...hubPreloadApi(ipcRenderer) }`.
import type { IpcRenderer, IpcRendererEvent } from 'electron'
import type { HubApi, HubProgress } from '../shared/hub-types'

export function hubPreloadApi(ipc: IpcRenderer): HubApi {
  return {
    hubWhoami: () => ipc.invoke('hub:whoami'),
    hubLogin: (token) => ipc.invoke('hub:login', token),
    hubLogout: () => ipc.invoke('hub:logout'),
    hubOpenTokenPage: () => ipc.invoke('hub:openTokenPage'),
    hubDirs: () => ipc.invoke('hub:dirs'),
    hubSearch: (q) => ipc.invoke('hub:search', q),
    hubFiles: (repoId) => ipc.invoke('hub:files', repoId),
    hubDownload: (req) => ipc.invoke('hub:download', req),
    hubCancel: (discard) => ipc.invoke('hub:cancel', discard),
    onHubProgress: (cb) => {
      const h = (_e: IpcRendererEvent, p: HubProgress) => cb(p)
      ipc.on('hub:progress', h)
      return () => { ipc.removeListener('hub:progress', h) }
    }
  }
}
