import { contextBridge, ipcRenderer } from 'electron'
import type { RendererApi } from '../shared/types'

const api: RendererApi = {
  scanSystem: () => ipcRenderer.invoke('system:scan'),
  detectRuntimes: () => ipcRenderer.invoke('runtimes:detect')
}
contextBridge.exposeInMainWorld('api', api)
