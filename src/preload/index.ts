import { contextBridge, ipcRenderer } from 'electron'
import type { RendererApi } from '../shared/types'

const api: RendererApi = {
  scanSystem: () => ipcRenderer.invoke('system:scan'),
  detectRuntimes: () => ipcRenderer.invoke('runtimes:detect'),
  listModels: () => ipcRenderer.invoke('models:list'),
  benchSmoke: (modelPath) => ipcRenderer.invoke('bench:smoke', modelPath)
}
contextBridge.exposeInMainWorld('api', api)
