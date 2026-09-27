import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import type { SessionEvent } from '../shared/bench-events'
import type { RendererApi } from '../shared/types'

const api: RendererApi = {
  scanSystem: () => ipcRenderer.invoke('system:scan'),
  detectRuntimes: () => ipcRenderer.invoke('runtimes:detect'),
  listModels: () => ipcRenderer.invoke('models:list'),
  benchSmoke: (modelPath) => ipcRenderer.invoke('bench:smoke', modelPath),
  getSettings: () => ipcRenderer.invoke('settings:get'),
  setWorkload: (w) => ipcRenderer.invoke('settings:setWorkload', w),
  listWorkloads: () => ipcRenderer.invoke('workloads:list'),
  listSessions: () => ipcRenderer.invoke('sessions:list'),
  getSession: (id) => ipcRenderer.invoke('sessions:get', id),
  latestRecommendation: (w) => ipcRenderer.invoke('recommendation:latest', w),
  startBench: (req) => ipcRenderer.invoke('bench:start', req),
  cancelBench: () => ipcRenderer.invoke('bench:cancel'),
  pauseBench: () => ipcRenderer.invoke('bench:pause'),
  resumeBench: (id, opts) => ipcRenderer.invoke('bench:resume', id, opts),
  installRuntime: () => ipcRenderer.invoke('runtime:install'),
  onRuntimeProgress: (cb) => {
    const h = (_e: IpcRendererEvent, msg: string) => cb(msg)
    ipcRenderer.on('runtime:progress', h)
    return () => { ipcRenderer.removeListener('runtime:progress', h) }
  },
  onBenchEvent: (cb) => {
    const h = (_e: IpcRendererEvent, ev: SessionEvent) => cb(ev)
    ipcRenderer.on('bench:event', h)
    return () => { ipcRenderer.removeListener('bench:event', h) }
  }
}
contextBridge.exposeInMainWorld('api', api)
