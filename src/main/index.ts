import { app, BrowserWindow, ipcMain } from 'electron'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { scanSystem } from '../core/system/scanner'
import { detectRuntimes } from '../core/runtimes'
import { LlamaCppBackend, killStaleServer } from '../core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../core/runtimes/llamacpp/parse'
import { openDb } from '../core/storage/db'
import type { SmokeResult } from '../shared/types'

// ponytail: app.getAppPath() is the project root in dev/preview; revisit for packaged builds.
const llamaDir = () => join(app.getAppPath(), 'vendor', 'llama.cpp')

// Large models live on D: (user decision); used when settings.json has no modelDirs key.
const DEFAULT_MODEL_DIRS = ['D:\\llm-models']

/** <project>/models plus dirs from userData/settings.json ({ "modelDirs": [...] }). Missing dirs are skipped. */
function modelDirs(): string[] {
  const file = join(app.getPath('userData'), 'settings.json')
  let extra = DEFAULT_MODEL_DIRS
  if (existsSync(file)) {
    const s = JSON.parse(readFileSync(file, 'utf8')) as { modelDirs?: unknown }
    if (Array.isArray(s.modelDirs)) extra = s.modelDirs.filter((d): d is string => typeof d === 'string')
  }
  return [join(app.getAppPath(), 'models'), ...extra]
}

const llama = new LlamaCppBackend(llamaDir(), { pidFile: join(app.getPath('userData'), 'llama-server.pid') })
let smokeBusy = false

ipcMain.handle('runtimes:detect', () => detectRuntimes(llamaDir()))
ipcMain.handle('system:scan', async () => {
  const [profile, runtimes] = await Promise.all([scanSystem(), detectRuntimes(llamaDir())])
  return { ...profile, runtimes }
})
ipcMain.handle('models:list', () => llama.enumerateModels(modelDirs()))
ipcMain.handle('bench:smoke', async (_e, modelPath: string): Promise<SmokeResult> => {
  if (typeof modelPath !== 'string' || !modelDirs().some((d) => modelPath.startsWith(d))) throw new Error('model path not in a configured model dir')
  if (smokeBusy) throw new Error('a smoke run is already in progress')
  smokeBusy = true
  try {
    // Deliberately tiny: ctx 2048, 32 tokens, one warmup + one measured request.
    const dev = pickDiscreteDevice(await llama.listDevices())
    if (!dev) throw new Error('no discrete Vulkan device reported by llama-server --list-devices')
    const load = await llama.loadModel({ modelPath, contextSize: 2048, gpuLayers: 99, device: dev.id })
    const text = 'Explain in one sentence what a GPU does.'
    await llama.warmup(text)
    const prompt = await llama.runPrompt({ prompt: text, maxTokens: 32, timeoutMs: 60_000 })
    return { load, prompt }
  } finally {
    await llama.unloadModel()
    smokeBusy = false
  }
})

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: '#0f1115',
    title: 'Local AI Optimizer',
    webPreferences: { preload: join(__dirname, '../preload/index.js'), sandbox: true, contextIsolation: true }
  })
  if (process.env.ELECTRON_RENDERER_URL) win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else win.loadFile(join(__dirname, '../renderer/index.html'))
}

app.whenReady().then(async () => {
  // Opened+migrated at startup so schema problems surface immediately; nothing writes yet.
  openDb(join(app.getPath('userData'), 'optimizer.db'))
  const stale = await killStaleServer(join(app.getPath('userData'), 'llama-server.pid')).catch((e: Error) => `stale-server check failed: ${e.message}`)
  if (stale) console.warn(stale)
  createWindow()
})
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => llama.killSync())
process.on('exit', () => llama.killSync())
