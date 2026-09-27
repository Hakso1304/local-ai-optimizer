import { app, BrowserWindow, ipcMain } from 'electron'
import { join } from 'node:path'
import { scanSystem } from '../core/system/scanner'
import { detectRuntimes } from '../core/runtimes'

// ponytail: app.getAppPath() is the project root in dev/preview; revisit for packaged builds.
const llamaDir = () => join(app.getAppPath(), 'vendor', 'llama.cpp')

ipcMain.handle('runtimes:detect', () => detectRuntimes(llamaDir()))
ipcMain.handle('system:scan', async () => {
  const [profile, runtimes] = await Promise.all([scanSystem(), detectRuntimes(llamaDir())])
  return { ...profile, runtimes }
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

app.whenReady().then(createWindow)
app.on('window-all-closed', () => app.quit())
