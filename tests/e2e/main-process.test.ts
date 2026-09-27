// Integrated main-process regression (W4c T01): the real built app (out/) with its real IPC handlers, driven over
// CDP like the manual verification runs. Uses a throwaway --user-data-dir, never the real app data. Skipped when
// there is no build or no Electron binary (run `npm run build` first).
import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { openDb } from '../../src/core/storage/db'
import { saveSession } from '../../src/core/storage/sessions'

const root = join(__dirname, '..', '..')
const electronExe = (() => { try { return createRequire(import.meta.url)('electron') as unknown as string } catch { return '' } })()
const canRun = existsSync(join(root, 'out', 'main', 'index.js')) && typeof electronExe === 'string' && existsSync(electronExe)
const PORT = 9399
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function cdp(port: number) {
  let targets: { type: string; webSocketDebuggerUrl: string }[] = []
  for (let i = 0; i < 60 && !targets.some((t) => t.type === 'page'); i++) {
    try { targets = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()) as typeof targets } catch { /* not up yet */ }
    await sleep(500)
  }
  const page = targets.find((t) => t.type === 'page')
  if (!page) throw new Error('app window never appeared')
  const ws = new WebSocket(page.webSocketDebuggerUrl)
  await new Promise((r) => ws.addEventListener('open', r))
  let id = 0
  const pending = new Map<number, (v: unknown) => void>()
  ws.addEventListener('message', (m) => { const d = JSON.parse(String(m.data)); if (d.id && pending.has(d.id)) { pending.get(d.id)!(d); pending.delete(d.id) } })
  const ev = async <T>(expr: string): Promise<T> => {
    const r = (await new Promise((res) => { const i = ++id; pending.set(i, res as (v: unknown) => void); ws.send(JSON.stringify({ id: i, method: 'Runtime.evaluate', params: { expression: expr, awaitPromise: true, returnByValue: true } })) })) as { result: { result: { value: T } } }
    return r.result.result.value
  }
  return { ev, close: () => ws.close() }
}

describe.skipIf(!canRun)('main process (real Electron, real IPC)', { timeout: 120_000 }, () => {
  const base = mkdtempSync(join(tmpdir(), 'lao-e2e-'))
  const userDataArg = join(base, 'ud') // dev build appends "-dev" to the userData path
  const userData = `${userDataArg}-dev`
  let app: ChildProcess
  let page: Awaited<ReturnType<typeof cdp>>

  beforeAll(async () => {
    mkdirSync(userData, { recursive: true })
    const db = openDb(join(userData, 'optimizer.db'))
    saveSession(db, { workload: 'coding', vramBytes: null, candidates: [] }, 'running') // left "running" by a crashed run
    db.close()
    app = spawn(electronExe, [root, `--user-data-dir=${userDataArg}`, `--remote-debugging-port=${PORT}`], { stdio: 'ignore' })
    page = await cdp(PORT)
    await sleep(1500)
  })
  afterAll(async () => {
    page?.close()
    app?.kill()
    await sleep(1500)
    rmSync(base, { recursive: true, force: true })
  })

  it('marks a session left running by a previous run as interrupted on launch', async () => {
    const list = await page.ev<{ status: string }[]>('window.api.listSessions()')
    expect(list.map((s) => s.status)).toEqual(['interrupted'])
  })

  it('bench:start rejects bad renderer input through the real handlers', async () => {
    const r = await page.ev<{ ok: boolean; error?: string }[]>(`Promise.all([
      window.api.startBench({ workload: '__proto__', modelIds: ['C:\\\\x.gguf'] }),
      window.api.startBench({ workload: 'coding', modelIds: ['C:\\\\Windows\\\\evil.gguf'] }),
      window.api.startBench({ workload: 'coding', modelIds: [], requiredContext: 12345 })
    ])`)
    expect(r.map((x) => x.ok)).toEqual([false, false, false])
    expect(r[0].error).toMatch(/unknown workload/)
    expect(r[1].error).toMatch(/not in a configured model dir/)
    const bad = await page.ev<string>(`window.api.getSession(-1).then(() => 'resolved', (e) => String(e))`)
    expect(bad).toMatch(/bad id/)
  })

  it('a second instance exits at once and leaves the first running', async () => {
    const second = spawn(electronExe, [root, `--user-data-dir=${userDataArg}`], { stdio: 'ignore' })
    const code = await new Promise<number | null>((r) => { second.on('exit', r); setTimeout(() => { second.kill(); r(-1) }, 30_000) })
    expect(code).toBe(0)
    expect(await page.ev<number>('document.querySelectorAll("nav button").length')).toBeGreaterThan(0)
  })
})
