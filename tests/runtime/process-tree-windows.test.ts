import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { LlamaCppBackend, windowsProcessTree, type ProcessIdentity } from '../../src/core/runtimes/llamacpp'
import { trackOwnedProcess } from '../../scripts/owned-process'

describe.skipIf(process.platform !== 'win32')('real Windows process identity, disposable Node only', () => {
  it('finds a live Node grandchild by identity and reaps both identities', async () => {
    const script = "const { spawn } = require('node:child_process'); const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore', windowsHide: true }); process.send({ pid: child.pid }); setInterval(() => {}, 1000)"
    const child = spawn(process.execPath, ['-e', script],
      { stdio: ['ignore', 'ignore', 'ignore', 'ipc'], windowsHide: true })
    let grandchildPid: number | null = null
    try {
      await once(child, 'spawn')
      expect(child.pid).toBeGreaterThan(0)
      const [message] = await Promise.race([
        once(child, 'message'),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error('Node grandchild PID not reported')), 5000))
      ])
      grandchildPid = (message as { pid: number }).pid
      expect(grandchildPid).toBeGreaterThan(0)
      const identity = await windowsProcessTree.inspect!(child.pid!)
      expect(identity).toMatchObject({ pid: child.pid, name: expect.stringMatching(/node/i), startedAt: expect.stringMatching(/^\d{4}-\d\d-\d\dT/) })
      const descendants = await windowsProcessTree.descendants(child.pid!)
      const grandchild = descendants.find((record) => record.pid === grandchildPid)
      expect(grandchild, 'the real descendant scan must find the live grandchild').toBeDefined()
      expect(grandchild).toMatchObject({ parentPid: child.pid, name: expect.stringMatching(/node/i), startedAt: expect.stringMatching(/^\d{4}-\d\d-\d\dT/) })
      expect(await windowsProcessTree.inspect!(grandchildPid!)).toEqual(grandchild)
      expect(await windowsProcessTree.killVerified!(grandchild!)).toBe(true)
      expect(await windowsProcessTree.isAlive(grandchildPid!)).toBe(false)
      const closed = once(child, 'close')
      expect(await windowsProcessTree.killVerified!(identity!)).toBe(true)
      await closed
      expect(await windowsProcessTree.isAlive(child.pid!)).toBe(false)
    } finally {
      if (grandchildPid != null) { try { process.kill(grandchildPid) } catch { /* already reaped */ } }
      if (child.exitCode === null && child.signalCode === null) child.kill()
    }
  }, 30_000)

  // Y1 (real-process probe): R → C (captured) ; R exits ; C then spawns G and exits. G's parent is a dead captured
  // intermediate, so a census seeded only at the root cannot reach it. stop() must reap G or reject — never resolve clean.
  it('Y1 real: an orphan of a dead captured child (root exited) is reaped or reported, never a clean stop', async () => {
    // Event-driven (CIM scans take 1–3 s each): R reports C's PID over IPC and exits on command; C spawns G only
    // after a go-file appears (i.e. after R is gone), writes G's PID and exits, leaving G parented to a dead PID.
    const tag = `${process.pid}-${Date.now()}`
    const goFile = join(tmpdir(), `y1-real-go-${tag}`), pidFile = join(tmpdir(), `y1-real-g-${tag}.txt`)
    const gCode = 'setInterval(() => {}, 1000)'
    const cCode = `const { spawn } = require('node:child_process'), fs = require('node:fs'); const t = setInterval(() => { if (!fs.existsSync(${JSON.stringify(goFile)})) return; clearInterval(t); const g = spawn(process.execPath, ['-e', ${JSON.stringify(gCode)}], { stdio: 'ignore', detached: true, windowsHide: true }); fs.writeFileSync(${JSON.stringify(pidFile)}, String(g.pid)); g.unref(); setTimeout(() => process.exit(0), 300) }, 100)`
    const rCode = `const { spawn } = require('node:child_process'); const c = spawn(process.execPath, ['-e', ${JSON.stringify(cCode)}], { stdio: 'ignore', detached: true, windowsHide: true }); c.unref(); process.send({ cPid: c.pid }); process.on('message', () => process.exit(0))`
    const root = spawn(process.execPath, ['-e', rCode], { stdio: ['ignore', 'ignore', 'ignore', 'ipc'], windowsHide: true })
    let gPid: number | null = null
    let cPid: number | null = null
    const waitFor = async (what: string, ok: () => Promise<boolean> | boolean, ms = 20_000) => {
      for (const end = Date.now() + ms; !(await ok());) { if (Date.now() > end) throw new Error(`timed out waiting for ${what}`); await new Promise((r) => setTimeout(r, 100)) }
    }
    try {
      const [message] = await once(root, 'message')
      cPid = (message as { cPid: number }).cPid
      await waitFor('C visible under R', async () => (await windowsProcessTree.descendants(root.pid!)).some((d) => d.pid === cPid))
      const owned = await trackOwnedProcess(root, windowsProcessTree, 60_000) // only this snapshot captures C
      expect([...owned.descendants.keys()], 'precondition: C captured while R lives').toContain(cPid)
      root.send('exit')
      await waitFor('root exit', () => root.exitCode !== null)
      writeFileSync(goFile, 'go')
      await waitFor('grandchild G', () => existsSync(pidFile))
      gPid = Number(readFileSync(pidFile, 'utf8'))
      await waitFor('C exit', async () => (await windowsProcessTree.inspect!(cPid!)) === null)
      expect(await windowsProcessTree.inspect!(gPid), 'precondition: G is alive and orphaned').not.toBeNull()
      const error = await owned.stop().then(() => null, (reason: unknown) => reason)
      const gAlive = (await windowsProcessTree.inspect!(gPid)) !== null
      expect(error !== null || !gAlive, `stop() resolved cleanly while orphan G ${gPid} (parent ${cPid}, dead) is alive`).toBe(true)
    } finally {
      for (const pid of [gPid, cPid]) if (pid != null) { try { process.kill(pid) } catch { /* already gone */ } }
      if (root.exitCode === null && root.signalCode === null) root.kill()
      rmSync(pidFile, { force: true }); rmSync(goFile, { force: true })
    }
  }, 90_000)

  // 54e7a71 regression guard: killSync compared Get-Process ticks with the CIM identity exactly, which differ by up
  // to 9 ticks (µs truncation), so on app quit it killed nothing.
  it('killSync kills an owned process identified by the production inspect', async () => {
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore', windowsHide: true })
    try {
      await once(child, 'spawn')
      const identity = await windowsProcessTree.inspect!(child.pid!)
      expect(identity).toMatchObject({ pid: child.pid })
      const b = new LlamaCppBackend('unused')
      ;(b as unknown as { ownedChildren: Map<number, ProcessIdentity> }).ownedChildren.set(child.pid!, identity!)
      const closed = once(child, 'close')
      b.killSync()
      await Promise.race([closed, new Promise((_, reject) => setTimeout(() => reject(new Error('killSync left the owned process alive')), 10_000))])
      expect(await windowsProcessTree.isAlive(child.pid!)).toBe(false)
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill()
    }
  }, 30_000)
})
