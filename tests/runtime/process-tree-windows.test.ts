import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { describe, expect, it } from 'vitest'
import { LlamaCppBackend, windowsProcessTree, type ProcessIdentity } from '../../src/core/runtimes/llamacpp'

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
