import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { describe, expect, it } from 'vitest'
import { windowsProcessTree } from '../../src/core/runtimes/llamacpp'

describe.skipIf(process.platform !== 'win32')('real Windows process identity, disposable Node only', () => {
  it('inspects an owned PID, scans descendants and kills only that creation-time identity', async () => {
    const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'],
      { stdio: 'ignore', windowsHide: true })
    try {
      await once(child, 'spawn')
      expect(child.pid).toBeGreaterThan(0)
      const identity = await windowsProcessTree.inspect!(child.pid!)
      expect(identity).toMatchObject({ pid: child.pid, name: expect.stringMatching(/node/i), startedAt: expect.stringMatching(/^\d{4}-\d\d-\d\dT/) })
      const descendants = await windowsProcessTree.descendants(child.pid!)
      expect(descendants.every((record) => Number.isSafeInteger(record.pid) && record.startedAt.endsWith('Z'))).toBe(true)
      expect(descendants.every((record) => record.parentPid === child.pid || descendants.some((parent) => parent.pid === record.parentPid))).toBe(true)
      const closed = once(child, 'close')
      expect(await windowsProcessTree.killVerified!(identity!)).toBe(true)
      await closed
      expect(await windowsProcessTree.isAlive(child.pid!)).toBe(false)
    } finally {
      if (child.exitCode === null && child.signalCode === null) child.kill()
    }
  }, 20_000)
})
