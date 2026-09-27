import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { describe, expect, it, vi } from 'vitest'
import { trackOwnedProcess } from '../scripts/owned-process'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'

const root: ProcessIdentity = { pid: 424243, name: 'node.exe', startedAt: '2026-09-28T00:00:00.0000000Z' }
const descendant: ProcessIdentity = { pid: 424247, parentPid: root.pid, name: 'child.exe', startedAt: '2026-09-28T00:00:01.0000000Z' }

function fakeProcess() {
  const events = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean }
  events.pid = root.pid; events.exitCode = null; events.signalCode = null
  events.kill = vi.fn(() => { events.exitCode = 0; events.emit('close', 0); return true })
  return events
}

describe('identity-bound harness process teardown (injected fake tree)', () => {
  it('reaps a captured child even when its parent exited before teardown', async () => {
    const p = fakeProcess()
    let rootLive = true, childLive = true
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async () => [descendant], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid ? rootLive ? root : null : pid === descendant.pid && childLive ? descendant : null,
      killVerified: async (record) => { killed.push(record.pid); if (record.pid === descendant.pid) childLive = false; return true }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    expect([...owned.descendants.values()]).toEqual([descendant])
    rootLive = false; p.exitCode = 0; p.emit('close', 0)
    await expect(owned.stop()).resolves.toBeUndefined()
    expect(killed).toEqual([descendant.pid])
  })

  it('does not kill a foreign process that reused a captured descendant PID', async () => {
    const p = fakeProcess()
    const foreign = { ...descendant, startedAt: '2026-09-28T00:02:00.0000000Z' }
    let childIdentity = descendant, rootLive = true
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async () => [descendant], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid ? rootLive ? root : null : pid === descendant.pid ? childIdentity : null,
      killVerified: async (record) => { killed.push(record.pid); if (record.pid === root.pid) { rootLive = false; p.exitCode = 0; p.emit('close', 0) } return true }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    childIdentity = foreign
    await expect(owned.stop()).resolves.toBeUndefined()
    expect(killed).toEqual([root.pid])
  })

  it('reports a surviving owned child as a teardown failure', async () => {
    const p = fakeProcess()
    let rootLive = true
    const tree: ProcessTree = {
      descendants: async () => [descendant], kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid ? (rootLive ? root : null) : pid === descendant.pid ? descendant : null,
      killVerified: async (record) => { if (record.pid === root.pid) { rootLive = false; p.exitCode = 0; p.emit('close', 0) } return true }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    await expect(owned.stop()).rejects.toThrow(/teardown could not be verified/)
    expect([...owned.descendants.values()]).toEqual([descendant])
  })
})
