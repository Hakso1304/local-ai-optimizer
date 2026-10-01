import type { ChildProcess } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { describe, expect, it, vi } from 'vitest'
import { trackOwnedProcess } from '../scripts/owned-process'
import type { ProcessIdentity, ProcessTree } from '../src/core/runtimes/llamacpp'

const root: ProcessIdentity = { pid: 424243, name: 'node.exe', startedAt: '2026-09-28T00:00:00.0000000Z' }
const descendant: ProcessIdentity = { pid: 424247, parentPid: root.pid, name: 'child.exe', startedAt: '2026-09-28T00:00:01.0000000Z' }
const grandchild: ProcessIdentity = { pid: 424248, parentPid: descendant.pid, name: 'grandchild.exe', startedAt: '2026-09-28T00:00:20.0000000Z' }

function fakeProcess() {
  const events = new EventEmitter() as EventEmitter & { pid: number; exitCode: number | null; signalCode: string | null; kill: () => boolean }
  events.pid = root.pid; events.exitCode = null; events.signalCode = null
  events.kill = vi.fn(() => { events.exitCode = 0; events.emit('close', 0); return true })
  return events
}


/** Source-aware fake census, like the real walkers: only rows reachable from `seed` through parentPid. A fake that
 *  returns every row for any seed hides the dead-intermediate orphan class (Y1 real-process probe). */
function reachableFrom(rows: ProcessIdentity[], seed: number): ProcessIdentity[] {
  const seen = new Set([seed]), out: ProcessIdentity[] = []
  for (let added = true; added;) {
    added = false
    for (const r of rows) if (!seen.has(r.pid) && r.parentPid !== undefined && seen.has(r.parentPid)) { seen.add(r.pid); out.push(r); added = true }
  }
  return out
}
describe('identity-bound harness process teardown (injected fake tree)', () => {
  it('reaps a captured child even when its parent exited before teardown', async () => {
    const p = fakeProcess()
    let rootLive = true, childLive = true
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async (seed) => reachableFrom([descendant], seed), kill: async () => {}, isAlive: async () => false,
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
      descendants: async (seed) => reachableFrom([descendant], seed), kill: async () => {}, isAlive: async () => false,
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
      descendants: async (seed) => reachableFrom([descendant], seed), kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid ? (rootLive ? root : null) : pid === descendant.pid ? descendant : null,
      killVerified: async (record) => { if (record.pid === root.pid) { rootLive = false; p.exitCode = 0; p.emit('close', 0) } return true }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    await expect(owned.stop()).rejects.toThrow(/teardown could not be verified/)
    expect([...owned.descendants.values()]).toEqual([descendant])
  })

  it('W1: never reports success when an owned child appears at root kill after every earlier scan', async () => {
    const p = fakeProcess()
    let rootLive = true, lateChildLive = false
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async (seed) => reachableFrom(lateChildLive && rootLive ? [descendant] : [], seed),
      kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid ? rootLive ? root : null : pid === descendant.pid && lateChildLive ? descendant : null,
      killVerified: async (record) => {
        killed.push(record.pid)
        if (record.pid === root.pid) {
          lateChildLive = true // born immediately before the root dies; earlier scans saw no child
          rootLive = false; p.exitCode = 0; p.emit('close', 0)
        } else if (record.pid === descendant.pid) lateChildLive = false
        return true
      }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    expect([...owned.descendants.values()]).toEqual([])
    const error = await owned.stop().then(() => null, (reason: unknown) => reason)
    expect(error !== null || !lateChildLive, 'uncaptured live child cannot be reported as clean teardown').toBe(true)
    if (error === null) expect(killed).toContain(descendant.pid)
  })

  it('Y1: never reports clean stop when a captured child creates a grandchild after natural root exit', async () => {
    const p = fakeProcess()
    let rootLive = true, childLive = true, grandchildLive = false
    let lateGrandchild = grandchild
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async (seed) => reachableFrom([descendant, ...(grandchildLive ? [lateGrandchild] : [])], seed),
      kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root :
        pid === descendant.pid && childLive ? descendant : pid === lateGrandchild.pid && grandchildLive ? lateGrandchild : null,
      killVerified: async (record) => {
        killed.push(record.pid)
        if (record.pid === descendant.pid) childLive = false
        if (record.pid === grandchild.pid) grandchildLive = false
        return true
      }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    expect([...owned.descendants.values()]).toEqual([descendant])
    rootLive = false; p.exitCode = 0; p.emit('close', 0)
    await new Promise((resolve) => setTimeout(resolve, 10))
    lateGrandchild = { ...grandchild, startedAt: new Date().toISOString() }
    grandchildLive = true
    const error = await owned.stop().then(() => null, (reason: unknown) => reason)
    expect(error !== null || !grandchildLive, 'owned grandchild must be reaped or reported as survivor').toBe(true)
    if (error === null) expect(killed).toContain(grandchild.pid)
  })

  it('Y1: reaps a grandchild a captured child spawns while being killed after root exit (born within its parent\x27s lifetime)', async () => {
    // w4y's harder shape: the grandchild is born INSIDE killVerified(child), with the parent still alive at that
    // moment and the root already gone. Born within the captured parent's lifetime [startedAt, exit], it is owned
    // (rule A′, childExitAt): it must be killed and gone, nothing foreign touched, and stop() resolves.
    const p = fakeProcess()
    let rootLive = true, childLive = true, grandchildLive = false
    let lateGrandchild = grandchild
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async (seed) => reachableFrom([...(childLive ? [descendant] : []), ...(grandchildLive ? [lateGrandchild] : [])], seed),
      kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root :
        pid === descendant.pid && childLive ? descendant : pid === lateGrandchild.pid && grandchildLive ? lateGrandchild : null,
      killVerified: async (record) => {
        killed.push(record.pid)
        if (record.pid === descendant.pid) {
          lateGrandchild = { ...grandchild, startedAt: new Date().toISOString() } // born while its parent still lives
          grandchildLive = true
          childLive = false
        }
        if (record.pid === grandchild.pid) grandchildLive = false
        return true
      }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    expect([...owned.descendants.values()]).toEqual([descendant])
    rootLive = false; p.exitCode = 0; p.emit('close', 0)
    await new Promise((resolve) => setTimeout(resolve, 10))
    await expect(owned.stop()).resolves.toBeUndefined()
    expect(killed).toContain(descendant.pid)
    expect(killed, 'the grandchild born inside its parent\x27s lifetime is reaped').toContain(grandchild.pid)
    expect(grandchildLive).toBe(false)
    expect(killed.every((pid) => pid === root.pid || pid === descendant.pid || pid === grandchild.pid), `foreign kill in ${killed}`).toBe(true)
  })

  it('Y1: fails closed without killing a foreign post-exit descendant lacking an owned parent chain', async () => {
    const p = fakeProcess()
    // Reachable shape: X (99999) is a post-exit child of the exited root PID (a PID reuser, skipped as foreign);
    // the row under X has no captured owned parent, so it must fail closed and nothing may be killed.
    let reuser: ProcessIdentity = { pid: 99999, parentPid: root.pid, name: 'reuser.exe', startedAt: root.startedAt }
    let foreign = { ...grandchild, parentPid: 99999 }
    let rootLive = true
    const killed: number[] = []
    const tree: ProcessTree = {
      descendants: async (seed) => reachableFrom(rootLive ? [] : [reuser, foreign], seed),
      kill: async () => {}, isAlive: async () => false,
      inspect: async (pid) => pid === root.pid && rootLive ? root : pid === foreign.pid ? foreign : pid === reuser.pid && !rootLive ? reuser : null,
      killVerified: async (record) => { killed.push(record.pid); return true }
    }
    const owned = await trackOwnedProcess(p as unknown as ChildProcess, tree, 10_000)
    rootLive = false; p.exitCode = 0; p.emit('close', 0)
    await new Promise((resolve) => setTimeout(resolve, 10))
    reuser = { ...reuser, startedAt: new Date().toISOString() }
    foreign = { ...foreign, startedAt: new Date().toISOString() }
    await expect(owned.stop()).rejects.toThrow(/unverified descendant|survivor|ancestry/i)
    expect(killed).not.toContain(foreign.pid)
    expect(killed).not.toContain(reuser.pid)
  })
})
