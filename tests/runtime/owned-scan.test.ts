import { describe, expect, it, vi } from 'vitest'
import { reconcileOwnedProcessScan, type ProcessIdentity, type ProcessTree } from '../../src/core/runtimes/llamacpp'

const root: ProcessIdentity = { pid: 81001, name: 'node.exe', startedAt: '2026-09-28T00:00:00.000Z' }
const child: ProcessIdentity = { pid: 81002, parentPid: root.pid, name: 'node.exe', startedAt: '2026-09-28T00:00:01.000Z' }
const grandchild: ProcessIdentity = { pid: 81003, parentPid: child.pid, name: 'node.exe', startedAt: '2026-09-28T00:00:20.000Z' }

function fakeTree(found: ProcessIdentity[], live: Map<number, ProcessIdentity>) {
  const killed: number[] = []
  const tree: ProcessTree = {
    descendants: vi.fn(async () => found),
    inspect: vi.fn(async (pid) => live.get(pid) ?? null),
    killVerified: vi.fn(async (record) => {
      if (live.get(record.pid)?.startedAt !== record.startedAt) return false
      killed.push(record.pid)
      live.delete(record.pid)
      return true
    }),
    kill: async () => { throw new Error('unverified numeric-PID kill must not be used') },
    isAlive: async (pid) => live.has(pid)
  }
  return { tree, killed }
}

describe('exported owned-process scan reconciliation (injected identities)', () => {
  it('reaps a newly observed child whose parent chain binds to the live root identity', async () => {
    const live = new Map([[root.pid, root], [child.pid, child]])
    const { tree, killed } = fakeTree([child], live)
    const children = new Map<number, ProcessIdentity>()
    const result = await reconcileOwnedProcessScan({ tree, root: root.pid, rootIdentity: root,
      children, rootWasLive: true, ownedExitAt: null })
    expect(children.get(child.pid)).toEqual(child)
    expect(result.remaining).toEqual([child])
    await result.reap()
    expect(killed).toEqual([child.pid])
    expect(live.has(child.pid)).toBe(false)
  })

  it('fails with survivor identity when an unknown child has no verified parent chain', async () => {
    const live = new Map([[child.pid, child]])
    const { tree, killed } = fakeTree([child], live)
    await expect(reconcileOwnedProcessScan({ tree, root: root.pid, rootIdentity: root,
      children: new Map(), rootWasLive: false, ownedExitAt: Date.parse('2026-09-28T00:00:10.000Z') }))
      .rejects.toThrow(/unverified descendant 81002/)
    expect(killed).toEqual([])
    expect(live.has(child.pid)).toBe(true)
  })

  it('leaves a foreign process alone when a recorded descendant PID has a new creation identity', async () => {
    const foreign = { ...child, startedAt: '2026-09-28T00:00:20.000Z' }
    const live = new Map([[foreign.pid, foreign]])
    const { tree, killed } = fakeTree([foreign], live)
    const children = new Map([[child.pid, child]])
    const result = await reconcileOwnedProcessScan({ tree, root: root.pid, rootIdentity: root,
      children, rootWasLive: false, ownedExitAt: Date.parse('2026-09-28T00:00:10.000Z') })
    expect(result.remaining).toEqual([])
    await result.reap()
    expect(killed).toEqual([])
    expect(live.get(foreign.pid)).toEqual(foreign)
  })

  it('Y1: reaps or reports a grandchild born after root exit through its captured parent identity', async () => {
    const live = new Map([[child.pid, child], [grandchild.pid, grandchild]])
    const { tree, killed } = fakeTree([grandchild], live)
    const children = new Map([[child.pid, child]])
    const error = await reconcileOwnedProcessScan({ tree, root: root.pid, rootIdentity: root,
      children, rootWasLive: false, ownedExitAt: Date.parse('2026-09-28T00:00:10.000Z') })
      .then(async (result) => { await result.reap(); return null }, (reason: unknown) => reason)
    expect(error !== null || !live.has(grandchild.pid), 'owned grandchild cannot silently survive reconciliation').toBe(true)
    if (error === null) expect(killed).toContain(grandchild.pid)
  })

  it('Y1 control: skips a post-exit row whose parent is the exited root PID (a PID reuser\'s child), neither adopted nor fail-closed', async () => {
    // A dead root cannot spawn: a row naming the exited root as parent and born after its exit belongs to whatever
    // reused that PID. Pins the distinction against the chain-less 99999 case below, which must fail closed.
    const reuserChild = { pid: 81004, parentPid: root.pid, name: 'node.exe', startedAt: '2026-09-28T00:00:20.000Z' }
    const live = new Map([[reuserChild.pid, reuserChild]])
    const { tree, killed } = fakeTree([reuserChild], live)
    const children = new Map<number, ProcessIdentity>()
    const result = await reconcileOwnedProcessScan({ tree, root: root.pid, rootIdentity: root,
      children, rootWasLive: false, ownedExitAt: Date.parse('2026-09-28T00:00:10.000Z') })
    expect(children.size).toBe(0)
    expect(result.remaining).toEqual([])
    await result.reap()
    expect(killed).toEqual([])
    expect(live.get(reuserChild.pid)).toEqual(reuserChild)
  })

  it('Y1: fails closed on a post-exit scan row with no captured parent-identity chain', async () => {
    const foreign = { ...grandchild, parentPid: 99999 }
    const live = new Map([[foreign.pid, foreign]])
    const { tree, killed } = fakeTree([foreign], live)
    await expect(reconcileOwnedProcessScan({ tree, root: root.pid, rootIdentity: root,
      children: new Map(), rootWasLive: false, ownedExitAt: Date.parse('2026-09-28T00:00:10.000Z') }))
      .rejects.toThrow(/unverified descendant|survivor|unknown ancestry/i)
    expect(killed).toEqual([])
    expect(live.get(foreign.pid)).toEqual(foreign)
  })
})
