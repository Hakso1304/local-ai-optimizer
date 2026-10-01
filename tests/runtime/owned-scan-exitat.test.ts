// Item 4 controls for the exported owned-process reconciliation: kill-time bound for orphans of captured
// descendants, reused parent PIDs, and cross-seed identity conflicts. Fake identities only (no real processes).
// The fake census is SOURCE-AWARE: descendants(seed) returns only rows whose parent chain reaches that seed, like
// the Windows census (a dead seed still finds rows that record it as their ParentProcessId).
// Contract (agreed with #2): OwnedScanOptions.childExitAt — captured pid → epoch ms at which we confirmed it gone.
import { describe, expect, it, vi } from 'vitest'
import { reconcileOwnedProcessScan, type OwnedScanOptions, type ProcessIdentity, type ProcessTree } from '../../src/core/runtimes/llamacpp'

const t = (s: number) => new Date(Date.UTC(2026, 9, 2, 0, 0, s)).toISOString()
const ms = (s: number) => Date.UTC(2026, 9, 2, 0, 0, s)
const root: ProcessIdentity = { pid: 82001, name: 'node.exe', startedAt: t(0) }
const P: ProcessIdentity = { pid: 82002, parentPid: root.pid, name: 'node.exe', startedAt: t(1) }
const rootExit = ms(10)

type Options = OwnedScanOptions & { childExitAt?: Map<number, number> }

/** rows: every row the OS would list. live: what inspect() sees now. */
function fakeTree(rows: ProcessIdentity[], live: Map<number, ProcessIdentity>, seedRows?: (seed: number) => ProcessIdentity[] | undefined) {
  const killed: number[] = []
  const reachable = (seed: number) => {
    const seen = new Set([seed]), out: ProcessIdentity[] = []
    for (let added = true; added;) {
      added = false
      for (const r of rows) if (!seen.has(r.pid) && r.parentPid !== undefined && seen.has(r.parentPid)) { seen.add(r.pid); out.push(r); added = true }
    }
    return out
  }
  const tree: ProcessTree = {
    descendants: vi.fn(async (seed: number) => seedRows?.(seed) ?? reachable(seed)),
    inspect: vi.fn(async (pid: number) => live.get(pid) ?? null),
    killVerified: vi.fn(async (record: ProcessIdentity) => {
      if (live.get(record.pid)?.startedAt !== record.startedAt) return false
      killed.push(record.pid); live.delete(record.pid); return true
    }),
    kill: async () => { throw new Error('unverified numeric-PID kill must not be used') },
    isAlive: async (pid: number) => live.has(pid)
  }
  return { tree, killed }
}

const run = (o: Options) => reconcileOwnedProcessScan(o).then(async (r) => { await r.reap(); return { ok: true as const, r } }, (e: unknown) => ({ ok: false as const, e }))

describe('owned scan: kill-time bound, reused parent PIDs, cross-seed conflicts (item 4)', () => {
  it('(a) adopts and reaps an orphan of a dead captured parent born within [P.startedAt, P.exitAt]', async () => {
    const X: ProcessIdentity = { pid: 82003, parentPid: P.pid, name: 'node.exe', startedAt: t(20) }
    const live = new Map([[X.pid, X]]) // P is gone (we killed it at 30 s); X is its orphan
    const { tree, killed } = fakeTree([X], live)
    const children = new Map([[P.pid, P]])
    const res = await run({ tree, root: root.pid, rootIdentity: root, children, rootWasLive: false, ownedExitAt: rootExit, childExitAt: new Map([[P.pid, ms(30)]]) })
    expect(res.ok, res.ok ? '' : String((res as { e: unknown }).e)).toBe(true)
    expect(children.get(X.pid)).toEqual(X) // adopted into the caller's Map (its survivor sweep reads it)
    expect(killed).toEqual([X.pid])
    expect(live.has(X.pid)).toBe(false)
  })

  it('(b) skips a row born after the dead captured parent\'s exit: not ours, not killed, no error', async () => {
    const X: ProcessIdentity = { pid: 82004, parentPid: P.pid, name: 'node.exe', startedAt: t(40) }
    const live = new Map([[X.pid, X]])
    const { tree, killed } = fakeTree([X], live)
    const children = new Map([[P.pid, P]])
    const res = await run({ tree, root: root.pid, rootIdentity: root, children, rootWasLive: false, ownedExitAt: rootExit, childExitAt: new Map([[P.pid, ms(30)]]) })
    expect(res.ok, res.ok ? '' : String((res as { e: unknown }).e)).toBe(true)
    expect(children.has(X.pid)).toBe(false)
    expect(killed).toEqual([])
    expect(live.get(X.pid)).toEqual(X)
  })

  it('(c) skips a child of a live process that reused the captured parent PID, born after the reuser started', async () => {
    const F: ProcessIdentity = { pid: P.pid, parentPid: 99999, name: 'other.exe', startedAt: t(25) } // P's PID, new owner
    const X: ProcessIdentity = { pid: 82005, parentPid: P.pid, name: 'helper.exe', startedAt: t(26) }
    const live = new Map([[F.pid, F], [X.pid, X]])
    const { tree, killed } = fakeTree([X], live)
    const children = new Map([[P.pid, P]])
    // P's own exit time is unknown (it exited by itself): only the reuser's start time can classify X
    const res = await run({ tree, root: root.pid, rootIdentity: root, children, rootWasLive: false, ownedExitAt: rootExit, childExitAt: new Map() })
    expect(res.ok, res.ok ? '' : String((res as { e: unknown }).e)).toBe(true) // no false ServerStuckError
    expect(killed).toEqual([])
    expect(live.get(X.pid)).toEqual(X)
    expect(live.get(F.pid)).toEqual(F)
  })

  it('(d) fails closed on a row under a reused parent PID born before the reuser started (orphan of ours, unprovable)', async () => {
    const F: ProcessIdentity = { pid: P.pid, parentPid: 99999, name: 'other.exe', startedAt: t(25) }
    const X: ProcessIdentity = { pid: 82006, parentPid: P.pid, name: 'node.exe', startedAt: t(24) }
    const live = new Map([[F.pid, F], [X.pid, X]])
    const { tree, killed } = fakeTree([X], live)
    const res = await run({ tree, root: root.pid, rootIdentity: root, children: new Map([[P.pid, P]]), rootWasLive: false, ownedExitAt: rootExit, childExitAt: new Map() })
    expect(res.ok).toBe(false)
    expect(String((res as { e: unknown }).e)).toMatch(/unverified descendant|survivor|ancestry|82006/i)
    expect(killed).toEqual([])
    expect(live.get(X.pid)).toEqual(X)
  })

  it('(e) fails closed when two census seeds report the same PID with different creation identities', async () => {
    const live = new Map([[P.pid, P]])
    const viaRoot: ProcessIdentity = { pid: 82007, parentPid: root.pid, name: 'node.exe', startedAt: t(5) }
    const viaP: ProcessIdentity = { pid: 82007, parentPid: P.pid, name: 'node.exe', startedAt: t(6) }
    live.set(viaP.pid, viaP)
    const { tree, killed } = fakeTree([], live, (seed) => (seed === root.pid ? [viaRoot] : seed === P.pid ? [viaP] : []))
    const res = await run({ tree, root: root.pid, rootIdentity: root, children: new Map([[P.pid, P]]), rootWasLive: false, ownedExitAt: rootExit, childExitAt: new Map() })
    expect(res.ok).toBe(false)
    expect(String((res as { e: unknown }).e)).toMatch(/conflict|unverified descendant|82007/i)
    expect(killed).toEqual([])
  })
})
