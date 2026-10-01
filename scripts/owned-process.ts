import { execFile, type ChildProcess } from 'node:child_process'
import { promisify } from 'node:util'
import { reconcileOwnedProcessScan, windowsProcessTree, type ProcessIdentity, type ProcessTree } from '../src/core/runtimes/llamacpp'

const same = (a: ProcessIdentity | null, b: ProcessIdentity) => !!a && a.pid === b.pid && a.startedAt === b.startedAt
const execFileAsync = promisify(execFile)

/** A root-independent final census still sees an orphaned Windows child by its recorded parent PID. */
async function finalDescendantCensus(root: number): Promise<ProcessIdentity[]> {
  const script = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,Name,@{n='CreationDate';e={$_.CreationDate.ToUniversalTime().ToString('o')}}) | ConvertTo-Json -Compress"
  const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script],
    { encoding: 'utf8', windowsHide: true, timeout: 10_000, maxBuffer: 16 * 1024 * 1024 })
  const parsed = JSON.parse(stdout.trim() || '[]') as { ProcessId: number; ParentProcessId: number; Name: string; CreationDate: string } | { ProcessId: number; ParentProcessId: number; Name: string; CreationDate: string }[]
  const rows = Array.isArray(parsed) ? parsed : [parsed]
  const seen = new Set([root]), result: ProcessIdentity[] = []
  for (let pass = 0; pass < rows.length; pass++) {
    let added = false
    for (const row of rows) if (!seen.has(row.ProcessId) && seen.has(row.ParentProcessId)) {
      if (!Number.isSafeInteger(row.ProcessId) || !Number.isFinite(Date.parse(row.CreationDate))) throw new Error(`unverifiable descendant of PID ${root}: ${JSON.stringify(row)}`)
      seen.add(row.ProcessId)
      result.push({ pid: row.ProcessId, parentPid: row.ParentProcessId, name: row.Name, startedAt: row.CreationDate })
      added = true
    }
    if (!added) break
  }
  return result
}

export interface OwnedProcess {
  readonly root: ProcessIdentity
  readonly descendants: ReadonlyMap<number, ProcessIdentity>
  snapshot(): Promise<void>
  stop(): Promise<void>
}

/** Capture the root identity immediately and descendants while its ancestry is still verifiable. */
export async function trackOwnedProcess(p: ChildProcess, tree: ProcessTree = windowsProcessTree, scanIntervalMs = 500): Promise<OwnedProcess> {
  if (!p.pid) throw new Error('spawned process has no PID')
  const pid = p.pid
  let exitAt: number | null = null
  const closed = new Promise<void>((resolve) => { const done = () => { exitAt ??= Date.now(); resolve() }; p.once('close', done); p.once('error', done) })
  let rootSeenByOs = false
  try { process.kill(pid, 0); rootSeenByOs = true } catch { /* injected fake PID or inaccessible process; fail closed if final census cannot prove cleanup */ }
  const root = await tree.inspect?.(pid)
  if (!root || p.exitCode !== null || p.signalCode !== null) throw new Error(`spawned process PID ${pid} identity could not be verified`)
  if (!tree.killVerified || !tree.inspect) throw new Error('process tree lacks identity-verified inspection and kill')
  const descendants = new Map<number, ProcessIdentity>()
  let fault: Error | null = null
  let scanning: Promise<void> | null = null
  let stopping: Promise<void> | null = null
  const snapshot = async (requireRoot = false) => {
    if (scanning) return scanning
    scanning = (async () => {
      const liveRoot = await tree.inspect!(pid)
      if (!liveRoot) {
        if (requireRoot) throw new Error(`owned PID ${pid} exited before descendant identity capture`)
        return // after parent exit, only previously captured children are owned
      }
      if (!same(liveRoot, root)) throw new Error(`PID ${pid} identity changed; refusing descendant scan`)
      const found = await tree.descendants(pid)
      for (const child of found) {
        if (!Number.isSafeInteger(child.pid) || !Number.isFinite(Date.parse(child.startedAt))) throw new Error(`invalid descendant identity under PID ${pid}`)
        const known = descendants.get(child.pid)
        if (known && !same(known, child)) throw new Error(`descendant PID ${child.pid} identity changed`)
        descendants.set(child.pid, child)
      }
    })()
    try { await scanning } finally { scanning = null }
  }
  await snapshot(true)
  const timer = setInterval(() => { void snapshot().catch((e) => { fault = e instanceof Error ? e : new Error(String(e)) }) }, scanIntervalMs)
  const stop = () => {
    if (stopping) return stopping
    stopping = (async () => {
      clearInterval(timer)
      const errors: unknown[] = []
      try { await snapshot() } catch (e) { errors.push(e) }
      if (fault) errors.push(fault)
      let rootWasLive = false
      const childExitAt = new Map<number, number>() // captured pid → time WE confirmed it killed (kill-time bound)
      try {
        rootWasLive = same(await tree.inspect!(pid), root)
        const before = await reconcileOwnedProcessScan({ tree, root: pid, rootIdentity: root, children: descendants, rootWasLive, ownedExitAt: exitAt, childExitAt })
        await before.reap()
      } catch (e) { errors.push(e) }
      for (const child of [...descendants.values()].reverse()) {
        try { if (same(await tree.inspect!(child.pid), child) && await tree.killVerified!(child)) childExitAt.set(child.pid, Date.now()) }
        catch (e) { errors.push(e) }
      }
      let rootKilledDuringStop = false
      try {
        if (same(await tree.inspect!(pid), root)) { rootKilledDuringStop = true; await tree.killVerified!(root) }
      } catch (e) { errors.push(e); p.kill() } // handle kill is safe even when OS identity inspection fails
      const didClose = await new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => resolve(false), 10_000)
        void closed.then(() => { clearTimeout(timer); resolve(true) })
      })
      if (!didClose) errors.push(new Error(`owned process PID ${pid} did not close`))
      try {
        // The injected tree can miss an orphan after root exit. Production adds an independent OS census.
        if (!rootSeenByOs && rootKilledDuringStop && descendants.size === 0) throw new Error(`owned PID ${pid} final descendant scan unavailable after root exit`)
        const finalTree: ProcessTree = rootSeenByOs ? { ...tree, descendants: async (source) => {
          const [treeRows, osRows] = await Promise.all([tree.descendants(source), finalDescendantCensus(source)])
          // Rule iii: the two sources must agree on every PID they both report; last-wins would mask a conflict.
          const merged = new Map<number, ProcessIdentity>()
          for (const child of [...treeRows, ...osRows]) {
            const prior = merged.get(child.pid)
            if (prior && (prior.startedAt !== child.startedAt || prior.parentPid !== child.parentPid)) {
              throw new Error(`owned PID ${pid}: census sources disagree on descendant ${child.pid}`)
            }
            merged.set(child.pid, child)
          }
          return [...merged.values()]
        } } : tree
        const deadline = Date.now() + 3_000
        while (true) {
          const result = await reconcileOwnedProcessScan({ tree: finalTree, root: pid, rootIdentity: root, children: descendants, rootWasLive: false, ownedExitAt: exitAt, childExitAt })
          if (!result.remaining.length) break
          if (Date.now() >= deadline) throw new Error(`owned PID ${pid} left descendants alive: ${result.remaining.map((x) => x.pid).join(', ')}`)
          await result.reap()
        }
      } catch (e) { errors.push(e) }
      for (const record of [...descendants.values(), root]) {
        try { if (same(await tree.inspect!(record.pid), record)) errors.push(new Error(`owned PID ${record.pid} survived teardown`)) }
        catch (e) { errors.push(e) }
      }
      if (errors.length) throw new AggregateError(errors, `owned PID chain ${pid} teardown could not be verified: ${errors.map((e) => e instanceof Error ? e.message : String(e)).join('; ')}`)
    })()
    return stopping
  }
  return { root, descendants, snapshot, stop }
}
