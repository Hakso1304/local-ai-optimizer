import type { ChildProcess } from 'node:child_process'
import { windowsProcessTree, type ProcessIdentity, type ProcessTree } from '../src/core/runtimes/llamacpp'

const same = (a: ProcessIdentity | null, b: ProcessIdentity) => !!a && a.pid === b.pid && a.startedAt === b.startedAt

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
  const closed = new Promise<void>((resolve) => { p.once('close', () => resolve()); p.once('error', () => resolve()) })
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
      for (const child of [...descendants.values()].reverse()) {
        try { if (same(await tree.inspect!(child.pid), child)) await tree.killVerified!(child) }
        catch (e) { errors.push(e) }
      }
      try {
        if (same(await tree.inspect!(pid), root)) await tree.killVerified!(root)
      } catch (e) { errors.push(e); p.kill() } // handle kill is safe even when OS identity inspection fails
      const didClose = await new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => resolve(false), 10_000)
        void closed.then(() => { clearTimeout(timer); resolve(true) })
      })
      if (!didClose) errors.push(new Error(`owned process PID ${pid} did not close`))
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
