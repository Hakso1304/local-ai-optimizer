import { spawn } from 'node:child_process'

// Out-of-process runner for model-written code. Why a child process and not worker_threads:
// verified on Node 24 that `new Array(1e9).fill(0)` and a string-doubling loop inside a Worker with
// resourceLimits hit V8's fatal "Reached heap limit" path and abort the WHOLE host process (exit 134).
// A child with --max-old-space-size dies alone. In Electron, process.execPath + ELECTRON_RUN_AS_NODE=1
// is plain Node (DESIGN §4). --permission denies fs/child_process/worker in the child as a second wall.

export interface SandboxOptions {
  timeoutMs: number
  memoryMb?: number // V8 old-space cap, default 256
}

export type SandboxOutcome = { ok: true; raw: string | null } | { ok: false; error: string }

const MARK = '\u0000QB-RESULT\u0000' // result follows the last marker; guards against stray stdout
const MAX_STDOUT = 64 * 1024

// Runs inside the child: read {script,timeoutMs,capBytes} JSON from stdin, run it in a fresh V8 context
// (same isolation as checkers.runHarness), write MARK + JSON outcome.
// Hardening (review-w4b F1): the host code is strict, so V8 CallSite.getFunction() never hands a host function to a
// model-installed Error.prepareStackTrace; Error is frozen in the context before the model runs; and a thrown value is
// never read through getters/toString — only own data properties of a native error, else a fixed diagnostic.
// F2: V8's old-space cap does not cover ArrayBuffer backing stores, so the child also checks its own RSS/arrayBuffers
// after the run (the parent polls RSS while it runs).
const CHILD = `'use strict';const vm=require('node:vm');const {isNativeError}=require('node:util').types;let s='';process.stdin.setEncoding('utf8');
const own=(e,k)=>{const d=Object.getOwnPropertyDescriptor(e,k);return d&&typeof d.value==='string'?d.value.slice(0,300):null};
const why=(e)=>typeof e==='string'?e.slice(0,300):typeof e!=='object'&&typeof e!=='function'?String(e):isNativeError(e)?(own(e,'message')??'error without a readable message'):'uncaught non-Error exception in model code';
process.stdin.on('data',d=>{s+=d}).on('end',()=>{const {script,timeoutMs,capBytes}=JSON.parse(s);let out;
try{const ctx=vm.createContext(vm.constants.DONT_CONTEXTIFY,{codeGeneration:{strings:false,wasm:false},microtaskMode:'afterEvaluate'});
vm.runInContext('Object.freeze(Error);Object.freeze(Error.prototype)',ctx);
const raw=vm.runInContext(script,ctx,{timeout:timeoutMs});out={ok:true,raw:typeof raw==='string'?raw:null}}
catch(e){out={ok:false,error:why(e)}}
const m=process.memoryUsage();if(m.arrayBuffers>capBytes||m.rss>capBytes*1.5+${48 * 1024 * 1024})out={ok:false,error:'memory limit exceeded (ArrayBuffer/RSS '+Math.round(Math.max(m.arrayBuffers,m.rss)/1048576)+' MiB)'};
process.stdout.write(${JSON.stringify(MARK)}+JSON.stringify(out))})`

/** Child working set in bytes via tasklist (no native Job Object API from Node); null if unreadable. */
function childRss(pid: number): Promise<number | null> {
  return new Promise((resolve) => {
    const t = spawn('tasklist', ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'], { windowsHide: true })
    let o = ''
    t.stdout.setEncoding('utf8').on('data', (d: string) => { o += d })
    t.on('error', () => resolve(null))
    t.on('close', () => {
      const kb = /"[^"]*","\d+","[^"]*","\d+","([\d.,\s]+)\s*K"/.exec(o)?.[1]?.replace(/[^\d]/g, '')
      resolve(kb ? Number(kb) * 1024 : null)
    })
  })
}

/** Launch policy of the sandbox child (exported so tests can probe the OS-level backstop under the exact same flags):
 *  --permission with no --allow-* grants denies fs, child_process and worker_threads. */
export const sandboxArgv = (memoryMb: number): string[] => ['--permission', `--max-old-space-size=${memoryMb}`, '--max-semi-space-size=16']
/** Minimal env: no NODE_OPTIONS (it could add --allow-* grants), only what Node needs on Windows. */
export const sandboxEnv = (): Record<string, string> => ({ ELECTRON_RUN_AS_NODE: '1', SYSTEMROOT: process.env.SYSTEMROOT ?? '' })

export function runSandboxed(script: string, opts: SandboxOptions): Promise<SandboxOutcome> {
  const memoryMb = opts.memoryMb ?? 256
  return new Promise((resolve) => {
    let settled = false
    const done = (o: SandboxOutcome) => { if (!settled) { settled = true; clearTimeout(hard); clearInterval(rssPoll); resolve(o) } }
    const capBytes = memoryMb * 1024 * 1024
    const child = spawn(process.execPath, [...sandboxArgv(memoryMb), '-e', CHILD],
      { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], env: sandboxEnv() })
    let out = ''
    let err = ''
    let killedFor: string | null = null
    const kill = (why: string) => { killedFor ??= why; child.kill() }
    // vm timeout should fire first; this catches anything that blocks outside the vm's clock.
    const hard = setTimeout(() => kill(`hard timeout: sandbox killed after ${opts.timeoutMs + 2000}ms`), opts.timeoutMs + 2000)
    // ponytail: RSS poll via tasklist every 250 ms (a slow allocation is caught mid-run; a fast one by the child's own
    // post-run check). Ceiling: a burst faster than 250 ms can briefly exceed the cap; a Job Object would need native code.
    let polling = false
    const rssPoll = setInterval(() => {
      if (polling || child.pid === undefined) return
      polling = true
      void childRss(child.pid).then((rss) => {
        polling = false
        if (rss != null && rss > capBytes * 1.5 + 48 * 1024 * 1024) kill(`memory limit exceeded (RSS ${Math.round(rss / 1048576)} MiB)`)
      })
    }, 250)
    child.stdout.setEncoding('utf8').on('data', (d: string) => {
      out += d
      if (out.length > MAX_STDOUT) kill('output limit exceeded')
    })
    child.stderr.setEncoding('utf8').on('data', (d: string) => { if (err.length < MAX_STDOUT) err += d })
    child.on('error', (e) => done({ ok: false, error: `sandbox failed to start: ${e.message}` }))
    child.on('close', (code) => {
      if (/heap out of memory|Reached heap limit/i.test(err) || code === 134) return done({ ok: false, error: `memory limit (${memoryMb} MiB) exceeded` })
      if (killedFor) return done({ ok: false, error: killedFor })
      const i = out.lastIndexOf(MARK)
      if (i < 0) return done({ ok: false, error: `sandbox exited with code ${code}: ${err.trim().split(/\r?\n/)[0] ?? ''}` })
      try {
        done(JSON.parse(out.slice(i + MARK.length)) as SandboxOutcome)
      } catch {
        done({ ok: false, error: 'sandbox result unreadable' })
      }
    })
    child.stdin.on('error', () => {}) // child may die before reading all input
    child.stdin.end(JSON.stringify({ script, timeoutMs: opts.timeoutMs, capBytes }))
  })
}
