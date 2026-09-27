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

// Runs inside the child: read {script,timeoutMs} JSON from stdin, run it in a fresh V8 context
// (same isolation as checkers.runHarness), write MARK + JSON outcome.
const CHILD = `const vm=require('node:vm');let s='';process.stdin.setEncoding('utf8');
process.stdin.on('data',d=>{s+=d}).on('end',()=>{const {script,timeoutMs}=JSON.parse(s);let out;
try{const ctx=vm.createContext(vm.constants.DONT_CONTEXTIFY,{codeGeneration:{strings:false,wasm:false},microtaskMode:'afterEvaluate'});
const raw=vm.runInContext(script,ctx,{timeout:timeoutMs});out={ok:true,raw:typeof raw==='string'?raw:null}}
catch(e){out={ok:false,error:String((e&&e.message)||e)}}
process.stdout.write(${JSON.stringify(MARK)}+JSON.stringify(out))})`

export function runSandboxed(script: string, opts: SandboxOptions): Promise<SandboxOutcome> {
  const memoryMb = opts.memoryMb ?? 256
  return new Promise((resolve) => {
    let settled = false
    const done = (o: SandboxOutcome) => { if (!settled) { settled = true; clearTimeout(hard); resolve(o) } }
    const child = spawn(
      process.execPath,
      ['--permission', `--max-old-space-size=${memoryMb}`, '--max-semi-space-size=16', '-e', CHILD],
      { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], env: { ELECTRON_RUN_AS_NODE: '1', SYSTEMROOT: process.env.SYSTEMROOT ?? '' } }
    )
    let out = ''
    let err = ''
    let killedFor: string | null = null
    const kill = (why: string) => { killedFor ??= why; child.kill() }
    // vm timeout should fire first; this catches anything that blocks outside the vm's clock.
    const hard = setTimeout(() => kill(`hard timeout: sandbox killed after ${opts.timeoutMs + 2000}ms`), opts.timeoutMs + 2000)
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
    child.stdin.end(JSON.stringify({ script, timeoutMs: opts.timeoutMs }))
  })
}
