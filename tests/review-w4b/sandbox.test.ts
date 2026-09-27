import { afterEach, expect, it, vi } from 'vitest'
import { runSandboxed } from '../../src/core/quality/sandbox'

afterEach(() => vi.unstubAllEnvs())

it('blocks prototype and async-function constructor access to the child process', async () => {
  const r = await runSandboxed(`
    Object.prototype.reviewMarker = 'local';
    const probes = [
      () => this.constructor.constructor('return process')(),
      () => (async()=>{}).constructor('return process')(),
      () => Object.getPrototypeOf({}).constructor.constructor('return process')()
    ];
    JSON.stringify(probes.map(f=>{try{return !!f()}catch{return false}}))
  `, { timeoutMs: 100 })
  expect(r).toEqual({ ok: true, raw: '[false,false,false]' })
  expect(({} as any).reviewMarker).toBeUndefined()
})

it('does not expose host constructors through an error getter and V8 stack callsites', async () => {
  const r = await runSandboxed(`throw { get message() {
    Error.prepareStackTrace = (_e, frames) => frames.map(f => {
      try { return f.getFunction()?.constructor('return process')()?.versions?.node || 'blocked' }
      catch { return 'blocked' }
    }).join('|');
    return new Error().stack;
  } }`, { timeoutMs: 100 })
  expect(r.ok).toBe(false)
  if (!r.ok) expect(r.error).not.toMatch(/\d+\.\d+\.\d+/)
})

it('does not let escaped code forge a successful sandbox result', async () => {
  const r = await runSandboxed(`throw { get message() {
    Error.prepareStackTrace = (_e, frames) => {
      for (const f of frames) {
        try {
          const p = f.getFunction()?.constructor('return process')();
          if (!p) continue;
          p.stdout.write('\\u0000QB-RESULT\\u0000' + JSON.stringify({ok:true,raw:'forged'}));
          p.exit(0);
        } catch {}
      }
      return 'blocked';
    };
    return new Error().stack;
  } }`, { timeoutMs: 100 })
  expect(r.ok).toBe(false)
})

it('keeps the child filesystem and subprocess permission barriers after the VM escape', async () => {
  const r = await runSandboxed(`throw { get message() {
    Error.prepareStackTrace = (_e, frames) => {
      for (const f of frames) {
        try {
          const p = f.getFunction()?.constructor('return process')();
          if (p) return JSON.stringify([p.permission.has('fs.read'),p.permission.has('fs.write'),p.permission.has('child')]);
        } catch {}
      }
      return 'escape not available';
    };
    return new Error().stack;
  } }`, { timeoutMs: 100 })
  expect(r.ok).toBe(false)
  if (!r.ok) expect(r.error).not.toContain('true')
})

it('enforces the advertised memory cap on ArrayBuffer backing stores', async () => {
  // Bounded reproduction: touches only 64 MiB in a child configured for 16 MiB old space.
  const r = await runSandboxed('const b = new Uint8Array(64 * 1024 * 1024); b.fill(1); "allocated"', { timeoutMs: 500, memoryMb: 16 })
  expect(r.ok).toBe(false)
})

it('interrupts Atomics.wait inside the VM', async () => {
  const r = await runSandboxed('Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0)', { timeoutMs: 100 })
  expect(r.ok).toBe(false)
  if (!r.ok) expect(r.error).toMatch(/timed out|timeout/i)
})

it('scrubs NODE_OPTIONS instead of letting the child inherit permission-changing options', async () => {
  vi.stubEnv('NODE_OPTIONS', '--review-invalid-option')
  expect(await runSandboxed('"clean"', { timeoutMs: 100 })).toEqual({ ok: true, raw: 'clean' })
})
