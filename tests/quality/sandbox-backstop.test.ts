// review-w4c T04: the OS-level backstop (--permission) must hold on its own, independent of VM isolation.
// A TRUSTED probe (not model code) runs in a child launched with the sandbox's exact argv/env and tries every
// capability the backstop is meant to deny. Removing --permission from sandboxArgv makes this test fail.
import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { sandboxArgv, sandboxEnv } from '../../src/core/quality/sandbox'

const dir = mkdtempSync(join(tmpdir(), 'lao-backstop-'))
const secret = join(dir, 'secret.txt')
writeFileSync(secret, 'top secret')
afterAll(() => rmSync(dir, { recursive: true, force: true }))

const PROBE = `
const out = {};
const t = (k, f) => { try { f(); out[k] = 'ALLOWED' } catch (e) { out[k] = String(e && e.code || e && e.message || e) } };
t('fsRead', () => require('node:fs').readFileSync(${JSON.stringify(secret)}, 'utf8'));
t('fsWrite', () => require('node:fs').writeFileSync(${JSON.stringify(join(dir, 'pwned.txt'))}, 'x'));
t('childSpawn', () => require('node:child_process').spawnSync(process.execPath, ['-e', '1']));
t('worker', () => new (require('node:worker_threads').Worker)('1', { eval: true }));
t('permissionApi', () => { if (process.permission.has('fs.read') || process.permission.has('child')) throw 0; else throw Object.assign(new Error(), { code: 'DENIED_BY_POLICY' }) });
process.stdout.write(JSON.stringify(out));
`

describe('sandbox launch policy', () => {
  it('argv enables --permission and grants nothing back', () => {
    const argv = sandboxArgv(256)
    expect(argv).toContain('--permission')
    expect(argv.some((a) => a.startsWith('--allow-'))).toBe(false)
    expect(sandboxEnv()).not.toHaveProperty('NODE_OPTIONS')
    expect(Object.keys(sandboxEnv()).sort()).toEqual(['ELECTRON_RUN_AS_NODE', 'SYSTEMROOT'])
  })

  it('a trusted probe under the exact policy is denied fs read, fs write, child spawn and workers', () => {
    const r = spawnSync(process.execPath, [...sandboxArgv(64), '-e', PROBE], { env: sandboxEnv(), encoding: 'utf8', timeout: 20_000, windowsHide: true })
    expect(r.status, r.stderr).toBe(0)
    const out = JSON.parse(r.stdout) as Record<string, string>
    expect(out).toEqual({
      fsRead: 'ERR_ACCESS_DENIED', fsWrite: 'ERR_ACCESS_DENIED', childSpawn: 'ERR_ACCESS_DENIED', worker: 'ERR_ACCESS_DENIED',
      permissionApi: 'DENIED_BY_POLICY'
    })
  })

  it('control: the same probe WITHOUT --permission is allowed (proves the probe detects the difference)', () => {
    const r = spawnSync(process.execPath, ['-e', PROBE], { env: sandboxEnv(), encoding: 'utf8', timeout: 20_000, windowsHide: true })
    const out = JSON.parse(r.stdout) as Record<string, string>
    expect(out.fsRead).toBe('ALLOWED')
    expect(out.childSpawn).toBe('ALLOWED')
  })
})
