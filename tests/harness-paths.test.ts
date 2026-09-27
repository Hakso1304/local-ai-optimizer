import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { existingDbPath } from '../scripts/harness-paths'

describe('hardware harness database path preflight', () => {
  it('expands an environment variable without case sensitivity and checks the resolved path', () => {
    const seen: string[] = []
    const result = existingDbPath('%mixed_root%\\optimizer.db', true,
      { MiXeD_RoOt: 'E:\\fixture' }, (path) => { seen.push(path); return true })
    expect(result).toBe(resolve('E:\\fixture\\optimizer.db'))
    expect(seen).toEqual([result])
  })

  it('resolves a relative existing path and rejects a missing file before openDb can create it', () => {
    const name = 'missing-stage3-fixture.db'
    expect(existingDbPath(name, false, {}, (path) => path === resolve(name))).toBe(resolve(name))
    expect(() => existingDbPath(name, true, {}, () => false)).toThrow(/--db does not exist/)
  })

  it('rejects an undefined environment variable and a required path omitted by scenario H', () => {
    expect(() => existingDbPath('%NO_SUCH_DB_ROOT%\\optimizer.db', true, {}, () => true))
      .toThrow(/undefined environment variable %NO_SUCH_DB_ROOT%/)
    expect(() => existingDbPath(undefined, true, {}, () => true)).toThrow(/scenario H requires --db/)
    expect(existingDbPath(undefined, false, {}, () => true)).toBeNull()
  })
})

describe('run-session H rejects malformed CLI before hardware work', () => {
  const tsx = resolve('node_modules/tsx/dist/cli.mjs')
  const script = resolve('scripts/run-session.ts')
  it.each([
    { name: 'missing explicit limits', args: [], reason: /requires explicit --request-cap-ms and --ram-abort-gib/ },
    { name: 'zero request cap', args: ['--request-cap-ms', '0', '--ram-abort-gib', '4'], reason: /request-cap-ms must be/ },
    { name: 'nonfinite request cap', args: ['--request-cap-ms', 'NaN', '--ram-abort-gib', '4'], reason: /request-cap-ms must be/ },
    { name: 'excess request cap', args: ['--request-cap-ms', '300001', '--ram-abort-gib', '4'], reason: /request-cap-ms must be/ },
    { name: 'low RAM floor', args: ['--request-cap-ms', '300000', '--ram-abort-gib', '3.99'], reason: /ram-abort-gib must be/ },
    { name: 'missing database', args: ['--request-cap-ms', '300000', '--ram-abort-gib', '4'], reason: /scenario H requires --db/ },
    { name: 'unknown option', args: ['--not-a-real-option'], reason: /unknown option/ }
  ])('$name', ({ args, reason }) => {
    const result = spawnSync(process.execPath, [tsx, script, 'H', ...args],
      { cwd: resolve('.'), encoding: 'utf8', timeout: 15_000, windowsHide: true })
    expect(result.error).toBeUndefined()
    expect(result.status).not.toBe(0)
    expect(result.stderr + result.stdout).toMatch(reason)
  })
})
