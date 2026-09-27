import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'

const query = vi.hoisted(() => ({ stdout: '', stderr: '', fail: false }))
vi.mock('node:child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:child_process')>()
  const { promisify } = await import('node:util')
  const fake = (() => { throw new Error('callback form is not used in this probe') }) as unknown as typeof actual.execFile
  Object.assign(fake, { [promisify.custom]: async () => {
    if (query.fail) throw new Error('Get-NetTCPConnection failed')
    return { stdout: query.stdout, stderr: query.stderr }
  } })
  return { ...actual, execFile: fake }
})

import { launch, reserveLoopbackPort } from '../scripts/ab-spill'

const GiB = 1024 ** 3

describe('A/B port-owner OS query fails closed before spawn (Node fake only)', () => {
  it.each([
    ['empty output', '', false],
    ['two listener PIDs', '1234\n5678', false],
    ['failed Get-NetTCPConnection', '', true]
  ] as const)('%s', async (_name, stdout, fail) => {
    query.stdout = stdout; query.stderr = ''; query.fail = fail
    const dir = mkdtempSync(join(tmpdir(), 'lao-ab-owner-'))
    const pidFile = join(dir, 'pid.txt'), previous = process.env.FAKE_AB_PID_FILE
    process.env.FAKE_AB_PID_FILE = pidFile
    try {
      const row = await launch('owner query', [join(__dirname, 'fixtures', 'fake-ab-spill-server.cjs'),
        '-m', 'fake.gguf', '-c', '2048', '--fake-mode', 'streams'], null, process.execPath,
      { readRam: () => 5 * GiB, countServers: () => 0, devices: () => [],
        startSampler: () => ({ rows: [], stop: async () => {} }),
        reservePort: () => reserveLoopbackPort(), settleMs: 0, postMs: 0, loadTimeoutMs: 500, healthTimeoutMs: 50 })
      expect(existsSync(pidFile)).toBe(false) // refusal precedes even the fake Node spawn
      expect(row.error).toMatch(/owner|listener|query|ambiguous|unavailable|Get-NetTCPConnection/i)
      expect(row.reps).toEqual([])
    } finally {
      if (previous === undefined) delete process.env.FAKE_AB_PID_FILE
      else process.env.FAKE_AB_PID_FILE = previous
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
