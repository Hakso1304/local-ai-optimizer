// review-w4c T03: the real typeperf lifecycle (spawn → rows → restart → stop → no orphan), not a mock.
// Skipped where typeperf is unavailable (non-Windows CI). Uses a dummy node child as the sampled pid.
import { spawn, spawnSync, execFileSync } from 'node:child_process'
import { describe, expect, it } from 'vitest'
import { startSampler } from '../src/core/telemetry/sampler'

const hasTypeperf = process.platform === 'win32' && !spawnSync('typeperf', ['-?'], { windowsHide: true }).error
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const until = async (f: () => boolean, ms: number) => { const t0 = Date.now(); while (!f() && Date.now() - t0 < ms) await sleep(100); return f() }

/** typeperf processes whose command line samples this pid (other sessions' samplers are not counted). */
function typeperfFor(pid: number): number {
  const out = execFileSync('powershell.exe', ['-NoProfile', '-Command',
    `@(Get-CimInstance Win32_Process -Filter "name='typeperf.exe'" | ? { $_.CommandLine -match 'pid_${pid}_' }).Count`], { encoding: 'utf8', windowsHide: true })
  return Number(out.trim())
}

describe.skipIf(!hasTypeperf)('startSampler (live typeperf)', () => {
  it('produces rows, restarts without leaking the old child, and stops with no orphan', async () => {
    const dummy = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { windowsHide: true })
    try {
      const pid = dummy.pid!
      const s = startSampler({ pid, procName: 'node' })
      expect(await until(() => s.samples.length >= 1, 8000), `no rows; errors: ${s.errors.join(' | ')}`).toBe(true)
      const first = s.samples[0]
      expect(typeof first.ramAvailBytes).toBe('number')
      expect(first.ramAvailBytes!).toBeGreaterThan(0)
      expect(s.hasPidColumns).not.toBeNull()

      const n = s.samples.length
      s.restart()
      expect(await until(() => s.samples.length > n, 8000), 'no rows after restart').toBe(true)
      expect(await until(() => typeperfFor(pid) === 1, 3000), 'old typeperf not killed by restart()').toBe(true)

      s.stop()
      expect(await until(() => typeperfFor(pid) === 0, 5000), 'typeperf orphaned after stop()').toBe(true)
      expect(s.errors.some((e) => /exited early|failed to start/.test(e)), s.errors.join(' | ')).toBe(false)
    } finally {
      dummy.kill()
    }
  }, 40_000)
})
