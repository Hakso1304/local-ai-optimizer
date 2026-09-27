import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { runProcess } from '../../src/core/exec'

const node = process.execPath
const dir = mkdtempSync(join(tmpdir(), 'lao-exec-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))

const isAlive = (pid: number) => {
  try { process.kill(pid, 0); return true } catch { return false }
}

describe('runProcess', () => {
  it('resolves stdout/stderr on success', async () => {
    expect(await runProcess(node, ['-e', 'process.stdout.write("hi");process.stderr.write("err")'], 10_000)).toEqual({ stdout: 'hi', stderr: 'err' })
  })

  it('timeout rejects AND the child is actually dead', async () => {
    const pidPath = join(dir, 'pid')
    const script = `require("fs").writeFileSync(${JSON.stringify(pidPath)}, String(process.pid)); setInterval(() => {}, 1000)`
    await expect(runProcess(node, ['-e', script], 1500)).rejects.toThrow(/timed out after 1500ms/)
    expect(existsSync(pidPath)).toBe(true)
    const pid = Number(readFileSync(pidPath, 'utf8'))
    await new Promise((r) => setTimeout(r, 200))
    expect(isAlive(pid)).toBe(false)
  })

  it('maxBuffer overflow is reported distinctly, not as a timeout', async () => {
    const err = await runProcess(node, ['-e', 'process.stdout.write("x".repeat(17 * 1024 * 1024))'], 30_000).catch((e: Error) => e)
    expect(err).toBeInstanceOf(Error)
    expect((err as Error).message).toMatch(/output exceeded 16 MiB/)
    expect((err as Error).message).not.toMatch(/timed out/)
  })

  it('missing executable and non-zero exit are distinguishable', async () => {
    await expect(runProcess(join(dir, 'nope.exe'), [], 5000)).rejects.toThrow(/executable not found/)
    await expect(runProcess(node, ['-e', 'console.error("bad thing");process.exit(3)'], 10_000)).rejects.toThrow(/exit code 3: bad thing/)
  })
})
