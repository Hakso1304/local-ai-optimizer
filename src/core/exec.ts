import { execFile } from 'node:child_process'

export interface ExecResult {
  stdout: string
  stderr: string
}

/** Run a process with a hard timeout. Rejects with a readable message on failure/timeout/non-zero exit. */
export function runProcess(file: string, args: string[], timeoutMs: number): Promise<ExecResult> {
  return new Promise((resolve, reject) => {
    execFile(
      file,
      args,
      { timeout: timeoutMs, windowsHide: true, maxBuffer: 16 * 1024 * 1024, encoding: 'utf8' },
      (err, stdout, stderr) => {
        if (!err) return resolve({ stdout, stderr })
        const e = err as NodeJS.ErrnoException & { killed?: boolean; signal?: string }
        const why = (e.code as string | undefined) === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER'
          ? 'output exceeded 16 MiB'
          : e.killed
          ? `timed out after ${timeoutMs}ms`
          : e.code === 'ENOENT'
            ? 'executable not found'
            : `exit code ${String(e.code)}`
        const detail = (stderr || stdout || e.message).trim().split(/\r?\n/).slice(0, 3).join(' | ')
        reject(new Error(`${file} ${why}${detail ? `: ${detail}` : ''}`))
      }
    )
  })
}

/** Run a PowerShell script (5.1 compatible). Output forced to UTF-8 so localized errors survive. */
export async function runPowerShell(script: string, timeoutMs = 20_000): Promise<string> {
  const full = `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n$ProgressPreference = 'SilentlyContinue'\n${script}`
  const encoded = Buffer.from(full, 'utf16le').toString('base64')
  const { stdout } = await runProcess(
    'powershell.exe',
    ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded],
    timeoutMs
  )
  return stdout
}
