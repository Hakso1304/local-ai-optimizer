// Every child process must be spawned with windowsHide: true. A windowless parent (the packaged app, or a harness
// started from an agent's shell) otherwise pops a console window for each console program (tasklist, typeperf, …).
// Source scan: each child_process call (and the backend's spawnFn seam) must pass windowsHide: true in its arguments.
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const ROOT = join(__dirname, '..')
const files = (dir: string): string[] => readdirSync(join(ROOT, dir), { recursive: true, encoding: 'utf8' })
  .filter((f) => /\.(ts|tsx|cjs|mjs|js)$/.test(f)).map((f) => join(dir, f))

/** The argument text of the call starting at `open` (index of its "("), parens balanced. */
function argsAt(src: string, open: number): string {
  let depth = 0
  for (let i = open; i < src.length; i++) {
    if (src[i] === '(') depth++
    else if (src[i] === ')' && --depth === 0) return src.slice(open, i + 1)
  }
  return src.slice(open)
}

describe('windows: no console windows for child processes', () => {
  const offenders: string[] = []
  let calls = 0
  for (const f of [...files('src'), ...files('scripts')]) {
    const src = readFileSync(join(ROOT, f), 'utf8')
    // only the names imported from child_process (a local helper called `exec` wraps a hidden execFile)
    const imported = /import\s*\{([^}]*)\}\s*from\s*'node:child_process'/.exec(src)?.[1]
      .split(',').map((x) => x.trim()).filter((x) => /^(spawn|spawnSync|execFile|execFileSync|exec|execSync|fork)$/.test(x)) ?? []
    if (!imported.length) continue
    for (const m of src.matchAll(new RegExp(`(?<![.\\w])(${imported.join('|')})\\(|this\\.spawnFn\\(`, 'g'))) {
      calls++
      const a = argsAt(src, m.index! + m[0].length - 1)
      if (!/windowsHide:\s*true/.test(a)) offenders.push(`${f}:${src.slice(0, m.index).split('\n').length} ${m[0]}`)
    }
  }

  it('every spawn/exec call in src/ and scripts/ passes windowsHide: true', () => {
    expect(calls).toBeGreaterThan(10) // the scan found the calls at all
    expect(offenders).toEqual([])
  })
})
