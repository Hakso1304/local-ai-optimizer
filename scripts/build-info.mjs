// Writes dist/BUILD-INFO.txt after `npm run package`, so every build says what it is: commit (and whether the tree
// was dirty), time, app + rules version, test results, and whether the HIP backend has been verified on hardware.
// Tests are run and recorded, not gating: a red suite is written into the file, never hidden.
// Env: LAO_BUILD_LABEL (e.g. "nightly 2026-09-28 pre-verdict"), LAO_HIP_VERIFIED=1 once a HIP calibration passed.
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const root = join(import.meta.dirname, '..')
const run = (cmd, args) => { try { return execFileSync(cmd, args, { cwd: root, encoding: 'utf8', windowsHide: true }).trim() } catch { return null } }
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))
const rules = JSON.parse(readFileSync(join(root, 'src/core/interpret/rules.v2.json'), 'utf8')).version

const commit = run('git', ['rev-parse', '--short', 'HEAD']) ?? 'unknown'
const dirty = (run('git', ['status', '--porcelain', '--untracked-files=no']) ?? '').length > 0

let tests = 'not run'
const tmp = mkdtempSync(join(tmpdir(), 'lao-bi-'))
try {
  const out = join(tmp, 'vitest.json')
  run(process.execPath, [join(root, 'node_modules', 'vitest', 'vitest.mjs'), 'run', '--reporter=json', `--outputFile=${out}`]) // exit≠0 when red; the JSON still lands
  const r = JSON.parse(readFileSync(out, 'utf8'))
  tests = `${r.numPassedTests}/${r.numTotalTests} passed${r.numFailedTests ? `, ${r.numFailedTests} FAILED` : ''}${r.numPendingTests ? `, ${r.numPendingTests} skipped` : ''}`
} catch (e) {
  tests = `could not run (${e.message.split('\n')[0]})`
} finally {
  rmSync(tmp, { recursive: true, force: true })
}

const lines = [
  `${pkg.build?.productName ?? pkg.name} ${pkg.version}${process.env.LAO_BUILD_LABEL ? ` — ${process.env.LAO_BUILD_LABEL}` : ''}`,
  `commit:  ${commit}${dirty ? ' (DIRTY working tree — not a clean build)' : ''}`,
  `built:   ${new Date().toISOString()}`,
  `rules:   ${rules}`,
  `tests:   ${tests}`,
  `HIP:     ${process.env.LAO_HIP_VERIFIED === '1' ? 'verified on hardware' : 'ROCm/HIP backend untested on real hardware'}`
]
writeFileSync(join(root, 'dist', 'BUILD-INFO.txt'), lines.join('\n') + '\n')
console.log(lines.join('\n'))
