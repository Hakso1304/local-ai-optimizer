// Writes dist/BUILD-INFO.txt after `npm run package`, so every build says what it is: commit (and whether the tree
// was dirty), time, app + rules version, test results, and whether the HIP backend has been verified on hardware.
// Tests are run and recorded, not gating: a red suite is written into the file, never hidden.
// Every skipped test is listed by name (a skip is accepted by the release gate only when it is named here).
// Env: LAO_BUILD_LABEL (e.g. "nightly 2026-09-28 pre-verdict"), LAO_HIP_VERIFIED=1 once a HIP calibration passed,
// LAO_BUILD_ROOT (stamp another checkout, e.g. the clean release worktree, with this script).
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const root = process.env.LAO_BUILD_ROOT ?? join(import.meta.dirname, '..')
const run = (cmd, args) => { try { return execFileSync(cmd, args, { cwd: root, encoding: 'utf8', windowsHide: true }).trim() } catch { return null } }
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))
const rules = JSON.parse(readFileSync(join(root, 'src/core/interpret/rules.v2.json'), 'utf8')).version

const commit = run('git', ['rev-parse', '--short', 'HEAD']) ?? 'unknown'
const dirty = (run('git', ['status', '--porcelain', '--untracked-files=no']) ?? '').length > 0
// origin/master at build time (local ref) and what separates it from the built commit
const origin = run('git', ['rev-parse', '--short', 'origin/master'])
let originLine = origin ?? 'unknown'
if (origin && origin !== commit) {
  const files = (run('git', ['diff', '--name-only', commit, origin]) ?? '').split(/\r?\n/).filter(Boolean)
  const ahead = run('git', ['merge-base', '--is-ancestor', commit, origin]) !== null
  const docsOnly = files.length > 0 && files.every((f) => f.startsWith('docs/'))
  originLine = `${origin} (${ahead ? `${files.length} file(s) ahead, ${docsOnly ? 'docs-only delta' : 'CODE delta'}` : 'not a descendant of the built commit'})`
}

let tests = 'not run'
let skipped = []
const tmp = mkdtempSync(join(tmpdir(), 'lao-bi-'))
try {
  const out = join(tmp, 'vitest.json')
  run(process.execPath, [join(root, 'node_modules', 'vitest', 'vitest.mjs'), 'run', '--reporter=json', `--outputFile=${out}`]) // exit≠0 when red; the JSON still lands
  const r = JSON.parse(readFileSync(out, 'utf8'))
  tests = `${r.numPassedTests}/${r.numTotalTests} passed${r.numFailedTests ? `, ${r.numFailedTests} FAILED` : ''}${r.numPendingTests ? `, ${r.numPendingTests} skipped` : ''}`
  skipped = r.testResults.flatMap((f) => f.assertionResults.filter((a) => a.status === 'pending' || a.status === 'skipped')
    .map((a) => `${f.name.replace(/\\/g, '/').split('/tests/').pop()} :: ${a.fullName}`))
} catch (e) {
  tests = `could not run (${e.message.split('\n')[0]})`
} finally {
  rmSync(tmp, { recursive: true, force: true })
}

const lines = [
  `${pkg.build?.productName ?? pkg.name} ${pkg.version}${process.env.LAO_BUILD_LABEL ? ` — ${process.env.LAO_BUILD_LABEL}` : ''}`,
  `commit:  ${commit}${dirty ? ' (DIRTY working tree — not a clean build)' : ''}`,
  `origin:  ${originLine}`,
  `built:   ${new Date().toISOString()}`,
  `rules:   ${rules}`,
  `tests:   ${tests}`,
  ...skipped.map((s) => `skipped: ${s}`),
  `HIP:     ${process.env.LAO_HIP_VERIFIED === '1' ? 'verified on hardware' : 'ROCm/HIP backend untested on real hardware'}`
]
// LAO_BUILD_INFO_OUT: write elsewhere (dry run, or stamping a build copied out of the release worktree)
writeFileSync(process.env.LAO_BUILD_INFO_OUT ?? join(root, 'dist', 'BUILD-INFO.txt'), lines.join('\n') + '\n')
console.log(lines.join('\n'))
