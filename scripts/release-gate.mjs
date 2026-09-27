// Automates the checkable boxes of docs/RELEASE-GATE.md. Prints PASS/STOP per item with the evidence it read;
// exit 1 on any STOP. Default mode is lease-safe (no GPU, no Electron, no full suite): typecheck, a few pure test
// files, tree/remote state, docs. --full (idle GPU lane only) also runs the full suite, build and package and checks
// dist/BUILD-INFO.txt. Boxes that need a human/reviewer judgement (Astra's verdict, STATUS currency) stay manual.
// Usage: node scripts/release-gate.mjs [--full]
import { execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const root = join(import.meta.dirname, '..')
const full = process.argv.includes('--full')
const results = []
const item = (name, ok, evidence) => { results.push({ name, ok }); console.log(`${ok ? 'PASS' : 'STOP'}  ${name}\n      ${evidence}`) }
/** Run a command; never throws. windowsHide: no console window from a windowless parent. */
const run = (cmd, args, env = {}) => {
  try {
    return { ok: true, out: execFileSync(cmd, args, { cwd: root, encoding: 'utf8', windowsHide: true, maxBuffer: 64 * 1024 * 1024, env: { ...process.env, ...env }, stdio: ['ignore', 'pipe', 'pipe'] }) }
  } catch (e) {
    return { ok: false, out: `${e.stdout ?? ''}${e.stderr ?? ''}` || e.message }
  }
}
const node = (script, args, env) => run(process.execPath, [join(root, ...script.split('/')), ...args], env)
const npm = (script, env) => run(process.env.ComSpec ?? 'cmd.exe', ['/d', '/s', '/c', `npm run ${script}`], env) // npm is a .cmd on Windows
const tail = (s, n = 3) => s.trim().split(/\r?\n/).filter(Boolean).slice(-n).join(' | ').slice(0, 300)
const read = (p) => (existsSync(join(root, p)) ? readFileSync(join(root, p), 'utf8') : null)
const git = (...a) => run('git', a)
const rules = JSON.parse(read('src/core/interpret/rules.v2.json')).version
const vitest = (files, extra = []) => node('node_modules/vitest/vitest.mjs', ['run', ...files, '--maxWorkers=1', ...extra])
const testLine = (out) => /Tests\s+([^\n]+)/.exec(out)?.[1]?.trim() ?? tail(out)

// ---- Code and tests (lease-safe) ----
const tsc = node('node_modules/typescript/bin/tsc', ['--noEmit', '-p', 'tsconfig.json'])
item('typecheck: tsc --noEmit -p tsconfig.json exits 0', tsc.ok, tsc.ok ? 'exit 0' : tail(tsc.out, 5))

const hide = vitest(['tests/windows-hide.test.ts'])
item('windows-hide scan: every spawn/exec in src/ and scripts/ passes windowsHide: true', hide.ok, `tests/windows-hide.test.ts: ${testLine(hide.out)}`)

const g08 = vitest(['tests/scoring/interp2-rereview.test.ts'], ['-t', 'lower effort is kept whatever the input order'])
item('G08 fixed: interp2-rereview "equal quality: the lower effort is kept whatever the input order"', g08.ok, testLine(g08.out))

// ---- Tree ----
const porcelain = git('status', '--porcelain')
const dirty = porcelain.out.trim().split(/\r?\n/).filter(Boolean)
item('tree: git status --porcelain is empty', porcelain.ok && dirty.length === 0, dirty.length ? `${dirty.length} entries, e.g. ${dirty.slice(0, 4).join('; ')}` : 'clean')

// origin/master == HEAD, or origin is a descendant whose delta is docs-only (recorded in BUILD-INFO "origin:")
const head = git('rev-parse', 'HEAD').out.trim(), remote = git('rev-parse', 'origin/master')
const rhead = remote.ok ? remote.out.trim() : ''
const delta = rhead && rhead !== head ? git('diff', '--name-only', head, rhead).out.trim().split(/\r?\n/).filter(Boolean) : []
const descendant = rhead && rhead !== head ? git('merge-base', '--is-ancestor', head, rhead).ok : true
const docsOnly = delta.every((f) => f.startsWith('docs/'))
item('pushed: origin/master == HEAD, or ahead by a docs-only delta', remote.ok && descendant && docsOnly,
  `HEAD ${head.slice(0, 7)}, origin/master ${rhead.slice(0, 7) || 'unknown'}${delta.length ? ` (${delta.length} file(s) ahead${docsOnly ? ', docs-only' : `, CODE: ${delta.filter((f) => !f.startsWith('docs/')).slice(0, 3).join(', ')}`})` : ''}`)

// ---- Docs and evidence ----
for (const doc of ['docs/STATUS.md', 'docs/EVIDENCE.md']) {
  const t = read(doc)
  item(`${doc} present and names the current rules version (${rules})`, !!t && t.includes(rules), t ? (t.includes(rules) ? `mentions ${rules}` : `no "${rules}" in ${doc}`) : 'missing')
}
const ev = read('docs/EVIDENCE.md') ?? ''
const e12 = /E-12[^\n]*contradict/i.test(ev)
item('EVIDENCE: E-12 marked contradicted', e12, e12 ? 'E-12 row says contradicted' : 'no "contradicted" on an E-12 line')

const lim = read('docs/LIMITATIONS.md') ?? ''
const caveats = [['spill/placement heuristic (I-2.8)', /I-2\.8[^\n]*(heuristic|not calibrated|uncalibrated)|(heuristic|not calibrated)[^\n]*I-2\.8/i],
  ['learned budget advisory until qualified', /advisory[^\n]*qualif/i], ['HIP unverified on hardware', /HIP[^\n]*(untested|unverified)/i]]
const missing = caveats.filter(([, re]) => !re.test(lim)).map(([n]) => n)
item('LIMITATIONS lists the three known caveats', missing.length === 0, missing.length ? `missing: ${missing.join('; ')}` : caveats.map(([n]) => n).join('; '))

// HIP: LAO_HIP_VERIFIED=1 only with stage 4 (ROCm0 listed) and stage 5 (Vulkan-vs-HIP A/B) passed in the overnight log.
const cal = read('docs/calibration-overnight-2026-09-28.md') ?? ''
const stagePassed = (n) => cal.split(/\r?\n/).some((l) => new RegExp(`stage\\s*${n}\\b`, 'i').test(l) && /\b(pass(ed)?|done|complete)\b/i.test(l) && !/\b(fail|blocked|pending|not run)\b/i.test(l))
const s4 = stagePassed(4), s5 = stagePassed(5), hipSet = process.env.LAO_HIP_VERIFIED === '1'
item('HIP flag consistent with calibration (LAO_HIP_VERIFIED=1 only after stages 4 and 5 passed)', hipSet ? s4 && s5 : true,
  `LAO_HIP_VERIFIED=${hipSet ? '1' : 'unset'}; calibration-overnight stage 4 ${s4 ? 'passed' : 'not passed'}, stage 5 ${s5 ? 'passed' : 'not passed'}${hipSet ? '' : ' → BUILD-INFO will say HIP untested (consistent)'}`)

// ---- Idle-lane only ----
if (full) {
  const t = node('node_modules/vitest/vitest.mjs', ['run'])
  item('full npm test passes with 0 failures', t.ok, testLine(t.out))
  const b = npm('build')
  item('npm run build succeeds', b.ok, b.ok ? 'ok' : tail(b.out, 5))
  const p = b.ok ? npm('package') : { ok: false, out: 'skipped (build failed)' }
  item('npm run package succeeds', p.ok, p.ok ? 'ok' : tail(p.out, 5))
  const bi = read('dist/BUILD-INFO.txt') ?? ''
  const commitOk = bi.includes(`commit:  ${head.slice(0, 7)}`) && !/DIRTY/.test(bi)
  // "P/T passed[, K skipped]" with P + K = T, no FAILED, and every skipped test named on a "skipped:" line
  const tm = /^tests:\s+(\d+)\/(\d+) passed(?:, (\d+) skipped)?\s*$/m.exec(bi)
  const named = (bi.match(/^skipped: \S.*$/gm) ?? []).length
  const testsOk = !!tm && !/FAILED/.test(bi) && Number(tm[1]) + Number(tm[3] ?? 0) === Number(tm[2]) && named === Number(tm[3] ?? 0)
  item('BUILD-INFO: packaged commit, not DIRTY, rules version, 0 failed, every skip named', p.ok && commitOk && bi.includes(`rules:   ${rules}`) && testsOk,
    bi ? bi.trim().split(/\r?\n/).join(' | ') : 'dist/BUILD-INFO.txt missing')
} else {
  console.log('\n(lease-safe mode: full suite, build, package and BUILD-INFO checks run only with --full on an idle GPU lane)')
}
console.log('\nManual boxes (reviewer judgement): Astra RECHECK verdict, RECHECK4 O1–O5 follow-ups, STATUS rows current, README/ACCEPTANCE claims.')

const stops = results.filter((r) => !r.ok).length
console.log(`\n${stops ? `STOP — ${stops} of ${results.length} items` : `PASS — all ${results.length} items`}`)
process.exit(stops ? 1 : 0)
