import { describe, expect, it } from 'vitest'
import { jsCodeAsync } from '../../src/core/quality/checkers'
import { defaultTestSet, evaluateAsync } from '../../src/core/quality'
import { runSandboxed } from '../../src/core/quality/sandbox'

const add = [{ expr: 'add(2,3)', expected: 5 }]

describe('child-process sandbox (jsCodeAsync)', () => {
  it('passes correct code and scores wrong code', async () => {
    expect(await jsCodeAsync('function add(a,b){return a+b}', add)).toMatchObject({ pass: true, score: 1 })
    expect(await jsCodeAsync('function add(a,b){return a-b}', add)).toMatchObject({ pass: false, score: 0 })
  })

  it.each([
    ['array fill bomb', 'new Array(1e9).fill(0)'],
    ['string doubling bomb', 'let s="ab"; while(true){ s = (s + s).split("").join("") }'],
    ['push bomb', 'const a=[]; while(true) a.push(new Array(1e6).fill(1))']
  ])('%s → memory limit, host survives', async (_n, code) => {
    const before = process.memoryUsage().rss
    const r = await jsCodeAsync(code, add, 10_000)
    expect(r).toMatchObject({ pass: false, score: 0 })
    expect(r.detail).toMatch(/memory limit/)
    expect(process.memoryUsage().rss - before).toBeLessThan(200 * 1024 ** 2) // bomb never touched our heap
  }, 20_000)

  it('infinite loop → vm timeout', async () => {
    const r = await jsCodeAsync('while(true){}', add, 300)
    expect(r.detail).toMatch(/timed out/)
  })

  it('no require/process/fetch in the context; --permission denies fs even if a host escape were found', async () => {
    const probe = [{ expr: '[typeof require, typeof process, typeof fetch, typeof setTimeout].join()', expected: 'undefined,undefined,undefined,undefined' }]
    expect((await jsCodeAsync('', probe)).pass).toBe(true)
    expect((await jsCodeAsync('', [{ expr: 'this.constructor.constructor("return typeof process")()', expected: 'object' }])).pass).toBe(false)
  })

  it('raw runner reports non-string results as null', async () => {
    expect(await runSandboxed('1+1', { timeoutMs: 1000 })).toEqual({ ok: true, raw: null })
    expect(await runSandboxed('"x"', { timeoutMs: 1000 })).toEqual({ ok: true, raw: 'x' })
  })

  it('evaluateAsync matches evaluate for the suite coding tests', async () => {
    const cd = defaultTestSet.tests.find((t) => t.id === 'CD-03')!
    const good = 'function groupBy(arr, key) { const o = {}; for (const x of arr) (o[x[key]] ??= []).push(x); return o }'
    expect(await evaluateAsync(cd, good)).toMatchObject({ testId: 'CD-03', pass: true, score: 1 })
    expect(await evaluateAsync(cd, 'function groupBy(a){return a}')).toMatchObject({ pass: false })
    const exact = defaultTestSet.tests.find((t) => t.id === 'RS-04')!
    expect(await evaluateAsync(exact, '100 mod 7 = 2\nAnswer: Friday')).toMatchObject({ pass: true })
  })
})
