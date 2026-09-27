import { describe, expect, it } from 'vitest'
import manifestJson from '../../src/core/quality/tests.v2.json'
import { runChecker, runCheckerAsync } from '../../src/core/quality/checkers'
import { arithmetic, wordProblem, extraction, schemaRecord, codeConstants, GENERATORS_V2, materializeV2, resolveSuiteV2, seedForItem, type GeneratorEntry, type V2Manifest } from '../../src/core/quality/generators.v2'

const manifest = manifestJson as V2Manifest
const variants = [1, 2, 3] as const
function accepts(test: ReturnType<typeof arithmetic>, good: string, bad: string) {
  const g = runChecker(test.checker, good), b = runChecker(test.checker, bad)
  expect(g.pass, g.detail).toBe(true)
  expect(g.score).toBe(1)
  expect(b.pass, b.detail).toBe(false)
}

describe('independently checked seed-1234 witnesses', () => {
  it('arithmetic obeys precedence, parentheses and exact division', () => {
    const answers = [142, 18, 142] // 16+14*9; (16-14)*9; 144/9+14*9
    for (const v of variants) {
      const t = arithmetic(1234, { variant: v })
      accepts(t, `Working through the expression.\nAnswer: ${answers[v - 1]}`, `Answer: ${answers[v - 1] + 1}`)
      expect(runChecker(t.checker, `Answer: 0 or ${answers[v - 1]}`).pass).toBe(false)
      expect(runChecker(t.checker, String(answers[v - 1])).pass).toBe(false)
    }
  })
  it('word problems distinguish total removals, one discount and reserve-before-division', () => {
    const answers = [31, 351, 4] // 4*10-9; 4*10*9-9; (49-9)/10
    const mistakes = [4, 324, 5]
    for (const v of variants) accepts(wordProblem(1234, { variant: v }), `Answer: ${answers[v - 1]}`, `Answer: ${mistakes[v - 1]}`)
  })
  it('entity/date extraction ignores a later unrelated event, a rejected date and obsolete reminders', () => {
    for (const v of variants) accepts(extraction(1234, { variant: v }),
      '{"reference":"EV-13686","date":"2032-11-22","event":"Aster"}',
      '{"reference":"EV-13686","date":"2032-11-23","event":"Aster"}')
  })
  it('schema instances require the randomized names, values and types', () => {
    for (const v of variants) {
      const t = schemaRecord(1234, { variant: v })
      const ready = v !== 2
      const good = { label_12c375: 'unit-9734', count_b412ce: 3, ready_e72192: ready }
      accepts(t, JSON.stringify(good), JSON.stringify({ ...good, count_b412ce: '3' }))
      for (const bad of [{ ...good, count_b412ce: 4 }, { ...good, ready_e72192: !ready }, {}, { label: 'unit-9734', count: 3, ready }]) {
        expect(runChecker(t.checker, JSON.stringify(bad)).pass).toBe(false)
      }
    }
  })
  it('code tasks catch omitted offset, wrong clamp boundary and JavaScript negative remainder', () => {
    const good = ['function transform(x){return x*2+13}', 'function transform(x){if(x < -2)return -2;if(x>13)return 13;return x}', 'function transform(x){return x-13*Math.floor(x/13)}']
    const bad = ['function transform(x){return x*2}', 'function transform(x){return Math.max(0,Math.min(13,x))}', 'function transform(x){return x%13}']
    for (const v of variants) accepts(codeConstants(1234, { variant: v }), good[v - 1], bad[v - 1])
  })
  it('the production async checker also accepts generated code and rejects its realistic bug', async () => {
    const t = codeConstants(1234, { variant: 3 })
    expect((await runCheckerAsync(t.checker, 'function transform(x){return x-13*Math.floor(x/13)}')).pass).toBe(true)
    expect((await runCheckerAsync(t.checker, 'function transform(x){return x%13}')).pass).toBe(false)
  })
})

describe('procedural expectations agree with the questions across seeds', () => {
  // Independent reference answers are read from the QUESTION, never from checker expectations.
  for (const seed of [0, 1, 7, 42, 8675309, 20260927, 0xffffffff]) {
    it(`seed ${seed}: arithmetic/word answers, extraction, schema and code contracts`, () => {
      for (const v of variants) {
        const a = arithmetic(seed, { variant: v }), p = a.prompt!
        const ns = p.slice(0, p.indexOf(' using')).match(/\d+/g)!.map(BigInt)
        const n = v === 1 ? ns[0] + ns[1] * ns[2] : v === 2 ? (ns[0] - ns[1]) * ns[2] : ns[0] / ns[1] + ns[2] * ns[3]
        accepts(a, `Answer: ${n}`, `Answer: ${n + 1n}`)

        const w = wordProblem(seed, { variant: v }), nums = w.prompt!.match(/\d+/g)!.map(Number)
        const answer = v === 1 ? nums[0] * nums[1] - nums[2] : v === 2 ? nums[0] * nums[1] * nums[2] - nums[3] : (nums[0] - nums[1]) / nums[2]
        expect(Number.isInteger(answer)).toBe(true)
        accepts(w, `Answer: ${answer}`, `Answer: ${answer + 1}`)

        const e = extraction(seed, { variant: v })
        const approved = /Final approved update: (\w+) is confirmed for ([\d-]+)/.exec(e.prompt!)!
        const ref = /Notice (EV-\d+):/.exec(e.prompt!)![1]
        accepts(e, JSON.stringify({event:approved[1], date:approved[2], reference:ref}), JSON.stringify({event:approved[1], date:'1900-01-01', reference:ref}))
        const [year, month, day] = approved[2].split('-').map(Number)
        expect(new Date(Date.UTC(year, month - 1, day)).toISOString().slice(0, 10)).toBe(approved[2])

        const s = schemaRecord(seed, { variant: v })
        const label = /string field "([^"]+)" must be "([^"]+)"/.exec(s.prompt!)!
        const count = /integer field "([^"]+)" must be (\d+)/.exec(s.prompt!)!
        const ready = /boolean field "([^"]+)" must be (true|false)/.exec(s.prompt!)!
        const record = { [label[1]]: label[2], [count[1]]: Number(count[2]), [ready[1]]: ready[2] === 'true' }
        accepts(s, JSON.stringify(record), JSON.stringify({ ...record, [count[1]]: Number(count[2]) + 1 }))

        const c = codeConstants(seed, { variant: v }), question = c.prompt!
        let solution: string, mistake: string
        if (v === 1) {
          const m = /return (\d+) \* x \+ (\d+)/.exec(question)!
          solution = `function transform(x){return ${m[2]} + x*${m[1]}}`
          mistake = `function transform(x){return x*${m[1]}}`
        } else if (v === 2) {
          const m = /interval \[(-\d+), (\d+)\]/.exec(question)!
          solution = `function transform(x){if(x<${m[1]})return ${m[1]};if(x>${m[2]})return ${m[2]};return x}`
          mistake = `function transform(x){if(x<0)return 0;if(x>${m[2]})return ${m[2]};return x}`
        } else {
          const mod = /r < (\d+)/.exec(question)![1]
          solution = `function transform(x){return x-${mod}*Math.floor(x/${mod})}`
          mistake = `function transform(x){return x%${mod}}`
        }
        accepts(c, solution, mistake)
      }
    })
  }
})

describe('materialization contract for index.ts integration', () => {
  it('is deterministic, order-independent and never mutates the manifest or shares mutable checker state', () => {
    const before = JSON.stringify(manifest)
    const a = resolveSuiteV2(manifest, 42), b = resolveSuiteV2(manifest, 42)
    expect(a).toEqual(b)
    const reverse = resolveSuiteV2({ ...manifest, tests: [...manifest.tests].reverse() }, 42)
    expect(reverse.tests.reverse()).toEqual(a.tests)
    a.tests[0].prompt = 'changed'
    expect(JSON.stringify(manifest)).toBe(before)
    expect(b.tests[0].prompt).not.toBe('changed')
    expect(a.tests[0].checker).not.toBe(b.tests[0].checker)
  })
  it('changing suite seed changes generated questions and leaves static items identical', () => {
    const a = resolveSuiteV2(manifest, 0), b = resolveSuiteV2(manifest, 1)
    const refs = manifest.tests.filter((t): t is GeneratorEntry => 'generator' in t)
    expect(refs).toHaveLength(13)
    expect(refs.filter(ref => a.tests.find(t=>t.id===ref.id)!.prompt !== b.tests.find(t=>t.id===ref.id)!.prompt).length).toBeGreaterThanOrEqual(12)
    for (const t of manifest.tests.filter(t=>!('generator' in t))) expect(materializeV2(t,0)).toEqual(materializeV2(t,1))
    expect(new Set(refs.map(t=>seedForItem(42,t.id))).size).toBe(13)
  })
  it('all generated instances carry replay provenance and honor manifest identity/budgets', () => {
    for (const ref of manifest.tests.filter((t): t is GeneratorEntry => 'generator' in t)) {
      const t = materializeV2(ref, 42)
      expect(t).toMatchObject({ id: ref.id, category: ref.category, weight: 1, maxTokens: ref.maxTokens, difficulty: ref.difficulty, skill: ref.skill, variant: ref.variant,
        instanceSeed: seedForItem(42,ref.id), generatorVersion: 'qbg-2.0.0' })
      expect(t).not.toHaveProperty('generator')
      expect(t.checker).toBeDefined()
    }
  })
  it('rejects invalid seeds, variants, generator names, category mismatches and duplicate IDs', () => {
    for (const f of Object.values(GENERATORS_V2)) {
      for (const seed of [-1, 1.5, NaN, Infinity, 2**32]) expect(()=>f(seed)).toThrow(/uint32/)
      expect(()=>f(1, {variant:4 as never})).toThrow(/variant/)
    }
    const ref = manifest.tests.find((t): t is GeneratorEntry => 'generator' in t)!
    expect(()=>materializeV2({...ref, generator:'__proto__' as never}, 0)).toThrow(/unknown v2 generator/)
    expect(()=>materializeV2({...ref, category:'context'}, 0)).toThrow(/category mismatch/)
    expect(()=>resolveSuiteV2({...manifest, tests:[ref,ref]},0)).toThrow(/duplicate/)
    expect(()=>resolveSuiteV2({...manifest, suite:'qb-1.1.0'},0)).toThrow(/qb-2.0.0/)
  })
})
