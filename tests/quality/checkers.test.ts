import { describe, expect, it } from 'vitest'
import {
  containsAll, exactMatch, extractCode, jsCode, jsonEqual, jsonSchema, needle, numberMatch, regex, stripThinking, validateSchema, wordCount
} from '../../src/core/quality/checkers'

describe('text checkers', () => {
  it('exactMatch normalizes case, whitespace, wrapping markdown and trailing period', () => {
    expect(exactMatch('  **Friday.**\n', 'Friday').pass).toBe(true)
    expect(exactMatch('friday', 'Friday').pass).toBe(true)
    expect(exactMatch('banana', 'BANANA', true).pass).toBe(false)
    expect(exactMatch('It is Friday', 'Friday')).toMatchObject({ pass: false, score: 0 })
    expect(exactMatch('a@x.io\n\n  b@y.io ', 'a@x.io\nb@y.io').pass).toBe(true)
    expect(exactMatch('b@y.io\na@x.io', 'a@x.io\nb@y.io').pass).toBe(false)
  })
  it('strips thinking blocks, closed or not', () => {
    expect(stripThinking('<think>Carol? Bob?</think>\nCarol')).toBe('Carol')
    expect(stripThinking('<think>never closed')).toBe('')
    expect(exactMatch('<think>hmm Dave</think>Carol', 'Carol').pass).toBe(true)
  })
  it('regex', () => {
    expect(regex('red,green,blue', '^red,green,blue$').pass).toBe(true)
    expect(regex('red, green, blue', '^red,green,blue$').pass).toBe(false)
  })
  it('numberMatch takes the last number, allows thousands separators', () => {
    expect(numberMatch('27 pens cost 36', 36).pass).toBe(true)
    expect(numberMatch('$1,428.00', 1428).pass).toBe(true)
    expect(numberMatch('thirty-six', 36)).toMatchObject({ pass: false, detail: 'no number in output' })
    expect(numberMatch('35', 36).pass).toBe(false)
  })
  it('containsAll gives partial score', () => {
    expect(containsAll('alpha Beta', ['alpha', 'beta'])).toMatchObject({ pass: true, score: 1 })
    expect(containsAll('alpha', ['alpha', 'beta'])).toMatchObject({ pass: false, score: 0.5 })
  })
  it('needle: full, partial, miss', () => {
    expect(needle('HELIOTROPE-5', 'HELIOTROPE-5')).toMatchObject({ pass: true, score: 1 })
    expect(needle('Heliotrope', 'HELIOTROPE-5')).toMatchObject({ pass: false, score: 0.5 })
    expect(needle('I do not know', 'HELIOTROPE-5')).toMatchObject({ pass: false, score: 0 })
  })
  it('wordCount', () => {
    expect(wordCount('The ocean is very deep.', 5, 5).pass).toBe(true)
    expect(wordCount("The ocean's waves crash loudly.", 5, 5).pass).toBe(true)
    expect(wordCount('The ocean is deep.', 5, 5).pass).toBe(false)
  })
})

describe('JSON checkers', () => {
  const schema = {
    type: 'object' as const, required: ['title', 'year'],
    properties: { title: { type: 'string' as const }, year: { type: 'integer' as const }, tags: { type: 'array' as const, minItems: 2, items: { type: 'string' as const } }, kind: { enum: ['a', 'b'] } }
  }
  it('validateSchema reports each violation path', () => {
    expect(validateSchema(schema, { title: 'x', year: 2020, tags: ['a', 'b'], kind: 'a' })).toEqual([])
    expect(validateSchema(schema, { title: 1, year: 20.5, tags: ['a'], kind: 'c' })).toEqual([
      '$.title: expected string, got number', '$.year: expected integer, got number', '$.tags: 1 items < minItems 2', '$.kind: not in enum'
    ])
    expect(validateSchema(schema, {})).toEqual(['$.title: required', '$.year: required'])
    expect(validateSchema(schema, [])).toEqual(['$: expected object, got array'])
  })
  it('jsonSchema accepts a ```json fence, rejects prose and invalid JSON', () => {
    expect(jsonSchema('```json\n{"title":"x","year":1999}\n```', schema).pass).toBe(true)
    expect(jsonSchema('Here it is: {"title":"x","year":1999}', schema)).toMatchObject({ pass: false, score: 0 })
    expect(jsonSchema('{"title":"x","year":"1999"}', schema).detail).toMatch(/year: expected integer/)
  })
  it('jsonEqual ignores key order, not array order', () => {
    expect(jsonEqual('[{"age":31,"name":"Tom"}]', [{ name: 'Tom', age: 31 }]).pass).toBe(true)
    expect(jsonEqual('[{"name":"Ana"},{"name":"Tom"}]', [{ name: 'Tom' }, { name: 'Ana' }]).pass).toBe(false)
    expect(jsonEqual('[{"name":"Tom","age":"31"}]', [{ name: 'Tom', age: 31 }]).pass).toBe(false)
  })
})

describe('jsCode sandbox', () => {
  const add = [{ expr: 'add(2,3)', expected: 5 }, { expr: 'add(-1,1)', expected: 0 }]

  it('extracts the first js fence and strips export', () => {
    expect(extractCode('text\n```python\nx=1\n```\n```javascript\nexport function f(){}\n```')).toBe('function f(){}')
    expect(extractCode('const g = 1')).toBe('const g = 1')
  })
  it('passes correct code, bare or fenced, and scores partial cases', () => {
    expect(jsCode('function add(a,b){return a+b}', add)).toMatchObject({ pass: true, score: 1 })
    expect(jsCode('Sure!\n```js\nconst add = (a, b) => a + b\n```\nDone.', add).pass).toBe(true)
    expect(jsCode('function add(a,b){return a>0?a+b:7}', add)).toMatchObject({ pass: false, score: 0.5 })
  })
  it('reports syntax errors and missing functions instead of throwing', () => {
    expect(jsCode('function add(a,b){', add)).toMatchObject({ pass: false, score: 0 })
    expect(jsCode('function sub(a,b){return a-b}', add).detail).toMatch(/add is not defined/)
  })
  it('times out an infinite loop, sync or microtask', () => {
    const t0 = Date.now()
    expect(jsCode('while(true){}', add, 200).detail).toMatch(/timed out/)
    expect(jsCode('Promise.resolve().then(function f(){return Promise.resolve().then(f)})', add, 200).detail).toMatch(/timed out/)
    expect(Date.now() - t0).toBeLessThan(3000)
  })
  it('has no require, process, fetch, timers, eval or host-realm escape', () => {
    const probe = (expr: string) => jsCode('', [{ expr, expected: 'undefined' }])
    for (const g of ['require', 'process', 'fetch', 'setTimeout', 'module.require', 'Buffer']) expect(probe(`typeof ${g}`).pass, g).toBe(true)
    expect(jsCode('', [{ expr: 'eval("1")', expected: 1 }]).detail).toMatch(/Code generation from strings disallowed/)
    expect(jsCode('', [{ expr: 'new Function("return 1")()', expected: 1 }]).pass).toBe(false)
    expect(jsCode('', [{ expr: 'this.constructor.constructor("return typeof process")()', expected: 'object' }]).pass).toBe(false)
    expect(jsCode('require("fs").readFileSync("x")', add).detail).toMatch(/require is not defined/)
  })
  it('CommonJS-style module.exports code still works', () => {
    expect(jsCode('function add(a,b){return a+b}\nmodule.exports = { add }', add).pass).toBe(true)
  })
})
