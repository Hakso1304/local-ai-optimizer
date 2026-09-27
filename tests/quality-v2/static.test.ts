import { describe, expect, it } from 'vitest'
import manifestJson from '../../src/core/quality/tests.v2.json'
import suiteV1 from '../../src/core/quality/tests.v1.json'
import { runChecker } from '../../src/core/quality/checkers'
import { buildQualityPrompts } from '../../src/core/quality'
import { resolveSuiteV2, type V2Manifest, type V2Test } from '../../src/core/quality/generators.v2'

// Independent canned solutions and realistic mutations. Never derive good answers from checker.expected.
const ANSWERS: Record<string, { good: string; bad: string }> = {
  "IF2-01": {
    "good": "READY",
    "bad": "Ready"
  },
  "IF2-02": {
    "good": "north|east|south",
    "bad": "north, east, south"
  },
  "IF2-03": {
    "good": "state=ok\nretries=0",
    "bad": "state=ok; retries=0"
  },
  "IF2-04": {
    "good": "blue,green",
    "bad": "red,blue,green"
  },
  "IF2-05": {
    "good": "oak,elm,ash",
    "bad": "oak,elm,ash,oak"
  },
  "IF2-06": {
    "good": "C/A",
    "bad": "A/C"
  },
  "IF2-07": {
    "good": "3,5,12,20",
    "bad": "12,20,3,5"
  },
  "IF2-08": {
    "good": "fig,pear,apple",
    "bad": "apple,fig,pear"
  },
  "IF2-09": {
    "good": "Ada,Bo,Cy",
    "bad": "Bo,Ada,Cy"
  },
  "IF2-10": {
    "good": "OPEN",
    "bad": "CLOSED"
  },
  "IF2-11": {
    "good": "NOW",
    "bad": "LATER"
  },
  "IF2-12": {
    "good": "SAFE",
    "bad": "GO"
  },
  "RS2-07": {
    "good": "Checking the given constraints.\nAnswer: Cy",
    "bad": "Checking the given constraints.\nAnswer: Bea"
  },
  "RS2-08": {
    "good": "Checking the given constraints.\nAnswer: Niko",
    "bad": "Checking the given constraints.\nAnswer: Oren"
  },
  "RS2-09": {
    "good": "Checking the given constraints.\nAnswer: blue",
    "bad": "Checking the given constraints.\nAnswer: red"
  },
  "RS2-10": {
    "good": "Checking the given constraints.\nAnswer: yes",
    "bad": "Checking the given constraints.\nAnswer: no"
  },
  "RS2-11": {
    "good": "Checking the given constraints.\nAnswer: no",
    "bad": "Checking the given constraints.\nAnswer: yes"
  },
  "RS2-12": {
    "good": "Checking the given constraints.\nAnswer: Cy",
    "bad": "Checking the given constraints.\nAnswer: Ada"
  },
  "CD2-01": {
    "good": "function positives(xs){return xs.filter(x=>x>0)}",
    "bad": "function positives(xs){return xs.filter(x=>x>=0)}"
  },
  "CD2-02": {
    "good": "function atLeast(xs,limit){return xs.filter(x=>x>=limit)}",
    "bad": "function atLeast(xs,limit){return xs.filter(x=>x>limit)}"
  },
  "CD2-03": {
    "good": "function evenPositions(xs){return xs.filter((_,i)=>i%2===0)}",
    "bad": "function evenPositions(xs){return xs.filter(x=>x%2===0)}"
  },
  "CD2-04": {
    "good": "function slug(s){return s.trim().toLowerCase().replace(/ +/g,\"-\")}",
    "bad": "function slug(s){return s.trim().toLowerCase().replace(/ /g,\"-\")}"
  },
  "CD2-05": {
    "good": "function removeVowels(s){return s.replace(/[aeiou]/gi,\"\")}",
    "bad": "function removeVowels(s){return s.replace(/[aeiou]/g,\"\")}"
  },
  "CD2-06": {
    "good": "function isAsciiPalindrome(s){const t=s.toLowerCase().replace(/[^a-z0-9]/g,\"\");return t===[...t].reverse().join(\"\")}",
    "bad": "function isAsciiPalindrome(s){const t=s.replace(/[^a-z0-9]/g,\"\");return t===[...t].reverse().join(\"\")}"
  },
  "CD2-07": {
    "good": "function sum(xs){return xs.reduce((a,b)=>a+b,0)}",
    "bad": "function sum(xs){return xs.reduce((a,b)=>a+b)}"
  },
  "CD2-08": {
    "good": "function maximum(xs){return xs.length?Math.max(...xs):null}",
    "bad": "function maximum(xs){return xs.length?Math.max(0,...xs):null}"
  },
  "CD2-09": {
    "good": "function prefixSums(xs){let s=0;return xs.map(x=>s+=x)}",
    "bad": "function prefixSums(xs){let s=0;return xs.map(x=>{const old=s;s+=x;return old})}"
  },
  "SO2-01": {
    "good": "[{\"name\": \"Ava\", \"age\": 9}, {\"name\": \"Ben\", \"age\": 12}]",
    "bad": "[{\"name\": \"Ava\", \"age\": \"9\"}, {\"name\": \"Ben\", \"age\": \"12\"}]"
  },
  "SO2-02": {
    "good": "[{\"id\": \"X\", \"active\": true}, {\"id\": \"Y\", \"active\": false}, {\"id\": \"Z\", \"active\": true}]",
    "bad": "[{\"id\": \"X\", \"active\": true}, {\"id\": \"Y\", \"active\": true}, {\"id\": \"Z\", \"active\": true}]"
  },
  "SO2-03": {
    "good": "{\"team\": {\"name\": \"Maple\", \"members\": [\"Ada\", \"Bo\"]}, \"open\": false}",
    "bad": "{\"team\": {\"name\": \"Maple\", \"members\": [\"Ada\", \"Bo\"]}, \"open\": \"false\"}"
  },
  "SO2-04": {
    "good": "{\"orders\": [{\"id\": \"A\", \"items\": [2, 3]}, {\"id\": \"B\", \"items\": []}]}",
    "bad": "{\"orders\": [{\"id\": \"A\", \"items\": [2, 3]}, {\"id\": \"B\", \"items\": null}]}"
  },
  "SO2-05": {
    "good": "[\"b\",\"c\",\"a\"]",
    "bad": "[\"a\",\"b\",\"c\"]"
  },
  "SO2-06": {
    "good": "[\"a\", \"z\"]",
    "bad": "[\"z\", \"a\"]"
  },
  "EX2-01": {
    "good": "{\"invoice\": \"INV-731\", \"total_due\": 88}",
    "bad": "{\"invoice\": \"INV-731\", \"total_due\": 80}"
  },
  "EX2-02": {
    "good": "{\"reference\": \"PAY-406\", \"amount_paid\": 7000}",
    "bad": "{\"reference\": \"PAY-406\", \"amount_paid\": 9200}"
  },
  "EX2-03": {
    "good": "help@birch.test\nops@cedar.test\nold@birch.test",
    "bad": "help@birch.test\nops@cedar.test"
  },
  "EX2-04": {
    "good": "accounts@elm.test",
    "bad": "bills@old.test"
  },
  "EX2-05": {
    "good": "{\"owner\": \"Bo\", \"status\": \"open\"}",
    "bad": "{\"owner\": \"Cy\", \"status\": \"open\"}"
  },
  "EX2-06": {
    "good": "{\"shipment\": \"S-42\", \"destination\": \"Birch\"}",
    "bad": "{\"shipment\": \"S-42\", \"destination\": \"Cedar\"}"
  },
  "CR2-01": {
    "good": "AMBER-417",
    "bad": "INDIGO-281"
  },
  "CR2-02": {
    "good": "JUNIPER-692",
    "bad": "WILLOW-803"
  },
  "CR2-03": {
    "good": "Bo",
    "bad": "Cy"
  },
  "CR2-04": {
    "good": "North",
    "bad": "West"
  },
  "CR2-05": {
    "good": "Gold",
    "bad": "Silver"
  },
  "CR2-06": {
    "good": "Mira",
    "bad": "Niko"
  },
  "CR2-07": {
    "good": "HARBOR-572",
    "bad": "HARBOR-999"
  },
  "CR2-08": {
    "good": "Gate-B",
    "bad": "Gate-A"
  }
}

const manifest = manifestJson as V2Manifest
const statics = manifest.tests.filter((t): t is V2Test => !('generator' in t))
describe('qb-2.0.0 static item validity', () => {
  it('the truth-count reasoning puzzle has exactly one solution', () => {
    const valid = ['Ada', 'Bo', 'Cy'].filter(owner =>
      [owner === 'Ada', owner === 'Ada', owner !== 'Bo'].filter(Boolean).length === 1)
    expect(valid).toEqual(['Cy'])
    expect(runChecker(statics.find(t=>t.id==='RS2-12')!.checker, `Answer: ${valid[0]}`).pass).toBe(true)
  })
  it.each(statics.map(t => [t.id, t] as const))('%s: competent solution passes and plausible mistake fails', (id, t) => {
    const good = runChecker(t.checker, ANSWERS[id].good)
    expect(good.pass, good.detail).toBe(true)
    expect(good.score).toBe(1)
    const bad = runChecker(t.checker, ANSWERS[id].bad)
    expect(bad.pass, bad.detail).toBe(false)
    expect(bad.score).toBeLessThan(1)
    expect(runChecker(t.checker, '').pass).toBe(false)
  })
  it('covers every static item exactly once, separately from the 13 generator references', () => {
    expect(Object.keys(ANSWERS).sort()).toEqual(statics.map(t => t.id).sort())
    expect(statics).toHaveLength(47)
    expect(manifest.tests).toHaveLength(60)
    expect(new Set(manifest.tests.map(t => t.id)).size).toBe(60)
  })
  it('has the requested category counts, v1 weights, three tiers and 2?3 variants per skill', () => {
    const counts: Record<string, number> = {}, skills = new Map<string, number[]>()
    for (const t of manifest.tests) {
      counts[t.category] = (counts[t.category] ?? 0) + 1
      expect(t.weight).toBe(1)
      expect([1,2,3]).toContain(t.difficulty)
      const k = `${t.category}:${t.skill}`
      skills.set(k, [...(skills.get(k) ?? []), t.variant])
    }
    expect(counts).toEqual({instruction:12, reasoning:12, coding:12, structured:8, extraction:8, context:8})
    expect(manifest.categoryWeights).toEqual(suiteV1.categoryWeights)
    for (const vs of skills.values()) {
      expect(vs.length).toBeGreaterThanOrEqual(2); expect(vs.length).toBeLessThanOrEqual(3)
      expect(new Set(vs).size).toBe(vs.length)
    }
    for (const cat of Object.keys(counts)) expect(new Set(manifest.tests.filter(t=>t.category===cat).map(t=>t.difficulty))).toEqual(new Set([1,2,3]))
  })
  it('resolved suite uses the existing prompt builder and checker vocabulary without touching v1', () => {
    const resolved = resolveSuiteV2(manifest, 20260927)
    const prompts = buildQualityPrompts(resolved)
    expect(prompts).toHaveLength(60)
    expect(prompts.every(p=>p.messages[0].content.length>0 && p.maxTokens>0)).toBe(true)
    for (const t of resolved.tests.filter(t=>t.category==='reasoning')) {
      expect(t.checker.type).toBe('finalAnswer')
      expect(t.prompt).toContain('Answer: X')
      expect(runChecker(t.checker, 'No final answer.').pass).toBe(false)
    }
    expect(suiteV1.suite).toBe('qb-1.1.0')
  })
  it('context items contain intervening material and distinct answers, not eight copies of the same needle', () => {
    const cs = statics.filter(t=>t.category==='context')
    expect(new Set(cs.map(t=>ANSWERS[t.id].good)).size).toBe(8)
    for (const t of cs) {
      expect(t.prompt!.length).toBeGreaterThan(4000)
      expect(t.prompt!.length).toBeLessThan(8000)
    }
  })
})
