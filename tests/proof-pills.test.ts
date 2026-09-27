import { describe, expect, it } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { ProofPills, notEvaluableReason, proofSummary } from '../src/renderer/src/ProofPills'
import type { Insight } from '../src/shared/interpret-types'

const req = { temperature: 0.6, topP: 0.95, topK: 20, minP: null, seed: 7 }
const acc = { temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0.05, seed: 7 }

describe('generation-config application proof (rows as stored)', () => {
  it('counts proved / reconstructed / unproved / contradicted, seed and sampling mismatches; pre-P1 rows apart', () => {
    const s = proofSummary([
      { renderProof: { status: 'proved' }, proofProvenance: { originalPromptHashPresent: true }, requestedSampling: req, acceptedSampling: acc },
      { renderProof: { status: 'proved' }, proofProvenance: { originalPromptHashPresent: false, status: 'reconstructed' }, requestedSampling: req, acceptedSampling: acc },
      { renderProof: { status: 'unproved' }, requestedSampling: req, acceptedSampling: { ...acc, seed: undefined } },
      { renderProof: { status: 'contradicted' }, requestedSampling: req, acceptedSampling: { ...acc, top_k: 40 } },
      { testId: 'old' } // written before the proof fields existed
    ])
    expect(s).toEqual({ rows: 5, preP1: 1, proved: 1, reconstructed: 1, unproved: 1, contradicted: 1, seedUnverified: 1, samplingMismatch: 1 })
    // a requested minP of null is not a requested sampler; an absent accepted block leaves the seed unverified
    expect(proofSummary([{ requestedSampling: req, acceptedSampling: null }])).toMatchObject({ seedUnverified: 1, samplingMismatch: 1, unproved: 1 })
  })

  it('renders the pills, and old rows say "no proof recorded (pre-P1)" instead of guessing', () => {
    const html = renderToStaticMarkup(createElement(ProofPills, { rows: [{ renderProof: { status: 'proved' }, requestedSampling: req, acceptedSampling: acc }, { renderProof: { status: 'contradicted' } }] }))
    expect(html).toContain('proved 1/2 rows')
    expect(html).toContain('contradicted 1')
    expect(renderToStaticMarkup(createElement(ProofPills, { rows: [{}, {}] }))).toContain('no proof recorded (pre-P1)')
  })

  it('the "not evaluable" reason comes from the engine\'s I-8.0 insight for that config and setting', () => {
    const ins = [{ ruleId: 'I-8.0', key: 'gen.comparable', severity: 'note', metric: 'gen', evidence: [], evaluable: false, configId: 'c1',
      text: '[I-8.0] Qwen: think-low T1 not comparable — effort kwarg not proven on 3 rows; it is not considered.' }] as Insight[]
    expect(notEvaluableReason(ins, 'c1', 'think-low T1')).toBe('effort kwarg not proven on 3 rows')
    expect(notEvaluableReason(ins, 'c2', 'think-low T1')).toBeNull() // other config
    expect(notEvaluableReason(ins, 'c1', 'off')).toBeNull()
  })
})
