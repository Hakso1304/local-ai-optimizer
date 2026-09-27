// Quality suite only (no ladder) on real models. Usage: npx tsx scripts/run-quality.ts [ctx=16384]
// Loads each model full-offload at ctx, applies the model's chat template (/apply-template), runs every test
// with evaluateAsync, prints per-category pass rates, and writes raw replies to docs/quality-run-<suite>-<ts>.json.
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { LlamaCppBackend } from '../src/core/runtimes/llamacpp'
import { pickDiscreteDevice } from '../src/core/runtimes/llamacpp/parse'
import { buildQualityPrompts, defaultTestSet, evaluateAsync, qualityScore, type QualityResult } from '../src/core/quality'

const MODELS = ['D:\\llm-models\\qwen2.5-1.5b-instruct-q4_k_m.gguf', 'D:\\llm-models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf']
const ctx = Number(process.argv[2] ?? 16384)
const servers = () =>
  execFileSync('tasklist', ['/FI', 'IMAGENAME eq llama-server.exe', '/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true }).split('\n').filter((l) => /^"llama-server\.exe"/i.test(l)).length

async function main(): Promise<void> {
  const b = new LlamaCppBackend(join('vendor', 'llama.cpp'))
  const dev = pickDiscreteDevice(await b.listDevices())
  if (!dev) throw new Error('no discrete GPU')
  const out: Record<string, { results: QualityResult[]; replies: { id: string; reply: string }[]; score: number | null }> = {}
  for (const model of MODELS) {
    for (let i = 0; servers() > 0; i++) {
      if (i >= 20) throw new Error('llama-server still running after 10 min')
      console.log('llama-server already running — waiting 30 s')
      await new Promise((r) => setTimeout(r, 30_000))
    }
    const name = model.split('\\').pop()!
    await b.loadModel({ modelPath: model, contextSize: ctx, gpuLayers: 999, device: dev.id, extraArgs: ['-fa', 'on'] })
    const results: QualityResult[] = []
    const replies: { id: string; reply: string }[] = []
    try {
      for (const p of buildQualityPrompts(defaultTestSet, { fillerTokens: Math.min(3000, Math.floor(ctx * 0.6)) })) {
        const test = defaultTestSet.tests.find((t) => t.id === p.testId)!
        const r = await b.runPrompt({ prompt: await b.applyTemplate(p.messages), maxTokens: p.maxTokens, temperature: 0, seed: 1, timeoutMs: 180_000 })
        const q = r.error ? { testId: test.id, category: test.category, weight: test.weight, pass: false, score: 0, detail: `request failed: ${r.error}` } : await evaluateAsync(test, r.text)
        results.push(q)
        replies.push({ id: test.id, reply: r.text })
        console.log(`${name.slice(0, 22)} ${test.id} ${q.pass ? 'PASS' : 'FAIL'} ${q.pass ? '' : q.detail.slice(0, 110)}`)
      }
    } finally {
      await b.unloadModel()
    }
    out[name] = { results, replies, score: qualityScore(results) }
    const by: Record<string, string> = {}
    for (const c of Object.keys(defaultTestSet.categoryWeights)) {
      const rs = results.filter((r) => r.category === c)
      by[c] = `${rs.filter((r) => r.pass).length}/${rs.length}`
    }
    console.log(`${name}: Q=${out[name].score?.toFixed(1)} ${JSON.stringify(by)}`)
  }
  const file = join('docs', `quality-run-${defaultTestSet.suite}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`)
  writeFileSync(file, JSON.stringify({ suite: defaultTestSet.suite, ctx, device: dev, out }, null, 1))
  console.log(`wrote ${file}; leftover llama-server ${servers()}`)
}

main().catch((e) => { console.error(e); process.exit(1) })
