import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { createServer } from 'node:http'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, sep } from 'node:path'
import { promisify } from 'node:util'
import { build } from 'esbuild'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { buildQualityPrompts, suiteFor } from '../../src/core/quality'
import { TEMPLATE_DATE } from '../../src/core/runtimes/llamacpp'

const runFile = promisify(execFile)
const sha = (s: string) => createHash('sha256').update(s).digest('hex')

describe('RECHECK6 R1: replay lineage', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lao-proof-replay-'))
  const modelPath = join(dir, 'fake.gguf')
  const first = join(dir, 'original.json'), second = join(dir, 'first-replay.json'), third = join(dir, 'second-replay.json')
  const template = 'fake-template'
  let server: ReturnType<typeof createServer>
  let port = 0

  beforeAll(async () => {
    writeFileSync(modelPath, 'x')
    server = createServer(async (req, res) => {
      res.setHeader('content-type', 'application/json')
      if (req.url === '/props') return res.end(JSON.stringify({ model_path: modelPath, chat_template: template }))
      if (req.url === '/apply-template') {
        const parts: Buffer[] = []
        for await (const part of req) parts.push(Buffer.from(part))
        const body = JSON.parse(Buffer.concat(parts).toString()) as { messages: { content: string }[]; chat_template_kwargs: { enable_thinking: boolean; date_string: string } }
        expect(body.chat_template_kwargs.date_string).toBe(TEMPLATE_DATE)
        return res.end(JSON.stringify({ prompt: `${body.messages[0].content}\nthink=${body.chat_template_kwargs.enable_thinking}` }))
      }
      res.statusCode = 404
      res.end('{}')
    })
    await new Promise<void>((done) => server.listen(0, '127.0.0.1', done))
    port = (server.address() as { port: number }).port
  })
  afterAll(async () => {
    await new Promise<void>((done) => server.close(() => done()))
    const root = resolve(tmpdir()) + sep
    if (!resolve(dir).startsWith(root)) throw new Error('temporary replay path escaped tmpdir')
    rmSync(dir, { recursive: true, force: true })
  })

  it('keeps reconstructed rows non-comparable after two runs of the real replay script', async () => {
    const suite = suiteFor('quick', 42)
    const item = suite.tests[0]
    expect(buildQualityPrompts(suite, { fillerTokens: Math.floor(2048 * 0.6), thinking: false })[0].testId).toBe(item.id)
    const artifact = {
      sessionId: 5,
      session: { payload: {
        request: { qualityMode: 'quick', qualitySeed: 42, genConfigs: [{ id: 'off', thinking: false, temperature: 0, source: 'default' }] },
        candidates: [{ config: { id: 'fake|ngl=all' }, model: { id: modelPath, fileBytes: 1, supportsThinking: true } }]
      } },
      qualityResults: [{ id: 1, model_id: modelPath, payload: {
        configId: 'fake|ngl=all', genId: 'off', testId: item.id, category: item.category,
        weight: item.weight, pass: true, score: 1, detail: '', ctx: 2048, sample: 1,
        checkerVersion: suite.suite, templateHash: sha(template), modelFingerprint: `${modelPath}#1`,
        requestedTemplateKwargs: { enable_thinking: false }, acceptedSampling: { temperature: 0, seed: 1 }
      } }]
    }
    writeFileSync(first, JSON.stringify(artifact))
    const script = resolve('scripts/prove-quality-rows.ts')
    const executable = join(dir, 'replay.mjs')
    await build({ entryPoints: [script], outfile: executable, bundle: true, platform: 'node', format: 'esm' })
    const replay = async (input: string, output: string) => {
      await runFile(process.execPath, [executable, '--in', input, '--out', output, '--port', String(port), '--session', '5'], { cwd: resolve('.'), timeout: 20_000 })
      return JSON.parse(readFileSync(output, 'utf8')) as typeof artifact & {
        qualityResults: { id: number; payload: typeof artifact.qualityResults[0]['payload'] & {
          promptSha256?: string; renderProof?: { status: string }; proofProvenance?: { status: string; originalPromptHashPresent: boolean }
        } }[]
      }
    }
    const once = await replay(first, second)
    expect(once.qualityResults[0].payload.proofProvenance).toMatchObject({ status: 'reconstructed', originalPromptHashPresent: false })
    expect(once.qualityResults[0].payload.renderProof?.status).toBe('reconstructed')
    const twice = await replay(second, third)
    expect(twice.qualityResults[0].payload.proofProvenance).toMatchObject({ status: 'reconstructed', originalPromptHashPresent: false })
    expect(twice.qualityResults[0].payload.renderProof?.status).toBe('reconstructed')
  })
})
