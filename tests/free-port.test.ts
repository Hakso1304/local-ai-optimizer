import { createServer } from 'node:net'
import { expect, it } from 'vitest'
import { freePort } from '../src/core/runtimes/llamacpp'

it('freePort(port) returns that port when free and rejects while something holds it', async () => {
  const p = await freePort()
  expect(await freePort(p)).toBe(p)
  const s = createServer()
  await new Promise<void>((ok) => s.listen(p, '127.0.0.1', ok))
  await expect(freePort(p)).rejects.toThrow()
  await new Promise((ok) => s.close(ok))
})
