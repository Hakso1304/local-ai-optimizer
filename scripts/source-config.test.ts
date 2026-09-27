import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { sourceStatus } from './measurement-launcher'

describe('source manifest includes tsx configuration', () => {
  it('includes root tsconfig and its local extends chain', async () => {
    const cwd = mkdtempSync(join(tmpdir(), 'lao-tsconfig-gate-'))
    try {
      writeFileSync(join(cwd, 'package.json'), '{}')
      writeFileSync(join(cwd, 'tsconfig.json'), '{"extends":"./tsconfig.base.json"}')
      writeFileSync(join(cwd, 'tsconfig.base.json'), '{"compilerOptions":{"target":"ES2022"}}')
      const files = await sourceStatus(cwd, 'a'.repeat(40))
      expect(files).toEqual(expect.arrayContaining(['package.json', 'tsconfig.json', 'tsconfig.base.json']))
    } finally { rmSync(cwd, { recursive: true, force: true }) }
  })
})
