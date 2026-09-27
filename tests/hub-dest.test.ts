import { mkdirSync, mkdtempSync, rmSync, symlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { isAllowedDest, resolvedProbe } from '../src/main/hub-logic'
import { rowId } from '../src/main/validate'

describe('download destination (W4b F4/F19) and row ids', () => {
  it.skipIf(process.platform !== 'win32')('rejects a junction inside a model dir that points elsewhere; free space is read on the resolved volume [Windows: junctions]', () => {
    const base = mkdtempSync(join(tmpdir(), 'lao-dest-'))
    try {
      const root = join(base, 'models'), outside = join(base, 'elsewhere')
      mkdirSync(root); mkdirSync(outside); mkdirSync(join(root, 'sub'))
      symlinkSync(outside, join(root, 'jump'), 'junction')
      expect(isAllowedDest(join(root, 'sub'), [root])).toBe(true)
      expect(isAllowedDest(join(root, 'new', 'deeper'), [root])).toBe(true) // not created yet: nearest existing = root
      expect(isAllowedDest(join(root, 'jump'), [root])).toBe(false)
      expect(isAllowedDest(join(root, 'jump', 'x'), [root])).toBe(false)
      expect(isAllowedDest(outside, [root])).toBe(false)
      expect(resolvedProbe(join(root, 'jump', 'not-yet'))?.toLowerCase()).toBe(outside.toLowerCase())
    } finally {
      rmSync(base, { recursive: true, force: true })
    }
  })

  it('rowId accepts positive safe integers only', () => {
    expect(rowId(7)).toBe(7)
    for (const bad of [0, -1, 1.5, '7', NaN, Number.MAX_SAFE_INTEGER + 1, null]) expect(() => rowId(bad)).toThrow(/bad id/)
  })
})
