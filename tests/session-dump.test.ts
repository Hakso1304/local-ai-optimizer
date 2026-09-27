import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve, sep } from 'node:path'
import { describe, expect, it } from 'vitest'
import { readSessionDump, reserveSessionDump } from '../scripts/session-dump'

function withDump(test: (path: string) => void): void {
  const dir = mkdtempSync(join(tmpdir(), 'lao-session-dump-'))
  try { test(join(dir, 'run.json')) } finally {
    if (!resolve(dir).startsWith(`${resolve(tmpdir())}${sep}`)) throw new Error('temp cleanup escaped temp directory')
    rmSync(dir, { recursive: true, force: true })
  }
}

describe('session dump checkpoints without hardware', () => {
  it('keeps a complete current JSON and the immediately prior complete checkpoint', () => withDump((path) => {
    const dump = reserveSessionDump(path, { phase: 'created', rows: 0 })
    expect(readSessionDump(path)).toEqual({ phase: 'created', rows: 0 })
    expect(existsSync(dump.previousPath)).toBe(false)

    dump.checkpoint({ phase: 'ladder', rows: 1 })
    expect(dump.current()).toEqual({ phase: 'ladder', rows: 1 })
    expect(readSessionDump(path)).toEqual({ phase: 'ladder', rows: 1 })
    expect(readSessionDump(dump.previousPath)).toEqual({ phase: 'created', rows: 0 })

    dump.checkpoint({ phase: 'quality', rows: 2 })
    expect(dump.current()).toEqual({ phase: 'quality', rows: 2 })
    expect(readSessionDump(path)).toEqual({ phase: 'quality', rows: 2 })
    expect(readSessionDump(dump.previousPath)).toEqual({ phase: 'ladder', rows: 1 })
    expect(readdirSync(resolve(path, '..')).sort()).toEqual(['run.json', 'run.json.previous'])
  }))

  it('refuses an already reserved output name without changing its bytes', () => withDump((path) => {
    reserveSessionDump(path, { phase: 'first' })
    const bytes = readFileSync(path)
    expect(() => reserveSessionDump(path, { phase: 'replacement' })).toThrow(/refusing to overwrite existing session dump/)
    expect(readFileSync(path)).toEqual(bytes)
  }))

  it('leaves the last complete checkpoint intact when serialization fails', () => withDump((path) => {
    const dump = reserveSessionDump(path, { phase: 'initial' } as { phase: string; invalid?: bigint })
    dump.checkpoint({ phase: 'saved' })
    const current = readFileSync(path), previous = readFileSync(dump.previousPath)
    expect(() => dump.checkpoint({ phase: 'broken', invalid: 1n })).toThrow()
    expect(readFileSync(path)).toEqual(current)
    expect(readFileSync(dump.previousPath)).toEqual(previous)
    expect(dump.current()).toEqual({ phase: 'saved' })
    expect(readdirSync(resolve(path, '..')).sort()).toEqual(['run.json', 'run.json.previous'])
  }))
})
