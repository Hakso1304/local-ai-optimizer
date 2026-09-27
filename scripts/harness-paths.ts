import { existsSync } from 'node:fs'
import { isAbsolute, resolve } from 'node:path'

/** Resolve a CLI database argument before opening it: openDb creates missing files. */
export function existingDbPath(raw: string | undefined, required: boolean, env: NodeJS.ProcessEnv = process.env, exists: (path: string) => boolean = existsSync): string | null {
  if (!raw) {
    if (required) throw new Error('scenario H requires --db <existing database>')
    return null
  }
  const expanded = raw.replace(/%([^%]+)%/g, (_, name: string) => {
    const key = Object.keys(env).find((k) => k.toUpperCase() === name.toUpperCase())
    if (!key || env[key] === undefined) throw new Error(`--db uses undefined environment variable %${name}%`)
    return env[key]!
  })
  const absolute = isAbsolute(expanded) ? expanded : resolve(expanded)
  if (!exists(absolute)) throw new Error(`--db does not exist: ${absolute}`)
  return absolute
}
