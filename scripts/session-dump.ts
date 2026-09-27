import { constants, copyFileSync, existsSync, fsyncSync, openSync, closeSync, readFileSync, renameSync, rmSync, writeSync } from 'node:fs'
import { randomUUID } from 'node:crypto'

export interface SessionDump<T extends object> {
  readonly path: string
  readonly previousPath: string
  current(): T
  checkpoint(value: T): void
}

function durableCreate(path: string, text: string): void {
  const fd = openSync(path, 'wx')
  try {
    const bytes = Buffer.from(text)
    let offset = 0
    while (offset < bytes.length) offset += writeSync(fd, bytes, offset, bytes.length - offset)
    fsyncSync(fd)
  } finally { closeSync(fd) }
}

/** Reserve the evidence name before probes; replace only from complete, synced JSON files. */
export function reserveSessionDump<T extends object>(path: string, initial: T): SessionDump<T> {
  if (existsSync(path)) throw new Error(`refusing to overwrite existing session dump: ${path}`)
  durableCreate(path, JSON.stringify(initial, null, 1))
  const previousPath = `${path}.previous`
  let latest = initial
  return {
    path, previousPath,
    current: () => latest,
    checkpoint(value) {
      const temp = `${path}.${process.pid}.${randomUUID()}.next`
      const previousTemp = `${previousPath}.${process.pid}.${randomUUID()}.next`
      try {
        durableCreate(temp, JSON.stringify(value, null, 1))
        copyFileSync(path, previousTemp, constants.COPYFILE_EXCL)
        renameSync(previousTemp, previousPath)
        renameSync(temp, path)
        latest = value
      } finally {
        rmSync(temp, { force: true })
        rmSync(previousTemp, { force: true })
      }
    }
  }
}

export function readSessionDump<T extends object>(path: string): T {
  return JSON.parse(readFileSync(path, 'utf8')) as T
}
