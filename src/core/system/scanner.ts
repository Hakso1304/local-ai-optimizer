import { runPowerShell, runProcess } from '../exec'
import type { DiskInfo, GpuInfo, GpuVendor, Sourced, SystemProfile } from '../../shared/types'

// One PowerShell spawn for everything (startup is ~0.5s). Each section is caught separately so one
// failing query never hides the others.
const GPU_CLASS = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'
export const SCAN_SCRIPT = String.raw`
function Q($b) { try { @{ ok = $true; data = (& $b) } } catch { @{ ok = $false; error = $_.Exception.Message } } }
$r = [ordered]@{}
$r.os = Q { Get-CimInstance Win32_OperatingSystem -ErrorAction Stop | Select-Object Caption, Version, BuildNumber, TotalVisibleMemorySize, FreePhysicalMemory }
$r.cpu = Q { @(Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors) }
$r.video = Q { @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object {
  $drv = $null
  try { $drv = (Get-ItemProperty ("HKLM:\SYSTEM\CurrentControlSet\Enum\" + $_.PNPDeviceID) -Name Driver -ErrorAction Stop).Driver } catch {}
  [pscustomobject]@{ Name = $_.Name; PNPDeviceID = $_.PNPDeviceID; DriverVersion = $_.DriverVersion; AdapterRAM = $_.AdapterRAM; Driver = $drv }
}) }
# SilentlyContinue: the class key has an ACL-protected 'Properties' subkey that aborts enumeration.
$r.gpuReg = Q { @(Get-ChildItem '${GPU_CLASS}' -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match '^\d{4}$' } | ForEach-Object {
  $p = $null
  try { $p = Get-ItemProperty $_.PSPath -ErrorAction Stop } catch {}
  [pscustomobject]@{ Key = $_.PSChildName; DriverDesc = $p.DriverDesc; QwMemorySize = $p.'HardwareInformation.qwMemorySize' }
}) }
$r.disks = Q { @(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' -ErrorAction Stop | Select-Object DeviceID, Size, FreeSpace) }
$r | ConvertTo-Json -Depth 6 -Compress
`

interface Section<T> {
  ok: boolean
  data?: T
  error?: string
}
interface RawScan {
  os: Section<{ Caption: string; Version: string; BuildNumber: string; TotalVisibleMemorySize: number; FreePhysicalMemory: number }>
  cpu: Section<OneOrMany<{ Name: string; NumberOfCores: number; NumberOfLogicalProcessors: number }>>
  video: Section<OneOrMany<{ Name: string; PNPDeviceID: string; DriverVersion: string | null; AdapterRAM: number | null; Driver: string | null }>>
  gpuReg: Section<OneOrMany<{ Key: string; DriverDesc: string | null; QwMemorySize: number | null }>>
  disks: Section<OneOrMany<{ DeviceID: string; Size: number; FreeSpace: number }>>
}
type OneOrMany<T> = T | T[]

export interface ScanDeps {
  powershell: (script: string) => Promise<string>
  exec: (file: string, args: string[]) => Promise<string>
}

const defaultDeps: ScanDeps = {
  powershell: (s) => runPowerShell(s, 30_000),
  exec: async (f, a) => (await runProcess(f, a, 8_000)).stdout
}

const arr = <T>(x: OneOrMany<T> | undefined | null): T[] => (x == null ? [] : Array.isArray(x) ? x : [x])
const ok = <T>(value: T, source: string): Sourced<T> => ({ value, status: 'available', source })
const fail = <T>(source: string, error: string): Sourced<T> => ({ value: null, status: 'unavailable', source, error })

function section<S, T>(s: Section<S> | undefined, source: string, map: (d: S) => T): Sourced<T> {
  if (!s) return fail(source, 'section missing from scan output')
  if (!s.ok || s.data == null) return fail(source, s.error ?? 'query returned no data')
  try {
    return ok(map(s.data), source)
  } catch (e) {
    return fail(source, `parse error: ${(e as Error).message}`)
  }
}

export function vendorFromPnp(pnp: string): GpuVendor {
  const ven = /VEN_([0-9A-F]{4})/i.exec(pnp)?.[1]?.toUpperCase()
  return ven === '10DE' ? 'nvidia' : ven === '1002' ? 'amd' : ven === '8086' ? 'intel' : 'other'
}

// ponytail: name/size heuristic; upgrade to DXGI adapter flags if it misclassifies real hardware.
function guessIntegrated(name: string, vram: number | null): boolean {
  if (/Radeon\(TM\) Graphics|Radeon Vega|Intel.*(UHD|HD Graphics|Iris)/i.test(name)) return true
  return vram != null && vram < 2 * 1024 ** 3
}

/** Join present adapters (Win32_VideoController) to the display class registry via the device's
 *  Driver key. Registry entries with no present adapter (stale drivers) are ignored. */
export function buildGpus(raw: RawScan): Sourced<GpuInfo[]> {
  const src = 'Win32_VideoController + registry HardwareInformation.qwMemorySize'
  return section(raw.video, src, (videos) => {
    const reg = new Map(arr(raw.gpuReg.ok ? raw.gpuReg.data : undefined).map((r) => [r.Key, r]))
    return arr(videos).map((v): GpuInfo => {
      const key = v.Driver?.split('\\').pop()
      const entry = key ? reg.get(key) : undefined
      const regSrc = `registry Class\\{4d36e968...}\\${key ?? '?'} HardwareInformation.qwMemorySize`
      let vram: Sourced<number>
      if (!raw.gpuReg.ok) vram = fail(regSrc, raw.gpuReg.error ?? 'registry query failed')
      else if (!entry) vram = fail(regSrc, 'no display-class registry entry linked to this adapter')
      else if (typeof entry.QwMemorySize !== 'number') vram = fail(regSrc, 'qwMemorySize not present for this adapter')
      else vram = ok(entry.QwMemorySize, regSrc)
      return {
        name: v.Name.trim(),
        vendor: vendorFromPnp(v.PNPDeviceID),
        pnpDeviceId: v.PNPDeviceID,
        driverVersion: v.DriverVersion ?? null,
        dedicatedVramBytes: vram,
        isIntegrated: guessIntegrated(v.Name, vram.value)
      }
    })
  })
}

async function detectCuda(gpus: Sourced<GpuInfo[]>, deps: ScanDeps): Promise<Sourced<{ available: boolean; version?: string }>> {
  const src = 'nvidia-smi'
  if (gpus.status === 'available' && !gpus.value!.some((g) => g.vendor === 'nvidia')) {
    return { value: { available: false }, status: 'unsupported', source: 'no NVIDIA adapter present' }
  }
  try {
    const out = await deps.exec('nvidia-smi', [])
    const version = /CUDA Version:\s*([\d.]+)/.exec(out)?.[1]
    if (!version) return fail(src, 'nvidia-smi ran but reported no CUDA version')
    return ok({ available: true, version }, src)
  } catch (e) {
    return fail(src, `NVIDIA telemetry unavailable: ${(e as Error).message}`)
  }
}

export function parseScan(json: string): Omit<SystemProfile, 'cuda' | 'runtimes' | 'scannedAt'> {
  const raw = JSON.parse(json) as RawScan
  return {
    os: section(raw.os, 'Win32_OperatingSystem', (o) => ({ name: o.Caption.trim(), version: o.Version, build: String(o.BuildNumber) })),
    cpu: section(raw.cpu, 'Win32_Processor', (c) => {
      const cpus = arr(c)
      if (!cpus.length) throw new Error('no processors returned')
      return {
        model: cpus[0].Name.trim(),
        physicalCores: cpus.reduce((n, p) => n + p.NumberOfCores, 0),
        logicalCores: cpus.reduce((n, p) => n + p.NumberOfLogicalProcessors, 0)
      }
    }),
    ram: section(raw.os, 'Win32_OperatingSystem TotalVisibleMemorySize/FreePhysicalMemory', (o) => ({
      totalBytes: o.TotalVisibleMemorySize * 1024,
      availableBytes: o.FreePhysicalMemory * 1024
    })),
    gpus: buildGpus(raw),
    disks: section(raw.disks, 'Win32_LogicalDisk (DriveType=3)', (d) =>
      arr(d).map((x): DiskInfo => ({ mount: x.DeviceID, totalBytes: x.Size, freeBytes: x.FreeSpace }))
    )
  }
}

/** Scan the machine. Never throws: failures surface as status 'unavailable' with an error. */
export async function scanSystem(deps: ScanDeps = defaultDeps): Promise<SystemProfile> {
  const scannedAt = new Date().toISOString()
  let parsed: ReturnType<typeof parseScan>
  try {
    parsed = parseScan(await deps.powershell(SCAN_SCRIPT))
  } catch (e) {
    const err = `PowerShell scan failed: ${(e as Error).message}`
    const f = <T>(s: string) => fail<T>(s, err)
    parsed = { os: f('Win32_OperatingSystem'), cpu: f('Win32_Processor'), ram: f('Win32_OperatingSystem'), gpus: f('Win32_VideoController'), disks: f('Win32_LogicalDisk') }
  }
  const cuda = await detectCuda(parsed.gpus, deps)
  return { scannedAt, ...parsed, cuda, runtimes: [] }
}
