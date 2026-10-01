import { useEffect, useState } from 'react'
import type { InstalledRuntime, Sourced, Status, SystemProfile } from '../../shared/types'

const gb = (b: number) => `${(b / 1024 ** 3).toFixed(1)} GiB`

function Badge({ status }: { status: Status }) {
  return <span className={`badge ${status}`}>{status.toUpperCase()}</span>
}

function Row({ label, s, fmt }: { label: string; s: Sourced<unknown>; fmt: () => string }) {
  return (
    <tr>
      <td>{label}</td>
      <td><Badge status={s.status} /></td>
      <td>{s.value != null ? fmt() : <span className="err">{s.error ?? '—'}</span>}</td>
      <td className="muted">{s.source}</td>
    </tr>
  )
}

function InstallRuntime({ onDone, hip = false, prism = false }: { onDone: () => void; hip?: boolean; prism?: boolean }) {
  const [busy, setBusy] = useState(false)
  const [lines, setLines] = useState<string[]>([])
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => window.api.onRuntimeProgress((m) => setLines((l) => [...l.slice(-20), m])), [])
  const go = () => {
    setBusy(true)
    setErr(null)
    ;(prism ? window.api.installPrismRuntime() : hip ? window.api.installHipRuntime() : window.api.installRuntime()).then(onDone, (e: Error) => setErr(e.message)).finally(() => setBusy(false))
  }
  return (
    <div className="card">
      {prism
        ? <p>Optional: the PrismML llama.cpp fork (Vulkan Windows build, ~31 MB) from github.com/PrismML-Eng/llama.cpp. It is the only runtime that loads ternary PQ2_0 / PTQ1_0 models (Bonsai 2); mainline llama.cpp rejects them. Installed next to the main build and used only for those models.</p>
        : hip
        ? <p>Optional: the official llama.cpp ROCm (HIP) build for AMD GPUs (~245 MB download, ~1.2 GB installed), next to the Vulkan build and at the same release. With both installed, benchmarks can compare the two backends on this GPU. It needs no HIP SDK, but whether your driver exposes this GPU to HIP is only known after the first run.</p>
        : <p>llama.cpp is not installed. The app downloads the official Windows Vulkan build from github.com/ggml-org/llama.cpp (~30 MB).</p>}
      <div className="bar"><button onClick={go} disabled={busy}>{busy ? 'Installing…' : prism ? 'Install PrismML ternary runtime' : hip ? 'Install ROCm (HIP) runtime' : 'Install llama.cpp runtime'}</button></div>
      {err && <p className="err">{err}</p>}
      {lines.length > 0 && <pre className="log">{lines.join('\n')}</pre>}
    </div>
  )
}

export function SystemPage() {
  const [p, setP] = useState<SystemProfile | null>(null)
  const [backends, setBackends] = useState<InstalledRuntime[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const scan = () => {
    setBusy(true)
    setErr(null)
    window.api.scanSystem().then(setP, (e: Error) => setErr(e.message)).finally(() => setBusy(false))
    window.api.installedBackends().then(setBackends, () => setBackends([]))
  }
  const amd = !!p?.gpus.value?.some((g) => g.vendor === 'amd' && !g.isIntegrated)
  const primaryOk = backends.some((b) => b.kind !== 'hip' && b.kind !== 'prism' && b.status === 'available')
  const hipOk = backends.some((b) => b.kind === 'hip' && b.status === 'available')
  const prismOk = backends.some((b) => b.kind === 'prism' && b.status === 'available')
  useEffect(scan, [])

  return (
    <section>
      <header className="bar">
        <h1>System</h1>
        <button onClick={scan} disabled={busy}>{busy ? 'Scanning…' : 'Rescan'}</button>
        {p && <span className="muted">scanned {new Date(p.scannedAt).toLocaleString()}</span>}
      </header>
      {err && <p className="err">{err}</p>}
      {p && (
        <>
          <table>
            <thead><tr><th>Item</th><th>Status</th><th>Value</th><th>Source</th></tr></thead>
            <tbody>
              <Row label="OS" s={p.os} fmt={() => `${p.os.value!.name} ${p.os.value!.version} (build ${p.os.value!.build})`} />
              <Row label="CPU" s={p.cpu} fmt={() => `${p.cpu.value!.model} — ${p.cpu.value!.physicalCores}C/${p.cpu.value!.logicalCores}T`} />
              <Row label="RAM" s={p.ram} fmt={() => `${gb(p.ram.value!.totalBytes)} total, ${gb(p.ram.value!.availableBytes)} available`} />
              <tr>
                <td>GPU temperature / power</td>
                <td><Badge status={p.nvidiaSmi?.available ? 'available' : 'unavailable'} /></td>
                <td>{p.nvidiaSmi?.available ? `NVIDIA via nvidia-smi${p.nvidiaSmi.cudaVersion ? ` (driver CUDA ${p.nvidiaSmi.cudaVersion})` : ''}`
                  : <span className="err">{p.gpus.value?.some((g) => g.vendor === 'amd' && !g.isIntegrated) ? 'AMD: no non-admin source (needs the ADLX native SDK)' : p.nvidiaSmi?.reason ?? 'no source'}</span>}</td>
                <td className="muted">nvidia-smi</td>
              </tr>
              <Row label="CUDA" s={p.cuda} fmt={() => (p.cuda.value!.available ? `yes, ${p.cuda.value!.driverCudaVersion}` : 'no')} />
              {p.gpus.value?.map((g) => (
                <Row key={g.pnpDeviceId} label={`GPU (${g.vendor}${g.isIntegrated ? ', integrated?' : ''})`} s={g.dedicatedVramBytes}
                  fmt={() => `${g.name} — ${gb(g.dedicatedVramBytes.value!)} VRAM, driver ${g.driverVersion ?? '?'}`} />
              )) ?? <Row label="GPUs" s={p.gpus} fmt={() => ''} />}
              {p.disks.value?.map((d) => (
                <Row key={d.mount} label={`Disk ${d.mount}`} s={p.disks} fmt={() => `${gb(d.freeBytes)} free of ${gb(d.totalBytes)}`} />
              )) ?? <Row label="Disks" s={p.disks} fmt={() => ''} />}
            </tbody>
          </table>
          <h2>Runtimes</h2>
          {p.runtimes.find((r) => r.id === 'llamacpp')?.status !== 'available' && <InstallRuntime onDone={scan} />}
          {amd && primaryOk && !hipOk && <InstallRuntime onDone={scan} hip />}
          {primaryOk && !prismOk && <InstallRuntime onDone={scan} prism />}
          {backends.length > 0 && (
            <>
              <h3>llama.cpp backends</h3>
              <table>
                <thead><tr><th>Backend</th><th>Status</th><th>Build</th><th>Directory</th></tr></thead>
                <tbody>
                  {backends.map((b) => (
                    <tr key={b.kind}>
                      <td>{b.kind === 'hip' ? 'ROCm (HIP)' : b.kind === 'cuda' ? 'CUDA' : b.kind === 'prism' ? 'PrismML ternary (Vulkan)' : 'Vulkan'}</td>
                      <td><Badge status={b.status} /></td>
                      <td>{b.build ?? <span className={b.kind === 'hip' || b.kind === 'prism' ? 'muted' : 'err'}>{(b.kind === 'hip' || b.kind === 'prism') && b.error?.startsWith('llama-server.exe not found') ? 'not installed (optional)' : b.error ?? '—'}</span>}</td>
                      <td className="muted">{b.vendorDir}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          <table>
            <thead><tr><th>Runtime</th><th>Status</th><th>Detail</th><th>Source</th></tr></thead>
            <tbody>
              {p.runtimes.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td><Badge status={r.status} /></td>
                  <td>{r.status === 'available'
                    ? [r.version, r.path, r.models && `${r.models.length} models`].filter(Boolean).join(' — ')
                    : <span className="err">{r.error}</span>}</td>
                  <td className="muted">{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  )
}
