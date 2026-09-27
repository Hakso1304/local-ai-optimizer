import { useEffect, useState } from 'react'
import type { Sourced, Status, SystemProfile } from '../../shared/types'

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

export function SystemPage() {
  const [p, setP] = useState<SystemProfile | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const scan = () => {
    setBusy(true)
    setErr(null)
    window.api.scanSystem().then(setP, (e: Error) => setErr(e.message)).finally(() => setBusy(false))
  }
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
