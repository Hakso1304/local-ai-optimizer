import { StrictMode, useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import type { RendererApi } from '../../shared/types'
import { BenchmarkPage } from './BenchmarkPage'
import { applyEvent, initialLive } from './benchState'
import { DashboardPage } from './DashboardPage'
import { HubPage } from './HubPage'
import type { BenchPreset } from './LargeCodingCard'
import { ModelsPage } from './ModelsPage'
import { ResultsPage } from './ResultsPage'
import { SystemPage } from './SystemPage'
import './styles.css'

declare global {
  interface Window { api: RendererApi }
}

const SECTIONS = ['Dashboard', 'Benchmark', 'Models', 'Download', 'Results', 'System'] as const
type Section = (typeof SECTIONS)[number]

function App() {
  const [section, setSection] = useState<Section>('Dashboard')
  const [sessionId, setSessionId] = useState<number | undefined>()
  const [preset, setPreset] = useState<BenchPreset | undefined>()
  const go = useCallback((s: Section, id?: number, p?: BenchPreset) => { setSection(s); setSessionId(id); setPreset(p) }, [])
  // Live benchmark state lives here so leaving the Benchmark page mid-run doesn't lose it.
  const [live, dispatch] = useReducer(applyEvent, initialLive)
  useEffect(() => window.api.onBenchEvent(dispatch), [])
  // Open Results only on an observed running → done transition of the session being watched.
  const prev = useRef(live.status)
  useEffect(() => {
    if (prev.current === 'running' && live.status === 'done' && live.sessionId) go('Results', Number(live.sessionId))
    prev.current = live.status
  }, [live.status, live.sessionId, go])
  return (
    <div className="app">
      <nav>
        <div className="brand">LOCAL AI OPTIMIZER</div>
        {SECTIONS.map((s) => (
          <button key={s} className={s === section ? 'active' : ''} onClick={() => go(s)}>{s}</button>
        ))}
      </nav>
      <main>
        {section === 'Dashboard' && <DashboardPage go={go} />}
        {section === 'Benchmark' && <BenchmarkPage key={preset ? 'preset' : 'plain'} live={live} preset={preset} onDownload={() => go('Download')} />}
        {section === 'Models' && <ModelsPage />}
        {section === 'Download' && <HubPage />}
        {section === 'Results' && <ResultsPage key={sessionId ?? 'none'} sessionId={sessionId} />}
        {section === 'System' && <SystemPage />}
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
