import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import type { RendererApi } from '../../shared/types'
import { BenchmarkPage } from './BenchmarkPage'
import { DashboardPage } from './DashboardPage'
import { ModelsPage } from './ModelsPage'
import { ResultsPage } from './ResultsPage'
import { SystemPage } from './SystemPage'
import './styles.css'

declare global {
  interface Window { api: RendererApi }
}

const SECTIONS = ['Dashboard', 'Benchmark', 'Models', 'Results', 'System'] as const
type Section = (typeof SECTIONS)[number]

function App() {
  const [section, setSection] = useState<Section>('Dashboard')
  const [sessionId, setSessionId] = useState<number | undefined>()
  const go = (s: Section, id?: number) => { setSection(s); setSessionId(id) }
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
        {section === 'Benchmark' && <BenchmarkPage />}
        {section === 'Models' && <ModelsPage />}
        {section === 'Results' && <ResultsPage key={sessionId ?? 'none'} sessionId={sessionId} />}
        {section === 'System' && <SystemPage />}
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
