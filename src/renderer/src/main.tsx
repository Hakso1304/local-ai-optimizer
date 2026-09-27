import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import type { RendererApi } from '../../shared/types'
import { SystemPage } from './SystemPage'
import './styles.css'

declare global {
  interface Window { api: RendererApi }
}

const SECTIONS = ['Dashboard', 'Benchmark', 'Models', 'Results', 'System'] as const
type Section = (typeof SECTIONS)[number]

function App() {
  const [section, setSection] = useState<Section>('System')
  return (
    <div className="app">
      <nav>
        <div className="brand">LOCAL AI OPTIMIZER</div>
        {SECTIONS.map((s) => (
          <button key={s} className={s === section ? 'active' : ''} onClick={() => setSection(s)}>{s}</button>
        ))}
      </nav>
      <main>{section === 'System' ? <SystemPage /> : <p className="muted">{section}: not implemented yet.</p>}</main>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
