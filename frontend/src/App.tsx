import { useState } from 'react'
import { LabProvider, useLab } from './state/LabContext'
import { Header } from './components/Header'
import { ControlsBar } from './components/ControlsBar'
import { AgentPanel } from './components/AgentPanel'
import { TaskCard } from './components/TaskCard'
import { PianoRollPanel } from './components/PianoRoll'
import { Timeline } from './components/Timeline'
import { TrajectoryPlot } from './components/TrajectoryPlot'
import { ScorePlot } from './components/ScorePlot'
import { EventLog, InterventionsPanel } from './components/InterventionsPanel'
import { PhraseEditor } from './components/PhraseEditor'
import { Inspector } from './components/Inspector'
import { ExperimentsView } from './components/ExperimentsView'
import { PresetsMenu } from './components/PresetsMenu'
import { Footer } from './components/Footer'
import { ErrorBanner } from './components/ErrorBanner'
import './App.css'

type SideTab = 'interventions' | 'compose' | 'log'

function Lab() {
  const { state } = useLab()
  const [tab, setTab] = useState<SideTab>('interventions')
  const agents = state.session?.agents ?? []
  return (
    <div className="app">
      <Header />
      <ControlsBar />
      <ErrorBanner />
      <main className="lab">
        <div className="lab__a">
          <AgentPanel agent={agents[0]} side="left" />
        </div>
        <div className="lab__center">
          <TaskCard />
          <PianoRollPanel />
          <div className="lab__plots">
            <TrajectoryPlot />
            <ScorePlot />
          </div>
          <Timeline />
        </div>
        <div className="lab__b">
          <AgentPanel agent={agents[1]} side="right" />
        </div>
        <aside className="lab__side">
          <div className="tabs" role="tablist" aria-label="Side panel">
            {(
              [
                ['interventions', 'Interventions'],
                ['compose', 'Compose'],
                ['log', `Log (${state.log.length})`],
              ] as const
            ).map(([k, label]) => (
              <button key={k} type="button" role="tab" aria-selected={tab === k} className={`tab${tab === k ? ' tab--on' : ''}`} onClick={() => setTab(k)}>
                {label}
              </button>
            ))}
          </div>
          <div className="lab__sidebody">
            {tab === 'interventions' && <InterventionsPanel />}
            {tab === 'compose' && <PhraseEditor />}
            {tab === 'log' && <EventLog />}
          </div>
        </aside>
      </main>
      <Footer />
      <Inspector />
      <ExperimentsView />
      <PresetsMenu />
    </div>
  )
}

export default function App() {
  return (
    <LabProvider>
      <Lab />
    </LabProvider>
  )
}
