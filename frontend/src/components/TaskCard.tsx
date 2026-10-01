import { useLab } from '../state/LabContext'
import { focusedEpisode } from '../state/store'
import { MOCK_SCORING_RULE, SCORING_RULE } from '../lib/definitions'
import { pct, fmt } from '../lib/format'
import { InfoTip } from './InfoTip'
import { PatternGrid } from './PatternGrid'
import { ScoreChip } from './ScoreChip'

export function TaskCard() {
  const { state } = useLab()
  const ep = focusedEpisode(state)
  const m = state.session?.metrics
  const agentName = (id: string | null) => state.session?.agents.find((a) => a.id === id)?.name ?? id ?? '—'
  return (
    <section className="panel task" aria-label="Current task" data-testid="task-card">
      <div className="task__head">
        <h3 className="panel__title">
          Task · step {ep ? ep.step : '—'}
          {state.selectedStep !== null && <span className="pill">selected</span>}
        </h3>
        <span className="muted small">
          {ep ? (
            <>
              {agentName(ep.senderId)} → {agentName(ep.receiverId)}
            </>
          ) : (
            'no episode yet'
          )}
        </span>
        <span className="task__score">
          score <ScoreChip score={ep?.score ?? null} />
          <InfoTip label="scoring rule" wide>
            <strong>Scoring rule</strong>
            <br />
            {SCORING_RULE}
            {state.source.kind === 'mock' && (
              <>
                <br />
                {MOCK_SCORING_RULE}
              </>
            )}
          </InfoTip>
        </span>
      </div>
      <div className="task__patterns">
        {state.patterns.map((p) => {
          const isTarget = ep?.targetId === p.id
          const isChosen = ep?.chosenId === p.id
          return (
            <div key={p.id} className={`taskpat${isTarget ? ' taskpat--target' : ''}${isChosen ? ' taskpat--chosen' : ''}`}>
              <div className="taskpat__name" title={`#${p.id} ${p.name}: onsets ${p.onsets.join(', ')}`}>
                <span className="mono">#{p.id}</span> {p.name}
              </div>
              <PatternGrid pattern={p} size={7} />
              <div className="taskpat__tags">
                {isTarget && <span className="tag tag--target" title="Shown to the experimenter only; the receiver never sees it">target (private to sender)</span>}
                {isChosen && <span className="tag tag--chosen">receiver's choice</span>}
              </div>
            </div>
          )
        })}
      </div>
      <div className="metrics" aria-label="Running metrics">
        <div className="metric">
          <span className="metric__k">episodes</span>
          <span className="metric__v" data-testid="episode-counter">
            {m?.episodes ?? 0}
            <span className="muted small"> / {state.episodesTarget}</span>
          </span>
        </div>
        <div className="metric">
          <span className="metric__k">mean score</span>
          <span className="metric__v">{fmt(m?.mean_score, 3)}</span>
        </div>
        <div className="metric">
          <span className="metric__k">rolling (last 20)</span>
          <span className="metric__v">{fmt(m?.rolling_score, 3)}</span>
        </div>
        <div className="metric">
          <span className="metric__k">success rate</span>
          <span className="metric__v">{pct(m?.success_rate, 1)}</span>
        </div>
      </div>
    </section>
  )
}
