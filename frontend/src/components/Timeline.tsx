import { Fragment, useMemo, useState } from 'react'
import { senderLabel, type InterventionRecord } from '../api/events'
import { useLab } from '../state/LabContext'
import { INTERVENTION_DEFS } from '../lib/definitions'
import type { InterventionKind } from '../api/types'
import { MiniRoll } from './MiniRoll'
import { ScoreChip } from './ScoreChip'

const PAGE = 120

function InterventionMarker({ iv }: { iv: InterventionRecord }) {
  const def = INTERVENTION_DEFS[iv.kind as InterventionKind]
  return (
    <div className="tl-marker" role="note">
      <span className="tl-marker__icon" aria-hidden>
        ⚑
      </span>
      <span>
        {def?.label ?? iv.kind}
        {iv.agentId ? ` [${iv.agentId}]` : ''} — effective from step {iv.effectiveFromStep}
        {iv.effect ? ` (${iv.effect})` : ''}
      </span>
      {iv.description && <span className="muted"> · {iv.description}</span>}
    </div>
  )
}

export function Timeline() {
  const { state, dispatch, actions } = useLab()
  const [limit, setLimit] = useState(PAGE)
  const live = !!state.session && state.session.mode !== 'replay'
  const name = (id: string | null) => state.session?.agents.find((a) => a.id === id)?.name ?? id ?? '—'
  const byStep = useMemo(() => {
    const m = new Map<number, InterventionRecord[]>()
    for (const iv of state.interventions) m.set(iv.effectiveFromStep, [...(m.get(iv.effectiveFromStep) ?? []), iv])
    return m
  }, [state.interventions])
  const rows = state.episodes.slice(-limit).reverse()
  const latestStep = state.episodes.length ? state.episodes[state.episodes.length - 1].step : -1
  const pending = state.interventions.filter((iv) => iv.effectiveFromStep > latestStep)
  const patName = (id: number | null) => (id === null ? '—' : `#${id}`)
  return (
    <section className="panel timeline" aria-label="Timeline of exchanges">
      <div className="panel__bar">
        <h3 className="panel__title">Timeline</h3>
        <span className="muted small">
          {state.episodes.length} episodes{state.episodes.length > limit ? ` · latest ${limit}` : ''} · click a row to select · ↻ = play{live ? ' + queue as next sent phrase' : ' audio only'}
        </span>
        <span className="spacer" />
        {state.selectedStep !== null && (
          <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'select', step: null })}>
            Follow latest
          </button>
        )}
      </div>
      <div className="timeline__scroll" data-testid="timeline">
        <table className="tl">
          <thead>
            <tr>
              <th>step</th>
              <th>exchange</th>
              <th>motif</th>
              <th>phrase</th>
              <th title="receiver's chosen pattern / target pattern (target private to sender)">chosen / target</th>
              <th>score</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {pending.map((iv) => (
              <tr key={`p${iv.seq}`} className="tl__markerrow">
                <td colSpan={7}>
                  <InterventionMarker iv={iv} />
                </td>
              </tr>
            ))}
            {rows.map((ep) => {
              const sel = state.selectedStep === ep.step
              const ivs = byStep.get(ep.step) ?? []
              return (
                <Fragment key={ep.step}>
                  <tr
                    className={`tl__row${sel ? ' tl__row--sel' : ''}`}
                    onClick={() => dispatch({ type: 'select', step: sel ? null : ep.step })}
                    aria-selected={sel}
                    data-testid="timeline-row"
                  >
                    <td className="mono">{ep.step}</td>
                    <td>
                      {senderLabel(ep, name)} → {name(ep.receiverId)}
                    </td>
                    <td className="mono small">{ep.motifId ?? (ep.motifIndex !== null ? `m${ep.motifIndex}` : '—')}</td>
                    <td>{ep.phrase ? <MiniRoll phrase={ep.phrase} /> : '—'}</td>
                    <td className="mono">
                      <span className={ep.chosenId !== null && ep.chosenId === ep.targetId ? 'match' : ''}>
                        {patName(ep.chosenId)} / {patName(ep.targetId)}
                      </span>
                    </td>
                    <td>
                      <ScoreChip score={ep.score} />
                    </td>
                    <td>
                      {ep.phrase && (
                        <button
                          type="button"
                          className="btn btn--tiny"
                          onClick={(e) => {
                            e.stopPropagation()
                            void actions.replayMotif(ep)
                          }}
                          title={live ? 'Plays audio locally AND posts replay_motif: the receiver receives this phrase at the next step' : 'Replay mode: plays audio locally only'}
                        >
                          ↻ {live ? 'Replay' : 'Play'}
                        </button>
                      )}
                    </td>
                  </tr>
                  {ivs.map((iv) => (
                    <tr key={`iv${iv.seq}`} className="tl__markerrow">
                      <td colSpan={7}>
                        <InterventionMarker iv={iv} />
                      </td>
                    </tr>
                  ))}
                </Fragment>
              )
            })}
          </tbody>
        </table>
        {state.episodes.length > limit && (
          <button type="button" className="btn btn--small tl__more" onClick={() => setLimit((l) => l + PAGE)}>
            Show older
          </button>
        )}
        {!state.episodes.length && <p className="muted empty">No exchanges yet.</p>}
      </div>
    </section>
  )
}
