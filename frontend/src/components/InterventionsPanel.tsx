import { useEffect, useState } from 'react'
import type { Perturbation, StateDim } from '../api/types'
import { useLab } from '../state/LabContext'
import { focusedEpisode } from '../state/store'
import { INTERVENTION_DEFS, PERTURBATION_DEFS, STATE_DIMS } from '../lib/definitions'
import { fmt } from '../lib/format'

function Effect({ when }: { when: 'next step' | 'immediately' }) {
  return <span className={`effect effect--${when === 'immediately' ? 'now' : 'next'}`}>takes effect: {when}</span>
}

function ParamSlider({ label, path, value, min, max, step, disabled, hint }: { label: string; path: string; value: number; min: number; max: number; step: number; disabled: boolean; hint?: string }) {
  const { actions } = useLab()
  const [v, setV] = useState(value)
  const [dirty, setDirty] = useState(false)
  useEffect(() => {
    if (!dirty) setV(value)
  }, [value, dirty])
  const commit = () => {
    if (!dirty) return
    setDirty(false)
    void actions.intervene({ kind: 'set_param', path, value: Number(v.toFixed(4)) })
  }
  return (
    <label className="slider" title={`set_param ${path}${hint ? ` — ${hint}` : ''}`}>
      <span className="slider__label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={v}
        disabled={disabled}
        onChange={(e) => {
          setV(Number(e.target.value))
          setDirty(true)
        }}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
      />
      <span className="mono slider__val">{fmt(v, 2)}</span>
    </label>
  )
}

export function InterventionsPanel() {
  const { state, actions } = useLab()
  const s = state.session
  const agents = s?.agents ?? []
  const disabled = !s || s.mode === 'replay'
  const [memAgent, setMemAgent] = useState<string>('A')
  const [stateAgent, setStateAgent] = useState<string>('A')
  const [transform, setTransform] = useState<Perturbation>('none')
  const ep = focusedEpisode(state)
  // '*' = both agents: the contract's agent_id null applies to all agents
  const target = (id: string) => (id === '*' ? null : agents.some((a) => a.id === id) ? id : (agents[0]?.id ?? null))
  const D = INTERVENTION_DEFS
  return (
    <section className="panel interventions" aria-label="Interventions">
      <h3 className="panel__title">Interventions</h3>
      {s?.mode === 'replay' && <p className="note">Replay mode: recorded events are re-emitted; interventions are disabled.</p>}
      <div className="iv">
        <div className="iv__head">
          <strong>{D.replay_motif.label}</strong> <Effect when={D.replay_motif.takesEffect} />
        </div>
        <p className="iv__sem">{D.replay_motif.semantics} Also plays it locally.</p>
        <button type="button" className="btn btn--small" disabled={disabled || !ep?.phrase} onClick={() => ep && void actions.replayMotif(ep)}>
          Replay {ep ? `step ${ep.step}` : 'selected'} motif
        </button>
      </div>
      <div className="iv">
        <div className="iv__head">
          <strong>{D.reset_memory.label}</strong> <Effect when={D.reset_memory.takesEffect} />
        </div>
        <p className="iv__sem">{D.reset_memory.semantics}</p>
        <div className="row">
          <select className="input" value={memAgent === '*' ? '*' : (target(memAgent) ?? '*')} onChange={(e) => setMemAgent(e.target.value)} aria-label="Agent for clear memory">
            {agents.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} ({a.id})
              </option>
            ))}
            <option value="*">both (agent_id null)</option>
          </select>
          <button type="button" className="btn btn--small" disabled={disabled} onClick={() => void actions.intervene({ kind: 'reset_memory', agent_id: target(memAgent) })}>
            Clear memory
          </button>
        </div>
      </div>
      <div className="iv">
        <div className="iv__head">
          <strong>{D.reset_state.label}</strong> <Effect when={D.reset_state.takesEffect} />
        </div>
        <p className="iv__sem">{D.reset_state.semantics}</p>
        <div className="row">
          <select className="input" value={stateAgent === '*' ? '*' : (target(stateAgent) ?? '*')} onChange={(e) => setStateAgent(e.target.value)} aria-label="Agent for reset state">
            {agents.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} ({a.id})
              </option>
            ))}
            <option value="*">both (agent_id null)</option>
          </select>
          <button type="button" className="btn btn--small" disabled={disabled} onClick={() => void actions.intervene({ kind: 'reset_state', agent_id: target(stateAgent) })}>
            Reset state
          </button>
        </div>
      </div>
      <div className="iv">
        <div className="iv__head">
          <strong>{D.freeze_state.label}</strong> / <strong>{D.set_coupling.label}</strong> <Effect when="next step" />
        </div>
        <p className="iv__sem">
          Freeze: {D.freeze_state.semantics} Coupling off: {D.set_coupling.semantics}
        </p>
        {agents.map((a) => (
          <div className="row" key={a.id}>
            <span className="agentchip" style={{ ['--agent' as string]: a.color }}>
              {a.name}
            </span>
            <label className="check small">
              <input type="checkbox" disabled={disabled} checked={a.frozen_state} onChange={(e) => void actions.intervene({ kind: 'freeze_state', agent_id: a.id, value: e.target.checked })} /> freeze state
            </label>
            <label className="check small">
              <input type="checkbox" disabled={disabled} checked={!a.coupling_enabled} onChange={(e) => void actions.intervene({ kind: 'set_coupling', agent_id: a.id, value: !e.target.checked })} /> disable coupling
            </label>
          </div>
        ))}
      </div>
      <div className="iv">
        <div className="iv__head">
          <strong>{D.swap_feature.label}</strong> <Effect when={D.swap_feature.takesEffect} />
        </div>
        <p className="iv__sem">{D.swap_feature.semantics}</p>
        <div className="row">
          <select className="input" value={transform} onChange={(e) => setTransform(e.target.value as Perturbation)} aria-label="Transform">
            {PERTURBATION_DEFS.map((p) => (
              <option key={p.kind} value={p.kind}>
                {p.kind}
                {p.preservesMessage === 'yes' ? ' — preserves task message' : p.preservesMessage === 'no' ? ' — destroys task message' : p.preservesMessage === 'partly' ? ' — partly destroys message' : ''}
              </option>
            ))}
          </select>
          <button type="button" className="btn btn--small" disabled={disabled} onClick={() => void actions.intervene({ kind: 'swap_feature', transform })}>
            Apply
          </button>
        </div>
        <p className="tiny muted">{PERTURBATION_DEFS.find((p) => p.kind === transform)?.description}</p>
      </div>
      <div className="iv">
        <div className="iv__head">
          <strong>Parameters</strong> <Effect when="next step" />
        </div>
        <p className="iv__sem">{D.set_param.semantics} Applied on release.</p>
        {agents.map((a, i) => (
          <ParamSlider key={a.id} label={`${a.name} sensitivity`} path={`agents.${i}.sensitivity`} value={a.params.sensitivity} min={0} max={3} step={0.05} disabled={disabled} hint="multiplier on all state drives" />
        ))}
        {STATE_DIMS.map((d: StateDim) => (
          <ParamSlider key={d} label={`decay ${d}`} path={`state.decay.${d}`} value={s?.config.state?.decay?.[d] ?? 0} min={0} max={0.5} step={0.005} disabled={disabled} hint="per-step pull toward baseline" />
        ))}
      </div>
    </section>
  )
}

export function EventLog() {
  const { state } = useLab()
  const items = [...state.log].reverse()
  return (
    <section className="panel eventlog" aria-label="Event log">
      <h3 className="panel__title">Event log</h3>
      <ol className="eventlog__list" data-testid="event-log">
        {items.map((l) => (
          <li key={l.id} className={`eventlog__item eventlog__item--${l.kind}`}>
            <span className="mono tiny muted">{l.at.slice(11, 19)}</span> <span className="eventlog__kind">{l.kind}</span> {l.text}
            {l.kind === 'intervention' && l.step !== undefined && (
              <span className="effect effect--next">
                effective from step {l.step}
                {l.effect ? ` (${l.effect})` : ''}
              </span>
            )}
          </li>
        ))}
        {!items.length && <li className="muted">No events yet.</li>}
      </ol>
    </section>
  )
}
