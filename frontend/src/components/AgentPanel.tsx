import type { AgentSnapshot, StateDim } from '../api/types'
import { useLab } from '../state/LabContext'
import { focusedEpisode } from '../state/store'
import { STATE_DIMS } from '../lib/definitions'
import { KindTag } from './KindTag'
import { ProbBars } from './ProbBars'
import { StateBars } from './StateBars'

export function AgentPanel({ agent, side }: { agent: AgentSnapshot | undefined; side: 'left' | 'right' }) {
  const { state, dispatch } = useLab()
  if (!agent) {
    return (
      <section className={`panel agent agent--${side} agent--empty`}>
        <p className="muted">No session. Choose a preset from the start menu.</p>
      </section>
    )
  }
  const ep = focusedEpisode(state)
  const role = ep ? (ep.senderId === agent.id && !ep.human ? 'sender' : ep.receiverId === agent.id ? 'receiver' : 'idle') : '—'
  const change = ep?.stateChanges[agent.id]
  const isLatest = state.selectedStep === null
  const shownState = isLatest ? agent.state : (change?.after ?? agent.state)
  const delta = change
    ? (Object.fromEntries(STATE_DIMS.map((d) => [d, change.after[d] - change.before[d]])) as Record<StateDim, number>)
    : undefined
  const trace = ep ? (role === 'sender' ? ep.senderTrace : role === 'receiver' ? ep.receiverTrace : null) : agent.last_trace ?? null
  const shownTrace = trace ?? agent.last_trace ?? null
  const labels =
    shownTrace?.role === 'receiver'
      ? state.patterns.map((p) => p.name)
      : Array.from({ length: shownTrace?.probabilities.length ?? 0 }, (_, i) => `m${i}`)
  const condFrozen = state.session?.condition.name === 'state_fixed'
  const condDecoupled = state.session?.condition.name === 'state_decoupled'
  return (
    <section className={`panel agent agent--${side}`} style={{ ['--agent' as string]: agent.color }} data-testid={`agent-panel-${agent.id}`} aria-label={`Agent ${agent.name}`}>
      <div className="agent__head">
        <span className="agent__dot" style={{ background: agent.color }} aria-hidden />
        <h2 className="agent__name">
          {agent.name} <span className="muted">({agent.id})</span>
        </h2>
        <span className={`role role--${role}`}>{role}</span>
      </div>
      <div className="agent__sub">
        <span>instrument: {agent.instrument}</span>
        <span>memory: {agent.memory_size}</span>
        <span>learner updates: {agent.learner.updates}</span>
      </div>
      <div className="agent__flags">
        <span className={`flag${agent.frozen_state ? ' flag--on' : ''}`} title={condFrozen ? 'state_fixed condition holds state at baseline' : 'freeze_state intervention'}>
          state {agent.frozen_state ? 'FROZEN' : 'updating'}
        </span>
        <span className={`flag${!agent.coupling_enabled ? ' flag--on' : ''}`} title={condDecoupled ? 'state_decoupled condition' : 'set_coupling intervention'}>
          coupling {agent.coupling_enabled ? 'on' : 'OFF'}
        </span>
        <span className="flag" title="Which policy chooses actions">
          policy: {agent.policy_kind}
        </span>
      </div>
      <div className="agent__section">
        <div className="section-title">
          State <KindTag kind="engineered" />
          <span className="muted small">{isLatest ? 'current' : `after step ${ep?.step}`}</span>
        </div>
        <StateBars state={shownState} baseline={agent.baseline} delta={delta} />
        <p className="tiny muted">right: Δ in the shown step · white tick = baseline</p>
      </div>
      <div className="agent__section">
        <div className="section-title">
          Policy {shownTrace ? `· ${shownTrace.role} (${shownTrace.role === 'receiver' ? 'per pattern' : 'per motif'})` : ''}
          <span className="muted small">{shownTrace ? `τ ${shownTrace.temperature.toFixed(2)}` : ''}</span>
        </div>
        {shownTrace ? <ProbBars trace={shownTrace} labels={labels} color={agent.color} /> : <p className="muted small">No decision yet.</p>}
      </div>
      <button type="button" className="btn btn--block" onClick={() => dispatch({ type: 'inspector', open: true, agentId: agent.id })} data-testid={`inspect-${agent.id}`}>
        Inspect {agent.name}
      </button>
    </section>
  )
}
