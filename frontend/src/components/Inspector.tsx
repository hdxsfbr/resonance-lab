import { useEffect, useState } from 'react'
import type { AgentInspection, Observation, PolicyTrace, StateUpdateInputs } from '../api/types'
import { useLab } from '../state/LabContext'
import { focusedEpisode } from '../state/store'
import type { EpisodeRecord } from '../api/events'
import { narrativeText } from '../api/events'
import { FEATURE_NAMES, STATE_DEFS, STATE_DIMS, HAND_AUTHORED_ACOUSTIC } from '../lib/definitions'
import { errorMessage, fmt, signed } from '../lib/format'
import { Heatmap } from './Heatmap'
import { KindTag } from './KindTag'
import { InfoTip } from './InfoTip'

function containsTarget(x: unknown): boolean {
  return JSON.stringify(x ?? null).toLowerCase().includes('target')
}

function StateChangeView({ ep, agentId, fallback }: { ep: EpisodeRecord | null; agentId: string; fallback: StateUpdateInputs | null | undefined }) {
  const ch = ep?.stateChanges[agentId]
  const inputs = ch?.inputs ?? (ep ? null : fallback) ?? null
  return (
    <div className="insp__block">
      <h4>
        State before → after <KindTag kind="engineered" /> {ep && <span className="muted small">step {ep.step}</span>}
      </h4>
      {ch ? (
        <table className="kvtable">
          <thead>
            <tr>
              <th>variable</th>
              <th>before</th>
              <th>after</th>
              <th>Δ</th>
            </tr>
          </thead>
          <tbody>
            {STATE_DIMS.map((d) => (
              <tr key={d}>
                <td>
                  {d}{' '}
                  <InfoTip label={`${d} definition`}>{STATE_DEFS[d].definition}</InfoTip>
                </td>
                <td className="mono">{fmt(ch.before[d], 3)}</td>
                <td className="mono">{fmt(ch.after[d], 3)}</td>
                <td className="mono">{signed(ch.after[d] - ch.before[d], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted small">No state_update recorded for this agent in this step.</p>
      )}
      <h5>Inputs that caused it</h5>
      {inputs ? (
        <dl className="kv">
          <dt>
            music drive <KindTag kind="hand" />
          </dt>
          <dd className="mono">
            {Object.entries(inputs.music_drive ?? {}).length
              ? Object.entries(inputs.music_drive ?? {})
                  .map(([k, v]) => `${k} ${signed(v, 3)}`)
                  .join(' · ')
              : '0 (disabled or no acoustic input)'}
            <InfoTip label="hand-authored">{HAND_AUTHORED_ACOUSTIC}</InfoTip>
          </dd>
          <dt>prediction error (score − expected_value)</dt>
          <dd className="mono">{signed(inputs.prediction_error ?? null, 3)}</dd>
          <dt>outcome (score)</dt>
          <dd className="mono">{fmt(inputs.outcome ?? null, 3)}</dd>
          <dt>partner similarity</dt>
          <dd className="mono">{fmt(inputs.partner_similarity ?? null, 3)}</dd>
          <dt>decay applied</dt>
          <dd>{inputs.decay_applied ? 'yes' : 'no'}</dd>
          <dt>frozen</dt>
          <dd>{inputs.frozen ? 'yes — update skipped' : 'no'}</dd>
          {inputs.notes && inputs.notes.length > 0 && (
            <>
              <dt>notes</dt>
              <dd>{inputs.notes.join('; ')}</dd>
            </>
          )}
        </dl>
      ) : (
        <p className="muted small">No inputs recorded.</p>
      )}
    </div>
  )
}

function PolicyView({ trace, labels }: { trace: PolicyTrace | null; labels: string[] }) {
  if (!trace) return <p className="muted small">This agent made no decision in the selected step.</p>
  return (
    <div>
      <dl className="kv kv--inline">
        <dt>role</dt>
        <dd>{trace.role}</dd>
        <dt>policy</dt>
        <dd>{trace.policy_kind}</dd>
        <dt>τ base → effective</dt>
        <dd className="mono">
          {fmt(trace.temperature_base, 3)} → {fmt(trace.temperature, 3)}
        </dd>
        <dt>coupling</dt>
        <dd>{trace.coupling_enabled ? 'on (state modulates τ / expression / η)' : 'off (τ = τ_base)'}</dd>
        <dt>draw</dt>
        <dd className="mono">{fmt(trace.exploration_draw ?? null, 3)}</dd>
      </dl>
      <table className="kvtable">
        <thead>
          <tr>
            <th>{trace.role === 'receiver' ? 'pattern' : 'motif'}</th>
            <th>score</th>
            <th>probability</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {trace.scores.map((sc, i) => (
            <tr key={i} className={i === trace.chosen ? 'row--chosen' : ''}>
              <td>{labels[i] ?? i}</td>
              <td className="mono">{fmt(sc, 3)}</td>
              <td className="mono">
                <span className="pbar" style={{ width: `${Math.round((trace.probabilities[i] ?? 0) * 60)}px` }} aria-hidden /> {fmt(trace.probabilities[i], 3)}
              </td>
              <td>{i === trace.chosen ? 'chosen' : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {Object.keys(trace.expressive_modulation ?? {}).length > 0 && (
        <p className="small">
          expressive modulation applied:{' '}
          <span className="mono">
            {Object.entries(trace.expressive_modulation ?? {})
              .map(([k, v]) => `${k} ${fmt(v, 3)}`)
              .join(' · ')}
          </span>
        </p>
      )}
      {trace.notes && trace.notes.length > 0 && <p className="small muted">{trace.notes.join('; ')}</p>}
    </div>
  )
}

function ObservationView({ obs }: { obs: Observation | null }) {
  if (!obs) return <p className="muted small">This agent received nothing in the selected step (it was the sender).</p>
  const leak = containsTarget(obs)
  return (
    <div>
      <p className={`small ${leak ? 'warn' : 'ok'}`} data-testid="isolation-check">
        {leak ? '⚠ the observation mentions "target" — information isolation violated' : '✓ no target field in what the receiver saw'}
      </p>
      <dl className="kv kv--inline">
        <dt>channel</dt>
        <dd>{obs.channel}</dd>
        <dt>sender</dt>
        <dd>{obs.sender_id}</dd>
        <dt>step</dt>
        <dd>{obs.step}</dd>
        {obs.channel === 'symbol' && (
          <>
            <dt>symbol id</dt>
            <dd className="mono">{obs.symbol_id ?? '—'}</dd>
          </>
        )}
      </dl>
      {obs.features && (
        <table className="kvtable kvtable--compact">
          <thead>
            <tr>
              <th>feature (φ)</th>
              <th>value</th>
            </tr>
          </thead>
          <tbody>
            {FEATURE_NAMES.map((f, i) => (
              <tr key={f}>
                <td>{f}</td>
                <td className="mono">{fmt(obs.features?.vector[i] ?? null, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <details>
        <summary className="small">Raw Observation JSON</summary>
        <pre className="json">{JSON.stringify(obs, null, 1)}</pre>
      </details>
    </div>
  )
}

function ModelCalls({ ep, agentId }: { ep: EpisodeRecord | null; agentId: string }) {
  const calls = (ep?.modelCalls ?? []).filter((e) => !e.agent_id || e.agent_id === agentId)
  const narratives = ep?.narratives ?? []
  return (
    <div className="insp__block insp__block--generated">
      <h4>
        Model request / response <KindTag kind="generated" />
      </h4>
      <p className="tiny">GENERATED — model output, not a measurement. Never read by the state update or the scoring.</p>
      {calls.length === 0 && <p className="muted small">No model calls for this agent in this step (policy is local).</p>}
      {calls.map((c) => {
        const p = (c.payload ?? {}) as {
          request?: { input_modality?: string; system_prompt?: string; user_prompt?: string; purpose?: string }
          response?: { ok?: boolean; parsed?: unknown; latency_ms?: number; error?: string | null; provider?: string; model_id?: string; content?: string }
        }
        return (
          <div key={c.seq} className="modelcall">
            <dl className="kv kv--inline">
              <dt>purpose</dt>
              <dd>{p.request?.purpose ?? '—'}</dd>
              <dt>input modality</dt>
              <dd>{p.request?.input_modality ?? '—'}</dd>
              <dt>provider</dt>
              <dd>
                {p.response?.provider ?? '—'} {p.response?.model_id ?? ''}
              </dd>
              <dt>latency</dt>
              <dd className="mono">{p.response?.latency_ms !== undefined ? `${fmt(p.response.latency_ms, 0)} ms` : '—'}</dd>
              <dt>ok</dt>
              <dd>{p.response?.ok === undefined ? '—' : p.response.ok ? 'yes' : 'no'}</dd>
              {p.response?.error && (
                <>
                  <dt>error</dt>
                  <dd className="warn">{p.response.error}</dd>
                </>
              )}
            </dl>
            <details>
              <summary className="small">Prompt text</summary>
              <pre className="json">{`${p.request?.system_prompt ?? ''}\n\n${p.request?.user_prompt ?? ''}`}</pre>
            </details>
            <details open>
              <summary className="small">Parsed JSON</summary>
              <pre className="json">{JSON.stringify(p.response?.parsed ?? null, null, 1)}</pre>
            </details>
          </div>
        )
      })}
      {narratives.map((n) => (
        <blockquote key={n.seq} className="narrative">
          <KindTag kind="generated" /> {narrativeText(n)}
        </blockquote>
      ))}
    </div>
  )
}

export function Inspector() {
  const { state, dispatch, api } = useLab()
  const { open, agentId } = state.inspector
  const s = state.session
  const [data, setData] = useState<AgentInspection | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const ep = focusedEpisode(state)
  const step = s?.step
  useEffect(() => {
    if (!open || !s || !agentId || !api) return
    let cancelled = false
    api
      .inspect(s.id, agentId)
      .then((d) => {
        if (!cancelled) {
          setData(d)
          setErr(null)
        }
      })
      .catch((e) => !cancelled && setErr(errorMessage(e)))
    return () => {
      cancelled = true
    }
  }, [open, agentId, api, s?.id, step, s])
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && dispatch({ type: 'inspector', open: false })
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, dispatch])
  if (!open || !s || !agentId) return null
  const agent = s.agents.find((a) => a.id === agentId)
  if (!agent) return null
  const role = ep ? (ep.senderId === agentId ? 'sender' : ep.receiverId === agentId ? 'receiver' : null) : null
  const trace = role === 'sender' ? (ep?.senderTrace ?? null) : role === 'receiver' ? (ep?.receiverTrace ?? null) : null
  const patternNames = state.patterns.map((p) => `#${p.id} ${p.name}`)
  const motifNames = Array.from({ length: agent.learner.sender_values[0]?.length ?? 0 }, (_, i) => `m${i}`)
  const featureCols = [...FEATURE_NAMES.slice(0, agent.learner.receiver_weights[0]?.length ?? 16), 'bias']
  const weights = agent.learner.receiver_weights.map((row, i) => [...row, agent.learner.receiver_bias[i] ?? 0])
  return (
    <aside className="drawer" role="dialog" aria-label={`Inspector for ${agent.name}`} data-testid="inspector">
      <div className="drawer__head">
        <h3>
          Inspector · <span style={{ color: 'var(--text)' }}>{agent.name}</span> <span className="muted">({agent.id})</span>
        </h3>
        <span className="seg" role="group" aria-label="Agent">
          {s.agents.map((a) => (
            <button key={a.id} type="button" className={`seg__btn${a.id === agentId ? ' seg__btn--on' : ''}`} onClick={() => dispatch({ type: 'inspector', open: true, agentId: a.id })}>
              {a.name}
            </button>
          ))}
        </span>
        <span className="spacer" />
        <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'inspector', open: false })} aria-label="Close inspector">
          ✕
        </button>
      </div>
      <div className="drawer__body">
        <p className="small muted">
          Step {ep ? ep.step : '—'} {state.selectedStep === null ? '(latest)' : '(selected in timeline)'} · role: {role ?? '—'}
          {err && <span className="warn"> · inspect failed: {err}</span>}
        </p>
        <StateChangeView ep={ep} agentId={agentId} fallback={data?.last_state_inputs} />
        <div className="insp__block">
          <h4>
            Policy: scores → probabilities <KindTag kind="engineered" />
          </h4>
          <PolicyView trace={trace} labels={role === 'receiver' ? patternNames : motifNames} />
        </div>
        <div className="insp__block">
          <h4>
            Information actually received <KindTag kind="measured" />
          </h4>
          <ObservationView obs={role === 'receiver' ? (ep?.observation ?? null) : null} />
        </div>
        <div className="insp__block">
          <h4>
            Retrieved memories <KindTag kind="learned" />
          </h4>
          <p className="tiny muted">For the agent's most recent observation (GET /agents/{agentId}/inspect), not necessarily the selected step.</p>
          {data?.retrieved && data.retrieved.length > 0 ? (
            <table className="kvtable">
              <thead>
                <tr>
                  <th>similarity</th>
                  <th>step</th>
                  <th>role</th>
                  <th>action</th>
                  <th>score</th>
                </tr>
              </thead>
              <tbody>
                {data.retrieved.map((r, i) => (
                  <tr key={i}>
                    <td className="mono">{fmt(r.similarity, 3)}</td>
                    <td className="mono">{r.item.step}</td>
                    <td>{r.item.role}</td>
                    <td className="mono">{r.item.role === 'receiver' ? `pattern #${r.item.action}` : `motif m${r.item.action}`}</td>
                    <td className="mono">{fmt(r.item.score, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted small">None (memory empty or not yet queried).</p>
          )}
          <p className="tiny muted">Memory size: {agent.memory_size} items.</p>
        </div>
        <div className="insp__block">
          <h4>
            Learned associations <KindTag kind="learned" />
          </h4>
          <p className="tiny muted">Current values after {agent.learner.updates} updates (not historical).</p>
          <Heatmap rows={patternNames} cols={featureCols} values={weights} mode="diverging" caption="Receiver weights W[pattern][feature] (+ bias)" cell={14} />
          <Heatmap
            rows={patternNames}
            cols={motifNames}
            values={agent.learner.sender_values}
            counts={agent.learner.sender_counts}
            mode="sequential"
            domain={[0, 1]}
            caption="Sender values Q[target][motif] (expected score)"
            showValues
            cell={22}
          />
        </div>
        <ModelCalls ep={ep} agentId={agentId} />
        {data && data.information_received && data.information_received.length > 0 && (
          <details className="insp__block">
            <summary>Information-received log (last {Math.min(50, data.information_received.length)})</summary>
            <pre className="json">{JSON.stringify(data.information_received.slice(-10), null, 1)}</pre>
          </details>
        )}
      </div>
    </aside>
  )
}
