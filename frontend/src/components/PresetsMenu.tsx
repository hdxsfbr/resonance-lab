import { useRef, useState } from 'react'
import type { ConditionName, Perturbation } from '../api/types'
import { useLab } from '../state/LabContext'
import { CONDITION_DEFS, PERTURBATION_DEFS } from '../lib/definitions'
import { fmt, humanise } from '../lib/format'
import { KindTag } from './KindTag'

function scalar(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : fmt(v, 3)
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (v === null || v === undefined) return '—'
  return String(v)
}

/** Readable rendering of preset comparison values (numbers, arrays, nested dicts). */
function ValueView({ k, v }: { k: string; v: unknown }) {
  if (Array.isArray(v)) {
    const probs = /prob/i.test(k)
    return <span className="mono">{v.map((x, i) => (probs ? `#${i} ${scalar(x)}` : scalar(x))).join(probs ? ' · ' : ', ')}</span>
  }
  if (v && typeof v === 'object') {
    return (
      <dl className="kv kv--nested">
        {Object.entries(v as Record<string, unknown>).map(([kk, vv]) => (
          <div key={kk} className="kv__pair">
            <dt>{kk}</dt>
            <dd>
              <ValueView k={kk} v={vv} />
            </dd>
          </div>
        ))}
      </dl>
    )
  }
  return <span className="mono">{scalar(v)}</span>
}

function ComparisonTable() {
  const { state } = useLab()
  const r = state.presetResult
  if (!r) return null
  const comp = r.comparison as Record<string, unknown>
  const labels = (typeof comp.labels === 'object' && comp.labels !== null ? comp.labels : {}) as Record<string, string>
  const entries = Object.entries(comp).filter(([k]) => k !== 'labels')
  return (
    <div className="comparison" data-testid="preset-comparison">
      <h4>
        Comparison · {r.preset} · seed {r.seed} <KindTag kind="measured" />
      </h4>
      <table className="kvtable">
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <th scope="row" title={k}>
                {labels[k] ?? humanise(k)}
              </th>
              <td>
                <ValueView k={k} v={v} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {r.sessions.length > 0 && (
        <p className="tiny muted">
          Sessions: {r.sessions.map((s) => `${s.id} (${s.condition.name}, ${s.step} steps)`).join(' · ')}
        </p>
      )}
      {r.narrative && (
        <blockquote className="narrative">
          <KindTag kind="generated" /> {r.narrative}
          <br />
          <span className="tiny muted">Generated or templated explanation — not a measurement.</span>
        </blockquote>
      )}
    </div>
  )
}

export function PresetsMenu() {
  const { state, dispatch, actions } = useLab()
  const [seed, setSeed] = useState<string>('')
  const [cond, setCond] = useState<ConditionName>('full')
  const [pert, setPert] = useState<Perturbation>('rhythm_shuffle')
  const [customSeed, setCustomSeed] = useState(7)
  const fileRef = useRef<HTMLInputElement | null>(null)
  if (!state.startMenuOpen) return null
  const seedNum = seed.trim() === '' ? null : Number(seed)
  return (
    <div className="modal" role="dialog" aria-label="Start menu" data-testid="start-menu">
      <div className="modal__card">
        <div className="modal__head">
          <h2>Resonance Lab · start</h2>
          <span className="spacer" />
          {state.session && (
            <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'startMenu', open: false })} aria-label="Close start menu">
              ✕
            </button>
          )}
        </div>
        <p className="small muted">
          Two engineered agents exchange short phrases in a cooperative timing task. Choose a preset, start a custom session, or import a recorded run.
          {state.source.kind === 'mock' && <strong> Data source: DEMO DATA (mock) — {state.source.reason}.</strong>}
        </p>
        <div className="presets">
          {state.presets.map((p) => (
            <article key={p.name} className="preset" data-testid={`preset-${p.name}`}>
              <h3>{p.title}</h3>
              <p className="small">{p.summary}</p>
              <dl className="kv">
                <dt>manipulation</dt>
                <dd>{p.manipulation}</dd>
                <dt>evidence criterion</dt>
                <dd>{p.evidence_criterion}</dd>
                {p.constructed_note && (
                  <>
                    <dt>constructed</dt>
                    <dd className="warn">{p.constructed_note}</dd>
                  </>
                )}
                <dt>default</dt>
                <dd>
                  {p.default_episodes} episodes · condition {p.condition.name}
                  {p.condition.perturbation !== 'none' ? ` (${p.condition.perturbation})` : ''}
                </dd>
              </dl>
              <div className="row">
                <button type="button" className="btn btn--primary btn--small" disabled={state.busy} onClick={() => void actions.startSession({ preset: p.name })} data-testid={`start-${p.name}`}>
                  Start live session
                </button>
                <button type="button" className="btn btn--small" disabled={state.busy} onClick={() => void actions.runPresetComparison(p.name, seedNum)}>
                  Run preset comparison
                </button>
              </div>
            </article>
          ))}
          {!state.presets.length && <p className="muted">No presets available (GET /api/presets).</p>}
        </div>
        <label className="small">
          seed for comparisons <input className="input input--num" value={seed} placeholder="default" onChange={(e) => setSeed(e.target.value)} />
        </label>
        {state.busy && <p className="small muted">Working…</p>}
        <ComparisonTable />
        <div className="custom">
          <h4>Custom session</h4>
          <div className="row wrap">
            <select className="input" value={cond} onChange={(e) => setCond(e.target.value as ConditionName)} aria-label="Condition">
              {CONDITION_DEFS.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
            {cond === 'perturbed' && (
              <select className="input" value={pert} onChange={(e) => setPert(e.target.value as Perturbation)} aria-label="Perturbation">
                {PERTURBATION_DEFS.filter((p) => p.kind !== 'none').map((p) => (
                  <option key={p.kind}>{p.kind}</option>
                ))}
              </select>
            )}
            <label className="small">
              seed <input className="input input--num" type="number" value={customSeed} onChange={(e) => setCustomSeed(Number(e.target.value) || 0)} />
            </label>
            <button
              type="button"
              className="btn btn--small"
              disabled={state.busy}
              onClick={() =>
                void actions.startSession({
                  condition: { name: cond, perturbation: cond === 'perturbed' ? pert : 'none', description: CONDITION_DEFS.find((c) => c.name === cond)?.summary ?? '' },
                  config: state.defaultConfig ? { ...state.defaultConfig, seed: customSeed } : null,
                })
              }
            >
              Start custom session
            </button>
            <button type="button" className="btn btn--small" onClick={() => fileRef.current?.click()}>
              ⤒ Import run JSON (replay)
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="application/json,.json"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void actions.importRun(f)
                e.target.value = ''
              }}
            />
          </div>
          <p className="tiny muted">{CONDITION_DEFS.find((c) => c.name === cond)?.summary}</p>
        </div>
      </div>
    </div>
  )
}
