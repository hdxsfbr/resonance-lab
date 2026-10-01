import { useEffect, useState } from 'react'
import type { ConditionAggregate, ConditionName, ConditionSpec, ExperimentResult, Perturbation } from '../api/types'
import { useLab } from '../state/LabContext'
import { CONDITION_DEFS, EXPERIMENT_CAVEAT, PERTURBATION_DEFS } from '../lib/definitions'
import { downloadJson, downloadText, safeName } from '../lib/download'
import { errorMessage, fmt } from '../lib/format'
import { mean, sd } from '../lib/stats'

function condLabel(a: { condition: string; perturbation: string }) {
  return a.perturbation && a.perturbation !== 'none' ? `${a.condition} · ${a.perturbation}` : a.condition
}

/** Per-condition dot plot: per-run points + mean ± sd (one series; no legend needed). */
function DotPlot({ rows, title, domain = [0, 1] }: { rows: { label: string; values: number[] }[]; title: string; domain?: [number, number] }) {
  const [hover, setHover] = useState<string | null>(null)
  const labelW = 170
  const W = 520
  const rowH = 30
  const H = rows.length * rowH + 30
  const sx = (v: number) => labelW + ((v - domain[0]) / (domain[1] - domain[0])) * (W - labelW - 16)
  return (
    <figure className="dotplot">
      <figcaption className="small">{title}</figcaption>
      <svg width={W} height={H} role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = domain[0] + t * (domain[1] - domain[0])
          return (
            <g key={t}>
              <line x1={sx(v)} x2={sx(v)} y1={4} y2={H - 22} className="gridline" />
              <text x={sx(v)} y={H - 8} className="axislabel" textAnchor="middle">
                {fmt(v, 2)}
              </text>
            </g>
          )
        })}
        {rows.map((r, i) => {
          const y = 14 + i * rowH
          const m = mean(r.values)
          const s = sd(r.values)
          return (
            <g key={r.label} onMouseEnter={() => setHover(r.label)} onMouseLeave={() => setHover(null)}>
              <text x={labelW - 8} y={y + 4} className="axislabel axislabel--strong" textAnchor="end">
                {r.label}
              </text>
              <line x1={sx(Math.max(domain[0], m - s))} x2={sx(Math.min(domain[1], m + s))} y1={y} y2={y} className="errbar" />
              {r.values.map((v, j) => (
                <circle key={j} cx={sx(v)} cy={y + ((j % 3) - 1) * 4} r={3.5} className="rundot">
                  <title>{`${r.label} run ${j + 1}: ${fmt(v, 3)}`}</title>
                </circle>
              ))}
              <rect x={sx(m) - 1.5} y={y - 8} width={3} height={16} rx={1} className="meanmark" />
              {hover === r.label && (
                <text x={W - 10} y={y - 9} className="axislabel" textAnchor="end">
                  mean {fmt(m, 3)} ± {fmt(s, 3)} (n={r.values.length})
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

/** Small multiples: one learning curve per condition, mean line + ±sd band, shared axes. */
function CurveMultiples({ aggs }: { aggs: ConditionAggregate[] }) {
  const W = 170
  const H = 110
  const pad = { l: 26, r: 6, t: 16, b: 18 }
  return (
    <div className="multiples">
      {aggs.map((a) => {
        const n = a.mean_learning_curve.length
        const sx = (i: number) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - pad.l - pad.r))
        const sy = (v: number) => pad.t + (1 - Math.max(0, Math.min(1, v))) * (H - pad.t - pad.b)
        const upper = a.mean_learning_curve.map((m, i) => `${sx(i)},${sy(m + (a.sd_learning_curve[i] ?? 0))}`)
        const lower = a.mean_learning_curve.map((m, i) => `${sx(i)},${sy(m - (a.sd_learning_curve[i] ?? 0))}`).reverse()
        return (
          <figure key={condLabel(a)} className="multiple">
            <svg width={W} height={H} role="img" aria-label={`Learning curve ${condLabel(a)}`}>
              <text x={pad.l} y={11} className="axislabel axislabel--strong">
                {condLabel(a)} (n={a.n_runs})
              </text>
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={pad.l} x2={W - pad.r} y1={sy(v)} y2={sy(v)} className={v === 0 ? 'axisline' : 'gridline'} />
                  <text x={pad.l - 3} y={sy(v) + 3} className="axislabel" textAnchor="end">
                    {v}
                  </text>
                </g>
              ))}
              <polygon points={[...upper, ...lower].join(' ')} className="band" />
              <polyline points={a.mean_learning_curve.map((m, i) => `${sx(i)},${sy(m)}`).join(' ')} className="curve" />
              <text x={(W + pad.l) / 2} y={H - 4} className="axislabel" textAnchor="middle">
                block 1–{n}
              </text>
            </svg>
          </figure>
        )
      })}
    </div>
  )
}

export function ExperimentsView() {
  const { state, dispatch, api } = useLab()
  const [selected, setSelected] = useState<Record<ConditionName, boolean>>({ full: true, no_history: true, state_fixed: true, state_decoupled: true, symbol: true, perturbed: true })
  const [perturbation, setPerturbation] = useState<Perturbation>('rhythm_shuffle')
  const [seeds, setSeeds] = useState('1,2,3,4,5')
  const [episodes, setEpisodes] = useState(120)
  const [name, setName] = useState('batch')
  const [running, setRunning] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [result, setResult] = useState<ExperimentResult | null>(null)
  const [previous, setPrevious] = useState<ExperimentResult[]>([])

  useEffect(() => {
    if (!state.experimentsOpen || !api) return
    api
      .listExperiments()
      .then(setPrevious)
      .catch(() => setPrevious([]))
  }, [state.experimentsOpen, api, result])

  useEffect(() => {
    if (!state.experimentsOpen) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && dispatch({ type: 'experiments', open: false })
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [state.experimentsOpen, dispatch])

  if (!state.experimentsOpen) return null

  const run = async () => {
    if (!api) return
    const seedList = seeds
      .split(/[\s,]+/)
      .map((x) => parseInt(x, 10))
      .filter((x) => Number.isFinite(x))
    const conditions: ConditionSpec[] = CONDITION_DEFS.filter((c) => selected[c.name]).map((c) => ({
      name: c.name,
      perturbation: c.name === 'perturbed' ? perturbation : 'none',
      description: c.summary,
    }))
    if (!seedList.length || !conditions.length) {
      setErr('Choose at least one condition and one seed.')
      return
    }
    setRunning(true)
    setErr(null)
    try {
      const base = state.defaultConfig
      const config = base ? { ...base, episodes } : null
      const r = await api.runExperiment({ name, conditions, seeds: seedList, config })
      setResult(r)
    } catch (e) {
      setErr(errorMessage(e))
    } finally {
      setRunning(false)
    }
  }

  const csv = async () => {
    if (!api || !result) return
    try {
      downloadText(await api.experimentCsv(result.id), `experiment-${safeName(result.id)}.csv`, 'text/csv')
    } catch (e) {
      setErr(errorMessage(e))
    }
  }

  const aggs = result?.aggregates ?? []
  return (
    <div className="modal" role="dialog" aria-label="Experiments" data-testid="experiments-view">
      <div className="modal__card modal__card--wide">
        <div className="modal__head">
          <h2>Experiments (batch, no audio)</h2>
          {state.source.kind === 'mock' && <span className="badge badge--demo">DEMO DATA (mock)</span>}
          <span className="spacer" />
          <button type="button" className="btn btn--small" onClick={() => dispatch({ type: 'experiments', open: false })} aria-label="Close experiments">
            ✕
          </button>
        </div>
        <p className="caveat" data-testid="experiment-caveat">
          {EXPERIMENT_CAVEAT}
        </p>
        <div className="exp">
          <form
            className="exp__form"
            onSubmit={(e) => {
              e.preventDefault()
              void run()
            }}
          >
            <fieldset>
              <legend>Conditions</legend>
              {CONDITION_DEFS.map((c) => (
                <label key={c.name} className="cond">
                  <input type="checkbox" checked={selected[c.name]} onChange={(e) => setSelected({ ...selected, [c.name]: e.target.checked })} />
                  <span>
                    <strong className="mono">{c.name}</strong>
                    <br />
                    <span className="small muted">{c.summary}</span>
                    {c.name === 'perturbed' && (
                      <select className="input" value={perturbation} onChange={(e) => setPerturbation(e.target.value as Perturbation)} aria-label="Perturbation kind">
                        {PERTURBATION_DEFS.filter((p) => p.kind !== 'none').map((p) => (
                          <option key={p.kind} value={p.kind}>
                            {p.kind} — {p.description}
                          </option>
                        ))}
                      </select>
                    )}
                  </span>
                </label>
              ))}
            </fieldset>
            <label className="field">
              seeds (comma list)
              <input className="input" value={seeds} onChange={(e) => setSeeds(e.target.value)} />
            </label>
            <label className="field">
              episodes per run
              <input className="input input--num" type="number" min={10} max={2000} value={episodes} onChange={(e) => setEpisodes(Math.max(10, Number(e.target.value) || 120))} />
            </label>
            <label className="field">
              name
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
            </label>
            <button type="submit" className="btn btn--primary" disabled={running} data-testid="run-experiment">
              {running ? 'Running…' : 'Run experiment'}
            </button>
            {err && <p className="warn small">{err}</p>}
            {previous.length > 0 && (
              <label className="field">
                previous experiments
                <select className="input" value={result?.id ?? ''} onChange={(e) => setResult(previous.find((p) => p.id === e.target.value) ?? null)}>
                  <option value="">—</option>
                  {previous.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.id} · {p.name} · {p.runs.length} runs
                    </option>
                  ))}
                </select>
              </label>
            )}
          </form>
          <div className="exp__results">
            {!result && <p className="muted">Run an experiment to see per-condition results. Each run = one seed × one condition.</p>}
            {result && (
              <>
                <div className="row">
                  <strong>
                    {result.name} · {result.id}
                  </strong>
                  <span className="muted small">
                    {result.runs.length} runs · {result.config.episodes} episodes each
                  </span>
                  <span className="spacer" />
                  <button type="button" className="btn btn--small" onClick={() => void csv()}>
                    ⤓ CSV
                  </button>
                  <button type="button" className="btn btn--small" onClick={() => downloadJson(result, `experiment-${safeName(result.id)}.json`)}>
                    ⤓ JSON
                  </button>
                </div>
                <table className="kvtable" data-testid="experiment-table">
                  <thead>
                    <tr>
                      <th>condition</th>
                      <th>n runs</th>
                      <th>final-block score (mean ± sd)</th>
                      <th>success rate (mean ± sd)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {aggs.map((a) => (
                      <tr key={condLabel(a)}>
                        <td className="mono">{condLabel(a)}</td>
                        <td className="mono">{a.n_runs}</td>
                        <td className="mono">
                          {fmt(a.mean_final_block_score, 3)} ± {fmt(a.sd_final_block_score, 3)}
                        </td>
                        <td className="mono">
                          {fmt(a.mean_success_rate, 3)} ± {fmt(a.sd_success_rate, 3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <DotPlot title="Final-block score per run (dots), mean (bar) ± sd (line)" rows={aggs.map((a) => ({ label: condLabel(a), values: a.per_run_final_block }))} />
                <DotPlot
                  title="Success rate per run (dots), mean (bar) ± sd (line)"
                  rows={aggs.map((a) => ({
                    label: condLabel(a),
                    values: result.runs.filter((r) => r.condition === a.condition && (r.perturbation ?? 'none') === (a.perturbation ?? 'none')).map((r) => r.success_rate),
                  }))}
                />
                <h4>Learning curves (block means, mean ± sd band across runs)</h4>
                <CurveMultiples aggs={aggs} />
                {result.notes && result.notes.length > 0 && (
                  <div className="notes">
                    <h4>Notes returned with the result</h4>
                    <ul>
                      {result.notes.map((n, i) => (
                        <li key={i} className="small">
                          {n}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
