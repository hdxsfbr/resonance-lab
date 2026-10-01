import { useMemo, useState } from 'react'
import type { StateDim } from '../api/types'
import type { StatePoint } from '../api/events'
import { useLab } from '../state/LabContext'
import { STATE_DEFS, STATE_DIMS } from '../lib/definitions'
import { DIM_COLORS } from '../lib/colors'
import { fmt } from '../lib/format'
import { useSize } from '../lib/useSize'
import { INTERVENTION_DEFS } from '../lib/definitions'
import type { InterventionKind } from '../api/types'
import { KindTag } from './KindTag'

const PAD = { l: 34, r: 8, t: 6, b: 24 }

function nearest(points: StatePoint[], x: number): StatePoint | null {
  let best: StatePoint | null = null
  for (const p of points) if (p.step <= x && (!best || p.step > best.step)) best = p
  return best
}

export function TrajectoryPlot() {
  const { state } = useLab()
  const [dims, setDims] = useState<Record<StateDim, boolean>>({ activation: true, expected_value: true, uncertainty: true, affiliation: true })
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [ref, size] = useSize<HTMLDivElement>({ width: 520, height: 170 })
  const agents = state.session?.agents ?? []
  const points = state.statePoints
  const maxX = useMemo(() => Math.max(10, ...Object.values(points).flatMap((ps) => ps.map((p) => p.step))), [points])
  const W = Math.max(200, size.width)
  const H = Math.max(60, size.height)
  const iw = W - PAD.l - PAD.r
  const ih = H - PAD.t - PAD.b
  const sx = (x: number) => PAD.l + (x / maxX) * iw
  const sy = (v: number) => PAD.t + ((1 - v) / 2) * ih
  const xTicks = useMemo(() => {
    const step = maxX <= 20 ? 5 : maxX <= 60 ? 10 : maxX <= 200 ? 25 : maxX <= 600 ? 100 : 250
    return Array.from({ length: Math.floor(maxX / step) + 1 }, (_, i) => i * step)
  }, [maxX])
  const off = state.stepOffset
  const selX = state.selectedStep !== null ? state.selectedStep + off : null
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const x = ((e.clientX - r.left - PAD.l) / iw) * maxX
    setHoverX(x < 0 || x > maxX ? null : Math.round(x))
  }
  return (
    <section className="panel plot" aria-label="State trajectories">
      <div className="panel__bar">
        <h3 className="panel__title">
          State trajectories <KindTag kind="engineered" />
        </h3>
        <span className="legend">
          {STATE_DIMS.map((d) => (
            <label key={d} className="legend__item" title={STATE_DEFS[d].definition}>
              <input type="checkbox" checked={dims[d]} onChange={(e) => setDims({ ...dims, [d]: e.target.checked })} />
              <span className="legend__line" style={{ background: DIM_COLORS[d] }} aria-hidden />
              {d}
            </label>
          ))}
          {agents.map((a, i) => (
            <span key={a.id} className="legend__item">
              <svg width={22} height={8} aria-hidden>
                <line x1={0} x2={22} y1={4} y2={4} className="legend__agentline" strokeDasharray={i === 0 ? undefined : '5 3'} />
              </svg>
              {a.name}
            </span>
          ))}
          <span className="legend__item">
            <svg width={22} height={8} aria-hidden>
              <line x1={0} x2={22} y1={4} y2={4} className="legend__agentline" strokeDasharray="1 3" />
            </svg>
            baseline
          </span>
        </span>
      </div>
      <div className="plot__body" ref={ref}>
        <svg width={W} height={H} onMouseMove={onMove} onMouseLeave={() => setHoverX(null)} role="img" aria-label="State value per episode, A solid, B dashed" data-testid="trajectory-plot">
          {[-1, -0.5, 0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={W - PAD.r} y1={sy(v)} y2={sy(v)} className={v === 0 ? 'axisline' : 'gridline'} />
              <text x={PAD.l - 5} y={sy(v) + 3} className="axislabel" textAnchor="end">
                {v}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={sx(t)} y={H - PAD.b + 13} className="axislabel" textAnchor="middle">
              {t}
            </text>
          ))}
          <text x={PAD.l + iw / 2} y={H - 2} className="axistitle" textAnchor="middle">
            episodes completed
          </text>
          <text x={10} y={PAD.t + ih / 2} className="axistitle" textAnchor="middle" transform={`rotate(-90 10 ${PAD.t + ih / 2})`}>
            value
          </text>
          {state.interventions.map((iv) => {
            const x = sx(Math.min(maxX, iv.effectiveFromStep + off - 1))
            const def = INTERVENTION_DEFS[iv.kind as InterventionKind]
            return (
              <g key={iv.seq}>
                <line x1={x} x2={x} y1={PAD.t} y2={PAD.t + ih} className="ivline" />
                <text x={x + 2} y={PAD.t + 9} className="ivlabel">
                  ⚑<title>{`${def?.label ?? iv.kind}${iv.agentId ? ` [${iv.agentId}]` : ''}: effective from step ${iv.effectiveFromStep}`}</title>
                </text>
              </g>
            )
          })}
          {selX !== null && <line x1={sx(selX)} x2={sx(selX)} y1={PAD.t} y2={PAD.t + ih} className="selline" />}
          {agents.map((a, ai) =>
            STATE_DIMS.filter((d) => dims[d]).map((d) => {
              const ps = points[a.id] ?? []
              const base = a.baseline[d]
              return (
                <g key={`${a.id}-${d}`}>
                  {(ai === 0 || agents[0].baseline[d] !== base) && (
                    <line x1={PAD.l} x2={W - PAD.r} y1={sy(base)} y2={sy(base)} stroke={DIM_COLORS[d]} strokeWidth={1} strokeDasharray="1 3" opacity={0.6} />
                  )}
                  <polyline
                    points={ps.map((p) => `${sx(p.step)},${sy(p.state[d])}`).join(' ')}
                    fill="none"
                    stroke={DIM_COLORS[d]}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    strokeDasharray={ai === 0 ? undefined : '5 3'}
                  />
                </g>
              )
            }),
          )}
          {hoverX !== null && (
            <line x1={sx(hoverX)} x2={sx(hoverX)} y1={PAD.t} y2={PAD.t + ih} className="crosshair" />
          )}
        </svg>
        {hoverX !== null && (
          <div className="tooltip" style={{ left: Math.min(sx(hoverX) + 12, W - 190), top: 8 }}>
            <div className="tooltip__title">after {hoverX} episodes</div>
            {agents.map((a) => {
              const p = nearest(points[a.id] ?? [], hoverX)
              return (
                <div key={a.id} className="tooltip__row">
                  <strong>{a.name}</strong>{' '}
                  {p
                    ? STATE_DIMS.filter((d) => dims[d])
                        .map((d) => `${d.slice(0, 3)} ${fmt(p.state[d], 2)}`)
                        .join(' · ')
                    : '—'}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}
