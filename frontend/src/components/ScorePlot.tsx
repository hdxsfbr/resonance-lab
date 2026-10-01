import { useMemo, useState } from 'react'
import { useLab } from '../state/LabContext'
import { rollingMean } from '../lib/stats'
import { fmt } from '../lib/format'
import { useSize } from '../lib/useSize'

const PAD = { l: 30, r: 8, t: 8, b: 24 }

export function ScorePlot() {
  const { state } = useLab()
  const history = state.session?.metrics.score_history
  const scores = useMemo(() => history ?? [], [history])
  const roll = useMemo(() => rollingMean(scores, 20), [scores])
  const [hover, setHover] = useState<number | null>(null)
  const [ref, size] = useSize<HTMLDivElement>({ width: 300, height: 170 })
  const W = Math.max(160, size.width)
  const H = Math.max(60, size.height)
  const n = Math.max(10, scores.length)
  const iw = W - PAD.l - PAD.r
  const ih = H - PAD.t - PAD.b
  const sx = (i: number) => PAD.l + ((i + 1) / n) * iw
  const sy = (v: number) => PAD.t + (1 - v) * ih
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const i = Math.round(((e.clientX - r.left - PAD.l) / iw) * n) - 1
    setHover(i >= 0 && i < scores.length ? i : null)
  }
  return (
    <section className="panel plot plot--score" aria-label="Score over episodes">
      <div className="panel__bar">
        <h3 className="panel__title">Score</h3>
        <span className="legend">
          <span className="legend__item">
            <span className="legend__dot" aria-hidden /> per episode
          </span>
          <span className="legend__item">
            <span className="legend__line legend__line--accent" aria-hidden /> rolling 20
          </span>
        </span>
      </div>
      <div className="plot__body" ref={ref}>
        <svg width={W} height={H} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label="Score per episode with rolling mean" data-testid="score-plot">
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={W - PAD.r} y1={sy(v)} y2={sy(v)} className={v === 0 ? 'axisline' : 'gridline'} />
              <text x={PAD.l - 4} y={sy(v) + 3} className="axislabel" textAnchor="end">
                {v}
              </text>
            </g>
          ))}
          <text x={PAD.l + iw / 2} y={H - 2} className="axistitle" textAnchor="middle">
            episode
          </text>
          <text x={sx(0)} y={H - 12} className="axislabel" textAnchor="start">
            1
          </text>
          <text x={sx(n - 1)} y={H - 12} className="axislabel" textAnchor="end">
            {n}
          </text>
          {scores.map((s, i) => (
            <circle key={i} cx={sx(i)} cy={sy(s)} r={2.2} className="scoredot" />
          ))}
          <polyline points={roll.map((v, i) => `${sx(i)},${sy(v)}`).join(' ')} className="rollline" />
          {hover !== null && <line x1={sx(hover)} x2={sx(hover)} y1={PAD.t} y2={PAD.t + ih} className="crosshair" />}
        </svg>
        {hover !== null && (
          <div className="tooltip" style={{ left: Math.min(sx(hover) + 10, W - 150), top: 6 }}>
            <div className="tooltip__title">episode {hover + 1}</div>
            <div className="tooltip__row">score {fmt(scores[hover], 2)}</div>
            <div className="tooltip__row">rolling {fmt(roll[hover], 3)}</div>
          </div>
        )}
      </div>
    </section>
  )
}
