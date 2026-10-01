import { diverging, sequential } from '../lib/colors'
import { fmt } from '../lib/format'

interface HeatmapProps {
  rows: string[]
  cols: string[]
  values: number[][]
  mode: 'diverging' | 'sequential'
  /** for sequential: [lo, hi]; for diverging: symmetric max |v| (auto when omitted) */
  domain?: [number, number]
  cell?: number
  showValues?: boolean
  caption: string
  counts?: number[][]
}

export function Heatmap({ rows, cols, values, mode, domain, cell = 18, showValues = false, caption, counts }: HeatmapProps) {
  const maxAbs = Math.max(1e-6, ...values.flat().map((v) => Math.abs(v)))
  const [lo, hi] = domain ?? (mode === 'diverging' ? [-maxAbs, maxAbs] : [0, 1])
  const color = (v: number) => (mode === 'diverging' ? diverging(v / Math.max(Math.abs(lo), Math.abs(hi))) : sequential((v - lo) / (hi - lo || 1)))
  const labelW = 96
  const headH = 78
  const W = labelW + cols.length * (cell + 2)
  const H = headH + rows.length * (cell + 2)
  return (
    <figure className="heatmap">
      <svg width={W} height={H} role="img" aria-label={caption}>
        {cols.map((c, j) => (
          <text key={c} x={labelW + j * (cell + 2) + cell / 2} y={headH - 4} className="axislabel" transform={`rotate(-60 ${labelW + j * (cell + 2) + cell / 2} ${headH - 4})`}>
            {c}
          </text>
        ))}
        {rows.map((r, i) => (
          <g key={r}>
            <text x={labelW - 6} y={headH + i * (cell + 2) + cell / 2 + 4} className="axislabel" textAnchor="end">
              {r}
            </text>
            {cols.map((c, j) => {
              const v = values[i]?.[j] ?? 0
              return (
                <g key={c}>
                  <rect x={labelW + j * (cell + 2)} y={headH + i * (cell + 2)} width={cell} height={cell} rx={2} fill={color(v)}>
                    <title>{`${r} × ${c}: ${fmt(v, 4)}${counts ? ` (n=${counts[i]?.[j] ?? 0})` : ''}`}</title>
                  </rect>
                  {showValues && (
                    <text x={labelW + j * (cell + 2) + cell / 2} y={headH + i * (cell + 2) + cell / 2 + 3} className="heatmap__val" textAnchor="middle">
                      {v.toFixed(2).replace(/^0\./, '.').replace(/^-0\./, '-.')}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        ))}
      </svg>
      <figcaption className="tiny muted">
        {caption} · {mode === 'diverging' ? `red < 0 < blue, scale ±${fmt(Math.max(Math.abs(lo), Math.abs(hi)), 3)}` : `dark ${fmt(lo, 2)} → light ${fmt(hi, 2)}`}
      </figcaption>
    </figure>
  )
}
