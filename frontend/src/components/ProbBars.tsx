import type { PolicyTrace } from '../api/types'
import { fmt } from '../lib/format'

/** Mini bar chart of a PolicyTrace's probabilities (one series; chosen bar marked). */
export function ProbBars({ trace, labels, color, height = 40 }: { trace: PolicyTrace; labels: string[]; color: string; height?: number }) {
  const n = trace.probabilities.length
  const bw = Math.min(22, Math.floor(200 / Math.max(n, 1)) - 4)
  const gap = 4
  const w = n * (bw + gap)
  const top = 12
  return (
    <svg className="probbars" width={w} height={height + top + 14} role="img" aria-label={`${trace.role} policy probabilities`}>
      {trace.probabilities.map((p, i) => {
        const h = Math.max(1, p * height)
        const x = i * (bw + gap)
        const chosen = i === trace.chosen
        return (
          <g key={i}>
            <title>{`${labels[i] ?? i}: p=${fmt(p, 3)} · score ${fmt(trace.scores[i], 3)}${chosen ? ' · chosen' : ''}`}</title>
            <rect x={x} y={top + height - h} width={bw} height={h} rx={2} fill={color} opacity={chosen ? 1 : 0.45} />
            {chosen && <text x={x + bw / 2} y={top + height - h - 3} className="probbars__chosen" textAnchor="middle">▼</text>}
            <text x={x + bw / 2} y={top + height + 11} className="probbars__label" textAnchor="middle">
              {labels[i]?.slice(0, 3) ?? i}
            </text>
          </g>
        )
      })}
      <line x1={0} x2={w} y1={top + height + 0.5} y2={top + height + 0.5} className="axisline" />
    </svg>
  )
}
