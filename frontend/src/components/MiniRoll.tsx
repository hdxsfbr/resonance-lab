import { memo } from 'react'
import type { Phrase } from '../api/types'
import { sequential } from '../lib/colors'

export const MiniRoll = memo(function MiniRoll({ phrase, width = 72, height = 22 }: { phrase: Phrase; width?: number; height?: number }) {
  const ps = phrase.notes.map((n) => n.pitch)
  const lo = Math.min(...ps, 60) - 1
  const hi = Math.max(...ps, lo + 12) + 1
  const sx = width / phrase.length_beats
  const sy = height / (hi - lo + 1)
  return (
    <svg width={width} height={height} className="miniroll" aria-hidden>
      <rect width={width} height={height} rx={2} className="miniroll__bg" />
      {phrase.notes.map((n, i) => (
        <rect key={i} x={n.onset * sx} y={(hi - n.pitch) * sy} width={Math.max(1.5, n.duration * sx - 0.5)} height={Math.max(1.5, sy)} fill={sequential(n.velocity / 127)} />
      ))}
    </svg>
  )
})
