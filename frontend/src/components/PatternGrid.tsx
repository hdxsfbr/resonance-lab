import type { TimingPattern } from '../api/types'

/** One timing pattern as a 4-beat onset grid at sixteenth-note resolution. */
export function PatternGrid({ pattern, size = 10 }: { pattern: TimingPattern; size?: number }) {
  const cellsPerBeat = 4
  const n = Math.round(pattern.length_beats * cellsPerBeat)
  const on = new Set(pattern.onsets.map((o) => Math.round(o * cellsPerBeat)))
  const gap = 2
  const w = n * (size + gap)
  return (
    <svg width={w} height={size + 2} role="img" aria-label={`${pattern.name}: onsets ${pattern.onsets.join(', ')} beats`} className="patterngrid">
      {Array.from({ length: n }, (_, i) => (
        <rect
          key={i}
          x={i * (size + gap)}
          y={1}
          width={size}
          height={size}
          rx={2}
          className={on.has(i) ? 'patterngrid__on' : i % cellsPerBeat === 0 ? 'patterngrid__beat' : 'patterngrid__off'}
        />
      ))}
    </svg>
  )
}
