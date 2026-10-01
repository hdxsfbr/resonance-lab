import { fmt } from '../lib/format'

export function ScoreChip({ score }: { score: number | null }) {
  if (score === null) return <span className="scorechip scorechip--none">—</span>
  const kind = score >= 0.999 ? 'exact' : score > 0 ? 'partial' : 'miss'
  const icon = kind === 'exact' ? '✓' : kind === 'partial' ? '≈' : '✕'
  const label = kind === 'exact' ? 'exact match' : kind === 'partial' ? 'partial credit' : 'no credit'
  return (
    <span className={`scorechip scorechip--${kind}`} title={label}>
      <span aria-hidden>{icon}</span> {fmt(score, 2)}
    </span>
  )
}
