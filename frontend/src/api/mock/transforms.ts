/** Phrase transforms (mock implementation of resonance/music transforms; definitions from schemas.Perturbation). */
import type { Perturbation, Phrase } from '../types'
import { Rng } from './rng'

export function transformPhrase(phrase: Phrase, kind: Perturbation, rng: Rng): Phrase {
  if (kind === 'none') return phrase
  const notes = [...phrase.notes].sort((a, b) => a.onset - b.onset).map((n) => ({ ...n }))
  let tempo = phrase.tempo_bpm
  switch (kind) {
    case 'transpose':
      for (const n of notes) n.pitch = Math.min(127, n.pitch + 5)
      break
    case 'velocity_flatten':
      for (const n of notes) n.velocity = 80
      break
    case 'tempo_shift':
      tempo = Math.min(400, tempo * 1.25)
      break
    case 'contour_invert': {
      const m = notes.reduce((a, n) => a + n.pitch, 0) / Math.max(1, notes.length)
      for (const n of notes) n.pitch = Math.max(0, Math.min(127, Math.round(2 * m - n.pitch)))
      break
    }
    case 'rhythm_shuffle': {
      if (notes.length > 2) {
        const iois = notes.slice(1).map((n, i) => n.onset - notes[i].onset)
        const shuffled = rng.shuffle(iois)
        let t = notes[0].onset
        for (let i = 1; i < notes.length; i++) {
          t += shuffled[i - 1]
          notes[i].onset = t
        }
      }
      break
    }
    case 'pitch_shuffle': {
      const ps = rng.shuffle(notes.map((n) => n.pitch))
      notes.forEach((n, i) => (n.pitch = ps[i]))
      break
    }
  }
  return {
    ...phrase,
    id: `${phrase.id}~${kind}`,
    notes,
    tempo_bpm: tempo,
    origin: { kind: 'transformed', agent_id: phrase.origin?.agent_id ?? null, source_phrase_id: phrase.id, transform: kind },
  }
}
