/**
 * Symbolic (note-list) features, browser implementation used by the MOCK data
 * source and the phrase editor preview. The authoritative implementation is the
 * backend's resonance/music/features.py (POST /api/phrases/features); values
 * here follow the field definitions in schemas.SymbolicFeatures but are not
 * guaranteed to be numerically identical.
 */
import type { Phrase, SymbolicFeatures } from '../api/types'
import { clamp, mean } from './stats'

const DISSONANT_PC = new Set([1, 6, 11])

export function phraseSeconds(p: Pick<Phrase, 'length_beats' | 'tempo_bpm'>): number {
  return (p.length_beats * 60) / p.tempo_bpm
}

function std(xs: number[]): number {
  if (xs.length < 2) return 0
  const m = mean(xs)
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)))
}

export function contourClass(pitches: number[]): SymbolicFeatures['contour_class'] {
  if (pitches.length < 2) return 'flat'
  const range = Math.max(...pitches) - Math.min(...pitches)
  if (range < 2) return 'flat'
  const iv = pitches.slice(1).map((p, i) => p - pitches[i]).filter((d) => d !== 0)
  let changes = 0
  for (let i = 1; i < iv.length; i++) if (Math.sign(iv[i]) !== Math.sign(iv[i - 1])) changes++
  if (iv.length >= 3 && changes / (iv.length - 1) > 0.6) return 'zigzag'
  const third = Math.max(1, Math.floor(pitches.length / 3))
  const start = mean(pitches.slice(0, third))
  const end = mean(pitches.slice(-third))
  const mid = mean(pitches.slice(third, pitches.length - third).length ? pitches.slice(third, pitches.length - third) : pitches)
  if (mid > start + 2 && mid > end + 2) return 'arch'
  if (mid < start - 2 && mid < end - 2) return 'valley'
  return end >= start ? 'rising' : 'falling'
}

/** Optional per-dimension standardisation applied to phi: clip((raw - mu) / max(sigma, floor) * scale, -1, 1). */
export interface FeatureNormaliser {
  mu: number[]
  sigma: number[]
  scale: number
  floor: number
}

export function computeSymbolicFeatures(phrase: Phrase, normaliser?: FeatureNormaliser): SymbolicFeatures {
  const notes = [...phrase.notes].sort((a, b) => a.onset - b.onset || a.pitch - b.pitch)
  const n = notes.length
  const duration_seconds = phraseSeconds(phrase)
  const onsets = notes.map((x) => x.onset)
  const pitches = notes.map((x) => x.pitch)
  const vels = notes.map((x) => x.velocity)
  const iois = onsets.slice(1).map((o, i) => o - onsets[i]).filter((d) => d > 1e-6)
  const ioiMean = mean(iois)
  const rhythmic_regularity = iois.length < 2 || ioiMean === 0 ? 1 : clamp(1 - std(iois) / ioiMean, 0, 1)
  const syncopation = n === 0 ? 0 : onsets.filter((o) => Math.abs(o * 2 - Math.round(o * 2)) > 0.02).length / n
  const mean_pitch = n ? mean(pitches) : 0
  const pitch_range = n ? Math.max(...pitches) - Math.min(...pitches) : 0
  // contour: least-squares slope of pitch over beats, scaled so ±24 semitones across the phrase -> ±1
  let contour = 0
  if (n >= 2) {
    const mo = mean(onsets)
    let num = 0
    let den = 0
    for (let i = 0; i < n; i++) {
      num += (onsets[i] - mo) * (pitches[i] - mean_pitch)
      den += (onsets[i] - mo) ** 2
    }
    const slope = den > 0 ? num / den : 0
    contour = clamp((slope * phrase.length_beats) / 24, -1, 1)
  }
  const intervals = pitches.slice(1).map((p, i) => p - pitches[i])
  const hist = [0, 0, 0, 0, 0]
  for (const d of intervals) {
    const a = Math.abs(d)
    hist[a === 0 ? 0 : a <= 2 ? 1 : a <= 4 ? 2 : a <= 7 ? 3 : 4]++
  }
  const interval_histogram = intervals.length ? hist.map((h) => h / intervals.length) : hist
  // repetition: fraction of (interval, ioi) bigrams already seen earlier in the phrase
  const seen = new Set<string>()
  let rep = 0
  let bigrams = 0
  for (let i = 1; i < n; i++) {
    const key = `${pitches[i] - pitches[i - 1]}|${(onsets[i] - onsets[i - 1]).toFixed(3)}`
    if (seen.has(key)) rep++
    seen.add(key)
    bigrams++
  }
  const repetition = bigrams ? rep / bigrams : 0
  const mean_velocity = n ? mean(vels) : 0
  const velocity_variation = mean_velocity ? std(vels) / mean_velocity : 0
  const dissonance_proxy = intervals.length
    ? intervals.filter((d) => DISSONANT_PC.has(((Math.abs(d) % 12) + 12) % 12)).length / intervals.length
    : 0
  const note_density = duration_seconds > 0 ? n / duration_seconds : 0
  const vector = [
    clamp(note_density / 8, 0, 1),
    rhythmic_regularity,
    syncopation,
    clamp((mean_pitch - 36) / 48, 0, 1),
    clamp(pitch_range / 24, 0, 1),
    contour,
    repetition,
    ...interval_histogram,
    mean_velocity / 127,
    clamp(velocity_variation, 0, 1),
    dissonance_proxy,
    clamp((phrase.tempo_bpm - 60) / 120, 0, 1),
  ]
  const vec = normaliser
    ? vector.map((x, i) => clamp(((x - normaliser.mu[i]) / Math.max(normaliser.sigma[i], normaliser.floor)) * normaliser.scale, -1, 1))
    : vector
  return {
    kind: 'symbolic',
    note_count: n,
    duration_seconds,
    note_density,
    rhythmic_regularity,
    syncopation,
    mean_pitch,
    pitch_range,
    contour,
    contour_class: contourClass(pitches),
    repetition,
    interval_histogram,
    mean_velocity,
    velocity_variation,
    dissonance_proxy,
    tempo_bpm: phrase.tempo_bpm,
    vector: vec,
  }
}
