/**
 * The mock's phi(phrase): range-normalised features, standardised against the
 * mock motif bank (per-dimension mean/sd over the 8 base motifs at default tempo),
 * scaled by 0.25 and clipped to [-1, 1]. This is the vector the mock receiver learns
 * from and the vector shown in its Observation. (Backend phi may differ.)
 */
import type { Phrase, SymbolicFeatures } from '../types'
import { computeSymbolicFeatures, type FeatureNormaliser } from '../../lib/features'
import { mean } from '../../lib/stats'
import { MOCK_MOTIFS, MOCK_TEMPO } from './data'

let cached: FeatureNormaliser | null = null

export function bankNormaliser(): FeatureNormaliser {
  if (cached) return cached
  const raws = MOCK_MOTIFS.map(
    (m) => computeSymbolicFeatures({ id: m.id, notes: m.notes, tempo_bpm: MOCK_TEMPO, length_beats: 8, instrument: 'pluck' }).vector,
  )
  const dims = raws[0].length
  const mu = Array.from({ length: dims }, (_, i) => mean(raws.map((r) => r[i])))
  const sigma = Array.from({ length: dims }, (_, i) => Math.sqrt(mean(raws.map((r) => (r[i] - mu[i]) ** 2))))
  cached = { mu, sigma, scale: 0.25, floor: 0.05 }
  return cached
}

export function mockFeatures(phrase: Phrase): SymbolicFeatures {
  return computeSymbolicFeatures(phrase, bankNormaliser())
}
