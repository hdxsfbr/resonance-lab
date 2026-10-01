/**
 * Instrument spec shared with the server numpy synth (see ./SYNTH_SPEC.md).
 * Pure functions only: no Web Audio here, so everything is unit-testable.
 */
import type { Instrument, Note, Phrase } from '../api/types'

export const SYNTH_SPEC_VERSION = 'resonance-synth-spec/1'

/** ADSR for sine / triangle / square (seconds; sustain is a level). */
export const ADSR = { a: 0.01, d: 0.1, s: 0.7, r: 0.15 } as const
/** Release applied after note-off for every instrument (= ADSR.r). */
export const RELEASE = ADSR.r
/** Silence appended after the phrase's nominal length. */
export const TAIL_SECONDS = 0.5

export interface PercussiveSpec {
  tau: number
  /** Linear attack ramp (s). 0 = instantaneous onset (pluck, bell), as in the server synth. */
  attack: number
  partials: { ratio: number; amp: number }[]
}

/** Exponential-decay instruments: env(t) = attack(t) * exp(-t / tau) * releaseGate(t), t from note onset. */
export const PERCUSSIVE: Record<'pluck' | 'bell' | 'marimba', PercussiveSpec> = {
  pluck: { tau: 0.35, attack: 0, partials: [{ ratio: 1, amp: 1 }] },
  bell: {
    tau: 0.8,
    attack: 0,
    partials: [
      { ratio: 1.0, amp: 1.0 },
      { ratio: 2.4, amp: 0.5 },
      { ratio: 5.95, amp: 0.25 },
    ],
  },
  marimba: {
    tau: 0.25,
    attack: 0.005,
    partials: [
      { ratio: 1.0, amp: 1.0 },
      { ratio: 4.0, amp: 0.3 },
    ],
  },
}

export const SQUARE_AMP = 0.5
export const SQUARE_LOWPASS_HZ = 2500
export const PLUCK_MIX = { triangle: 0.6, sawtooth: 0.4 } as const
export const PLUCK_SWEEP_HZ = { from: 4000, to: 800 } as const
/** Envelope curve sample rate for setValueCurveAtTime (linear interpolation between samples). */
export const ENVELOPE_RATE_HZ = 2000

export function isPercussive(i: Instrument): i is 'pluck' | 'bell' | 'marimba' {
  return i === 'pluck' || i === 'bell' || i === 'marimba'
}

export function beatsToSeconds(beats: number, tempoBpm: number): number {
  return (beats * 60) / tempoBpm
}

export function secondsToBeats(seconds: number, tempoBpm: number): number {
  return (seconds * tempoBpm) / 60
}

export function midiToHz(pitch: number): number {
  return 440 * 2 ** ((pitch - 69) / 12)
}

/** gain = (velocity/127)^1.5 * 0.5 */
export function velocityGain(velocity: number): number {
  const v = Math.max(0, Math.min(127, velocity))
  return (v / 127) ** 1.5 * 0.5
}

/** Nominal phrase duration (length_beats at tempo), without the tail. */
export function phraseSeconds(p: Pick<Phrase, 'length_beats' | 'tempo_bpm'>): number {
  return beatsToSeconds(p.length_beats, p.tempo_bpm)
}

/** Total rendered duration: max(nominal length, last note-off + release) + tail. */
export function renderSeconds(p: Pick<Phrase, 'length_beats' | 'tempo_bpm' | 'notes'>): number {
  let end = phraseSeconds(p)
  for (const n of p.notes) end = Math.max(end, beatsToSeconds(n.onset + n.duration, p.tempo_bpm) + RELEASE)
  return end + TAIL_SECONDS
}

export interface NoteTiming {
  start: number
  dur: number
  off: number
  end: number
}

export function noteTiming(n: Pick<Note, 'onset' | 'duration'>, tempoBpm: number): NoteTiming {
  const start = beatsToSeconds(n.onset, tempoBpm)
  const dur = beatsToSeconds(n.duration, tempoBpm)
  return { start, dur, off: start + dur, end: start + dur + RELEASE }
}

const clip01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x)

/**
 * ADSR (sine / triangle / square), identical to the server's `_adsr`:
 * attack 0.01 linear 0->1, decay 0.10 linear 1->0.7, sustain 0.7 until T,
 * then a 0.15 s linear release from the level reached at T.
 */
export function adsrValueAt(t: number, T: number, peak = 1): number {
  const { a, d, s, r } = ADSR
  const pre = (x: number) => (x < a ? clip01(x / a) : 1 - (1 - s) * clip01((x - a) / d))
  if (t < 0) return 0
  if (t < T) return peak * pre(t)
  return peak * pre(T) * clip01(1 - (t - T) / r)
}

/** 1 until T, then a linear ramp to 0 over RELEASE (percussive voices). */
export function releaseGate(t: number, T: number): number {
  return t < T ? 1 : clip01(1 - (t - T) / RELEASE)
}

/** Percussive envelope: attack(t) * exp(-t / tau) * releaseGate(t), t measured from note onset. */
export function percValueAt(t: number, T: number, spec: Pick<PercussiveSpec, 'tau' | 'attack'>, peak = 1): number {
  if (t < 0) return 0
  const att = spec.attack > 0 ? clip01(t / spec.attack) : 1
  return peak * att * Math.exp(-t / spec.tau) * releaseGate(t, T)
}

/** Envelope value for any instrument. */
export function envelopeValueAt(instrument: Instrument, t: number, T: number, peak = 1): number {
  return isPercussive(instrument) ? percValueAt(t, T, PERCUSSIVE[instrument], peak) : adsrValueAt(t, T, peak)
}

/** Sampled envelope from onset to T + RELEASE (inclusive) for setValueCurveAtTime. */
export function envelopeCurve(instrument: Instrument, T: number, peak = 1, rate = ENVELOPE_RATE_HZ): Float32Array {
  const total = T + RELEASE
  const n = Math.max(2, Math.ceil(total * rate) + 1)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = envelopeValueAt(instrument, (i / (n - 1)) * total, T, peak)
  return out
}

/** Pluck low-pass cutoff (Hz) at time t into a note of duration T: exponential sweep 4 kHz -> 800 Hz, held in the release. */
export function pluckCutoffAt(t: number, T: number): number {
  const { from, to } = PLUCK_SWEEP_HZ
  return from * (to / from) ** clip01(t / Math.max(T, 1e-6))
}

/** One-pole low-pass coefficient a = 1 - exp(-2 pi fc / sr); y[n] = y[n-1] + a (x[n] - y[n-1]). */
export function onePoleCoefficient(fc: number, sampleRate: number): number {
  return 1 - Math.exp((-2 * Math.PI * fc) / sampleRate)
}

/** Soft clip applied to the summed mix (server: output = tanh(mix)). */
export function softClip(x: number): number {
  return Math.tanh(x)
}

export type Speed = '0.5x' | '1x' | '2x' | '4x' | 'fast'
export const SPEEDS: Speed[] = ['0.5x', '1x', '2x', '4x', 'fast']

/**
 * Wall-clock wait between successive step requests while running.
 * Speed changes the PACING between episodes, never a phrase's tempo or pitch.
 */
export function pacingSeconds(p: Pick<Phrase, 'length_beats' | 'tempo_bpm' | 'notes'>, speed: Speed): number {
  const full = renderSeconds(p)
  switch (speed) {
    case '0.5x':
      return full * 2
    case '1x':
      return full
    case '2x':
      return full / 2
    case '4x':
      return full / 4
    case 'fast':
      return 0
  }
}

export function stepsPerRequest(speed: Speed): number {
  return speed === 'fast' ? 10 : 1
}
