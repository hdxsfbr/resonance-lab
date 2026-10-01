import { describe, expect, it } from 'vitest'
import {
  ADSR,
  PERCUSSIVE,
  RELEASE,
  TAIL_SECONDS,
  adsrValueAt,
  beatsToSeconds,
  envelopeCurve,
  envelopeValueAt,
  noteTiming,
  onePoleCoefficient,
  pacingSeconds,
  percValueAt,
  phraseSeconds,
  pluckCutoffAt,
  renderSeconds,
  secondsToBeats,
  stepsPerRequest,
  velocityGain,
} from './synthSpec'

const phrase = (notes: { pitch: number; onset: number; duration: number; velocity: number }[], tempo = 120, length = 8) => ({
  notes,
  tempo_bpm: tempo,
  length_beats: length,
})

describe('scheduling math', () => {
  it('converts beats to seconds as beats * 60 / tempo', () => {
    expect(beatsToSeconds(8, 120)).toBe(4)
    expect(beatsToSeconds(1, 60)).toBe(1)
    expect(beatsToSeconds(3, 90)).toBeCloseTo(2, 12)
    expect(secondsToBeats(beatsToSeconds(5.5, 137), 137)).toBeCloseTo(5.5, 12)
  })

  it('computes note timing including the 0.15 s release', () => {
    const t = noteTiming({ onset: 2, duration: 1 }, 120)
    expect(t.start).toBe(1)
    expect(t.dur).toBe(0.5)
    expect(t.off).toBe(1.5)
    expect(t.end).toBeCloseTo(1.5 + RELEASE, 12)
  })

  it('phrase duration = length_beats at tempo; render adds release overhang + 0.5 s tail', () => {
    const p = phrase([{ pitch: 60, onset: 0, duration: 1, velocity: 100 }])
    expect(phraseSeconds(p)).toBe(4)
    expect(renderSeconds(p)).toBeCloseTo(4 + TAIL_SECONDS, 12)
    // a note running to the very end extends the render by the release
    const q = phrase([{ pitch: 60, onset: 7, duration: 1, velocity: 100 }])
    expect(renderSeconds(q)).toBeCloseTo(4 + RELEASE + TAIL_SECONDS, 12)
  })

  it('velocity gain = (v/127)^1.5 * 0.5', () => {
    expect(velocityGain(127)).toBeCloseTo(0.5, 12)
    expect(velocityGain(0)).toBe(0)
    expect(velocityGain(64)).toBeCloseTo((64 / 127) ** 1.5 * 0.5, 12)
  })

  it('pacing: 1x waits for the full render, fast does not wait and steps 10 at a time', () => {
    const p = phrase([{ pitch: 60, onset: 0, duration: 1, velocity: 100 }])
    const full = renderSeconds(p)
    expect(pacingSeconds(p, '1x')).toBeCloseTo(full, 12)
    expect(pacingSeconds(p, '0.5x')).toBeCloseTo(2 * full, 12)
    expect(pacingSeconds(p, '2x')).toBeCloseTo(full / 2, 12)
    expect(pacingSeconds(p, '4x')).toBeCloseTo(full / 4, 12)
    expect(pacingSeconds(p, 'fast')).toBe(0)
    expect(stepsPerRequest('fast')).toBe(10)
    expect(stepsPerRequest('1x')).toBe(1)
  })
})

describe('envelopes', () => {
  it('ADSR: linear attack to 1 at 10 ms, decay to 0.7 at 110 ms, sustain, 150 ms linear release', () => {
    const T = 1
    expect(adsrValueAt(0, T)).toBe(0)
    expect(adsrValueAt(ADSR.a / 2, T)).toBeCloseTo(0.5, 9)
    expect(adsrValueAt(ADSR.a, T)).toBeCloseTo(1, 9)
    expect(adsrValueAt(ADSR.a + ADSR.d / 2, T)).toBeCloseTo(0.85, 9)
    expect(adsrValueAt(ADSR.a + ADSR.d, T)).toBeCloseTo(0.7, 9)
    expect(adsrValueAt(0.5, T)).toBeCloseTo(0.7, 9)
    expect(adsrValueAt(T + RELEASE / 2, T)).toBeCloseTo(0.35, 9)
    expect(adsrValueAt(T + RELEASE, T)).toBeCloseTo(0, 9)
    expect(adsrValueAt(T + 1, T)).toBe(0)
  })

  it('ADSR: a note shorter than attack+decay releases from the level reached at note-off', () => {
    const T = 0.06 // mid-decay: 1 - 0.3 * 0.5 = 0.85
    expect(adsrValueAt(T, T)).toBeCloseTo(0.85, 9)
    expect(adsrValueAt(T + RELEASE / 2, T)).toBeCloseTo(0.425, 9)
    const tiny = 0.005 // mid-attack
    expect(adsrValueAt(tiny + RELEASE / 2, tiny)).toBeCloseTo(0.25, 9)
  })

  it('percussive: exp(-t/tau) from onset, gated by a linear release after note-off', () => {
    const T = 0.5
    const pl = PERCUSSIVE.pluck
    expect(percValueAt(0, T, pl)).toBe(1) // no attack ramp for pluck (server parity)
    expect(percValueAt(0.35, T, pl)).toBeCloseTo(Math.exp(-1), 9)
    expect(percValueAt(T + RELEASE / 2, T, pl)).toBeCloseTo(Math.exp(-(T + RELEASE / 2) / 0.35) * 0.5, 9)
    expect(percValueAt(T + RELEASE, T, pl)).toBeCloseTo(0, 9)
    const bell = PERCUSSIVE.bell
    expect(percValueAt(0.8, 2, bell)).toBeCloseTo(Math.exp(-1), 9)
    const m = PERCUSSIVE.marimba
    expect(m.attack).toBe(0.005)
    expect(percValueAt(0.0025, T, m)).toBeCloseTo(0.5 * Math.exp(-0.0025 / 0.25), 9)
    expect(percValueAt(0.25, T, m)).toBeCloseTo(Math.exp(-1), 9)
  })

  it('bell and marimba partials follow the spec', () => {
    expect(PERCUSSIVE.bell.partials).toEqual([
      { ratio: 1, amp: 1 },
      { ratio: 2.4, amp: 0.5 },
      { ratio: 5.95, amp: 0.25 },
    ])
    expect(PERCUSSIVE.marimba.partials).toEqual([
      { ratio: 1, amp: 1 },
      { ratio: 4, amp: 0.3 },
    ])
    expect(PERCUSSIVE.pluck.tau).toBe(0.35)
    expect(PERCUSSIVE.bell.tau).toBe(0.8)
    expect(PERCUSSIVE.marimba.tau).toBe(0.25)
  })

  it('sampled curve spans onset..note-off+release and matches the analytic envelope', () => {
    for (const inst of ['sine', 'square', 'pluck', 'bell', 'marimba'] as const) {
      const T = 0.4
      const peak = velocityGain(100)
      const c = envelopeCurve(inst, T, peak, 1000)
      expect(c.length).toBe(Math.ceil((T + RELEASE) * 1000) + 1)
      expect(c[c.length - 1]).toBeCloseTo(0, 6)
      const i = 123
      const t = (i / (c.length - 1)) * (T + RELEASE)
      expect(c[i]).toBeCloseTo(envelopeValueAt(inst, t, T, peak), 6)
      expect(Math.max(...c)).toBeLessThanOrEqual(peak + 1e-6)
    }
  })

  it('pluck cutoff sweeps exponentially 4 kHz -> 800 Hz over the note and holds', () => {
    expect(pluckCutoffAt(0, 1)).toBe(4000)
    expect(pluckCutoffAt(0.5, 1)).toBeCloseTo(Math.sqrt(4000 * 800), 6)
    expect(pluckCutoffAt(1, 1)).toBeCloseTo(800, 9)
    expect(pluckCutoffAt(1.1, 1)).toBeCloseTo(800, 9)
  })

  it('one-pole coefficient matches the server definition', () => {
    expect(onePoleCoefficient(2500, 22050)).toBeCloseTo(1 - Math.exp((-2 * Math.PI * 2500) / 22050), 12)
  })
})
