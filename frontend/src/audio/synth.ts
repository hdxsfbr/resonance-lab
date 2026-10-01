/**
 * Web Audio implementation of the instrument spec (./SYNTH_SPEC.md), aligned with
 * the server numpy synth (backend/resonance/music/synth.py, "numpy-synth-1").
 * Works with both AudioContext (live playback) and OfflineAudioContext (WAV export).
 */
import type { Instrument, Note, Phrase } from '../api/types'
import {
  PERCUSSIVE,
  PLUCK_MIX,
  PLUCK_SWEEP_HZ,
  SQUARE_AMP,
  SQUARE_LOWPASS_HZ,
  envelopeCurve,
  midiToHz,
  noteTiming,
  onePoleCoefficient,
  softClip,
  velocityGain,
} from './synthSpec'

// ---------------------------------------------------------------------------
// One-pole low-pass with an a-rate cutoff (needed for the pluck's swept filter).
// Same difference equation as the server: y[n] = y[n-1] + a (x[n] - y[n-1]).
// ---------------------------------------------------------------------------

const ONE_POLE_NAME = 'resonance-one-pole'
const ONE_POLE_SRC = `
class ResonanceOnePole extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [{ name: 'cutoff', defaultValue: 4000, minValue: 1, maxValue: 20000, automationRate: 'a-rate' }]
  }
  constructor() { super(); this.y = 0 }
  process(inputs, outputs, parameters) {
    const input = inputs[0]
    const out = outputs[0][0]
    if (!out) return input.length > 0
    const x = input[0]
    const c = parameters.cutoff
    const k = -2 * Math.PI / sampleRate
    for (let i = 0; i < out.length; i++) {
      const fc = c.length > 1 ? c[i] : c[0]
      const a = 1 - Math.exp(k * fc)
      this.y += a * ((x ? x[i] : 0) - this.y)
      out[i] = this.y
    }
    return input.length > 0
  }
}
registerProcessor('${ONE_POLE_NAME}', ResonanceOnePole)
`

const workletReady = new WeakSet<BaseAudioContext>()
const workletPending = new WeakMap<BaseAudioContext, Promise<boolean>>()

/** Load the one-pole worklet into a context. Resolves false (biquad fallback) if unavailable. */
export function ensureOnePole(ctx: BaseAudioContext): Promise<boolean> {
  if (workletReady.has(ctx)) return Promise.resolve(true)
  const pending = workletPending.get(ctx)
  if (pending) return pending
  const p = (async () => {
    try {
      if (!ctx.audioWorklet || typeof URL.createObjectURL !== 'function') return false
      const url = URL.createObjectURL(new Blob([ONE_POLE_SRC], { type: 'application/javascript' }))
      await ctx.audioWorklet.addModule(url)
      URL.revokeObjectURL(url)
      workletReady.add(ctx)
      return true
    } catch {
      return false
    }
  })()
  workletPending.set(ctx, p)
  return p
}

export function hasOnePole(ctx: BaseAudioContext): boolean {
  return workletReady.has(ctx)
}

function osc(ctx: BaseAudioContext, type: OscillatorType, hz: number): OscillatorNode {
  const o = ctx.createOscillator()
  o.type = type
  o.frequency.value = hz
  return o
}

function scaled(ctx: BaseAudioContext, amp: number): GainNode {
  const g = ctx.createGain()
  g.gain.value = amp
  return g
}

/** Fixed one-pole low-pass as an IIR filter (exactly the server's square-wave filter). */
function onePoleFixed(ctx: BaseAudioContext, fc: number): AudioNode {
  const a = onePoleCoefficient(fc, ctx.sampleRate)
  return ctx.createIIRFilter([a], [1, -(1 - a)])
}

/** Swept one-pole (worklet) or, if worklets are unavailable, a 2-pole biquad approximation. */
function sweptLowpass(ctx: BaseAudioContext, ts: number, T: number): AudioNode {
  if (hasOnePole(ctx)) {
    const node = new AudioWorkletNode(ctx, ONE_POLE_NAME, { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] })
    const cutoff = node.parameters.get('cutoff')!
    cutoff.setValueAtTime(PLUCK_SWEEP_HZ.from, ts)
    cutoff.exponentialRampToValueAtTime(PLUCK_SWEEP_HZ.to, ts + Math.max(T, 1e-3))
    return node
  }
  const f = ctx.createBiquadFilter()
  f.type = 'lowpass'
  f.Q.value = 0.5
  f.frequency.setValueAtTime(PLUCK_SWEEP_HZ.from, ts)
  f.frequency.exponentialRampToValueAtTime(PLUCK_SWEEP_HZ.to, ts + Math.max(T, 1e-3))
  return f
}

/**
 * Schedule one note. `t0` is the phrase start in context time.
 * Returns the source nodes (so playback can be cancelled).
 */
export function scheduleNote(ctx: BaseAudioContext, dest: AudioNode, note: Note, instrument: Instrument, tempoBpm: number, t0: number): OscillatorNode[] {
  const { start, dur, end } = noteTiming(note, tempoBpm)
  const ts = t0 + start
  const hz = midiToHz(note.pitch)
  const env = ctx.createGain()
  env.gain.value = 0
  env.connect(dest)
  const curve = envelopeCurve(instrument, dur, velocityGain(note.velocity))
  env.gain.setValueCurveAtTime(curve, ts, end - start)
  const sources: OscillatorNode[] = []
  const add = (into: AudioNode, src: OscillatorNode) => {
    src.connect(into)
    sources.push(src)
  }
  switch (instrument) {
    case 'sine':
    case 'triangle':
      add(env, osc(ctx, instrument, hz))
      break
    case 'square': {
      const amp = scaled(ctx, SQUARE_AMP)
      amp.connect(onePoleFixed(ctx, SQUARE_LOWPASS_HZ)).connect(env)
      add(amp, osc(ctx, 'square', hz))
      break
    }
    case 'pluck': {
      const lp = sweptLowpass(ctx, ts, dur)
      lp.connect(env)
      const tri = scaled(ctx, PLUCK_MIX.triangle)
      const saw = scaled(ctx, PLUCK_MIX.sawtooth)
      tri.connect(lp)
      saw.connect(lp)
      add(tri, osc(ctx, 'triangle', hz))
      add(saw, osc(ctx, 'sawtooth', hz))
      break
    }
    case 'bell':
    case 'marimba':
      for (const p of PERCUSSIVE[instrument].partials) {
        const g = scaled(ctx, p.amp)
        g.connect(env)
        add(g, osc(ctx, 'sine', hz * p.ratio))
      }
      break
  }
  for (const s of sources) {
    s.start(ts)
    s.stop(t0 + end + 0.01)
  }
  return sources
}

/** Schedule every note of a phrase at context time t0. */
export function schedulePhrase(ctx: BaseAudioContext, dest: AudioNode, phrase: Phrase, t0: number): OscillatorNode[] {
  const out: OscillatorNode[] = []
  for (const n of phrase.notes) out.push(...scheduleNote(ctx, dest, n, phrase.instrument, phrase.tempo_bpm, t0))
  return out
}

let tanhCurve: Float32Array<ArrayBuffer> | null = null
function softClipCurve(): Float32Array<ArrayBuffer> {
  if (tanhCurve) return tanhCurve
  const n = 4096
  const c = new Float32Array(new ArrayBuffer(n * 4))
  // WaveShaper maps input [-1, 1] to the curve; scale so the curve spans [-4, 4] and pre-attenuate by 1/4.
  for (let i = 0; i < n; i++) c[i] = softClip(((i / (n - 1)) * 2 - 1) * 4)
  tanhCurve = c
  return c
}

/**
 * Master chain: voices -> mix bus -> soft limiter (tanh, as the server) -> volume -> destination.
 * The shaper input is pre-scaled by 1/4 because a WaveShaper's domain is [-1, 1].
 */
export function createMasterChain(ctx: BaseAudioContext, dest: AudioNode): { input: GainNode; volume: GainNode } {
  const input = ctx.createGain()
  input.gain.value = 0.25
  const shaper = ctx.createWaveShaper()
  shaper.curve = softClipCurve()
  shaper.oversample = '4x'
  const volume = ctx.createGain()
  volume.gain.value = 1
  input.connect(shaper).connect(volume).connect(dest)
  return { input, volume }
}
