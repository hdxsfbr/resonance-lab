# Browser synth — Web Audio realisation of the instrument spec

The instrument spec is shared with the server numpy synth
(`backend/resonance/music/synth.py`, `synth_version = "numpy-synth-1"`). The canonical text
is `docs/SYNTH_SPEC.md`; this file records how the browser implements it and where it can
differ. Pure math lives in `synthSpec.ts` (unit-tested in `synthSpec.test.ts`); the Web Audio
graph is in `synth.ts`; playback/transport in `engine.ts`; offline WAV in `wav.ts`.

## The spec (as implemented)

| quantity | definition |
|---|---|
| time | `seconds = beats * 60 / tempo_bpm`; note start = onset·spb, note length `T` = duration·spb |
| pitch | `f = 440 * 2^((pitch - 69) / 12)` |
| note gain | `g = (velocity / 127)^1.5 * 0.5` |
| ADSR (sine, triangle, square) | attack 0.01 s linear 0→1, decay 0.10 s linear 1→0.7, sustain 0.7 until `T`, release 0.15 s linear from the level reached at `T` to 0 |
| percussive (pluck, bell, marimba) | `attack(t) · exp(-t/τ) · gate(t)` with `t` from the note onset; `gate` = 1 until `T`, then a 0.15 s linear ramp to 0; pluck/bell: no attack ramp; marimba: 0.005 s linear attack |
| sine / triangle | oscillator × ADSR |
| square | 0.5 × square → one-pole low-pass 2500 Hz → × ADSR |
| pluck | (0.6 × triangle + 0.4 × sawtooth) → one-pole low-pass whose cutoff sweeps exponentially 4000 → 800 Hz over `T` (held at 800 Hz during the release) → × exp decay, τ = 0.35 s |
| bell | sine partials f (1.0), 2.4f (0.5), 5.95f (0.25) × exp decay, τ = 0.8 s |
| marimba | sine partials f (1.0), 4f (0.3) × exp decay, τ = 0.25 s, attack 0.005 s |
| mix | notes summed, soft limiter `out = tanh(mix)` |
| length | `max(length_beats·spb, last note end + 0.15 s) + 0.5 s` tail |

## Web Audio mapping

| part | implementation | parity with server |
|---|---|---|
| envelopes | `envelopeCurve()` evaluates the formulas above at 2 kHz; applied with `AudioParam.setValueCurveAtTime` (linear interpolation between samples) on a per-note `GainNode` | exact up to the 0.5 ms sampling |
| square filter | `IIRFilterNode(feedforward [a], feedback [1, -(1-a)])`, `a = 1 - exp(-2π·2500/sr)` at the context's sample rate | exact difference equation |
| pluck filter | AudioWorklet `resonance-one-pole` (inline module) with an a-rate `cutoff` param: `setValueAtTime(4000)` + `exponentialRampToValueAtTime(800, T)` | exact difference equation; if AudioWorklet is unavailable a `BiquadFilterNode` low-pass (Q 0.5) is used and the engine reports `pluckFilter: 'biquad fallback'` |
| oscillators | `OscillatorNode` (`sine`, `triangle`, `sawtooth`, `square`) | sine exact; triangle/sawtooth/square are band-limited wavetables in Web Audio, the server uses naive waveforms (aliasing) — small timbral difference |
| soft limiter | voices → `GainNode(0.25)` → `WaveShaperNode` (tanh curve over [-4, 4], 4× oversampling) → volume/mute `GainNode` → destination, i.e. `volume · tanh(mix)` for \|mix\| ≤ 4 | exact at volume 1 (within the curve resolution) |
| sample rate | the device's `AudioContext` rate (live); 44.1 kHz for the in-browser WAV (mock only) | server WAV is 22.05 kHz |

## API

```ts
import { playPhrase } from './engine'
const h = playPhrase(phrase, { when: 0.05, onProgress: (beat) => { /* beat or null at end */ } })
h.stop()          // fade out within ~15 ms and stop all voices
await h.done      // resolves after release + tail (or on stop)
```

- `audioEngine.enable()` creates/resumes the `AudioContext` **only from a user gesture**
  (the "Enable audio" button), respecting browser autoplay policy. Until then `playPhrase`
  returns a *silent* handle that still keeps time, so pacing and the piano-roll playhead behave
  identically with or without sound.
- `audioEngine.setVolume(v)`, `setMuted(m)`, `suspend()` (pause output; the simulation is unaffected).
- The global playhead (`playhead.ts`) is a tiny pub-sub read by the piano roll on animation frames.

## Transport / pacing (frontend owns time)

The run loop (`state/LabContext.tsx`) requests a step, plays the sent phrase, then waits
`pacingSeconds(phrase, speed)` before the next request:

| speed | step request | wait before next request |
|---|---|---|
| 0.5× | n = 1 | 2 × (phrase length incl. release + tail) |
| 1× | n = 1 | phrase length incl. release + 0.5 s tail (the phrase finishes) |
| 2× / 4× | n = 1 | ½ / ¼ of that; the previous phrase is cut off when the next starts |
| no audio / fast | n = 10 | none (no audio) |

Speed changes pacing between episodes, never a phrase's tempo or pitch. Senders alternate,
so consecutive performances alternate between the agents' instruments.

## WAV export

- Live data source: `POST /api/phrases/render` (server numpy synth).
- Mock data source: `renderPhraseWav()` renders the same Web Audio graph in an
  `OfflineAudioContext` (44.1 kHz mono) and encodes 16-bit PCM (`encodeWav`).
- "⤓ JSON" always downloads the symbolic phrase (notes, tempo, length, instrument).
