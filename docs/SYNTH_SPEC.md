# Instrument spec (shared by the numpy server synth and the browser Web Audio synth)

Server implementation: `backend/resonance/music/synth.py` (`synth_version = "numpy-synth-1"`),
used for WAV export (`POST /api/phrases/render`) and audio-measured features.
Browser playback implements the same spec with Web Audio; the two are separate
implementations and may differ in small numerical details (filter discretisation,
oscillator band-limiting). Simulation runs never depend on audio.

## Timing

- `seconds_per_beat = 60 / phrase.tempo_bpm`
- note start = `onset_beats * seconds_per_beat`; note length `T = duration_beats * seconds_per_beat`
- frequency `f = 440 * 2^((pitch - 69) / 12)`

## Gain

`g = (velocity / 127)^1.5 * 0.5` per note.

## Oscillators (phase starts at 0)

| name | definition |
|---|---|
| sine | `sin(p)` |
| triangle | `(2/pi) * asin(sin(p))` |
| sawtooth | `2 * (p/2pi - floor(p/2pi + 0.5))` |
| square | `sign(sin(p))` |

## Envelopes

- **ADSR** (sine, triangle, square): attack 0.01 s linear 0→1; decay 0.10 s linear 1→0.7;
  sustain 0.7 until `T`; release 0.15 s linear from the level reached at `T` down to 0.
- **Percussive** (pluck, bell, marimba): exponential decay `exp(-t/tau)` from the onset; the voice
  sounds for `T` and then shares the same 0.15 s linear release ramp to 0. Pluck and bell have no
  attack ramp; marimba has a 0.005 s linear attack.

## Instruments

| instrument | source | filter | envelope |
|---|---|---|---|
| sine | sine | – | ADSR |
| triangle | triangle | – | ADSR |
| square | 0.5 × square | one-pole lowpass 2500 Hz | ADSR |
| pluck | 0.6 × triangle + 0.4 × sawtooth | one-pole lowpass, cutoff sweeps exponentially 4000 Hz → 800 Hz over `T` (held at 800 Hz during release) | exp decay tau = 0.35 s |
| bell | partials f (1.0), 2.4 f (0.5), 5.95 f (0.25), sines | – | exp decay tau = 0.8 s |
| marimba | partials f (1.0), 4 f (0.3), sines | – | exp decay tau = 0.25 s, attack 0.005 s |

One-pole lowpass: `y[n] = y[n-1] + a (x[n] - y[n-1])`, `a = 1 - exp(-2π fc / sr)`
(Web Audio: a `BiquadFilterNode` lowpass with Q ≈ 0.5 is an acceptable approximation; note it as such).

## Mix

Notes are summed, then soft-clipped: `out = tanh(mix)`. Buffer length = max(phrase end
`length_beats * seconds_per_beat`, last note end incl. release) + 0.5 s tail.
Server sample rate 22050 Hz, WAV = 16-bit PCM mono.

## Audio-measured features (server)

`rms`, `peak`, spectral centroid (FFT of the whole buffer: Σ f|X| / Σ |X|), and
`onset_count_estimate` from spectral flux (Hann 1024 / hop 256, log1p magnitudes, positive
differences summed, normalised to max, local maxima > 0.3 at least 80 ms apart). The estimate may
differ from `note_count` (e.g. for square/pluck timbres with strong spectral motion).
