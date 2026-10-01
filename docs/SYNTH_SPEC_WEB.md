# Browser synth (Web Audio) — implementation notes

The browser plays phrases with a Web Audio implementation of `docs/SYNTH_SPEC.md`.
Full notes: `frontend/src/audio/SYNTH_SPEC.md`. Summary of parity with the server synth:

- Envelopes (ADSR and percussive) use the same formulas, sampled at 2 kHz into
  `setValueCurveAtTime` → equal up to sampling.
- Square low-pass: `IIRFilterNode` implementing the same one-pole difference equation (exact).
- Pluck low-pass: an AudioWorklet one-pole with an a-rate cutoff swept exponentially
  4000 → 800 Hz over the note (exact equation); BiquadFilter fallback only if AudioWorklet is unavailable.
- Mix: `tanh` soft clip via a WaveShaper (exact within curve resolution), then volume/mute.
- Known difference: Web Audio's triangle/sawtooth/square oscillators are band-limited; the server's are naive.
- Tail and lengths identical: `max(phrase end, last note end + 0.15 s) + 0.5 s`.
