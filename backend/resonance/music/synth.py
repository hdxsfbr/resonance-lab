"""Server-side numpy synthesiser (synth_version "numpy-synth-1").

Used for WAV export (`/api/phrases/render`) and for AUDIO-MEASURED features. The
browser plays phrases with a separate Web Audio implementation of the SAME
instrument spec (docs/SYNTH_SPEC.md). Simulation runs never need audio.

INSTRUMENT SPEC (times in seconds; f = 440 * 2**((pitch - 69) / 12)):

  note gain  g = (velocity / 127) ** 1.5 * 0.5
  note length T = duration_beats * 60 / tempo_bpm; onset time = onset_beats * 60 / tempo_bpm
  Oscillators start at phase 0:
      sine(p) = sin(p);  triangle(p) = (2/pi) * arcsin(sin(p));
      sawtooth(p) = 2 * (p/2pi - floor(p/2pi + 0.5));  square(p) = sign(sin(p))
  ADSR (sine, triangle, square): attack 0.01 linear 0->1, decay 0.10 linear 1->0.7,
      sustain 0.7 until T, release 0.15 linear from the level reached at T down to 0.
  sine:     sine * ADSR
  triangle: triangle * ADSR
  square:   0.5 * square, one-pole lowpass fc = 2500 Hz, * ADSR
  pluck:    (0.6 * triangle + 0.4 * sawtooth), one-pole lowpass whose cutoff sweeps
            exponentially 4000 Hz -> 800 Hz over T (held at 800 Hz in the release),
            * exp(-t / 0.35)
  bell:     sin(f) * 1.0 + sin(2.4 f) * 0.5 + sin(5.95 f) * 0.25, * exp(-t / 0.8)
  marimba:  sin(f) * 1.0 + sin(4 f) * 0.3, * exp(-t / 0.25), linear attack 0.005
  Percussive voices (pluck, bell, marimba) sound for T and then share the
  0.15 s linear release used by the ADSR voices (no attack ramp for pluck/bell).
  one-pole lowpass: y[n] = y[n-1] + a * (x[n] - y[n-1]),  a = 1 - exp(-2 pi fc / sr)
  Mix: notes summed; output = tanh(mix); buffer length = max(phrase end, last note
  end incl. release) + 0.5 s tail.
"""

from __future__ import annotations

import io
import math
import wave

import numpy as np

from resonance.schemas import AudioFeatures, Note, Phrase

SYNTH_VERSION = "numpy-synth-1"
DEFAULT_SR = 22050
ATTACK, DECAY, SUSTAIN, RELEASE = 0.01, 0.10, 0.7, 0.15
TAIL_SECONDS = 0.5
SQUARE_CUTOFF = 2500.0
PLUCK_CUTOFF_START, PLUCK_CUTOFF_END = 4000.0, 800.0
DECAY_TAU = {"pluck": 0.35, "bell": 0.8, "marimba": 0.25}
MARIMBA_ATTACK = 0.005


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def note_gain(velocity: int) -> float:
    return (velocity / 127.0) ** 1.5 * 0.5


def _phase(f: float, t: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * f * t


def _triangle(p: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arcsin(np.sin(p))


def _sawtooth(p: np.ndarray) -> np.ndarray:
    c = p / (2.0 * np.pi)
    return 2.0 * (c - np.floor(c + 0.5))


def _adsr(t: np.ndarray, T: float) -> np.ndarray:
    """Linear ADSR held until T, then a linear release from the level reached at T."""

    def pre(x: np.ndarray) -> np.ndarray:
        a = np.clip(x / ATTACK, 0.0, 1.0)
        d = 1.0 - (1.0 - SUSTAIN) * np.clip((x - ATTACK) / DECAY, 0.0, 1.0)
        return np.where(x < ATTACK, a, d)

    level_at_T = float(pre(np.array([T]))[0])
    rel = level_at_T * np.clip(1.0 - (t - T) / RELEASE, 0.0, 1.0)
    return np.where(t < T, pre(t), rel)


def _release_gate(t: np.ndarray, T: float) -> np.ndarray:
    """1 until T, then linear ramp to 0 over RELEASE (percussive voices)."""
    return np.clip(1.0 - (t - T) / RELEASE, 0.0, 1.0)


def _one_pole(x: np.ndarray, cutoff: np.ndarray | float, sr: int) -> np.ndarray:
    """y[n] = y[n-1] + a[n] * (x[n] - y[n-1]),  a = 1 - exp(-2 pi fc / sr)."""
    fc = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = (1.0 - np.exp(-2.0 * np.pi * fc / sr)).tolist()
    xs = x.tolist()
    out = [0.0] * len(xs)
    y = 0.0
    for i, (xi, ai) in enumerate(zip(xs, a, strict=True)):
        y += ai * (xi - y)
        out[i] = y
    return np.asarray(out)


def _voice(note: Note, instrument: str, T: float, sr: int) -> np.ndarray:
    """Waveform of one note (without gain), length T + RELEASE."""
    n = int(math.ceil((T + RELEASE) * sr))
    t = np.arange(n) / sr
    f = midi_to_hz(note.pitch)
    p = _phase(f, t)
    if instrument == "sine":
        return np.sin(p) * _adsr(t, T)
    if instrument == "triangle":
        return _triangle(p) * _adsr(t, T)
    if instrument == "square":
        return _one_pole(0.5 * np.sign(np.sin(p)), SQUARE_CUTOFF, sr) * _adsr(t, T)
    if instrument == "pluck":
        raw = 0.6 * _triangle(p) + 0.4 * _sawtooth(p)
        frac = np.clip(t / max(T, 1e-6), 0.0, 1.0)
        cutoff = PLUCK_CUTOFF_START * (PLUCK_CUTOFF_END / PLUCK_CUTOFF_START) ** frac
        return _one_pole(raw, cutoff, sr) * np.exp(-t / DECAY_TAU["pluck"]) * _release_gate(t, T)
    if instrument == "bell":
        raw = np.sin(p) + 0.5 * np.sin(_phase(2.4 * f, t)) + 0.25 * np.sin(_phase(5.95 * f, t))
        return raw * np.exp(-t / DECAY_TAU["bell"]) * _release_gate(t, T)
    if instrument == "marimba":
        raw = np.sin(p) + 0.3 * np.sin(_phase(4.0 * f, t))
        attack = np.clip(t / MARIMBA_ATTACK, 0.0, 1.0)
        return raw * attack * np.exp(-t / DECAY_TAU["marimba"]) * _release_gate(t, T)
    raise ValueError(f"unknown instrument {instrument!r}")


def render(phrase: Phrase, sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """Render to a float waveform in (-1, 1) (tanh soft clip)."""
    spb = 60.0 / phrase.tempo_bpm
    end = phrase.length_beats * spb
    for n in phrase.notes:
        end = max(end, (n.onset + n.duration) * spb + RELEASE)
    buf = np.zeros(int(math.ceil((end + TAIL_SECONDS) * sample_rate)))
    for n in phrase.notes:
        T = n.duration * spb
        v = _voice(n, phrase.instrument, T, sample_rate) * note_gain(n.velocity)
        start = int(round(n.onset * spb * sample_rate))
        stop = min(len(buf), start + len(v))
        buf[start:stop] += v[: stop - start]
    return np.tanh(buf)


def wav_bytes(phrase: Phrase, sample_rate: int = DEFAULT_SR) -> bytes:
    """16-bit PCM mono WAV."""
    pcm = (np.clip(render(phrase, sample_rate), -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def _onset_estimate(x: np.ndarray, sr: int) -> int:
    """Spectral-flux onset count (an estimate; may differ from note_count).

    STFT: Hann window 1024, hop 256, one window of leading zeros;
    flux[t] = sum_k max(0, log1p|X_t,k| - log1p|X_t-1,k|), normalised by its max;
    a frame is an onset if flux is a local maximum > 0.3 and >= 80 ms after the previous onset.
    """
    hop, win = 256, 1024
    padded = np.concatenate([np.zeros(win), x])
    if len(padded) < 2 * win:
        return 0
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[::hop] * np.hanning(win)
    logmag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(0.0, np.diff(logmag, axis=0)).sum(axis=1)
    flux = flux / (flux.max() + 1e-12)
    min_gap = int(0.08 * sr / hop) + 1
    count, last = 0, -min_gap
    for i in range(1, len(flux) - 1):
        if flux[i] > 0.3 and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and i - last >= min_gap:
            count += 1
            last = i
    return count


def audio_features(phrase: Phrase, sample_rate: int = DEFAULT_SR) -> AudioFeatures:
    x = render(phrase, sample_rate)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    centroid = float(np.sum(freqs * mag) / np.sum(mag)) if np.sum(mag) > 0 else 0.0
    return AudioFeatures(
        sample_rate=sample_rate,
        duration_seconds=round(len(x) / sample_rate, 6),
        rms=round(float(np.sqrt(np.mean(x**2))), 6),
        peak=round(float(np.max(np.abs(x))), 6),
        spectral_centroid_hz=round(centroid, 3),
        onset_count_estimate=_onset_estimate(x, sample_rate),
        synth_version=SYNTH_VERSION,
    )
