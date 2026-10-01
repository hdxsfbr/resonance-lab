"""MEASURED symbolic features, computed deterministically from a phrase's note list.

Nothing here reads audio; see `synth.audio_features` for waveform measurements.
Definitions (notes sorted by (onset, pitch); IOI = inter-onset interval in beats):

  note_density         note_count / duration_seconds           (notes per second)
  rhythmic_regularity  clip(1 - std(IOI) / mean(IOI), 0, 1)    (1.0 if < 2 notes)
  syncopation          fraction of onsets farther than GRID_TOLERANCE beats from the
                       eighth-note grid (multiples of 0.5 beat)
  mean_pitch           mean MIDI pitch
  pitch_range          max - min pitch (semitones)
  contour              least-squares slope of pitch vs onset (semitones/beat) times the
                       onset span, divided by 12, clipped to [-1, 1]
                       (= fitted pitch change across the phrase in octaves)
  contour_class        rising | falling | arch | valley | flat | zigzag (rules in `_contour_class`)
  repetition           over the n-1 (interval, IOI) bigrams between consecutive notes:
                       (count - unique) / count, IOI quantised to 1/24 beat
  interval_histogram   fractions of |melodic interval| in classes
                       [unison 0, step 1-2, third 3-4, fourth/fifth 5-7, large >7]
  mean_velocity        mean velocity
  velocity_variation   std(velocity) / mean(velocity)
  dissonance_proxy     fraction of melodic intervals with (|interval| mod 12) in {1, 6, 11}
                       (minor 2nd, tritone, major 7th). A defined proxy, not a perceptual claim.
  tempo_bpm            phrase tempo

The learner input phi(phrase) = `vector` uses FEATURE_NAMES order with the
normalisation constants in NORMALISATION (each then clipped to [0, 1]) and is
finally multiplied element-wise by `music.feature_weights` (default 1.0).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from resonance.schemas import FEATURE_NAMES, Phrase, SymbolicFeatures

GRID_TOLERANCE = 0.06  # beats; onsets within this of a multiple of 0.5 count as "on the eighth grid"
IOI_QUANTUM = 1.0 / 24.0  # beats; IOI resolution for repetition bigrams (robust to jitter)
DISSONANT_PITCH_CLASSES = frozenset({1, 6, 11})

# Normalisation x_norm = (x - offset) / scale, then clipped to [0, 1].
# (offset, scale) per FEATURE_NAMES entry. Documented constants, chosen so the
# default motif bank and plausible human phrases spread over [0, 1].
NORMALISATION: dict[str, tuple[float, float]] = {
    "note_density": (0.0, 6.0),  # notes/s; 6/s -> 1.0
    "rhythmic_regularity": (0.0, 1.0),
    "syncopation": (0.0, 1.0),
    "mean_pitch": (48.0, 36.0),  # MIDI 48 -> 0, MIDI 84 -> 1
    "pitch_range": (0.0, 24.0),  # two octaves -> 1
    "contour": (-1.0, 2.0),  # [-1, 1] -> [0, 1]
    "repetition": (0.0, 1.0),
    "iv_unison": (0.0, 1.0),
    "iv_step": (0.0, 1.0),
    "iv_third": (0.0, 1.0),
    "iv_fourth_fifth": (0.0, 1.0),
    "iv_large": (0.0, 1.0),
    "mean_velocity": (0.0, 127.0),
    "velocity_variation": (0.0, 0.5),  # std/mean of 0.5 -> 1
    "dissonance_proxy": (0.0, 1.0),
    "tempo": (60.0, 120.0),  # 60 bpm -> 0, 180 bpm -> 1
}


def _sorted_arrays(phrase: Phrase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    notes = sorted(phrase.notes, key=lambda n: (n.onset, n.pitch))
    onsets = np.array([n.onset for n in notes], dtype=float)
    pitches = np.array([n.pitch for n in notes], dtype=float)
    vels = np.array([n.velocity for n in notes], dtype=float)
    return onsets, pitches, vels


def _regularity(iois: np.ndarray) -> float:
    if len(iois) < 2 or float(np.mean(iois)) <= 1e-9:
        return 1.0
    return float(np.clip(1.0 - np.std(iois) / np.mean(iois), 0.0, 1.0))


def _syncopation(onsets: np.ndarray) -> float:
    if len(onsets) == 0:
        return 0.0
    dist = np.abs(onsets / 0.5 - np.round(onsets / 0.5)) * 0.5  # distance to grid in beats
    return float(np.mean(dist > GRID_TOLERANCE))


def _contour(onsets: np.ndarray, pitches: np.ndarray) -> float:
    if len(onsets) < 2:
        return 0.0
    span = float(onsets.max() - onsets.min())
    if span <= 1e-9:
        return 0.0
    slope = float(np.polyfit(onsets, pitches, 1)[0])  # semitones per beat
    return float(np.clip(slope * span / 12.0, -1.0, 1.0))


def _contour_class(pitches: np.ndarray, contour: float) -> str:
    """Deterministic rules, applied in order:
    1. range <= 2 semitones                              -> flat
    2. >= 3 direction changes and changes >= 40% of the possible ones -> zigzag
    3. interior max >= 3 above both endpoints (and above the interior min's depth) -> arch
    4. interior min >= 3 below both endpoints            -> valley
    5. contour > 0.15 -> rising; contour < -0.15 -> falling
    6. otherwise zigzag if >= 2 direction changes else flat
    """
    if len(pitches) < 2 or float(pitches.max() - pitches.min()) <= 2:
        return "flat"
    diffs = np.diff(pitches)
    signs = np.sign(diffs[diffs != 0])
    changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    if changes >= 3 and changes >= 0.4 * max(1, len(signs) - 1):
        return "zigzag"
    first, last = float(pitches[0]), float(pitches[-1])
    interior = pitches[1:-1]
    peak_excess = float(interior.max()) - max(first, last) if len(interior) else 0.0
    trough_excess = min(first, last) - float(interior.min()) if len(interior) else 0.0
    if peak_excess >= 3 and peak_excess >= trough_excess:
        return "arch"
    if trough_excess >= 3:
        return "valley"
    if contour > 0.15:
        return "rising"
    if contour < -0.15:
        return "falling"
    return "zigzag" if changes >= 2 else "flat"


def _repetition(intervals: np.ndarray, iois: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    tokens = [(int(iv), int(round(ioi / IOI_QUANTUM))) for iv, ioi in zip(intervals, iois, strict=True)]
    return (len(tokens) - len(set(tokens))) / len(tokens)


def _interval_histogram(intervals: np.ndarray) -> list[float]:
    if len(intervals) == 0:
        return [0.0] * 5
    a = np.abs(intervals)
    classes = [a == 0, (a >= 1) & (a <= 2), (a >= 3) & (a <= 4), (a >= 5) & (a <= 7), a > 7]
    return [float(np.mean(c)) for c in classes]


def _dissonance(intervals: np.ndarray) -> float:
    if len(intervals) == 0:
        return 0.0
    pcs = np.abs(intervals).astype(int) % 12
    return float(np.mean([pc in DISSONANT_PITCH_CLASSES for pc in pcs]))


def normalise(raw: Mapping[str, float]) -> list[float]:
    """Map raw feature values (keyed by FEATURE_NAMES) to [0, 1] with NORMALISATION."""
    out = []
    for name in FEATURE_NAMES:
        offset, scale = NORMALISATION[name]
        out.append(float(np.clip((raw[name] - offset) / scale, 0.0, 1.0)))
    return out


def apply_weights(vector: Sequence[float], weights: Mapping[str, float] | None) -> list[float]:
    """Element-wise multiply by `music.feature_weights` (missing names -> 1.0)."""
    if not weights:
        return [float(v) for v in vector]
    return [float(v) * float(weights.get(name, 1.0)) for v, name in zip(vector, FEATURE_NAMES, strict=True)]


def symbolic_features(phrase: Phrase, feature_weights: Mapping[str, float] | None = None) -> SymbolicFeatures:
    onsets, pitches, vels = _sorted_arrays(phrase)
    n = len(onsets)
    duration_s = phrase.duration_seconds
    iois = np.diff(onsets)
    intervals = np.diff(pitches)
    contour = _contour(onsets, pitches)
    hist = _interval_histogram(intervals)
    mean_vel = float(np.mean(vels)) if n else 0.0
    vel_var = float(np.std(vels) / mean_vel) if n and mean_vel > 0 else 0.0
    raw = {
        "note_density": n / duration_s if duration_s > 0 else 0.0,
        "rhythmic_regularity": _regularity(iois),
        "syncopation": _syncopation(onsets),
        "mean_pitch": float(np.mean(pitches)) if n else 0.0,
        "pitch_range": float(pitches.max() - pitches.min()) if n else 0.0,
        "contour": contour,
        "repetition": _repetition(intervals, iois),
        "iv_unison": hist[0],
        "iv_step": hist[1],
        "iv_third": hist[2],
        "iv_fourth_fifth": hist[3],
        "iv_large": hist[4],
        "mean_velocity": mean_vel,
        "velocity_variation": vel_var,
        "dissonance_proxy": _dissonance(intervals),
        "tempo": float(phrase.tempo_bpm),
    }
    vector = apply_weights(normalise(raw), feature_weights)
    return SymbolicFeatures(
        note_count=n,
        duration_seconds=round(duration_s, 6),
        note_density=round(raw["note_density"], 6),
        rhythmic_regularity=round(raw["rhythmic_regularity"], 6),
        syncopation=round(raw["syncopation"], 6),
        mean_pitch=round(raw["mean_pitch"], 6),
        pitch_range=int(raw["pitch_range"]),
        contour=round(contour, 6),
        contour_class=_contour_class(pitches, contour),  # type: ignore[arg-type]
        repetition=round(raw["repetition"], 6),
        interval_histogram=[round(h, 6) for h in hist],
        mean_velocity=round(mean_vel, 6),
        velocity_variation=round(vel_var, 6),
        dissonance_proxy=round(raw["dissonance_proxy"], 6),
        tempo_bpm=float(phrase.tempo_bpm),
        vector=[round(v, 6) for v in vector],
    )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """cos(a, b); 0.0 if either vector has zero norm."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def centred(vector: Sequence[float], centre: float = 0.5) -> np.ndarray:
    """x - 0.5: features centred on the middle of their normalised range.

    Used for partner similarity and as the receiver learner's input so that the
    shared "average phrase" component does not dominate dot products.
    """
    return np.asarray(vector, dtype=float) - centre
