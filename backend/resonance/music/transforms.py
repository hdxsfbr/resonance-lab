"""Channel perturbations (transforms) applied to a phrase.

Which transforms preserve the sender's message?
The receiver's learner reads the normalised feature vector, so "message" here is
whatever part of phi(phrase) distinguishes one motif from another.

  transpose (+5 semitones)   PRESERVES intervals, contour, rhythm, dynamics; shifts mean_pitch only.
  velocity_flatten (all 80)  PRESERVES pitch and rhythm; removes dynamics (mean_velocity,
                             velocity_variation) - mostly preserves the message.
  tempo_shift (x1.25)        PRESERVES beat-relative structure (onsets in beats, intervals);
                             changes note_density and tempo features.
  contour_invert             DESTROYS contour (mirror around the mean pitch); keeps rhythm,
                             interval sizes (up to rounding/clipping) and dynamics.
  rhythm_shuffle             DESTROYS rhythmic content: permutes inter-onset intervals
                             (regularity of the IOI multiset is kept, order and syncopation change).
  pitch_shuffle              DESTROYS contour and interval sequence: permutes pitch order;
                             keeps rhythm, pitch set and dynamics.

All randomness comes from the `rng` argument (the session's `rng_perturb` stream).
"""

from __future__ import annotations

import numpy as np

from resonance.schemas import Note, Perturbation, Phrase, PhraseOrigin

TRANSPOSE_SEMITONES = 5
FLAT_VELOCITY = 80
TEMPO_FACTOR = 1.25
INVERT_PITCH_BOUNDS = (36, 96)
MAX_SHUFFLE_TRIES = 8

PRESERVES_MESSAGE: dict[str, bool] = {
    "none": True,
    "transpose": True,
    "velocity_flatten": True,
    "tempo_shift": True,
    "contour_invert": False,
    "rhythm_shuffle": False,
    "pitch_shuffle": False,
}


def _sorted_notes(phrase: Phrase) -> list[Note]:
    return sorted(phrase.notes, key=lambda n: (n.onset, n.pitch))


def _permutation(n: int, rng: np.random.Generator, values: list[float]) -> np.ndarray:
    """A permutation that changes the value ORDER when that is possible."""
    perm = np.arange(n)
    if n < 2 or len(set(values)) < 2:
        return perm
    for _ in range(MAX_SHUFFLE_TRIES):
        perm = rng.permutation(n)
        if any(values[i] != values[j] for i, j in enumerate(perm)):
            return perm
    return perm


def _transpose(notes: list[Note]) -> list[Note]:
    return [n.model_copy(update={"pitch": int(np.clip(n.pitch + TRANSPOSE_SEMITONES, 0, 127))}) for n in notes]


def _velocity_flatten(notes: list[Note]) -> list[Note]:
    return [n.model_copy(update={"velocity": FLAT_VELOCITY}) for n in notes]


def _contour_invert(notes: list[Note]) -> list[Note]:
    if not notes:
        return notes
    mean = float(np.mean([n.pitch for n in notes]))
    lo, hi = INVERT_PITCH_BOUNDS
    return [n.model_copy(update={"pitch": int(np.clip(round(2 * mean - n.pitch), lo, hi))}) for n in notes]


def _rhythm_shuffle(notes: list[Note], length_beats: float, rng: np.random.Generator) -> list[Note]:
    """Permute the inter-onset intervals; first onset kept; duration_i <= new IOI_i."""
    if len(notes) < 3:
        return notes
    onsets = [n.onset for n in notes]
    iois = list(np.diff(onsets))
    perm = _permutation(len(iois), rng, iois)
    new_iois = [iois[j] for j in perm]
    new_onsets = [onsets[0]] + list(onsets[0] + np.cumsum(new_iois))
    out = []
    for i, n in enumerate(notes):
        limit = new_iois[i] if i < len(new_iois) else max(length_beats - new_onsets[i], n.duration)
        dur = max(1e-3, min(n.duration, limit))
        out.append(n.model_copy(update={"onset": round(float(new_onsets[i]), 4), "duration": round(dur, 4)}))
    return out


def _pitch_shuffle(notes: list[Note], rng: np.random.Generator) -> list[Note]:
    pitches = [n.pitch for n in notes]
    perm = _permutation(len(pitches), rng, [float(p) for p in pitches])
    return [n.model_copy(update={"pitch": pitches[j]}) for n, j in zip(notes, perm, strict=True)]


def apply_transform(phrase: Phrase, kind: Perturbation, rng: np.random.Generator | None = None) -> Phrase:
    """Return a NEW phrase with `kind` applied (origin.kind="transformed"). 'none' returns the input."""
    if kind == "none":
        return phrase
    rng = rng if rng is not None else np.random.default_rng(0)
    notes = _sorted_notes(phrase)
    tempo = phrase.tempo_bpm
    if kind == "transpose":
        notes = _transpose(notes)
    elif kind == "velocity_flatten":
        notes = _velocity_flatten(notes)
    elif kind == "tempo_shift":
        tempo = min(400.0, round(phrase.tempo_bpm * TEMPO_FACTOR, 3))
    elif kind == "contour_invert":
        notes = _contour_invert(notes)
    elif kind == "rhythm_shuffle":
        notes = _rhythm_shuffle(notes, phrase.length_beats, rng)
    elif kind == "pitch_shuffle":
        notes = _pitch_shuffle(notes, rng)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown transform {kind!r}")
    length = max(phrase.length_beats, max((n.onset + n.duration for n in notes), default=0.0))
    return phrase.model_copy(
        update={
            "id": f"{phrase.id}~{kind}",
            "notes": notes,
            "tempo_bpm": tempo,
            "length_beats": length,
            "origin": PhraseOrigin(
                kind="transformed",
                agent_id=phrase.origin.agent_id,
                source_phrase_id=phrase.id,
                transform=kind,
            ),
            "tags": [*phrase.tags, f"transform:{kind}"],
        }
    )
