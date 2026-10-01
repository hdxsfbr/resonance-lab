"""Motif banks: the sender's action space.

A motif is a short, fixed note pattern on an 8-beat phrase. The sender chooses a
motif INDEX; `generate()` turns it into a concrete `Phrase` at a tempo, on an
instrument, with optional expressive modulation (tempo multiplier / velocity
offset, derived from engineered state when coupling is enabled) and small
random jitter drawn from the dedicated `rng_gen` stream.

The "default" bank has 8 motifs drawn from C major / C major pentatonic around
MIDI 60-79. They are hand-designed to have DISTINCT measured feature profiles
(density, contour, rhythm regularity, syncopation, interval content, dynamics,
dissonance proxy, characteristic tempo) so that a linear learner over the
normalised feature vector can separate them, while related variants (e.g. a
transposition) stay close in feature space.

Phrase ids are deterministic: f"ph-{step}-{agent}-{motif}" so replays line up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resonance.schemas import Instrument, Note, Phrase, PhraseOrigin

# (onset_beats, duration_beats, midi_pitch, velocity)
NoteSpec = tuple[float, float, int, int]


@dataclass(frozen=True)
class MotifSpec:
    name: str
    description: str
    base_tempo: float  # characteristic tempo in bpm (8 beats -> 3.4-5.7 s)
    notes: tuple[NoteSpec, ...]


def _seq(onsets: list[float], durs: list[float] | float, pitches: list[int], vels: list[int] | int) -> tuple[NoteSpec, ...]:
    n = len(onsets)
    d = durs if isinstance(durs, list) else [durs] * n
    v = vels if isinstance(vels, list) else [vels] * n
    assert len(d) == len(pitches) == len(v) == n
    return tuple((float(o), float(dd), int(p), int(vv)) for o, dd, p, vv in zip(onsets, d, pitches, v, strict=True))


_THIRD = 1.0 / 3.0

DEFAULT_BANK: tuple[MotifSpec, ...] = (
    MotifSpec(
        "ascent",
        "Even quarter notes climbing the C major scale; medium loud; perfectly regular.",
        100.0,
        _seq([0, 1, 2, 3, 4, 5, 6, 7], 0.9, [60, 62, 64, 65, 67, 69, 71, 72], 80),
    ),
    MotifSpec(
        "cascade",
        "Five loud notes falling by fourths/fifths from G5 to G3 in an uneven long-short-long rhythm; sparse.",
        90.0,
        _seq([0, 1.5, 2, 4, 6], [1.4, 0.45, 1.9, 1.9, 1.9], [79, 74, 67, 62, 55], 112),
    ),
    MotifSpec(
        "flutter",
        "Sixteen soft eighth notes rising then falling through the pentatonic scale (arch); dense.",
        132.0,
        _seq(
            [i * 0.5 for i in range(16)],
            0.45,
            [60, 62, 64, 67, 69, 72, 74, 76, 76, 74, 72, 69, 67, 64, 62, 60],
            52,
        ),
    ),
    MotifSpec(
        "hop",
        "Uneven, mostly off-the-eighth-grid onsets zig-zagging by thirds/fifths with strong accents.",
        116.0,
        _seq(
            [0, 0.75, 1.25, 2.25, 3.0, 3.75, 4.25, 5.25, 6.0, 6.75],
            [0.6, 0.45, 0.9, 0.6, 0.6, 0.45, 0.9, 0.6, 0.6, 0.6],
            [64, 67, 64, 69, 64, 67, 62, 67, 64, 69],
            [110, 58, 110, 58, 110, 58, 110, 58, 110, 58],
        ),
    ),
    MotifSpec(
        "chime",
        "Three long, soft, high notes far apart with wide leaps (falls then rises).",
        84.0,
        _seq([0, 3, 5.5], [2.6, 2.2, 2.4], [79, 67, 76], 48),
    ),
    MotifSpec(
        "pulse",
        "Twelve loud repeated G4 eighth notes then a rest: flat, highly repetitive, regular.",
        140.0,
        _seq([i * 0.5 for i in range(12)], 0.3, [67] * 12, [104, 96] * 6),
    ),
    MotifSpec(
        "swirl",
        "Triplet-spaced notes dipping by semitones and tritones (valley); off the eighth grid.",
        108.0,
        _seq(
            [i * (2 * _THIRD) for i in range(10)],
            0.55,
            [72, 71, 65, 64, 60, 60, 64, 65, 71, 72],
            [70, 74, 78, 82, 86, 86, 82, 78, 74, 70],
        ),
    ),
    MotifSpec(
        "stomp",
        "Low dotted rhythm (long-short pairs) with a strong crescendo; mixed intervals.",
        96.0,
        _seq(
            [0, 0.75, 1, 1.75, 2, 2.75, 3, 3.75, 4, 6],
            [0.7, 0.25, 0.7, 0.25, 0.7, 0.25, 0.7, 0.25, 1.9, 1.9],
            [60, 60, 62, 62, 64, 64, 67, 67, 55, 60],
            [40, 50, 58, 66, 74, 82, 90, 98, 110, 122],
        ),
    ),
)

BANKS: dict[str, tuple[MotifSpec, ...]] = {"default": DEFAULT_BANK}

# Generation jitter scales (multiplied by `music.generation_noise`, default 0.05):
VELOCITY_JITTER_SCALE = 100.0  # velocity sd = 100 * noise  (5 units at noise 0.05)
ONSET_JITTER_SCALE = 0.2  # onset sd (beats) = 0.2 * noise (0.01 beat at noise 0.05)


def get_bank(name: str = "default") -> tuple[MotifSpec, ...]:
    if name not in BANKS:
        raise KeyError(f"unknown motif bank {name!r}; available: {sorted(BANKS)}")
    return BANKS[name]


def motif_names(bank: str = "default", n: int | None = None) -> list[str]:
    specs = get_bank(bank)
    return [m.name for m in specs[: (n or len(specs))]]


def motif_id(index: int, bank: str = "default") -> str:
    """Stable identifier of a base motif, e.g. 'm3-hop'."""
    return f"m{index}-{get_bank(bank)[index].name}"


def motif_index_from_id(mid: str | None, bank: str = "default") -> int | None:
    """Inverse of motif_id (returns None for non-bank phrases)."""
    if not mid or not mid.startswith("m") or "-" not in mid:
        return None
    head = mid[1:].split("-", 1)[0]
    if not head.isdigit():
        return None
    idx = int(head)
    return idx if idx < len(get_bank(bank)) else None


def generate(
    motif_index: int,
    tempo_bpm: float | None = None,
    instrument: Instrument = "pluck",
    rng_gen: np.random.Generator | None = None,
    noise: float = 0.0,
    tempo_multiplier: float = 1.0,
    velocity_offset: float = 0,
    agent_id: str | None = None,
    *,
    step: int = 0,
    bank: str = "default",
    length_beats: float = 8.0,
    tempo_bounds: tuple[float, float] = (80.0, 160.0),
    origin_kind: str = "agent",
    phrase_id: str | None = None,
) -> Phrase:
    """Build a concrete Phrase from base motif `motif_index`.

    tempo = clip((tempo_bpm or motif.base_tempo) * tempo_multiplier, *tempo_bounds)
    velocity_i = clip(round(v_i + velocity_offset + N(0, 100*noise)), 1, 127)
    onset_i    = max(0, o_i + N(0, 0.2*noise))      (beats; both jitters from rng_gen only)
    """
    spec = get_bank(bank)[motif_index]
    base = spec.base_tempo if tempo_bpm is None else float(tempo_bpm)
    tempo = float(np.clip(base * tempo_multiplier, tempo_bounds[0], tempo_bounds[1]))
    n = len(spec.notes)
    if rng_gen is not None and noise > 0:
        vel_jit = rng_gen.normal(0.0, VELOCITY_JITTER_SCALE * noise, n)
        onset_jit = rng_gen.normal(0.0, ONSET_JITTER_SCALE * noise, n)
    else:
        vel_jit = np.zeros(n)
        onset_jit = np.zeros(n)
    notes: list[Note] = []
    for i, (onset, dur, pitch, vel) in enumerate(spec.notes):
        o = max(0.0, onset + float(onset_jit[i]))
        v = int(np.clip(round(vel + velocity_offset + float(vel_jit[i])), 1, 127))
        notes.append(Note(pitch=pitch, onset=round(o, 4), duration=dur, velocity=v))
    pid = phrase_id or f"ph-{step}-{agent_id or 'x'}-{motif_index}"
    return Phrase(
        id=pid,
        notes=notes,
        tempo_bpm=round(tempo, 3),
        length_beats=length_beats,
        instrument=instrument,
        origin=PhraseOrigin(kind=origin_kind, agent_id=agent_id),  # type: ignore[arg-type]
        motif_id=motif_id(motif_index, bank),
        tags=[spec.name],
    )


def base_phrases(bank: str = "default", instrument: Instrument = "pluck", n: int | None = None) -> list[Phrase]:
    """The bank at each motif's characteristic tempo, without jitter (used by /api/motifs and probes)."""
    specs = get_bank(bank)[: (n or len(get_bank(bank)))]
    return [
        generate(i, None, instrument, None, 0.0, origin_kind="preset", phrase_id=f"motif-{motif_id(i, bank)}", bank=bank)
        for i in range(len(specs))
    ]
