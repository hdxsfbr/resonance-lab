"""Shared fixtures. Tests never touch the real data/resonance.db."""

from __future__ import annotations

import os
import tempfile

import pytest

# Make sure nothing in the suite writes to the project database.
os.environ.setdefault("RESONANCE_DB", os.path.join(tempfile.mkdtemp(prefix="resonance-test-"), "test.db"))

from resonance.config import load_config  # noqa: E402
from resonance.schemas import ConditionSpec, ExperimentConfig, Note, Phrase  # noqa: E402


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config()


@pytest.fixture
def full() -> ConditionSpec:
    return ConditionSpec()


def make_phrase(pitches: list[int], onsets: list[float] | None = None, velocities: list[int] | None = None,
                tempo: float = 120.0, length: float = 8.0, duration: float = 0.4, pid: str = "t") -> Phrase:
    onsets = onsets if onsets is not None else [i * 0.5 for i in range(len(pitches))]
    velocities = velocities if velocities is not None else [80] * len(pitches)
    notes = [Note(pitch=p, onset=o, duration=duration, velocity=v) for p, o, v in zip(pitches, onsets, velocities, strict=True)]
    return Phrase(id=pid, notes=notes, tempo_bpm=tempo, length_beats=length)
