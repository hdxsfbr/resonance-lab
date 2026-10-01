import io
import math
import wave

import numpy as np
import pytest

from resonance.music.motifs import base_phrases
from resonance.music.synth import (
    DEFAULT_SR,
    RELEASE,
    SYNTH_VERSION,
    TAIL_SECONDS,
    audio_features,
    render,
    wav_bytes,
)


@pytest.mark.parametrize("instrument", ["sine", "triangle", "square", "pluck", "bell", "marimba"])
def test_render_all_instruments(instrument):
    p = base_phrases(instrument=instrument)[4]
    x = render(p)
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) < 1.0 and np.max(np.abs(x)) > 0.01


def test_render_length():
    p = base_phrases()[0]
    spb = 60.0 / p.tempo_bpm
    end = max(p.length_beats * spb, max((n.onset + n.duration) * spb + RELEASE for n in p.notes))
    assert len(render(p)) == math.ceil((end + TAIL_SECONDS) * DEFAULT_SR)


def test_wav_header():
    data = wav_bytes(base_phrases()[2])
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == DEFAULT_SR
        assert w.getnframes() == len(render(base_phrases()[2]))


def test_audio_features():
    for p in base_phrases(instrument="marimba"):
        a = audio_features(p)
        assert a.synth_version == SYNTH_VERSION
        assert 0 < a.rms <= a.peak < 1.0
        assert 100 < a.spectral_centroid_hz < 5000
        assert abs(a.onset_count_estimate - len(p.notes)) <= 1
