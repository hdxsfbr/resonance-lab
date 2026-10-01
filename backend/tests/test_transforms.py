import numpy as np
import pytest

from resonance.music.features import symbolic_features
from resonance.music.motifs import base_phrases
from resonance.music.transforms import PRESERVES_MESSAGE, apply_transform


def _sorted(p):
    return sorted(p.notes, key=lambda n: (n.onset, n.pitch))


def test_transpose_preserves_intervals_and_rhythm():
    p = base_phrases()[3]
    t = apply_transform(p, "transpose")
    a, b = _sorted(p), _sorted(t)
    assert [n.pitch + 5 for n in a] == [n.pitch for n in b]
    assert [(n.onset, n.duration, n.velocity) for n in a] == [(n.onset, n.duration, n.velocity) for n in b]
    fa, fb = symbolic_features(p), symbolic_features(t)
    assert fa.interval_histogram == fb.interval_histogram and fa.contour == fb.contour
    assert fb.mean_pitch == pytest.approx(fa.mean_pitch + 5)
    assert t.origin.kind == "transformed" and t.origin.source_phrase_id == p.id and t.origin.transform == "transpose"


def test_rhythm_shuffle_keeps_count_changes_iois():
    p = base_phrases()[7]  # stomp: uneven IOIs
    t = apply_transform(p, "rhythm_shuffle", np.random.default_rng(3))
    a, b = _sorted(p), _sorted(t)
    assert len(a) == len(b)
    ioi_a, ioi_b = np.diff([n.onset for n in a]), np.diff([n.onset for n in b])
    assert sorted(np.round(ioi_a, 4)) == sorted(np.round(ioi_b, 4))  # same multiset
    assert not np.allclose(ioi_a, ioi_b)  # different order
    for n, nxt in zip(b, b[1:], strict=False):
        assert n.duration <= nxt.onset - n.onset + 1e-6


def test_other_transforms():
    p = base_phrases()[6]
    assert {n.velocity for n in apply_transform(p, "velocity_flatten").notes} == {80}
    assert apply_transform(p, "tempo_shift").tempo_bpm == pytest.approx(p.tempo_bpm * 1.25)
    inv = apply_transform(p, "contour_invert")
    mean = np.mean([n.pitch for n in p.notes])
    assert [n.pitch for n in _sorted(inv)] == [int(np.clip(round(2 * mean - n.pitch), 36, 96)) for n in _sorted(p)]
    sh = apply_transform(p, "pitch_shuffle", np.random.default_rng(0))
    assert sorted(n.pitch for n in sh.notes) == sorted(n.pitch for n in p.notes)
    assert [n.pitch for n in _sorted(sh)] != [n.pitch for n in _sorted(p)]
    assert apply_transform(p, "none") is p


def test_message_preservation_documented():
    assert PRESERVES_MESSAGE["transpose"] and not PRESERVES_MESSAGE["rhythm_shuffle"]
