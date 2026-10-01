import numpy as np
import pytest

from resonance.music.features import NORMALISATION, cosine_similarity, symbolic_features
from resonance.music.motifs import base_phrases, generate
from resonance.schemas import FEATURE_NAMES
from tests.conftest import make_phrase


def test_deterministic():
    p = generate(3, None, "pluck", np.random.default_rng(1), 0.05, step=4, agent_id="A")
    assert symbolic_features(p) == symbolic_features(p)
    q = generate(3, None, "pluck", np.random.default_rng(1), 0.05, step=4, agent_id="A")
    assert p == q and p.id == "ph-4-A-3"


def test_ranges_and_vector_order():
    assert list(NORMALISATION) == FEATURE_NAMES
    for p in base_phrases():
        f = symbolic_features(p)
        assert len(f.vector) == len(FEATURE_NAMES)
        assert all(0.0 <= v <= 1.0 for v in f.vector)
        for val in (f.rhythmic_regularity, f.syncopation, f.repetition, f.dissonance_proxy):
            assert 0.0 <= val <= 1.0
        assert -1.0 <= f.contour <= 1.0
        assert sum(f.interval_histogram) == pytest.approx(1.0, abs=1e-5)
        assert 2.0 <= p.duration_seconds <= 8.0


def test_bank_profiles_are_distinct():
    vecs = np.array([symbolic_features(p).vector for p in base_phrases()])
    d = np.linalg.norm(vecs[:, None] - vecs[None], axis=-1)
    assert d[np.triu_indices(len(vecs), 1)].min() > 0.3


@pytest.mark.parametrize(
    "pitches,expected",
    [
        ([60, 62, 64, 65, 67, 69], "rising"),
        ([72, 69, 67, 65, 64, 62], "falling"),
        ([60, 64, 67, 72, 67, 64, 60], "arch"),
        ([72, 67, 64, 60, 64, 67, 72], "valley"),
        ([67, 67, 67, 67, 68], "flat"),
        ([60, 67, 60, 67, 60, 67], "zigzag"),
    ],
)
def test_contour_classes(pitches, expected):
    assert symbolic_features(make_phrase(pitches)).contour_class == expected


def test_contour_sign():
    assert symbolic_features(make_phrase([60, 62, 64, 65, 67, 69])).contour > 0.3
    assert symbolic_features(make_phrase([69, 67, 65, 64, 62, 60])).contour < -0.3


def test_dissonance_definition():
    # intervals: +1 (m2), +6 (tritone), +11 (M7), +2 (step) -> 3 of 4 dissonant
    f = symbolic_features(make_phrase([60, 61, 67, 78, 80]))
    assert f.dissonance_proxy == pytest.approx(0.75)
    # compound intervals reduce mod 12: 13 semitones -> pitch class 1 -> dissonant
    assert symbolic_features(make_phrase([60, 73])).dissonance_proxy == pytest.approx(1.0)
    assert symbolic_features(make_phrase([60, 64, 67, 72])).dissonance_proxy == 0.0


def test_syncopation_eighth_grid():
    f = symbolic_features(make_phrase([60, 62, 64, 65], onsets=[0.0, 0.5, 0.75, 1.25]))
    assert f.syncopation == pytest.approx(0.5)
    assert symbolic_features(make_phrase([60, 62, 64], onsets=[0.0, 1.5, 3.0])).syncopation == 0.0


def test_regularity_and_repetition():
    regular = symbolic_features(make_phrase([60] * 6))
    assert regular.rhythmic_regularity == pytest.approx(1.0)
    assert regular.repetition == pytest.approx(4 / 5)
    irregular = symbolic_features(make_phrase([60, 62, 64, 65], onsets=[0, 0.25, 2.0, 2.5]))
    assert irregular.rhythmic_regularity < 0.5


def test_interval_histogram_classes():
    f = symbolic_features(make_phrase([60, 60, 62, 66, 73, 85]))  # 0, 2, 4, 7, 12
    assert f.interval_histogram == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2])


def test_feature_weights_multiply():
    p = base_phrases()[0]
    base = symbolic_features(p).vector
    w = symbolic_features(p, {"tempo": 0.0, "note_density": 2.0}).vector
    assert w[FEATURE_NAMES.index("tempo")] == 0.0
    assert w[0] == pytest.approx(2 * base[0], abs=1e-5)


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
