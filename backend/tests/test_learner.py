import numpy as np
import pytest

from resonance.agents.learner import (
    ReceiverLearner,
    SenderLearner,
    encode_music,
    encode_symbol,
    reference_vector,
)
from resonance.music.features import symbolic_features
from resonance.music.motifs import base_phrases
from resonance.music.transforms import apply_transform
from resonance.schemas import FEATURE_NAMES, LearningConfig


def test_sender_q_moves_toward_score():
    s = SenderLearner(4, 8, LearningConfig(optimistic_init=0.5))
    assert np.all(s.Q == 0.5)
    s.update(1, 3, 1.0, 0.3)
    assert s.Q[1, 3] == pytest.approx(0.65) and s.counts[1, 3] == 1
    for _ in range(50):
        s.update(1, 3, 0.0, 0.3)
    assert s.Q[1, 3] < 0.01 and s.Q[0, 0] == 0.5
    s.reset()
    assert np.all(s.Q == 0.5) and s.counts.sum() == 0


def test_receiver_generalises_to_transposed_motif():
    ref = reference_vector()
    bank = base_phrases()
    learner = ReceiverLearner(len(FEATURE_NAMES), 4, LearningConfig())
    x_train = encode_music(symbolic_features(bank[0]).vector, ref)
    for _ in range(30):
        learner.update(x_train, 1, 1.0, 0.3)
    x_transposed = encode_music(symbolic_features(apply_transform(bank[0], "transpose")).vector, ref)
    x_unrelated = encode_music(symbolic_features(bank[4]).vector, ref)
    assert learner.scores(x_transposed)[1] > learner.scores(x_unrelated)[1] + 0.2
    assert learner.scores(x_train)[1] == pytest.approx(1.0, abs=0.05)


def test_receiver_only_chosen_row_changes_and_reset():
    learner = ReceiverLearner(8, 4, LearningConfig())
    learner.update(encode_symbol(2, 8), 3, 1.0, 0.5)
    assert np.all(learner.W[:3] == 0) and learner.W[3, 2] > 0 and learner.b[3] > 0
    assert learner.updates == 1
    learner.reset()
    assert np.all(learner.W == 0) and np.all(learner.b == 0) and learner.updates == 0


def test_bandit_update_rule_exact():
    cfg = LearningConfig(receiver_l2=0.01)
    learner = ReceiverLearner(2, 2, cfg)
    learner.W[0] = [0.5, -0.5]
    x = np.array([1.0, 2.0])
    q = learner.scores(x)[0]  # -0.5
    learner.update(x, 0, 1.0, 0.1)
    err = 1.0 - q
    np.testing.assert_allclose(learner.W[0], np.array([0.5, -0.5]) + 0.1 * err * x - 0.01 * np.array([0.5, -0.5]))
    assert learner.b[0] == pytest.approx(0.1 * err)
