import pytest

from resonance.agents.agent import Agent
from resonance.agents.state import StateDrive, StateEngine
from resonance.music.features import symbolic_features
from resonance.music.motifs import base_phrases
from resonance.schemas import STATE_DIMS, StateConfig, StateVector

BASE = StateVector(activation=0.5, expected_value=0.5, uncertainty=0.5, affiliation=0.0)
LOUD_DENSE = symbolic_features(base_phrases()[5])  # pulse: dense and loud


def test_bounds_hold_under_extreme_inputs():
    cfg = StateConfig(acoustic_activation_gain=10.0, outcome_gain={"expected_value": 5, "affiliation": 5})
    s = BASE
    for i in range(200):
        s, _ = StateEngine.update(s, StateDrive(score=float(i % 2), heard=LOUD_DENSE, partner_similarity=1.0), cfg, 3.0)
        assert 0 <= s.activation <= 1 and 0 <= s.expected_value <= 1 and 0 <= s.uncertainty <= 1
        assert -1 <= s.affiliation <= 1


def test_inertia_one_blocks_drive():
    cfg = StateConfig(inertia={"activation": 1.0}, decay={"activation": 0.0})
    s, _ = StateEngine.update(BASE, StateDrive(score=1.0, heard=LOUD_DENSE), cfg)
    assert s.activation == pytest.approx(0.5)


def test_decay_to_baseline():
    cfg = StateConfig()
    s = StateVector(activation=0.95, expected_value=0.5, uncertainty=0.5, affiliation=0.0)
    for _ in range(100):
        s, _ = StateEngine.update(s, StateDrive(), cfg)
    assert s.activation == pytest.approx(0.5, abs=1e-3)


def test_frozen_returns_state_unchanged():
    s, rec = StateEngine.update(BASE, StateDrive(score=1.0, heard=LOUD_DENSE), StateConfig(), frozen=True)
    assert s == BASE and rec.frozen and not rec.decay_applied


def test_acoustic_switch_off_zero_music_drive():
    on_state, on = StateEngine.update(BASE, StateDrive(heard=LOUD_DENSE), StateConfig())
    off_state, off = StateEngine.update(BASE, StateDrive(heard=LOUD_DENSE), StateConfig(acoustic_activation_enabled=False))
    assert on.music_drive["activation"] > 0 and any("hand-authored" in n for n in on.notes)
    assert all(v == 0.0 for v in off.music_drive.values())
    assert off_state.activation == pytest.approx(0.5) and on_state.activation > 0.5


def test_max_step_bounds_delta():
    cfg = StateConfig(max_step={"activation": 0.05, "expected_value": 0.05, "uncertainty": 0.05, "affiliation": 0.05},
                      acoustic_activation_gain=50.0, outcome_gain={"expected_value": 50.0, "affiliation": 50.0})
    s, _ = StateEngine.update(BASE, StateDrive(score=1.0, heard=LOUD_DENSE, partner_similarity=1.0), cfg)
    for k in STATE_DIMS:
        assert abs(getattr(s, k) - getattr(BASE, k)) <= 0.05 + 1e-9


def test_prediction_error_uses_pre_update_expectation():
    state = BASE.model_copy(update={"expected_value": 0.2})
    s, rec = StateEngine.update(state, StateDrive(score=1.0), StateConfig())
    assert rec.prediction_error == pytest.approx(0.8)
    # expected_value moves toward the score by (1 - inertia) * gain * pe, plus decay toward 0.5
    assert s.expected_value == pytest.approx(0.2 + 0.2 * 0.8 + 0.02 * 0.3)


def test_baseline_override_respected(config):
    params = config.agents[0].model_copy(update={"baseline_override": StateVector(activation=0.8, expected_value=0.3, uncertainty=0.2, affiliation=0.1)})
    agent = Agent(params, config)
    assert agent.state.activation == 0.8
    s = agent.state.model_copy(update={"activation": 0.2})
    for _ in range(100):
        s, _ = StateEngine.update(s, StateDrive(), config.state, baseline=agent.baseline)
    assert s.activation == pytest.approx(0.8, abs=1e-3)
