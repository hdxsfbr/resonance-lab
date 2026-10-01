import numpy as np
import pytest

from resonance.agents.policy import expressive_modulation, softmax, temperature
from resonance.schemas import Intervention, StateVector
from resonance.session import Session


def state(u: float = 0.5, a: float = 0.5) -> StateVector:
    return StateVector(activation=a, expected_value=0.5, uncertainty=u, affiliation=0.0)


def test_coupling_on_temperature_tracks_uncertainty(config):
    lo, hi = temperature(state(u=0.2), config, True), temperature(state(u=0.8), config, True)
    assert lo < config.learning.temperature_base < hi
    base, k = config.learning.temperature_base, config.coupling.uncertainty_to_temperature
    assert hi == pytest.approx(base * (1 + k * 0.3))
    # higher activation -> lower temperature
    assert temperature(state(a=0.9), config, True) < temperature(state(a=0.1), config, True)


def test_coupling_off_identity(config):
    for s in (state(u=0.1, a=0.9), state(u=0.9, a=0.1)):
        assert temperature(s, config, False) == config.learning.temperature_base
        assert expressive_modulation(s, config, False) == {"tempo_multiplier": 1.0, "velocity_offset": 0.0}
    mod = expressive_modulation(state(a=1.0), config, True)
    assert mod["tempo_multiplier"] == pytest.approx(1 + config.coupling.activation_to_tempo * 0.5)
    assert mod["velocity_offset"] == pytest.approx(config.coupling.activation_to_velocity * 0.5)


def test_softmax_temperature():
    assert np.allclose(softmax([0.0, 0.0], 0.1), [0.5, 0.5])
    assert softmax([1.0, 0.0], 0.05)[0] > softmax([1.0, 0.0], 1.0)[0]


def _traces(session: Session) -> list[dict]:
    return [e.payload["trace"] for e in session.events if e.type == "action_chosen"]


def test_set_coupling_intervention_changes_trace(config, full):
    s = Session(config, full, seed=5)
    s.step(12)
    assert all(t["coupling_enabled"] for t in _traces(s))
    assert any(t["temperature"] != t["temperature_base"] for t in _traces(s))
    ev = s.intervene(Intervention(kind="set_coupling", value=False))
    assert ev.payload["effective_from_step"] == 12
    s.step(6)
    after = _traces(s)[12:]
    assert all(not t["coupling_enabled"] and t["temperature"] == t["temperature_base"] for t in after)
    sent = [e.payload["trace"] for e in s.events if e.type == "phrase_sent" and e.step >= 12]
    assert all(t["expressive_modulation"] == {"tempo_multiplier": 1.0, "velocity_offset": 0.0} for t in sent)
