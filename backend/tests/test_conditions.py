import numpy as np
import pytest

from resonance.conditions import describe, effective_settings, standard_conditions
from resonance.schemas import ConditionSpec
from resonance.session import Session


def test_six_conditions_build_and_run(config):
    specs = standard_conditions()
    assert {c.name.value for c in specs} == {"full", "no_history", "state_fixed", "state_decoupled", "symbol", "perturbed"}
    for spec in specs:
        assert spec.description
        s = Session(config, spec, seed=4)
        s.step(20)
        assert len(s.scores) == 20


def test_description_autofilled(config):
    s = Session(config, ConditionSpec(name="perturbed", perturbation="pitch_shuffle"), seed=1)
    assert "pitch_shuffle" in s.condition.description
    assert describe(ConditionSpec(name="symbol")).startswith("Symbol baseline")


@pytest.mark.parametrize(
    "name,key,value",
    [("no_history", "learning_enabled", False), ("no_history", "memory_capacity", 0), ("state_fixed", "state_frozen", True),
     ("state_decoupled", "coupling_enabled", False), ("symbol", "channel", "symbol")],
)
def test_settings_table(config, name, key, value):
    assert effective_settings(ConditionSpec(name=name), config)[key] == value


def test_condition_effects(config):
    nh = Session(config, ConditionSpec(name="no_history"), seed=4)
    nh.step(20)
    assert all(np.all(a.receiver_learner.W == 0) and len(a.memory) == 0 for a in nh.agents)
    sf = Session(config, ConditionSpec(name="state_fixed"), seed=4)
    sf.step(20)
    assert all(a.state == a.baseline for a in sf.agents)
    sd = Session(config, ConditionSpec(name="state_decoupled"), seed=4)
    sd.step(20)
    traces = [e.payload["trace"] for e in sd.events if e.type == "action_chosen"]
    assert all(t["temperature"] == t["temperature_base"] for t in traces)
    assert any(a.state != a.baseline for a in sd.agents)


def test_symbol_observation_has_no_features(config):
    s = Session(config, ConditionSpec(name="symbol"), seed=4)
    s.step(10)
    for e in s.events:
        if e.type == "phrase_received":
            assert e.payload["observation"]["features"] is None
        if e.type == "state_update":
            assert e.payload["inputs"]["music_drive"]["activation"] == 0.0
            assert e.payload["inputs"]["partner_similarity"] is None
    assert s.agents[0].receiver_learner.W.shape == (4, 8)
