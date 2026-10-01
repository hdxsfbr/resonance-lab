"""policy_kind="model" must degrade to the local policy (with a logged model_call event)
whether the WS3 model layer is absent, failing, or configured with provider 'none'."""

import sys

from resonance.agents.policy import LocalPolicy
from resonance.session import Session


def _model_config(config):
    agents = [config.agents[0].model_copy(update={"policy_kind": "model"}), config.agents[1]]
    return config.model_copy(update={"agents": agents})


def test_provider_none_falls_back(config, full):
    s = Session(_model_config(config), full, seed=1)
    assert isinstance(s.agents[0].policy, LocalPolicy)
    calls = [e for e in s.events if e.type == "model_call"]
    assert len(calls) == 1 and calls[0].payload["ok"] is False and calls[0].payload["fallback"] == "local"
    s.step(4)


def test_missing_registry_falls_back(config, full, monkeypatch):
    monkeypatch.setitem(sys.modules, "resonance.models.registry", None)  # import -> ImportError
    s = Session(_model_config(config), full, seed=1)
    calls = [e for e in s.events if e.type == "model_call"]
    assert calls and calls[0].payload["error"].split(":")[0] in {"ImportError", "ModuleNotFoundError"}
    s.step(2)
    assert len(s.scores) == 2
