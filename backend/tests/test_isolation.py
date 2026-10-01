"""Information isolation: the receiver never sees the target."""

from dataclasses import fields
from typing import Any

from resonance.agents.policy import LocalPolicy
from resonance.policy_types import ReceiverContext
from resonance.schemas import ConditionSpec
from resonance.session import Session


def keys_recursive(obj: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            out |= keys_recursive(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            out |= keys_recursive(v)
    return out


class SpyPolicy(LocalPolicy):
    def __init__(self) -> None:
        self.contexts: list[ReceiverContext] = []

    def choose_pattern(self, ctx):  # noqa: ANN001, ANN201
        self.contexts.append(ctx)
        return super().choose_pattern(ctx)


def _run(config, condition, episodes=40):
    s = Session(config, condition, seed=11)
    spies = []
    for a in s.agents:
        a.policy = SpyPolicy()
        spies.append(a.policy)
    s.step(episodes)
    return s, [c for sp in spies for c in sp.contexts]


def test_receiver_context_has_no_target_field():
    assert not any("target" in f.name for f in fields(ReceiverContext))


def test_no_target_reaches_receiver(config, full):
    s, contexts = _run(config, full)
    assert len(contexts) == 40
    targets = {e.step: e.payload["target_id"] for e in s.events if e.type == "target_assigned"}
    for ev in s.events:
        if ev.type == "phrase_received":
            assert set(ev.payload) == {"observation"}
            assert not any("target" in k for k in keys_recursive(ev.payload))
    for a in s.agents:
        for entry in a.information_received:
            if entry["role"] == "receiver":
                assert not any("target" in k for k in keys_recursive(entry))
    for ctx in contexts:
        assert not any("target" in k for k in vars(ctx))
        dumped = ctx.observation.model_dump(mode="json")
        assert not any("target" in k for k in keys_recursive(dumped))
        # the only per-episode private value is the target pattern; it must not be attached to the context
        assert ctx.patterns == s.task.patterns()  # public set only
        assert ctx.step in targets


def test_target_assigned_is_sender_private(config, full):
    s, _ = _run(config, full, episodes=6)
    assert all(e.visibility == "sender_private" for e in s.events if e.type == "target_assigned")


def test_symbol_observation_only_symbol_id(config):
    s, contexts = _run(config, ConditionSpec(name="symbol"), episodes=20)
    for ev in s.events:
        if ev.type == "phrase_received":
            obs = ev.payload["observation"]
            assert obs["channel"] == "symbol" and obs["phrase"] is None and obs["features"] is None
            assert isinstance(obs["symbol_id"], int)
    assert all(c.observation.features is None and c.observation.phrase is None for c in contexts)
