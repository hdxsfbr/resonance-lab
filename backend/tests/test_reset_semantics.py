import numpy as np

from resonance.schemas import Intervention
from resonance.session import Session


def test_reset_memory_keeps_state(config, full):
    s = Session(config, full, seed=2)
    s.step(30)
    a = s.agent_by_id["A"]
    assert len(a.memory) > 0 and a.receiver_learner.updates > 0
    state_before = a.state
    s.intervene(Intervention(kind="reset_memory", agent_id="A"))
    assert len(a.memory) == 0
    assert np.all(a.receiver_learner.W == 0) and np.all(a.receiver_learner.b == 0)
    assert np.all(a.sender_learner.Q == config.learning.optimistic_init) and a.sender_learner.counts.sum() == 0
    assert a.state == state_before
    # the other agent is untouched
    assert len(s.agent_by_id["B"].memory) > 0


def test_reset_state_keeps_memory(config, full):
    s = Session(config, full, seed=2)
    s.step(30)
    b = s.agent_by_id["B"]
    mem, W, Q = len(b.memory), b.receiver_learner.W.copy(), b.sender_learner.Q.copy()
    assert b.state != b.baseline
    ev = s.intervene(Intervention(kind="reset_state", agent_id="B"))
    assert b.state == b.baseline and ev.payload["state_after"]["B"] == b.baseline.model_dump()
    assert len(b.memory) == mem and np.array_equal(b.receiver_learner.W, W) and np.array_equal(b.sender_learner.Q, Q)


def test_freeze_state_intervention(config, full):
    s = Session(config, full, seed=2)
    s.step(4)
    s.intervene(Intervention(kind="freeze_state", agent_id="A", value=True))
    frozen = s.agent_by_id["A"].state
    s.step(6)
    assert s.agent_by_id["A"].state == frozen
    updates = [e for e in s.events if e.type == "state_update" and e.agent_id == "A" and e.step >= 4]
    assert updates and all(e.payload["inputs"]["frozen"] for e in updates)


def test_session_reset_restarts(config, full):
    s = Session(config, full, seed=2)
    s.step(10)
    fresh = s.reset_copy()
    assert fresh.id == s.id and fresh.step_index == 0 and fresh.events[0].type == "session_reset"
    fresh.step(10)
    assert fresh.scores == s.scores
