import numpy as np

from resonance.schemas import EpisodeSummary, Intervention, RunExport
from resonance.session import Session


def _summaries(events):
    return [EpisodeSummary.model_validate(e.payload) for e in events if e.type == "episode_complete"]


def test_export_import_replay(config, full):
    s = Session(config, full, seed=8)
    s.step(10)
    s.intervene(Intervention(kind="reset_state", agent_id="A"))
    s.intervene(Intervention(kind="set_coupling", agent_id="B", value=False))
    s.step(10)
    s.intervene(Intervention(kind="reset_memory", agent_id="B"))
    s.step(5)
    export = RunExport.model_validate_json(s.export().model_dump_json())
    r = Session.from_export(export)
    assert r.mode == "replay" and r.status == "ready" and r.step_index == 0
    first = r.replay_step(3)
    assert [e.seq for e in first] == [e.seq for e in s.events if e.step < 3]
    r.replay_step(100)
    assert r.status == "finished"
    assert _summaries(r.events) == _summaries(s.events)
    assert r.scores == s.scores
    for a, b in zip(r.agents, s.agents, strict=True):
        assert a.state == b.state
        assert a.coupling_enabled == b.coupling_enabled
        np.testing.assert_allclose(a.receiver_learner.W, b.receiver_learner.W, atol=1e-6)
        np.testing.assert_allclose(a.sender_learner.Q, b.sender_learner.Q, atol=1e-6)
        assert len(a.memory) == len(b.memory)
    snap_r, snap_s = r.snapshot(), s.snapshot()
    assert snap_r.metrics == snap_s.metrics and snap_r.current_episode == snap_s.current_episode


def test_partial_replay_export_contains_everything(config, full):
    s = Session(config, full, seed=8)
    s.step(6)
    r = Session.from_export(s.export())
    r.replay_step(2)
    exp = r.export()
    assert len(exp.events) == len(s.events) and exp.final_snapshot.metrics == s.snapshot().metrics
