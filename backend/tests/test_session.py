import pytest

from resonance.music.motifs import base_phrases
from resonance.schemas import HumanPhraseRequest, Intervention
from resonance.session import Session

EPISODE_TYPES = ["target_assigned", "phrase_sent", "phrase_received", "action_chosen", "outcome",
                 "learning_update", "state_update", "learning_update", "state_update", "episode_complete"]


def test_sixty_episodes(config, full):
    cfg = config.model_copy(update={"episodes": 60})
    s = Session(cfg, full, seed=1)
    assert s.status == "ready" and s.events[0].type == "session_created"
    s.step(30)
    assert s.status == "running"
    s.step(30)
    snap = s.snapshot()
    assert snap.status == "finished" and snap.step == 60
    m = snap.metrics
    assert m.episodes == 60 and len(m.score_history) == 60
    assert 0 <= m.mean_score <= 1 and 0 <= m.success_rate <= 1
    assert m.rolling_score == pytest.approx(sum(m.score_history[-20:]) / 20)
    senders = [e.payload["sender_id"] for e in s.events if e.type == "episode_complete"]
    assert senders == ["A", "B"] * 30
    assert [e.type for e in s.events if e.step == 5] == EPISODE_TYPES
    s.step(2)  # stepping past the configured length is allowed
    assert s.snapshot().step == 62 and s.status == "finished"
    assert len(snap.agents) == 2 and snap.current_episode is not None and len(snap.patterns) == 4


def test_human_phrase_episode(config, full):
    s = Session(config, full, seed=1)
    s.step(3)  # next receiver: A (step 3 odd -> B sends, A receives)
    phrase = base_phrases()[2]
    q_before = s.agent_by_id["B"].sender_learner.Q.copy()
    events = s.human_phrase(HumanPhraseRequest(phrase=phrase, target_id=1))
    sent = next(e for e in events if e.type == "phrase_sent")
    assert sent.payload["phrase"]["origin"]["kind"] == "human" and sent.agent_id is None
    assert next(e for e in events if e.type == "outcome").payload["target_id"] == 1
    roles = {(e.payload["role"], e.agent_id) for e in events if e.type == "learning_update"}
    assert roles == {("receiver", "A")}
    assert (s.agent_by_id["B"].sender_learner.Q == q_before).all()
    assert s.step_index == 4


def test_replay_motif_and_swap_feature(config, full):
    s = Session(config, full, seed=1)
    s.step(2)
    first = next(e.payload["phrase"]["id"] for e in s.events if e.type == "phrase_sent")
    s.intervene(Intervention(kind="replay_motif", phrase_id=first))
    s.intervene(Intervention(kind="swap_feature", transform="transpose"))
    events = s.step(2)
    sent = [e for e in events if e.type == "phrase_sent"]
    assert sent[0].payload["phrase"]["origin"] == {"kind": "replay", "agent_id": "A", "source_phrase_id": first, "transform": None}
    assert sent[0].payload["trace"]["policy_kind"] == "replay"
    assert sent[1].payload["phrase"]["origin"]["kind"] == "agent"
    received = [e.payload["observation"]["phrase"] for e in events if e.type == "phrase_received"]
    assert all(r["origin"]["transform"] == "transpose" for r in received)
    skipped = [e for e in events if e.type == "learning_update" and e.payload["role"] == "sender"][0]
    assert skipped.payload["skipped"] is True


def test_set_param_validates(config, full):
    s = Session(config, full, seed=1)
    s.intervene(Intervention(kind="set_param", path="state.decay.activation", value=0.3))
    assert s.config.state.decay["activation"] == 0.3
    with pytest.raises(ValueError):
        s.intervene(Intervention(kind="set_param", path="learning.temperature_base", value="hot"))
    with pytest.raises(ValueError):
        s.intervene(Intervention(kind="set_param", path="task.n_patterns", value=5))
    s.intervene(Intervention(kind="set_param", path="memory.capacity", value=3))
    s.step(10)
    assert all(len(a.memory) <= 3 for a in s.agents)


def test_inspect(config, full):
    s = Session(config, full, seed=1)
    s.step(6)
    ins = s.inspect("A")
    assert len(ins.state_history) == 7 and len(ins.information_received) == 6
    assert ins.last_state_inputs is not None and ins.memory


def test_generated_narrative_is_logged_not_used(config, full):
    class FakeNarrator:
        kind = "local"

        def __init__(self, inner):
            self.inner = inner

        def choose_motif(self, ctx):  # noqa: ANN001, ANN201
            return self.inner.choose_motif(ctx)

        def choose_pattern(self, ctx):  # noqa: ANN001, ANN201
            return self.inner.choose_pattern(ctx)

        def generate_narrative(self, summary, before, after, emit, step):  # noqa: ANN001, ANN201
            return {"text": f"episode {step}", "generated_by": "fake"}

    cfg = config.model_copy(update={"model": config.model.model_copy(update={"narrative_enabled": True})})
    s = Session(cfg, full, seed=1)
    plain = Session(cfg, full, seed=1)
    s.agents[0].policy = FakeNarrator(s.agents[0].policy)
    s.step(4)
    plain.step(4)
    narr = [e for e in s.events if e.type == "generated_narrative"]
    assert len(narr) == 4 and all(e.payload["kind"] == "generated_narrative" and e.visibility == "experimenter" for e in narr)
    assert s.scores == plain.scores  # narrative never feeds back into the simulation
