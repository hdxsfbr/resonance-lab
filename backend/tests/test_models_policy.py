"""ModelPolicy behaviour with deterministic providers (no network, no model)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from resonance import schemas as S
from resonance.models.narrative import generate_narrative
from resonance.models.policy import (
    BUDGET_EXHAUSTED,
    NARRATIVE_LABEL,
    ModelPolicy,
    feature_summary,
    narrative_event_payload,
    validate_phrase,
)
from resonance.models.prompts import REQUIRED_TEMPLATES, PromptError, PromptLibrary
from resonance.models.registry import PROJECT_ROOT, build_model_policy
from resonance.models.scripted import ScriptedProvider
from resonance.models.testing import (
    EventSink,
    FakeLocalPolicy,
    make_receiver_ctx,
    make_sender_ctx,
)
from resonance.policy_types import PolicyProtocol

PROMPTS = PromptLibrary(PROJECT_ROOT / "prompts")


def policy(script, *, use_for=("receiver", "sender"), **cfg_kw) -> tuple[ModelPolicy, FakeLocalPolicy]:
    fb = FakeLocalPolicy()
    cfg = S.ModelConfig(provider="scripted", use_for=list(use_for), **cfg_kw)
    return ModelPolicy(ScriptedProvider(script), fb, cfg, PROMPTS), fb


def test_implements_protocol():
    pol, _ = policy([])
    assert isinstance(pol, ModelPolicy) and pol.kind == "model"
    _proto: PolicyProtocol = pol  # structural typing check (mypy); runtime: methods exist
    assert callable(pol.choose_pattern) and callable(pol.choose_motif)


def test_choose_pattern_happy_path():
    pol, fb = policy([{"pattern_id": 2, "confidence": 0.8, "rationale": "regular, sparse"}])
    sink = EventSink()
    ctx = make_receiver_ctx(emit=sink)
    local = FakeLocalPolicy().choose_pattern(make_receiver_ctx())
    trace = pol.choose_pattern(ctx)
    assert trace.policy_kind == "model" and trace.chosen == 2
    assert trace.probabilities[2] == pytest.approx(0.8)
    assert sum(trace.probabilities) == pytest.approx(1.0)
    # the local policy's temperature fields are retained
    assert trace.temperature == pytest.approx(local.temperature)
    assert trace.temperature_base == pytest.approx(ctx.config.learning.temperature_base)
    assert trace.exploration_draw is None
    assert any("model decision" in n for n in trace.notes)
    assert fb.calls == 1
    calls = sink.of("model_call")
    assert len(calls) == 1
    ev = calls[0]
    assert ev["ok"] and not ev["fallback_used"] and ev["purpose"] == "choose_pattern"
    assert ev["input_modality"] == "symbolic_features" and ev["accepts_audio"] is False
    assert ev["request"]["system"] and ev["request"]["user"]
    assert ev["response"]["parsed"]["pattern_id"] == 2
    assert ev["budget_remaining"] == 199 and ev["visibility_hint"] == "public"
    json.dumps(sink.events)  # payloads are JSON-serialisable


def test_local_rng_consumed_exactly_once():
    """The model path must leave ctx.rng exactly where a pure local run leaves it."""
    pol, _ = policy([{"pattern_id": 0, "confidence": 0.9, "rationale": "x"}])
    ctx_model = make_receiver_ctx(seed=123)
    ctx_local = make_receiver_ctx(seed=123)
    pol.choose_pattern(ctx_model)
    FakeLocalPolicy().choose_pattern(ctx_local)
    assert ctx_model.rng.random() == ctx_local.rng.random()


def test_invalid_pattern_id_falls_back_and_notes():
    pol, fb = policy([{"pattern_id": 9, "confidence": 0.9, "rationale": "x"}] * 3, max_retries=1)
    sink = EventSink()
    ctx = make_receiver_ctx(emit=sink, seed=5)
    trace = pol.choose_pattern(ctx)
    expected = FakeLocalPolicy().choose_pattern(make_receiver_ctx(seed=5))
    assert trace.policy_kind == "local" and trace.chosen == expected.chosen
    assert any("model fallback" in n and "out of range" in n for n in trace.notes)
    calls = sink.of("model_call")
    assert [c["attempt"] for c in calls] == [1, 2]
    assert calls[0]["will_retry"] and not calls[0]["fallback_used"]
    assert calls[1]["fallback_used"] and not calls[1]["will_retry"]
    assert fb.calls == 1


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        {"pattern_id": "two", "confidence": 0.5, "rationale": ""},
        {"pattern_id": 1, "confidence": 1.7, "rationale": ""},
        {"pattern_id": True, "confidence": 0.5, "rationale": ""},
    ],
)
def test_malformed_outputs_fall_back(bad):
    pol, _ = policy([bad], max_retries=0)
    trace = pol.choose_pattern(make_receiver_ctx())
    assert trace.policy_kind == "local"
    assert any("model fallback" in n for n in trace.notes)


def test_exception_retries_then_falls_back():
    pol, _ = policy([RuntimeError("boom"), TimeoutError("slow")], max_retries=1)
    sink = EventSink()
    trace = pol.choose_pattern(make_receiver_ctx(emit=sink))
    assert trace.policy_kind == "local"
    calls = sink.of("model_call")
    assert len(calls) == 2 and pol.calls_made == 2
    assert "RuntimeError" in calls[0]["response"]["error"] and "TimeoutError" in calls[1]["response"]["error"]
    assert calls[1]["fallback_used"]


def test_exception_then_success_on_retry():
    pol, _ = policy(
        [RuntimeError("transient"), {"pattern_id": 3, "confidence": 0.6, "rationale": "ok"}], max_retries=1
    )
    trace = pol.choose_pattern(make_receiver_ctx())
    assert trace.policy_kind == "model" and trace.chosen == 3


def test_budget_exhaustion_falls_back_with_single_event():
    script = [{"pattern_id": 1, "confidence": 0.7, "rationale": "a"}] * 10
    pol, fb = policy(script, call_budget=2, max_retries=0)
    sink = EventSink()
    kinds = [pol.choose_pattern(make_receiver_ctx(emit=sink)).policy_kind for _ in range(5)]
    assert kinds == ["model", "model", "local", "local", "local"]
    assert pol.calls_made == 2 and pol.budget_remaining == 0
    exhausted = [c for c in sink.of("model_call") if c["response"]["error"] == BUDGET_EXHAUSTED]
    assert len(exhausted) == 1 and exhausted[0]["fallback_used"] and exhausted[0]["request"] is None
    assert fb.calls == 5
    assert pol.status().calls_made == 2


def test_receiver_prompt_never_contains_target():
    """Information isolation: nothing about the target reaches the receiver prompt,
    even if a phrase id/tag were (wrongly) to carry it."""
    pol, _ = policy([{"pattern_id": 0, "confidence": 0.5, "rationale": "x"}])
    sink = EventSink()
    ctx = make_receiver_ctx(emit=sink, phrase_tags=["target_id=3", "secret-target-3"])
    ctx.observation.phrase.id = "phrase-target-3"
    system, user, schema = pol.receiver_prompts(ctx)
    for text in (system, user, json.dumps(schema)):
        assert "target" not in text.lower()
    pol.choose_pattern(ctx)
    req = sink.of("model_call")[0]["request"]
    assert "target" not in (req["system"] + req["user"]).lower()
    # The receiver context type itself has no target field.
    assert not any("target" in f for f in type(ctx).__dataclass_fields__)


def test_symbol_channel_summary():
    ctx = make_receiver_ctx(channel="symbol")
    assert feature_summary(ctx.observation).startswith("symbol #5")
    pol, _ = policy([{"pattern_id": 1, "confidence": 0.5, "rationale": "x"}])
    _, user, _ = pol.receiver_prompts(ctx)
    assert "symbol #5" in user and "note_density" not in user


def test_use_for_gates_model_calls():
    pol, fb = policy([{"pattern_id": 1, "confidence": 0.5, "rationale": "x"}], use_for=())
    sink = EventSink()
    trace = pol.choose_pattern(make_receiver_ctx(emit=sink))
    assert trace.policy_kind == "local" and sink.events == [] and pol.calls_made == 0
    assert any("use_for" in n for n in trace.notes)


def test_choose_motif_happy_path_is_sender_private():
    pol, _ = policy([{"motif_index": 6, "confidence": 0.7, "rationale": "jab marks syncopation"}])
    sink = EventSink()
    ctx = make_sender_ctx(emit=sink)
    trace = pol.choose_motif(ctx)
    assert trace.role == "sender" and trace.policy_kind == "model" and trace.chosen == 6
    ev = sink.of("model_call")[0]
    assert ev["purpose"] == "choose_motif" and ev["visibility_hint"] == "sender_private"
    assert "syncopated" in ev["request"]["user"]  # the sender legitimately sees the target


def test_propose_phrase_valid_and_invalid():
    good = {
        "tempo_bpm": 120,
        "rationale": "on the onsets",
        "notes": [
            {"pitch": 60, "onset": 0.0, "duration": 0.5, "velocity": 90},
            {"pitch": 64, "onset": 0.75, "duration": 0.5, "velocity": 90},
            {"pitch": 67, "onset": 7.0, "duration": 1.0, "velocity": 100},
        ],
    }
    pol, _ = policy([good])
    ph = pol.propose_phrase(make_sender_ctx())
    assert isinstance(ph, S.Phrase) and len(ph.notes) == 3 and ph.origin.kind == "model"
    assert ph.length_beats == 8.0 and ph.tempo_bpm == 120

    overflow = json.loads(json.dumps(good))
    overflow["notes"][2]["onset"] = 7.5  # 7.5 + 1.0 > 8 beats
    pol2, _ = policy([overflow], max_retries=0)
    assert pol2.propose_phrase(make_sender_ctx()) is None

    pol3, _ = policy([good], use_for=("receiver",))
    assert pol3.propose_phrase(make_sender_ctx()) is None and pol3.calls_made == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(notes=[]),
        lambda d: d.update(notes=[d["notes"][0]] * 17),
        lambda d: d["notes"][0].update(pitch=47),
        lambda d: d["notes"][0].update(velocity=121),
        lambda d: d["notes"][0].update(duration=0.2),
        lambda d: d.update(tempo_bpm=200),
    ],
)
def test_validate_phrase_rejects(mutate):
    d = {
        "tempo_bpm": 100,
        "rationale": "",
        "notes": [{"pitch": 60, "onset": 0, "duration": 1, "velocity": 80}],
    }
    assert validate_phrase(d, 8.0, 80, 160)[0] is not None
    mutate(d)
    assert validate_phrase(d, 8.0, 80, 160)[0] is None


def test_build_model_policy_none_returns_fallback_itself():
    fb = FakeLocalPolicy()
    assert build_model_policy(S.ModelConfig(provider="none"), fb) is fb


def test_build_model_policy_unavailable_provider_returns_fallback(tmp_path):
    fb = FakeLocalPolicy()
    cfg = S.ModelConfig(provider="llamacpp", local_model_path=str(tmp_path / "missing.gguf"))
    assert build_model_policy(cfg, fb) is fb
    cfg2 = S.ModelConfig(provider="openai_compatible", base_url=None)
    assert build_model_policy(cfg2, fb) is fb


def test_build_model_policy_scripted_and_missing_prompts(tmp_path):
    fb = FakeLocalPolicy()
    pol = build_model_policy(S.ModelConfig(provider="scripted", use_for=["receiver"]), fb)
    assert isinstance(pol, ModelPolicy)
    # default scripted answers are deterministic and schema-valid
    t1 = pol.choose_pattern(make_receiver_ctx())
    t2 = pol.choose_pattern(make_receiver_ctx())
    assert t1.policy_kind == "model" and t1.chosen == t2.chosen
    assert build_model_policy(S.ModelConfig(provider="scripted"), fb, prompts_dir=tmp_path) is fb


def test_prompt_templates_complete_and_truthful():
    assert PROMPTS.missing_templates() == []
    for name in REQUIRED_TEMPLATES:
        text = PROMPTS.get(name)
        if name.endswith(".system"):
            assert "JSON" in text
            assert "not heard audio" in text
    assert "SYMBOLIC FEATURE SUMMARY computed from the note list" in PROMPTS.get("choose_pattern.system")
    with pytest.raises(PromptError):
        PROMPTS.render("choose_pattern.system", n_patterns=4)  # missing placeholders


def test_narrative_payload_and_gating():
    payload = narrative_event_payload("A sent; B chose 2.", "scripted", "scripted-v1")
    assert payload["label"] == NARRATIVE_LABEL == "generated interpretation — not a measurement"
    prov = ScriptedProvider([{"narrative": "Agent A sent motif 2; B chose pattern 2; score 1."}] * 3)
    args = ({"step": 1, "score": 1.0}, {"A": {"activation": 0.5}}, {"A": {"activation": 0.6}})
    off = S.ModelConfig(provider="scripted")
    assert generate_narrative(prov, PROMPTS, *args, cfg=off) is None
    on = S.ModelConfig(provider="scripted", narrative_enabled=True, use_for=["narrative"])
    assert generate_narrative(prov, PROMPTS, *args, cfg=on).startswith("Agent A")
    pol = ModelPolicy(prov, FakeLocalPolicy(), on, PROMPTS)
    sink = EventSink()
    out = pol.generate_narrative(*args, emit=sink)
    assert out["label"] == NARRATIVE_LABEL and out["provider"] == "scripted"
    assert sink.of("model_call")[0]["visibility_hint"] == "experimenter"


def test_rng_untouched_by_failures():
    pol, _ = policy([RuntimeError("x")] * 4, max_retries=1)
    a, b = make_receiver_ctx(seed=9), make_receiver_ctx(seed=9)
    pol.choose_pattern(a)
    FakeLocalPolicy().choose_pattern(b)
    assert np.array_equal(a.rng.random(3), b.rng.random(3))
