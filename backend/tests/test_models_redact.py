"""Redaction: credentials must never appear in emitted model payloads."""

from __future__ import annotations

from resonance import schemas as S
from resonance.models.policy import ModelPolicy
from resonance.models.prompts import PromptLibrary
from resonance.models.redact import MASK, redact, redact_obj, register_secret_env
from resonance.models.registry import PROJECT_ROOT
from resonance.models.scripted import ScriptedProvider
from resonance.models.testing import EventSink, FakeLocalPolicy, make_receiver_ctx


def test_masks_sk_keys():
    text = "use key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 and sk-proj-zzzzzzzzzzzz1234"
    out = redact(text)
    assert "AbCdEfGhIj" not in out and "zzzzzzzzzzzz1234" not in out
    assert out.count(MASK) == 2


def test_masks_labelled_tokens_and_bearer():
    hexrun = "0123456789abcdef0123456789abcdef"
    for text in (
        f"api_key={hexrun}",
        f'"token": "{hexrun}"',
        f"Authorization: Bearer {hexrun}",
        f"secret: {hexrun}",
    ):
        out = redact(text)
        assert hexrun not in out, text
        assert MASK in out


def test_leaves_ordinary_text_alone():
    text = "pattern_id 2, confidence 0.71, rhythmic_regularity: 0.42, phrase p-demo"
    assert redact(text) == text


def test_never_leaks_registered_env_var_value(monkeypatch):
    secret = "not-a-real-key-but-secret-value-7f3a"  # no recognisable key shape on purpose
    monkeypatch.setenv("RESONANCE_TEST_PROVIDER_KEY", secret)
    register_secret_env("RESONANCE_TEST_PROVIDER_KEY")
    assert secret not in redact(f"the prompt contains {secret} verbatim")
    nested = {"request": {"user": f"x {secret} y"}, "list": [secret], secret: 1}
    assert secret not in repr(redact_obj(nested))


def test_policy_emits_redacted_payloads(monkeypatch):
    secret = "zz-env-secret-value-0042-qq"
    monkeypatch.setenv("RESONANCE_TEST_PROVIDER_KEY2", secret)
    register_secret_env("RESONANCE_TEST_PROVIDER_KEY2")
    leaky = {"pattern_id": 1, "confidence": 0.6, "rationale": f"sk-live-ABCDEFGHIJKLMNOP and {secret}"}
    provider = ScriptedProvider([leaky])
    cfg = S.ModelConfig(provider="scripted", use_for=["receiver"])
    pol = ModelPolicy(provider, FakeLocalPolicy(), cfg, PromptLibrary(PROJECT_ROOT / "prompts"))
    sink = EventSink()
    trace = pol.choose_pattern(make_receiver_ctx(emit=sink))
    dumped = repr(sink.events) + repr(trace.notes)
    assert secret not in dumped
    assert "ABCDEFGHIJKLMNOP" not in dumped
    assert trace.chosen == 1
