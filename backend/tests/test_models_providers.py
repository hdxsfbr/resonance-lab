"""Provider plumbing tests. No external network: the OpenAI-compatible provider
is exercised against a local http.server stub; the Anthropic provider is checked
offline (status without a key, request building, and a fake SDK client)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from resonance import schemas as S
from resonance.models import anthropic_provider as ap
from resonance.models.anthropic_provider import AnthropicProvider, anthropic_schema
from resonance.models.base import parse_json_object
from resonance.models.openai_compatible import OpenAICompatibleProvider
from resonance.models.policy import ModelPolicy, choice_schema
from resonance.models.prompts import PromptLibrary
from resonance.models.registry import PROJECT_ROOT, build_provider, resolve_path, status
from resonance.models.scripted import ScriptedProvider
from resonance.models.testing import EventSink, FakeLocalPolicy, make_receiver_ctx

NO_KEY_ENV = "RESONANCE_TEST_DEFINITELY_UNSET_KEY"


def _req(schema=True) -> S.ModelRequest:
    return S.ModelRequest(
        purpose="choose_pattern",
        system_prompt="sys",
        user_prompt="usr",
        json_schema=choice_schema("pattern_id", 4) if schema else None,
        max_tokens=64,
    )


# -- helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 2}\n```', {"a": 2}),
        ('Sure! Here it is: {"a": 3} hope that helps', {"a": 3}),
        ("[1, 2]", None),
        ("nope", None),
        ("", None),
    ],
)
def test_parse_json_object(text, expected):
    assert parse_json_object(text) == expected


def test_resolve_path_against_project_root():
    assert resolve_path("prompts") == PROJECT_ROOT / "prompts"
    assert (PROJECT_ROOT / "backend").is_dir()


# -- scripted ----------------------------------------------------------------


def test_scripted_provider_status_and_exhaustion():
    p = ScriptedProvider([{"pattern_id": 1, "confidence": 0.5, "rationale": "x"}])
    st = p.status()
    assert st.available and st.tested_in_this_environment and not st.accepts_audio
    assert st.input_modality == "symbolic_features"
    assert p.complete(_req()).parsed["pattern_id"] == 1
    r = p.complete(_req())
    assert not r.ok and r.error == "script exhausted"


# -- OpenAI-compatible against a local stub -----------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    received: list[dict] = []
    reply_content = '{"pattern_id": 2, "confidence": 0.75, "rationale": "stub"}'
    status_code = 200

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        type(self).received.append(
            {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
        )
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        if type(self).status_code != 200:
            payload = {"error": {"message": "stub failure", "type": "server_error"}}
        else:
            payload = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": type(self).reply_content},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        data = json.dumps(payload).encode()
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def stub_server():
    _StubHandler.received = []
    _StubHandler.status_code = 200
    _StubHandler.reply_content = '{"pattern_id": 2, "confidence": 0.75, "rationale": "stub"}'
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_openai_compatible_against_local_stub(stub_server, monkeypatch):
    monkeypatch.setenv("RESONANCE_TEST_STUB_KEY", "local-stub-key-123456")
    p = OpenAICompatibleProvider("stub-model", stub_server, api_key_env="RESONANCE_TEST_STUB_KEY", timeout=5)
    r = p.complete(_req())
    assert r.ok, r.error
    assert r.parsed == {"pattern_id": 2, "confidence": 0.75, "rationale": "stub"}
    assert r.usage["total_tokens"] == 18 and r.latency_ms > 0 and r.tested_in_this_environment is False
    sent = _StubHandler.received[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["model"] == "stub-model"
    assert sent["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]
    assert sent["auth"] == "Bearer local-stub-key-123456"
    assert "local-stub-key-123456" not in repr(p.__dict__)  # key is not stored on the provider


def test_openai_compatible_stub_errors_and_bad_json(stub_server):
    p = OpenAICompatibleProvider("stub-model", stub_server, timeout=5)
    _StubHandler.reply_content = "I think pattern two"
    r = p.complete(_req())
    assert not r.ok and "JSON" in r.error
    _StubHandler.status_code = 500
    r = p.complete(_req())
    assert not r.ok and "500" in r.error


def test_openai_compatible_policy_end_to_end_via_stub(stub_server):
    cfg = S.ModelConfig(
        provider="openai_compatible",
        model_id="stub-model",
        base_url=stub_server,
        use_for=["receiver"],
        timeout_seconds=5,
    )
    provider = build_provider(cfg)
    assert isinstance(provider, OpenAICompatibleProvider)
    pol = ModelPolicy(provider, FakeLocalPolicy(), cfg, PromptLibrary(PROJECT_ROOT / "prompts"))
    sink = EventSink()
    trace = pol.choose_pattern(make_receiver_ctx(emit=sink))
    assert trace.policy_kind == "model" and trace.chosen == 2
    assert sink.of("model_call")[0]["provider"] == "openai_compatible"


def test_openai_compatible_unreachable_is_graceful():
    p = OpenAICompatibleProvider("m", "http://127.0.0.1:9/v1", timeout=1)
    r = p.complete(_req())
    assert not r.ok and r.error


def test_openai_compatible_status():
    st = OpenAICompatibleProvider("m", None).status()
    assert not st.available and not st.tested_in_this_environment and not st.accepts_audio
    st = OpenAICompatibleProvider("m", "http://localhost:11434/v1").status()
    assert st.available and not st.tested_in_this_environment
    assert "Untested against a live service" in st.detail and "local HTTP stub" in st.detail


# -- Anthropic (offline only) ----------------------------------------------------


def test_anthropic_status_without_key(monkeypatch):
    monkeypatch.delenv(NO_KEY_ENV, raising=False)
    p = AnthropicProvider(model_id="claude-opus-5-5", api_key_env=NO_KEY_ENV)
    st = p.status()
    assert st.available is False and st.tested_in_this_environment is False
    assert st.accepts_audio is False and "untested against the live API" in st.detail
    st2 = status(S.ModelConfig(provider="anthropic", api_key_env=NO_KEY_ENV))
    assert st2.available is False and st2.tested_in_this_environment is False


def test_anthropic_complete_without_key_makes_no_call(monkeypatch):
    monkeypatch.delenv(NO_KEY_ENV, raising=False)

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("client must not be constructed without a key")

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", Boom)
    r = AnthropicProvider(api_key_env=NO_KEY_ENV).complete(_req())
    assert not r.ok and NO_KEY_ENV in r.error
    assert build_provider(S.ModelConfig(provider="anthropic", api_key_env=NO_KEY_ENV)) is None


def test_anthropic_request_building():
    p = AnthropicProvider(model_id="claude-opus-5-5", api_key_env=NO_KEY_ENV)
    kw = p.build_kwargs(_req())
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and kw["output_config"]["effort"] == "low"
    assert "temperature" not in kw and kw["max_tokens"] >= ap.MIN_MAX_TOKENS
    assert kw["system"] == "sys" and kw["messages"] == [{"role": "user", "content": "usr"}]
    assert kw["fallbacks"] == "default" and kw["betas"] == [ap.FALLBACK_BETA]
    sch = json.dumps(fmt["schema"])
    assert "maxLength" not in sch and "minimum" not in sch and fmt["schema"]["additionalProperties"] is False
    other = AnthropicProvider(
        model_id="claude-haiku-4-5", api_key_env=NO_KEY_ENV, base_url="http://proxy.local"
    )
    kw2 = other.build_kwargs(_req(schema=False))
    assert "fallbacks" not in kw2 and kw2["temperature"] == pytest.approx(0.2) and "output_config" not in kw2


def test_anthropic_schema_strips_unsupported():
    s = anthropic_schema(
        {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 3},
                "n": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {}}},
            },
        }
    )
    assert s["properties"]["x"] == {"type": "integer"}
    assert s["properties"]["n"]["items"]["additionalProperties"] is False


def test_anthropic_response_parsing_with_fake_client(monkeypatch):
    """Exercise complete() end to end with a stand-in SDK client (no network)."""
    import anthropic

    monkeypatch.setenv("RESONANCE_TEST_ANTHROPIC_KEY", "sk-ant-test-0000000000000000")
    captured = {}

    class Block:
        type = "text"
        text = '{"pattern_id": 1, "confidence": 0.9, "rationale": "fake"}'

    class Usage:
        input_tokens = 100
        output_tokens = 20

    class Resp:
        content = [Block()]
        stop_reason = "end_turn"
        usage = Usage()

    class Messages:
        def create(self, **kw):
            captured.update(kw)
            return Resp()

    class Beta:
        messages = Messages()

    class FakeClient:
        def __init__(self, **kw):
            captured["client_kwargs"] = kw
            self.messages = Messages()
            self.beta = Beta()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    p = AnthropicProvider(model_id="claude-opus-5-5", api_key_env="RESONANCE_TEST_ANTHROPIC_KEY")
    r = p.complete(_req())
    assert r.ok and r.parsed["pattern_id"] == 1 and r.usage == {"input_tokens": 100, "output_tokens": 20}
    assert r.tested_in_this_environment is False
    assert captured["client_kwargs"]["max_retries"] == 0
    assert "sk-ant-test" not in repr(p.__dict__)

    Resp.stop_reason = "refusal"
    r2 = p.complete(_req())
    assert not r2.ok and "refusal" in r2.error


# -- registry ------------------------------------------------------------------


def test_registry_status_none_and_llamacpp_missing(tmp_path):
    st = status(S.ModelConfig(provider="none"))
    assert st.provider == "none" and not st.available and st.input_modality == "none" and not st.accepts_audio
    st = status(S.ModelConfig(provider="llamacpp", local_model_path=str(tmp_path / "nope.gguf")))
    assert not st.available and not st.tested_in_this_environment and "not found" in st.detail


def test_no_provider_accepts_audio():
    for prov in ("none", "scripted", "llamacpp", "anthropic", "openai_compatible"):
        assert status(S.ModelConfig(provider=prov, api_key_env=NO_KEY_ENV)).accepts_audio is False
