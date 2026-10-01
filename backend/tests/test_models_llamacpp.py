"""Real local-model test (Qwen2.5-0.5B-Instruct Q4_K_M via llama-cpp-python).

Skips cleanly if the GGUF file or llama_cpp is unavailable. It is slow (a few
seconds per call on CPU); deselect with `-k "not slow"` or set
RESONANCE_SKIP_SLOW=1.
"""

from __future__ import annotations

import os
import time

import pytest

from resonance import schemas as S
from resonance.models.llamacpp import LlamaCppProvider, llama_cpp_importable
from resonance.models.policy import ModelPolicy, validate_choice
from resonance.models.prompts import PromptLibrary
from resonance.models.registry import PROJECT_ROOT, build_model_policy, resolve_path, status
from resonance.models.testing import EventSink, FakeLocalPolicy, make_receiver_ctx

MODEL_PATH = resolve_path(S.ModelConfig().local_model_path)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file() or not llama_cpp_importable() or os.environ.get("RESONANCE_SKIP_SLOW") == "1",
    reason=f"local GGUF model or llama_cpp unavailable ({MODEL_PATH}), or RESONANCE_SKIP_SLOW=1",
)


def test_llamacpp_status_is_truthful():
    st = status(S.ModelConfig(provider="llamacpp"))
    assert st.available and st.tested_in_this_environment and not st.accepts_audio
    assert st.input_modality == "symbolic_features"
    assert "Qwen2.5-0.5B-Instruct-Q4_K_M" in st.detail and "weak" in st.detail


def test_llamacpp_real_choose_pattern_slow():
    cfg = S.ModelConfig(provider="llamacpp", use_for=["receiver"], max_retries=0)
    pol = build_model_policy(cfg, FakeLocalPolicy())
    assert isinstance(pol, ModelPolicy) and isinstance(pol.provider, LlamaCppProvider)
    sink = EventSink()
    t0 = time.perf_counter()
    trace = pol.choose_pattern(make_receiver_ctx(emit=sink))
    elapsed = time.perf_counter() - t0
    ev = sink.of("model_call")[0]
    print(
        f"\nllama.cpp choose_pattern: {elapsed:.2f}s wall, model latency {ev['response']['latency_ms']} ms, "
        f"usage {ev['response']['usage']}, content {ev['response']['content']}"
    )
    # Grammar-constrained decoding guarantees parseable JSON with pattern_id in the enum.
    assert ev["response"]["parsed"] is not None
    assert ev["response"]["parsed"]["pattern_id"] in range(4)
    assert isinstance(ev["response"]["parsed"]["rationale"], str)
    value, _why = validate_choice(ev["response"]["parsed"], "pattern_id", 4)
    if value is not None:  # the policy keeps at most 200 characters of rationale
        assert len(value["rationale"]) <= 200
    if ev["ok"]:
        assert trace.policy_kind == "model" and trace.chosen == ev["response"]["parsed"]["pattern_id"]
        assert 0.0 <= ev["response"]["parsed"]["confidence"] <= 1.0
    else:  # e.g. confidence outside [0,1]: grammar cannot enforce numeric bounds
        assert trace.policy_kind == "local" and ev["fallback_used"]
    assert ev["input_modality"] == "symbolic_features" and ev["tested_in_this_environment"]
    # No wall-time assertion: latency depends on machine load (~10-11 s idle on 4 cores here,
    # >60 s observed while other CPU-heavy jobs ran). Timing is printed above instead.


def test_llamacpp_missing_file_is_graceful(tmp_path):
    p = LlamaCppProvider(tmp_path / "missing.gguf")
    r = p.complete(S.ModelRequest(purpose="choose_pattern", system_prompt="s", user_prompt="u"))
    assert not r.ok and "not found" in r.error
    assert not p.status().available
    fb = FakeLocalPolicy()
    assert (
        build_model_policy(S.ModelConfig(provider="llamacpp", local_model_path=str(tmp_path / "x.gguf")), fb)
        is fb
    )
    assert PromptLibrary(PROJECT_ROOT / "prompts").missing_templates() == []
