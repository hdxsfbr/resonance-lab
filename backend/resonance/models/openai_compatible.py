"""OpenAI-compatible chat-completions provider (official `openai` SDK).

Works with any server exposing `/v1/chat/completions`: OpenAI, Ollama, vLLM,
LM Studio, llama.cpp's own server, etc.

UNTESTED AGAINST ANY LIVE SERVICE IN THIS ENVIRONMENT. The request/response
plumbing IS tested here, against a local HTTP stub (http.server in a thread)
that implements /v1/chat/completions (see tests/test_models_providers.py).

* `base_url` is required (e.g. http://localhost:11434/v1 for Ollama).
* The key is optional (local servers usually need none); if `api_key_env` names
  a variable it is read at call time, never stored, and registered for redaction.
* JSON is requested with `response_format={"type": "json_object"}`; the schema
  itself is described in the prompt and re-validated by the caller.
* SDK retries are disabled; `ModelPolicy` owns retries and the call budget.
* Text only: `accepts_audio=False` (no audio is ever sent).
"""

from __future__ import annotations

import os
import time
from typing import Any

from resonance import schemas as S
from resonance.models.base import error_response, parse_json_object
from resonance.models.redact import register_secret_env

UNTESTED_DETAIL = (
    "Untested against a live service in this environment; request/response plumbing is tested "
    "against a local HTTP stub implementing /v1/chat/completions."
)


def _sdk_importable() -> bool:
    try:
        import openai  # noqa: F401
    except Exception:
        return False
    return True


class OpenAICompatibleProvider:
    name = "openai_compatible"
    accepts_audio = False
    input_modality = "symbolic_features"

    def __init__(self, model_id: str, base_url: str | None, api_key_env: str = "", timeout: float = 20.0):
        self.model_id = model_id or "unspecified-model"
        self.base_url = base_url or None
        self.api_key_env = api_key_env or ""
        self.timeout = float(timeout)
        register_secret_env(self.api_key_env)

    def build_kwargs(self, req: S.ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.user_prompt},
            ],
            "max_tokens": int(req.max_tokens),
            "temperature": float(req.temperature),
        }
        if req.json_schema:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def complete(self, req: S.ModelRequest) -> S.ModelResponse:
        t0 = time.perf_counter()
        if not self.base_url:
            return error_response(self.name, self.model_id, "model.base_url is not set")
        try:
            import openai
        except Exception as exc:
            return error_response(self.name, self.model_id, f"openai SDK not importable: {exc}")
        key = (os.environ.get(self.api_key_env) if self.api_key_env else None) or "not-needed"
        try:
            client = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
            resp = client.chat.completions.create(**self.build_kwargs(req))
            latency = (time.perf_counter() - t0) * 1000.0
            content = (resp.choices[0].message.content or "") if resp.choices else ""
            usage: dict[str, int] = {}
            if resp.usage is not None:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(resp.usage, k, None)
                    if isinstance(v, int):
                        usage[k] = v
            parsed = parse_json_object(content)
            ok = parsed is not None or not req.json_schema
            return S.ModelResponse(
                ok=ok,
                provider=self.name,
                model_id=self.model_id,
                content=content,
                parsed=parsed,
                latency_ms=latency,
                usage=usage,
                error=None if ok else "response is not a JSON object",
            )
        except openai.APITimeoutError:
            return error_response(
                self.name,
                self.model_id,
                f"timeout after {self.timeout}s",
                (time.perf_counter() - t0) * 1000.0,
            )
        except openai.APIStatusError as exc:
            return error_response(
                self.name, self.model_id, f"API error {exc.status_code}", (time.perf_counter() - t0) * 1000.0
            )
        except openai.APIConnectionError:
            return error_response(
                self.name, self.model_id, "connection error", (time.perf_counter() - t0) * 1000.0
            )
        except Exception as exc:
            return error_response(
                self.name, self.model_id, f"{type(exc).__name__}: {exc}", (time.perf_counter() - t0) * 1000.0
            )

    def status(self) -> S.ModelStatus:
        importable = _sdk_importable()
        available = bool(self.base_url) and importable
        why = (
            "openai SDK not installed."
            if not importable
            else f"Configured base_url: {self.base_url}."
            if self.base_url
            else "model.base_url is not set."
        )
        return S.ModelStatus(
            provider=self.name,
            model_id=self.model_id,
            available=available,
            tested_in_this_environment=False,
            accepts_audio=False,
            input_modality="symbolic_features",
            detail=f"{why} {UNTESTED_DETAIL}",
        )
