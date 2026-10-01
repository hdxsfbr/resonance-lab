"""Anthropic Messages API provider (official `anthropic` SDK).

UNTESTED AGAINST THE LIVE API IN THIS ENVIRONMENT: no credentials were present.
The request-building and response-parsing plumbing is unit-tested against a
stub client. Treat the first live run as an integration test.

* The API key is read from `os.environ[api_key_env]` at call time. It is never
  stored on this object, never logged, and its value is registered for redaction.
* Input is text only. The Anthropic Messages API does not accept audio input, so
  this provider reports `accepts_audio=False`; prompts state that the model gets
  a symbolic feature summary computed from the note list.
* When the request carries a JSON schema we use structured outputs
  (`output_config.format` with type `json_schema`). Numeric and string-length
  constraints (minimum/maximum/maxLength/...) are not supported there, so they are
  stripped from the schema sent to the API; the caller re-validates every field.
* SDK retries are disabled (`max_retries=0`): retries are owned by `ModelPolicy`
  (cfg.max_retries) so the call budget counts every attempt.
* On recent models (Opus 5.5 etc.) thinking is always on and sampling parameters
  are rejected, so temperature is not sent; effort is set to "low" and
  max_tokens is raised to MIN_MAX_TOKENS so thinking cannot starve the answer.
* For models that support it, server-side refusal fallbacks (`fallbacks="default"`,
  beta `server-side-fallback-2026-07-01`) are enabled on the first-party API;
  a final `stop_reason == "refusal"` is still returned as ok=False.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

from resonance import schemas as S
from resonance.models.base import error_response, parse_json_object
from resonance.models.redact import register_secret_env

DEFAULT_MODEL_ID = "claude-opus-5-5"
MIN_MAX_TOKENS = 2048
# Models with adaptive thinking + effort and no sampling parameters.
EFFORT_MODELS = {
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
}
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
}
UNTESTED_DETAIL = "No credentials present in this environment; integration untested against the live API."


def anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy of `schema` restricted to what structured outputs accept."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
            if out.get("type") == "object":
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(copy.deepcopy(schema))


def _sdk_importable() -> bool:
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


class AnthropicProvider:
    name = "anthropic"
    accepts_audio = False  # the Messages API takes text/images/documents, not audio
    input_modality = "symbolic_features"

    def __init__(
        self,
        model_id: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 20.0,
        base_url: str | None = None,
        server_side_fallbacks: bool = True,
    ):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.api_key_env = api_key_env or "ANTHROPIC_API_KEY"
        self.timeout = float(timeout)
        self.base_url = base_url or None
        self.server_side_fallbacks = server_side_fallbacks
        register_secret_env(self.api_key_env)

    def build_kwargs(self, req: S.ModelRequest) -> dict[str, Any]:
        """Request body (minus credentials). Separated out so tests can check it offline."""
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": int(req.max_tokens),
            "system": req.system_prompt,
            "messages": [{"role": "user", "content": req.user_prompt}],
        }
        output_config: dict[str, Any] = {}
        if self.model_id in EFFORT_MODELS:
            output_config["effort"] = "low"
            kwargs["max_tokens"] = max(kwargs["max_tokens"], MIN_MAX_TOKENS)
        else:
            kwargs["temperature"] = float(req.temperature)
        if req.json_schema:
            output_config["format"] = {"type": "json_schema", "schema": anthropic_schema(req.json_schema)}
        if output_config:
            kwargs["output_config"] = output_config
        if self.server_side_fallbacks and self.base_url is None and self.model_id in FALLBACK_MODELS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def complete(self, req: S.ModelRequest) -> S.ModelResponse:
        t0 = time.perf_counter()
        key = os.environ.get(self.api_key_env)
        if not key:
            return error_response(
                self.name, self.model_id, f"environment variable {self.api_key_env} is not set"
            )
        try:
            import anthropic
        except Exception as exc:
            return error_response(self.name, self.model_id, f"anthropic SDK not importable: {exc}")
        try:
            client = anthropic.Anthropic(
                api_key=key, base_url=self.base_url, timeout=self.timeout, max_retries=0
            )
            kwargs = self.build_kwargs(req)
            if "fallbacks" in kwargs:
                resp = client.beta.messages.create(**kwargs)
            else:
                resp = client.messages.create(**kwargs)
            latency = (time.perf_counter() - t0) * 1000.0
            usage: dict[str, int] = {}
            if getattr(resp, "usage", None) is not None:
                for k in ("input_tokens", "output_tokens"):
                    v = getattr(resp.usage, k, None)
                    if isinstance(v, int):
                        usage[k] = v
            if getattr(resp, "stop_reason", None) == "refusal":
                return S.ModelResponse(
                    ok=False,
                    provider=self.name,
                    model_id=self.model_id,
                    latency_ms=latency,
                    usage=usage,
                    error="model declined (stop_reason=refusal)",
                )
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
            parsed = parse_json_object(text)
            ok = parsed is not None or not req.json_schema
            err = None if ok else "response is not a JSON object"
            if getattr(resp, "stop_reason", None) == "max_tokens" and not ok:
                err = "output truncated at max_tokens"
            return S.ModelResponse(
                ok=ok,
                provider=self.name,
                model_id=self.model_id,
                content=text,
                parsed=parsed,
                latency_ms=latency,
                usage=usage,
                error=err,
            )
        except anthropic.AuthenticationError:
            return error_response(
                self.name,
                self.model_id,
                f"authentication failed (check the key in {self.api_key_env})",
                (time.perf_counter() - t0) * 1000.0,
            )
        except anthropic.RateLimitError:
            return error_response(
                self.name, self.model_id, "rate limited (429)", (time.perf_counter() - t0) * 1000.0
            )
        except anthropic.APITimeoutError:
            return error_response(
                self.name,
                self.model_id,
                f"timeout after {self.timeout}s",
                (time.perf_counter() - t0) * 1000.0,
            )
        except anthropic.APIStatusError as exc:
            return error_response(
                self.name, self.model_id, f"API error {exc.status_code}", (time.perf_counter() - t0) * 1000.0
            )
        except anthropic.APIConnectionError:
            return error_response(
                self.name, self.model_id, "connection error", (time.perf_counter() - t0) * 1000.0
            )
        except Exception as exc:
            return error_response(
                self.name, self.model_id, f"{type(exc).__name__}: {exc}", (time.perf_counter() - t0) * 1000.0
            )

    def status(self) -> S.ModelStatus:
        has_key = bool(os.environ.get(self.api_key_env))
        importable = _sdk_importable()
        if not importable:
            why = "anthropic SDK not installed."
        elif not has_key:
            why = f"Environment variable {self.api_key_env} is not set."
        else:
            why = f"Key found in {self.api_key_env}."
        return S.ModelStatus(
            provider=self.name,
            model_id=self.model_id,
            available=has_key and importable,
            tested_in_this_environment=False,
            accepts_audio=False,
            input_modality="symbolic_features",
            detail=f"{why} {UNTESTED_DETAIL} The Anthropic Messages API does not accept audio input.",
        )
