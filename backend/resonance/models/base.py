"""Provider protocol and small shared helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from resonance import schemas as S


@runtime_checkable
class ModelProvider(Protocol):
    """A backend that answers one `ModelRequest` at a time.

    Implementations must never raise from `complete`: every failure (import,
    network, timeout, invalid output) is returned as `ModelResponse(ok=False,
    error=...)`. Callers still guard against exceptions defensively.
    """

    name: str
    model_id: str
    accepts_audio: bool
    input_modality: str

    def complete(self, req: S.ModelRequest) -> S.ModelResponse: ...

    def status(self) -> S.ModelStatus: ...


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Parse a JSON object from model output.

    Tolerates surrounding prose and ```json code fences```. Returns None unless
    the result is a JSON object (dict).
    """
    if not text:
        return None
    candidates: list[str] = [text.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            val = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(val, dict):
            return val
    return None


def error_response(
    provider: str, model_id: str, error: str, latency_ms: float = 0.0, tested: bool = False
) -> S.ModelResponse:
    return S.ModelResponse(
        ok=False,
        provider=provider,
        model_id=model_id,
        error=error,
        latency_ms=latency_ms,
        tested_in_this_environment=tested,
    )


def schema_for_prompt(schema: dict[str, Any] | None) -> str:
    """Compact JSON rendering of a schema for inclusion in a prompt."""
    if not schema:
        return "{}"
    return json.dumps(schema, separators=(",", ":"), sort_keys=False)


REPAIRED_FLAG = "_repaired_truncated_output"


def repair_truncated_json(text: str | None) -> dict[str, Any] | None:
    """Best-effort repair of a JSON object cut off by max_tokens.

    Only closes what is open (an unterminated string, then brackets/braces) after
    dropping a dangling comma or partial escape. Returns None if that does not
    yield a JSON object. The result is marked with REPAIRED_FLAG so callers can
    record that the output was truncated.
    """
    if not text or "{" not in text:
        return None
    s = text[text.find("{") :].rstrip()
    in_str, esc, stack = False, False, []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if not stack and not in_str:
        return None  # not truncated; ordinary parsing already failed
    if esc:
        s = s[:-1]
    if in_str:
        s += '"'
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1]
    s += "".join(reversed(stack))
    try:
        val = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(val, dict):
        return None
    val[REPAIRED_FLAG] = True
    return val
