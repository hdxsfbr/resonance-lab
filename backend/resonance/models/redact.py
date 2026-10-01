"""Masking of credentials in anything the model layer emits or logs.

`redact(text)` masks:
  * OpenAI/Anthropic-style keys:            sk-..., sk-ant-...
  * bearer tokens:                          "Bearer <token>"
  * values that follow a key-like label:    api_key=..., "token": "...", secret: ...
    (runs of 16+ characters from the hex/base64/url-safe alphabet)
  * the exact VALUES of environment variables whose NAMES were registered with
    `register_secret_env()` (providers register their `api_key_env`), plus the
    well-known provider key variables in `DEFAULT_SECRET_ENVS`.

It never reads or reports any other environment variable. It is applied to every
`model_call` / narrative payload before emission (see `redact_obj`).
"""

from __future__ import annotations

import os
import re
from typing import Any

MASK = "***REDACTED***"

DEFAULT_SECRET_ENVS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
_registered_envs: set[str] = set(DEFAULT_SECRET_ENVS)

_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9_\-\.=+/]{8,})")
_LABELLED_RE = re.compile(
    r"(?i)((?:api[_\-]?key|access[_\-]?key|secret[_\-]?key|key|token|secret|password|passwd|authorization)"
    r"[\"']?\s*[:=]\s*[\"']?)"
    r"([A-Za-z0-9_\-\.=+/]{16,})"
)


def register_secret_env(name: str | None) -> None:
    """Register an env var NAME whose value must be masked wherever it appears."""
    if name:
        _registered_envs.add(name)


def _secret_values(extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    vals = [v for v in extra if v and len(v) >= 6]
    for name in _registered_envs:
        v = os.environ.get(name)
        if v and len(v) >= 6:
            vals.append(v)
    # Longest first so a value that contains another is masked whole.
    return sorted(set(vals), key=len, reverse=True)


def redact(text: str, secrets: tuple[str, ...] | list[str] = ()) -> str:
    """Return `text` with credentials masked. Safe on any string."""
    if not text:
        return text
    out = text
    for v in _secret_values(secrets):
        out = out.replace(v, MASK)
    out = _SK_RE.sub("sk-" + MASK, out)
    out = _BEARER_RE.sub(lambda m: m.group(1) + MASK, out)
    out = _LABELLED_RE.sub(lambda m: m.group(1) + MASK, out)
    return out


def redact_obj(obj: Any, secrets: tuple[str, ...] | list[str] = ()) -> Any:
    """Recursively redact every string in a JSON-like structure (dict keys included)."""
    if isinstance(obj, str):
        return redact(obj, secrets)
    if isinstance(obj, dict):
        return {redact(str(k), secrets): redact_obj(v, secrets) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, secrets) for v in obj]
    return obj
