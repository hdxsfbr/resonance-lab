"""Deterministic scripted provider, for tests and for `model.provider: scripted`.

It is a real, exercised code path (tested_in_this_environment=True), but it is
NOT a language model: answers come from a fixed script or a pure function of
the request, so model-assisted runs with it are fully reproducible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from resonance import schemas as S
from resonance.models.base import parse_json_object

ScriptItem = dict[str, Any] | str | BaseException
Script = list[ScriptItem] | Callable[[S.ModelRequest], ScriptItem]


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def default_script(req: S.ModelRequest) -> dict[str, Any]:
    """A deterministic, schema-valid answer derived only from the request text.

    Integer fields with minimum/maximum get `hash(user_prompt) mod range`, so the
    same prompt always yields the same answer. Used by `provider: scripted`.
    """
    h = _stable_int(req.user_prompt)
    props = (req.json_schema or {}).get("properties", {})
    if req.purpose == "narrative":
        return {"narrative": "Scripted placeholder narrative (no language model was called)."}
    if "notes" in props:  # propose_phrase
        pitches = [60, 62, 64, 67]
        return {
            "tempo_bpm": 100,
            "notes": [
                {"pitch": pitches[(h + i) % 4], "onset": float(i), "duration": 0.5, "velocity": 90}
                for i in range(4)
            ],
            "confidence": 0.5,
            "rationale": "scripted",
        }
    out: dict[str, Any] = {}
    for key, spec in props.items():
        if "enum" in spec and spec["enum"]:
            out[key] = spec["enum"][h % len(spec["enum"])]
        elif spec.get("type") == "integer":
            lo, hi = int(spec.get("minimum", 0)), int(spec.get("maximum", 0))
            out[key] = lo + (h % (hi - lo + 1)) if hi >= lo else lo
        elif spec.get("type") == "number":
            out[key] = 0.5
        elif spec.get("type") == "string":
            out[key] = "scripted deterministic choice"
    return out


class ScriptedProvider:
    name = "scripted"
    accepts_audio = False
    input_modality = "symbolic_features"

    def __init__(
        self, responses: Script | None = None, model_id: str = "scripted-v1", latency_ms: float = 0.0
    ):
        self.model_id = model_id or "scripted-v1"
        self._script: Script = responses if responses is not None else default_script
        self._index = 0
        self._latency_ms = latency_ms
        self.requests: list[S.ModelRequest] = []

    def complete(self, req: S.ModelRequest) -> S.ModelResponse:
        """Return the next scripted answer.

        Script items: dict -> JSON content; str -> raw content (parsed if JSON);
        an exception instance -> RAISED (lets tests exercise the caller's guard).
        """
        self.requests.append(req)
        if callable(self._script):
            item = self._script(req)
        else:
            if self._index >= len(self._script):
                return S.ModelResponse(
                    ok=False,
                    provider=self.name,
                    model_id=self.model_id,
                    error="script exhausted",
                    tested_in_this_environment=True,
                )
            item = self._script[self._index]
            self._index += 1
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, dict):
            content, parsed = json.dumps(item), item
        else:
            content, parsed = str(item), parse_json_object(str(item))
        # Fixed (default 0) rather than measured, so scripted runs give bit-identical event logs.
        latency = self._latency_ms
        ok = parsed is not None or req.json_schema is None
        return S.ModelResponse(
            ok=ok,
            provider=self.name,
            model_id=self.model_id,
            content=content,
            parsed=parsed,
            latency_ms=latency,
            error=None if ok else "response is not a JSON object",
            usage={},
            tested_in_this_environment=True,
        )

    def status(self) -> S.ModelStatus:
        return S.ModelStatus(
            provider=self.name,
            model_id=self.model_id,
            available=True,
            tested_in_this_environment=True,
            accepts_audio=False,
            input_modality="symbolic_features",
            detail=(
                "Deterministic scripted provider (not a language model). Answers are a fixed "
                "script or a hash of the prompt; exercised by the test suite in this environment."
            ),
        )
