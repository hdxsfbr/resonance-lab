"""Local GGUF model via llama-cpp-python (the provider actually tested here).

Model: Qwen2.5-0.5B-Instruct, Q4_K_M quantisation, run on CPU. This is a very
small model: its decisions are an honest-but-weak baseline, not a strong agent.

Notes
-----
* The Llama object is loaded lazily, once per (path, n_threads), and cached at
  module level. Loading takes ~1 s; a constrained decision ~3 s on 4 threads.
* JSON output is enforced with `response_format={"type": "json_object",
  "schema": ...}` (grammar-constrained decoding), so the output is valid JSON
  matching the schema's structure; numeric ranges are still re-validated by
  the caller.
* String-length keywords (maxLength/minLength) are STRIPPED from the schema
  before it becomes a grammar: llama.cpp expands them into deep repetition rules
  that made constrained sampling 5-10x slower here (measured 353-662 ms/token
  with maxLength 200/80 vs 68 ms/token without, Qwen2.5-0.5B, 4 CPU threads).
  String length is then bounded by max_tokens. If output is cut off by
  max_tokens, `repair_truncated_json` closes the open string/object and marks
  the result with `_repaired_truncated_output`; the policy notes it in the trace.
* There is NO wall-clock timeout for llama.cpp: generation runs in-process. The
  bound is `max_tokens` (capped at MAX_TOKENS_CAP) — documented in MODEL_SETUP.md.
* Calls are serialised with a lock (a Llama instance is not thread-safe).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from resonance import schemas as S
from resonance.models.base import error_response, parse_json_object, repair_truncated_json

MODEL_LABEL = "Qwen2.5-0.5B-Instruct-Q4_K_M (local llama.cpp)"
MAX_TOKENS_CAP = 512
N_CTX = 2048

_CACHE: dict[tuple[str, int], Any] = {}
_CACHE_LOCK = threading.Lock()
_CALL_LOCK = threading.Lock()
_successful_calls = 0


_SLOW_GRAMMAR_KEYS = {"maxLength", "minLength"}


def grammar_schema(schema: Any) -> Any:
    """Copy of a JSON schema without keywords that make llama.cpp grammars slow."""
    if isinstance(schema, dict):
        return {k: grammar_schema(v) for k, v in schema.items() if k not in _SLOW_GRAMMAR_KEYS}
    if isinstance(schema, list):
        return [grammar_schema(v) for v in schema]
    return schema


def llama_cpp_importable() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except Exception:
        return False
    return True


def _load(path: str, n_threads: int) -> Any:
    key = (path, n_threads)
    with _CACHE_LOCK:
        if key not in _CACHE:
            from llama_cpp import Llama

            _CACHE[key] = Llama(model_path=path, n_ctx=N_CTX, n_threads=n_threads, verbose=False)
        return _CACHE[key]


class LlamaCppProvider:
    name = "llamacpp"
    accepts_audio = False
    input_modality = "symbolic_features"

    def __init__(
        self,
        model_path: str | Path,
        n_threads: int = 4,
        temperature: float = 0.2,
        model_id: str = "",
        seed: int | None = None,
    ):
        self.model_path = str(model_path)
        self.n_threads = int(n_threads)
        self.temperature = float(temperature)
        self.model_id = model_id or Path(self.model_path).name
        self.seed = seed

    def complete(self, req: S.ModelRequest) -> S.ModelResponse:
        global _successful_calls
        t0 = time.perf_counter()
        try:
            if not Path(self.model_path).is_file():
                return error_response(self.name, self.model_id, f"model file not found: {self.model_path}")
            llm = _load(self.model_path, self.n_threads)
            kwargs: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": req.system_prompt},
                    {"role": "user", "content": req.user_prompt},
                ],
                "max_tokens": min(int(req.max_tokens), MAX_TOKENS_CAP),
                "temperature": float(req.temperature),
            }
            if self.seed is not None:
                kwargs["seed"] = self.seed
            if req.json_schema:
                kwargs["response_format"] = {"type": "json_object", "schema": grammar_schema(req.json_schema)}
            with _CALL_LOCK:
                out = llm.create_chat_completion(**kwargs)
            latency = (time.perf_counter() - t0) * 1000.0
            choice = out["choices"][0]
            content = choice["message"].get("content") or ""
            usage = {k: int(v) for k, v in (out.get("usage") or {}).items() if isinstance(v, (int, float))}
            parsed = parse_json_object(content)
            if parsed is None and req.json_schema and choice.get("finish_reason") == "length":
                # Cut off by max_tokens (typically inside the free-text rationale, since
                # maxLength is not given to the grammar). Close it and flag the repair.
                parsed = repair_truncated_json(content)
            if req.json_schema and parsed is None:
                return S.ModelResponse(
                    ok=False,
                    provider=self.name,
                    model_id=self.model_id,
                    content=content,
                    latency_ms=latency,
                    error="response is not a JSON object",
                    usage=usage,
                    tested_in_this_environment=True,
                )
            _successful_calls += 1
            return S.ModelResponse(
                ok=True,
                provider=self.name,
                model_id=self.model_id,
                content=content,
                parsed=parsed,
                latency_ms=latency,
                usage=usage,
                tested_in_this_environment=True,
            )
        except Exception as exc:  # never raise to the caller
            return error_response(
                self.name, self.model_id, f"{type(exc).__name__}: {exc}", (time.perf_counter() - t0) * 1000.0
            )

    def status(self) -> S.ModelStatus:
        exists = Path(self.model_path).is_file()
        importable = llama_cpp_importable()
        available = exists and importable
        if not exists:
            why = f"model file not found at {self.model_path}"
        elif not importable:
            why = "llama_cpp (llama-cpp-python) is not importable"
        else:
            why = "model file present and llama_cpp importable"
        detail = (
            f"{MODEL_LABEL}: {why}. Input modality: symbolic feature summary computed from the note "
            f"list (no audio). This is a small 0.5B model; its decisions are an honest-but-weak "
            f"baseline. Real calls succeeded in this process: {_successful_calls}. No wall-clock "
            f"timeout (in-process); output bounded by max_tokens<={MAX_TOKENS_CAP}."
        )
        return S.ModelStatus(
            provider=self.name,
            model_id=self.model_id,
            available=available,
            tested_in_this_environment=available,
            accepts_audio=False,
            input_modality="symbolic_features",
            detail=detail,
        )
