"""Build providers / model policies from `ModelConfig`; report `ModelStatus`.

Entry points used by the simulation core:

    build_model_policy(model_cfg, fallback=<LocalPolicy>, prompts_dir=...)
        -> ModelPolicy, or `fallback` itself when provider is "none" or the
           provider cannot be built (missing file/SDK/key/base_url, missing
           prompt templates). Reasons are logged and kept in `last_build_error`.
    status(model_cfg, policy=None) -> ModelStatus

Relative paths (`local_model_path`, `prompts_dir`) resolve against the project
root (the parent of `backend/`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from resonance import schemas as S
from resonance.models.anthropic_provider import AnthropicProvider
from resonance.models.base import ModelProvider
from resonance.models.llamacpp import LlamaCppProvider, llama_cpp_importable
from resonance.models.openai_compatible import OpenAICompatibleProvider
from resonance.models.policy import ModelPolicy
from resonance.models.prompts import PromptLibrary
from resonance.models.redact import redact
from resonance.models.scripted import ScriptedProvider
from resonance.policy_types import PolicyProtocol

log = logging.getLogger("resonance.models")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LLAMACPP_MODEL_ID = "qwen2.5-0.5b-instruct-q4_k_m"

last_build_error: dict[str, str] = {}


def resolve_path(p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _construct(cfg: S.ModelConfig) -> ModelProvider | None:
    """Instantiate the configured provider without checking availability."""
    if cfg.provider == "scripted":
        return ScriptedProvider(model_id=cfg.model_id or "scripted-v1")
    if cfg.provider == "llamacpp":
        return LlamaCppProvider(
            resolve_path(cfg.local_model_path),
            n_threads=cfg.n_threads,
            temperature=cfg.temperature,
            model_id=cfg.model_id or DEFAULT_LLAMACPP_MODEL_ID,
        )
    if cfg.provider == "anthropic":
        return AnthropicProvider(
            model_id=cfg.model_id,
            api_key_env=cfg.api_key_env or "ANTHROPIC_API_KEY",
            timeout=cfg.timeout_seconds,
            base_url=cfg.base_url,
        )
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model_id=cfg.model_id,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            timeout=cfg.timeout_seconds,
        )
    return None


def build_provider(cfg: S.ModelConfig) -> ModelProvider | None:
    """Return a ready provider, or None (with a logged reason) if unavailable."""
    last_build_error.pop(cfg.provider, None)
    if cfg.provider == "none":
        return None
    try:
        provider = _construct(cfg)
    except Exception as exc:
        reason = f"could not construct provider {cfg.provider!r}: {type(exc).__name__}: {exc}"
        last_build_error[cfg.provider] = redact(reason)
        log.warning(redact(reason))
        return None
    if provider is None:
        last_build_error[cfg.provider] = f"unknown provider {cfg.provider!r}"
        log.warning(last_build_error[cfg.provider])
        return None
    st = provider.status()
    if not st.available:
        last_build_error[cfg.provider] = redact(st.detail)
        log.warning(
            "model provider %s unavailable; using local policy only: %s", cfg.provider, redact(st.detail)
        )
        return None
    return provider


def load_prompts(cfg: S.ModelConfig, prompts_dir: str | Path | None = None) -> PromptLibrary:
    return PromptLibrary(resolve_path(prompts_dir or cfg.prompts_dir))


def build_model_policy(
    cfg: S.ModelConfig, fallback: PolicyProtocol, prompts_dir: str | Path | None = None
) -> PolicyProtocol:
    """ModelPolicy wrapping `fallback`, or `fallback` itself if no provider is usable."""
    if cfg.provider == "none":
        return fallback
    provider = build_provider(cfg)
    if provider is None:
        return fallback
    prompts = load_prompts(cfg, prompts_dir)
    missing = prompts.missing_templates()
    if missing:
        last_build_error[cfg.provider] = f"missing prompt templates in {prompts.directory}: {missing}"
        log.warning(last_build_error[cfg.provider])
        return fallback
    return ModelPolicy(provider, fallback, cfg, prompts)


def status(cfg: S.ModelConfig, policy: object | None = None) -> S.ModelStatus:
    """Truthful status of the configured provider (does not load models or call APIs)."""
    if cfg.provider == "none":
        return S.ModelStatus(
            provider="none",
            model_id="",
            available=False,
            tested_in_this_environment=False,
            accepts_audio=False,
            input_modality="none",
            call_budget=int(cfg.call_budget),
            detail=(
                "Model assistance disabled (model.provider: none). Local policy only. "
                "Local llama.cpp provider "
                + ("is installable here" if llama_cpp_importable() else "is not importable here")
                + "; no provider in this build accepts audio."
            ),
        )
    try:
        provider = _construct(cfg)
    except Exception as exc:
        provider = None
        err = f"{type(exc).__name__}: {exc}"
    else:
        err = f"unknown provider {cfg.provider!r}"
    if provider is None:
        return S.ModelStatus(
            provider=cfg.provider,
            model_id=cfg.model_id,
            available=False,
            tested_in_this_environment=False,
            accepts_audio=False,
            input_modality="none",
            detail=redact(err),
            call_budget=int(cfg.call_budget),
        )
    st = provider.status()
    calls = getattr(policy, "calls_made", 0) if isinstance(policy, ModelPolicy) else 0
    detail = st.detail
    if cfg.use_for:
        detail += f" Used for: {', '.join(cfg.use_for)}."
    else:
        detail += " model.use_for is empty, so no decisions are delegated to the model."
    return st.model_copy(
        update={"detail": redact(detail), "calls_made": int(calls), "call_budget": int(cfg.call_budget)}
    )
