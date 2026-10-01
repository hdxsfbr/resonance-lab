"""Generated narrative for an episode — strictly a LABELLED INTERPRETATION.

The caller wraps the returned text as an event of type "generated_narrative"
with payload `narrative_event_payload(...)`:
    {text, provider, model_id, label: "generated interpretation — not a measurement"}

Invariant: narrative text is display-only. It must never be read by the
simulation — not by state updates, learners, memory, metrics or experiments.
Nothing in `resonance/` other than this package and the event log handles it.

`generate_narrative` makes ONE call with no retries and no budget accounting;
use `ModelPolicy.generate_narrative` for the budgeted, event-emitting variant.
"""

from __future__ import annotations

from typing import Any

from resonance import schemas as S
from resonance.models.base import ModelProvider
from resonance.models.policy import (
    NARRATIVE_LABEL,
    NARRATIVE_MAX_TOKENS,
    NARRATIVE_SCHEMA,
    narrative_enabled,
    narrative_event_payload,
    narrative_prompts,
    validate_narrative,
)
from resonance.models.prompts import PromptLibrary
from resonance.models.redact import redact

__all__ = ["generate_narrative", "narrative_event_payload", "narrative_enabled", "NARRATIVE_LABEL"]


def generate_narrative(
    provider: ModelProvider,
    prompts: PromptLibrary,
    episode_summary_dict: dict[str, Any],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    cfg: S.ModelConfig | None = None,
    temperature: float = 0.2,
) -> str | None:
    """Return generated narrative text, or None on any failure / when disabled.

    If `cfg` is given, nothing is generated unless cfg.narrative_enabled and
    "narrative" in cfg.use_for.
    """
    if cfg is not None and not narrative_enabled(cfg):
        return None
    try:
        system, user = narrative_prompts(prompts, episode_summary_dict, state_before, state_after)
        req = S.ModelRequest(
            purpose="narrative",
            system_prompt=system,
            user_prompt=user,
            json_schema=NARRATIVE_SCHEMA,
            input_modality="symbolic_features",
            max_tokens=NARRATIVE_MAX_TOKENS,
            temperature=cfg.temperature if cfg is not None else temperature,
        )
        resp = provider.complete(req)
    except Exception:
        return None
    if not resp.ok:
        return None
    value, _ = validate_narrative(resp.parsed)
    return redact(value["narrative"]) if value else None
