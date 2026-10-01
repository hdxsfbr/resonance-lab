"""Experimental conditions (see docs/CONTRACTS.md, "Conditions").

| name            | learning | memory     | state updates          | coupling | channel |
|-----------------|----------|------------|------------------------|----------|---------|
| full            | on       | on         | on                     | on       | music   |
| no_history      | off      | capacity 0 | on                     | on       | music   |
| state_fixed     | on       | on         | off (held at baseline) | n/a      | music   |
| state_decoupled | on       | on         | on                     | off      | music   |
| symbol          | on       | on         | on                     | on       | symbol  |
| perturbed       | on       | on         | on                     | on       | music + transform in channel |

The SYMBOL baseline is meant to be a strong, fair comparison, never deliberately weakened.
What IS matched between `symbol` and `full`:
  - message identity: the receiver gets symbol_id = the sender's chosen motif index, so every
    distinction the sender can make is transmitted without loss or noise;
  - cardinality and action spaces: same n_motifs messages, same n_patterns targets;
  - learner and update rule: the same linear contextual bandit (one-hot input = tabular case)
    and the same sender Q-learner, learning rates, temperatures and coupling;
  - same task, targets, seeds and RNG streams.
What is NOT matched (by design, these are what the music channel adds or costs):
  - no graded similarity between messages (one-hot codes are orthogonal, so nothing generalises
    from one message to a related one, and partner-similarity input to affiliation is absent);
  - no expressive modulation reaches the receiver (tempo/velocity changes are not transmitted);
  - no acoustic state drive (the hand-authored density/loudness influence needs a heard phrase);
  - no generation jitter or channel perturbation noise.
"""

from __future__ import annotations

from typing import Any

from resonance.schemas import ConditionName, ConditionSpec, ExperimentConfig

_DESCRIPTIONS: dict[str, str] = {
    "full": "Full model: learning, episodic memory, state updates and state-to-policy coupling on; music channel.",
    "no_history": "No history: learning off and memory capacity 0, so behaviour cannot depend on past episodes; state still updates and is coupled.",
    "state_fixed": "State fixed: engineered state held at baseline (no updates), so coupling has nothing to transmit; learning and memory on.",
    "state_decoupled": "State decoupled: state evolves but cannot influence temperature, expressive tempo/velocity or learning rate.",
    "symbol": "Symbol baseline: receiver gets the sender's motif index as a one-hot code (identity preserved) instead of the phrase; same learners and seeds.",
}


def describe(condition: ConditionSpec) -> str:
    name = ConditionName(condition.name).value
    if name == "perturbed":
        return f"Perturbed channel: every sent phrase is transformed by '{condition.perturbation}' before the receiver hears it; everything else as in full."
    return _DESCRIPTIONS[name]


def with_description(condition: ConditionSpec) -> ConditionSpec:
    """Fill `description` with an accurate one-sentence label when it is empty."""
    if condition.description:
        return condition
    return condition.model_copy(update={"description": describe(condition)})


def effective_settings(condition: ConditionSpec, config: ExperimentConfig) -> dict[str, Any]:
    """Resolve what a condition switches on/off, on top of the (live) config."""
    name = ConditionName(condition.name).value
    settings: dict[str, Any] = {
        "learning_enabled": config.learning.enabled,
        "memory_capacity": config.memory.capacity,
        "state_frozen": False,
        "coupling_enabled": config.coupling.enabled,
        "channel": "music",
        "perturbation": "none",
    }
    if name == "no_history":
        settings["learning_enabled"] = False
        settings["memory_capacity"] = 0
    elif name == "state_fixed":
        settings["state_frozen"] = True
    elif name == "state_decoupled":
        settings["coupling_enabled"] = False
    elif name == "symbol":
        settings["channel"] = "symbol"
    elif name == "perturbed":
        settings["perturbation"] = condition.perturbation
    return settings


def standard_conditions(perturbations: tuple[str, ...] = ("transpose", "rhythm_shuffle")) -> list[ConditionSpec]:
    """The six conditions used by batch experiments (perturbed once per perturbation)."""
    specs = [ConditionSpec(name=ConditionName(n)) for n in ("full", "no_history", "state_fixed", "state_decoupled", "symbol")]
    specs += [ConditionSpec(name=ConditionName.perturbed, perturbation=p) for p in perturbations]  # type: ignore[arg-type]
    return [with_description(s) for s in specs]
