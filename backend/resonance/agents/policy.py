"""Local (non-model) policy: softmax over learned scores with a state-coupled temperature.

  temperature = clip(tau_base * (1 + k_u * (U - 0.5)) * (1 + k_a * (0.5 - A)), tau_min, tau_max)
                if coupling is enabled, else tau_base
  p_i = softmax(scores / temperature)_i
  chosen = first i with cumsum(p)_i > u,  u ~ Uniform[0,1) from the agent's policy stream

Expressive modulation (sender's phrase rendering), when coupling is enabled:
  tempo_multiplier = 1 + activation_to_tempo * (A - 0.5)
  velocity_offset  = activation_to_velocity * (A - 0.5)
otherwise {tempo_multiplier: 1.0, velocity_offset: 0.0}.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from resonance.policy_types import ReceiverContext, SenderContext
from resonance.schemas import ExperimentConfig, PolicyTrace, StateVector


def temperature(state: StateVector, cfg: ExperimentConfig, coupling_enabled: bool) -> float:
    base = cfg.learning.temperature_base
    if not coupling_enabled:
        return base
    c = cfg.coupling
    tau = base * (1.0 + c.uncertainty_to_temperature * (state.uncertainty - 0.5))
    tau *= 1.0 + c.activation_to_temperature * (0.5 - state.activation)
    return float(np.clip(tau, c.temperature_min, c.temperature_max))


def expressive_modulation(state: StateVector, cfg: ExperimentConfig, coupling_enabled: bool) -> dict[str, float]:
    if not coupling_enabled:
        return {"tempo_multiplier": 1.0, "velocity_offset": 0.0}
    c = cfg.coupling
    return {
        "tempo_multiplier": round(1.0 + c.activation_to_tempo * (state.activation - 0.5), 6),
        "velocity_offset": round(c.activation_to_velocity * (state.activation - 0.5), 6),
    }


def softmax(scores: Sequence[float], tau: float) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / max(tau, 1e-9)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sample(probs: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    u = float(rng.random())
    idx = int(np.searchsorted(np.cumsum(probs), u, side="right"))
    return min(idx, len(probs) - 1), u


def _trace(
    role: str, scores: Sequence[float], state: StateVector, cfg: ExperimentConfig, coupling: bool, rng: np.random.Generator
) -> PolicyTrace:
    tau = temperature(state, cfg, coupling)
    probs = softmax(scores, tau)
    chosen, u = sample(probs, rng)
    notes = [f"temperature={'state-coupled' if coupling else 'fixed (coupling off)'}"]
    return PolicyTrace(
        role=role,  # type: ignore[arg-type]
        policy_kind="local",
        scores=[round(float(s), 6) for s in scores],
        probabilities=[round(float(p), 6) for p in probs],
        temperature=round(tau, 6),
        temperature_base=cfg.learning.temperature_base,
        coupling_enabled=coupling,
        chosen=chosen,
        exploration_draw=round(u, 6),
        expressive_modulation=expressive_modulation(state, cfg, coupling),
        notes=notes,
    )


class LocalPolicy:
    kind = "local"

    def choose_motif(self, ctx: SenderContext) -> PolicyTrace:
        trace = _trace("sender", ctx.learner_values, ctx.state, ctx.config, ctx.coupling_enabled, ctx.rng)
        trace.notes.append("scores = sender Q[target][motif] (learned)")
        return trace

    def choose_pattern(self, ctx: ReceiverContext) -> PolicyTrace:
        trace = _trace("receiver", ctx.learner_scores, ctx.state, ctx.config, ctx.coupling_enabled, ctx.rng)
        trace.notes.append("scores = receiver q = W x + b (learned)")
        return trace
