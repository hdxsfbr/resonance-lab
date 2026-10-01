"""An agent: engineered state + episodic memory + learned associations + a policy.

Every agent can act as sender or receiver; roles alternate each episode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from resonance.agents.learner import ReceiverLearner, SenderLearner
from resonance.agents.memory import EpisodicMemory
from resonance.agents.policy import LocalPolicy
from resonance.config import resolve_project_path
from resonance.policy_types import PolicyProtocol
from resonance.schemas import (
    FEATURE_NAMES,
    AgentParams,
    AgentSnapshot,
    ExperimentConfig,
    LearnerSnapshot,
    PolicyTrace,
    RetrievedMemory,
    StateUpdateInputs,
    StateVector,
)

log = logging.getLogger(__name__)


def build_policy(params: AgentParams, config: ExperimentConfig, allow_model: bool = True) -> tuple[PolicyProtocol, list[dict[str, Any]]]:
    """LocalPolicy, or the WS3 model policy (with LocalPolicy as fallback) if requested.

    Returns (policy, warnings); each warning becomes a `model_call` event (ok=False).
    Works when `resonance.models.registry` is absent or fails.
    """
    local = LocalPolicy()
    if params.policy_kind != "model" or not allow_model:
        return local, []
    try:
        from resonance.models.registry import build_model_policy  # type: ignore[import-not-found]

        prompts_dir: Path = resolve_project_path(config.model.prompts_dir)
        policy = build_model_policy(config.model, fallback=local, prompts_dir=prompts_dir)
    except Exception as exc:  # ImportError or any provider/setup failure -> local policy
        log.warning("model policy unavailable for agent %s: %s", params.id, exc)
        return local, [{"ok": False, "error": f"{type(exc).__name__}: {exc}", "fallback": "local", "agent_id": params.id}]
    if policy is local or getattr(policy, "kind", "local") != "model":
        reason = f"no usable model provider (model.provider={config.model.provider!r})"
        return local, [{"ok": False, "error": reason, "fallback": "local", "agent_id": params.id}]
    return policy, []


class Agent:
    def __init__(
        self,
        params: AgentParams,
        config: ExperimentConfig,
        *,
        channel: str = "music",
        memory_capacity: int | None = None,
        coupling_enabled: bool = True,
        frozen_state: bool = False,
        init_seed: int = 0,
        allow_model: bool = True,
    ) -> None:
        self.id = params.id
        self.params = params
        self.baseline: StateVector = params.baseline_override or config.state.baseline
        self.state: StateVector = self.baseline.model_copy()
        self.memory = EpisodicMemory(config.memory.capacity if memory_capacity is None else memory_capacity)
        n_motifs, n_patterns = config.music.n_motifs, config.task.n_patterns
        input_dim = len(FEATURE_NAMES) if channel == "music" else n_motifs
        self.receiver_learner = ReceiverLearner(input_dim, n_patterns, config.learning, init_seed)
        self.sender_learner = SenderLearner(n_patterns, n_motifs, config.learning)
        self.policy, self.init_warnings = build_policy(params, config, allow_model)
        self.frozen_state = frozen_state
        self.coupling_enabled = coupling_enabled
        self.state_history: list[StateVector] = [self.state]
        self.information_received: list[dict[str, Any]] = []
        self.last_trace: PolicyTrace | None = None
        self.last_state_inputs: StateUpdateInputs | None = None
        self.last_retrieved: list[RetrievedMemory] = []
        self.last_sent_vector: list[float] | None = None  # own last phrase phi (partner similarity)

    # -- learning-rate coupling -------------------------------------------------
    def effective_learning_rate(self, base: float, affiliation_to_lr: float = 0.5) -> float:
        """lr_eff = base * (1 + k * affiliation) if coupled else base; clipped to [0.01, 1]."""
        lr = base * (1.0 + affiliation_to_lr * self.state.affiliation) if self.coupling_enabled else base
        return float(np.clip(lr, 0.01, 1.0))

    # -- resets -----------------------------------------------------------------
    def reset_memory(self) -> None:
        """Clear episodic memory AND learned associations. State untouched."""
        self.memory.clear()
        self.receiver_learner.reset()
        self.sender_learner.reset()

    def reset_state(self) -> None:
        """State := baseline. Memory and learned associations untouched."""
        self.state = self.baseline.model_copy()
        self.state_history.append(self.state)

    def set_state(self, state: StateVector) -> None:
        self.state = state
        self.state_history.append(state)

    # -- inspection ---------------------------------------------------------------
    def learner_snapshot(self) -> LearnerSnapshot:
        r, s = self.receiver_learner, self.sender_learner
        return LearnerSnapshot(
            receiver_weights=[[round(float(v), 6) for v in row] for row in r.W],
            receiver_bias=[round(float(v), 6) for v in r.b],
            sender_values=[[round(float(v), 6) for v in row] for row in s.Q],
            sender_counts=[[int(v) for v in row] for row in s.counts],
            updates=r.updates + s.updates,
        )

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            id=self.id,
            name=self.params.name,
            color=self.params.color,
            instrument=self.params.instrument,
            params=self.params,
            state=self.state,
            baseline=self.baseline,
            memory_size=len(self.memory),
            learner=self.learner_snapshot(),
            last_trace=self.last_trace,
            frozen_state=self.frozen_state,
            coupling_enabled=self.coupling_enabled,
            policy_kind=self.params.policy_kind,
        )
