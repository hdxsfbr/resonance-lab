"""Policy interface shared by the local simulation policy (WS1) and the
model-assisted policy (WS3). Both implement `PolicyProtocol`.

Contexts carry EVERYTHING a policy may legitimately see. In particular the
`ReceiverContext` has no target: information isolation is enforced by
construction here, and checked by tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from resonance import schemas as S

EmitFn = Callable[[str, dict[str, Any]], None]
"""emit(event_type, payload) — append an event for the current step (e.g. "model_call")."""


@dataclass
class SenderContext:
    agent_id: str
    partner_id: str
    step: int
    target_id: int  # the sender legitimately knows the target
    target_pattern: S.TimingPattern
    n_motifs: int
    motif_names: list[str]
    learner_values: list[float]  # Q[target][motif] from the sender learner (local estimate)
    state: S.StateVector
    coupling_enabled: bool
    config: S.ExperimentConfig
    rng: np.random.Generator  # this agent's policy stream
    retrieved: list[S.RetrievedMemory] = field(default_factory=list)
    emit: EmitFn = lambda _t, _p: None


@dataclass
class ReceiverContext:
    agent_id: str
    partner_id: str
    step: int
    observation: S.Observation  # exactly what was received; never contains the target
    n_patterns: int
    patterns: list[S.TimingPattern]
    learner_scores: list[float]  # per-pattern scores from the receiver learner (local estimate)
    state: S.StateVector
    coupling_enabled: bool
    config: S.ExperimentConfig
    rng: np.random.Generator
    retrieved: list[S.RetrievedMemory] = field(default_factory=list)
    emit: EmitFn = lambda _t, _p: None


class PolicyProtocol(Protocol):
    kind: str  # "local" | "model"

    def choose_motif(self, ctx: SenderContext) -> S.PolicyTrace: ...

    def choose_pattern(self, ctx: ReceiverContext) -> S.PolicyTrace: ...
