"""Test/demo support for the model layer: a minimal local policy and context
builders. Not used by the simulation. WS1's real LocalPolicy replaces
`FakeLocalPolicy` in production; this exists so the model layer can be tested
without importing simulation-core modules.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from resonance import schemas as S
from resonance.policy_types import ReceiverContext, SenderContext

DEMO_PATTERNS: list[S.TimingPattern] = [
    S.TimingPattern(id=0, name="four_on_floor", onsets=[0.0, 1.0, 2.0, 3.0]),
    S.TimingPattern(id=1, name="offbeats", onsets=[0.5, 1.5, 2.5, 3.5]),
    S.TimingPattern(id=2, name="syncopated", onsets=[0.0, 0.75, 1.5, 2.5, 3.0]),
    S.TimingPattern(id=3, name="sparse", onsets=[0.0, 2.0]),
]
DEMO_MOTIFS = [
    "rising_steps",
    "falling_steps",
    "arch",
    "valley",
    "repeated_note",
    "wide_leaps",
    "syncopated_jab",
    "slow_drone",
]


class FakeLocalPolicy:
    """Softmax over the local learner's estimates, sampled with ctx.rng (one draw)."""

    kind = "local"

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _softmax(values: list[float], tau: float) -> list[float]:
        m = max(values)
        ex = [math.exp((v - m) / tau) for v in values]
        z = sum(ex)
        return [e / z for e in ex]

    def _decide(self, values: list[float], ctx: Any, role: str) -> S.PolicyTrace:
        self.calls += 1
        tau = float(ctx.config.learning.temperature_base)
        probs = self._softmax(list(values), tau)
        draw = float(ctx.rng.random())
        acc, chosen = 0.0, len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if draw < acc:
                chosen = i
                break
        return S.PolicyTrace(
            role=role,
            policy_kind="local",
            scores=list(values),
            probabilities=probs,
            temperature=tau,
            temperature_base=tau,
            coupling_enabled=ctx.coupling_enabled,
            chosen=chosen,
            exploration_draw=draw,
            notes=["fake local policy (tests)"],
        )

    def choose_pattern(self, ctx: ReceiverContext) -> S.PolicyTrace:
        return self._decide(ctx.learner_scores, ctx, "receiver")

    def choose_motif(self, ctx: SenderContext) -> S.PolicyTrace:
        return self._decide(ctx.learner_values, ctx, "sender")


def demo_phrase(phrase_id: str = "p-demo", tags: list[str] | None = None) -> S.Phrase:
    notes = [
        S.Note(pitch=p, onset=o, duration=0.5, velocity=v)
        for p, o, v in [(60, 0.0, 90), (62, 1.0, 84), (64, 2.0, 96), (67, 3.0, 88), (64, 4.5, 80)]
    ]
    return S.Phrase(id=phrase_id, notes=notes, tempo_bpm=110, length_beats=8.0, tags=tags or [])


def demo_features() -> S.SymbolicFeatures:
    return S.SymbolicFeatures(
        note_count=5,
        duration_seconds=4.36,
        note_density=1.15,
        rhythmic_regularity=0.71,
        syncopation=0.0,
        mean_pitch=63.4,
        pitch_range=7,
        contour=0.42,
        contour_class="arch",
        repetition=0.25,
        interval_histogram=[0.0, 0.5, 0.5, 0.0, 0.0],
        mean_velocity=87.6,
        velocity_variation=0.06,
        dissonance_proxy=0.0,
        tempo_bpm=110.0,
        vector=[0.38, 0.71, 0.0, 0.55, 0.29, 0.71, 0.25, 0.0, 0.5, 0.5, 0.0, 0.0, 0.62, 0.12, 0.0, 0.38],
    )


def demo_memory(
    step: int, action: int, score: float, sim: float, role: str = "receiver"
) -> S.RetrievedMemory:
    st = S.StateVector(activation=0.5, expected_value=0.5, uncertainty=0.5, affiliation=0.0)
    item = S.MemoryItem(
        step=step,
        role=role,
        partner_id="A",
        phrase_id=f"p{step}",
        motif_id="arch",
        feature_vector=demo_features().vector,
        action=action,
        score=score,
        state_before=st,
        state_after=st,
    )
    return S.RetrievedMemory(item=item, similarity=sim)


def make_receiver_ctx(
    *,
    channel: str = "music",
    seed: int = 0,
    learner_scores: list[float] | None = None,
    emit=None,
    config: S.ExperimentConfig | None = None,
    phrase_tags: list[str] | None = None,
    retrieved: list[S.RetrievedMemory] | None = None,
    step: int = 12,
) -> ReceiverContext:
    cfg = config or S.ExperimentConfig()
    if channel == "symbol":
        obs = S.Observation(channel="symbol", symbol_id=5, sender_id="A", step=step)
    else:
        obs = S.Observation(
            channel="music",
            phrase=demo_phrase(tags=phrase_tags),
            features=demo_features(),
            sender_id="A",
            step=step,
        )
    kw: dict[str, Any] = {}
    if emit is not None:
        kw["emit"] = emit
    return ReceiverContext(
        agent_id="B",
        partner_id="A",
        step=step,
        observation=obs,
        n_patterns=4,
        patterns=list(DEMO_PATTERNS),
        learner_scores=learner_scores or [0.10, 0.62, 0.20, 0.08],
        state=S.StateVector(activation=0.61, expected_value=0.48, uncertainty=0.37, affiliation=0.12),
        coupling_enabled=True,
        config=cfg,
        rng=np.random.default_rng(seed),
        retrieved=retrieved
        if retrieved is not None
        else [demo_memory(7, 1, 1.0, 0.93), demo_memory(3, 2, 0.25, 0.71)],
        **kw,
    )


def make_sender_ctx(
    *, target_id: int = 2, seed: int = 0, emit=None, config: S.ExperimentConfig | None = None
) -> SenderContext:
    cfg = config or S.ExperimentConfig()
    kw: dict[str, Any] = {}
    if emit is not None:
        kw["emit"] = emit
    return SenderContext(
        agent_id="A",
        partner_id="B",
        step=13,
        target_id=target_id,
        target_pattern=DEMO_PATTERNS[target_id],
        n_motifs=8,
        motif_names=list(DEMO_MOTIFS),
        learner_values=[0.5, 0.5, 0.71, 0.5, 0.42, 0.5, 0.66, 0.31],
        state=S.StateVector(activation=0.55, expected_value=0.52, uncertainty=0.41, affiliation=0.2),
        coupling_enabled=True,
        config=cfg,
        rng=np.random.default_rng(seed),
        retrieved=[demo_memory(9, 2, 1.0, 0.88, role="sender")],
        **kw,
    )


class EventSink:
    """Collects (event_type, payload) pairs emitted by a policy."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    def of(self, event_type: str) -> list[dict[str, Any]]:
        return [p for t, p in self.events if t == event_type]
