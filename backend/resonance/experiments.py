"""Headless batch experiments: conditions x seeds -> per-run metrics -> per-condition aggregates.

Runs (one seed, one condition) are the unit of analysis. Turns within a run are
correlated and are never treated as independent samples. Aggregates report mean
and sample SD over runs.

Probe-style metrics are evaluated AFTER the run, at `temperature_base`, so they
reflect learned associations rather than the momentary state:

  policy_shift_kl[agent_role]   mean over contexts of KL(p_final || p_initial).
                                receiver contexts: the bank motifs at characteristic tempo
                                (one-hot motif codes in the symbol channel); sender contexts: targets.
  state_effect_persistence[id]  copy the agent's end state, add +0.3 activation (pulse), then apply
                                StateEngine updates with neutral inputs (no phrase, score = expected_value
                                so pe = 0) until |activation - baseline| < 0.05; steps counted, cap 200.
  familiar_vs_unfamiliar        mean max-probability over bank motifs (familiar) vs the same motifs
                                transposed +5 and tempo-shifted x1.25 (unfamiliar), per agent as receiver.
  generalization_score          for each direction and target: sender's greedy motif, transposed +5,
                                receiver chooses greedily; mean task score.
  state_similarity              Pearson r of the two agents' activation histories (None if constant).
"""

from __future__ import annotations

import csv
import io
import uuid
from typing import Any

import numpy as np

from resonance.agents.agent import Agent
from resonance.agents.learner import encode_observation
from resonance.agents.policy import softmax
from resonance.agents.state import StateDrive, StateEngine
from resonance.channel import music_observation, symbol_observation
from resonance.conditions import with_description
from resonance.config import load_config
from resonance.music.motifs import base_phrases
from resonance.music.transforms import apply_transform
from resonance.schemas import (
    ConditionAggregate,
    ConditionSpec,
    ExperimentConfig,
    ExperimentRequest,
    ExperimentResult,
    Observation,
    Phrase,
    RunMetrics,
    StateVector,
)
from resonance.session import Session, now_iso

N_BLOCKS = 10
PULSE = 0.3
PERSISTENCE_TOLERANCE = 0.05
PERSISTENCE_CAP = 200
EPS = 1e-12

EXPERIMENT_NOTES = [
    "Runs are the unit of analysis; turns within a run are correlated.",
    "Controls may match or exceed the full condition; report as-is.",
    "State variables are engineered quantities with operational definitions, not measurements of feelings.",
    "state_fixed and state_decoupled differ only in whether state values evolve; in this design state reaches behaviour only through coupling, so their behaviour can coincide.",
    "Probe metrics (KL, familiarity, generalization) are evaluated at temperature_base after the run.",
]


# ---------------------------------------------------------------- probe helpers
def probe_observation(session: Session, phrase: Phrase, sender_id: str, motif_index: int) -> Observation:
    if session.settings["channel"] == "symbol":
        return symbol_observation(motif_index, sender_id, session.step_index)
    return music_observation(phrase, sender_id, session.step_index, session.config.music.feature_weights)


def receiver_probs(session: Session, agent: Agent, obs: Observation, tau: float | None = None) -> np.ndarray:
    x = encode_observation(obs, session.config.music.n_motifs, session.reference)
    return softmax(agent.receiver_learner.scores(x), tau or session.config.learning.temperature_base)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.clip(p, EPS, 1.0), np.clip(q, EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _bank(session: Session, agent: Agent) -> list[Phrase]:
    return base_phrases(session.config.music.motif_bank, agent.params.instrument, n=session.config.music.n_motifs)


def policy_shift(session: Session) -> dict[str, float]:
    tau = session.config.learning.temperature_base
    out: dict[str, float] = {}
    for agent in session.agents:
        partner = session.agents[1] if agent is session.agents[0] else session.agents[0]
        fresh = Agent(agent.params, session.config, channel=session.settings["channel"], allow_model=False)
        recv = []
        for i, phrase in enumerate(_bank(session, partner)):
            obs = probe_observation(session, phrase, partner.id, i)
            recv.append(kl(receiver_probs(session, agent, obs, tau), receiver_probs(session, fresh, obs, tau)))
        send = [
            kl(softmax(agent.sender_learner.values(t), tau), softmax(fresh.sender_learner.values(t), tau))
            for t in range(session.task.n_patterns)
        ]
        out[f"{agent.id}_receiver"] = round(float(np.mean(recv)), 6)
        out[f"{agent.id}_sender"] = round(float(np.mean(send)), 6)
    return out


def persistence(state: StateVector, agent: Agent, session: Session) -> int:
    """Steps for activation to return within tolerance of baseline after a +0.3 pulse."""
    s = state.model_copy(update={"activation": min(1.0, state.activation + PULSE)})
    base_a = agent.baseline.activation
    for steps in range(PERSISTENCE_CAP + 1):
        if abs(s.activation - base_a) < PERSISTENCE_TOLERANCE:
            return steps
        drive = StateDrive(score=s.expected_value, heard=None, partner_similarity=None)
        s, _ = StateEngine.update(s, drive, session.config.state, agent.params.sensitivity, False, agent.baseline)
    return PERSISTENCE_CAP


def familiarity(session: Session) -> dict[str, float]:
    tau = session.config.learning.temperature_base
    out: dict[str, float] = {}
    rng = np.random.default_rng(0)
    for agent in session.agents:
        partner = session.agents[1] if agent is session.agents[0] else session.agents[0]
        fam, unf = [], []
        for i, phrase in enumerate(_bank(session, partner)):
            novel = apply_transform(apply_transform(phrase, "transpose", rng), "tempo_shift", rng)
            fam.append(receiver_probs(session, agent, probe_observation(session, phrase, partner.id, i), tau).max())
            unf.append(receiver_probs(session, agent, probe_observation(session, novel, partner.id, i), tau).max())
        out[f"{agent.id}_familiar"] = round(float(np.mean(fam)), 6)
        out[f"{agent.id}_unfamiliar"] = round(float(np.mean(unf)), 6)
    return out


def generalization(session: Session) -> float:
    scores = []
    rng = np.random.default_rng(0)
    for sender in session.agents:
        receiver = session.agents[1] if sender is session.agents[0] else session.agents[0]
        bank = _bank(session, sender)
        for t in range(session.task.n_patterns):
            m = int(np.argmax(sender.sender_learner.values(t)))
            phrase = apply_transform(bank[m], "transpose", rng)
            obs = probe_observation(session, phrase, sender.id, m)
            x = encode_observation(obs, session.config.music.n_motifs, session.reference)
            chosen = int(np.argmax(receiver.receiver_learner.scores(x)))
            scores.append(session.task.score(chosen, t))
    return round(float(np.mean(scores)), 6)


def activation_correlation(session: Session) -> float | None:
    a = np.array([s.activation for s in session.agents[0].state_history])
    b = np.array([s.activation for s in session.agents[1].state_history])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 6)


# ---------------------------------------------------------------- runs
def block_means(scores: list[float], n_blocks: int = N_BLOCKS) -> list[float]:
    if not scores:
        return []
    return [round(float(np.mean(b)), 6) for b in np.array_split(np.asarray(scores), min(n_blocks, len(scores)))]


def metrics_from_session(session: Session) -> RunMetrics:
    scores = session.scores
    n = len(scores)
    k = max(1, int(round(0.2 * n)))
    extra: dict[str, float] = {"chance_score": round(session.task.chance_score(), 6)}
    for a in session.agents:
        for dim, v in a.state.model_dump().items():
            extra[f"{a.id}_final_{dim}"] = round(float(v), 6)
    return RunMetrics(
        seed=session.seed,
        condition=session.condition.name.value if hasattr(session.condition.name, "value") else str(session.condition.name),
        perturbation=session.condition.perturbation,
        episodes=n,
        mean_score=round(float(np.mean(scores)), 6) if n else 0.0,
        final_block_score=round(float(np.mean(scores[-k:])), 6) if n else 0.0,
        first_block_score=round(float(np.mean(scores[:k])), 6) if n else 0.0,
        success_rate=round(float(np.mean([s == 1.0 for s in scores])), 6) if n else 0.0,
        learning_curve=block_means(scores),
        policy_shift_kl=policy_shift(session),
        state_effect_persistence={a.id: float(persistence(a.state, a, session)) for a in session.agents},
        familiar_vs_unfamiliar=familiarity(session),
        generalization_score=generalization(session),
        state_similarity=activation_correlation(session),
        extra=extra,
    )


def run_single(config: ExperimentConfig, condition: ConditionSpec, seed: int) -> RunMetrics:
    session = Session(config, with_description(condition), seed, mode="live")
    session.step(config.episodes)
    return metrics_from_session(session)


def _sd(values: list[float]) -> float:
    return round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0


def aggregate(runs: list[RunMetrics]) -> list[ConditionAggregate]:
    groups: dict[tuple[str, str], list[RunMetrics]] = {}
    for r in runs:
        groups.setdefault((r.condition, r.perturbation), []).append(r)
    out = []
    for (cond, pert), rs in groups.items():
        finals = [r.final_block_score for r in rs]
        succ = [r.success_rate for r in rs]
        width = min(len(r.learning_curve) for r in rs)
        curves = np.array([r.learning_curve[:width] for r in rs])
        out.append(
            ConditionAggregate(
                condition=cond,
                perturbation=pert,
                n_runs=len(rs),
                mean_final_block_score=round(float(np.mean(finals)), 6),
                sd_final_block_score=_sd(finals),
                mean_success_rate=round(float(np.mean(succ)), 6),
                sd_success_rate=_sd(succ),
                mean_learning_curve=[round(float(v), 6) for v in curves.mean(axis=0)],
                sd_learning_curve=[round(float(v), 6) for v in (curves.std(axis=0, ddof=1) if len(rs) > 1 else np.zeros(width))],
                per_run_final_block=finals,
            )
        )
    return out


def run_experiment(req: ExperimentRequest) -> ExperimentResult:
    config = req.config or load_config()
    runs = [run_single(config, with_description(c), seed) for c in req.conditions for seed in req.seeds]
    return ExperimentResult(
        id=uuid.uuid4().hex[:12],
        name=req.name,
        created_at=now_iso(),
        config=config,
        runs=runs,
        aggregates=aggregate(runs),
        notes=list(EXPERIMENT_NOTES),
    )


# ---------------------------------------------------------------- CSV
def _run_row(r: RunMetrics) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": "run",
        "condition": r.condition,
        "perturbation": r.perturbation,
        "seed": r.seed,
        "n_runs": 1,
        "episodes": r.episodes,
        "mean_score": r.mean_score,
        "first_block_score": r.first_block_score,
        "final_block_score": r.final_block_score,
        "success_rate": r.success_rate,
        "generalization_score": r.generalization_score,
        "state_similarity": r.state_similarity,
    }
    for prefix, d in (("kl_", r.policy_shift_kl), ("persistence_", r.state_effect_persistence), ("fam_", r.familiar_vs_unfamiliar)):
        row.update({f"{prefix}{k}": v for k, v in d.items()})
    row.update({f"curve_{i}": v for i, v in enumerate(r.learning_curve)})
    return row


def _aggregate_row(a: ConditionAggregate) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": "aggregate",
        "condition": a.condition,
        "perturbation": a.perturbation,
        "n_runs": a.n_runs,
        "final_block_score": a.mean_final_block_score,
        "sd_final_block_score": a.sd_final_block_score,
        "success_rate": a.mean_success_rate,
        "sd_success_rate": a.sd_success_rate,
    }
    row.update({f"curve_{i}": v for i, v in enumerate(a.mean_learning_curve)})
    return row


def to_csv(result: ExperimentResult) -> str:
    rows = [_run_row(r) for r in result.runs] + [_aggregate_row(a) for a in result.aggregates]
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buf.getvalue()
