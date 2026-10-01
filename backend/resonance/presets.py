"""Guided presets. Each one names its manipulation and what would count as evidence of
the intended COMPUTATIONAL effect (in mechanism terms: learned associations, engineered
state, coupling). Nothing here is evidence about feelings.

Comparisons are measured by probing agents with identical phrase, state and RNG draw,
so differences are attributable to what differs by construction (learned associations).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from resonance.agents.agent import Agent
from resonance.agents.learner import encode_observation
from resonance.agents.policy import sample, softmax, temperature
from resonance.agents.state import StateDrive, StateEngine
from resonance.channel import partner_similarity
from resonance.config import load_config
from resonance.experiments import block_means, policy_shift, probe_observation
from resonance.music.motifs import generate, motif_index_from_id
from resonance.schemas import (
    STATE_DIMS,
    ConditionName,
    ConditionSpec,
    ExperimentConfig,
    Phrase,
    PresetInfo,
    PresetResult,
    StateVector,
)
from resonance.session import EpisodeScript, Session

SCRIPTED_MOTIF = 3  # "hop"
SCRIPTED_TARGET = 2  # "gallop"
TEMPLATED = "[templated summary, not model-generated] "

PRESETS: dict[str, PresetInfo] = {
    "first_encounter": PresetInfo(
        name="first_encounter",
        title="First encounter",
        summary="Two fresh agents with no shared history play the timing task for 120 episodes under the full condition.",
        manipulation=(
            "None beyond starting from zero: receiver weights 0, sender Q = optimistic_init, empty episodic "
            "memory, engineered state at baseline."
        ),
        evidence_criterion=(
            "A shared convention forms if the block learning curve rises above the chance score of a random "
            "receiver (0.326 for 4 patterns with partial credit) and the final 20% of episodes beats the first 20%; "
            "policy-shift KL > 0 shows the learned associations moved both policies. A curve flat at chance means "
            "no convention formed in this run."
        ),
        default_episodes=120,
        condition=ConditionSpec(name=ConditionName.full),
    ),
    "shared_history": PresetInfo(
        name="shared_history",
        title="Shared history vs memory reset",
        summary=(
            "After 120 episodes of shared history, the motif the next receiver has most often seen succeed is "
            "replayed to that receiver and to a clone of it whose memory and learned associations were reset."
        ),
        manipulation=(
            "reset_memory on a clone of the receiver (episodic memory cleared and learner weights re-initialised); "
            "the clone keeps the identical state vector and hears the identical phrase with the identical RNG draw."
        ),
        evidence_criterion=(
            "History dependence is shown if the trained receiver concentrates probability on the pattern the "
            "motif has predicted (lower entropy, that pattern as top choice) while the reset clone stays near "
            "uniform, and if the resulting state change differs (through the outcome-driven terms). Because phrase, "
            "state and RNG draw are identical, differences are attributable to learned associations."
        ),
        constructed_note="The probe motif is selected post hoc as the one most often followed by an exact match for this receiver.",
        default_episodes=120,
        condition=ConditionSpec(name=ConditionName.full),
    ),
    "same_phrase_different_history": PresetInfo(
        name="same_phrase_different_history",
        title="Same phrase, different history",
        summary=(
            "Two sessions with the same seed; a scripted sender plays motif 'hop' on every episode. In session 1 "
            "the target is always pattern 2; in session 2 it is drawn uniformly. Then both receivers hear 'hop' "
            "with identical state and RNG."
        ),
        manipulation=(
            "Only the statistical relation between the phrase and the target differs: predictive (session 1) vs "
            "uninformative (session 2). Sender learning is off (scripted); receivers learn normally."
        ),
        evidence_criterion=(
            "The same phrase has a history-dependent effect if the session-1 receiver puts high probability on "
            "pattern 2 with low entropy while the session-2 receiver does not, and if the state responses to the "
            "identical probe (identical starting state) differ via the outcome-driven terms. The end-of-training "
            "states also differ (expected_value, uncertainty, affiliation) because the histories differ."
        ),
        constructed_note=(
            "Constructed: this contrast is built by the training schedule (scripted sender, fixed vs uniform "
            "targets). It shows that the mechanism can produce history-dependent responses to an identical "
            "phrase; it is not evidence that such histories arise spontaneously between the agents."
        ),
        default_episodes=60,
        condition=ConditionSpec(name=ConditionName.full),
    ),
}


def _get(name: str) -> PresetInfo:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]


def preset_session_config(
    name: str, base: ExperimentConfig | None = None, episodes: int | None = None
) -> tuple[ExperimentConfig, ConditionSpec, EpisodeScript | None]:
    """(config, condition, script) for starting a LIVE session of this preset.

    For same_phrase_different_history the live session is the 'predictive' variant (session 1).
    """
    info = _get(name)
    config = (base or load_config()).model_copy(update={"episodes": episodes or info.default_episodes}, deep=True)
    script = None
    if name == "same_phrase_different_history":
        script = EpisodeScript(SCRIPTED_MOTIF, SCRIPTED_TARGET, "motif 'hop' every episode; target always pattern 2")
    return config, info.condition, script


# ---------------------------------------------------------------- probes
def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return round(float(-np.sum(p * np.log(p))), 6)


def _delta(before: StateVector, after: StateVector) -> dict[str, float]:
    b, a = before.model_dump(), after.model_dump()
    return {k: round(a[k] - b[k], 6) for k in STATE_DIMS}


def probe_receiver(
    session: Session, agent: Agent, phrase: Phrase, motif_index: int, target: int, rng: np.random.Generator,
    state: StateVector | None = None,
) -> dict[str, Any]:
    """Present `phrase` to `agent` WITHOUT mutating it: policy probabilities, sampled choice, score,
    and the state change one StateEngine update would produce from `state` (default: agent.state)."""
    state = state or agent.state
    sender_id = next(a.id for a in session.agents if a.id != agent.id)
    obs = probe_observation(session, phrase, sender_id, motif_index)
    x = encode_observation(obs, session.config.music.n_motifs, session.reference)
    tau = temperature(state, session.config, agent.coupling_enabled)
    probs = softmax(agent.receiver_learner.scores(x), tau)
    chosen, _ = sample(probs, rng)
    score = session.task.score(chosen, target)
    heard = obs.features if obs.channel == "music" else None
    vec = obs.features.vector if obs.features is not None else None
    sim = partner_similarity(vec, agent.last_sent_vector, session.reference) if heard is not None else None
    after, _ = StateEngine.update(
        state, StateDrive(score=score, heard=heard, partner_similarity=sim), session.config.state,
        agent.params.sensitivity, agent.frozen_state, agent.baseline,
    )
    return {
        "probs": [round(float(p), 6) for p in probs],
        "top": int(np.argmax(probs)),
        "entropy": _entropy(probs),
        "chosen": chosen,
        "score": round(score, 6),
        "temperature": round(tau, 6),
        "state_delta": _delta(state, after),
    }


def reset_clone(session: Session, agent: Agent) -> Agent:
    """A copy of `agent` after reset_memory: same params/state/flags, empty memory, fresh learners."""
    clone = Agent(
        agent.params, session.config, channel=session.settings["channel"],
        memory_capacity=agent.memory.capacity, coupling_enabled=agent.coupling_enabled,
        frozen_state=agent.frozen_state, allow_model=False,
    )
    clone.state = agent.state.model_copy()
    clone.baseline = agent.baseline
    clone.last_sent_vector = agent.last_sent_vector
    return clone


def _probe_phrase(session: Session, motif: int) -> Phrase:
    sender, _ = session.roles(session.step_index)
    mc = session.config.music
    return generate(
        motif, None, sender.params.instrument, None, 0.0, agent_id=sender.id, step=session.step_index,
        bank=mc.motif_bank, length_beats=mc.phrase_length_beats, tempo_bounds=(mc.tempo_min, mc.tempo_max),
        origin_kind="replay", phrase_id=f"probe-{session.step_index}-m{motif}",
    )


def most_successful_motif(agent: Agent, bank: str) -> tuple[int, int]:
    """(motif index, pattern) most often followed by an exact match in the agent's receiver memories."""
    items = [m for m in agent.memory.items() if m.role == "receiver" and m.motif_id]
    wins = [m for m in items if m.score == 1.0] or items
    if not wins:
        return 0, 0
    motif_id, _ = Counter(m.motif_id for m in wins).most_common(1)[0]
    pattern, _ = Counter(m.action for m in wins if m.motif_id == motif_id).most_common(1)[0]
    return motif_index_from_id(motif_id, bank) or 0, int(pattern)


# ---------------------------------------------------------------- runners
def _run_first_encounter(seed: int, config: ExperimentConfig) -> tuple[dict[str, Any], list[Session], str]:
    s = Session(config, PRESETS["first_encounter"].condition, seed, preset="first_encounter")
    s.step(config.episodes)
    n = len(s.scores)
    k = max(1, round(0.2 * n))
    comp = {
        "episodes": n,
        "chance_score": round(s.task.chance_score(), 6),
        "first_block_score": round(float(np.mean(s.scores[:k])), 6),
        "final_block_score": round(float(np.mean(s.scores[-k:])), 6),
        "success_rate": s.metrics().success_rate,
        "learning_curve": block_means(s.scores),
        "policy_shift_kl": policy_shift(s),
        "final_states": {a.id: a.state.model_dump() for a in s.agents},
    }
    narrative = (
        f"{TEMPLATED}Over {n} episodes the mean score moved from {comp['first_block_score']:.2f} (first 20%) to "
        f"{comp['final_block_score']:.2f} (last 20%); chance is {comp['chance_score']:.2f}."
    )
    return comp, [s], narrative


def _run_shared_history(seed: int, config: ExperimentConfig) -> tuple[dict[str, Any], list[Session], str]:
    s = Session(config, PRESETS["shared_history"].condition, seed, preset="shared_history")
    s.step(config.episodes)
    _, receiver = s.roles(s.step_index)
    motif, target = most_successful_motif(receiver, s.config.music.motif_bank)
    phrase = _probe_phrase(s, motif)
    probe_seed = np.random.SeedSequence([seed, 7919])
    trained = probe_receiver(s, receiver, phrase, motif, target, np.random.default_rng(probe_seed))
    clone = reset_clone(s, receiver)
    reset = probe_receiver(s, clone, phrase, motif, target, np.random.default_rng(probe_seed))
    comp = {
        "receiver_id": receiver.id,
        "probe_motif": motif,
        "probe_phrase_id": phrase.id,
        "probe_target": target,
        "trained_probs": trained["probs"],
        "trained_top": trained["top"],
        "reset_probs": reset["probs"],
        "reset_top": reset["top"],
        "entropy_trained": trained["entropy"],
        "entropy_reset": reset["entropy"],
        "state_delta_trained": trained["state_delta"],
        "state_delta_reset": reset["state_delta"],
        "trained_choice": trained["chosen"],
        "reset_choice": reset["chosen"],
        "trained_score": trained["score"],
        "reset_score": reset["score"],
        "state_at_probe": receiver.state.model_dump(),
    }
    narrative = (
        f"{TEMPLATED}Probe motif {motif} (most often followed by success, pattern {target}). Trained receiver: "
        f"top={trained['top']}, entropy={trained['entropy']:.2f}; reset clone: top={reset['top']}, "
        f"entropy={reset['entropy']:.2f}. Same phrase, state and RNG draw."
    )
    return comp, [s], narrative


def _run_same_phrase(seed: int, config: ExperimentConfig) -> tuple[dict[str, Any], list[Session], str]:
    cond = PRESETS["same_phrase_different_history"].condition
    s1 = Session(config, cond, seed, preset="same_phrase_different_history",
                 script=EpisodeScript(SCRIPTED_MOTIF, SCRIPTED_TARGET, "motif 'hop' every episode; target always pattern 2"))
    s2 = Session(config, cond, seed, preset="same_phrase_different_history",
                 script=EpisodeScript(SCRIPTED_MOTIF, None, "motif 'hop' every episode; target drawn uniformly"))
    s1.step(config.episodes)
    s2.step(config.episodes)
    r1, r2 = s1.roles(s1.step_index)[1], s2.roles(s2.step_index)[1]
    phrase = _probe_phrase(s1, SCRIPTED_MOTIF)
    probe_state = r1.baseline  # identical starting state for both probes
    probe_seed = np.random.SeedSequence([seed, 104729])
    p1 = probe_receiver(s1, r1, phrase, SCRIPTED_MOTIF, SCRIPTED_TARGET, np.random.default_rng(probe_seed), probe_state)
    p2 = probe_receiver(s2, r2, phrase, SCRIPTED_MOTIF, SCRIPTED_TARGET, np.random.default_rng(probe_seed), probe_state)
    comp = {
        "receiver_id": r1.id,
        "probe_motif": SCRIPTED_MOTIF,
        "probe_target": SCRIPTED_TARGET,
        "probe_state": probe_state.model_dump(),
        "predictive_probs": p1["probs"],
        "uninformative_probs": p2["probs"],
        "predictive_top": p1["top"],
        "uninformative_top": p2["top"],
        "predictive_p_target": p1["probs"][SCRIPTED_TARGET],
        "uninformative_p_target": p2["probs"][SCRIPTED_TARGET],
        "entropy_predictive": p1["entropy"],
        "entropy_uninformative": p2["entropy"],
        "state_response_predictive": p1["state_delta"],
        "state_response_uninformative": p2["state_delta"],
        "end_state_predictive": r1.state.model_dump(),
        "end_state_uninformative": r2.state.model_dump(),
        "mean_score_predictive": round(float(np.mean(s1.scores)), 6),
        "mean_score_uninformative": round(float(np.mean(s2.scores)), 6),
    }
    narrative = (
        f"{TEMPLATED}Same phrase, same probe state and RNG. Predictive history: P(pattern 2)="
        f"{comp['predictive_p_target']:.2f}, entropy {p1['entropy']:.2f}. Uninformative history: P(pattern 2)="
        f"{comp['uninformative_p_target']:.2f}, entropy {p2['entropy']:.2f}. Constructed by the training schedule."
    )
    return comp, [s1, s2], narrative


_RUNNERS = {
    "first_encounter": _run_first_encounter,
    "shared_history": _run_shared_history,
    "same_phrase_different_history": _run_same_phrase,
}


def run_preset_sessions(
    name: str, seed: int = 7, config: ExperimentConfig | None = None, episodes: int | None = None
) -> tuple[PresetResult, list[Session]]:
    cfg, _, _ = preset_session_config(name, config, episodes)
    comp, sessions, narrative = _RUNNERS[name](seed, cfg)
    result = PresetResult(
        preset=name, seed=seed, sessions=[s.summary() for s in sessions], comparison=comp, narrative=narrative
    )
    return result, sessions


def run_preset(name: str, seed: int = 7, config: ExperimentConfig | None = None, episodes: int | None = None) -> PresetResult:
    return run_preset_sessions(name, seed, config, episodes)[0]

