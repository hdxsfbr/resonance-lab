"""ENGINEERED state dynamics (not a model of feelings).

Four bounded dims (see `StateVector` for operational definitions). One update:

  pe = score - expected_value                       (computed BEFORE the update)

  drive_k = sensitivity * ( music_drive_k
                          + pe_gain_k      * pe_term_k
                          + outcome_gain_k * outcome_term_k
                          + partner_similarity_gain * similarity   [affiliation only] )

    pe_term_k      = |pe| - uncertainty     for k = uncertainty   (running |prediction error|)
                   = pe                     otherwise
    outcome_term_k = pe                     for k = expected_value (running mean of score)
                   = score - 0.5            otherwise
    music_drive_k  = acoustic_activation_gain * ((density_norm - 0.5) + (velocity_norm - 0.5))
                     for k = activation when acoustic_activation_enabled and the agent heard a
                     phrase (HAND-AUTHORED acoustic influence); 0 for every other dim / case.
                     density_norm = note_density / 6, velocity_norm = mean_velocity / 127, clipped [0, 1].

  delta_k = clip((1 - inertia_k) * drive_k + decay_k * (baseline_k - s_k), -max_step_k, +max_step_k)
  s_k'    = clip(s_k + delta_k, bounds_k)     bounds: [0,1] except affiliation [-1,1]

Gains missing from a gain dict are 0. When `score is None` (e.g. a probe with no
outcome) the pe and outcome terms are 0. If `frozen`, the state is returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from resonance.music.features import NORMALISATION
from resonance.schemas import STATE_DIMS, StateConfig, StateUpdateInputs, StateVector, SymbolicFeatures

BOUNDS: dict[str, tuple[float, float]] = {
    "activation": (0.0, 1.0),
    "expected_value": (0.0, 1.0),
    "uncertainty": (0.0, 1.0),
    "affiliation": (-1.0, 1.0),
}


@dataclass
class StateDrive:
    """Raw inputs for one update."""

    score: float | None = None
    heard: SymbolicFeatures | None = None  # phrase this agent HEARD (receiver in music channel)
    partner_similarity: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class StateStepDetail:
    drive: dict[str, float]
    delta: dict[str, float]


def acoustic_drive(heard: SymbolicFeatures, gain: float) -> float:
    """HAND-AUTHORED: gain * ((density_norm - 0.5) + (velocity_norm - 0.5))."""
    d_off, d_scale = NORMALISATION["note_density"]
    v_off, v_scale = NORMALISATION["mean_velocity"]
    density_norm = float(np.clip((heard.note_density - d_off) / d_scale, 0.0, 1.0))
    velocity_norm = float(np.clip((heard.mean_velocity - v_off) / v_scale, 0.0, 1.0))
    return gain * ((density_norm - 0.5) + (velocity_norm - 0.5))


class StateEngine:
    @staticmethod
    def update(
        state: StateVector,
        inputs: StateDrive,
        cfg: StateConfig,
        sensitivity: float = 1.0,
        frozen: bool = False,
        baseline: StateVector | None = None,
    ) -> tuple[StateVector, StateUpdateInputs]:
        new_state, record, _ = StateEngine.update_detailed(state, inputs, cfg, sensitivity, frozen, baseline)
        return new_state, record

    @staticmethod
    def update_detailed(
        state: StateVector,
        inputs: StateDrive,
        cfg: StateConfig,
        sensitivity: float = 1.0,
        frozen: bool = False,
        baseline: StateVector | None = None,
    ) -> tuple[StateVector, StateUpdateInputs, StateStepDetail]:
        base = baseline or cfg.baseline
        s = state.model_dump()
        b = base.model_dump()
        pe = None if inputs.score is None else inputs.score - s["expected_value"]
        music = StateEngine._music_drive(inputs, cfg)
        notes = list(inputs.notes)
        if music["activation"] != 0.0:
            notes.append("hand-authored acoustic influence on activation")
        elif inputs.heard is not None and not cfg.acoustic_activation_enabled:
            notes.append("hand-authored acoustic influence disabled")
        record = StateUpdateInputs(
            music_drive=music,
            prediction_error=None if pe is None else round(pe, 6),
            outcome=inputs.score,
            partner_similarity=inputs.partner_similarity,
            decay_applied=not frozen,
            frozen=frozen,
            notes=notes,
        )
        if frozen:
            zeros = dict.fromkeys(STATE_DIMS, 0.0)
            return state, record, StateStepDetail(drive=zeros, delta=zeros)
        drive: dict[str, float] = {}
        delta: dict[str, float] = {}
        new: dict[str, float] = {}
        for k in STATE_DIMS:
            drive[k] = sensitivity * StateEngine._raw_drive(k, s, pe, inputs, music[k], cfg)
            step = (1.0 - cfg.inertia.get(k, 0.0)) * drive[k] + cfg.decay.get(k, 0.0) * (b[k] - s[k])
            cap = cfg.max_step.get(k, 1.0)
            delta[k] = float(np.clip(step, -cap, cap))
            lo, hi = BOUNDS[k]
            new[k] = float(np.clip(s[k] + delta[k], lo, hi))
        rounded = {k: round(v, 9) for k, v in new.items()}
        detail = StateStepDetail(drive={k: round(v, 6) for k, v in drive.items()}, delta={k: round(v, 6) for k, v in delta.items()})
        return StateVector(**rounded), record, detail

    @staticmethod
    def _music_drive(inputs: StateDrive, cfg: StateConfig) -> dict[str, float]:
        out = dict.fromkeys(STATE_DIMS, 0.0)
        if inputs.heard is not None and cfg.acoustic_activation_enabled:
            out["activation"] = round(acoustic_drive(inputs.heard, cfg.acoustic_activation_gain), 6)
        return out

    @staticmethod
    def _raw_drive(
        k: str, s: dict[str, float], pe: float | None, inputs: StateDrive, music_k: float, cfg: StateConfig
    ) -> float:
        total = music_k
        if pe is not None and inputs.score is not None:
            pe_term = abs(pe) - s["uncertainty"] if k == "uncertainty" else pe
            outcome_term = pe if k == "expected_value" else inputs.score - 0.5
            total += cfg.prediction_error_gain.get(k, 0.0) * pe_term
            total += cfg.outcome_gain.get(k, 0.0) * outcome_term
        if k == "affiliation" and inputs.partner_similarity is not None:
            total += cfg.partner_similarity_gain * inputs.partner_similarity
        return total
