"""LEARNED associations.

ReceiverLearner — a linear contextual bandit. The receiver never sees the target;
it only sees its own chosen pattern a and the resulting score r.

    q(x) = W x + b                      (one row per timing pattern; q_a estimates E[score | x, a])
    after choosing a and receiving r:
        err  = r - q_a(x)
        w_a <- w_a + lr_eff * err * x - l2 * w_a
        b_a <- b_a + lr_eff * err
    Only the chosen row changes. The policy uses q as its per-pattern scores.

Receiver input x (built by `encode_music` / `encode_symbol`):
    music channel : x = phi(phrase) - phi_ref, where phi_ref is the mean feature vector of
                    the motif bank at characteristic tempo (a fixed reference, NOT learned).
                    Centring on a typical phrase keeps the shared "average phrase" component
                    from dominating dot products, so updates for one motif interfere less with
                    others, while related phrases (e.g. transpositions) still share weights.
    symbol channel: x = one_hot(symbol_id, n_motifs). Same update rule (this is the tabular case).

SenderLearner — tabular bandit over (target, motif):
    Q[t][m] <- Q[t][m] + lr_eff * (r - Q[t][m]);  counts[t][m] += 1;  Q initialised to optimistic_init.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

from resonance.music.features import symbolic_features
from resonance.music.motifs import base_phrases
from resonance.schemas import LearningConfig, Observation


@lru_cache(maxsize=32)
def _reference_cached(bank: str, n_motifs: int, weights_key: tuple[tuple[str, float], ...]) -> tuple[float, ...]:
    weights = dict(weights_key)
    vecs = [symbolic_features(p, weights).vector for p in base_phrases(bank, n=n_motifs)]
    return tuple(float(v) for v in np.mean(np.asarray(vecs), axis=0))


def reference_vector(bank: str = "default", n_motifs: int = 8, weights: Mapping[str, float] | None = None) -> np.ndarray:
    """phi_ref: mean normalised feature vector over the motif bank (fixed constant)."""
    key = tuple(sorted((weights or {}).items()))
    return np.asarray(_reference_cached(bank, n_motifs, key))


def encode_music(vector: Sequence[float], reference: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=float) - reference


def encode_symbol(symbol_id: int, n_motifs: int) -> np.ndarray:
    x = np.zeros(n_motifs)
    if 0 <= symbol_id < n_motifs:
        x[symbol_id] = 1.0
    return x


def encode_observation(obs: Observation, n_motifs: int, reference: np.ndarray) -> np.ndarray:
    if obs.channel == "symbol":
        return encode_symbol(int(obs.symbol_id if obs.symbol_id is not None else -1), n_motifs)
    assert obs.features is not None, "music observation must carry features"
    return encode_music(obs.features.vector, reference)


class ReceiverLearner:
    def __init__(self, input_dim: int, n_patterns: int, cfg: LearningConfig, init_seed: int = 0) -> None:
        self.input_dim = input_dim
        self.n_patterns = n_patterns
        self.l2 = cfg.receiver_l2
        self.init_scale = cfg.receiver_init_scale
        self.init_seed = init_seed
        self.reset()

    def reset(self) -> None:
        if self.init_scale > 0:
            rng = np.random.default_rng(self.init_seed)
            self.W = rng.normal(0.0, self.init_scale, (self.n_patterns, self.input_dim))
        else:
            self.W = np.zeros((self.n_patterns, self.input_dim))
        self.b = np.zeros(self.n_patterns)
        self.updates = 0

    def scores(self, x: np.ndarray) -> np.ndarray:
        """q = W x + b (expected score per pattern)."""
        return self.W @ x + self.b

    def update(self, x: np.ndarray, action: int, score: float, lr: float) -> dict[str, object]:
        q_before = float(self.scores(x)[action])
        err = score - q_before
        w_before = self.W[action].copy()
        b_before = float(self.b[action])
        self.W[action] = self.W[action] + lr * err * x - self.l2 * self.W[action]
        self.b[action] = self.b[action] + lr * err
        self.updates += 1
        return {
            "row": action,
            "q_before": round(q_before, 6),
            "q_after": round(float(self.scores(x)[action]), 6),
            "error": round(err, 6),
            "weights_before": [round(float(v), 6) for v in w_before],
            "weights_after": [round(float(v), 6) for v in self.W[action]],
            "bias_before": round(b_before, 6),
            "bias_after": round(float(self.b[action]), 6),
        }

    def set_row(self, row: int, weights: Sequence[float], bias: float) -> None:
        """Replay helper: apply recorded values (no recomputation)."""
        self.W[row] = np.asarray(weights, dtype=float)
        self.b[row] = float(bias)
        self.updates += 1


class SenderLearner:
    def __init__(self, n_targets: int, n_motifs: int, cfg: LearningConfig) -> None:
        self.n_targets = n_targets
        self.n_motifs = n_motifs
        self.optimistic_init = cfg.optimistic_init
        self.reset()

    def reset(self) -> None:
        self.Q = np.full((self.n_targets, self.n_motifs), float(self.optimistic_init))
        self.counts = np.zeros((self.n_targets, self.n_motifs), dtype=int)
        self.updates = 0

    def values(self, target: int) -> np.ndarray:
        return self.Q[target].copy()

    def update(self, target: int, motif: int, score: float, lr: float) -> dict[str, object]:
        q_before = float(self.Q[target, motif])
        self.Q[target, motif] = q_before + lr * (score - q_before)
        self.counts[target, motif] += 1
        self.updates += 1
        return {
            "target": target,
            "motif": motif,
            "q_before": round(q_before, 6),
            "q_after": round(float(self.Q[target, motif]), 6),
            "count_after": int(self.counts[target, motif]),
        }

    def set_value(self, target: int, motif: int, q: float, count: int) -> None:
        """Replay helper: apply recorded values (no recomputation)."""
        self.Q[target, motif] = q
        self.counts[target, motif] = count
        self.updates += 1
