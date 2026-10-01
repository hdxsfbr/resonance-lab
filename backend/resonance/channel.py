"""The channel between sender and receiver: perturbation and observation building.

The Observation is EXACTLY what the receiver gets. It is built only from the sent
phrase (or, in the symbol channel, the sender's motif index) and never from the target.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from resonance.agents.learner import reference_vector
from resonance.music.features import cosine_similarity, symbolic_features
from resonance.music.motifs import base_phrases, motif_index_from_id
from resonance.music.transforms import apply_transform
from resonance.schemas import Observation, Perturbation, Phrase


def perturb(phrase: Phrase, kinds: list[Perturbation], rng: np.random.Generator) -> tuple[Phrase, list[str]]:
    """Apply each non-'none' transform in order (randomness from the rng_perturb stream)."""
    applied: list[str] = []
    for kind in kinds:
        if kind != "none":
            phrase = apply_transform(phrase, kind, rng)
            applied.append(kind)
    return phrase, applied


def nearest_motif(phrase: Phrase, bank: str, n_motifs: int, weights: Mapping[str, float] | None = None) -> int:
    """Index of the bank motif whose (centred) feature vector is most similar to `phrase`.

    Used only to give a non-bank phrase (human / replayed) a symbol in the symbol channel.
    """
    idx = motif_index_from_id(phrase.motif_id, bank)
    if idx is not None and idx < n_motifs:
        return idx
    ref = reference_vector(bank, n_motifs, weights)
    v = np.asarray(symbolic_features(phrase, weights).vector) - ref
    sims = [cosine_similarity(v, np.asarray(symbolic_features(p, weights).vector) - ref) for p in base_phrases(bank, n=n_motifs)]
    return int(np.argmax(sims))


def music_observation(heard: Phrase, sender_id: str, step: int, weights: Mapping[str, float] | None) -> Observation:
    return Observation(
        channel="music", phrase=heard, features=symbolic_features(heard, weights), sender_id=sender_id, step=step
    )


def symbol_observation(symbol_id: int, sender_id: str, step: int) -> Observation:
    return Observation(channel="symbol", symbol_id=symbol_id, sender_id=sender_id, step=step)


def partner_similarity(a: list[float] | None, b: list[float] | None, reference: np.ndarray) -> float | None:
    """HAND-AUTHORED input to affiliation: max(0, cos(phi_a - phi_ref, phi_b - phi_ref)) in [0, 1]."""
    if a is None or b is None:
        return None
    sim = cosine_similarity(np.asarray(a) - reference, np.asarray(b) - reference)
    return round(max(0.0, sim), 6)
