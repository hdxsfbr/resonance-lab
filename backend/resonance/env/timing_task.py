"""Cooperative timing task.

The sender privately knows a target 4-beat onset pattern; the receiver must pick
that pattern from the public set after hearing one phrase.

score(chosen, target) = 1.0                                   if chosen == target
                      = 0.5 * Jaccard(grid(chosen), grid(target))   if partial_credit
                      = 0.0                                   otherwise
grid(pattern) = set of onsets rounded to the 0.25-beat grid. Partial credit is
therefore capped at 0.5 (only reachable by identical grids), so an exact match is
always clearly distinguishable from a near miss.
"""

from __future__ import annotations

import numpy as np

from resonance.schemas import TimingPattern

GRID = 0.25

PATTERN_LIBRARY: list[tuple[str, list[float]]] = [
    ("steady quarters", [0.0, 1.0, 2.0, 3.0]),
    ("offbeat", [0.5, 1.5, 2.5, 3.5]),
    ("gallop", [0.0, 0.75, 1.0, 2.0, 2.75, 3.0]),
    ("sparse", [0.0, 2.5]),
    ("triplet-ish", [0.0, 1.33, 2.67]),
    ("syncopated", [0.0, 1.5, 2.0, 3.5]),
    ("dense eighths", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
    ("clave", [0.0, 0.75, 1.5, 2.5, 3.0]),
]


def _grid(onsets: list[float]) -> set[int]:
    return {int(round(o / GRID)) for o in onsets}


def jaccard(a: list[float], b: list[float]) -> float:
    ga, gb = _grid(a), _grid(b)
    union = ga | gb
    return len(ga & gb) / len(union) if union else 1.0


class TimingTask:
    name = "timing"

    def __init__(self, n_patterns: int = 4, partial_credit: bool = True) -> None:
        if not 2 <= n_patterns <= len(PATTERN_LIBRARY):
            raise ValueError(f"n_patterns must be in [2, {len(PATTERN_LIBRARY)}]")
        self.n_patterns = n_patterns
        self.partial_credit = partial_credit
        self._patterns = [
            TimingPattern(id=i, name=name, onsets=onsets) for i, (name, onsets) in enumerate(PATTERN_LIBRARY[:n_patterns])
        ]

    def patterns(self) -> list[TimingPattern]:
        return list(self._patterns)

    def draw_target(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n_patterns))

    def score(self, chosen_id: int, target_id: int) -> float:
        if chosen_id == target_id:
            return 1.0
        if not self.partial_credit or not 0 <= chosen_id < self.n_patterns:
            return 0.0
        return 0.5 * jaccard(self._patterns[chosen_id].onsets, self._patterns[target_id].onsets)

    def chance_score(self) -> float:
        """Expected score of a uniformly random receiver with uniform targets."""
        n = self.n_patterns
        return float(np.mean([[self.score(c, t) for c in range(n)] for t in range(n)]))
