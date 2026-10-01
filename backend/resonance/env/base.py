"""Task protocol. The environment is replaceable: anything with these members works."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from resonance.schemas import TimingPattern


class Task(Protocol):
    name: str

    def patterns(self) -> list[TimingPattern]:
        """The public set of possible targets (which one is the target is sender-private)."""
        ...

    def draw_target(self, rng: np.random.Generator) -> int:
        """Draw a target id using the environment RNG stream."""
        ...

    def score(self, chosen_id: int, target_id: int) -> float:
        """Coordination score in [0, 1]."""
        ...
