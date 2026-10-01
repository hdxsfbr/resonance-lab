"""Episodic memory: a FIFO store of past interactions, retrievable by feature similarity.

In the LOCAL policy, decisions are made from the learner's associations; memory is
retrieved into the policy context (used by the model-assisted policy, the inspector,
familiarity readouts and preset probes). `capacity = 0` means nothing is ever stored.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from resonance.music.features import cosine_similarity
from resonance.schemas import MemoryItem, RetrievedMemory


class EpisodicMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self._items: deque[MemoryItem] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def append(self, item: MemoryItem) -> None:
        if self.capacity == 0:
            return
        self._items.append(item)
        while len(self._items) > self.capacity:
            self._items.popleft()  # FIFO: oldest forgotten first

    def set_capacity(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        while len(self._items) > self.capacity:
            self._items.popleft()

    def retrieve(self, vector: Sequence[float], k: int) -> list[RetrievedMemory]:
        """Top-k items by cosine similarity (ties: more recent first)."""
        if k <= 0 or not self._items:
            return []
        scored = [
            (cosine_similarity(vector, it.feature_vector), idx, it)
            for idx, it in enumerate(self._items)
            if len(it.feature_vector) == len(vector)
        ]
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [RetrievedMemory(item=it, similarity=round(sim, 6)) for sim, _, it in scored[:k]]

    def familiarity(self, vector: Sequence[float]) -> float:
        """Max cosine similarity to any stored item (0 if empty)."""
        sims = [cosine_similarity(vector, it.feature_vector) for it in self._items if len(it.feature_vector) == len(vector)]
        return max(sims) if sims else 0.0

    def clear(self) -> None:
        self._items.clear()
