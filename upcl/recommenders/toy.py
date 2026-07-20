"""
Minimal synthetic recommenders used purely to exercise UPCL end to end
without requiring any external dataset. These are intentionally simple
(deterministic lookup tables / linear scoring functions) so that the
numbers in experiments/example_alice.py match the paper's worked example
exactly, and so that unit tests are fully reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Hashable, Tuple

from ..context import World

Item = Hashable
User = Hashable


class LookupWorldwiseRecommender:
    """A world-wise recommender f(u, i, omega) backed by an explicit table.

    Used to reproduce the paper's running example verbatim (Section 5.1's
    running example: r(A, w1)=4.8, r(A, w2)=3.2, r(B, w1)=3.9, r(B, w2)=4.5).
    """

    def __init__(self, table: Dict[Tuple[User, Item, World], float]):
        self._table = table

    def __call__(self, user: User, item: Item, world: World) -> float:
        try:
            return self._table[(user, item, world)]
        except KeyError as exc:
            raise KeyError(
                f"No score registered for (user={user!r}, item={item!r}, world={world!r})."
            ) from exc


class HashedContextualRecommender:
    """A deterministic, dependency-free stand-in for a trained recommender.

    Produces a reproducible pseudo-score in [0, 5] for any (user, item,
    world) triple by hashing their string representation. This lets the
    synthetic benchmark (experiments/synthetic_benchmark.py) exercise UPCL
    at arbitrary scale without training an actual model or shipping
    dataset files, while remaining fully deterministic across runs (same
    seed -> identical numbers), which is what the reproducibility
    material is meant to guarantee.
    """

    def __init__(self, seed: int = 0, scale: float = 5.0):
        self.seed = seed
        self.scale = scale

    def _hash_unit(self, *parts: object) -> float:
        key = f"{self.seed}|" + "|".join(str(p) for p in parts)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # Use the first 8 hex chars as a uniform draw in [0, 1).
        return int(digest[:8], 16) / 0xFFFFFFFF

    def worldwise(self, user: User, item: Item, world: World) -> float:
        return self.scale * self._hash_unit("world", user, item, world)

    def base(self, user: User, item: Item) -> float:
        return self.scale * self._hash_unit("base", user, item)

    def post_adjustment(self, r0: float, user: User, item: Item, world: World) -> float:
        # Additive contextual perturbation bounded to keep scores in range.
        delta = (self._hash_unit("adjust", user, item, world) - 0.5) * 1.5
        return float(min(self.scale, max(0.0, r0 + delta)))
