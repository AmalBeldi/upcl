"""Decision operator Psi (Definition 10): argmax item selection."""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

Item = Hashable


def psi(scores: Dict[Item, float]) -> Item:
    """i* = Psi(S) = argmax_{i in I} S(u, i)."""
    if not scores:
        raise ValueError("Cannot select an item from an empty score dictionary.")
    return max(scores.items(), key=lambda kv: kv[1])[0]


def psi_ranked(scores: Dict[Item, float]) -> List[Tuple[Item, float]]:
    """Full ranked list, most useful for ranking-quality evaluation (Section 7)."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
