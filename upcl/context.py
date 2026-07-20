"""
Probabilistic context representation for the Unified Probabilistic
Context Layer (UPCL).

Implements, directly from the paper:
  - Definition 1: Contextual Variable
  - Definition 2: Probabilistic Context Instance
  - Definition 3: Possible Contextual World
  - Definition 4: Contextual Probability Measure
  - Definition 5: Probabilistic Context Space
  - Definitions 6-7 + transformation operator Phi (generator G, assignment A)
  - TID / BID / Hybrid probability assignment semantics (Section 4.5)

No external dependencies beyond the standard library and itertools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Hashable, List, Sequence, Tuple

Value = Hashable
World = Tuple[Tuple[str, Value], ...]  # ((var_name, value), ...), order-stable


@dataclass(frozen=True)
class ContextualVariable:
    """Definition 1. A discrete random variable C_k over a finite domain."""

    name: str
    distribution: Dict[Value, float]

    def __post_init__(self) -> None:
        total = sum(self.distribution.values())
        if any(p < 0 for p in self.distribution.values()):
            raise ValueError(f"Negative probability in variable '{self.name}'.")
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Distribution of '{self.name}' must sum to 1 (got {total})."
            )

    @property
    def domain(self) -> Tuple[Value, ...]:
        return tuple(self.distribution.keys())


@dataclass(frozen=True)
class BooleanFact:
    """A single independent TID fact tau_j with marginal probability p_j."""

    name: str
    probability: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"Fact '{self.name}' probability must be in [0,1].")


@dataclass
class ProbabilisticContextInstance:
    """Definition 2. kappa^t_u = {P(C_1), ..., P(C_K)} plus optional TID facts."""

    categorical_variables: List[ContextualVariable] = field(default_factory=list)
    boolean_facts: List[BooleanFact] = field(default_factory=list)


@dataclass
class ProbabilisticContextSpace:
    """Definition 5. PCS^t_u = (C, Omega^t_u, P^t_u)."""

    variables: List[str]
    worlds: List[World]
    probabilities: List[float]

    def __post_init__(self) -> None:
        total = sum(self.probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Contextual probability measure does not sum to 1 (got {total})."
            )
        if any(p < 0 for p in self.probabilities):
            raise ValueError("Contextual probability measure has negative mass.")

    def as_dict(self) -> Dict[World, float]:
        return dict(zip(self.worlds, self.probabilities))

    def support(self, eps: float = 0.0) -> "ProbabilisticContextSpace":
        """Restrict to worlds with strictly positive probability (Section 4.4)."""
        kept = [(w, p) for w, p in zip(self.worlds, self.probabilities) if p > eps]
        worlds, probs = zip(*kept) if kept else ([], [])
        # Renormalize defensively against floating point drift.
        s = sum(probs)
        probs = tuple(p / s for p in probs) if s > 0 else probs
        return ProbabilisticContextSpace(self.variables, list(worlds), list(probs))


def _bid_worlds(categorical_variables: Sequence[ContextualVariable]) -> Tuple[List[World], List[float]]:
    """Definition 6/BID (Section 4.5.0.0.2): cartesian product across independent
    blocks, exactly one value drawn per block. Corresponds to G for a purely
    categorical (BID) context instance, followed by A under block independence.
    """
    if not categorical_variables:
        return [tuple()], [1.0]

    names = [v.name for v in categorical_variables]
    domains = [list(v.distribution.items()) for v in categorical_variables]

    worlds: List[World] = []
    probs: List[float] = []
    for combo in product(*domains):
        values = tuple(val for val, _ in combo)
        p = 1.0
        for _, pj in combo:
            p *= pj
        worlds.append(tuple(zip(names, values)))
        probs.append(p)
    return worlds, probs


def _tid_worlds(boolean_facts: Sequence[BooleanFact]) -> Tuple[List[World], List[float]]:
    """Section 4.5.0.0.1: TID semantics over independent Boolean facts.
    Every subset of facts is a possible world; P(omega) = prod p_j * prod (1-p_j).
    """
    if not boolean_facts:
        return [tuple()], [1.0]

    names = [f.name for f in boolean_facts]
    probs_j = [f.probability for f in boolean_facts]

    worlds: List[World] = []
    probs: List[float] = []
    for combo in product([False, True], repeat=len(boolean_facts)):
        p = 1.0
        for present, pj in zip(combo, probs_j):
            p *= pj if present else (1.0 - pj)
        worlds.append(tuple(zip(names, combo)))
        probs.append(p)
    return worlds, probs


def phi(
    context_instance: ProbabilisticContextInstance,
    prune_zero_probability: bool = True,
) -> ProbabilisticContextSpace:
    """The probabilistic context transformation operator Phi (Definition 6-7,
    Section 4.4): kappa^t_u -> (Omega^t_u, P^t_u).

    Supports the hybrid TID-BID representation of Section 4.5.0.0.3: the
    world space is the Cartesian product of the BID block worlds and the
    TID subset worlds, with independence assumed between the two components
    (P(omega) = P_TID(omega_TID) * P_BID(omega_BID)).
    """
    bid_worlds, bid_probs = _bid_worlds(context_instance.categorical_variables)
    tid_worlds, tid_probs = _tid_worlds(context_instance.boolean_facts)

    worlds: List[World] = []
    probs: List[float] = []
    for (bw, bp) in zip(bid_worlds, bid_probs):
        for (tw, tp) in zip(tid_worlds, tid_probs):
            worlds.append(bw + tw)
            probs.append(bp * tp)

    variables = [v.name for v in context_instance.categorical_variables] + [
        f.name for f in context_instance.boolean_facts
    ]

    space = ProbabilisticContextSpace(variables=variables, worlds=worlds, probabilities=probs)
    if prune_zero_probability:
        space = space.support()
    return space


def degenerate_context(assignment: Dict[str, Value]) -> ProbabilisticContextSpace:
    """Builds a Probabilistic Context Space with a single world of probability 1.

    Used to instantiate Theorem 2 (Reduction to Deterministic Context): when
    P(omega*) = 1, UPCL-Pre must reduce exactly to deterministic CARS.
    """
    world = tuple(assignment.items())
    return ProbabilisticContextSpace(
        variables=list(assignment.keys()), worlds=[world], probabilities=[1.0]
    )
