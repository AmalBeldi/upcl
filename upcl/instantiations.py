"""
The three UPCL integration strategies (Section 5), unified by the same
Probabilistic Context Space (Proposition 11) but differing in where the
recommendation function interacts with contextual uncertainty:

  - UPCL-Pre  : world-wise evaluation, aggregated after prediction (5.1)
  - UPCL-Model: probabilistic context encoded and fed into the recommender (5.2)
  - UPCL-Post : base recommender adjusted per-world, then aggregated (5.3)

All three expose the same call signature: given a Probabilistic Context
Space and a candidate item set, they return (i*, {item: score}).
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Iterable, List, Tuple

from .aggregation import AGGREGATION_OPERATORS
from .context import ProbabilisticContextSpace, World
from .decision import psi

Item = Hashable
User = Hashable

# f(u, i, world) -> score, where `world` is the World tuple from Definition 3.
WorldwiseRecommender = Callable[[User, Item, World], float]
# f0(u, i) -> base score, independent of context.
BaseRecommender = Callable[[User, Item], float]
# h(r0, u, i, world) -> contextually adjusted score.
PostAdjustment = Callable[[float, User, Item, World], float]
# Encoder(worlds, probs) -> fixed-size contextual representation z.
ContextEncoder = Callable[[List[World], List[float]], object]
# f_phi(u, i, z) -> score, given the encoded probabilistic context.
ModelRecommender = Callable[[User, Item, object], float]


def _resolve_gamma(policy: str, **policy_kwargs):
    if policy not in AGGREGATION_OPERATORS:
        raise ValueError(f"Unknown aggregation policy '{policy}'. "
                          f"Choose from {list(AGGREGATION_OPERATORS)}.")
    gamma = AGGREGATION_OPERATORS[policy]

    def call(scores, probs):
        if policy == "maxmin":
            return gamma(scores)
        if policy == "hurwicz":
            return gamma(scores, probs, **policy_kwargs)
        if policy == "cvar":
            return gamma(scores, probs, **policy_kwargs)
        if policy == "dro":
            return gamma(scores, probs, **policy_kwargs)
        return gamma(scores, probs)

    return call


def upcl_pre(
    user: User,
    items: Iterable[Item],
    space: ProbabilisticContextSpace,
    recommender: WorldwiseRecommender,
    policy: str = "eu",
    **policy_kwargs,
) -> Tuple[Item, Dict[Item, float]]:
    """Section 5.1 (Eq. for S^Pre_theta):
        kappa -> Phi -> {r(u,i,omega)}_omega -> Gamma_theta -> S^Pre_theta -> Psi -> i*
    """
    gamma = _resolve_gamma(policy, **policy_kwargs)
    scores: Dict[Item, float] = {}
    for item in items:
        world_scores = [recommender(user, item, w) for w in space.worlds]
        scores[item] = gamma(world_scores, space.probabilities)
    return psi(scores), scores


def upcl_model(
    user: User,
    items: Iterable[Item],
    space: ProbabilisticContextSpace,
    encoder: ContextEncoder,
    recommender: ModelRecommender,
) -> Tuple[Item, Dict[Item, float]]:
    """Section 5.2:
        kappa -> Phi -> E_phi -> z^t_u -> f_phi(u,i,z) -> S^Model_phi -> Psi -> i*

    Unlike UPCL-Pre/Post, uncertainty enters the recommender itself through
    the encoded representation z; there is no separate Gamma_theta step
    here because aggregation is (by design) absorbed into f_phi's learned
    parameters (Proposition 7). `encoder` and `recommender` are injected so
    that any concrete architecture (MF, GNN, sequential, LLM-based, ...)
    can be plugged in without changing this function.
    """
    z = encoder(space.worlds, space.probabilities)
    scores: Dict[Item, float] = {item: recommender(user, item, z) for item in items}
    return psi(scores), scores


def upcl_post(
    user: User,
    items: Iterable[Item],
    space: ProbabilisticContextSpace,
    base_recommender: BaseRecommender,
    adjustment: PostAdjustment,
    policy: str = "eu",
    **policy_kwargs,
) -> Tuple[Item, Dict[Item, float]]:
    """Section 5.3:
        r0(u,i) -> h_eta(r0,u,i,omega) -> {r^Post(u,i,omega)}_omega
                 -> Gamma_theta -> S^Post_theta -> Psi -> i*

    `base_recommender` is treated as a black box, unaware of contextual
    uncertainty (Proposition 8); only `adjustment` and `Gamma_theta` see it.
    """
    gamma = _resolve_gamma(policy, **policy_kwargs)
    scores: Dict[Item, float] = {}
    for item in items:
        r0 = base_recommender(user, item)
        world_scores = [adjustment(r0, user, item, w) for w in space.worlds]
        scores[item] = gamma(world_scores, space.probabilities)
    return psi(scores), scores
