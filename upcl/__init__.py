"""
UPCL: Unified Probabilistic Context Layer
------------------------------------------
Reference implementation accompanying the paper "Towards a Unified
Probabilistic Framework for Uncertainty-Aware Context Modeling in
Recommender Systems".

This package is a reproducibility / supplementary-material artifact and is
released as an anonymous companion repository for double-blind review.
"""

from .context import (
    BooleanFact,
    ContextualVariable,
    ProbabilisticContextInstance,
    ProbabilisticContextSpace,
    degenerate_context,
    phi,
)
from .decision import psi, psi_ranked
from .instantiations import upcl_model, upcl_post, upcl_pre
from .monte_carlo import (
    hoeffding_bound,
    monte_carlo_expected_utility,
    required_sample_size,
    sample_worlds,
)

__all__ = [
    "BooleanFact",
    "ContextualVariable",
    "ProbabilisticContextInstance",
    "ProbabilisticContextSpace",
    "degenerate_context",
    "phi",
    "psi",
    "psi_ranked",
    "upcl_pre",
    "upcl_model",
    "upcl_post",
    "hoeffding_bound",
    "monte_carlo_expected_utility",
    "required_sample_size",
    "sample_worlds",
]

__version__ = "0.1.0"
