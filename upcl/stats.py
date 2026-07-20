"""
Paired statistical significance testing (Section 7.3).

Wraps standard paired tests used in recommender-systems evaluation to
compare UPCL against deterministic/baseline methods across users or folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class SignificanceResult:
    test_name: str
    statistic: float
    p_value: float
    mean_difference: float
    n: int
    significant_at_05: bool

    def __str__(self) -> str:
        sig = "significant" if self.significant_at_05 else "not significant"
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p={self.p_value:.4g} ({sig} at alpha=0.05), "
            f"mean_diff={self.mean_difference:.4f}, n={self.n}"
        )


def paired_ttest(method_a: Sequence[float], method_b: Sequence[float]) -> SignificanceResult:
    """Paired (two-sided) t-test: is method_a's per-user/fold metric
    significantly different from method_b's?
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("method_a and method_b must have matched length.")
    stat, p = scipy_stats.ttest_rel(a, b)
    return SignificanceResult(
        test_name="Paired t-test",
        statistic=float(stat),
        p_value=float(p),
        mean_difference=float(np.mean(a - b)),
        n=len(a),
        significant_at_05=bool(p < 0.05),
    )


def wilcoxon_signed_rank(method_a: Sequence[float], method_b: Sequence[float]) -> SignificanceResult:
    """Non-parametric alternative to the paired t-test, recommended when
    per-user metric differences are not approximately normal.
    """
    a = np.asarray(method_a, dtype=float)
    b = np.asarray(method_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("method_a and method_b must have matched length.")
    diffs = a - b
    if np.allclose(diffs, 0.0):
        stat, p = 0.0, 1.0
    else:
        stat, p = scipy_stats.wilcoxon(a, b)
    return SignificanceResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(stat),
        p_value=float(p),
        mean_difference=float(np.mean(diffs)),
        n=len(a),
        significant_at_05=bool(p < 0.05),
    )
