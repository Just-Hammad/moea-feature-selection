"""Non-parametric comparison pipeline.

API-compatible with the shared contract in `12-build-guides.md` section 2, so that
when `metaheuristics-bench` lands this module can be replaced by::

    from mhbench.stats.compare import friedman_ranks, posthoc_holm, cliffs_delta

Nothing else in this package needs to change.

The two-level distinction the contract insists on:

* ``friedman_ranks`` / ``posthoc_holm`` operate on an ``algorithms x problems``
  matrix of **per-problem aggregates** (one number per algorithm per problem).
* ``cliffs_delta`` operates on **raw runs within a single problem**.

Mixing those up is the most common statistical error in this literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

try:  # scikit-posthocs is optional at import time so the module can be unit-tested bare
    import scikit_posthocs as sp
except ImportError:  # pragma: no cover
    sp = None


# Romano et al. cut-points. Reported alongside the number so a reader never has
# to look the scale up.
_CLIFF_BOUNDS = ((0.147, "negligible"), (0.330, "small"), (0.474, "medium"))


def friedman_ranks(matrix: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """Friedman test across algorithms, with problems as blocks.

    Parameters
    ----------
    matrix
        Rows = problems (blocks), columns = algorithms. Values are the
        per-problem aggregate (median across runs), **lower is better**.

    Returns
    -------
    (statistic, p_value, mean_ranks)
        ``mean_ranks`` is ascending, so rank 1 is the best algorithm.
    """
    if matrix.shape[1] < 3:
        raise ValueError(
            f"Friedman needs >=3 groups, got {matrix.shape[1]}. "
            "For two algorithms use a paired Wilcoxon signed-rank test directly."
        )
    if matrix.shape[0] < 2:
        raise ValueError("Friedman needs >=2 blocks (problems).")

    stat, p = friedmanchisquare(*[matrix[c].to_numpy() for c in matrix.columns])
    ranks = matrix.rank(axis=1, ascending=True).mean(axis=0)
    return float(stat), float(p), ranks.sort_values()


def posthoc_holm(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank with Holm step-down correction.

    Paired across problems, so this answers "is A better than B *across the
    suite*" -- not "on this problem". Only meaningful when ``friedman_ranks``
    was significant; running it otherwise is the multiplicity error the
    correction exists to prevent.
    """
    if sp is None:  # pragma: no cover
        raise ImportError("scikit-posthocs is required for posthoc_holm()")

    # posthoc_wilcoxon takes a list of groups; our columns are the groups.
    groups = [matrix[c].to_numpy().tolist() for c in matrix.columns]
    out = sp.posthoc_wilcoxon(groups, p_adjust="holm")
    out.index = list(matrix.columns)
    out.columns = list(matrix.columns)
    return out


def cliffs_delta(a, b) -> float:
    """Cliff's delta: stochastic dominance of ``a`` over ``b``.

    Run this on **raw runs within one problem**, never on per-problem
    aggregates -- on aggregates it merely restates the Friedman ranks.

    Returns a value in [-1, 1]. Positive means ``a`` tends to exceed ``b``.
    Because our objectives are errors (lower is better), a *negative* delta
    means ``a`` is the better algorithm.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("cliffs_delta() needs two non-empty samples")

    diff = a[:, None] - b[None, :]
    greater = int((diff > 0).sum())
    less = int((diff < 0).sum())
    return (greater - less) / (a.size * b.size)


def cliffs_delta_label(delta: float) -> str:
    """Magnitude label for a Cliff's delta, using the Romano et al. cut-points."""
    magnitude = abs(delta)
    for bound, label in _CLIFF_BOUNDS:
        if magnitude < bound:
            return label
    return "large"


def pairwise_cliffs(
    runs: dict[str, np.ndarray], labels: bool = True
) -> pd.DataFrame:
    """Cliff's delta for every ordered pair of algorithms, on raw runs.

    ``runs`` maps algorithm name -> 1-D array of per-run values for a single
    problem.
    """
    names = list(runs)
    frame = pd.DataFrame(index=names, columns=names, dtype=object if labels else float)
    for i in names:
        for j in names:
            if i == j:
                frame.loc[i, j] = "-" if labels else 0.0
                continue
            d = cliffs_delta(runs[i], runs[j])
            frame.loc[i, j] = f"{d:+.3f} ({cliffs_delta_label(d)})" if labels else d
    return frame
