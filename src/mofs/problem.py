"""pymoo problem wrapper around `InnerObjective`.

Kept deliberately thin: the objective lives in `objective.py` because the
filter, embedded, knockoff and random baselines all need it too, and none of
them are pymoo problems.
"""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from .objective import InnerObjective


class FeatureSelectionProblem(Problem):
    """Binary bi-objective feature selection.

    A cardinality-limiting repair is applied *before* evaluation rather than as
    a penalty afterwards. With p > 10^4 a uniformly random binary initial
    population selects ~p/2 features and every individual looks identical; the
    repair is what makes the search well-posed at high p.
    """

    def __init__(self, objective: InnerObjective, max_features: int | None = None):
        self.objective = objective
        self.max_features = max_features
        super().__init__(n_var=objective.p, n_obj=2, n_ieq_constr=0, vtype=bool)

    def _evaluate(self, X, out, *args, **kwargs):
        masks = np.asarray(X, dtype=bool)
        if self.max_features is not None:
            masks = np.array([repair_cardinality(m, self.max_features) for m in masks])
        out["F"] = self.objective.evaluate_masks(masks)
        out["_masks"] = masks


def repair_cardinality(mask: np.ndarray, max_features: int, rng=None) -> np.ndarray:
    """Trim a mask to at most ``max_features`` selected entries.

    Deterministic by default (keeps the lowest indices) so that repair does not
    silently inject extra stochasticity into a run whose seed is logged.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    k = int(mask.sum())
    if k <= max_features:
        return mask
    chosen = np.flatnonzero(mask)
    if rng is not None:
        chosen = rng.permutation(chosen)
    mask[:] = False
    mask[chosen[:max_features]] = True
    return mask


def random_masks(p: int, sizes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random boolean masks with the given cardinalities."""
    out = np.zeros((len(sizes), p), dtype=bool)
    for i, k in enumerate(sizes):
        k = int(max(1, min(p, k)))
        out[i, rng.choice(p, size=k, replace=False)] = True
    return out
