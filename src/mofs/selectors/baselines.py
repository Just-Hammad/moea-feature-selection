"""Non-evolutionary baselines.

The important one is `RandomSubsetSelector`. Guide 4's argument is that on
small-n-large-p data, random feature subsets of matched cardinality are a
shockingly strong baseline, and that if the evolutionary search does not
clearly beat them the search contributed nothing. It is given exactly the same
evaluation budget as NSGA-II so the comparison isolates *search* from *budget*.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from ..objective import InnerObjective
from ..problem import random_masks
from ..stats.multiobj import non_dominated_mask
from .base import SelectionResult, pick_from_front


def _front_from_masks(objective: InnerObjective, masks: np.ndarray):
    """Evaluate masks and return their non-dominated subset."""
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    if len(masks) == 0:
        empty = np.zeros(objective.p, dtype=bool)
        return np.zeros((0, 2)), np.zeros((0, objective.p), dtype=bool), empty
    F = objective.evaluate_masks(masks)
    keep = non_dominated_mask(F)
    return F[keep], masks[keep], pick_from_front(F[keep], masks[keep])


class RandomSubsetSelector:
    """Uniformly random feature subsets at matched cardinalities.

    THE ablation. Same budget, same cardinality distribution as the
    evolutionary initial population, no search whatsoever.
    """

    name = "Random subsets"
    is_multiobjective = True

    def __init__(self, budget: int = 2000, k_range: tuple[int, int] = (2, 200)):
        self.budget = budget
        self.k_range = k_range

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        k_hi = int(min(self.k_range[1], objective.p))
        k_lo = int(max(1, min(self.k_range[0], k_hi)))
        sizes = np.exp(
            rng.uniform(np.log(k_lo), np.log(k_hi), size=self.budget)
        ).astype(int)
        masks = random_masks(objective.p, sizes, rng)
        front, front_masks, support = _front_from_masks(objective, masks)
        return SelectionResult(
            support=support,
            front=front,
            front_masks=front_masks,
            meta={"budget": self.budget, "front_size": int(len(front))},
        )


class UnivariateFilterSelector:
    """Two-sample t-test per feature, Benjamini-Hochberg FDR control.

    Produces a front by sweeping the number of top-ranked features retained,
    so it is comparable to the evolutionary methods by hypervolume rather than
    only at a single operating point.
    """

    name = "t-test + BH"
    is_multiobjective = True

    def __init__(self, q: float = 0.1, k_grid: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200)):
        self.q = q
        self.k_grid = k_grid

    def _pvalues(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        a, b = X[y == 0], X[y == 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            _, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
        return np.nan_to_num(p, nan=1.0)

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        pvals = self._pvalues(objective.X, objective.y)
        order = np.argsort(pvals)

        # BH threshold, reported alongside the swept front.
        m = len(pvals)
        ranked = pvals[order]
        crit = self.q * np.arange(1, m + 1) / m
        below = np.flatnonzero(ranked <= crit)
        n_bh = int(below[-1] + 1) if below.size else 0

        grid = sorted({k for k in (*self.k_grid, n_bh) if 0 < k <= objective.p})
        masks = np.zeros((len(grid), objective.p), dtype=bool)
        for i, k in enumerate(grid):
            masks[i, order[:k]] = True

        front, front_masks, support = _front_from_masks(objective, masks)
        return SelectionResult(
            support=support,
            front=front,
            front_masks=front_masks,
            meta={"bh_selected": n_bh, "q": self.q, "front_size": int(len(front))},
        )


class MRMRSelector:
    """Greedy minimum-redundancy maximum-relevance selection.

    Relevance is the absolute t-statistic; redundancy is mean absolute Pearson
    correlation with the already-selected set. The prefix at every k is
    recorded, which gives a front for free.
    """

    name = "mRMR"
    is_multiobjective = True

    def __init__(self, k_max: int = 50, k_grid: tuple[int, ...] = (1, 2, 5, 10, 20, 35, 50)):
        self.k_max = k_max
        self.k_grid = k_grid

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        X, y = objective.X, objective.y
        a, b = X[y == 0], X[y == 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            t, _ = stats.ttest_ind(a, b, axis=0, equal_var=False)
        relevance = np.nan_to_num(np.abs(t), nan=0.0)

        Xz = (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))
        n = X.shape[0]

        selected: list[int] = [int(np.argmax(relevance))]
        red_sum = np.abs(Xz.T @ Xz[:, selected[0]] / n)

        k_max = int(min(self.k_max, objective.p))
        while len(selected) < k_max:
            score = relevance - red_sum / len(selected)
            score[selected] = -np.inf
            nxt = int(np.argmax(score))
            selected.append(nxt)
            red_sum = red_sum + np.abs(Xz.T @ Xz[:, nxt] / n)

        grid = sorted({k for k in self.k_grid if 0 < k <= len(selected)})
        masks = np.zeros((len(grid), objective.p), dtype=bool)
        for i, k in enumerate(grid):
            masks[i, selected[:k]] = True

        front, front_masks, support = _front_from_masks(objective, masks)
        return SelectionResult(
            support=support, front=front, front_masks=front_masks,
            meta={"order": selected[:20], "front_size": int(len(front))},
        )


class LassoSelector:
    """L1-penalised logistic regression; the regularisation path gives the front."""

    name = "LASSO path"
    is_multiobjective = True

    def __init__(self, n_lambda: int = 12, c_range: tuple[float, float] = (1e-3, 1e1)):
        self.n_lambda = n_lambda
        self.c_range = c_range

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        Cs = np.logspace(np.log10(self.c_range[0]), np.log10(self.c_range[1]), self.n_lambda)
        masks, used = [], []
        for C in Cs:
            model = LogisticRegression(  # l1_ratio=1 is lasso
                l1_ratio=1.0, C=float(C), solver="liblinear", max_iter=2000, random_state=0
            )
            model.fit(objective.X, objective.y)
            mask = np.abs(model.coef_.ravel()) > 1e-10
            if mask.sum() == 0:
                continue
            masks.append(mask)
            used.append(float(C))
        if not masks:
            return SelectionResult(support=np.zeros(objective.p, dtype=bool), meta={"empty": True})

        front, front_masks, support = _front_from_masks(objective, np.array(masks))
        return SelectionResult(
            support=support, front=front, front_masks=front_masks,
            meta={"C_grid": used, "front_size": int(len(front))},
        )


class OracleSelector:
    """Uses the true informative support. Only defined for synthetic data.

    Not a competitor -- a **ceiling**. Without it, a table in which every
    method sits near chance is ambiguous between "these methods are weak" and
    "this dataset carries no learnable signal". The oracle separates those two
    readings, and on small-n-large-p data the gap between it and the best real
    selector is usually the most informative number in the study.
    """

    name = "Oracle (true support)"
    is_multiobjective = False
    needs_true_support = True

    def __init__(self) -> None:
        self.true_support: np.ndarray | None = None

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        if self.true_support is None:
            raise RuntimeError(
                "OracleSelector requires a dataset with a known true_support; "
                "it is undefined on real data and must be excluded there."
            )
        support = np.asarray(self.true_support, dtype=bool)
        front = np.array([objective(support)], dtype=float)
        return SelectionResult(
            support=support,
            front=front,
            front_masks=support.reshape(1, -1),
            meta={"ceiling": True, "n_true": int(support.sum())},
        )
