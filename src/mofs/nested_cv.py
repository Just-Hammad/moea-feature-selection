"""Nested cross-validation -- the correctness gate.

The failure this file exists to prevent: selecting features using
cross-validated error and then reporting that same cross-validated error as
the performance of the selected set. The CV error has been optimised against
directly, so it is a training metric wearing a validation metric's clothes. On
an 80-sample, 2000-feature dataset an evolutionary algorithm will drive it to
zero on pure noise.

    OUTER fold k   <- the only honest estimate lives here
     |- training portion
     |    |- scaler + InnerObjective built HERE, from training rows only
     |    |- selector runs, using inner CV only
     |    '- chosen support refit on the full training portion
     '- OUTER TEST FOLD -> reported error. Never touched by selection.

Two numbers come out of every fold and they answer different questions:

* ``hypervolume`` is computed on the **inner** front and measures search quality.
* ``outer_error`` is computed on the **held-out** fold and measures generalisation.

Keeping them apart is the entire point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .objective import InnerObjective, default_classifier
from .selectors.base import SelectionResult
from .stats.multiobj import REFERENCE_POINT, hypervolume


@dataclass
class FoldResult:
    dataset: str
    selector: str
    seed: int
    fold: int
    n_selected: int
    outer_error: float
    outer_balanced_error: float
    majority_baseline: float
    inner_error: float
    hypervolume: float
    front_size: int
    n_evaluations: int
    support: np.ndarray = field(repr=False)
    front: np.ndarray | None = field(default=None, repr=False)
    selection_precision: float | None = None
    selection_recall: float | None = None
    realised_fdr: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in {"support", "front", "meta"}
        }


def _support_quality(support: np.ndarray, truth: np.ndarray | None) -> dict[str, float | None]:
    """Precision / recall / realised FDR against a known support.

    Only computable on synthetic data. On real data nobody knows the answer,
    which is exactly why the synthetic arm of the experiment exists.
    """
    if truth is None:
        return {"selection_precision": None, "selection_recall": None, "realised_fdr": None}
    selected = int(support.sum())
    if selected == 0:
        return {"selection_precision": 0.0, "selection_recall": 0.0, "realised_fdr": 0.0}
    true_positive = int((support & truth).sum())
    return {
        "selection_precision": true_positive / selected,
        "selection_recall": true_positive / max(1, int(truth.sum())),
        "realised_fdr": 1.0 - true_positive / selected,
    }


def run_nested_cv(
    dataset: Dataset,
    selector_factory: Callable[[], Any],
    selector_name: str,
    seed: int = 0,
    n_outer: int = 5,
    inner_folds: int = 3,
    permute_labels: bool = False,
    size_scale: int | None = None,
) -> list[FoldResult]:
    """Run one selector over every outer fold.

    ``permute_labels`` shuffles y *once, before splitting*, which destroys all
    signal while preserving the class balance and the feature correlation
    structure. Outer error must then sit at the majority-class baseline. If it
    does not, there is a leak, and no amount of staring at the code finds it
    faster than this check does.
    """
    X, y = dataset.X, dataset.y
    rng = np.random.default_rng(seed)

    if permute_labels:
        y = rng.permutation(y)

    truth = dataset.true_support if not permute_labels else None
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    results: list[FoldResult] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fitted on training rows only. A scaler fitted on the full matrix is
        # a leak -- small, but real, and it is the kind reviewers look for.
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        objective = InnerObjective(
            X_train_s, y_train, n_folds=inner_folds, seed=seed, size_scale=size_scale
        )
        fold_rng = np.random.default_rng([seed, fold])
        selector = selector_factory()
        if getattr(selector, "needs_true_support", False):
            selector.true_support = dataset.true_support
        result: SelectionResult = selector.fit_select(objective, fold_rng)

        support = result.support
        n_selected = int(support.sum())

        if n_selected == 0:
            # Degenerate selection: predict the training majority class.
            majority = int(np.bincount(y_train).argmax())
            pred = np.full(len(y_test), majority)
            inner_error = 1.0
        else:
            model = clone(default_classifier())
            model.fit(X_train_s[:, support], y_train)
            pred = model.predict(X_test_s[:, support])
            inner_error = float(objective(support)[1])

        outer_error = float(np.mean(pred != y_test))

        # Balanced error, because these cohorts are rarely balanced and plain
        # accuracy flatters a majority-class predictor.
        per_class = [
            float(np.mean(pred[y_test == c] != c)) for c in np.unique(y_test)
        ]
        balanced_error = float(np.mean(per_class))
        majority_rate = float(1.0 - np.bincount(y_train).max() / len(y_train))

        front = result.front if result.front is not None else np.zeros((0, 2))
        hv = hypervolume(front, REFERENCE_POINT) if len(front) else 0.0

        results.append(
            FoldResult(
                dataset=dataset.name,
                selector=selector_name,
                seed=seed,
                fold=fold,
                n_selected=n_selected,
                outer_error=outer_error,
                outer_balanced_error=balanced_error,
                majority_baseline=majority_rate,
                inner_error=inner_error,
                hypervolume=float(hv),
                front_size=int(len(front)),
                n_evaluations=int(objective.n_evaluations),
                support=support,
                front=front,
                **_support_quality(support, truth),
                meta={**result.meta, "budget": objective.budget_report()},
            )
        )

    return results


def stability(supports: list[np.ndarray]) -> dict[str, float]:
    """Across-fold feature-set stability.

    Reported because in this literature the Jaccard overlap between feature
    sets chosen on different folds is routinely near zero, and almost nobody
    publishes it. An unstable selector that predicts well has found *a*
    predictive subspace, not *the* biomarkers -- a materially different claim.
    """
    from .selectors.base import jaccard

    pairs = [
        jaccard(supports[i], supports[j])
        for i in range(len(supports))
        for j in range(i + 1, len(supports))
    ]
    if not pairs:
        return {"jaccard_mean": float("nan"), "jaccard_median": float("nan")}
    return {
        "jaccard_mean": float(np.mean(pairs)),
        "jaccard_median": float(np.median(pairs)),
        "n_pairs": len(pairs),
    }
