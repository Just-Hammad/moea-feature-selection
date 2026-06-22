"""The shared bi-objective evaluator.

Every selector -- evolutionary, filter, embedded, knockoff, random -- is scored
through *this* object, with the same inner folds, the same classifier and the
same budget accounting. If each method brought its own evaluation protocol the
comparison would measure the protocols, not the methods.

Objectives, both minimised and both normalised to [0, 1]:

    f1 = min(1, |S| / size_scale)   selected features, normalised
    f2 = inner-CV error              misclassification rate over the inner folds

Normalisation is what lets the hypervolume reference point be the fixed,
problem-derived (1.0, 1.0) in `stats/multiobj.py` rather than something chosen
after seeing results.

``size_scale`` defaults to the cardinality ceiling, **not** to p. Dividing by p
looks more natural but is wrong here: at p = 2000 with a 200-feature cap, every
attainable solution has f1 <= 0.1, the size axis collapses, and hypervolume
silently degenerates into an error-only indicator that cannot tell 2 features
from 200. Scaling by the ceiling makes f1 span the range the search actually
operates over. The scale is recorded in the manifest, because a hypervolume is
only comparable to another computed under the same normalisation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def default_classifier() -> LogisticRegression:
    """Fixed, fast, regularised. Deliberately *not* tuned.

    We are studying selection, not classification. Tuning the inner model
    confounds the two and makes every evaluation slow enough to destroy the
    run count, which is the resource the statistics actually need.
    """
    # l1_ratio=0 is ridge. scikit-learn 1.8 deprecated `penalty=`; the two
    # spellings were verified to produce identical coefficients before switching.
    return LogisticRegression(
        l1_ratio=0.0,
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=0,
    )


class InnerObjective:
    """Evaluate feature masks by inner cross-validated error.

    Fold indices are computed once in the constructor, so two selectors
    evaluating the same mask always get exactly the same number.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 3,
        seed: int = 0,
        classifier=None,
        max_features: int | None = None,
        size_scale: int | None = None,
    ) -> None:
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=int)
        self.p = self.X.shape[1]
        self.max_features = max_features
        # Normalise the size objective by the range the search operates over.
        self.size_scale = int(size_scale or max_features or self.p)
        self._proto = classifier if classifier is not None else default_classifier()

        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        self._folds = list(splitter.split(self.X, self.y))

        self._cache: dict[bytes, tuple[float, float]] = {}
        self.n_evaluations = 0  # unique mask evaluations -- the budget counter
        self.n_calls = 0        # including cache hits

    # ------------------------------------------------------------------ #

    def __call__(self, mask: np.ndarray) -> tuple[float, float]:
        mask = np.asarray(mask, dtype=bool).ravel()
        if mask.size != self.p:
            raise ValueError(f"mask length {mask.size} != p {self.p}")

        self.n_calls += 1
        key = np.packbits(mask).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        self.n_evaluations += 1
        result = self._evaluate(mask)
        self._cache[key] = result
        return result

    def _evaluate(self, mask: np.ndarray) -> tuple[float, float]:
        k = int(mask.sum())
        frac = min(1.0, k / self.size_scale)

        # An empty selection cannot be fitted. Assign the worst possible error
        # rather than skipping it, so the optimiser learns to avoid it instead
        # of exploiting a hole in the objective.
        if k == 0:
            return 1.0, 1.0
        if self.max_features is not None and k > self.max_features:
            return frac, 1.0

        Xs = self.X[:, mask]
        errors = []
        for train_idx, test_idx in self._folds:
            model = clone(self._proto)
            model.fit(Xs[train_idx], self.y[train_idx])
            pred = model.predict(Xs[test_idx])
            errors.append(float(np.mean(pred != self.y[test_idx])))
        return frac, float(np.mean(errors))

    # ------------------------------------------------------------------ #

    def evaluate_masks(self, masks: np.ndarray) -> np.ndarray:
        """Evaluate an (m, p) boolean array; returns an (m, 2) objective array."""
        return np.array([self(m) for m in np.atleast_2d(masks)], dtype=float)

    def budget_report(self) -> dict[str, int]:
        return {
            "unique_evaluations": self.n_evaluations,
            "total_calls": self.n_calls,
            "cache_hits": self.n_calls - self.n_evaluations,
        }
