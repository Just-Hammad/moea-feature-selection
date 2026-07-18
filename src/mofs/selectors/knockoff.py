"""Model-X Gaussian knockoffs with false-discovery-rate control.

Barber & Candes / Candes, Fan, Janson & Lv. Knockoff copies of each feature
preserve the correlation structure but are conditionally independent of the
response; a feature selected far more strongly than its own knockoff is real,
and the threshold rule controls FDR at a chosen level q.

**Why this sits next to NSGA-II.** The evolutionary front offers a trade-off
curve with *no error guarantee*; the knockoff filter offers one set with a
provable FDR bound. That contrast is the interesting comparison, and it is the
bridge to Liu `51988` ("A Knock-off Enhanced KP-GNN Framework") without paying
a GNN's compute cost.

**Assumptions, stated because they matter here.** Model-X knockoffs assume the
feature distribution is known. On genomic data with p >> n it is emphatically
not, and a second-order (Gaussian, shrinkage-estimated) construction is an
approximation with *two distinct* failure modes that pull in opposite
directions. Both are measured and recorded on every call:

1. **Strong correlation collapses ``s``.** ``s = min(1, 2 * lambda_min(R))``,
   so as features become strongly correlated lambda_min falls, ``s`` falls with
   it, and the knockoffs converge on exact copies of the features. Power goes
   to zero. Measured here at rho=0.9: ``s`` around 0.07.

2. **p >> n inflates ``s`` while destroying the estimate.** Ledoit-Wolf shrinks
   toward a scaled identity, which *raises* lambda_min by construction -- at
   n=20, p=300 of pure iid noise ``s`` sits at a perfectly healthy 1.0 while
   the off-diagonal correlation error is 0.17. The knockoffs then mimic a
   correlation structure the data does not have, exchangeability is violated,
   and FDR is inflated above nominal.

The trap is that these interact: **heavy shrinkage masks failure mode 1 by
artificially inflating lambda_min, so ``s`` looks healthiest precisely when the
covariance estimate is least trustworthy.** ``shrinkage`` is therefore recorded
alongside ``s``, and a high value means the ``s`` next to it should not be
believed. Reading only one of the two is how a knockoff analysis silently
reports an FDR it is not delivering.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

from ..objective import InnerObjective
from .base import SelectionResult


def gaussian_knockoffs(
    X: np.ndarray, rng: np.random.Generator, jitter: float = 1e-10
) -> tuple[np.ndarray, dict]:
    """Equicorrelated second-order Gaussian knockoffs.

    Returns the knockoff matrix and a diagnostics dict.
    """
    n, p = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd

    # Ledoit-Wolf shrinkage keeps the estimate positive definite when p > n,
    # which the plain sample covariance is not.
    estimator = LedoitWolf(assume_centered=False).fit(Z)
    Sigma = estimator.covariance_
    d = np.sqrt(np.clip(np.diag(Sigma), jitter, None))
    R = Sigma / np.outer(d, d)
    R = (R + R.T) / 2.0

    eigvals = np.linalg.eigvalsh(R)
    eig_min = float(max(eigvals.min(), 0.0))
    s_val = float(min(1.0, 2.0 * eig_min))
    s = np.full(p, s_val)

    shrinkage = float(getattr(estimator, "shrinkage_", float("nan")))
    diagnostics = {
        "eig_min": eig_min,
        "s": s_val,
        # Fraction of the estimate that is the identity target. Above ~0.5 the
        # correlation structure is mostly invented and `s` cannot be believed.
        "shrinkage": shrinkage,
        "degenerate": bool(s_val <= 1e-6),
        "low_power": bool(s_val < 0.2),
        "estimate_unreliable": bool(shrinkage > 0.5),
    }

    if s_val <= 1e-6:
        # Knockoffs would be near-exact copies of X. Return them anyway and let
        # the caller report zero power -- silently substituting a different
        # construction here would hide the finding.
        return (Z * sd + mu), diagnostics

    R_inv = np.linalg.pinv(R)
    # Conditional mean:  Z - Z R^-1 diag(s)
    M = Z - (Z @ R_inv) * s
    # Conditional covariance:  2 diag(s) - diag(s) R^-1 diag(s)
    V = 2.0 * np.diag(s) - np.outer(s, s) * R_inv
    V = (V + V.T) / 2.0

    w, Q = np.linalg.eigh(V)
    w = np.clip(w, 0.0, None)
    L = Q * np.sqrt(w)
    E = rng.standard_normal((n, p)) @ L.T

    return (M + E) * sd + mu, diagnostics


def lasso_coefficient_difference(
    X: np.ndarray, X_knockoff: np.ndarray, y: np.ndarray, C: float = 0.1
) -> np.ndarray:
    """W_j = |beta_j| - |beta_{j+p}| from an L1 logistic fit on [X, X~]."""
    p = X.shape[1]
    augmented = np.hstack([X, X_knockoff])
    model = LogisticRegression(  # l1_ratio=1 is lasso
        l1_ratio=1.0, C=float(C), solver="liblinear", max_iter=5000, random_state=0
    )
    model.fit(augmented, y)
    beta = np.abs(model.coef_.ravel())
    return beta[:p] - beta[p:]


def knockoff_threshold(W: np.ndarray, q: float = 0.1, offset: int = 1) -> float:
    """Knockoff (offset=0) / knockoff+ (offset=1) data-dependent threshold.

    tau = min{ t > 0 : (offset + #{W_j <= -t}) / max(1, #{W_j >= t}) <= q }
    """
    W = np.asarray(W, dtype=float)
    candidates = np.sort(np.unique(np.abs(W[W != 0])))
    for t in candidates:
        numerator = offset + int(np.sum(W <= -t))
        denominator = max(1, int(np.sum(W >= t)))
        if numerator / denominator <= q:
            return float(t)
    return float("inf")


class KnockoffSelector:
    """Model-X knockoff filter at a target FDR.

    Single-answer method: it returns one set, not a front. The front field is
    therefore a single point -- an honest representation of what the procedure
    offers, and what makes its hypervolume directly comparable to the
    evolutionary methods' at the same operating point.
    """

    name = "Knockoff+"
    is_multiobjective = False

    def __init__(self, q: float = 0.1, C: float = 0.1, screen_to: int | None = 500, offset: int = 1):
        self.q = q
        self.C = C
        self.screen_to = screen_to
        self.offset = offset

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        X, y = objective.X, objective.y
        p_full = X.shape[1]
        screen_idx = np.arange(p_full)

        # Screening keeps the p x p eigendecomposition tractable. It is applied
        # to the same data used for selection, which BREAKS the strict FDR
        # guarantee -- the honest name for what follows is "screened
        # knockoffs", and it is recorded in the metadata and reported.
        screened = False
        if self.screen_to is not None and p_full > self.screen_to:
            with np.errstate(invalid="ignore", divide="ignore"):
                t, _ = stats.ttest_ind(X[y == 0], X[y == 1], axis=0, equal_var=False)
            rank = np.argsort(-np.nan_to_num(np.abs(t), nan=0.0))
            screen_idx = np.sort(rank[: self.screen_to])
            X = X[:, screen_idx]
            screened = True

        X_knockoff, diagnostics = gaussian_knockoffs(X, rng)
        W = lasso_coefficient_difference(X, X_knockoff, y, C=self.C)
        tau = knockoff_threshold(W, q=self.q, offset=self.offset)

        support = np.zeros(p_full, dtype=bool)
        chosen_local = np.flatnonzero(W >= tau) if np.isfinite(tau) else np.array([], dtype=int)
        support[screen_idx[chosen_local]] = True

        if support.sum() > 0:
            front = np.array([objective(support)], dtype=float)
            front_masks = support.reshape(1, -1)
        else:
            front = np.zeros((0, 2))
            front_masks = np.zeros((0, p_full), dtype=bool)

        return SelectionResult(
            support=support,
            front=front,
            front_masks=front_masks,
            meta={
                "q": self.q,
                "threshold": tau if np.isfinite(tau) else None,
                "n_selected": int(support.sum()),
                "screened": screened,
                "screen_to": self.screen_to if screened else None,
                "fdr_guarantee": "broken by same-data screening" if screened else "nominal",
                **diagnostics,
            },
        )
