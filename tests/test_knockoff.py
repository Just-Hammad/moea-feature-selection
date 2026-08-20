"""Knockoff filter: threshold arithmetic exactly, FDR control empirically."""

import numpy as np
import pytest

from mofs.data import make_synthetic
from mofs.objective import InnerObjective
from mofs.selectors.knockoff import (
    KnockoffSelector,
    gaussian_knockoffs,
    knockoff_threshold,
)


# --- threshold arithmetic (deterministic, so assert it exactly) -------------

def test_threshold_picks_smallest_t_meeting_the_ratio():
    # Strong positives, no negatives: knockoff+ needs (1 + 0)/count <= q.
    W = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    tau = knockoff_threshold(W, q=0.2, offset=1)
    # at t=1: (1+0)/5 = 0.2 <= 0.2  -> accepted, and 1 is the smallest candidate
    assert tau == pytest.approx(1.0)
    assert int(np.sum(W >= tau)) == 5


def test_threshold_is_infinite_when_fdr_unreachable():
    # Every positive is mirrored by an equally large negative: nothing is safe.
    W = np.array([1.0, -1.0, 2.0, -2.0])
    assert knockoff_threshold(W, q=0.01, offset=1) == float("inf")


def test_knockoff_plus_is_more_conservative_than_knockoff():
    W = np.array([3.0, 2.0, 1.0, -1.0])
    assert knockoff_threshold(W, q=0.5, offset=1) >= knockoff_threshold(W, q=0.5, offset=0)


def test_threshold_ignores_zero_statistics():
    W = np.array([0.0, 0.0, 3.0, 2.0])
    assert np.isfinite(knockoff_threshold(W, q=0.5, offset=1))


# --- the construction -------------------------------------------------------

def test_knockoffs_have_the_same_shape_and_are_not_copies():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 15))
    Xk, diag = gaussian_knockoffs(X, rng)
    assert Xk.shape == X.shape
    assert not diag["degenerate"]
    assert not np.allclose(Xk, X)


def test_knockoffs_roughly_preserve_marginal_scale():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 8)) * 3.0 + 5.0
    Xk, _ = gaussian_knockoffs(X, rng)
    assert np.allclose(Xk.mean(0), X.mean(0), atol=0.8)
    assert np.allclose(Xk.std(0), X.std(0), rtol=0.4)


def test_strong_correlation_collapses_s_and_is_flagged():
    """Failure mode 1: correlated features drive lambda_min, and s, toward 0."""
    ds = make_synthetic(n=80, p=200, block_size=50, rho=0.9, seed=1)
    _, diag = gaussian_knockoffs(ds.X, np.random.default_rng(0))
    assert diag["s"] < 0.3, "strongly correlated features should collapse s"
    assert diag["low_power"] is True


def test_heavy_shrinkage_inflates_s_and_is_also_flagged():
    """Failure mode 2: p >> n makes s look healthy while the estimate is junk.

    This is the trap the diagnostics exist for -- s alone would report no
    problem at all here.
    """
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 300))
    _, diag = gaussian_knockoffs(X, rng)
    assert diag["s"] == pytest.approx(1.0)          # looks perfectly healthy
    assert diag["shrinkage"] > 0.5                  # and yet
    assert diag["estimate_unreliable"] is True


def test_both_diagnostics_are_always_present():
    rng = np.random.default_rng(3)
    _, diag = gaussian_knockoffs(rng.standard_normal((50, 20)), rng)
    for key in ("eig_min", "s", "shrinkage", "degenerate", "low_power", "estimate_unreliable"):
        assert key in diag


# --- empirical FDR ----------------------------------------------------------

@pytest.mark.slow
def test_realised_fdr_is_controlled_on_synthetic_data():
    """Across repetitions the mean realised FDR should sit near or below q.

    This is an expectation over draws, not a per-run guarantee, so the check is
    deliberately loose -- it catches a broken threshold rule, not a subtle
    power loss.
    """
    q = 0.2
    fdrs = []
    for seed in range(6):
        ds = make_synthetic(n=150, p=120, n_informative=12, block_size=30,
                            rho=0.2, effect=2.5, seed=seed)
        obj = InnerObjective(ds.X, ds.y, n_folds=3, seed=seed)
        res = KnockoffSelector(q=q, screen_to=None).fit_select(
            obj, np.random.default_rng(seed)
        )
        k = res.n_selected
        if k == 0:
            continue
        false_positives = int((res.support & ~ds.true_support).sum())
        fdrs.append(false_positives / k)

    assert fdrs, "knockoff filter selected nothing in every repetition"
    assert np.mean(fdrs) <= q + 0.25
