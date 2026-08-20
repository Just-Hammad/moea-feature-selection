"""The leak detector.

These two tests are a matched pair and must be read together:

* with real signal the pipeline must find it (low outer error), and
* with permuted labels the pipeline must fail (chance outer error).

Either alone is worthless. A pipeline that is simply broken passes the
permutation test trivially; a pipeline that leaks passes the signal test
spectacularly.
"""

import numpy as np
import pytest

from mofs.data import make_synthetic
from mofs.nested_cv import run_nested_cv, stability
from mofs.selectors.baselines import UnivariateFilterSelector
from mofs.selectors.base import jaccard


def _mean_balanced_error(results):
    return float(np.mean([r.outer_balanced_error for r in results]))


@pytest.fixture(scope="module")
def dataset():
    return make_synthetic(n=120, p=400, n_informative=15, block_size=40,
                          rho=0.3, effect=2.0, seed=7)


def test_pipeline_detects_real_signal(dataset):
    results = []
    for seed in range(3):
        results += run_nested_cv(
            dataset, UnivariateFilterSelector, "t-test + BH",
            seed=seed, n_outer=4, inner_folds=3, permute_labels=False,
        )
    assert _mean_balanced_error(results) < 0.35, "pipeline cannot find planted signal"


def test_permuted_labels_give_chance_performance(dataset):
    """THE gate. Outer error below chance here means selection saw the test fold."""
    results = []
    for seed in range(3):
        results += run_nested_cv(
            dataset, UnivariateFilterSelector, "t-test + BH",
            seed=seed, n_outer=4, inner_folds=3, permute_labels=True,
        )
    mean_error = _mean_balanced_error(results)
    assert mean_error > 0.35, (
        f"balanced error {mean_error:.3f} under permuted labels is better than chance "
        "-- the selector is seeing the outer test fold"
    )
    assert mean_error < 0.70


def test_permutation_clears_the_ground_truth(dataset):
    """Under permutation there is no true support, so selection quality is undefined."""
    results = run_nested_cv(
        dataset, UnivariateFilterSelector, "t-test + BH",
        seed=0, n_outer=3, permute_labels=True,
    )
    assert all(r.selection_precision is None for r in results)


def test_support_quality_is_measured_on_synthetic_data(dataset):
    results = run_nested_cv(
        dataset, UnivariateFilterSelector, "t-test + BH", seed=0, n_outer=3,
    )
    for r in results:
        assert r.selection_precision is not None
        assert 0.0 <= r.selection_precision <= 1.0
        assert r.realised_fdr == pytest.approx(1.0 - r.selection_precision)


def test_every_fold_reports_a_majority_baseline(dataset):
    results = run_nested_cv(
        dataset, UnivariateFilterSelector, "t-test + BH", seed=0, n_outer=3,
    )
    assert all(0.0 <= r.majority_baseline <= 0.5 for r in results)


def test_jaccard_bounds():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert jaccard(a, a) == 1.0
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert jaccard(a, np.zeros(4, dtype=bool)) == 0.0


def test_stability_summarises_pairwise_overlap():
    supports = [np.array([True, True, False]), np.array([True, False, False])]
    out = stability(supports)
    assert out["n_pairs"] == 1
    assert out["jaccard_mean"] == pytest.approx(0.5)
