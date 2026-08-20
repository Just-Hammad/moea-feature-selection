"""The statistics module is the most reused code in the plan. Test it properly."""

import numpy as np
import pandas as pd
import pytest

from mofs.stats.compare import (
    cliffs_delta,
    cliffs_delta_label,
    friedman_ranks,
    posthoc_holm,
)
from mofs.stats.multiobj import REFERENCE_POINT, hypervolume, non_dominated


# --- the three assertions the shared contract names explicitly --------------

def test_cliffs_delta_against_itself_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=50)
    assert cliffs_delta(sample, sample) == pytest.approx(0.0)


def test_cliffs_delta_strict_dominance_is_plus_one():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)


def test_cliffs_delta_strict_domination_is_minus_one():
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_cliffs_delta_is_antisymmetric():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=30), rng.normal(loc=0.6, size=30)
    assert cliffs_delta(a, b) == pytest.approx(-cliffs_delta(b, a))


def test_cliffs_labels_use_romano_cutpoints():
    assert cliffs_delta_label(0.10) == "negligible"
    assert cliffs_delta_label(0.20) == "small"
    assert cliffs_delta_label(0.40) == "medium"
    assert cliffs_delta_label(0.90) == "large"
    assert cliffs_delta_label(-0.90) == "large"  # magnitude, not sign


def test_cliffs_delta_rejects_empty():
    with pytest.raises(ValueError):
        cliffs_delta([], [1, 2])


# --- Friedman ---------------------------------------------------------------

def _matrix(offsets, n_problems=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {name: rng.normal(loc=off, scale=0.05, size=n_problems) for name, off in offsets.items()}
    )


def test_friedman_ranks_orders_best_first():
    m = _matrix({"good": 0.1, "mid": 0.5, "bad": 0.9})
    stat, p, ranks = friedman_ranks(m)
    assert p < 0.01
    assert list(ranks.index) == ["good", "mid", "bad"]
    assert ranks["good"] < ranks["bad"]


def test_friedman_not_significant_when_identical():
    m = _matrix({"a": 0.5, "b": 0.5, "c": 0.5}, seed=3)
    _, p, _ = friedman_ranks(m)
    assert p > 0.05


def test_friedman_requires_three_groups():
    with pytest.raises(ValueError, match=">=3 groups"):
        friedman_ranks(_matrix({"a": 0.1, "b": 0.9}))


def test_posthoc_holm_is_square_and_labelled():
    m = _matrix({"good": 0.1, "mid": 0.5, "bad": 0.9})
    out = posthoc_holm(m)
    assert list(out.index) == list(m.columns)
    assert out.shape == (3, 3)
    # Holm-corrected p-values are >= the raw ones, so never below zero.
    assert (out.to_numpy() >= 0).all()


# --- multi-objective indicators --------------------------------------------

def test_hypervolume_matches_hand_computation():
    # Two points, ref (1,1): 0.56 + 0.54 - 0.42 overlap = 0.68
    front = np.array([[0.2, 0.3], [0.4, 0.1]])
    assert hypervolume(front, (1.0, 1.0)) == pytest.approx(0.68, abs=1e-9)


def test_hypervolume_of_empty_front_is_zero():
    assert hypervolume(np.zeros((0, 2))) == 0.0


def test_points_not_dominating_reference_contribute_nothing():
    assert hypervolume(np.array([[1.5, 1.5]])) == 0.0


def test_reference_point_is_the_problem_derived_nadir():
    assert REFERENCE_POINT == (1.0, 1.0)


def test_non_dominated_drops_dominated_points():
    pts = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5], [0.95, 0.95]])
    nd = non_dominated(pts)
    assert len(nd) == 3
    assert not any(np.allclose(row, [0.95, 0.95]) for row in nd)


# --- attainment surface -----------------------------------------------------

def test_attainment_surface_returns_two_columns():
    from mofs.stats.multiobj import attainment_surface

    rng = np.random.default_rng(0)
    fronts = [np.sort(rng.random((5, 2)), axis=0) for _ in range(4)]
    surface = attainment_surface(fronts, percentile=50.0)
    assert surface.ndim == 2 and surface.shape[1] == 2
    assert len(surface) > 0


def test_attainment_surface_of_identical_fronts_is_that_front():
    from mofs.stats.multiobj import attainment_surface

    front = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    surface = attainment_surface([front] * 5, percentile=50.0)
    assert len(surface) == len(front)


def test_attainment_surface_handles_empty_input():
    from mofs.stats.multiobj import attainment_surface

    assert len(attainment_surface([])) == 0
    assert len(attainment_surface([np.zeros((0, 2))])) == 0


def test_non_dominated_mask_keeps_duplicate_objective_vectors_by_index():
    """Two different feature sets can score identically -- both must survive."""
    from mofs.stats.multiobj import non_dominated_mask

    pts = np.array([[0.2, 0.4], [0.2, 0.4], [0.9, 0.9]])
    keep = non_dominated_mask(pts)
    assert keep.tolist() == [True, True, False]


def test_sparse_sampling_is_reproducible_across_instances():
    """Guard: pymoo's seed does not reach a default_rng(), so ours must be explicit."""
    from pymoo.core.problem import Problem

    from mofs.selectors.evolutionary import SparseBinarySampling

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=200, n_obj=2, vtype=bool)

    a = SparseBinarySampling(2, 50, seed=123)._do(_P(), 20)
    b = SparseBinarySampling(2, 50, seed=123)._do(_P(), 20)
    c = SparseBinarySampling(2, 50, seed=999)._do(_P(), 20)
    assert np.array_equal(a, b), "same seed must give the same initial population"
    assert not np.array_equal(a, c), "different seeds must differ"
    assert a.sum(axis=1).max() <= 50
