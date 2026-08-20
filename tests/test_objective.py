import numpy as np
import pytest

from mofs.data import make_synthetic
from mofs.objective import InnerObjective
from mofs.problem import repair_cardinality


@pytest.fixture(scope="module")
def objective():
    ds = make_synthetic(n=40, p=60, n_informative=5, block_size=20, seed=0)
    return InnerObjective(ds.X, ds.y, n_folds=3, seed=0)


def test_empty_mask_is_worst_not_an_exception(objective):
    frac, err = objective(np.zeros(objective.p, dtype=bool))
    assert (frac, err) == (1.0, 1.0)


def test_objectives_are_normalised_to_unit_square(objective):
    rng = np.random.default_rng(0)
    for _ in range(10):
        mask = rng.random(objective.p) < 0.2
        frac, err = objective(mask)
        assert 0.0 <= frac <= 1.0
        assert 0.0 <= err <= 1.0


def test_first_objective_is_the_selected_fraction(objective):
    mask = np.zeros(objective.p, dtype=bool)
    mask[:6] = True
    frac, _ = objective(mask)
    assert frac == pytest.approx(6 / objective.p)


def test_identical_masks_give_identical_scores(objective):
    mask = np.zeros(objective.p, dtype=bool)
    mask[[1, 4, 9]] = True
    assert objective(mask) == objective(mask.copy())


def test_cache_makes_repeats_free(objective):
    mask = np.zeros(objective.p, dtype=bool)
    mask[[2, 3]] = True
    objective(mask)
    before = objective.n_evaluations
    for _ in range(5):
        objective(mask)
    assert objective.n_evaluations == before
    assert objective.budget_report()["cache_hits"] >= 5


def test_budget_counter_tracks_unique_masks():
    ds = make_synthetic(n=30, p=40, n_informative=4, block_size=10, seed=2)
    obj = InnerObjective(ds.X, ds.y, n_folds=3, seed=0)
    rng = np.random.default_rng(0)
    masks = [rng.random(obj.p) < 0.3 for _ in range(20)]
    for m in masks:
        obj(m)
    assert obj.n_evaluations <= 20
    assert obj.n_calls == 20


def test_mask_length_is_validated(objective):
    with pytest.raises(ValueError, match="mask length"):
        objective(np.ones(objective.p + 1, dtype=bool))


def test_repair_caps_cardinality_and_is_deterministic():
    mask = np.ones(50, dtype=bool)
    out = repair_cardinality(mask, 7)
    assert out.sum() == 7
    assert np.array_equal(out, repair_cardinality(mask, 7))


def test_repair_leaves_small_masks_alone():
    mask = np.zeros(20, dtype=bool)
    mask[[1, 2]] = True
    assert np.array_equal(repair_cardinality(mask, 10), mask)


def test_classifier_uses_no_deprecated_sklearn_api():
    """Regression guard: `penalty=` is removed in scikit-learn 1.10.

    It also emitted one FutureWarning per fit -- 86,000 of them in a full run,
    which buried the progress output entirely.
    """
    import warnings

    from mofs.objective import default_classifier

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        clf = default_classifier()
        ds = make_synthetic(n=40, p=30, n_informative=4, block_size=10, seed=0)
        clf.fit(ds.X, ds.y)
