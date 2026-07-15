"""NSGA-II / NSGA-III feature selection via pymoo."""

from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from ..objective import InnerObjective
from ..problem import FeatureSelectionProblem
from .base import SelectionResult, pick_from_front


class SparseBinarySampling(Sampling):
    """Initial population with controlled sparsity.

    pymoo's stock ``BinaryRandomSampling`` draws each bit at p=0.5, so on a
    20,000-gene dataset every initial individual selects ~10,000 features and
    the whole population is indistinguishable noise. We instead draw a target
    cardinality per individual, log-uniformly between ``k_min`` and ``k_max``,
    which spreads the initial population across the size axis the search is
    supposed to trade off along.
    """

    def __init__(self, k_min: int = 2, k_max: int = 100, seed: int | None = None):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        # Seeded explicitly. pymoo's `minimize(seed=...)` seeds numpy's *legacy*
        # global state, which a Generator created by default_rng() does not
        # observe -- so relying on it here would leave initialisation silently
        # non-deterministic while every other part of the run was reproducible.
        self.seed = seed

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(self.seed if self.seed is not None else kwargs.get("seed"))
        p = problem.n_var
        k_hi = int(min(self.k_max, p))
        k_lo = int(max(1, min(self.k_min, k_hi)))
        out = np.zeros((n_samples, p), dtype=bool)
        sizes = np.exp(rng.uniform(np.log(k_lo), np.log(k_hi), size=n_samples)).astype(int)
        for i, k in enumerate(sizes):
            out[i, rng.choice(p, size=int(max(1, k)), replace=False)] = True
        return out


class _EvolutionarySelector:
    is_multiobjective = True

    def __init__(
        self,
        pop_size: int = 40,
        n_gen: int = 50,
        max_features: int | None = 200,
        k_init: tuple[int, int] = (2, 100),
        pick_rule: str = "best_error",
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.max_features = max_features
        self.k_init = k_init
        self.pick_rule = pick_rule

    def _algorithm(self, seed: int):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        seed = int(rng.integers(0, 2**31 - 1))
        problem = FeatureSelectionProblem(objective, max_features=self.max_features)

        res = minimize(
            problem,
            self._algorithm(seed),
            termination=("n_gen", self.n_gen),
            seed=seed,
            save_history=False,
            verbose=False,
        )

        masks = np.atleast_2d(np.asarray(res.X, dtype=bool))
        front = np.atleast_2d(np.asarray(res.F, dtype=float))
        support = pick_from_front(front, masks, rule=self.pick_rule)

        return SelectionResult(
            support=support,
            front=front,
            front_masks=masks,
            meta={
                "seed": seed,
                "pop_size": self.pop_size,
                "n_gen": self.n_gen,
                "front_size": int(len(front)),
                "budget": objective.budget_report(),
            },
        )


class NSGA2Selector(_EvolutionarySelector):
    name = "NSGA-II"

    def _algorithm(self, seed: int):
        return NSGA2(
            pop_size=self.pop_size,
            sampling=SparseBinarySampling(*self.k_init, seed=seed),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(prob=0.9),
            eliminate_duplicates=True,
        )


class NSGA3Selector(_EvolutionarySelector):
    name = "NSGA-III"

    def _algorithm(self, seed: int):
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=max(2, self.pop_size - 1))
        return NSGA3(
            ref_dirs=ref_dirs,
            pop_size=self.pop_size,
            sampling=SparseBinarySampling(*self.k_init, seed=seed),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(prob=0.9),
            eliminate_duplicates=True,
        )
