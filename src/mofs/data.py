"""Datasets, with provenance recorded alongside the arrays.

`10-papers-labs-and-timeline.md` section 1 is explicit that describing a
dataset's provenance is part of what makes something research rather than a
project, and in omics the preprocessing *is* half the result. So every dataset
carries a `Provenance` record that is serialised into the results and printed
in the report.

Two kinds of dataset are deliberately supported:

* **Synthetic, with known ground truth.** Real data can never tell you whether
  a selector found the *right* features -- only whether it predicted well.
  Synthetic data with a known support lets us measure selection precision and
  recall directly, and is what the FDR claims in the knockoff baseline are
  validated against.
* **Real microarray**, small-n-large-p, fetched from OpenML.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Provenance:
    source: str
    identifier: str
    preprocessing: tuple[str, ...]
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    provenance: Provenance
    #: Boolean mask of the truly informative features. Only ever available for
    #: synthetic data; ``None`` for real data, where nobody knows it.
    true_support: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def fingerprint(self) -> str:
        """Content hash, so a results file can prove which data produced it."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.y, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> str:
        truth = "known" if self.true_support is not None else "unknown"
        return (
            f"{self.name}: n={self.n}, p={self.p}, "
            f"class balance={np.bincount(self.y).tolist()}, "
            f"true support {truth}, sha256:{self.fingerprint()}"
        )


def make_synthetic(
    n: int = 80,
    p: int = 2000,
    n_informative: int = 20,
    block_size: int = 50,
    rho: float = 0.5,
    effect: float = 1.2,
    seed: int = 0,
) -> Dataset:
    """Block-correlated Gaussian design with a known informative support.

    Real omics data has correlated gene modules, and an independent-features
    generator makes feature selection look far easier than it is -- so features
    are generated in equicorrelated blocks of ``block_size`` with within-block
    correlation ``rho``.

    The informative features are spread across blocks, so every informative
    feature has correlated *null* neighbours. That is what makes the problem
    hard, and it is exactly the situation a false-discovery-rate guarantee is
    supposed to handle.
    """
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(p / block_size))
    blocks = []
    for _ in range(n_blocks):
        size = min(block_size, p - sum(b.shape[1] for b in blocks))
        common = rng.standard_normal((n, 1))
        idio = rng.standard_normal((n, size))
        blocks.append(np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio)
    X = np.hstack(blocks)[:, :p]

    # One informative feature per block until we run out, so signal never
    # clusters inside a single correlated block.
    informative = (np.arange(n_informative) * block_size) % p
    informative = np.unique(informative)[:n_informative]
    if len(informative) < n_informative:  # pragma: no cover - tiny p
        extra = rng.choice(np.setdiff1d(np.arange(p), informative),
                           n_informative - len(informative), replace=False)
        informative = np.concatenate([informative, extra])

    beta = np.zeros(p)
    beta[informative] = effect * rng.choice([-1.0, 1.0], size=len(informative))

    logits = X @ beta
    logits -= logits.mean()
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)

    # Guard against a degenerate draw producing a single class.
    if len(np.unique(y)) < 2:  # pragma: no cover
        y[np.argsort(logits)[: n // 2]] = 0
        y[np.argsort(logits)[n // 2 :]] = 1

    support = np.zeros(p, dtype=bool)
    support[informative] = True

    return Dataset(
        name=f"synthetic_n{n}_p{p}_k{n_informative}",
        X=X,
        y=y,
        provenance=Provenance(
            source="simulated",
            identifier=f"make_synthetic(seed={seed})",
            preprocessing=("block-equicorrelated gaussian", f"rho={rho}", f"block={block_size}"),
            notes=(
                "Ground-truth support is known, so selection precision/recall and "
                "realised FDR are measurable. Informative features are spread one "
                "per correlated block."
            ),
        ),
        true_support=support,
        meta={"n_informative": int(support.sum()), "rho": rho, "effect": effect},
    )


#: Real small-n-large-p microarray benchmarks. Resolved lazily -- fetching
#: requires network access, and the runner records which ones were reachable.
OPENML_REGISTRY: dict[str, dict[str, Any]] = {
    "leukemia": {
        "data_id": 1104,
        "notes": "Golub et al. 1999 ALL/AML microarray. The canonical small-n-large-p benchmark.",
    },
    "colon": {
        "data_id": 45087,
        "notes": "Alon et al. 1999 colon tumour vs normal tissue microarray.",
    },
}


def load_openml(key: str, standardise: bool = True) -> Dataset:
    """Fetch a registered OpenML microarray dataset.

    Standardisation is applied to the *whole* array here only because these
    benchmarks are distributed pre-normalised per array and the scaling is not
    label-dependent. Every label-dependent step happens strictly inside the
    outer training fold -- see `nested_cv.py`.
    """
    from sklearn.datasets import fetch_openml

    if key not in OPENML_REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(OPENML_REGISTRY)}")
    spec = OPENML_REGISTRY[key]

    bunch = fetch_openml(data_id=spec["data_id"], as_frame=True, parser="auto")
    X = bunch.data.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    y_raw = np.asarray(bunch.target)
    classes, y = np.unique(y_raw, return_inverse=True)
    if len(classes) != 2:
        raise ValueError(f"{key} is not binary: {classes}")

    steps = ["numeric columns only"]
    if standardise:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
        steps.append("z-scored per feature (label-independent)")

    return Dataset(
        name=key,
        X=X,
        y=y.astype(int),
        provenance=Provenance(
            source="OpenML",
            identifier=f"data_id={spec['data_id']}",
            preprocessing=tuple(steps),
            notes=spec["notes"] + f" Classes: {classes.tolist()}.",
        ),
        true_support=None,
    )


def load(spec: dict[str, Any]) -> Dataset:
    """Build a dataset from a config block."""
    kind = spec["kind"]
    if kind == "synthetic":
        return make_synthetic(**{k: v for k, v in spec.items() if k != "kind"})
    if kind == "openml":
        return load_openml(spec["key"], standardise=spec.get("standardise", True))
    raise ValueError(f"unknown dataset kind {kind!r}")
