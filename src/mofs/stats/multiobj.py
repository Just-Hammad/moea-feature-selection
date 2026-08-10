"""Pareto-front quality metrics.

Uses `moocore` -- maintained by Lopez-Ibanez, Fonseca and Paquete, i.e. the
hypervolume and attainment-function literature itself -- rather than
hand-rolling the indicators. Falls back to pymoo only if moocore is absent.

The reference point is *derived from the problem and never tuned*: objectives
are normalised to [0, 1] x [0, 1] (selected fraction, error rate), so the
nadir is exactly (1.0, 1.0). Choosing a reference point after seeing results
is p-hacking with extra steps, so it is a module constant, not an argument
with a default someone can quietly change per-experiment.
"""

from __future__ import annotations

import numpy as np

#: Reference point for hypervolume. Objectives are (fraction of features
#: selected, classification error), both in [0, 1] and both minimised, so the
#: worst attainable point is (1, 1). Stated in every caption. Never tuned.
REFERENCE_POINT: tuple[float, float] = (1.0, 1.0)

try:
    import moocore

    _BACKEND = "moocore"
except ImportError:  # pragma: no cover
    moocore = None
    _BACKEND = "pymoo"


def backend() -> str:
    """Which indicator implementation is in use; recorded in results metadata."""
    return _BACKEND


def non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated subset (minimisation, both objectives).

    Returning a mask rather than the points themselves lets callers carry the
    corresponding feature masks along by index. Matching points back to their
    masks by *value* is fragile -- duplicate objective vectors are common, since
    many different feature sets achieve the same size and error.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not keep[i]:
            continue
        dominated = np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)
        keep &= ~dominated
        keep[i] = True
    return keep


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset of ``points`` (minimisation, both objectives)."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    return pts[non_dominated_mask(pts)]


def hypervolume(front: np.ndarray, ref_point: tuple[float, float] = REFERENCE_POINT) -> float:
    """Hypervolume of a minimisation front relative to ``ref_point``.

    Points that do not dominate the reference point contribute nothing and are
    dropped, which is the standard convention -- but we drop them explicitly
    rather than letting the backend decide, so the number is reproducible
    across backends.
    """
    pts = np.asarray(front, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected an (n, 2) front, got shape {pts.shape}")

    ref = np.asarray(ref_point, dtype=float)
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    if _BACKEND == "moocore":
        return float(moocore.hypervolume(pts, ref=ref))

    from pymoo.indicators.hv import HV  # pragma: no cover

    return float(HV(ref_point=ref)(pts))  # pragma: no cover


def igd(front: np.ndarray, reference_set: np.ndarray) -> float:
    """Inverted generational distance against a reference set."""
    pts = np.asarray(front, dtype=float)
    ref = np.asarray(reference_set, dtype=float)
    if len(pts) == 0:
        return float("inf")
    if _BACKEND == "moocore":
        return float(moocore.igd(pts, ref=ref))
    d = np.sqrt(((ref[:, None, :] - pts[None, :, :]) ** 2).sum(-1))  # pragma: no cover
    return float(d.min(axis=1).mean())  # pragma: no cover


def attainment_surface(fronts: list[np.ndarray], percentile: float = 50.0) -> np.ndarray:
    """Empirical attainment surface across a *collection* of fronts.

    The correct way to visualise a distribution of fronts: one front from one
    seed is one sample, and plotting only the best of them is the mistake the
    shared contract warns about. The 50th percentile surface is the region
    attained by at least half the runs.

    ``moocore.eaf`` takes points and their set membership as two separate
    arguments and returns an (m, 3) array whose last column is the percentile;
    getting that signature wrong silently produces a plausible-looking but
    meaningless curve, so the shape is asserted rather than trusted.
    """
    usable = [np.asarray(f, dtype=float).reshape(-1, 2) for f in fronts if len(f)]
    if not usable:
        return np.zeros((0, 2))
    if _BACKEND != "moocore":  # pragma: no cover
        return non_dominated(np.vstack(usable))

    points = np.vstack(usable)
    sets = np.concatenate([np.full(len(f), i) for i, f in enumerate(usable)])
    out = np.asarray(moocore.eaf(points, sets, percentiles=[float(percentile)]))
    if out.ndim != 2 or out.shape[1] != 3:  # pragma: no cover
        raise RuntimeError(f"unexpected eaf output shape {out.shape}; expected (m, 3)")
    return out[np.isclose(out[:, 2], percentile)][:, :2]
