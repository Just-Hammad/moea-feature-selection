"""Common selector interface.

Every selector receives the *same* `InnerObjective` instance, so every method
is scored on identical folds with an identical classifier, and every method's
evaluations are charged to the same budget counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..objective import InnerObjective


@dataclass
class SelectionResult:
    """What a selector returns.

    ``support`` is the single feature set that will be carried to the outer
    test fold. ``front`` is the multi-objective trade-off curve where the
    method produces one; single-answer methods (knockoffs) leave it as a single
    point, which is itself the honest representation of what they offer.
    """

    support: np.ndarray
    front: np.ndarray | None = None
    front_masks: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.support = np.asarray(self.support, dtype=bool).ravel()
        if self.front is not None:
            self.front = np.asarray(self.front, dtype=float).reshape(-1, 2)

    @property
    def n_selected(self) -> int:
        return int(self.support.sum())


class Selector(Protocol):
    name: str
    is_multiobjective: bool

    def fit_select(self, objective: InnerObjective, rng: np.random.Generator) -> SelectionResult:
        ...


def pick_from_front(
    front: np.ndarray, masks: np.ndarray, rule: str = "best_error"
) -> np.ndarray:
    """Choose one solution from a front using INNER information only.

    This function must never see the outer test fold. Choosing the front point
    that happens to do best on held-out data is the selection-bias error the
    whole nested design exists to prevent, so the rule is restricted to the
    inner objectives that produced the front.

    Rules
    -----
    best_error
        Lowest inner-CV error; ties broken toward fewer features.
    knee
        Closest point to the utopia corner after min-max normalisation.
    """
    front = np.asarray(front, dtype=float).reshape(-1, 2)
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    if len(front) == 0:
        return np.zeros(masks.shape[1], dtype=bool)

    if rule == "best_error":
        order = np.lexsort((front[:, 0], front[:, 1]))  # error first, then size
        return masks[order[0]]

    if rule == "knee":
        lo, hi = front.min(axis=0), front.max(axis=0)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        norm = (front - lo) / span
        return masks[int(np.argmin(np.linalg.norm(norm, axis=1)))]

    raise ValueError(f"unknown selection rule {rule!r}")


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap between two boolean supports."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0
