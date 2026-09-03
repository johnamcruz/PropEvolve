"""Training-only audits for causal winner/failure representation quality."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .decision import Action


@dataclass(frozen=True)
class TemporalSeparabilityResult:
    status: str
    rows: int
    feature_count: int
    train_rows: int
    test_rows: int
    train_positive_rows: int
    test_positive_rows: int
    roc_auc: float | None
    balanced_accuracy: float | None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def canonical_side_context(
    context: np.ndarray,
    side: Action,
) -> np.ndarray:
    """Mirror Short evidence into the same causal coordinate system as Long."""
    values = np.asarray(context, dtype=np.float32)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("causal separability context is invalid")
    if side == Action.ENTER_LONG_1:
        return values
    if side == Action.ENTER_SHORT_1:
        return values[[2, 3, 0, 1, 4, 5, 6]]
    raise ValueError("causal separability side is invalid")


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = len(labels) - positive_count
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def temporal_ridge_separability(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    train_fraction: float = 0.7,
    ridge: float = 1e-3,
) -> TemporalSeparabilityResult:
    """Fit on older rows and score newer rows without contaminating the audit."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    ended_at = np.asarray(timestamps)
    if (
        x.ndim != 2
        or y.ndim != 1
        or ended_at.ndim != 1
        or len(x) != len(y)
        or len(x) != len(ended_at)
        or len(x) < 4
        or x.shape[1] < 1
        or not np.isfinite(x).all()
        or not np.isfinite(ended_at).all()
        or not np.isin(y, (0, 1)).all()
        or isinstance(train_fraction, bool)
        or not np.isfinite(train_fraction)
        or not 0.5 <= train_fraction < 1.0
        or isinstance(ridge, bool)
        or not np.isfinite(ridge)
        or ridge <= 0.0
    ):
        raise ValueError("causal separability audit rows are invalid")
    y = y.astype(np.int8, copy=False)
    order = np.argsort(ended_at, kind="stable")
    x = x[order]
    y = y[order]
    target_split = int(round(len(y) * train_fraction))
    candidates = [
        split
        for split in range(2, len(y) - 1)
        if np.unique(y[:split]).size == 2
        and np.unique(y[split:]).size == 2
        and ended_at[order][split - 1] < ended_at[order][split]
    ]
    if not candidates:
        return TemporalSeparabilityResult(
            status="insufficient_temporal_class_coverage",
            rows=len(y),
            feature_count=x.shape[1],
            train_rows=0,
            test_rows=0,
            train_positive_rows=0,
            test_positive_rows=0,
            roc_auc=None,
            balanced_accuracy=None,
        )
    split = min(candidates, key=lambda value: (abs(value - target_split), value))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train_design = np.column_stack((
        np.ones(len(train_x), dtype=np.float64),
        (train_x - mean) / scale,
    ))
    test_design = np.column_stack((
        np.ones(len(test_x), dtype=np.float64),
        (test_x - mean) / scale,
    ))
    penalty = np.eye(train_design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_y.astype(np.float64),
    )
    scores = test_design @ coefficients
    predicted = scores >= 0.5
    true_positive_rate = float(predicted[test_y == 1].mean())
    true_negative_rate = float((~predicted[test_y == 0]).mean())
    return TemporalSeparabilityResult(
        status="ok",
        rows=len(y),
        feature_count=x.shape[1],
        train_rows=len(train_y),
        test_rows=len(test_y),
        train_positive_rows=int(train_y.sum()),
        test_positive_rows=int(test_y.sum()),
        roc_auc=_binary_auc(test_y, scores),
        balanced_accuracy=0.5 * (true_positive_rate + true_negative_rate),
    )
