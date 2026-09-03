from __future__ import annotations

import numpy as np
import pytest

from propevolve.causal_separability import (
    canonical_side_context,
    temporal_ridge_separability,
)
from propevolve.decision import Action


def test_canonical_side_context_mirrors_short_into_shared_direction_space() -> None:
    context = np.asarray(
        (0.9, 0.8, 0.2, 0.1, 0.3, 0.6, 0.1),
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        canonical_side_context(context, Action.ENTER_LONG_1),
        context,
    )
    np.testing.assert_array_equal(
        canonical_side_context(context, Action.ENTER_SHORT_1),
        context[[2, 3, 0, 1, 4, 5, 6]],
    )


def test_temporal_probe_detects_causal_holdout_separability() -> None:
    labels = np.tile(np.asarray((0, 1), dtype=np.int8), 40)
    features = np.column_stack((
        labels.astype(np.float64),
        1.0 - labels.astype(np.float64),
    ))

    result = temporal_ridge_separability(
        features,
        labels,
        np.arange(len(labels), dtype=np.int64),
    )

    assert result.status == "ok"
    assert result.roc_auc == pytest.approx(1.0)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.train_positive_rows > 0
    assert result.test_positive_rows > 0


def test_temporal_probe_reports_indistinguishable_causal_rows() -> None:
    labels = np.tile(np.asarray((0, 1), dtype=np.int8), 40)
    features = np.ones((len(labels), 3), dtype=np.float64)

    result = temporal_ridge_separability(
        features,
        labels,
        np.arange(len(labels), dtype=np.int64),
    )

    assert result.status == "ok"
    assert result.roc_auc == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("features", "labels", "timestamps"),
    (
        (np.ones((4, 2)), np.asarray((0, 1, 0)), np.arange(4)),
        (np.asarray(((0.0,), (np.nan,))), np.asarray((0, 1)), np.arange(2)),
        (np.ones((4, 2)), np.asarray((0, 1, 2, 0)), np.arange(4)),
    ),
)
def test_temporal_probe_rejects_invalid_audit_rows(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="causal separability"):
        temporal_ridge_separability(features, labels, timestamps)
