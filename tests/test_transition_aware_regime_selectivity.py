from __future__ import annotations

import numpy as np
import pytest
import torch

from propevolve.balance_aware_regime_selectivity import (
    BalanceAwareRegimeSelectivity,
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
)
from propevolve.decision import Action
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS


CHANNELS = (*EXPANSION_CHANNELS, *REGIME_CHANNELS)


def _teacher_row(
    *,
    chop_no_trend: float,
    chop_end_transition: float = 0.0,
    expansion_trend: float = 0.0,
) -> torch.Tensor:
    values = np.full(len(CHANNELS), 0.1, dtype=np.float32)
    updates = {
        "chop_no_trend_probability": chop_no_trend,
        "chop_end_transition_probability": chop_end_transition,
        "expansion_trend_probability": expansion_trend,
    }
    for channel, value in updates.items():
        values[CHANNELS.index(channel)] = value
    return torch.from_numpy(values)


def _selectivity() -> BalanceAwareRegimeSelectivity:
    return BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    )


def _wait_weight(row: torch.Tensor) -> torch.Tensor:
    return _selectivity().exact_wait_negative_weights(
        row[None], torch.tensor([int(Action.WAIT)])
    )[0]


def test_direct_chop_no_trend_probability_raises_exact_wait_emphasis() -> None:
    transient_chop = _teacher_row(chop_no_trend=0.10)
    persistent_dead_chop = _teacher_row(chop_no_trend=0.90)

    assert _wait_weight(persistent_dead_chop) > _wait_weight(transient_chop)


def test_transition_and_expansion_are_direct_ready_evidence() -> None:
    rows = torch.stack((
        _teacher_row(chop_no_trend=0.90),
        _teacher_row(chop_no_trend=0.10, chop_end_transition=0.80),
        _teacher_row(chop_no_trend=0.10, expansion_trend=0.80),
    ))
    evidence = _selectivity().exact_wait_negative_weight_evidence(
        rows,
        torch.tensor([int(Action.WAIT)] * 3),
    )

    assert evidence.persistent_dead_chop_membership.tolist() == pytest.approx(
        [0.90, 0.10, 0.10]
    )
    assert evidence.transition_ready_membership.tolist() == pytest.approx(
        [0.0, 0.80, 0.80]
    )


def test_only_exact_wait_rows_receive_negative_emphasis_for_both_sides() -> None:
    rows = torch.stack(
        (
            _teacher_row(chop_no_trend=0.90),
            _teacher_row(chop_no_trend=0.90),
            _teacher_row(chop_no_trend=0.90),
        )
    )
    weights = _selectivity().exact_wait_negative_weights(
        rows,
        torch.tensor(
            [
                int(Action.WAIT),
                int(Action.ENTER_LONG_1),
                int(Action.ENTER_SHORT_1),
            ]
        ),
    )

    assert weights[0] > 1.0
    assert weights[1:].tolist() == [0.0, 0.0]


def test_three_state_mode_cannot_be_misused_to_soften_positive_targets() -> None:
    selectivity = _selectivity()

    with pytest.raises(ValueError, match="exact WAIT negative weights"):
        selectivity.target_probabilities(
            _teacher_row(chop_no_trend=0.90)[None],
            torch.tensor([0.5]),
            torch.tensor([int(Action.ENTER_LONG_1)]),
        )


def test_three_state_compilation_does_not_synchronize_through_tensor_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_item(*args, **kwargs):
        raise AssertionError("hot-loop tensor.item synchronization is forbidden")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    weights = _selectivity().exact_wait_negative_weights(
        _teacher_row(
            chop_no_trend=0.10,
            chop_end_transition=0.80,
            expansion_trend=0.10,
        )[None],
        torch.tensor([int(Action.WAIT)]),
    )

    assert weights.shape == (1,)
