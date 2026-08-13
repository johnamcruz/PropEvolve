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
    chop_persistence: float,
    trend_onset: float = 0.0,
    trend_persistence: float = 0.0,
    volatility_expansion_onset: float = 0.0,
    volatility_high_persistence: float = 0.0,
    kaufman_efficiency: float = 0.0,
    volatility_percentile: float = 0.0,
) -> torch.Tensor:
    values = np.full(len(CHANNELS), 0.1, dtype=np.float32)
    updates = {
        "structure_chop_persistence_probability": chop_persistence,
        "structure_trend_onset_probability": trend_onset,
        "structure_trend_persistence_probability": trend_persistence,
        "volatility_expansion_onset_probability": volatility_expansion_onset,
        "volatility_high_persistence_probability": volatility_high_persistence,
        "kaufman_efficiency": kaufman_efficiency,
        "volatility_percentile": volatility_percentile,
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


def test_persistent_dead_chop_raises_exact_wait_emphasis() -> None:
    transient_chop = _teacher_row(chop_persistence=0.10)
    persistent_dead_chop = _teacher_row(chop_persistence=0.90)

    assert _wait_weight(persistent_dead_chop) > _wait_weight(transient_chop)


def test_transition_ready_compression_relieves_persistent_chop_emphasis() -> None:
    dead_chop = _teacher_row(chop_persistence=0.90)
    transition_ready = _teacher_row(
        chop_persistence=0.90,
        trend_onset=0.80,
        trend_persistence=0.60,
        volatility_expansion_onset=0.90,
        volatility_high_persistence=0.70,
        kaufman_efficiency=0.85,
        volatility_percentile=0.80,
    )

    assert _wait_weight(transition_ready) < _wait_weight(dead_chop)
    assert _wait_weight(transition_ready) >= 1.0


def test_volatility_alone_without_efficiency_cannot_erase_chop_emphasis() -> None:
    dead_chop = _teacher_row(chop_persistence=0.90)
    volatile_but_inefficient = _teacher_row(
        chop_persistence=0.90,
        volatility_expansion_onset=1.0,
        volatility_high_persistence=1.0,
        kaufman_efficiency=0.0,
        volatility_percentile=1.0,
    )

    torch.testing.assert_close(
        _wait_weight(volatile_but_inefficient),
        _wait_weight(dead_chop),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "channel",
    (
        "structure_trend_onset_probability",
        "structure_trend_persistence_probability",
        "volatility_expansion_onset_probability",
        "volatility_high_persistence_probability",
        "volatility_percentile",
    ),
)
def test_each_transition_channel_can_relieve_persistent_chop(
    channel: str,
) -> None:
    dead_chop = _teacher_row(
        chop_persistence=0.90,
        kaufman_efficiency=0.80,
    )
    values = {
        "chop_persistence": 0.90,
        "kaufman_efficiency": 0.80,
    }
    argument = {
        "structure_trend_onset_probability": "trend_onset",
        "structure_trend_persistence_probability": "trend_persistence",
        "volatility_expansion_onset_probability": "volatility_expansion_onset",
        "volatility_high_persistence_probability": "volatility_high_persistence",
        "volatility_percentile": "volatility_percentile",
    }[channel]
    values[argument] = 1.0
    transition = _teacher_row(**values)

    assert _wait_weight(transition) < _wait_weight(dead_chop)


def test_kaufman_efficiency_gates_transition_relief() -> None:
    inefficient = _teacher_row(
        chop_persistence=0.90,
        trend_onset=0.90,
        volatility_expansion_onset=0.90,
        kaufman_efficiency=0.10,
    )
    efficient = _teacher_row(
        chop_persistence=0.90,
        trend_onset=0.90,
        volatility_expansion_onset=0.90,
        kaufman_efficiency=0.90,
    )

    assert _wait_weight(efficient) < _wait_weight(inefficient)


def test_only_exact_wait_rows_receive_negative_emphasis_for_both_sides() -> None:
    rows = torch.stack(
        (
            _teacher_row(chop_persistence=0.90),
            _teacher_row(chop_persistence=0.90),
            _teacher_row(chop_persistence=0.90),
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


def test_transition_mode_cannot_be_misused_to_soften_positive_targets() -> None:
    selectivity = _selectivity()

    with pytest.raises(ValueError, match="exact WAIT negative weights"):
        selectivity.target_probabilities(
            _teacher_row(chop_persistence=0.90)[None],
            torch.tensor([0.5]),
            torch.tensor([int(Action.ENTER_LONG_1)]),
        )


def test_transition_compilation_does_not_synchronize_through_tensor_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_item(*args, **kwargs):
        raise AssertionError("hot-loop tensor.item synchronization is forbidden")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    weights = _selectivity().exact_wait_negative_weights(
        _teacher_row(
            chop_persistence=0.90,
            trend_onset=0.80,
            volatility_expansion_onset=0.80,
            kaufman_efficiency=0.80,
            volatility_percentile=0.80,
        )[None],
        torch.tensor([int(Action.WAIT)]),
    )

    assert weights.shape == (1,)
