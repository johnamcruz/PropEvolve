from __future__ import annotations

import torch
import pytest

from propevolve.decision import Action
from propevolve.trend_start_confluence import (
    causal_trend_directional_scores,
    trend_start_confluence_rank_loss,
)


def test_trend_start_confluence_respects_economic_label_authority() -> None:
    q_values = torch.zeros((8, 3), dtype=torch.float32, requires_grad=True)
    action_targets = torch.tensor([
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
        int(Action.WAIT),
        int(Action.WAIT),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
        int(Action.WAIT),
        int(Action.WAIT),
    ])
    economic_sides = torch.tensor([
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
    ])
    economic_wins = torch.tensor([
        True, True, False, False, True, True, False, False,
    ])
    # launch Long, launch Short, quality Long, quality Short
    trend_targets = torch.tensor([
        [0.9, 0.1, 0.8, 0.2],  # aligned Long winner
        [0.1, 0.9, 0.2, 0.8],  # aligned Short winner
        [0.1, 0.9, 0.2, 0.8],  # countertrend failed Long
        [0.9, 0.1, 0.8, 0.2],  # countertrend failed Short
        [0.1, 0.9, 0.2, 0.8],  # profitable countertrend Long
        [0.5, 0.5, 0.5, 0.5],  # ambiguous Short winner
        [0.9, 0.1, 0.8, 0.2],  # aligned failed Long remains WAIT
        [0.1, 0.9, 0.2, 0.8],  # aligned failed Short remains WAIT
    ])

    result = trend_start_confluence_rank_loss(
        q_values,
        action_targets=action_targets,
        economic_sides=economic_sides,
        economic_wins=economic_wins,
        directional_scores=torch.stack((
            trend_targets[:, 0] * trend_targets[:, 2],
            trend_targets[:, 1] * trend_targets[:, 3],
        ), dim=-1),
        margin=0.25,
    )

    assert result.aligned_long_winner_rows.item() == 1
    assert result.aligned_short_winner_rows.item() == 1
    assert result.countertrend_long_failure_rows.item() == 1
    assert result.countertrend_short_failure_rows.item() == 1
    assert result.active_rows.item() == 4

    result.loss.backward()
    gradients = q_values.grad
    assert gradients is not None
    assert gradients[0, int(Action.ENTER_LONG_1)] < 0
    assert gradients[0, int(Action.WAIT)] > 0
    assert gradients[1, int(Action.ENTER_SHORT_1)] < 0
    assert gradients[1, int(Action.WAIT)] > 0
    assert gradients[2, int(Action.WAIT)] < 0
    assert gradients[2, int(Action.ENTER_LONG_1)] > 0
    assert gradients[3, int(Action.WAIT)] < 0
    assert gradients[3, int(Action.ENTER_SHORT_1)] > 0
    # Trend may not reverse an economic label or resolve a tie. In particular,
    # an aligned Trend cannot turn an authenticated failure into ENTER.
    assert torch.equal(gradients[4], torch.zeros(3))
    assert torch.equal(gradients[5], torch.zeros(3))
    assert torch.equal(gradients[6], torch.zeros(3))
    assert torch.equal(gradients[7], torch.zeros(3))


def test_trend_start_confluence_rejects_corrupt_economic_pairs() -> None:
    with torch.no_grad(), pytest.raises(ValueError):
        trend_start_confluence_rank_loss(
            torch.zeros((1, 3)),
            action_targets=torch.tensor([int(Action.WAIT)]),
            economic_sides=torch.tensor([int(Action.ENTER_LONG_1)]),
            economic_wins=torch.tensor([True]),
            directional_scores=torch.tensor([[0.72, 0.02]]),
            margin=0.25,
        )


def test_recent_trend_start_confirms_later_established_trend_causally() -> None:
    targets = torch.tensor([[[
        0.9, 0.1, 0.8, 0.2,
    ], [
        0.5, 0.5, 0.5, 0.5,
    ], [
        0.5, 0.5, 0.5, 0.5,
    ]]], dtype=torch.float32)

    scores = causal_trend_directional_scores(
        targets,
        confirmation_lookback_bars=3,
    )

    assert scores[0, 0].tolist() == pytest.approx([0.72, 0.02])
    assert scores[0, 2].tolist() == pytest.approx([0.72, 0.25])

    short_after_long = targets.clone()
    short_after_long[0, 2] = torch.tensor([0.1, 0.95, 0.2, 0.9])
    scores = causal_trend_directional_scores(
        short_after_long,
        confirmation_lookback_bars=3,
    )
    assert scores[0, 2, 1] > scores[0, 2, 0]

    padded = torch.cat((
        targets,
        torch.full((1, 1, 4), torch.nan),
    ), dim=1)
    scores = causal_trend_directional_scores(
        padded,
        confirmation_lookback_bars=3,
    )
    assert scores[0, 3].tolist() == pytest.approx([0.25, 0.25])

    corrupt = padded.clone()
    corrupt[0, 3, 0] = 0.5
    with pytest.raises(ValueError):
        causal_trend_directional_scores(
            corrupt,
            confirmation_lookback_bars=3,
        )
