"""Training-only Trend Start confluence for authenticated economic labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decision import Action


TREND_START_CONFLUENCE_SCHEMA = "trend_start_confluence_v1"
TREND_START_CONFLUENCE_SEMANTICS = (
    "economic_label_authority_causal_directional_confirmation_v1"
)
TREND_START_SCORE_FORMULA = (
    "causal_recent_max_launch_probability_times_conditional_quality"
)
TREND_START_CHANNELS = (
    "long_launch_probability",
    "short_launch_probability",
    "long_conditional_quality",
    "short_conditional_quality",
)


@dataclass(frozen=True)
class TrendStartConfluenceResult:
    loss: torch.Tensor
    opportunity_loss: torch.Tensor
    safety_loss: torch.Tensor
    active_rows: torch.Tensor
    aligned_long_winner_rows: torch.Tensor
    aligned_short_winner_rows: torch.Tensor
    countertrend_long_failure_rows: torch.Tensor
    countertrend_short_failure_rows: torch.Tensor
    dominance_mass: torch.Tensor


def causal_trend_directional_scores(
    trend_targets: torch.Tensor,
    *,
    confirmation_lookback_bars: int,
) -> torch.Tensor:
    """Carry start evidence causally into a bounded established-trend window."""
    finite = torch.isfinite(trend_targets)
    complete_rows = finite.all(dim=-1)
    missing_rows = ~finite.any(dim=-1)
    if (
        trend_targets.ndim != 3
        or trend_targets.shape[-1] != 4
        or isinstance(confirmation_lookback_bars, bool)
        or int(confirmation_lookback_bars) < 1
        or not bool((complete_rows | missing_rows).all().item())
        or bool((
            complete_rows[..., None]
            & ((trend_targets < 0.0) | (trend_targets > 1.0))
        ).any().item())
    ):
        raise ValueError("Trend Start confirmation inputs are invalid")
    trend_targets = torch.where(
        complete_rows[..., None], trend_targets, torch.zeros_like(trend_targets)
    )
    current = torch.stack((
        trend_targets[..., 0] * trend_targets[..., 2],
        trend_targets[..., 1] * trend_targets[..., 3],
    ), dim=-1)
    history = []
    for lag in range(min(int(confirmation_lookback_bars), current.shape[1])):
        shifted = torch.zeros_like(current)
        if lag == 0:
            shifted = current
        else:
            shifted[:, lag:] = current[:, :-lag]
        history.append(shifted)
    return torch.stack(history, dim=0).amax(dim=0)


def trend_start_confluence_rank_loss(
    q_values: torch.Tensor,
    *,
    action_targets: torch.Tensor,
    economic_sides: torch.Tensor,
    economic_wins: torch.Tensor,
    directional_scores: torch.Tensor,
    margin: float,
) -> TrendStartConfluenceResult:
    """Weight existing economic margins by continuous Trend side dominance.

    Trend never creates or reverses an action target. It only reinforces an
    authenticated winner when Trend agrees, or an authenticated failed side
    when Trend favors the opposite side. Exact ties add no pressure.
    """
    if (
        q_values.ndim != 2
        or q_values.shape[-1] != 3
        or action_targets.shape != q_values.shape[:1]
        or economic_sides.shape != q_values.shape[:1]
        or economic_wins.shape != q_values.shape[:1]
        or directional_scores.shape != (*q_values.shape[:1], 2)
        or isinstance(margin, bool)
        or not torch.isfinite(torch.tensor(float(margin)))
        or float(margin) < 0.0
        or not torch.isfinite(q_values).all()
        or not torch.isfinite(directional_scores).all()
        or bool(((directional_scores < 0.0) | (directional_scores > 1.0)).any().item())
    ):
        raise ValueError("Trend Start confluence inputs are invalid")

    wait = int(Action.WAIT)
    long = int(Action.ENTER_LONG_1)
    short = int(Action.ENTER_SHORT_1)
    valid_sides = (economic_sides == long) | (economic_sides == short)
    valid_targets = (
        (economic_wins & (action_targets == economic_sides))
        | (~economic_wins & (action_targets == wait))
    )
    if not bool((valid_sides & valid_targets).all().item()):
        raise ValueError("Trend Start confluence economic evidence is invalid")

    long_score = directional_scores[:, 0]
    short_score = directional_scores[:, 1]
    long_dominance = (long_score - short_score).clamp_min(0.0)
    short_dominance = (short_score - long_score).clamp_min(0.0)

    aligned_long_winner = economic_wins & (economic_sides == long) & (
        long_dominance > 0.0
    )
    aligned_short_winner = economic_wins & (economic_sides == short) & (
        short_dominance > 0.0
    )
    countertrend_long_failure = ~economic_wins & (economic_sides == long) & (
        short_dominance > 0.0
    )
    countertrend_short_failure = ~economic_wins & (economic_sides == short) & (
        long_dominance > 0.0
    )

    opportunity_weights = (
        aligned_long_winner.to(q_values.dtype) * long_dominance
        + aligned_short_winner.to(q_values.dtype) * short_dominance
    )
    safety_weights = (
        countertrend_long_failure.to(q_values.dtype) * short_dominance
        + countertrend_short_failure.to(q_values.dtype) * long_dominance
    )

    long_winner_headroom = q_values[:, long] - torch.maximum(
        q_values[:, wait], q_values[:, short]
    )
    short_winner_headroom = q_values[:, short] - torch.maximum(
        q_values[:, wait], q_values[:, long]
    )
    winner_headroom = torch.where(
        economic_sides == long,
        long_winner_headroom,
        short_winner_headroom,
    )
    failed_side_q = torch.where(
        economic_sides == long,
        q_values[:, long],
        q_values[:, short],
    )
    failure_headroom = q_values[:, wait] - failed_side_q

    opportunity_loss = (
        nn.functional.softplus(float(margin) - winner_headroom)
        * opportunity_weights
    ).sum() / opportunity_weights.sum().clamp_min(1.0)
    safety_loss = (
        nn.functional.softplus(float(margin) - failure_headroom)
        * safety_weights
    ).sum() / safety_weights.sum().clamp_min(1.0)
    active_rows = (opportunity_weights > 0.0).sum() + (
        safety_weights > 0.0
    ).sum()
    active_groups = (opportunity_weights.sum() > 0.0).to(q_values.dtype) + (
        safety_weights.sum() > 0.0
    ).to(q_values.dtype)
    loss = (opportunity_loss + safety_loss) / active_groups.clamp_min(1.0)
    return TrendStartConfluenceResult(
        loss=loss,
        opportunity_loss=opportunity_loss,
        safety_loss=safety_loss,
        active_rows=active_rows.to(torch.float32),
        aligned_long_winner_rows=aligned_long_winner.sum().to(torch.float32),
        aligned_short_winner_rows=aligned_short_winner.sum().to(torch.float32),
        countertrend_long_failure_rows=(
            countertrend_long_failure.sum().to(torch.float32)
        ),
        countertrend_short_failure_rows=(
            countertrend_short_failure.sum().to(torch.float32)
        ),
        dominance_mass=(opportunity_weights + safety_weights).sum(),
    )


__all__ = [
    "TREND_START_CONFLUENCE_SCHEMA",
    "TREND_START_CONFLUENCE_SEMANTICS",
    "TREND_START_SCORE_FORMULA",
    "TREND_START_CHANNELS",
    "TrendStartConfluenceResult",
    "causal_trend_directional_scores",
    "trend_start_confluence_rank_loss",
]
