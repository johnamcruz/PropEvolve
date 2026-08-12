"""Training-only balance-aware Regime selectivity targets.

Expansion scores are compared with authenticated fit-only base rates in log-odds
space.  Shrinking MLL headroom subtracts evidence from both Entry actions, and
dominant chop can only increase that subtraction.  WAIT is the zero-evidence
reference action.  Nothing in this module masks or changes inference actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch


EXPANSION_CHANNELS = (
    "long_attempt_probability",
    "long_clean_retained_given_attempt_probability",
    "short_attempt_probability",
    "short_clean_retained_given_attempt_probability",
)
REGIME_STATE_CHANNELS = (
    "structure_chop_probability",
    "structure_neutral_probability",
    "structure_trend_probability",
)
ACTION_ORDER = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1")
SCHEMA = "balance_aware_regime_selectivity_v1"
TARGET_SOURCE = "post_launch_entry_action_target"
FORMULA = (
    "wait_vs_declared_side_softmax(relative_expansion_log_odds"
    "-headroom_pressure*(1-mll_headroom_fraction)"
    "-dominant_chop_pressure*max(0,chop-max(neutral,trend)))"
)


@dataclass(frozen=True)
class BalanceAwareRegimeSelectivity:
    """Compile calibrated Expansion/Regime evidence into soft Entry targets.

    The declared formula for side ``s`` is::

        expansion_s = attempt_s * clean_given_attempt_s
        evidence_s = logit(expansion_s) - logit(fit_only_center_s)
        chop_dominance = max(0, chop - max(neutral, trend))
        pressure = headroom_pressure * (1 - mll_headroom_fraction)
                   + dominant_chop_pressure * chop_dominance
        logits = [0, evidence_declared_side - pressure]
        target[WAIT, declared_side] = softmax(logits)
        target[opposite_side] = 0

    The zero WAIT logit makes centers meaningful without a raw score threshold.
    Regime and balance are downside-only: they may demand stronger Expansion
    evidence, but cannot manufacture an Entry target.
    """

    channel_names: tuple[str, ...] | Sequence[str]
    expansion_centers: tuple[float, float] | Sequence[float]
    probability_epsilon: float = 1e-6
    headroom_pressure: float = 1.0
    dominant_chop_pressure: float = 2.0
    _indices: tuple[int, ...] = field(init=False, repr=False)
    _center_logits: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.channel_names)
        centers = tuple(float(value) for value in self.expansion_centers)
        required = (*EXPANSION_CHANNELS, *REGIME_STATE_CHANNELS)
        if (
            len(names) < len(required)
            or names[: len(required)] != required
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "balance-aware Regime selectivity channel order is invalid"
            )
        if (
            len(centers) != 2
            or not 0.0 < float(self.probability_epsilon) < 0.5
            or any(
                not float(self.probability_epsilon)
                < center
                < 1.0 - float(self.probability_epsilon)
                for center in centers
            )
            or not np.isfinite(self.headroom_pressure)
            or float(self.headroom_pressure) < 0.0
            or not np.isfinite(self.dominant_chop_pressure)
            or float(self.dominant_chop_pressure) < 0.0
        ):
            raise ValueError("balance-aware Regime selectivity contract is invalid")
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "expansion_centers", centers)
        object.__setattr__(
            self,
            "_center_logits",
            tuple(math.log(center / (1.0 - center)) for center in centers),
        )
        object.__setattr__(
            self,
            "_indices",
            tuple(names.index(channel) for channel in required),
        )

    def target_probabilities(
        self,
        teacher_probabilities: torch.Tensor,
        mll_headroom_fraction: torch.Tensor,
        entry_action_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Soften exact bar 1-5 labels toward WAIT without inventing entries."""
        # Numeric ranges are authenticated once when teacher targets and
        # replay rows are ingested. Repeating reductions here would force an
        # MPS-to-CPU synchronization on every optimizer update.
        if (
            teacher_probabilities.ndim < 1
            or teacher_probabilities.shape[-1] != len(self.channel_names)
            or mll_headroom_fraction.shape != teacher_probabilities.shape[:-1]
            or entry_action_targets.shape != teacher_probabilities.shape[:-1]
        ):
            raise ValueError(
                "teacher probabilities or MLL headroom violate the selectivity contract"
            )
        if (
            entry_action_targets.dtype == torch.bool
            or torch.is_floating_point(entry_action_targets)
        ):
            raise ValueError("Regime selectivity requires exact flat-action labels")
        selected = teacher_probabilities[..., list(self._indices)]
        epsilon = float(self.probability_epsilon)
        long_score = (selected[..., 0] * selected[..., 1]).clamp(
            epsilon, 1.0 - epsilon
        )
        short_score = (selected[..., 2] * selected[..., 3]).clamp(
            epsilon, 1.0 - epsilon
        )
        evidence = torch.stack(
            (
                torch.logit(long_score) - self._center_logits[0],
                torch.logit(short_score) - self._center_logits[1],
            ),
            dim=-1,
        )
        chop_dominance = (
            selected[..., 4] - torch.maximum(selected[..., 5], selected[..., 6])
        ).clamp_min(0.0)
        pressure = (
            float(self.headroom_pressure)
            * (1.0 - mll_headroom_fraction).clamp(0.0, 1.0)
            + float(self.dominant_chop_pressure) * chop_dominance
        )
        side_logits = evidence - pressure[..., None]
        pair_probabilities = torch.stack(
            (torch.zeros_like(side_logits), side_logits), dim=-1
        ).softmax(-1)
        wait_rows = entry_action_targets == 0
        long_rows = entry_action_targets == 1
        short_rows = entry_action_targets == 2
        wait_probability = (
            wait_rows.to(teacher_probabilities.dtype)
            + long_rows.to(teacher_probabilities.dtype)
            * pair_probabilities[..., 0, 0]
            + short_rows.to(teacher_probabilities.dtype)
            * pair_probabilities[..., 1, 0]
        )
        return torch.stack(
            (
                wait_probability,
                long_rows.to(teacher_probabilities.dtype)
                * pair_probabilities[..., 0, 1],
                short_rows.to(teacher_probabilities.dtype)
                * pair_probabilities[..., 1, 1],
            ),
            dim=-1,
        )


__all__ = [
    "ACTION_ORDER",
    "BalanceAwareRegimeSelectivity",
    "EXPANSION_CHANNELS",
    "FORMULA",
    "REGIME_STATE_CHANNELS",
    "SCHEMA",
    "TARGET_SOURCE",
]
