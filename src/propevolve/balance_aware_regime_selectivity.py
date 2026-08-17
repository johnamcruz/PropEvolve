"""Training-only balance-aware Regime selectivity targets.

Expansion scores are compared with authenticated fit-only base rates in log-odds
space.  Shrinking MLL headroom subtracts evidence from both Entry actions, and
dominant chop can only increase that subtraction.  WAIT is the zero-evidence
reference action.  Nothing in this module masks or changes inference actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import NamedTuple, Sequence

import numpy as np
import torch


EXPANSION_CHANNELS = (
    "long_attempt_probability",
    "long_clean_retained_given_attempt_probability",
    "short_attempt_probability",
    "short_clean_retained_given_attempt_probability",
)
REGIME_STATE_CHANNELS = (
    "chop_no_trend_probability",
    "chop_end_transition_probability",
    "expansion_trend_probability",
)
REGIME_TRANSITION_CHANNELS = REGIME_STATE_CHANNELS
REGIME_TEACHER_CHANNELS = (
    "chop_no_trend_probability",
    "chop_end_transition_probability",
    "expansion_trend_probability",
)
ACTION_ORDER = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1")
SCHEMA = "balance_aware_regime_selectivity_v1"
TARGET_SOURCE = "post_launch_entry_action_target"
STATIC_STATE_SEMANTICS = "static_state_v1"
PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS = (
    "persistent_chop_negative_weight_v1"
)
PERSISTENT_CHOP_ASSOCIATION_SEMANTICS = "persistent_chop_association_v2"
EXPANSION_REGIME_CONFLUENCE_SEMANTICS = "expansion_regime_confluence_v3"
SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS = (
    "side_conditioned_expansion_regime_confluence_v4"
)
ALL_DOMINANT_CHOP_MARGIN_SEMANTICS = (
    "side_conditioned_expansion_regime_all_dominant_chop_margin_v5"
)
FORMULA = (
    "wait_vs_declared_side_softmax(relative_expansion_log_odds"
    "-headroom_pressure*(1-mll_headroom_fraction)"
    "-dominant_chop_pressure*max(0,chop-max(neutral,trend)))"
)
PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA = (
    "exact_wait*(1+persistent_chop_negative_emphasis*"
    "chop_no_trend_probability)"
)
PERSISTENT_CHOP_ASSOCIATION_FORMULA = (
    "equal_present_group_mean(exact_wait_weighted_ce,exact_long_ce,"
    "exact_short_ce,zero_margin_dead_vs_transition_positive_wait_rank)"
)
EXPANSION_REGIME_CONFLUENCE_FORMULA = (
    "equal_present_group_mean("
    "exact_wait_expansion_regime_confluence_weighted_ce,"
    "exact_long_ce,exact_short_ce,"
    "zero_margin_dead_vs_transition_positive_wait_rank)"
)
SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA = (
    "equal_present_group_mean("
    "exact_wait_expansion_regime_confluence_weighted_ce,"
    "exact_long_ce,exact_short_ce,"
    "dead_vs_transition_positive_wait_rank,"
    "failed_long_vs_valid_long_wait_rank,"
    "failed_short_vs_valid_short_wait_rank)"
)
CHOP_MARGIN_EXPANSION_REGIME_CONFLUENCE_FORMULA = (
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA
    + "+membership_weighted_mean(dominant_chop_wait_margin,"
    "failed_long_wait_margin,"
    "failed_short_wait_margin)"
)
ALL_DOMINANT_CHOP_MARGIN_FORMULA = (
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA
    + "+membership_weighted_mean(all_action_dominant_chop_wait_margin,"
    "failed_long_wait_margin,"
    "failed_short_wait_margin)"
)


class PersistentChopEvidence(NamedTuple):
    """Continuous compiler evidence for loss weighting and mechanism gates."""

    exact_wait_weights: torch.Tensor
    persistent_dead_chop_membership: torch.Tensor
    transition_ready_membership: torch.Tensor
    transition_positive_long_membership: torch.Tensor
    transition_positive_short_membership: torch.Tensor
    failed_setup_confluence_membership: torch.Tensor
    failed_long_confluence_membership: torch.Tensor
    failed_short_confluence_membership: torch.Tensor


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
    semantics: str = STATIC_STATE_SEMANTICS
    persistent_chop_negative_emphasis: float = 1.0
    _indices: tuple[int, ...] = field(init=False, repr=False)
    _transition_indices: tuple[int, ...] = field(init=False, repr=False)
    _center_logits: tuple[float, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.channel_names)
        centers = tuple(float(value) for value in self.expansion_centers)
        semantics = str(self.semantics)
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
            or semantics
            not in (
                STATIC_STATE_SEMANTICS,
                PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            )
            or not np.isfinite(self.persistent_chop_negative_emphasis)
            or float(self.persistent_chop_negative_emphasis) < 0.0
        ):
            raise ValueError("balance-aware Regime selectivity contract is invalid")
        if semantics in {
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        } and any(
            channel not in names for channel in REGIME_TRANSITION_CHANNELS
        ):
            raise ValueError(
                "transition-aware Regime selectivity channels are incomplete"
            )
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "expansion_centers", centers)
        object.__setattr__(self, "semantics", semantics)
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
        object.__setattr__(
            self,
            "_transition_indices",
            tuple(
                names.index(channel)
                for channel in REGIME_TRANSITION_CHANNELS
                if channel in names
            ),
        )

    def dominant_chop_margin_membership(
        self,
        teacher_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Return soft dominance mass for learned WAIT pressure on every action."""
        if self.semantics != ALL_DOMINANT_CHOP_MARGIN_SEMANTICS:
            raise ValueError(
                "all-action dominant-chop margin requires its frozen semantics"
            )
        if (
            teacher_probabilities.ndim < 1
            or teacher_probabilities.shape[-1] != len(self.channel_names)
        ):
            raise ValueError(
                "teacher probabilities violate the dominant-chop margin contract"
            )
        selected = teacher_probabilities[..., list(self._transition_indices)]
        return (
            selected[..., 0]
            - torch.maximum(selected[..., 1], selected[..., 2])
        ).clamp_min(0.0)

    def target_probabilities(
        self,
        teacher_probabilities: torch.Tensor,
        mll_headroom_fraction: torch.Tensor,
        entry_action_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Soften exact bar 1-5 labels toward WAIT without inventing entries."""
        if self.semantics != STATIC_STATE_SEMANTICS:
            raise ValueError(
                "persistent-chop semantics compile exact WAIT negative weights; "
                "they cannot soften positive Entry targets"
            )
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

    def exact_wait_negative_weights(
        self,
        teacher_probabilities: torch.Tensor,
        entry_action_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compile persistent-dead-chop emphasis for exact WAIT rows only.

        The new Expansion-anchored classifier supplies the lifecycle directly:
        class 0 is persistent dead chop, while classes 1 and 2 are transition
        and active Expansion. The result is continuous and bounded in
        ``[1, 1 + persistent_chop_negative_emphasis]`` on exact WAIT rows.
        Long and Short rows receive exactly zero mass and therefore cannot be
        softened or redirected by this compiler. The consumer is responsible
        for normalizing aggregate WAIT class mass.
        """
        evidence = self.exact_wait_negative_weight_evidence(
            teacher_probabilities,
            entry_action_targets,
        )
        return evidence.exact_wait_weights

    def exact_wait_replay_priorities(
        self,
        teacher_probabilities: torch.Tensor,
        entry_action_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded replay mass for hard exact-WAIT confluence rows.

        This reuses the loss compiler's authenticated memberships so replay
        cannot invent a second definition of dominant chop or failed setup.
        Positive Entry targets always receive zero priority.
        """
        evidence = self.exact_wait_negative_weight_evidence(
            teacher_probabilities,
            entry_action_targets,
        )
        return (
            evidence.persistent_dead_chop_membership
            + evidence.failed_setup_confluence_membership
        ).clamp(0.0, 1.0)

    def exact_wait_negative_weight_evidence(
        self,
        teacher_probabilities: torch.Tensor,
        entry_action_targets: torch.Tensor,
    ) -> PersistentChopEvidence:
        """Compile WAIT weights and continuous dead/ready membership masses."""
        if self.semantics not in {
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        }:
            raise ValueError(
                "exact WAIT negative weights require persistent-chop semantics"
            )
        if (
            teacher_probabilities.ndim < 1
            or teacher_probabilities.shape[-1] != len(self.channel_names)
            or entry_action_targets.shape != teacher_probabilities.shape[:-1]
        ):
            raise ValueError(
                "teacher probabilities or Entry labels violate the "
                "transition-aware selectivity contract"
            )
        if (
            entry_action_targets.dtype == torch.bool
            or torch.is_floating_point(entry_action_targets)
        ):
            raise ValueError(
                "transition-aware selectivity requires exact flat-action labels"
            )

        selected = teacher_probabilities[..., list(self._transition_indices)]
        persistent_dead_chop = selected[..., 0].clamp(0.0, 1.0)
        transition_ready_chop = (
            selected[..., 1] + selected[..., 2]
        ).clamp(0.0, 1.0)
        wait_rows = (entry_action_targets == 0).to(teacher_probabilities.dtype)
        long_rows = (entry_action_targets == 1).to(teacher_probabilities.dtype)
        short_rows = (entry_action_targets == 2).to(teacher_probabilities.dtype)
        failed_setup_confluence = torch.zeros_like(persistent_dead_chop)
        failed_long_confluence = torch.zeros_like(persistent_dead_chop)
        failed_short_confluence = torch.zeros_like(persistent_dead_chop)
        transition_positive_long = long_rows * transition_ready_chop
        transition_positive_short = short_rows * transition_ready_chop
        if self.semantics in {
            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        }:
            epsilon = float(self.probability_epsilon)
            long_score = (
                teacher_probabilities[..., self.channel_names.index(
                    "long_attempt_probability"
                )]
                * teacher_probabilities[..., self.channel_names.index(
                    "long_clean_retained_given_attempt_probability"
                )]
            ).clamp(epsilon, 1.0 - epsilon)
            short_score = (
                teacher_probabilities[..., self.channel_names.index(
                    "short_attempt_probability"
                )]
                * teacher_probabilities[..., self.channel_names.index(
                    "short_clean_retained_given_attempt_probability"
                )]
            ).clamp(epsilon, 1.0 - epsilon)
            long_expansion_evidence = torch.sigmoid(
                torch.logit(long_score) - self._center_logits[0]
            )
            short_expansion_evidence = torch.sigmoid(
                torch.logit(short_score) - self._center_logits[1]
            )
            transition_positive_long = (
                transition_positive_long * long_expansion_evidence
            )
            transition_positive_short = (
                transition_positive_short * short_expansion_evidence
            )
            if (
                self.semantics
                in {
                    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                    ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
                }
            ):
                evidence_sum = (
                    long_expansion_evidence + short_expansion_evidence
                ).clamp_min(torch.finfo(teacher_probabilities.dtype).tiny)
                transition_positive_long = (
                    transition_positive_long
                    * long_expansion_evidence
                    / evidence_sum
                )
                transition_positive_short = (
                    transition_positive_short
                    * short_expansion_evidence
                    / evidence_sum
                )
            if self.semantics in {
                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            }:
                failed_setup_confluence = (
                    wait_rows
                    * transition_ready_chop
                    * torch.maximum(
                        long_expansion_evidence,
                        short_expansion_evidence,
                    )
                ).clamp(0.0, 1.0)
                if (
                    self.semantics
                    in {
                        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                        ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
                    }
                ):
                    failed_long_confluence = (
                        failed_setup_confluence
                        * long_expansion_evidence
                        / evidence_sum
                    )
                    failed_short_confluence = (
                        failed_setup_confluence
                        * short_expansion_evidence
                        / evidence_sum
                    )
        wait_emphasis = (
            persistent_dead_chop + failed_setup_confluence
        ).clamp(0.0, 1.0)
        wait_weight = 1.0 + (
            float(self.persistent_chop_negative_emphasis) * wait_emphasis
        )
        return PersistentChopEvidence(
            exact_wait_weights=wait_rows * wait_weight,
            persistent_dead_chop_membership=wait_rows * persistent_dead_chop,
            transition_ready_membership=wait_rows * transition_ready_chop,
            transition_positive_long_membership=transition_positive_long,
            transition_positive_short_membership=transition_positive_short,
            failed_setup_confluence_membership=failed_setup_confluence,
            failed_long_confluence_membership=failed_long_confluence,
            failed_short_confluence_membership=failed_short_confluence,
        )


__all__ = [
    "ACTION_ORDER",
    "ALL_DOMINANT_CHOP_MARGIN_FORMULA",
    "ALL_DOMINANT_CHOP_MARGIN_SEMANTICS",
    "BalanceAwareRegimeSelectivity",
    "EXPANSION_REGIME_CONFLUENCE_FORMULA",
    "EXPANSION_REGIME_CONFLUENCE_SEMANTICS",
    "EXPANSION_CHANNELS",
    "FORMULA",
    "PERSISTENT_CHOP_ASSOCIATION_FORMULA",
    "PERSISTENT_CHOP_ASSOCIATION_SEMANTICS",
    "PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA",
    "PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS",
    "PersistentChopEvidence",
    "REGIME_STATE_CHANNELS",
    "REGIME_TEACHER_CHANNELS",
    "REGIME_TRANSITION_CHANNELS",
    "SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA",
    "SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS",
    "SCHEMA",
    "STATIC_STATE_SEMANTICS",
    "TARGET_SOURCE",
]
