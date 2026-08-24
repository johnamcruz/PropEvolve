from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.balance_aware_regime_selectivity import (
    BalanceAwareRegimeSelectivity,
    ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
    PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
    PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    STATIC_STATE_SEMANTICS,
)
from propevolve.agent import (
    RecurrentC51Agent,
    exact_action_margin_losses,
    paired_a_plus_rank_loss,
    paired_recurrent_a_plus_rank_loss,
    side_conditioned_wait_rank_loss,
    chop_specific_wait_margin_losses,
)
from propevolve.decision import Action
from propevolve.config import materialize_effective_config
from propevolve.replay import Transition
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS
from propevolve.teachers.trend import CHANNELS as TREND_CHANNELS
from propevolve.training import (
    _bounded_regime_selectivity_headroom,
    _regime_selectivity_episode_diagnostic,
    _regime_selectivity_evaluation_metrics,
    _regime_selectivity_frozen_contract,
    _training_resume_identity,
    _training_evaluation_gates,
    _write_training_diagnostic_summary,
)


CHANNELS = (*EXPANSION_CHANNELS, *REGIME_CHANNELS)


def test_all_dominant_chop_actions_receive_learned_wait_margin_membership(
) -> None:
    compiler = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.1, 0.1),
        semantics=ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
    )
    teacher_probabilities = torch.tensor(
        [
            [0.4, 0.8, 0.1, 0.8, 0.70, 0.20, 0.10],
            [0.1, 0.8, 0.4, 0.8, 0.60, 0.25, 0.15],
            [0.1, 0.1, 0.1, 0.1, 0.80, 0.10, 0.10],
            [0.4, 0.8, 0.1, 0.8, 0.10, 0.55, 0.35],
        ],
        dtype=torch.float32,
    )
    economic_targets = torch.tensor([1, 2, 0, 1], dtype=torch.long)

    membership = compiler.dominant_chop_margin_membership(
        teacher_probabilities,
    )

    assert economic_targets.tolist() == [1, 2, 0, 1]
    torch.testing.assert_close(
        membership,
        torch.tensor([0.50, 0.35, 0.70, 0.0], dtype=torch.float32),
        rtol=0.0,
        atol=1e-7,
    )


def test_exact_action_margin_penalizes_correct_but_weak_action_separation() -> None:
    action_values = torch.tensor(
        [
            [1.00, 0.90, 0.00],
            [0.80, 1.00, 0.90],
            [0.90, 0.80, 1.00],
        ],
        dtype=torch.float64,
    )

    losses = exact_action_margin_losses(
        action_values,
        torch.tensor([0, 1, 2]),
        margin=0.25,
    )

    torch.testing.assert_close(
        losses,
        torch.tensor([0.15, 0.15, 0.15], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("target", (0, 1, 2))
def test_exact_action_margin_is_zero_after_each_action_clears_the_gap(
    target: int,
) -> None:
    action_values = torch.zeros((1, 3), dtype=torch.float64)
    action_values[0, target] = 0.25

    losses = exact_action_margin_losses(
        action_values,
        torch.tensor([target]),
        margin=0.25,
    )

    torch.testing.assert_close(
        losses,
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_zero_exact_action_margin_preserves_legacy_loss() -> None:
    action_values = torch.tensor([[0.7, 0.6, 0.5]], dtype=torch.float64)

    losses = exact_action_margin_losses(
        action_values,
        torch.tensor([2]),
        margin=0.0,
    )

    torch.testing.assert_close(
        losses,
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("margin", (-0.1, float("nan"), True))
def test_agent_rejects_invalid_exact_action_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="teacher settings must be nonnegative"):
        _agent(
            seed=509,
            selectivity_weight=0.0,
            entry_action_margin=margin,
        )


def test_side_conditioned_rank_loss_separates_wait_long_and_wait_short_gradients(
) -> None:
    action_values = torch.zeros((4, 3), dtype=torch.float64, requires_grad=True)
    loss, active_sides = side_conditioned_wait_rank_loss(
        action_values,
        failed_long_membership=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        valid_long_membership=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        failed_short_membership=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        valid_short_membership=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        q_temperature=1.0,
    )

    gradient, = torch.autograd.grad(loss, action_values)

    assert active_sides.item() == 2.0
    torch.testing.assert_close(
        gradient,
        torch.tensor([
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.0, 0.5],
            [0.5, 0.0, -0.5],
        ], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_paired_a_plus_rank_loss_requires_good_long_to_outrank_matched_failure(
) -> None:
    action_values = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -0.5, 0.0],
        ],
        dtype=torch.float64,
    )

    result = paired_a_plus_rank_loss(
        action_values,
        failed_long_membership=torch.tensor([0.0, 1.0]),
        failed_short_membership=torch.zeros(2),
        valid_long_membership=torch.tensor([1.0, 0.0]),
        valid_short_membership=torch.zeros(2),
        regime_probabilities=torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        margin=0.25,
    )

    assert result.active_groups.item() == 1.0
    assert result.pair_count.item() == 1.0
    assert result.pair_mass.item() == 1.0
    assert result.good_advantage_sum.item() == 1.0
    assert result.bad_advantage_sum.item() == -0.5
    assert result.loss.item() == pytest.approx(0.2519290813453729)


def test_paired_recurrent_a_plus_uses_only_explicit_economic_pairs() -> None:
    action_values = torch.tensor(
        [
            [0.0, 1.0, 0.0],   # Long economic winner.
            [0.0, -0.5, 0.0],  # Matched Long economic failure.
            [0.0, 0.0, 0.8],   # Short economic winner.
            [0.0, 0.0, -0.2],  # Matched Short economic failure.
            [0.0, 9.0, 9.0],   # Unpaired row must not affect this loss.
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = paired_recurrent_a_plus_rank_loss(
        action_values,
        pair_ids=torch.tensor([10, 10, 11, 11, -1]),
        pair_sides=torch.tensor([1, 1, 2, 2, -1]),
        economic_wins=torch.tensor([True, False, True, False, False]),
        population_weights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
        margin=0.25,
        action_margin=0.25,
    )
    gradient, = torch.autograd.grad(result.loss, action_values)

    assert result.pair_count.item() == 2.0
    assert result.good_advantage_sum.item() == pytest.approx(1.8)
    assert result.bad_advantage_sum.item() == pytest.approx(-0.7)
    assert gradient[4].abs().sum().item() == 0.0
    assert gradient[0, Action.ENTER_LONG_1] < 0.0
    assert gradient[1, Action.ENTER_LONG_1] > 0.0
    assert gradient[2, Action.ENTER_SHORT_1] < 0.0
    assert gradient[3, Action.ENTER_SHORT_1] > 0.0


def test_paired_recurrent_a_plus_anchors_winner_and_failure_absolutely() -> None:
    action_values = torch.tensor(
        [
            [0.0, 0.38, 0.0],  # Winner advantage already exceeds failure.
            [0.0, 0.09, 0.0],  # Failure still incorrectly prefers ENTER.
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = paired_recurrent_a_plus_rank_loss(
        action_values,
        pair_ids=torch.tensor([7, 7]),
        pair_sides=torch.tensor([1, 1]),
        economic_wins=torch.tensor([True, False]),
        population_weights=torch.ones(2, dtype=torch.float64),
        margin=0.25,
        action_margin=0.25,
    )
    gradient, = torch.autograd.grad(result.loss, action_values)

    # The worked three-term mean is:
    # mean(softplus(-0.04), softplus(-0.13), softplus(0.34)).
    assert result.loss.item() == pytest.approx(0.7270442627858384)
    assert gradient[0, Action.ENTER_LONG_1] < 0.0
    assert gradient[0, Action.WAIT] > 0.0
    assert gradient[1, Action.ENTER_LONG_1] > 0.0
    assert gradient[1, Action.WAIT] < 0.0


def test_paired_recurrent_a_plus_corrects_balanced_pairs_to_population_prior(
) -> None:
    action_values = torch.tensor(
        [
            [0.0, 0.10, 0.0],  # Winner still lacks the required margin.
            [0.0, 0.10, 0.0],  # Failure still incorrectly prefers ENTER.
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    uncorrected_values = action_values.detach().clone().requires_grad_(True)
    uncorrected = paired_recurrent_a_plus_rank_loss(
        uncorrected_values,
        pair_ids=torch.tensor([7, 7]),
        pair_sides=torch.tensor([1, 1]),
        economic_wins=torch.tensor([True, False]),
        population_weights=torch.ones(2, dtype=torch.float64),
        margin=0.25,
        action_margin=0.25,
    )
    uncorrected_gradient, = torch.autograd.grad(
        uncorrected.loss, uncorrected_values
    )

    result = paired_recurrent_a_plus_rank_loss(
        action_values,
        pair_ids=torch.tensor([7, 7]),
        pair_sides=torch.tensor([1, 1]),
        economic_wins=torch.tensor([True, False]),
        population_weights=torch.tensor([0.4, 1.6], dtype=torch.float64),
        margin=0.25,
        action_margin=0.25,
    )
    gradient, = torch.autograd.grad(result.loss, action_values)

    relative = torch.nn.functional.softplus(torch.tensor(0.25, dtype=torch.float64))
    winner = torch.nn.functional.softplus(torch.tensor(0.15, dtype=torch.float64))
    failure = torch.nn.functional.softplus(torch.tensor(0.35, dtype=torch.float64))
    expected = (relative + 0.4 * winner + 1.6 * failure) / 3.0
    assert result.loss.item() == pytest.approx(expected.item())
    # The naturally more common failures exert stronger absolute WAIT pressure
    # than the oversampled winners exert ENTER pressure.
    assert gradient[1, Action.ENTER_LONG_1] > -gradient[0, Action.ENTER_LONG_1]
    assert gradient[1, Action.ENTER_LONG_1] > (
        uncorrected_gradient[1, Action.ENTER_LONG_1]
    )
    assert -gradient[0, Action.ENTER_LONG_1] < (
        -uncorrected_gradient[0, Action.ENTER_LONG_1]
    )


def test_population_corrected_pair_loss_teaches_long_and_short_winner_boundaries(
) -> None:
    action_values = torch.tensor(
        [
            [0.0, 0.10, 0.0],  # Long winner.
            [0.0, 0.10, 0.0],  # Matched Long failure.
            [0.0, 0.0, 0.10],  # Short winner.
            [0.0, 0.0, 0.10],  # Matched Short failure.
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = paired_recurrent_a_plus_rank_loss(
        action_values,
        pair_ids=torch.tensor([7, 7, 8, 8]),
        pair_sides=torch.tensor([1, 1, 2, 2]),
        economic_wins=torch.tensor([True, False, True, False]),
        population_weights=torch.tensor(
            [0.4, 1.6, 0.35, 1.65], dtype=torch.float64
        ),
        margin=0.25,
        action_margin=0.25,
    )
    gradient, = torch.autograd.grad(result.loss, action_values)

    assert result.pair_count.item() == 2.0
    assert gradient[0, Action.ENTER_LONG_1] < 0.0
    assert gradient[1, Action.ENTER_LONG_1] > 0.0
    assert gradient[2, Action.ENTER_SHORT_1] < 0.0
    assert gradient[3, Action.ENTER_SHORT_1] > 0.0
    assert result.group_metrics[
        "long_failure_population_weight_sum"
    ].item() == pytest.approx(1.6)
    assert result.group_metrics[
        "short_failure_population_weight_sum"
    ].item() == pytest.approx(1.65)


@pytest.mark.parametrize(
    "population_weights",
    (
        torch.tensor([0.0, 2.0]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([float("nan"), 1.0]),
    ),
)
def test_paired_recurrent_a_plus_rejects_invalid_population_correction(
    population_weights: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="population weights"):
        paired_recurrent_a_plus_rank_loss(
            torch.zeros((2, 3), dtype=torch.float32),
            pair_ids=torch.tensor([7, 7]),
            pair_sides=torch.tensor([1, 1]),
            economic_wins=torch.tensor([True, False]),
            population_weights=population_weights,
            margin=0.25,
            action_margin=0.25,
        )


def test_paired_economic_and_exact_action_losses_cover_full_decision_boundary(
) -> None:
    action_values = torch.tensor(
        [
            # The opposite entry is strongest on each winner. Both entries
            # incorrectly beat WAIT on each failure.
            [0.00, 0.38, 0.80],  # Long winner.
            [0.00, 0.09, 0.60],  # Long failure.
            [0.00, 0.70, 0.38],  # Short winner.
            [0.00, 0.60, 0.09],  # Short failure.
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([
        int(Action.ENTER_LONG_1),
        int(Action.WAIT),
        int(Action.ENTER_SHORT_1),
        int(Action.WAIT),
    ])

    paired = paired_recurrent_a_plus_rank_loss(
        action_values,
        pair_ids=torch.tensor([10, 10, 11, 11]),
        pair_sides=torch.tensor([1, 1, 2, 2]),
        economic_wins=torch.tensor([True, False, True, False]),
        population_weights=torch.ones(4, dtype=torch.float64),
        margin=0.25,
        action_margin=0.25,
    )
    exact = exact_action_margin_losses(
        action_values,
        targets,
        margin=0.25,
    )
    gradient, = torch.autograd.grad(paired.loss + exact.mean(), action_values)

    # Gradient descent raises the correct entry and suppresses WAIT plus the
    # opposite side for both winner families.
    assert gradient[0, Action.ENTER_LONG_1] < 0.0
    assert gradient[0, Action.WAIT] > 0.0
    assert gradient[0, Action.ENTER_SHORT_1] > 0.0
    assert gradient[2, Action.ENTER_SHORT_1] < 0.0
    assert gradient[2, Action.WAIT] > 0.0
    assert gradient[2, Action.ENTER_LONG_1] > 0.0
    # Gradient descent raises WAIT and suppresses both entries for failures,
    # including the non-declared side that the paired loss does not address.
    assert gradient[1, Action.WAIT] < 0.0
    assert gradient[1, Action.ENTER_LONG_1] > 0.0
    assert gradient[1, Action.ENTER_SHORT_1] > 0.0
    assert gradient[3, Action.WAIT] < 0.0
    assert gradient[3, Action.ENTER_LONG_1] > 0.0
    assert gradient[3, Action.ENTER_SHORT_1] > 0.0


def test_paired_recurrent_a_plus_rejects_cross_side_or_non_economic_pairs() -> None:
    values = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="explicit economic pair"):
        paired_recurrent_a_plus_rank_loss(
            values,
            pair_ids=torch.tensor([3, 3]),
            pair_sides=torch.tensor([1, 2]),
            economic_wins=torch.tensor([True, False]),
            population_weights=torch.ones(2),
            margin=0.25,
            action_margin=0.25,
        )
    with pytest.raises(ValueError, match="explicit economic pair"):
        paired_recurrent_a_plus_rank_loss(
            values,
            pair_ids=torch.tensor([3, 3]),
            pair_sides=torch.tensor([1, 1]),
            economic_wins=torch.tensor([True, True]),
            population_weights=torch.ones(2),
            margin=0.25,
            action_margin=0.25,
        )


def test_paired_recurrent_a_plus_skips_an_orphaned_unlearnable_anchor() -> None:
    values = torch.tensor(
        [[0.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    result = paired_recurrent_a_plus_rank_loss(
        values,
        pair_ids=torch.tensor([3]),
        pair_sides=torch.tensor([1]),
        economic_wins=torch.tensor([True]),
        population_weights=torch.ones(1),
        margin=0.25,
        action_margin=0.25,
    )
    gradient, = torch.autograd.grad(result.loss, values)

    assert result.pair_count.item() == 0.0
    assert result.loss.item() == 0.0
    assert gradient.abs().sum().item() == 0.0


def test_paired_recurrent_a_plus_keeps_complete_pairs_when_an_orphan_is_skipped(
) -> None:
    values = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 9.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    result = paired_recurrent_a_plus_rank_loss(
        values,
        pair_ids=torch.tensor([3, 3, 4]),
        pair_sides=torch.tensor([1, 1, 2]),
        economic_wins=torch.tensor([True, False, True]),
        population_weights=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        margin=0.25,
        action_margin=0.25,
    )
    gradient, = torch.autograd.grad(result.loss, values)

    assert result.pair_count.item() == 1.0
    assert result.group_metrics["long_pair_count"].item() == 1.0
    assert result.group_metrics["short_pair_count"].item() == 0.0
    assert gradient[2].abs().sum().item() == 0.0


def test_paired_recurrent_a_plus_rejects_more_than_two_rows_for_one_pair() -> None:
    with pytest.raises(ValueError, match="explicit economic pair"):
        paired_recurrent_a_plus_rank_loss(
            torch.zeros((3, 3), dtype=torch.float32),
            pair_ids=torch.tensor([3, 3, 3]),
            pair_sides=torch.tensor([1, 1, 1]),
            economic_wins=torch.tensor([True, False, False]),
            population_weights=torch.ones(3),
            margin=0.25,
            action_margin=0.25,
        )


def test_paired_a_plus_uses_continuous_regime_similarity_without_argmax() -> None:
    result = paired_a_plus_rank_loss(
        torch.tensor([
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
        ]),
        failed_long_membership=torch.tensor([0.0, 1.0]),
        failed_short_membership=torch.zeros(2),
        valid_long_membership=torch.tensor([1.0, 0.0]),
        valid_short_membership=torch.zeros(2),
        regime_probabilities=torch.tensor([
            [0.00, 0.49, 0.51],
            [0.00, 0.51, 0.49],
        ]),
        margin=0.25,
    )

    assert result.active_groups.item() == 2.0
    assert result.pair_mass.item() == pytest.approx(0.4998)
    assert result.loss.item() == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(-1.75)).item()
    )


def test_paired_a_plus_rank_loss_balances_present_long_and_short_sides() -> None:
    action_values = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Long good.
            [0.0, 0.0, 0.0],  # Long failure.
            [0.0, 0.0, 2.0],  # Short good, transition Regime.
            [0.0, 0.0, -2.0],  # Matched Short failure.
            [0.0, 0.0, 2.0],  # Short good, expansion-trend Regime.
            [0.0, 0.0, -2.0],  # Matched Short failure.
        ],
        dtype=torch.float64,
    )
    result = paired_a_plus_rank_loss(
        action_values,
        failed_long_membership=torch.tensor([0, 1, 0, 0, 0, 0.0]),
        valid_long_membership=torch.tensor([1, 0, 0, 0, 0, 0.0]),
        failed_short_membership=torch.tensor([0, 0, 0, 1, 0, 1.0]),
        valid_short_membership=torch.tensor([0, 0, 1, 0, 1, 0.0]),
        regime_probabilities=torch.nn.functional.one_hot(
            torch.tensor([0, 0, 1, 1, 2, 2]), num_classes=3
        ).float(),
        margin=0.25,
    )

    assert result.active_groups.item() == 3.0
    assert result.loss.item() == pytest.approx(0.42459244212563435)


@pytest.mark.parametrize(
    ("failed_long", "failed_short", "valid_long", "valid_short", "regimes"),
    (
        # Opposite sides never form an A+ pair.
        ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [2, 2]),
        # The same side in different learned Regime classes never forms a pair.
        ([0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1, 2]),
        # Missing failed cohorts produce no invented supervision.
        ([0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2, 2]),
    ),
)
def test_paired_a_plus_rank_loss_never_invents_cross_context_pairs(
    failed_long: list[float],
    failed_short: list[float],
    valid_long: list[float],
    valid_short: list[float],
    regimes: list[int],
) -> None:
    result = paired_a_plus_rank_loss(
        torch.tensor([
            [0.0, 0.5, -0.5],
            [0.0, -0.5, 0.5],
        ]),
        failed_long_membership=torch.tensor(failed_long),
        failed_short_membership=torch.tensor(failed_short),
        valid_long_membership=torch.tensor(valid_long),
        valid_short_membership=torch.tensor(valid_short),
        regime_probabilities=torch.nn.functional.one_hot(
            torch.tensor(regimes), num_classes=3
        ).float(),
        margin=0.25,
    )

    assert result.loss.item() == 0.0
    assert result.active_groups.item() == 0.0
    assert result.pair_mass.item() == 0.0


def test_paired_a_plus_margin_cannot_drift_from_its_declared_semantics() -> None:
    with pytest.raises(ValueError, match=r"paired A\+"):
        _agent(
            seed=528,
            selectivity_weight=1.0,
            side_balance="equal_long_short_v1",
            selectivity_semantics=ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            persistent_chop_negative_emphasis=2.0,
            paired_a_plus_margin=0.25,
        )
    with pytest.raises(ValueError, match=r"paired A\+"):
        _agent(
            seed=529,
            selectivity_weight=1.0,
            side_balance="equal_long_short_v1",
            selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
            persistent_chop_negative_emphasis=2.0,
            paired_a_plus_margin=0.0,
        )


def test_chop_specific_margin_requires_wait_over_entries_without_touching_valid_short(
) -> None:
    action_values = torch.tensor(
        [
            [0.10, 0.30, 0.40],  # dominant chop: WAIT below both entries
            [0.20, 0.60, 0.10],  # failed Long: WAIT below Long
            [0.20, 0.10, 0.50],  # failed Short: WAIT below Short
            [0.10, 0.20, 0.70],  # valid Short: not a WAIT-margin row
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    losses = chop_specific_wait_margin_losses(
        action_values,
        dominant_chop_membership=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        failed_long_membership=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        failed_short_membership=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        chop_margin=0.25,
        failed_confluence_margin=0.35,
    )

    torch.testing.assert_close(
        losses,
        torch.tensor([0.55, 0.75, 0.65, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_each_chop_margin_can_be_disabled_independently() -> None:
    action_values = torch.tensor(
        [[0.0, 1.0, 0.5], [0.0, 0.5, 1.0]], dtype=torch.float64
    )

    losses = chop_specific_wait_margin_losses(
        action_values,
        dominant_chop_membership=torch.tensor([1.0, 0.0]),
        failed_long_membership=torch.tensor([0.0, 0.0]),
        failed_short_membership=torch.tensor([0.0, 1.0]),
        chop_margin=0.0,
        failed_confluence_margin=0.25,
    )

    torch.testing.assert_close(
        losses,
        torch.tensor([0.0, 1.25], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def _teacher_row(
    *,
    long_attempt: float,
    long_clean: float,
    short_attempt: float,
    short_clean: float,
    chop: float,
    neutral: float,
    trend: float,
) -> torch.Tensor:
    values = np.full(len(CHANNELS), 0.1, dtype=np.float32)
    values[:4] = (long_attempt, long_clean, short_attempt, short_clean)
    values[4:7] = (chop, neutral, trend)
    return torch.from_numpy(values)


def _selectivity() -> BalanceAwareRegimeSelectivity:
    return BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        probability_epsilon=1e-6,
        headroom_pressure=1.0,
        dominant_chop_pressure=2.0,
    )


def test_frozen_confluence_contract_requires_side_dominance_and_preserves_exact_classes(
) -> None:
    selectivity = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        probability_epsilon=1e-6,
        headroom_pressure=1.0,
        dominant_chop_pressure=2.0,
        semantics=SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
    )
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    rows = torch.stack((
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.90,
            short_clean=0.90,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.95,
            short_clean=0.95,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.90,
            short_clean=0.90,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.90,
            short_clean=0.90,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.05,
            long_clean=0.05,
            short_attempt=0.05,
            short_clean=0.05,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.20,
            long_clean=0.20,
            short_attempt=0.20,
            short_clean=0.20,
            chop=0.95,
            neutral=0.03,
            trend=0.02,
        ),
    ))
    targets = torch.tensor((
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
        int(Action.ENTER_LONG_1),
        int(Action.WAIT),
        int(Action.WAIT),
        int(Action.WAIT),
        int(Action.WAIT),
        int(Action.WAIT),
    ))

    evidence = selectivity.exact_wait_negative_weight_evidence(rows, targets)

    assert evidence.transition_positive_long_membership[0] > 0.0
    assert evidence.transition_positive_short_membership[1] > 0.0
    assert (
        evidence.transition_positive_long_membership[0]
        > evidence.transition_positive_long_membership[2]
    )
    assert evidence.failed_long_confluence_membership[3] > 0.0
    assert evidence.failed_short_confluence_membership[4] > 0.0
    assert evidence.exact_wait_weights[3] > evidence.exact_wait_weights[6]
    assert evidence.exact_wait_weights[4] > evidence.exact_wait_weights[6]
    assert evidence.transition_positive_long_membership[3] == 0.0
    assert evidence.transition_positive_short_membership[4] == 0.0
    assert evidence.exact_wait_weights[5] > evidence.exact_wait_weights[6]
    assert evidence.exact_wait_weights[7] > evidence.exact_wait_weights[6]


@pytest.mark.parametrize(
    ("target", "declared_membership", "declared_scores", "opposite_scores"),
    (
        (
            Action.ENTER_LONG_1,
            "transition_positive_long_membership",
            (0.90, 0.90),
            (0.10, 0.10),
        ),
        (
            Action.ENTER_SHORT_1,
            "transition_positive_short_membership",
            (0.90, 0.90),
            (0.10, 0.10),
        ),
    ),
    ids=("long", "short"),
)
def test_entry_rule_requires_declared_expansion_ready_regime_and_side_dominance(
    target: Action,
    declared_membership: str,
    declared_scores: tuple[float, float],
    opposite_scores: tuple[float, float],
) -> None:
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    long_scores, short_scores = (
        (declared_scores, opposite_scores)
        if target == Action.ENTER_LONG_1
        else (opposite_scores, declared_scores)
    )
    dominant = _teacher_row(
        long_attempt=long_scores[0],
        long_clean=long_scores[1],
        short_attempt=short_scores[0],
        short_clean=short_scores[1],
        **ready,
    )
    conflicting = _teacher_row(
        long_attempt=0.90,
        long_clean=0.90,
        short_attempt=0.95,
        short_clean=0.95,
        **ready,
    )
    selectivity = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        probability_epsilon=1e-6,
        headroom_pressure=1.0,
        dominant_chop_pressure=2.0,
        semantics=SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
    )

    evidence = selectivity.exact_wait_negative_weight_evidence(
        torch.stack((dominant, conflicting)),
        torch.tensor((int(target), int(target))),
    )
    membership = getattr(evidence, declared_membership)

    assert membership[0] > membership[1]


@pytest.mark.parametrize(
    ("teacher", "failed_membership"),
    (
        (
            dict(
                long_attempt=0.90,
                long_clean=0.90,
                short_attempt=0.10,
                short_clean=0.10,
            ),
            "failed_long_confluence_membership",
        ),
        (
            dict(
                long_attempt=0.10,
                long_clean=0.10,
                short_attempt=0.90,
                short_clean=0.90,
            ),
            "failed_short_confluence_membership",
        ),
    ),
    ids=("failed-long", "failed-short"),
)
def test_failed_directional_confluence_strengthens_wait_never_enter(
    teacher: dict[str, float],
    failed_membership: str,
) -> None:
    selectivity = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        probability_epsilon=1e-6,
        headroom_pressure=1.0,
        dominant_chop_pressure=2.0,
        semantics=SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
    )
    row = _teacher_row(
        **teacher,
        chop=0.05,
        neutral=0.45,
        trend=0.50,
    )

    evidence = selectivity.exact_wait_negative_weight_evidence(
        row[None],
        torch.tensor((int(Action.WAIT),)),
    )

    assert getattr(evidence, failed_membership)[0] > 0.0
    assert evidence.transition_positive_long_membership[0] == 0.0
    assert evidence.transition_positive_short_membership[0] == 0.0


def test_conflict_weak_expansion_and_persistent_chop_strengthen_wait() -> None:
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    rows = torch.stack((
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.90,
            short_clean=0.90,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.05,
            long_clean=0.05,
            short_attempt=0.05,
            short_clean=0.05,
            **ready,
        ),
        _teacher_row(
            long_attempt=0.20,
            long_clean=0.20,
            short_attempt=0.20,
            short_clean=0.20,
            chop=0.95,
            neutral=0.03,
            trend=0.02,
        ),
    ))
    selectivity = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        probability_epsilon=1e-6,
        headroom_pressure=1.0,
        dominant_chop_pressure=2.0,
        semantics=SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
    )

    evidence = selectivity.exact_wait_negative_weight_evidence(
        rows,
        torch.full((3,), int(Action.WAIT)),
    )

    conflict, weak, chop = evidence.exact_wait_weights
    assert conflict > weak
    assert chop > weak


def test_hard_wait_replay_priority_uses_exact_wait_confluence_only() -> None:
    selectivity = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=(0.10, 0.10),
        probability_epsilon=1e-6,
        headroom_pressure=1.0,
        dominant_chop_pressure=2.0,
        semantics=SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
    )
    rows = torch.stack((
        _teacher_row(
            long_attempt=0.20,
            long_clean=0.20,
            short_attempt=0.20,
            short_clean=0.20,
            chop=0.95,
            neutral=0.03,
            trend=0.02,
        ),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.05,
            short_clean=0.05,
            chop=0.05,
            neutral=0.45,
            trend=0.50,
        ),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.05,
            short_clean=0.05,
            chop=0.05,
            neutral=0.45,
            trend=0.50,
        ),
    ))

    priorities = selectivity.exact_wait_replay_priorities(
        rows,
        torch.tensor([
            int(Action.WAIT),
            int(Action.WAIT),
            int(Action.ENTER_LONG_1),
        ]),
    )

    assert priorities[0] > 0.9
    assert priorities[1] > 0.5
    assert priorities[2] == 0.0


def _agent(
    *,
    seed: int,
    selectivity_weight: float,
    n_step_return: int = 1,
    recurrent_burn_in: int = 0,
    entry_action_weight: float = 0.0,
    entry_action_margin: float = 0.0,
    entry_action_loss_reduction: str = "population_weighted_mean_v1",
    side_balance: str | None = None,
    selectivity_semantics: str | None = None,
    persistent_chop_negative_emphasis: float | None = None,
    chop_wait_margin: float = 0.0,
    failed_confluence_margin: float = 0.0,
    paired_a_plus_margin: float = 0.0,
    expansion_centers: tuple[float, float] = (0.10, 0.10),
    device: str = "cpu",
) -> RecurrentC51Agent:
    optional_settings = (
        {}
        if side_balance is None
        else {"regime_selectivity_side_balance": side_balance}
    )
    if selectivity_semantics is not None:
        optional_settings["regime_selectivity_semantics"] = selectivity_semantics
    if persistent_chop_negative_emphasis is not None:
        optional_settings[
            "regime_selectivity_persistent_chop_negative_emphasis"
        ] = persistent_chop_negative_emphasis
    return RecurrentC51Agent(
        3,
        hidden_dim=24,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.03,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        n_step_return=n_step_return,
        recurrent_burn_in=recurrent_burn_in,
        device=device,
        seed=seed,
        teacher_channels=len(CHANNELS),
        teacher_channel_names=CHANNELS,
        teacher_loss_weight=1e-6,
        teacher_entry_search_centers=expansion_centers,
        entry_action_loss_weight=entry_action_weight,
        entry_action_margin=entry_action_margin,
        entry_action_loss_reduction=entry_action_loss_reduction,
        regime_selectivity_loss_weight=selectivity_weight,
        regime_selectivity_expansion_centers=expansion_centers,
        regime_selectivity_chop_wait_margin=chop_wait_margin,
        regime_selectivity_failed_confluence_margin=failed_confluence_margin,
        regime_selectivity_paired_a_plus_margin=paired_a_plus_margin,
        **optional_settings,
    )


def _sequence(
    observation: tuple[float, float, float],
    teacher: torch.Tensor,
    *,
    headroom: float,
    target: Action,
    teacher_imitation_visible: bool = True,
    pair_id: int | None = None,
    pair_side: Action | None = None,
    economic_win: bool | None = None,
    training_valid: bool = True,
) -> tuple[Transition, ...]:
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    return (
        Transition(
            observation=np.asarray(observation, np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.zeros(3, np.float32),
            terminated=True,
            valid_actions=flat,
            next_valid_actions=(),
            teacher_target=teacher.numpy(),
            teacher_imitation_visible=teacher_imitation_visible,
            entry_action_target=target,
            regime_selectivity_headroom_fraction=headroom,
            paired_a_plus_pair_id=pair_id,
            paired_a_plus_pair_side=pair_side,
            paired_a_plus_economic_win=economic_win,
            paired_a_plus_population_weight=(
                1.0 if pair_id is not None else None
            ),
            training_valid=training_valid,
        ),
    )


def test_paired_recurrent_sequences_both_receive_td_and_anchor_ranking() -> None:
    agent = _agent(
        seed=502,
        selectivity_weight=0.3,
        entry_action_margin=0.25,
        side_balance="paired_recurrent_long_short_v1",
        selectivity_semantics=PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    teacher = _teacher_row(
        long_attempt=0.90,
        long_clean=0.90,
        short_attempt=0.10,
        short_clean=0.10,
        **ready,
    )
    winner = _sequence(
        (1.0, 0.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        pair_id=7,
        pair_side=Action.ENTER_LONG_1,
        economic_win=True,
    )
    failure = _sequence(
        (0.0, 1.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.WAIT,
        pair_id=7,
        pair_side=Action.ENTER_LONG_1,
        economic_win=False,
    )

    agent.train_batch((winner, failure))

    assert agent.last_train_metrics["sampled_valid_learning_rows"] == 2.0
    assert agent.last_train_metrics["rl_loss"] > 0.0
    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_pair_count"
    ] == 1.0


def test_paired_recurrent_learner_rejects_missing_population_correction() -> None:
    agent = _agent(
        seed=503,
        selectivity_weight=0.3,
        entry_action_margin=0.25,
        side_balance="paired_recurrent_long_short_v1",
        selectivity_semantics=PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    teacher = _teacher_row(
        long_attempt=0.90,
        long_clean=0.90,
        short_attempt=0.10,
        short_clean=0.10,
        chop=0.05,
        neutral=0.45,
        trend=0.50,
    )
    winner = _sequence(
        (1.0, 0.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        pair_id=7,
        pair_side=Action.ENTER_LONG_1,
        economic_win=True,
    )
    failure = _sequence(
        (0.0, 1.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.WAIT,
        pair_id=7,
        pair_side=Action.ENTER_LONG_1,
        economic_win=False,
    )
    winner_without_correction = (
        replace(winner[0], paired_a_plus_population_weight=None),
    )

    with pytest.raises(ValueError, match="batch evidence is incomplete"):
        agent.train_batch((winner_without_correction, failure))


def test_paired_recurrent_batch_keeps_td_when_one_pair_anchor_is_unlearnable() -> None:
    agent = _agent(
        seed=504,
        selectivity_weight=0.3,
        side_balance="paired_recurrent_long_short_v1",
        selectivity_semantics=PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    teacher = _teacher_row(
        long_attempt=0.90,
        long_clean=0.90,
        short_attempt=0.10,
        short_clean=0.10,
        chop=0.05,
        neutral=0.45,
        trend=0.50,
    )
    winner = _sequence(
        (1.0, 0.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        pair_id=7,
        pair_side=Action.ENTER_LONG_1,
        economic_win=True,
    )
    unlearnable_failure = _sequence(
        (0.0, 1.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.WAIT,
        pair_id=7,
        pair_side=Action.ENTER_LONG_1,
        economic_win=False,
        training_valid=False,
    )

    agent.train_batch((winner, unlearnable_failure))

    assert agent.last_train_metrics["sampled_valid_learning_rows"] == 1.0
    assert agent.last_train_metrics["rl_loss"] > 0.0
    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_pair_count"
    ] == 0.0


def test_teacher_dropout_only_disables_imitation_not_exact_or_confluence_losses(
) -> None:
    agent = _agent(
        seed=503,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        entry_action_margin=0.25,
        side_balance="equal_long_short_v1",
        selectivity_semantics=(
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS
        ),
        persistent_chop_negative_emphasis=2.0,
    )
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    long_winner = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            **ready,
        ),
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        teacher_imitation_visible=False,
    )
    failed_long = _sequence(
        (0.0, 1.0, 0.0),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            **ready,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )

    agent.train_batch(
        (long_winner, failed_long),
        teacher_weight_scale=1.0,
        entry_action_weight_scale=1.0,
    )

    assert agent.last_train_metrics["teacher_loss"] == 0.0
    assert agent.last_train_metrics["entry_action_loss"] > 0.0
    assert agent.last_train_metrics["entry_action_margin_loss"] > 0.0
    assert agent.last_train_metrics["entry_action_supervised_rows"] == 2.0
    assert agent.last_train_metrics["regime_selectivity_loss"] > 0.0
    assert agent.last_train_metrics["regime_selectivity_supervised_rows"] == 2.0
    assert agent.last_train_metrics[
        "regime_selectivity_failed_long_confluence_rows"
    ] > 0.0


def test_paired_a_plus_supervision_remains_active_after_teacher_dropout() -> None:
    agent = _agent(
        seed=524,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    batch = (
        _sequence(
            (1.0, 0.0, 0.0),
            _teacher_row(
                long_attempt=0.90,
                long_clean=0.90,
                short_attempt=0.10,
                short_clean=0.10,
                **ready,
            ),
            headroom=1.0,
            target=Action.ENTER_LONG_1,
            teacher_imitation_visible=False,
        ),
        _sequence(
            (0.0, 1.0, 0.0),
            _teacher_row(
                long_attempt=0.90,
                long_clean=0.90,
                short_attempt=0.10,
                short_clean=0.10,
                **ready,
            ),
            headroom=1.0,
            target=Action.WAIT,
            teacher_imitation_visible=False,
        ),
        _sequence(
            (0.0, 0.0, 1.0),
            _teacher_row(
                long_attempt=0.10,
                long_clean=0.10,
                short_attempt=0.90,
                short_clean=0.90,
                **ready,
            ),
            headroom=1.0,
            target=Action.ENTER_SHORT_1,
            teacher_imitation_visible=False,
        ),
        _sequence(
            (1.0, 1.0, 0.0),
            _teacher_row(
                long_attempt=0.10,
                long_clean=0.10,
                short_attempt=0.90,
                short_clean=0.90,
                **ready,
            ),
            headroom=1.0,
            target=Action.WAIT,
            teacher_imitation_visible=False,
        ),
    )

    agent.train_batch(
        batch,
        teacher_weight_scale=1.0,
        entry_action_weight_scale=1.0,
    )

    assert agent.last_train_metrics["teacher_loss"] == 0.0
    assert agent.last_train_metrics["entry_action_supervised_rows"] == 4.0
    # Continuous opposite-side evidence stays in the weighted population;
    # no fixed cutoff turns weak evidence into a handcrafted hard class.
    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_pair_count"
    ] == 4.0
    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_long_expansion_trend_"
        "pair_count"
    ] == 2.0
    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_short_expansion_trend_"
        "pair_count"
    ] == 2.0
    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_loss"
    ] > 0.0


def test_paired_a_plus_teacher_free_checkpoint_round_trip_preserves_policy(
    tmp_path: Path,
) -> None:
    agent = _agent(
        seed=525,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    observation = np.asarray((0.25, -0.50, 0.75), dtype=np.float32)
    _, _, expected_values = agent.select_action(
        observation,
        hidden=None,
        valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        epsilon=0.0,
        return_action_values=True,
    )

    agent.discard_teacher()
    checkpoint = agent.save(tmp_path / "paired-a-plus-teacher-free.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    restored.assert_teacher_free()
    _, _, actual_values = restored.select_action(
        observation,
        hidden=None,
        valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        epsilon=0.0,
        return_action_values=True,
    )

    assert restored.regime_selectivity_chop_wait_margin == 0.0
    assert restored.regime_selectivity_failed_confluence_margin == 0.0
    assert restored.regime_selectivity_paired_a_plus_margin == 0.0
    np.testing.assert_array_equal(actual_values, expected_values)


def test_load_normalizes_legacy_teacher_free_paired_a_plus_margins(
    tmp_path: Path,
) -> None:
    agent = _agent(
        seed=527,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    agent.discard_teacher()
    # v19 stripped the teacher and semantics but serialized these inactive
    # training-only values, making its otherwise valid policy unloadable.
    agent.regime_selectivity_chop_wait_margin = 0.25
    agent.regime_selectivity_failed_confluence_margin = 0.25
    agent.regime_selectivity_paired_a_plus_margin = 0.25
    checkpoint = agent.save(tmp_path / "legacy-paired-a-plus.pt", manifest={})

    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")

    restored.assert_teacher_free()
    assert restored.regime_selectivity_chop_wait_margin == 0.0
    assert restored.regime_selectivity_failed_confluence_margin == 0.0
    assert restored.regime_selectivity_paired_a_plus_margin == 0.0


def test_paired_a_plus_pressure_is_additive_without_weakening_v18_losses() -> None:
    common = dict(
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
    )
    baseline = _agent(
        seed=526,
        selectivity_semantics=ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        **common,
    )
    paired = _agent(
        seed=526,
        selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        paired_a_plus_margin=0.25,
        **common,
    )
    ready = dict(chop=0.05, neutral=0.45, trend=0.50)
    batch = (
        _sequence(
            (1.0, 0.0, 0.0),
            _teacher_row(
                long_attempt=0.90,
                long_clean=0.90,
                short_attempt=0.10,
                short_clean=0.10,
                **ready,
            ),
            headroom=1.0,
            target=Action.ENTER_LONG_1,
            teacher_imitation_visible=False,
        ),
        _sequence(
            (0.0, 1.0, 0.0),
            _teacher_row(
                long_attempt=0.90,
                long_clean=0.90,
                short_attempt=0.10,
                short_clean=0.10,
                **ready,
            ),
            headroom=1.0,
            target=Action.WAIT,
            teacher_imitation_visible=False,
        ),
    )

    baseline.train_batch(batch)
    paired.train_batch(batch)

    assert paired.last_train_metrics["regime_selectivity_loss"] == pytest.approx(
        baseline.last_train_metrics["regime_selectivity_loss"]
        + paired.last_train_metrics["regime_selectivity_paired_a_plus_loss"]
    )


def test_paired_a_plus_is_exactly_v18_when_no_matched_pair_exists() -> None:
    common = dict(
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
    )
    baseline = _agent(
        seed=527,
        selectivity_semantics=ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        **common,
    )
    paired = _agent(
        seed=527,
        selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        paired_a_plus_margin=0.25,
        **common,
    )
    valid_only = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            chop=0.05,
            neutral=0.45,
            trend=0.50,
        ),
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        teacher_imitation_visible=False,
    )

    baseline.train_batch((valid_only,))
    paired.train_batch((valid_only,))

    assert paired.last_train_metrics[
        "regime_selectivity_paired_a_plus_pair_mass"
    ] == 0.0
    assert paired.last_train_metrics["regime_selectivity_loss"] == pytest.approx(
        baseline.last_train_metrics["regime_selectivity_loss"]
    )


def test_chop_wait_margins_remain_active_after_teacher_dropout() -> None:
    settings = dict(
        seed=521,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        selectivity_semantics=(
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS
        ),
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
    )
    baseline = _agent(**settings)
    margin_agent = _agent(
        **settings,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.35,
    )
    dead_chop = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.20,
            long_clean=0.20,
            short_attempt=0.80,
            short_clean=0.80,
            chop=0.90,
            neutral=0.05,
            trend=0.05,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )
    failed_short = _sequence(
        (0.0, 1.0, 0.0),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.90,
            short_clean=0.90,
            chop=0.05,
            neutral=0.45,
            trend=0.50,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )
    valid_short = _sequence(
        (0.0, 0.0, 1.0),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.90,
            short_clean=0.90,
            chop=0.05,
            neutral=0.45,
            trend=0.50,
        ),
        headroom=1.0,
        target=Action.ENTER_SHORT_1,
        teacher_imitation_visible=False,
    )

    batch = (dead_chop, failed_short, valid_short)
    baseline.train_batch(batch, teacher_weight_scale=1.0)
    margin_agent.train_batch(batch, teacher_weight_scale=1.0)

    assert margin_agent.last_train_metrics["teacher_loss"] == 0.0
    assert margin_agent.last_train_metrics["regime_selectivity_loss"] > (
        baseline.last_train_metrics["regime_selectivity_loss"]
    )
    assert margin_agent.last_train_metrics[
        "regime_selectivity_transition_positive_short_rows"
    ] > 0.0


def test_all_action_dominant_chop_margin_learns_wait_without_relabeling(
) -> None:
    settings = dict(
        seed=522,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        selectivity_semantics=ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        side_balance="equal_long_short_v1",
        persistent_chop_negative_emphasis=2.0,
    )
    baseline = _agent(**settings)
    margin_agent = _agent(
        **settings,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
    )
    dominant = dict(chop=0.90, neutral=0.05, trend=0.05)
    dominant_long = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            **dominant,
        ),
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        teacher_imitation_visible=False,
    )
    dominant_short = _sequence(
        (0.0, 0.0, 1.0),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.90,
            short_clean=0.90,
            **dominant,
        ),
        headroom=1.0,
        target=Action.ENTER_SHORT_1,
        teacher_imitation_visible=False,
    )

    baseline.train_batch(
        (dominant_long, dominant_short),
        teacher_weight_scale=1.0,
        entry_action_weight_scale=1.0,
    )
    margin_agent.train_batch(
        (dominant_long, dominant_short),
        teacher_weight_scale=1.0,
        entry_action_weight_scale=1.0,
    )

    assert margin_agent.last_train_metrics["teacher_loss"] == 0.0
    assert margin_agent.last_train_metrics["entry_action_target_wait_rows"] == 0.0
    assert margin_agent.last_train_metrics["entry_action_target_long_rows"] == 1.0
    assert margin_agent.last_train_metrics["entry_action_target_short_rows"] == 1.0
    assert margin_agent.last_train_metrics["regime_selectivity_loss"] > (
        baseline.last_train_metrics["regime_selectivity_loss"]
    )


def test_chop_margin_cohorts_are_learned_after_recurrent_burn_in() -> None:
    agent = _agent(
        seed=523,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        recurrent_burn_in=1,
        side_balance="equal_long_short_v1",
        selectivity_semantics=(
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS
        ),
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
    )
    neutral_burn = _sequence(
        (0.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.10,
            short_clean=0.10,
            chop=0.20,
            neutral=0.60,
            trend=0.20,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )[0]
    dead_chop = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.20,
            long_clean=0.20,
            short_attempt=0.20,
            short_clean=0.20,
            chop=0.90,
            neutral=0.05,
            trend=0.05,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )[0]
    failed_short = _sequence(
        (0.0, 1.0, 0.0),
        _teacher_row(
            long_attempt=0.10,
            long_clean=0.10,
            short_attempt=0.90,
            short_clean=0.90,
            chop=0.05,
            neutral=0.45,
            trend=0.50,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )[0]

    agent.train_batch(((neutral_burn, dead_chop), (neutral_burn, failed_short)))

    assert agent.last_train_metrics[
        "regime_selectivity_persistent_dead_chop_rows"
    ] > 0.0
    assert agent.last_train_metrics[
        "regime_selectivity_failed_short_confluence_rows"
    ] > 0.0


def test_paired_a_plus_uses_learning_rows_after_recurrent_burn_in() -> None:
    agent = _agent(
        seed=525,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
        recurrent_burn_in=1,
        side_balance="equal_long_short_v1",
        selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        persistent_chop_negative_emphasis=2.0,
        chop_wait_margin=0.25,
        failed_confluence_margin=0.25,
        paired_a_plus_margin=0.25,
    )
    burn = _sequence(
        (0.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.90,
            long_clean=0.90,
            short_attempt=0.10,
            short_clean=0.10,
            chop=0.05,
            neutral=0.90,
            trend=0.05,
        ),
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )[0]
    teacher = _teacher_row(
        long_attempt=0.90,
        long_clean=0.90,
        short_attempt=0.10,
        short_clean=0.10,
        chop=0.05,
        neutral=0.45,
        trend=0.50,
    )
    good = _sequence(
        (1.0, 0.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.ENTER_LONG_1,
        teacher_imitation_visible=False,
    )[0]
    failed = _sequence(
        (0.0, 1.0, 0.0),
        teacher,
        headroom=1.0,
        target=Action.WAIT,
        teacher_imitation_visible=False,
    )[0]

    agent.train_batch(((burn, good), (burn, failed)))

    assert agent.last_train_metrics[
        "regime_selectivity_paired_a_plus_long_expansion_trend_pair_count"
    ] == 1.0


def _greedy(agent: RecurrentC51Agent, observation: tuple[float, ...]) -> Action:
    action, _, _ = agent.select_action(
        np.asarray(observation, np.float32),
        hidden=None,
        valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        epsilon=0.0,
    )
    return action


def _transition_teacher_row(
    *,
    chop_persistence: float,
    transition_readiness: float,
    long_attempt: float = 0.1,
    long_clean: float = 0.1,
    short_attempt: float = 0.1,
    short_clean: float = 0.1,
) -> torch.Tensor:
    values = _teacher_row(
        long_attempt=long_attempt,
        long_clean=long_clean,
        short_attempt=short_attempt,
        short_clean=short_clean,
        chop=0.1,
        neutral=0.1,
        trend=0.8,
    ).clone()
    dead_chop = chop_persistence * (1.0 - transition_readiness)
    transition = chop_persistence * transition_readiness
    updates = {
        "chop_no_trend_probability": dead_chop,
        "chop_end_transition_probability": transition,
        "expansion_trend_probability": 1.0 - dead_chop - transition,
    }
    for channel, value in updates.items():
        values[CHANNELS.index(channel)] = value
    return values


def test_wait_probability_increases_with_chop_and_lower_headroom() -> None:
    selectivity = _selectivity()
    clean = _teacher_row(
        long_attempt=0.4,
        long_clean=0.4,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    chop = _teacher_row(
        long_attempt=0.4,
        long_clean=0.4,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.90,
        neutral=0.05,
        trend=0.05,
    )

    enter_long = torch.tensor([int(Action.ENTER_LONG_1)])
    clean_safe = selectivity.target_probabilities(
        clean[None], torch.tensor([1.0]), enter_long
    )
    clean_low = selectivity.target_probabilities(
        clean[None], torch.tensor([0.1]), enter_long
    )
    chop_safe = selectivity.target_probabilities(
        chop[None], torch.tensor([1.0]), enter_long
    )
    chop_low = selectivity.target_probabilities(
        chop[None], torch.tensor([0.1]), enter_long
    )

    assert clean_safe[0, 0].item() == pytest.approx(0.368421, abs=1e-6)
    assert clean_low[0, 0] > clean_safe[0, 0]
    assert chop_safe[0, 0] > clean_safe[0, 0]
    assert chop_low[0, 0] > clean_low[0, 0]


@pytest.mark.parametrize(
    ("long_values", "short_values", "expected"),
    [
        ((0.90, 0.80), (0.10, 0.10), Action.ENTER_LONG_1),
        ((0.10, 0.10), (0.90, 0.80), Action.ENTER_SHORT_1),
    ],
)
def test_clean_directional_confluence_is_preserved(
    long_values: tuple[float, float],
    short_values: tuple[float, float],
    expected: Action,
) -> None:
    probabilities = _selectivity().target_probabilities(
        _teacher_row(
            long_attempt=long_values[0],
            long_clean=long_values[1],
            short_attempt=short_values[0],
            short_clean=short_values[1],
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        )[None],
        torch.tensor([0.10]),
        torch.tensor([int(expected)]),
    )

    assert int(probabilities.argmax(-1).item()) == int(expected)


def test_recovery_requires_stronger_relative_expansion_evidence() -> None:
    selectivity = _selectivity()
    medium = _teacher_row(
        long_attempt=0.4,
        long_clean=0.4,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    strong = _teacher_row(
        long_attempt=0.9,
        long_clean=0.8,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    targets = selectivity.target_probabilities(
        torch.stack((medium, strong)),
        torch.tensor([0.10, 0.10]),
        torch.tensor([
            int(Action.ENTER_LONG_1),
            int(Action.ENTER_LONG_1),
        ]),
    )

    assert int(targets[0].argmax().item()) == int(Action.WAIT)
    assert int(targets[1].argmax().item()) == int(Action.ENTER_LONG_1)


def test_literal_fixture_rejects_universal_wait_and_all_enter() -> None:
    selectivity = _selectivity()
    rows = torch.stack((
        _teacher_row(
            long_attempt=0.4,
            long_clean=0.4,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        ),
        _teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        ),
        _teacher_row(
            long_attempt=0.1,
            long_clean=0.1,
            short_attempt=0.9,
            short_clean=0.8,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        ),
    ))
    actions = selectivity.target_probabilities(
        rows,
        torch.tensor([0.10, 0.10, 0.10]),
        torch.tensor([
            int(Action.ENTER_LONG_1),
            int(Action.ENTER_LONG_1),
            int(Action.ENTER_SHORT_1),
        ]),
    ).argmax(-1)

    assert actions.tolist() == [
        int(Action.WAIT),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
    ]


def test_channel_lineage_and_tensor_shape_contract_fail_closed() -> None:
    with pytest.raises(ValueError, match="channel order"):
        BalanceAwareRegimeSelectivity(
            channel_names=tuple(reversed(CHANNELS)),
            expansion_centers=(0.10, 0.10),
        )
    selectivity = _selectivity()
    with pytest.raises(ValueError, match="headroom"):
        selectivity.target_probabilities(
            torch.ones((1, len(CHANNELS))),
            torch.tensor([0.1, 0.2]),
            torch.tensor([int(Action.WAIT)]),
        )


def test_agent_rejects_nonfinite_selectivity_loss_weight() -> None:
    with pytest.raises(ValueError, match="teacher settings"):
        _agent(seed=139, selectivity_weight=float("nan"))


def test_agent_rejects_unknown_selectivity_side_balance() -> None:
    with pytest.raises(ValueError, match="side balance"):
        _agent(
            seed=139,
            selectivity_weight=1.0,
            side_balance="inverse_frequency",
        )


@pytest.mark.parametrize(
    ("semantics", "emphasis"),
    (("static_chop", 1.0), (STATIC_STATE_SEMANTICS, float("nan")), (-1, -1.0)),
)
def test_agent_rejects_invalid_selectivity_semantics_or_emphasis(
    semantics,
    emphasis: float,
) -> None:
    with pytest.raises(ValueError, match="semantics"):
        _agent(
            seed=139,
            selectivity_weight=1.0,
            selectivity_semantics=semantics,
            persistent_chop_negative_emphasis=emphasis,
        )


def test_persistent_chop_semantics_requires_equal_action_groups() -> None:
    with pytest.raises(ValueError, match="equal Long/Short"):
        _agent(
            seed=139,
            selectivity_weight=1.0,
            side_balance="none",
            selectivity_semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        )


def test_target_compilation_does_not_synchronize_through_tensor_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_item(*args, **kwargs):
        raise AssertionError("hot-loop tensor.item synchronization is forbidden")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)
    result = _selectivity().target_probabilities(
        _teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        )[None],
        torch.tensor([0.1]),
        torch.tensor([int(Action.ENTER_LONG_1)]),
    )

    assert result.shape == (1, 3)


def test_exact_wait_label_cannot_become_entry_under_strong_teacher_scores() -> None:
    targets = _selectivity().target_probabilities(
        _teacher_row(
            long_attempt=0.99,
            long_clean=0.99,
            short_attempt=0.99,
            short_clean=0.99,
            chop=0.01,
            neutral=0.09,
            trend=0.90,
        )[None],
        torch.tensor([1.0]),
        torch.tensor([int(Action.WAIT)]),
    )

    torch.testing.assert_close(
        targets,
        torch.tensor([[1.0, 0.0, 0.0]]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("target", "declared_index", "opposite_index"),
    (
        (Action.ENTER_LONG_1, 1, 2),
        (Action.ENTER_SHORT_1, 2, 1),
    ),
)
def test_positive_label_has_nonzero_declared_response_and_zero_opposite_mass(
    target: Action,
    declared_index: int,
    opposite_index: int,
) -> None:
    probabilities = _selectivity().target_probabilities(
        _teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.9,
            short_clean=0.8,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        )[None],
        torch.tensor([0.5]),
        torch.tensor([int(target)]),
    )

    assert probabilities[0, declared_index] > 0.0
    assert probabilities[0, opposite_index] == 0.0
    assert probabilities.sum() == pytest.approx(1.0)


def test_unlabeled_row_cannot_activate_selectivity_supervision() -> None:
    agent = _agent(seed=101, selectivity_weight=20.0)
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    unlabeled = Transition(
        observation=np.asarray((1.0, 0.0, 0.0), np.float32),
        action=Action.WAIT,
        reward=0.0,
        next_observation=np.zeros(3, np.float32),
        terminated=True,
        valid_actions=flat,
        next_valid_actions=(),
        teacher_target=_teacher_row(
            long_attempt=0.99,
            long_clean=0.99,
            short_attempt=0.01,
            short_clean=0.01,
            chop=0.01,
            neutral=0.09,
            trend=0.90,
        ).numpy(),
        entry_action_target=None,
        regime_selectivity_headroom_fraction=None,
    )

    agent.train_batch(((unlabeled,), (unlabeled,)))

    assert agent.last_train_metrics["regime_selectivity_loss"] == 0.0
    assert agent.last_train_metrics["regime_selectivity_supervised_rows"] == 0.0


def test_selectivity_diagnostics_are_additive_across_declared_strata() -> None:
    common = {
        "long_attempt": 0.9,
        "long_clean": 0.8,
        "short_attempt": 0.9,
        "short_clean": 0.8,
    }
    chop = _teacher_row(
        **common,
        chop=0.90,
        neutral=0.05,
        trend=0.05,
    )
    nonchop = _teacher_row(
        **common,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    sequences = (
        _sequence((1.0, 0.0, 0.0), chop, headroom=0.10, target=Action.ENTER_LONG_1),
        _sequence((-1.0, 0.0, 0.0), chop, headroom=0.90, target=Action.ENTER_SHORT_1),
        _sequence((0.0, 1.0, 0.0), nonchop, headroom=0.10, target=Action.ENTER_LONG_1),
        _sequence((0.0, -1.0, 0.0), nonchop, headroom=0.90, target=Action.ENTER_SHORT_1),
    )
    agent = _agent(seed=127, selectivity_weight=1.0)

    agent.train_batch(sequences)
    metrics = agent.last_train_metrics

    assert metrics["regime_selectivity_positive_long_short_rows"] == 4.0
    assert metrics["regime_selectivity_dominant_chop_rows"] == 2.0
    assert metrics["regime_selectivity_nonchop_rows"] == 2.0
    assert metrics["regime_selectivity_low_headroom_le_0_25_rows"] == 2.0
    assert metrics["regime_selectivity_safe_headroom_ge_0_75_rows"] == 2.0
    assert (
        metrics["regime_selectivity_dominant_chop_target_wait_probability_mean"]
        > metrics["regime_selectivity_nonchop_target_wait_probability_mean"]
    )
    assert (
        metrics[
            "regime_selectivity_low_headroom_le_0_25_target_wait_probability_mean"
        ]
        > metrics[
            "regime_selectivity_safe_headroom_ge_0_75_target_wait_probability_mean"
        ]
    )
    for stratum in (
        "positive_long_short",
        "dominant_chop",
        "nonchop",
        "low_headroom_le_0_25",
        "safe_headroom_ge_0_75",
    ):
        rows = metrics[f"regime_selectivity_{stratum}_rows"]
        assert metrics[
            f"regime_selectivity_{stratum}_target_wait_probability_sum"
        ] == pytest.approx(
            rows
            * metrics[
                f"regime_selectivity_{stratum}_target_wait_probability_mean"
            ]
        )
        assert metrics[
            f"regime_selectivity_{stratum}_model_wait_probability_sum"
        ] == pytest.approx(
            rows
            * metrics[
                f"regime_selectivity_{stratum}_model_wait_probability_mean"
            ]
        )
        assert metrics[
            f"regime_selectivity_{stratum}_declared_side_probability_sum"
        ] > 0.0


def test_regime_diagnostics_vectorize_exact_channels_and_exclude_trend() -> None:
    all_channels = (*CHANNELS, *TREND_CHANNELS)
    agent = RecurrentC51Agent(
        3,
        hidden_dim=16,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        device="cpu",
        seed=151,
        teacher_channels=len(all_channels),
        teacher_channel_names=all_channels,
        teacher_loss_weight=1e-6,
        regime_selectivity_loss_weight=1.0,
        regime_selectivity_expansion_centers=(0.10, 0.10),
    )
    teacher = np.concatenate((
        _teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.9,
            neutral=0.05,
            trend=0.05,
        ).numpy(),
        np.asarray((0.2, 0.3, 0.4, 0.5), np.float32),
    ))
    with torch.no_grad():
        assert agent.online.teacher_output is not None
        agent.online.teacher_output.weight.zero_()
        agent.online.teacher_output.bias.zero_()
        agent.online.output.weight.zero_()
        agent.online.output.bias.zero_()
        agent.online.output.bias[agent.atoms - 1] = 20.0
    sequence = _sequence(
        (1.0, 0.0, 0.0),
        torch.from_numpy(teacher),
        headroom=0.10,
        target=Action.ENTER_LONG_1,
    )

    agent.train_batch((sequence, sequence))
    metrics = agent.last_train_metrics

    assert agent.regime_teacher_channel_names == REGIME_CHANNELS
    assert metrics[
        "regime_teacher_channel_chop_no_trend_probability_rows"
    ] == 2.0
    assert metrics[
        "regime_teacher_channel_chop_no_trend_probability_"
        "target_probability_sum"
    ] == pytest.approx(1.8)
    assert metrics[
        "regime_teacher_channel_chop_no_trend_probability_"
        "model_probability_sum"
    ] == pytest.approx(1.0)
    assert metrics[
        "regime_teacher_channel_chop_no_trend_probability_absolute_error_sum"
    ] == pytest.approx(0.8)
    assert not any(
        f"regime_teacher_channel_{channel}_" in key
        for channel in TREND_CHANNELS
        for key in metrics
    )
    assert metrics[
        "regime_selectivity_positive_long_target_long_predicted_wait_rows"
    ] == 2.0
    assert metrics["regime_selectivity_positive_long_accuracy"] == 0.0


def test_padding_is_excluded_and_zero_strata_are_explicit() -> None:
    agent = _agent(seed=131, selectivity_weight=1.0)
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    teacher = _teacher_row(
        long_attempt=0.9,
        long_clean=0.8,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    ).numpy()
    sequence = (
        Transition(
            observation=np.asarray((1.0, 0.0, 0.0), np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.asarray((0.0, 1.0, 0.0), np.float32),
            terminated=True,
            valid_actions=flat,
            next_valid_actions=(),
            teacher_target=teacher,
            entry_action_target=Action.ENTER_LONG_1,
            regime_selectivity_headroom_fraction=0.10,
        ),
        Transition(
            observation=np.asarray((0.0, 1.0, 0.0), np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.zeros(3, np.float32),
            terminated=True,
            valid_actions=flat,
            next_valid_actions=(),
            teacher_target=teacher,
            entry_action_target=Action.ENTER_SHORT_1,
            regime_selectivity_headroom_fraction=0.90,
            training_valid=False,
        ),
    )

    agent.train_batch((sequence, sequence))
    metrics = agent.last_train_metrics

    assert metrics["regime_selectivity_positive_long_short_rows"] == 2.0
    assert metrics["regime_selectivity_low_headroom_le_0_25_rows"] == 2.0
    assert metrics["regime_selectivity_safe_headroom_ge_0_75_rows"] == 0.0
    for suffix in (
        "target_wait_probability_sum",
        "target_wait_probability_mean",
        "model_wait_probability_sum",
        "model_wait_probability_mean",
        "greedy_wait_rows",
        "greedy_wait_rate",
        "declared_side_probability_sum",
        "declared_side_probability_mean",
        "greedy_entry_rows",
        "greedy_entry_rate",
    ):
        assert (
            metrics[
                f"regime_selectivity_safe_headroom_ge_0_75_{suffix}"
            ]
            == 0.0
        )


def test_exact_wait_rows_are_not_reweighted_by_regime_selectivity() -> None:
    """Balanced hard Entry CE remains the only supervisor for exact WAIT rows."""
    agent = _agent(
        seed=109,
        selectivity_weight=20.0,
        entry_action_weight=1.0,
    )
    sequence = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.99,
            long_clean=0.99,
            short_attempt=0.01,
            short_clean=0.01,
            chop=0.01,
            neutral=0.09,
            trend=0.90,
        ),
        headroom=1.0,
        target=Action.WAIT,
    )

    agent.train_batch((sequence, sequence))

    assert agent.last_train_metrics["regime_selectivity_loss"] == 0.0
    assert agent.last_train_metrics["regime_selectivity_supervised_rows"] == 0.0
    assert agent.last_train_metrics["entry_action_target_wait_rows"] == 2.0


def test_selectivity_never_trains_on_the_n_step_tail() -> None:
    """Soft confluence is aligned to the same authentic rows as hard Entry CE."""
    agent = _agent(seed=113, selectivity_weight=20.0, n_step_return=2)
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    teacher = _teacher_row(
        long_attempt=0.9,
        long_clean=0.8,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    ).numpy()
    sequence = (
        Transition(
            observation=np.asarray((1.0, 0.0, 0.0), np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.asarray((0.0, 1.0, 0.0), np.float32),
            terminated=False,
            valid_actions=flat,
            next_valid_actions=flat,
            teacher_target=teacher,
        ),
        Transition(
            observation=np.asarray((0.0, 1.0, 0.0), np.float32),
            action=Action.WAIT,
            reward=1.0,
            next_observation=np.zeros(3, np.float32),
            terminated=True,
            valid_actions=flat,
            next_valid_actions=(),
            teacher_target=teacher,
            entry_action_target=Action.ENTER_LONG_1,
            regime_selectivity_headroom_fraction=0.1,
        ),
    )

    agent.train_batch((sequence, sequence))

    assert agent.last_train_metrics["regime_selectivity_loss"] == 0.0
    assert agent.last_train_metrics["regime_selectivity_supervised_rows"] == 0.0
    assert agent.last_train_metrics["entry_action_supervised_rows"] == 0.0


def test_zero_exact_and_imitation_scales_have_exact_rl_update_parity() -> None:
    control = _agent(seed=97, selectivity_weight=0.0)
    taught = _agent(
        seed=97,
        selectivity_weight=20.0,
        side_balance="equal_long_short_v1",
    )
    sequence = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        ),
        headroom=0.1,
        target=Action.ENTER_LONG_1,
    )

    control.train_batch(
        (sequence, sequence),
        teacher_weight_scale=0.0,
        entry_action_weight_scale=0.0,
    )
    taught.train_batch(
        (sequence, sequence),
        teacher_weight_scale=0.0,
        entry_action_weight_scale=0.0,
    )

    assert taught.last_train_metrics["regime_selectivity_loss"] == 0.0
    assert taught.last_train_metrics["regime_selectivity_supervised_rows"] == 0.0
    for name, value in control.online.state_dict().items():
        torch.testing.assert_close(
            value, taught.online.state_dict()[name], rtol=0, atol=0
        )


def test_soft_selectivity_learns_wait_long_short_then_discards_and_round_trips(
    tmp_path,
) -> None:
    medium = _teacher_row(
        long_attempt=0.4,
        long_clean=0.4,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    strong_long = _teacher_row(
        long_attempt=0.9,
        long_clean=0.8,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    strong_short = _teacher_row(
        long_attempt=0.1,
        long_clean=0.1,
        short_attempt=0.9,
        short_clean=0.8,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    sequences = (
        _sequence(
            (0.0, 1.0, 0.0), medium,
            headroom=0.1, target=Action.ENTER_LONG_1,
        ),
        _sequence(
            (1.0, 0.0, 0.0), strong_long,
            headroom=0.1, target=Action.ENTER_LONG_1,
        ),
        _sequence(
            (-1.0, 0.0, 0.0), strong_short,
            headroom=0.1, target=Action.ENTER_SHORT_1,
        ),
    )
    agent = _agent(
        seed=103,
        selectivity_weight=20.0,
        side_balance="equal_long_short_v1",
    )

    for _ in range(100):
        agent.train_batch(sequences)

    observations = (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    expected = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert tuple(_greedy(agent, row) for row in observations) == expected
    assert agent.last_train_metrics["regime_selectivity_loss"] > 0.0
    assert agent.last_train_metrics["regime_selectivity_supervised_rows"] == 3.0
    before = tuple(_greedy(agent, row) for row in observations)
    resumable = agent.save(tmp_path / "balanced-stage2a-resume.pt", manifest={})
    resumed, _ = RecurrentC51Agent.load(resumable, device="cpu")

    assert resumed.regime_selectivity_side_balance == "equal_long_short_v1"
    assert tuple(_greedy(resumed, row) for row in observations) == before

    agent.discard_teacher()
    after = tuple(_greedy(agent, row) for row in observations)
    checkpoint = agent.save(tmp_path / "teacher-free-stage2a.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")

    assert agent.regime_selectivity is None
    assert agent.regime_selectivity_loss_weight == 0.0
    assert agent.regime_selectivity_side_balance == "none"
    assert after == before
    assert tuple(_greedy(restored, row) for row in observations) == before


def test_equal_side_reduction_prevents_short_from_being_lost_in_long_heavy_rows(
) -> None:
    """A sampled side must not lose its gradient merely because Long dominates."""
    medium_long = _teacher_row(
        long_attempt=0.4,
        long_clean=0.4,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    strong_long = _teacher_row(
        long_attempt=0.9,
        long_clean=0.8,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    strong_short = _teacher_row(
        long_attempt=0.1,
        long_clean=0.1,
        short_attempt=0.9,
        short_clean=0.8,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    wait = _sequence(
        (0.0, 1.0, 0.0),
        medium_long,
        headroom=0.1,
        target=Action.ENTER_LONG_1,
    )
    enter_long = _sequence(
        (1.0, 0.0, 0.0),
        strong_long,
        headroom=0.1,
        target=Action.ENTER_LONG_1,
    )
    enter_short = _sequence(
        (-1.0, 0.0, 0.0),
        strong_short,
        headroom=0.1,
        target=Action.ENTER_SHORT_1,
    )
    # This literal 56:1 Long:Short fixture reproduces the Stage 2 learning
    # boundary without stochastic replay sampling.
    long_heavy_batch = (wait,) * 32 + (enter_long,) * 24 + (enter_short,)

    legacy = _agent(seed=211, selectivity_weight=20.0)
    for _ in range(10):
        legacy.train_batch(long_heavy_batch)
    assert tuple(
        _greedy(legacy, row)
        for row in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    ) == (Action.WAIT, Action.ENTER_LONG_1, Action.WAIT)

    balanced = _agent(
        seed=211,
        selectivity_weight=20.0,
        side_balance="equal_long_short_v1",
    )
    for _ in range(10):
        balanced.train_batch(long_heavy_batch)

    assert tuple(
        _greedy(balanced, row)
        for row in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    ) == (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    assert balanced.last_train_metrics["regime_selectivity_positive_long_rows"] == 56.0
    assert balanced.last_train_metrics["regime_selectivity_positive_short_rows"] == 1.0


def test_equal_side_reduction_uses_the_present_side_when_short_is_absent() -> None:
    teacher = _teacher_row(
        long_attempt=0.9,
        long_clean=0.8,
        short_attempt=0.1,
        short_clean=0.1,
        chop=0.05,
        neutral=0.10,
        trend=0.85,
    )
    batch = (
        _sequence(
            (1.0, 0.0, 0.0),
            teacher,
            headroom=0.1,
            target=Action.ENTER_LONG_1,
        ),
    ) * 4
    legacy = _agent(seed=223, selectivity_weight=20.0)
    balanced = _agent(
        seed=223,
        selectivity_weight=20.0,
        side_balance="equal_long_short_v1",
    )

    legacy.train_batch(batch)
    balanced.train_batch(batch)

    assert balanced.last_train_metrics["regime_selectivity_loss"] == pytest.approx(
        legacy.last_train_metrics["regime_selectivity_loss"]
    )
    assert balanced.last_train_metrics["regime_selectivity_positive_long_rows"] == 4.0
    assert balanced.last_train_metrics["regime_selectivity_positive_short_rows"] == 0.0
    assert balanced.last_train_metrics["regime_selectivity_positive_long_loss"] > 0.0
    assert balanced.last_train_metrics["regime_selectivity_positive_short_loss"] == 0.0
    for name, value in legacy.online.state_dict().items():
        torch.testing.assert_close(
            value,
            balanced.online.state_dict()[name],
            rtol=0,
            atol=0,
        )


def test_persistent_chop_objective_learns_wait_and_retains_both_entry_sides(
    tmp_path: Path,
) -> None:
    persistent_chop = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.0,
    )
    transition_long = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.95,
        long_attempt=0.95,
        long_clean=0.95,
    )
    transition_short = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.95,
        short_attempt=0.95,
        short_clean=0.95,
    )
    rows = (
        _sequence(
            (0.0, 1.0, 0.0),
            persistent_chop,
            headroom=1.0,
            target=Action.WAIT,
        ),
        _sequence(
            (1.0, 0.0, 0.0),
            transition_long,
            headroom=1.0,
            target=Action.ENTER_LONG_1,
        ),
        _sequence(
            (-1.0, 0.0, 0.0),
            transition_short,
            headroom=1.0,
            target=Action.ENTER_SHORT_1,
        ),
    )
    control = _agent(seed=229, selectivity_weight=0.0)
    taught = _agent(
        seed=229,
        selectivity_weight=20.0,
        side_balance="equal_long_short_v1",
        selectivity_semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        persistent_chop_negative_emphasis=4.0,
    )

    for _ in range(12):
        control.train_batch(rows)
        taught.train_batch(rows)

    observations = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    expected = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    assert tuple(_greedy(taught, row) for row in observations) == expected
    assert _greedy(control, observations[0]) != Action.WAIT
    assert taught.last_train_metrics["regime_selectivity_exact_wait_loss"] > 0.0
    assert taught.last_train_metrics["regime_selectivity_positive_long_loss"] > 0.0
    assert taught.last_train_metrics["regime_selectivity_positive_short_loss"] > 0.0
    assert taught.last_train_metrics[
        "regime_selectivity_transition_positive_long_declared_side_probability_mean"
    ] > 0.5
    assert taught.last_train_metrics[
        "regime_selectivity_transition_positive_short_declared_side_probability_mean"
    ] > 0.5

    resumable = taught.save(tmp_path / "persistent-chop-resume.pt", manifest={})
    resumed, _ = RecurrentC51Agent.load(resumable, device="cpu")
    assert resumed.regime_selectivity_semantics == (
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
    )
    assert resumed.regime_selectivity_persistent_chop_negative_emphasis == 4.0
    assert tuple(_greedy(resumed, row) for row in observations) == expected

    taught.discard_teacher()
    assert taught.regime_selectivity_semantics == STATIC_STATE_SEMANTICS
    assert tuple(_greedy(taught, row) for row in observations) == expected
    teacher_free = taught.save(tmp_path / "persistent-chop-free.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(teacher_free, device="cpu")
    assert tuple(_greedy(restored, row) for row in observations) == expected


def _hostile_stage2_v2_entry_rows() -> tuple[
    tuple[float, float],
    tuple[tuple[Transition, ...], ...],
]:
    centers = (0.10249102659218842, 0.10399580328775007)
    hostile = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.0,
        long_attempt=0.3391180487263731,
        long_clean=0.3391180487263731,
        short_attempt=0.20604401637062572,
        short_clean=0.20604401637062572,
    )
    rows = (
        *(
            _sequence(
                (0.0, 1.0, 0.0),
                hostile,
                headroom=1.0,
                target=Action.WAIT,
            )
            for _ in range(10)
        ),
        *(
            _sequence(
                (1.0, 0.0, 0.0),
                hostile,
                headroom=1.0,
                target=Action.ENTER_LONG_1,
            )
            for _ in range(10)
        ),
        *(
            _sequence(
                (-1.0, 0.0, 0.0),
                hostile,
                headroom=1.0,
                target=Action.ENTER_SHORT_1,
            )
            for _ in range(10)
        ),
    )
    return centers, rows


def test_static_regime_objective_uses_expansion_anchored_dead_chop() -> None:
    centers, rows = _hostile_stage2_v2_entry_rows()
    policy = _agent(
        seed=271,
        selectivity_weight=0.3,
        entry_action_weight=0.3,
        entry_action_loss_reduction="equal_present_class_mean_v1",
        side_balance="equal_long_short_v1",
        expansion_centers=centers,
    )

    for _ in range(6):
        policy.train_batch(rows)

    assert policy.entry_action_loss_weight == 0.3
    assert policy.regime_selectivity_loss_weight == 0.3
    for side in ("wait", "long", "short"):
        assert policy.last_train_metrics[
            f"entry_balance_{side}_weighted_mass"
        ] == pytest.approx(1.0 / 3.0)
    assert policy.last_train_metrics[
        "regime_entry_conflict_long_target_wait_probability_mean"
    ] == pytest.approx(0.8416820957958787, abs=1e-6)
    assert policy.last_train_metrics[
        "regime_entry_conflict_short_target_wait_probability_mean"
    ] == pytest.approx(0.9406073169730343, abs=1e-6)
    observations = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert tuple(_greedy(policy, row) for row in observations) == (
        Action.WAIT,
        Action.WAIT,
        Action.WAIT,
    )


def test_negative_only_regime_objective_survives_hostile_short_teacher(
    tmp_path: Path,
) -> None:
    """Regime evidence may strengthen WAIT but never relabel a true Entry."""
    centers, rows = _hostile_stage2_v2_entry_rows()
    policy = _agent(
        seed=271,
        selectivity_weight=0.3,
        entry_action_weight=0.3,
        entry_action_loss_reduction="equal_present_class_mean_v1",
        side_balance="equal_long_short_v1",
        selectivity_semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        persistent_chop_negative_emphasis=1.0,
        expansion_centers=centers,
    )

    for _ in range(6):
        policy.train_batch(rows)

    for side in ("long", "short"):
        assert policy.last_train_metrics[
            f"regime_entry_conflict_{side}_target_wait_probability_mean"
        ] == 0.0
        assert policy.last_train_metrics[
            f"regime_entry_conflict_{side}_target_declared_side_probability_mean"
        ] == 1.0
        assert policy.last_train_metrics[
            f"regime_entry_conflict_{side}_soft_wait_disagreement_rows"
        ] == 0.0

    observations = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    expected = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    assert tuple(_greedy(policy, row) for row in observations) == expected

    guided = policy.save(tmp_path / "hostile-short-guided.pt", manifest={})
    resumed, _ = RecurrentC51Agent.load(guided, device="cpu")
    assert resumed.regime_selectivity_semantics == (
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
    )
    assert tuple(_greedy(resumed, row) for row in observations) == expected

    zero_entry_updates = 0
    for update in range(1, 289):
        progress = 0.80 + 0.20 * update / 288.0
        entry_action_weight_scale = max(0.0, 1.0 - progress / 0.95)
        zero_entry_updates += int(entry_action_weight_scale == 0.0)
        resumed.train_batch(
            rows,
            teacher_weight_scale=0.0,
            entry_action_weight_scale=entry_action_weight_scale,
        )
        assert tuple(
            _greedy(resumed, row) for row in observations
        ) == expected
    # Inclusive sampling hits the 95% boundary on update 216, so 73 of these
    # 288 tail updates exercise the fully autonomous contract.
    assert zero_entry_updates == 73

    resumed.discard_teacher()
    assert resumed.teacher_channels == 0
    assert tuple(
        _greedy(resumed, row) for row in observations
    ) == expected
    teacher_free = resumed.save(
        tmp_path / "hostile-short-teacher-free.pt",
        manifest={},
    )
    restored, _ = RecurrentC51Agent.load(teacher_free, device="cpu")
    assert tuple(
        _greedy(restored, row) for row in observations
    ) == expected


def test_persistent_chop_diagnostics_are_additive_without_thresholds() -> None:
    dead_chop = _transition_teacher_row(
        chop_persistence=0.90,
        transition_readiness=0.0,
    )
    transition_ready = _transition_teacher_row(
        chop_persistence=0.90,
        transition_readiness=0.80,
    )
    rows = (
        _sequence(
            (0.0, 1.0, 0.0),
            dead_chop,
            headroom=1.0,
            target=Action.WAIT,
        ),
        _sequence(
            (0.0, -1.0, 0.0),
            transition_ready,
            headroom=1.0,
            target=Action.WAIT,
        ),
        _sequence(
            (1.0, 0.0, 0.0),
            transition_ready,
            headroom=1.0,
            target=Action.ENTER_LONG_1,
        ),
    )
    agent = _agent(
        seed=233,
        selectivity_weight=1.0,
        side_balance="equal_long_short_v1",
        selectivity_semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        persistent_chop_negative_emphasis=4.0,
    )

    agent.train_batch(rows)
    metrics = agent.last_train_metrics

    # The Expansion-anchored classes directly encode dead chop (.90/.18) and
    # transition-or-expansion readiness (.10/.82), without Kaufman proxies.
    assert metrics["regime_selectivity_exact_wait_rows"] == 2.0
    assert metrics["regime_selectivity_exact_wait_weight_sum"] == pytest.approx(
        4.6 + 1.72,
    )
    assert metrics["regime_selectivity_exact_wait_weight_mean"] == pytest.approx(
        (4.6 + 1.72) / 2.0,
    )
    assert metrics["regime_selectivity_persistent_chop_weight_sum"] == pytest.approx(
        metrics["regime_selectivity_exact_wait_weight_sum"]
    )
    assert metrics["regime_selectivity_persistent_dead_chop_rows"] == pytest.approx(
        0.9 + 0.18,
    )
    assert metrics[
        "regime_selectivity_persistent_dead_chop_weight_sum"
    ] == pytest.approx(0.9 * 4.6 + 0.18 * 1.72)
    assert metrics["regime_selectivity_transition_ready_rows"] == pytest.approx(
        0.1 + 0.82,
    )
    assert metrics[
        "regime_selectivity_transition_ready_weight_sum"
    ] == pytest.approx(0.1 * 4.6 + 0.82 * 1.72)
    assert metrics["regime_selectivity_transition_positive_long_rows"] == (
        pytest.approx(0.82)
    )


def test_persistent_chop_objective_learns_same_label_wait_regime_contrast() -> None:
    dead_chop = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.0,
    )
    transition_ready = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.95,
    )
    rows = (
        _sequence(
            (0.0, 1.0, 0.0),
            dead_chop,
            headroom=1.0,
            target=Action.WAIT,
        ),
        _sequence(
            (0.0, -1.0, 0.0),
            transition_ready,
            headroom=1.0,
            target=Action.WAIT,
        ),
    )
    control = _agent(seed=241, selectivity_weight=0.0)
    taught = _agent(
        seed=241,
        selectivity_weight=20.0,
        side_balance="equal_long_short_v1",
        selectivity_semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        persistent_chop_negative_emphasis=8.0,
    )

    for _ in range(8):
        control.train_batch(rows)
        taught.train_batch(rows)

    dead_mean = taught.last_train_metrics[
        "regime_selectivity_persistent_dead_chop_model_wait_probability_mean"
    ]
    ready_mean = taught.last_train_metrics[
        "regime_selectivity_transition_ready_model_wait_probability_mean"
    ]
    def observed_wait_probability(
        policy: RecurrentC51Agent,
        observation: tuple[float, ...],
    ) -> tuple[Action, float]:
        action, _, values = policy.select_action(
            np.asarray(observation, np.float32),
            hidden=None,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            epsilon=0.0,
            return_action_values=True,
        )
        assert values is not None
        flat = values[[
            int(Action.WAIT),
            int(Action.ENTER_LONG_1),
            int(Action.ENTER_SHORT_1),
        ]]
        normalized = np.exp(flat - flat.max())
        return action, float(normalized[0] / normalized.sum())

    control_dead, control_dead_probability = observed_wait_probability(
        control, (0.0, 1.0, 0.0)
    )
    control_ready, control_ready_probability = observed_wait_probability(
        control, (0.0, -1.0, 0.0)
    )
    taught_dead, taught_dead_probability = observed_wait_probability(
        taught, (0.0, 1.0, 0.0)
    )
    taught_ready, taught_ready_probability = observed_wait_probability(
        taught, (0.0, -1.0, 0.0)
    )

    assert dead_mean > ready_mean
    assert taught.last_train_metrics[
        "regime_selectivity_transition_ready_rows"
    ] > 0.0
    assert (control_dead, control_ready) == (
        Action.ENTER_LONG_1,
        Action.ENTER_LONG_1,
    )
    assert (taught_dead, taught_ready) == (Action.WAIT, Action.WAIT)
    assert taught_dead_probability - taught_ready_probability > 0.04
    assert abs(control_dead_probability - control_ready_probability) < 0.01


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_persistent_chop_optimizer_update_runs_on_mps() -> None:
    dead_chop = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.0,
    )
    transition_ready = _transition_teacher_row(
        chop_persistence=0.95,
        transition_readiness=0.95,
    )
    agent = _agent(
        seed=251,
        selectivity_weight=1.0,
        side_balance="equal_long_short_v1",
        selectivity_semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        persistent_chop_negative_emphasis=4.0,
        device="mps",
    )
    rows = (
        _sequence(
            (0.0, 1.0, 0.0),
            dead_chop,
            headroom=1.0,
            target=Action.WAIT,
        ),
        _sequence(
            (1.0, 0.0, 0.0),
            transition_ready,
            headroom=1.0,
            target=Action.ENTER_LONG_1,
        ),
        _sequence(
            (-1.0, 0.0, 0.0),
            transition_ready,
            headroom=1.0,
            target=Action.ENTER_SHORT_1,
        ),
    )

    loss = agent.train_batch(rows)

    assert np.isfinite(loss)
    assert agent.last_train_metrics["regime_selectivity_exact_wait_rows"] == 1.0
    assert agent.last_train_metrics[
        "regime_selectivity_transition_positive_long_rows"
    ] > 0.0
    assert agent.last_train_metrics[
        "regime_selectivity_transition_positive_short_rows"
    ] > 0.0


def test_static_selectivity_reports_zero_persistent_chop_diagnostics() -> None:
    agent = _agent(seed=239, selectivity_weight=1.0)
    agent.train_batch((
        _sequence(
            (1.0, 0.0, 0.0),
            _teacher_row(
                long_attempt=0.9,
                long_clean=0.8,
                short_attempt=0.1,
                short_clean=0.1,
                chop=0.05,
                neutral=0.10,
                trend=0.85,
            ),
            headroom=1.0,
            target=Action.ENTER_LONG_1,
        ),
    ))

    for name in (
        "regime_selectivity_exact_wait_rows",
        "regime_selectivity_exact_wait_weight_sum",
        "regime_selectivity_exact_wait_weight_mean",
        "regime_selectivity_exact_wait_model_wait_probability_sum",
        "regime_selectivity_exact_wait_model_wait_probability_mean",
        "regime_selectivity_persistent_chop_weight_sum",
        "regime_selectivity_persistent_chop_weight_mean",
        "regime_selectivity_persistent_dead_chop_rows",
        "regime_selectivity_persistent_dead_chop_weight_sum",
        "regime_selectivity_persistent_dead_chop_weight_mean",
        "regime_selectivity_persistent_dead_chop_model_wait_probability_sum",
        "regime_selectivity_persistent_dead_chop_model_wait_probability_mean",
        "regime_selectivity_transition_ready_rows",
        "regime_selectivity_transition_ready_weight_sum",
        "regime_selectivity_transition_ready_weight_mean",
        "regime_selectivity_transition_ready_model_wait_probability_sum",
        "regime_selectivity_transition_ready_model_wait_probability_mean",
        "regime_selectivity_transition_positive_long_rows",
        "regime_selectivity_transition_positive_long_declared_side_probability_sum",
        "regime_selectivity_transition_positive_long_declared_side_probability_mean",
        "regime_selectivity_transition_positive_short_rows",
        "regime_selectivity_transition_positive_short_declared_side_probability_sum",
        "regime_selectivity_transition_positive_short_declared_side_probability_mean",
    ):
        assert agent.last_train_metrics[name] == 0.0


def test_episode_and_training_summary_preserve_row_additive_diagnostics(
    tmp_path: Path,
) -> None:
    update_metrics = {
        "regime_selectivity_positive_long_short_rows": [2.0, 3.0],
        "regime_selectivity_positive_long_short_target_wait_probability_sum": [
            1.5,
            1.0,
        ],
        "regime_selectivity_positive_long_short_model_wait_probability_sum": [
            0.5,
            1.5,
        ],
        "regime_selectivity_positive_long_short_greedy_wait_rows": [1.0, 1.0],
        "regime_selectivity_positive_long_short_declared_side_probability_sum": [
            1.0,
            2.0,
        ],
        "regime_selectivity_positive_long_short_greedy_entry_rows": [1.0, 2.0],
    }
    first = _regime_selectivity_episode_diagnostic(update_metrics)
    second = _regime_selectivity_episode_diagnostic({})

    assert {
        key: first["positive_long_short"][key]
        for key in (
            "rows",
            "target_wait_probability_sum",
            "target_wait_probability_mean",
            "model_wait_probability_sum",
            "model_wait_probability_mean",
            "greedy_wait_rows",
            "greedy_wait_rate",
            "declared_side_probability_sum",
            "declared_side_probability_mean",
            "greedy_entry_rows",
            "greedy_entry_rate",
        )
    } == pytest.approx({
        "rows": 5,
        "target_wait_probability_sum": 2.5,
        "target_wait_probability_mean": 0.5,
        "model_wait_probability_sum": 2.0,
        "model_wait_probability_mean": 0.4,
        "greedy_wait_rows": 2,
        "greedy_wait_rate": 0.4,
        "declared_side_probability_sum": 3.0,
        "declared_side_probability_mean": 0.6,
        "greedy_entry_rows": 3,
        "greedy_entry_rate": 0.6,
    })
    assert {
        key: second["safe_headroom_ge_0_75"][key]
        for key in (
            "rows",
            "target_wait_probability_sum",
            "target_wait_probability_mean",
            "model_wait_probability_sum",
            "model_wait_probability_mean",
            "greedy_wait_rows",
            "greedy_wait_rate",
            "declared_side_probability_sum",
            "declared_side_probability_mean",
            "greedy_entry_rows",
            "greedy_entry_rate",
        )
    } == {
        "rows": 0,
        "target_wait_probability_sum": 0.0,
        "target_wait_probability_mean": 0.0,
        "model_wait_probability_sum": 0.0,
        "model_wait_probability_mean": 0.0,
        "greedy_wait_rows": 0,
        "greedy_wait_rate": 0.0,
        "declared_side_probability_sum": 0.0,
        "declared_side_probability_mean": 0.0,
        "greedy_entry_rows": 0,
        "greedy_entry_rate": 0.0,
    }

    source = tmp_path / "training-diagnostics.jsonl"
    destination = tmp_path / "training-diagnostic-summary.json"
    rows = (
        {
            "ticker": "NQ",
            "outcome": "pass",
            "regime_selectivity": first,
        },
        {
            "ticker": "NQ",
            "outcome": "timeout",
            "regime_selectivity": second,
        },
    )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    _write_training_diagnostic_summary(source, destination)
    summary = json.loads(destination.read_text())["overall"]["regime_selectivity"]

    assert {
        key: value
        for key, value in summary["positive_long_short"].items()
        if key != "confusion"
    } == pytest.approx({
        key: value
        for key, value in first["positive_long_short"].items()
        if key != "confusion"
    })
    assert summary["positive_long_short"]["confusion"] == first[
        "positive_long_short"
    ]["confusion"]
    assert summary["dominant_chop"] == second["dominant_chop"]
    assert summary["safe_headroom_ge_0_75"] == second[
        "safe_headroom_ge_0_75"
    ]


def test_stage2a_training_gate_requires_rows_side_response_and_wait_deltas() -> None:
    metrics = {
        "short_circuited": 0.0,
        "sampled_entry_action_long_rows": 10.0,
        "sampled_entry_action_short_rows": 10.0,
        "sampled_entry_action_long_recall": 0.5,
        "sampled_entry_action_short_recall": 0.5,
        "regime_selectivity_positive_long_rows": 10.0,
        "regime_selectivity_positive_short_rows": 10.0,
        "regime_selectivity_positive_long_declared_side_probability_sum": 5.0,
        "regime_selectivity_positive_short_declared_side_probability_sum": 5.0,
        "regime_selectivity_dominant_chop_rows": 10.0,
        "regime_selectivity_nonchop_rows": 10.0,
        "final_regime_probe_wait_rows": 32.0,
        "final_regime_probe_long_rows": 32.0,
        "final_regime_probe_short_rows": 32.0,
        "final_regime_probe_long_recall": 0.5,
        "final_regime_probe_short_recall": 0.5,
        "final_regime_probe_wait_recall": 0.75,
        "final_regime_probe_dominant_chop_rows": 16.0,
        "final_regime_probe_nonchop_rows": 16.0,
        "final_regime_probe_chop_minus_nonchop_wait": 0.2,
    }
    gates = _training_evaluation_gates(regime_selectivity_active=True)

    assert all(gate.passes(metrics) for gate in gates)
    metrics["final_regime_probe_chop_minus_nonchop_wait"] = 0.0
    assert not all(gate.passes(metrics) for gate in gates)

    for action in ("wait", "long", "short"):
        metrics[f"entry_balance_{action}_rows"] = 10.0
        metrics[
            f"entry_balance_{action}_weighted_mass_fraction"
        ] = 1.0 / 3.0
    repair_gates = _training_evaluation_gates(
        regime_selectivity_active=True,
        entry_action_loss_reduction="equal_present_class_mean_v1",
        entry_action_supervision_active=True,
    )
    assert all(gate.passes(metrics) for gate in repair_gates)


def test_active_selectivity_is_plain_in_candidate_contract_and_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = {
        "schema": "balance_aware_regime_selectivity_v1",
        "training_only": True,
        "target_source": "post_launch_entry_action_target",
        "loss_weight": np.float64(0.3),
        "expansion_long_center": np.float64(0.10),
        "expansion_short_center": np.float64(0.11),
        "probability_epsilon": np.float64(1e-6),
        "headroom_pressure": np.float64(1.0),
        "dominant_chop_pressure": np.float64(2.0),
        "q_temperature": np.float64(1.0),
    }
    frozen = _regime_selectivity_frozen_contract(specification)

    assert frozen is not specification
    assert frozen == {
        key: (float(value) if isinstance(value, np.floating) else value)
        for key, value in specification.items()
    }
    json.dumps(frozen)

    cache_root = tmp_path / "cache"
    ticker_root = cache_root / "NQ"
    ticker_root.mkdir(parents=True)
    (ticker_root / "manifest.json").write_text("{}")
    config = materialize_effective_config({
        "_root": str(tmp_path),
        "tickers": ["NQ"],
        "regime_selectivity": specification,
    })
    baseline = _training_resume_identity(config, cache_root, ())
    selectivity_path = Path(__file__).parents[1] / "src" / "propevolve" / (
        "balance_aware_regime_selectivity.py"
    )
    original_read_bytes = Path.read_bytes

    def changed_selectivity(path: Path) -> bytes:
        value = original_read_bytes(path)
        return value + b"\n# changed\n" if path == selectivity_path else value

    monkeypatch.setattr(Path, "read_bytes", changed_selectivity)

    assert _training_resume_identity(config, cache_root, ()) != baseline


@pytest.mark.parametrize(
    "relative_dependency",
    (
        "config.py",
        "decision.py",
        "observation.py",
        "evolution.py",
        "teachers/composition.py",
        "teachers/expansion.py",
        "teachers/regime.py",
    ),
)
def test_training_resume_identity_binds_runtime_semantic_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_dependency: str,
) -> None:
    cache_root = tmp_path / "cache"
    ticker_root = cache_root / "NQ"
    ticker_root.mkdir(parents=True)
    (ticker_root / "manifest.json").write_text("{}")
    config = materialize_effective_config({
        "_root": str(tmp_path),
        "tickers": ["NQ"],
    })
    baseline = _training_resume_identity(config, cache_root, ())
    dependency = (
        Path(__file__).parents[1] / "src" / "propevolve" / relative_dependency
    )
    original_read_bytes = Path.read_bytes

    def changed_dependency(path: Path) -> bytes:
        value = original_read_bytes(path)
        return value + b"\n# changed\n" if path == dependency else value

    monkeypatch.setattr(Path, "read_bytes", changed_dependency)

    assert _training_resume_identity(config, cache_root, ()) != baseline


def test_profitable_headroom_is_bounded_without_rejecting_valid_state() -> None:
    assert _bounded_regime_selectivity_headroom(1.75) == 1.0
    assert _bounded_regime_selectivity_headroom(0.25) == 0.25
    with pytest.raises(ValueError, match="headroom fraction is invalid"):
        _bounded_regime_selectivity_headroom(-0.01)


def test_stage2a_training_update_has_no_new_tensor_item_synchronization() -> None:
    source = Path(__file__).parents[1] / "src" / "propevolve" / "agent.py"
    block = source.read_text().split("def train_batch(", 1)[1].split(
        "def retain_policy(", 1
    )[0]

    assert "learnable_rows.any().item()" not in block
    assert "diagnostic_target_rows.any().item()" not in block
    selectivity_block = block.split(
        "if self.regime_selectivity is not None:", 1
    )[1].split("if self.entry_action_loss_weight", 1)[0]
    assert ".item()" not in selectivity_block


def test_fixed_training_tensors_are_reused_across_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(
        seed=137,
        selectivity_weight=1.0,
        entry_action_weight=1.0,
    )
    fixed_values = {
        id(agent.teacher_channel_loss_weights),
        id(agent.entry_action_class_weights),
    }
    original_as_tensor = torch.as_tensor

    def guard_as_tensor(value, *args, **kwargs):
        if id(value) in fixed_values:
            raise AssertionError("fixed training tensor was recreated")
        return original_as_tensor(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", guard_as_tensor)
    sequence = _sequence(
        (1.0, 0.0, 0.0),
        _teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        ),
        headroom=0.1,
        target=Action.ENTER_LONG_1,
    )

    agent.train_batch((sequence, sequence))
    agent.train_batch((sequence, sequence))


def test_legacy_agent_skips_entry_target_scan_and_bincount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(seed=149, selectivity_weight=0.0, entry_action_weight=0.0)
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    transition = Transition(
        observation=np.asarray((1.0, 0.0, 0.0), np.float32),
        action=Action.WAIT,
        reward=0.0,
        next_observation=np.zeros(3, np.float32),
        terminated=True,
        valid_actions=flat,
        next_valid_actions=(),
        teacher_target=_teacher_row(
            long_attempt=0.9,
            long_clean=0.8,
            short_attempt=0.1,
            short_clean=0.1,
            chop=0.05,
            neutral=0.10,
            trend=0.85,
        ).numpy(),
        # Deliberately malformed: an inactive entry auxiliary must not inspect it.
        entry_action_target=999,
    )

    def fail_bincount(*args, **kwargs):
        raise AssertionError("inactive entry diagnostics invoked bincount")

    monkeypatch.setattr(torch, "bincount", fail_bincount)

    agent.train_batch(((transition,), (transition,)))

    assert agent.last_train_metrics["entry_action_supervised_rows"] == 0.0
    assert agent.last_train_metrics[
        "regime_selectivity_positive_long_short_rows"
    ] == 0.0
