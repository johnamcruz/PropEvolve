from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent
from propevolve.balance_aware_regime_selectivity import (
    PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
)
from propevolve.decision import Action
from propevolve.replay import Transition
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS


CHANNELS = (*EXPANSION_CHANNELS, *REGIME_CHANNELS)
CENTERS = (0.10249102659218842, 0.10399580328775007)
HIERARCHICAL_REDUCTION = "hierarchical_enter_wait_direction_v1"
FLAT_ACTIONS = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)


def _teacher_row(
    *,
    chop_persistence: float,
    readiness: float,
    long_score: float = 0.01,
    short_score: float = 0.01,
) -> np.ndarray:
    values = np.full(len(CHANNELS), 0.1, dtype=np.float32)
    values[:4] = (
        np.sqrt(long_score),
        np.sqrt(long_score),
        np.sqrt(short_score),
        np.sqrt(short_score),
    )
    values[CHANNELS.index("structure_chop_probability")] = chop_persistence
    values[CHANNELS.index("structure_neutral_probability")] = 0.1
    values[CHANNELS.index("structure_trend_probability")] = readiness
    for name in (
        "structure_trend_onset_probability",
        "structure_trend_persistence_probability",
        "volatility_expansion_onset_probability",
        "volatility_high_persistence_probability",
        "kaufman_efficiency",
        "volatility_percentile",
    ):
        values[CHANNELS.index(name)] = readiness
    values[
        CHANNELS.index("structure_chop_persistence_probability")
    ] = chop_persistence
    return values


def _flat_transition(
    observation: tuple[float, float, float],
    *,
    target: Action,
    teacher: np.ndarray | None = None,
    reward: float = 0.0,
) -> tuple[Transition, ...]:
    return (
        Transition(
            observation=np.asarray(observation, dtype=np.float32),
            action=target,
            reward=reward,
            next_observation=np.zeros(3, dtype=np.float32),
            terminated=True,
            valid_actions=FLAT_ACTIONS,
            next_valid_actions=(),
            teacher_target=teacher,
            entry_action_target=target,
            regime_selectivity_headroom_fraction=(
                1.0 if teacher is not None else None
            ),
        ),
    )


def _agent(
    *,
    reduction: str,
    seed: int = 314159,
    association: bool = False,
    entry_weight: float = 0.3,
    learning_rate: float = 0.03,
    device: str = "cpu",
) -> RecurrentC51Agent:
    teacher_settings = {}
    if association:
        teacher_settings = {
            "teacher_channels": len(CHANNELS),
            "teacher_channel_names": CHANNELS,
            "teacher_loss_weight": 1e-6,
            "regime_selectivity_loss_weight": 0.3,
            "regime_selectivity_expansion_centers": CENTERS,
            "regime_selectivity_side_balance": "equal_long_short_v1",
            "regime_selectivity_semantics": (
                PERSISTENT_CHOP_ASSOCIATION_SEMANTICS
            ),
            "regime_selectivity_persistent_chop_negative_emphasis": 1.0,
        }
    return RecurrentC51Agent(
        3,
        hidden_dim=24,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=learning_rate,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        device=device,
        seed=seed,
        entry_action_loss_weight=entry_weight,
        entry_action_loss_reduction=reduction,
        **teacher_settings,
    )


def _probe_actions(
    policy: RecurrentC51Agent,
    rows: tuple[tuple[Transition, ...], ...],
) -> tuple[Action, ...]:
    actions, _ = policy.greedy_sequence_action_values(rows)
    return tuple(Action(value) for value in actions[:, -1])


def test_hierarchical_entry_loss_closes_the_v6_sampled_vs_fixed_wait_gap(
) -> None:
    """A positive sampled association is insufficient if fixed chop still enters."""
    dead = _teacher_row(chop_persistence=0.99, readiness=0.0)
    ready_long = _teacher_row(
        chop_persistence=0.99,
        readiness=0.99,
        long_score=0.90,
    )
    ready_short = _teacher_row(
        chop_persistence=0.99,
        readiness=0.99,
        short_score=0.90,
    )
    sampled = (
        *(
            _flat_transition((0.0, 0.0, 1.0), target=Action.WAIT, teacher=dead)
            for _ in range(16)
        ),
        *(
            _flat_transition((0.50, 0.0, 0.50), target=Action.WAIT, teacher=dead)
            for _ in range(15)
        ),
        _flat_transition(
            (1.0, 0.0, 0.0),
            target=Action.ENTER_LONG_1,
            teacher=ready_long,
            reward=1.0,
        ),
        _flat_transition(
            (-1.0, 0.0, 0.0),
            target=Action.ENTER_SHORT_1,
            teacher=ready_short,
            reward=1.0,
        ),
    )
    sampled_probe = (sampled[0], sampled[-2], sampled[-1])
    fixed_probe = (
        _flat_transition((0.70, 0.0, 0.30), target=Action.WAIT, teacher=dead),
        sampled[-2],
        sampled[-1],
    )

    legacy = _agent(
        reduction="equal_present_class_mean_v1",
        association=True,
    )
    for _ in range(8):
        legacy.train_batch(sampled)

    assert legacy.last_train_metrics[
        "regime_selectivity_dead_wait_minus_"
        "transition_positive_model_wait"
    ] > 0.0
    assert _probe_actions(legacy, sampled_probe) == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert _probe_actions(legacy, fixed_probe)[0] != Action.WAIT

    hierarchical = _agent(
        reduction=HIERARCHICAL_REDUCTION,
        association=True,
    )
    for _ in range(8):
        hierarchical.train_batch(sampled)

    assert hierarchical.last_train_metrics[
        "regime_selectivity_dead_wait_minus_"
        "transition_positive_model_wait"
    ] > 0.0
    assert _probe_actions(hierarchical, fixed_probe) == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )


def test_hierarchical_entry_loss_balances_timing_and_conditional_direction(
) -> None:
    rows = (
        *(
            _flat_transition((0.0, 0.0, 1.0), target=Action.WAIT)
            for _ in range(7)
        ),
        *(
            _flat_transition(
                (1.0, 0.0, 0.0),
                target=Action.ENTER_LONG_1,
                reward=1.0,
            )
            for _ in range(5)
        ),
        _flat_transition(
            (-1.0, 0.0, 0.0),
            target=Action.ENTER_SHORT_1,
            reward=1.0,
        ),
    )
    policy = _agent(
        reduction=HIERARCHICAL_REDUCTION,
        association=False,
        entry_weight=1.0,
    )

    policy.train_batch(rows)

    metrics = policy.last_train_metrics
    assert metrics["entry_timing_wait_rows"] == 7.0
    assert metrics["entry_timing_enter_rows"] == 6.0
    assert metrics["entry_timing_wait_weighted_mass"] == pytest.approx(0.5)
    assert metrics["entry_timing_enter_weighted_mass"] == pytest.approx(0.5)
    assert metrics["entry_timing_wait_weighted_mass_fraction"] == pytest.approx(
        0.5
    )
    assert metrics["entry_timing_enter_weighted_mass_fraction"] == pytest.approx(
        0.5
    )
    assert metrics["entry_direction_long_rows"] == 5.0
    assert metrics["entry_direction_short_rows"] == 1.0
    assert metrics["entry_direction_long_weighted_mass"] == pytest.approx(0.5)
    assert metrics["entry_direction_short_weighted_mass"] == pytest.approx(0.5)
    assert metrics[
        "entry_direction_long_weighted_mass_fraction"
    ] == pytest.approx(0.5)
    assert metrics[
        "entry_direction_short_weighted_mass_fraction"
    ] == pytest.approx(0.5)
    assert metrics["entry_timing_loss"] > 0.0
    assert metrics["entry_direction_loss"] > 0.0
    assert metrics["entry_action_loss"] == pytest.approx(
        metrics["entry_timing_loss"] + metrics["entry_direction_loss"]
    )
    assert (
        metrics["entry_timing_wait_weighted_loss_contribution"]
        + metrics["entry_timing_enter_weighted_loss_contribution"]
    ) == pytest.approx(metrics["entry_timing_loss"])
    assert (
        metrics["entry_direction_long_weighted_loss_contribution"]
        + metrics["entry_direction_short_weighted_loss_contribution"]
    ) == pytest.approx(metrics["entry_direction_loss"])


def test_hierarchical_entry_policy_survives_autonomous_tail_and_round_trip(
    tmp_path: Path,
) -> None:
    rows = (
        _flat_transition((0.75, 0.0, 0.25), target=Action.WAIT),
        _flat_transition((-0.75, 0.0, 0.25), target=Action.WAIT),
        _flat_transition(
            (1.0, 0.0, 0.0),
            target=Action.ENTER_LONG_1,
            reward=1.0,
        ),
        _flat_transition(
            (-1.0, 0.0, 0.0),
            target=Action.ENTER_SHORT_1,
            reward=1.0,
        ),
    )
    expected = (
        Action.WAIT,
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    policy = _agent(
        reduction=HIERARCHICAL_REDUCTION,
        seed=811,
        entry_weight=10.0,
        learning_rate=0.02,
    )

    for _ in range(40):
        policy.train_batch(rows)
    for _ in range(20):
        policy.train_batch(rows, entry_action_weight_scale=0.0)

    assert _probe_actions(policy, rows) == expected
    before_actions, before_values = policy.greedy_sequence_action_values(rows)
    policy.discard_teacher()
    policy.assert_teacher_free()
    after_actions, after_values = policy.greedy_sequence_action_values(rows)
    np.testing.assert_array_equal(after_actions, before_actions)
    np.testing.assert_allclose(after_values, before_values, rtol=0, atol=0)

    checkpoint = policy.save(
        tmp_path / "hierarchical-entry-teacher-free.pt",
        manifest={},
    )
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    restored.assert_teacher_free()
    assert restored.entry_action_loss_reduction == HIERARCHICAL_REDUCTION
    restored_actions, restored_values = restored.greedy_sequence_action_values(rows)
    np.testing.assert_array_equal(restored_actions, before_actions)
    np.testing.assert_allclose(restored_values, before_values, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host",
)
def test_hierarchical_entry_optimizer_update_runs_on_mps() -> None:
    dead = _teacher_row(chop_persistence=0.99, readiness=0.0)
    ready_long = _teacher_row(
        chop_persistence=0.99,
        readiness=0.99,
        long_score=0.90,
    )
    ready_short = _teacher_row(
        chop_persistence=0.99,
        readiness=0.99,
        short_score=0.90,
    )
    rows = (
        _flat_transition((0.0, 0.0, 1.0), target=Action.WAIT, teacher=dead),
        _flat_transition(
            (1.0, 0.0, 0.0),
            target=Action.ENTER_LONG_1,
            teacher=ready_long,
            reward=1.0,
        ),
        _flat_transition(
            (-1.0, 0.0, 0.0),
            target=Action.ENTER_SHORT_1,
            teacher=ready_short,
            reward=1.0,
        ),
    )
    policy = _agent(
        reduction=HIERARCHICAL_REDUCTION,
        association=True,
        device="mps",
    )

    loss = policy.train_batch(rows)
    metrics = policy.last_train_metrics

    assert np.isfinite(loss)
    assert metrics["entry_timing_wait_weighted_mass_fraction"] == pytest.approx(
        0.5
    )
    assert metrics["entry_timing_enter_weighted_mass_fraction"] == pytest.approx(
        0.5
    )
    assert metrics[
        "entry_direction_long_weighted_mass_fraction"
    ] == pytest.approx(0.5)
    assert metrics[
        "entry_direction_short_weighted_mass_fraction"
    ] == pytest.approx(0.5)
    assert metrics[
        "regime_entry_conflict_long_soft_wait_disagreement_rows"
    ] == 0.0
    assert metrics[
        "regime_entry_conflict_short_soft_wait_disagreement_rows"
    ] == 0.0
