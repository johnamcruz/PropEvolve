from __future__ import annotations

import hashlib
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
FLAT_ACTIONS = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
HIERARCHICAL_ENTRY_LOSS = "hierarchical_enter_wait_direction_v1"


def _teacher_context(*, transition_ready: bool) -> np.ndarray:
    """Hold Expansion constant while changing only the Regime transition."""
    values = np.full(len(CHANNELS), 0.1, dtype=np.float32)
    # The dead and ready fixtures deliberately have identical, strong, and
    # direction-ambiguous Expansion evidence. Regime is the only timing clue.
    values[:4] = (0.95, 0.95, 0.95, 0.95)
    readiness = 0.95 if transition_ready else 0.0
    values[CHANNELS.index("structure_chop_probability")] = 0.95
    values[CHANNELS.index("structure_neutral_probability")] = 0.025
    values[CHANNELS.index("structure_trend_probability")] = readiness
    values[CHANNELS.index("structure_chop_persistence_probability")] = 0.95
    for channel in (
        "structure_trend_onset_probability",
        "structure_trend_persistence_probability",
        "volatility_expansion_onset_probability",
        "volatility_high_persistence_probability",
        "kaufman_efficiency",
        "volatility_percentile",
    ):
        values[CHANNELS.index(channel)] = readiness
    return values


def _flat_sequence(
    *,
    direction: float,
    transition_ready: bool,
    target: Action,
) -> tuple[Transition, ...]:
    # Coordinates 0-1 are the look-alike Expansion/side context. Coordinates
    # 2-3 are causal Regime context available to the deployed policy.
    observation = np.asarray(
        (0.95, direction, 0.95, 0.95 if transition_ready else 0.0),
        dtype=np.float32,
    )
    return (
        Transition(
            observation=observation,
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.zeros(4, dtype=np.float32),
            terminated=True,
            valid_actions=FLAT_ACTIONS,
            next_valid_actions=(),
            teacher_target=_teacher_context(transition_ready=transition_ready),
            entry_action_target=target,
            regime_selectivity_headroom_fraction=1.0,
        ),
    )


def _agent(
    *,
    seed: int = 811,
    entry_loss_reduction: str = HIERARCHICAL_ENTRY_LOSS,
) -> RecurrentC51Agent:
    return RecurrentC51Agent(
        4,
        hidden_dim=24,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.02,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        device="cpu",
        seed=seed,
        teacher_channels=len(CHANNELS),
        teacher_channel_names=CHANNELS,
        teacher_loss_weight=0.5,
        entry_action_loss_weight=10.0,
        entry_action_loss_reduction=entry_loss_reduction,
        regime_selectivity_loss_weight=1.0,
        regime_selectivity_expansion_centers=CENTERS,
        regime_selectivity_side_balance="equal_long_short_v1",
        regime_selectivity_semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
        regime_selectivity_persistent_chop_negative_emphasis=1.0,
    )


def _plain_agent(
    *,
    seed: int,
    entry_loss_weight: float = 0.0,
    entry_loss_reduction: str = "population_weighted_mean_v1",
    device: str = "cpu",
) -> RecurrentC51Agent:
    return RecurrentC51Agent(
        4,
        hidden_dim=24,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=0.02,
        weight_decay=0.0,
        gradient_clip=10.0,
        target_sync_updates=250,
        device=device,
        seed=seed,
        entry_action_loss_weight=entry_loss_weight,
        entry_action_loss_reduction=entry_loss_reduction,
    )


def _flat_scores(
    policy: RecurrentC51Agent,
    rows: tuple[tuple[Transition, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    actions, values = policy.greedy_sequence_action_values(rows)
    probabilities = torch.softmax(torch.as_tensor(values[:, 0, :3]), dim=-1)
    return actions[:, 0], probabilities.numpy()


def test_hierarchical_entry_learning_uses_regime_to_separate_lookalike_expansion(
    tmp_path: Path,
) -> None:
    """Dead rotation must become WAIT without vetoing transition-ready entries."""
    rows = (
        _flat_sequence(
            direction=1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
        _flat_sequence(
            direction=-1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
        _flat_sequence(
            direction=1.0,
            transition_ready=True,
            target=Action.ENTER_LONG_1,
        ),
        _flat_sequence(
            direction=-1.0,
            transition_ready=True,
            target=Action.ENTER_SHORT_1,
        ),
    )
    policy = _agent()

    policy.train_batch(rows)
    initial_regime_error = policy.last_train_metrics[
        "regime_teacher_channel_structure_chop_persistence_probability_"
        "mean_absolute_error"
    ] + policy.last_train_metrics[
        "regime_teacher_channel_structure_trend_onset_probability_"
        "mean_absolute_error"
    ]
    for _ in range(79):
        policy.train_batch(rows)

    final_regime_error = policy.last_train_metrics[
        "regime_teacher_channel_structure_chop_persistence_probability_"
        "mean_absolute_error"
    ] + policy.last_train_metrics[
        "regime_teacher_channel_structure_trend_onset_probability_"
        "mean_absolute_error"
    ]
    assert final_regime_error < initial_regime_error
    assert policy.last_train_metrics["entry_timing_loss"] > 0.0
    assert policy.last_train_metrics["entry_direction_loss"] > 0.0
    assert policy.last_train_metrics[
        "regime_entry_conflict_long_soft_wait_disagreement_rows"
    ] == 0.0
    assert policy.last_train_metrics[
        "regime_entry_conflict_short_soft_wait_disagreement_rows"
    ] == 0.0

    actions, probabilities = _flat_scores(policy, rows)
    assert actions.tolist() == [
        int(Action.WAIT),
        int(Action.WAIT),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
    ]
    assert probabilities[0, int(Action.WAIT)] > probabilities[2, int(Action.WAIT)]
    assert probabilities[1, int(Action.WAIT)] > probabilities[3, int(Action.WAIT)]

    policy.discard_teacher()
    policy.assert_teacher_free()
    discarded_actions, discarded_probabilities = _flat_scores(policy, rows)
    np.testing.assert_array_equal(discarded_actions, actions)
    np.testing.assert_allclose(discarded_probabilities, probabilities, rtol=0, atol=0)

    path = policy.save(tmp_path / "hierarchical-regime-teacher-free.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(path, device="cpu")
    restored.assert_teacher_free()
    restored_actions, restored_probabilities = _flat_scores(restored, rows)
    np.testing.assert_array_equal(restored_actions, actions)
    np.testing.assert_allclose(restored_probabilities, probabilities, rtol=0, atol=0)


def test_stage1_warm_start_has_exact_policy_and_shared_shape_parity(
    tmp_path: Path,
) -> None:
    stage1 = _plain_agent(seed=821)
    probes = (
        _flat_sequence(
            direction=1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
        _flat_sequence(
            direction=1.0,
            transition_ready=True,
            target=Action.ENTER_LONG_1,
        ),
        _flat_sequence(
            direction=-1.0,
            transition_ready=True,
            target=Action.ENTER_SHORT_1,
        ),
    )
    stage1_actions, stage1_values = stage1.greedy_sequence_action_values(probes)
    stage1_path = stage1.save(tmp_path / "immutable-stage1.pt", manifest={})
    before_sha = hashlib.sha256(stage1_path.read_bytes()).hexdigest()

    child_config = {
        "observation_dim": 4,
        "hidden_dim": 24,
        "atoms": 11,
        "value_min": -3.0,
        "value_max": 3.0,
        "gamma": 0.997,
        "learning_rate": 0.02,
        "weight_decay": 0.0,
        "gradient_clip": 10.0,
        "target_sync_updates": 250,
        "device": "cpu",
        "seed": 823,
        "teacher_channels": len(CHANNELS),
        "teacher_channel_names": CHANNELS,
        "teacher_loss_weight": 0.5,
        "entry_action_loss_weight": 10.0,
        "entry_action_loss_reduction": HIERARCHICAL_ENTRY_LOSS,
        "regime_selectivity_loss_weight": 1.0,
        "regime_selectivity_expansion_centers": CENTERS,
        "regime_selectivity_side_balance": "equal_long_short_v1",
        "regime_selectivity_semantics": PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
        "regime_selectivity_persistent_chop_negative_emphasis": 1.0,
    }
    child, _ = RecurrentC51Agent.warm_start(stage1_path, config=child_config)

    child_actions, child_values = child.greedy_sequence_action_values(probes)
    np.testing.assert_array_equal(child_actions, stage1_actions)
    np.testing.assert_allclose(child_values, stage1_values, rtol=0, atol=0)
    shared_child_state = {
        name: value
        for name, value in child.online.state_dict().items()
        if not name.startswith("teacher_output.")
    }
    assert shared_child_state.keys() == stage1.online.state_dict().keys()
    for name, stage1_value in stage1.online.state_dict().items():
        assert shared_child_state[name].shape == stage1_value.shape
        torch.testing.assert_close(shared_child_state[name], stage1_value, rtol=0, atol=0)
    assert child.entry_action_loss_reduction == HIERARCHICAL_ENTRY_LOSS
    assert hashlib.sha256(stage1_path.read_bytes()).hexdigest() == before_sha


def test_hierarchical_entry_loss_does_not_directly_target_management_rows() -> None:
    management = (
        Transition(
            observation=np.asarray((0.95, 1.0, 0.0, 0.95), dtype=np.float32),
            action=Action.HOLD,
            reward=0.25,
            next_observation=np.zeros(4, dtype=np.float32),
            terminated=True,
            valid_actions=(Action.HOLD, Action.CLOSE),
            next_valid_actions=(),
        ),
    )
    control = _plain_agent(seed=829)
    hierarchical = _plain_agent(
        seed=829,
        entry_loss_weight=10.0,
        entry_loss_reduction=HIERARCHICAL_ENTRY_LOSS,
    )

    control.train_batch((management, management))
    hierarchical.train_batch((management, management))

    assert hierarchical.last_train_metrics["entry_action_supervised_rows"] == 0.0
    assert hierarchical.last_train_metrics["entry_timing_loss"] == 0.0
    assert hierarchical.last_train_metrics["entry_direction_loss"] == 0.0
    for group in ("wait", "enter"):
        assert hierarchical.last_train_metrics[f"entry_timing_{group}_rows"] == 0.0
    for group in ("long", "short"):
        assert hierarchical.last_train_metrics[f"entry_direction_{group}_rows"] == 0.0
    for name, expected in control.online.state_dict().items():
        torch.testing.assert_close(
            hierarchical.online.state_dict()[name], expected, rtol=0, atol=0
        )


def test_wait_only_hierarchical_batch_has_no_direction_host_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty conditional-direction task must stay finite and on-device."""
    rows = (
        _flat_sequence(
            direction=1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
        _flat_sequence(
            direction=-1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
    )
    control = _plain_agent(
        seed=839,
        entry_loss_weight=1.0,
        entry_loss_reduction="equal_present_class_mean_v1",
    )
    hierarchical = _plain_agent(
        seed=839,
        entry_loss_weight=1.0,
        entry_loss_reduction=HIERARCHICAL_ENTRY_LOSS,
    )
    original_item = torch.Tensor.item
    phase = "control"
    boolean_scalar_extractions = {"control": 0, "hierarchical": 0}

    def tracked_item(tensor: torch.Tensor, *args, **kwargs):
        if tensor.ndim == 0 and tensor.dtype == torch.bool:
            boolean_scalar_extractions[phase] += 1
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", tracked_item)
    control.train_batch(rows)
    phase = "hierarchical"
    loss = hierarchical.train_batch(rows)

    # The shared public input-validation syncs are identical. Hierarchical
    # reduction must not add another host extraction merely to discover that
    # this batch contains no ENTER target.
    assert boolean_scalar_extractions["hierarchical"] == (
        boolean_scalar_extractions["control"]
    )
    assert np.isfinite(loss)
    assert hierarchical.last_train_metrics["entry_timing_wait_rows"] == 2.0
    assert hierarchical.last_train_metrics["entry_timing_enter_rows"] == 0.0
    assert hierarchical.last_train_metrics["entry_direction_loss"] == 0.0
    for group in ("long", "short"):
        assert hierarchical.last_train_metrics[f"entry_direction_{group}_rows"] == 0.0
        assert hierarchical.last_train_metrics[
            f"entry_direction_{group}_weighted_mass"
        ] == 0.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host",
)
def test_wait_only_hierarchical_direction_reduction_is_empty_safe_on_mps() -> None:
    rows = (
        _flat_sequence(
            direction=1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
        _flat_sequence(
            direction=-1.0,
            transition_ready=False,
            target=Action.WAIT,
        ),
    )
    policy = _plain_agent(
        seed=853,
        entry_loss_weight=1.0,
        entry_loss_reduction=HIERARCHICAL_ENTRY_LOSS,
        device="mps",
    )

    loss = policy.train_batch(rows)

    assert np.isfinite(loss)
    assert policy.last_train_metrics["entry_direction_loss"] == 0.0
    for group in ("long", "short"):
        assert policy.last_train_metrics[f"entry_direction_{group}_rows"] == 0.0
        assert policy.last_train_metrics[
            f"entry_direction_{group}_weighted_loss_contribution"
        ] == 0.0
