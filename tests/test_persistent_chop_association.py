from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.agent import (
    RecurrentC51Agent,
    persistent_chop_association_rank_loss,
)
from propevolve.balance_aware_regime_selectivity import (
    BalanceAwareRegimeSelectivity,
    PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
)
from propevolve.decision import Action
from propevolve.replay import Transition
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS


CHANNELS = (*EXPANSION_CHANNELS, *REGIME_CHANNELS)
CENTERS = (0.10249102659218842, 0.10399580328775007)


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
    values[CHANNELS.index("chop_no_trend_probability")] = chop_persistence
    values[CHANNELS.index("chop_end_transition_probability")] = readiness
    values[CHANNELS.index("expansion_trend_probability")] = readiness
    return values


def _transition(
    observation: tuple[float, float, float],
    teacher: np.ndarray,
    target: Action,
) -> tuple[Transition, ...]:
    flat = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    return (
        Transition(
            observation=np.asarray(observation, dtype=np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.zeros(3, dtype=np.float32),
            terminated=True,
            valid_actions=flat,
            next_valid_actions=(),
            teacher_target=teacher,
            entry_action_target=target,
            regime_selectivity_headroom_fraction=1.0,
        ),
    )


def _agent(
    *,
    semantics: str,
    selectivity_weight: float = 0.3,
    device: str = "cpu",
) -> RecurrentC51Agent:
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
        device=device,
        seed=314159,
        teacher_channels=len(CHANNELS),
        teacher_channel_names=CHANNELS,
        teacher_loss_weight=1e-6,
        entry_action_loss_weight=0.3,
        entry_action_loss_reduction="equal_present_class_mean_v1",
        regime_selectivity_loss_weight=selectivity_weight,
        regime_selectivity_expansion_centers=CENTERS,
        regime_selectivity_side_balance="equal_long_short_v1",
        regime_selectivity_semantics=semantics,
        regime_selectivity_persistent_chop_negative_emphasis=1.0,
    )


def _flat_probabilities(
    policy: RecurrentC51Agent,
    rows: tuple[tuple[Transition, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    actions, action_values = policy.greedy_sequence_action_values(rows)
    flat_values = torch.as_tensor(action_values[:, 0, :3])
    return actions[:, 0], torch.softmax(flat_values, dim=-1).numpy()


def test_negative_only_objective_reproduces_v5_inverted_wait_association() -> None:
    dead = _teacher_row(chop_persistence=0.99, readiness=0.0)
    ready = _teacher_row(chop_persistence=0.99, readiness=0.99)
    rows = (
        *(
            _transition((-1.0, -1.0, -1.0), dead, Action.WAIT)
            for _ in range(31)
        ),
        _transition((0.0, 1.0, 0.0), ready, Action.WAIT),
    )
    policy = _agent(semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS)

    for _ in range(4):
        policy.train_batch(rows)

    metrics = policy.last_train_metrics
    assert metrics[
        "regime_selectivity_persistent_dead_chop_model_wait_probability_mean"
    ] == pytest.approx(0.5829402839)
    assert metrics[
        "regime_selectivity_transition_ready_model_wait_probability_mean"
    ] == pytest.approx(0.5959601998)
    assert metrics[
        "regime_selectivity_persistent_dead_chop_model_wait_probability_mean"
    ] < metrics[
        "regime_selectivity_transition_ready_model_wait_probability_mean"
    ]


def test_compiler_authenticates_exact_cohorts_and_centers_expansion_evidence() -> None:
    compiler = BalanceAwareRegimeSelectivity(
        channel_names=CHANNELS,
        expansion_centers=CENTERS,
        semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
        persistent_chop_negative_emphasis=1.0,
    )
    below_center = _teacher_row(
        chop_persistence=1.0,
        readiness=1.0,
        long_score=CENTERS[0] / 9.0,
    )
    above_center = _teacher_row(
        chop_persistence=1.0,
        readiness=1.0,
        long_score=0.9,
    )
    teachers = torch.as_tensor(
        np.stack((below_center, above_center, above_center, above_center))
    )
    actions = torch.tensor(
        (Action.ENTER_LONG_1, Action.ENTER_LONG_1, Action.WAIT, Action.ENTER_SHORT_1)
    )

    evidence = compiler.exact_wait_negative_weight_evidence(teachers, actions)

    assert evidence.transition_positive_long_membership[0] < 0.5
    assert evidence.transition_positive_long_membership[1] > 0.5
    assert evidence.transition_positive_long_membership[2] == 0.0
    assert evidence.transition_positive_long_membership[3] == 0.0
    assert evidence.transition_positive_short_membership[3] > 0.0


def test_association_semantics_learns_dead_wait_above_exact_transition_entries(
    tmp_path: Path,
) -> None:
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
        *(
            _transition((-1.0, -1.0, -1.0), dead, Action.WAIT)
            for _ in range(31)
        ),
        _transition((1.0, 0.0, 0.0), ready_long, Action.ENTER_LONG_1),
        _transition((0.0, 1.0, 0.0), ready_short, Action.ENTER_SHORT_1),
    )
    probes = (rows[0], rows[-2], rows[-1])
    policy = _agent(semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS)

    for _ in range(12):
        policy.train_batch(rows)

    # Replay the frozen v5/v6 autonomy tail: Regime guidance reaches zero at
    # 80% while exact Entry supervision reaches zero at 95%.
    for progress in np.linspace(0.8, 1.0, 21):
        policy.train_batch(
            rows,
            teacher_weight_scale=max(0.0, 1.0 - progress / 0.8),
            entry_action_weight_scale=max(0.0, 1.0 - progress / 0.95),
        )

    metrics = policy.last_train_metrics
    assert metrics["teacher_weight_scale"] == 0.0
    assert metrics["entry_action_weight_scale"] == 0.0
    assert metrics["regime_entry_conflict_long_soft_wait_disagreement_rows"] == 0.0
    assert metrics["regime_entry_conflict_short_soft_wait_disagreement_rows"] == 0.0
    actions, probabilities = _flat_probabilities(policy, probes)
    assert actions.tolist() == [
        int(Action.WAIT),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
    ]
    assert probabilities[0, int(Action.WAIT)] > probabilities[1, int(Action.WAIT)]
    assert probabilities[0, int(Action.WAIT)] > probabilities[2, int(Action.WAIT)]

    path = policy.save(tmp_path / "association.pt", manifest={})
    resumed, _ = RecurrentC51Agent.load(path, device="cpu")
    assert resumed.regime_selectivity_semantics == PERSISTENT_CHOP_ASSOCIATION_SEMANTICS
    resumed_actions, resumed_probabilities = _flat_probabilities(resumed, probes)
    np.testing.assert_array_equal(resumed_actions, actions)
    np.testing.assert_allclose(resumed_probabilities, probabilities, rtol=0, atol=0)
    resumed.discard_teacher()
    assert resumed.regime_selectivity_semantics != PERSISTENT_CHOP_ASSOCIATION_SEMANTICS
    discarded_actions, discarded_probabilities = _flat_probabilities(resumed, probes)
    np.testing.assert_array_equal(discarded_actions, actions)
    np.testing.assert_allclose(discarded_probabilities, probabilities, rtol=0, atol=0)
    deployed_path = resumed.save(tmp_path / "association-teacher-free.pt", manifest={})
    deployed, _ = RecurrentC51Agent.load(deployed_path, device="cpu")
    deployed_actions, deployed_probabilities = _flat_probabilities(deployed, probes)
    np.testing.assert_array_equal(deployed_actions, actions)
    np.testing.assert_allclose(deployed_probabilities, probabilities, rtol=0, atol=0)


def test_association_semantics_skips_finitely_without_each_authenticated_cohort() -> None:
    ready_long = _teacher_row(
        chop_persistence=0.99,
        readiness=0.99,
        long_score=0.90,
    )
    policy = _agent(semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS)

    loss = policy.train_batch((
        _transition((1.0, 0.0, 0.0), ready_long, Action.ENTER_LONG_1),
    ))

    assert np.isfinite(loss)
    assert policy.last_train_metrics["regime_selectivity_association_loss"] == 0.0
    assert policy.last_train_metrics["regime_selectivity_association_active"] == 0.0
    assert policy.last_train_metrics["regime_selectivity_association_skipped"] == 1.0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host",
)
def test_association_optimizer_update_runs_on_mps() -> None:
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
        _transition((-1.0, -1.0, -1.0), dead, Action.WAIT),
        _transition((1.0, 0.0, 0.0), ready_long, Action.ENTER_LONG_1),
        _transition((0.0, 1.0, 0.0), ready_short, Action.ENTER_SHORT_1),
    )
    policy = _agent(
        semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
        device="mps",
    )

    loss = policy.train_batch(rows)
    metrics = policy.last_train_metrics

    assert np.isfinite(loss)
    assert metrics["regime_selectivity_association_active"] == 1.0
    assert metrics["regime_selectivity_association_skipped"] == 0.0
    assert metrics["regime_entry_conflict_long_soft_wait_disagreement_rows"] == 0.0
    assert metrics["regime_entry_conflict_short_soft_wait_disagreement_rows"] == 0.0


def test_association_semantics_equal_normalizes_long_and_short_exposure() -> None:
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
    base_rows = (
        _transition((-1.0, -1.0, -1.0), dead, Action.WAIT),
        _transition((1.0, 0.0, 0.0), ready_long, Action.ENTER_LONG_1),
        _transition((0.0, 1.0, 0.0), ready_short, Action.ENTER_SHORT_1),
    )
    duplicated_rows = (
        base_rows[0],
        *(base_rows[1] for _ in range(31)),
        base_rows[2],
    )
    balanced = _agent(semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS)
    duplicated = _agent(semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS)

    balanced.train_batch(base_rows)
    duplicated.train_batch(duplicated_rows)

    assert duplicated.last_train_metrics[
        "regime_selectivity_association_loss"
    ] == pytest.approx(
        balanced.last_train_metrics["regime_selectivity_association_loss"],
        rel=1e-5,
    )


def test_association_gradient_changes_only_authenticated_wait_coordinates() -> None:
    flat_q = torch.tensor(
        (
            (0.0, 0.1, -0.2),
            (0.2, 0.3, -0.1),
            (-0.1, 0.2, 0.4),
            (0.5, 0.6, 0.7),  # transition-ready exact WAIT: excluded
            (-0.5, -0.4, -0.3),  # persistent-dead exact Long: excluded
        ),
        requires_grad=True,
    )
    loss, active = persistent_chop_association_rank_loss(
        flat_q,
        dead_membership=torch.tensor((1.0, 0.0, 0.0, 0.0, 0.0)),
        transition_positive_long_membership=torch.tensor(
            (0.0, 0.25, 0.0, 0.0, 0.0)
        ),
        transition_positive_short_membership=torch.tensor(
            (0.0, 0.0, 0.75, 0.0, 0.0)
        ),
        q_temperature=1.0,
    )

    gradient = torch.autograd.grad(loss, flat_q)[0]

    assert active == 1.0
    torch.testing.assert_close(
        gradient[:, 1:], torch.zeros_like(gradient[:, 1:]), rtol=0, atol=0
    )
    assert gradient[0, 0] < 0.0
    assert gradient[1, 0] > 0.0
    assert gradient[2, 0] > 0.0
    torch.testing.assert_close(
        gradient[3:], torch.zeros_like(gradient[3:]), rtol=0, atol=0
    )


def test_weight_zero_has_exact_legacy_parameter_and_checkpoint_shape_parity(
    tmp_path: Path,
) -> None:
    legacy = _agent(
        semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        selectivity_weight=0.0,
    )
    disabled = _agent(
        semantics=PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
        selectivity_weight=0.0,
    )
    assert tuple(legacy.online.state_dict()) == tuple(disabled.online.state_dict())
    for name, tensor in legacy.online.state_dict().items():
        torch.testing.assert_close(tensor, disabled.online.state_dict()[name], rtol=0, atol=0)

    legacy_path = legacy.save(tmp_path / "legacy.pt", manifest={})
    disabled_path = disabled.save(tmp_path / "disabled.pt", manifest={})
    legacy_payload = torch.load(legacy_path, weights_only=False)
    disabled_payload = torch.load(disabled_path, weights_only=False)
    assert {
        name: tuple(value.shape) for name, value in legacy_payload["online"].items()
    } == {
        name: tuple(value.shape) for name, value in disabled_payload["online"].items()
    }
