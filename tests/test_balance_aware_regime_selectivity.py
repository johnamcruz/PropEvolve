from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.balance_aware_regime_selectivity import (
    BalanceAwareRegimeSelectivity,
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    STATIC_STATE_SEMANTICS,
)
from propevolve.agent import RecurrentC51Agent
from propevolve.decision import Action
from propevolve.replay import Transition
from propevolve.teachers.expansion import CHANNELS as EXPANSION_CHANNELS
from propevolve.teachers.regime import CHANNELS as REGIME_CHANNELS
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


def _agent(
    *,
    seed: int,
    selectivity_weight: float,
    n_step_return: int = 1,
    entry_action_weight: float = 0.0,
    side_balance: str | None = None,
    selectivity_semantics: str | None = None,
    persistent_chop_negative_emphasis: float | None = None,
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
        device=device,
        seed=seed,
        teacher_channels=len(CHANNELS),
        teacher_channel_names=CHANNELS,
        teacher_loss_weight=1e-6,
        teacher_entry_search_centers=(0.10, 0.10),
        entry_action_loss_weight=entry_action_weight,
        regime_selectivity_loss_weight=selectivity_weight,
        regime_selectivity_expansion_centers=(0.10, 0.10),
        **optional_settings,
    )


def _sequence(
    observation: tuple[float, float, float],
    teacher: torch.Tensor,
    *,
    headroom: float,
    target: Action,
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
            entry_action_target=target,
            regime_selectivity_headroom_fraction=headroom,
        ),
    )


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
    updates = {
        "structure_chop_persistence_probability": chop_persistence,
        "structure_trend_onset_probability": transition_readiness,
        "structure_trend_persistence_probability": transition_readiness,
        "volatility_expansion_onset_probability": transition_readiness,
        "volatility_high_persistence_probability": transition_readiness,
        "kaufman_efficiency": transition_readiness,
        "volatility_percentile": transition_readiness,
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


def test_zero_curriculum_scale_has_exact_rl_update_parity() -> None:
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

    control.train_batch((sequence, sequence), teacher_weight_scale=0.0)
    taught.train_batch((sequence, sequence), teacher_weight_scale=0.0)

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

    # transition_readiness = Kaufman 0.8 * mean(0.8, ..., 0.8) = 0.64.
    # Effective memberships are continuous formula components, not thresholded
    # row masses: dead = 0.9 + 0.9*(1-0.64) on exact WAIT rows, while
    # transition-ready mass = 0.9*0.64 on both a WAIT and positive Entry row.
    assert metrics["regime_selectivity_exact_wait_rows"] == 2.0
    assert metrics["regime_selectivity_exact_wait_weight_sum"] == pytest.approx(
        4.6 + 2.296,
    )
    assert metrics["regime_selectivity_exact_wait_weight_mean"] == pytest.approx(
        (4.6 + 2.296) / 2.0,
    )
    assert metrics["regime_selectivity_persistent_chop_weight_sum"] == pytest.approx(
        metrics["regime_selectivity_exact_wait_weight_sum"]
    )
    assert metrics["regime_selectivity_persistent_dead_chop_rows"] == pytest.approx(
        0.9 + 0.324,
    )
    assert metrics[
        "regime_selectivity_persistent_dead_chop_weight_sum"
    ] == pytest.approx(0.9 * 4.6 + 0.324 * 2.296)
    assert metrics["regime_selectivity_transition_ready_rows"] == pytest.approx(
        0.576,
    )
    assert metrics[
        "regime_selectivity_transition_ready_weight_sum"
    ] == pytest.approx(0.576 * 2.296)
    assert metrics["regime_selectivity_transition_positive_long_rows"] == (
        pytest.approx(0.576)
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

    assert first["positive_long_short"] == pytest.approx({
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
    assert second["safe_headroom_ge_0_75"] == {
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

    assert summary["positive_long_short"] == pytest.approx(
        first["positive_long_short"]
    )
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
    config = {
        "_root": str(tmp_path),
        "tickers": ["NQ"],
        "regime_selectivity": specification,
    }
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
    config = {"_root": str(tmp_path), "tickers": ["NQ"]}
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
