from __future__ import annotations

import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent
from propevolve.decision import Action
from propevolve.replay import Transition
from propevolve.recovery import (
    RecoveryBranchResult,
    RecoveryValueStore,
    RecoveryValueTarget,
    build_recovery_value_target,
    recovery_action_values,
    recovery_value_kl,
)


def _branches(
    *,
    wait: RecoveryBranchResult,
    long: RecoveryBranchResult,
    short: RecoveryBranchResult,
) -> dict[Action, RecoveryBranchResult]:
    return {
        Action.WAIT: wait,
        Action.ENTER_LONG_1: long,
        Action.ENTER_SHORT_1: short,
    }


def test_recovery_values_rank_long_recovery_above_wait_progress_and_short_blow() -> None:
    target = recovery_action_values(
        observation=np.array([0.25, -0.5], np.float32),
        branches=_branches(
            wait=RecoveryBranchResult("timeout", -2_600.0, False),
            long=RecoveryBranchResult("timeout", 0.0, True),
            short=RecoveryBranchResult("blow", -3_000.0, False),
        ),
        start_pnl=-2_700.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="a" * 64,
    )

    assert isinstance(target, RecoveryValueTarget)
    assert target.action_values == pytest.approx((100.0 / 2_700.0, 1.0, -1.0))
    assert target.best_action is Action.ENTER_LONG_1


def test_recovery_values_mirror_the_winning_short_branch() -> None:
    target = recovery_action_values(
        observation=np.array([1.0], np.float32),
        branches=_branches(
            wait=RecoveryBranchResult("timeout", -2_700.0, False),
            long=RecoveryBranchResult("blow", -3_000.0, False),
            short=RecoveryBranchResult("pass", 6_000.0, True),
        ),
        start_pnl=-2_700.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="b" * 64,
    )

    assert target.action_values == pytest.approx((0.0, -1.0, 1.0))
    assert target.best_action is Action.ENTER_SHORT_1


def test_recovery_values_rank_wait_first_when_both_entries_damage_headroom() -> None:
    target = recovery_action_values(
        observation=np.array([1.0], np.float32),
        branches=_branches(
            wait=RecoveryBranchResult("timeout", -2_650.0, False),
            long=RecoveryBranchResult("timeout", -2_900.0, False),
            short=RecoveryBranchResult("timeout", -2_800.0, False),
        ),
        start_pnl=-2_700.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="c" * 64,
    )

    assert target.action_values == pytest.approx(
        (50.0 / 2_700.0, -200.0 / 2_700.0, -100.0 / 2_700.0)
    )
    assert target.best_action is Action.WAIT


def test_recovery_values_treat_an_eventual_blow_as_failure_after_early_profit() -> None:
    target = recovery_action_values(
        observation=np.array([1.0], np.float32),
        branches=_branches(
            wait=RecoveryBranchResult("timeout", -2_700.0, False),
            long=RecoveryBranchResult(
                "blow", -3_050.0, False, maximum_pnl=-1_500.0
            ),
            short=RecoveryBranchResult("timeout", -2_600.0, False),
        ),
        start_pnl=-2_700.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="d" * 64,
    )

    assert target.action_values[1] == -1.0
    assert target.best_action is Action.ENTER_SHORT_1


def test_recovery_values_reject_nontraining_or_mismatched_lineage() -> None:
    branches = _branches(
        wait=RecoveryBranchResult("timeout", -2_700.0, False),
        long=RecoveryBranchResult("timeout", -2_700.0, False),
        short=RecoveryBranchResult("timeout", -2_700.0, False),
    )
    with pytest.raises(ValueError, match="training role"):
        recovery_action_values(
            observation=np.array([1.0], np.float32),
            branches=branches,
            start_pnl=-2_700.0,
            recovery_success_pnl=0.0,
            source_role="validation",
            source_identity_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="source identity"):
        recovery_action_values(
            observation=np.array([1.0], np.float32),
            branches=branches,
            start_pnl=-2_700.0,
            recovery_success_pnl=0.0,
            source_role="training",
            source_identity_sha256="not-a-sha",
        )


def test_recovery_target_forces_each_first_action_from_the_same_causal_state() -> None:
    outcomes = {
        Action.WAIT: ("timeout", -2_650.0, False),
        Action.ENTER_LONG_1: ("timeout", 0.0, True),
        Action.ENTER_SHORT_1: ("blow", -3_000.0, False),
    }

    class BranchEnvironment:
        def __init__(self) -> None:
            self.resets: list[dict[str, object]] = []

        def reset(self, *, options):
            self.resets.append(dict(options))
            return np.array([0.25, -0.5], np.float32), {
                "ticker": options["ticker"],
                "start": options["start"],
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "realized_pnl": -2_700.0,
                "equity_pnl": -2_700.0,
            }

        def step(self, action):
            outcome, pnl, recovered = outcomes[Action(action)]
            return np.array([0.5, 0.5], np.float32), 0.0, True, False, {
                "outcome": outcome,
                "valid_actions": (),
                "realized_pnl": pnl,
                "equity_pnl": pnl,
                "recovery_status": "recovered" if recovered else "not_recovered",
            }

    class FrozenPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            self.calls += 1
            assert epsilon == 0.0
            return Action.WAIT, None, None

    environment = BranchEnvironment()
    policy = FrozenPolicy()
    options = {
        "ticker": "NQ",
        "start": 71,
        "challenge_start_state": object(),
    }
    target = build_recovery_value_target(
        policy,
        environment,
        reset_options=options,
        recurrent_horizon=64,
        start_pnl=-2_700.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="f" * 64,
    )

    assert target.action_values == pytest.approx((50.0 / 2_700.0, 1.0, -1.0))
    assert policy.calls == 3
    assert environment.resets == [options, options, options]


def test_recovery_value_store_round_trip_resumes_the_same_sample() -> None:
    store = RecoveryValueStore(capacity=3, seed=19)
    for index in range(3):
        store.add(RecoveryValueTarget(
            observation=np.array([float(index)], np.float32),
            action_values=(0.0, float(index) / 2.0, -1.0),
            source_identity_sha256=f"{index + 1:064x}",
        ))
    state = store.state_dict()
    expected = store.sample()

    restored = RecoveryValueStore(capacity=3, seed=19)
    restored.load_state_dict(state)

    assert restored.sample().identity_sha256 == expected.identity_sha256


def test_recovery_value_store_does_not_accept_replay_transitions() -> None:
    store = RecoveryValueStore(capacity=1, seed=3)
    with pytest.raises(TypeError, match="RecoveryValueTarget"):
        store.add(Transition(
            observation=np.zeros(1, np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.zeros(1, np.float32),
            terminated=True,
            valid_actions=(Action.WAIT,),
            next_valid_actions=(),
        ))


def _agent(seed: int, **overrides) -> RecurrentC51Agent:
    return RecurrentC51Agent(
        2,
        hidden_dim=8,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=1e-4,
        weight_decay=1e-5,
        gradient_clip=10.0,
        target_sync_updates=250,
        device="cpu",
        seed=seed,
        **overrides,
    )


def _ordinary_v21_batch() -> tuple[tuple[Transition, ...], ...]:
    sequence = tuple(
        Transition(
            observation=np.array([float(index), 1.0], np.float32),
            action=Action.WAIT,
            reward=0.1 if index == 3 else 0.0,
            next_observation=np.array([float(index + 1), 1.0], np.float32),
            terminated=index == 3,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            next_valid_actions=(
                ()
                if index == 3
                else (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            ),
            recurrent_reset=index == 0,
        )
        for index in range(4)
    )
    return (sequence, sequence)


def test_recovery_kl_moves_the_best_action_up_and_alternatives_down() -> None:
    q_values = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    loss = recovery_value_kl(
        q_values,
        torch.tensor([-1.0, 1.0, 0.0]),
        temperature=1.0,
    )

    loss.backward()

    assert q_values.grad is not None
    assert q_values.grad[1] < 0.0
    assert q_values.grad[0] > 0.0
    assert q_values.grad[2] > 0.0


def test_recovery_disabled_keeps_the_complete_v21_update_exact() -> None:
    control = _agent(101)
    candidate = _agent(101)
    batch = _ordinary_v21_batch()

    control_loss = control.train_batch(batch)
    candidate_loss = candidate.train_batch(
        batch,
        recovery_target=None,
        recovery_value_loss_weight=0.25,
        recovery_value_temperature=1.0,
    )

    assert candidate_loss == control_loss
    assert candidate.last_train_metrics == control.last_train_metrics
    for expected, actual in zip(
        control.online.state_dict().values(),
        candidate.online.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recovery_target_adds_one_loss_to_the_existing_optimizer_step() -> None:
    control = _agent(211)
    candidate = _agent(211)
    batch = _ordinary_v21_batch()
    target = RecoveryValueTarget(
        observation=np.array([0.0, 1.0], np.float32),
        action_values=(-1.0, 1.0, 0.0),
        source_identity_sha256="1" * 64,
    )

    control.train_batch(batch)
    candidate.train_batch(
        batch,
        recovery_target=target,
        recovery_value_loss_weight=0.25,
        recovery_value_temperature=1.0,
    )

    assert candidate.optimizer_updates == control.optimizer_updates == 1
    assert candidate.last_train_metrics["recovery_value_rows"] == 1.0
    assert candidate.last_train_metrics["recovery_value_loss"] > 0.0
    assert candidate.last_train_metrics["total_loss"] > (
        candidate.last_train_metrics["rl_loss"]
    )


def test_recovery_training_retains_v21_entry_policy_only_at_nonnegative_pnl() -> None:
    agent = _agent(307, policy_retention_loss_weight=1.0)
    agent.retain_policy(apply_to_all_management_rows=True)
    with torch.no_grad():
        agent.online.output.bias.view(len(Action), agent.atoms)[
            int(Action.ENTER_SHORT_1)
        ].add_(3.0)
    healthy = tuple(
        Transition(
            **{
                **item.__dict__,
                "recovery_active": False,
            }
        )
        for item in _ordinary_v21_batch()[0]
    )
    recovery = tuple(
        Transition(
            **{
                **item.__dict__,
                "recovery_active": True,
            }
        )
        for item in _ordinary_v21_batch()[0]
    )

    agent.train_batch(
        (healthy, recovery),
        retain_nonnegative_entry_policy=True,
    )

    assert agent.last_train_metrics[
        "healthy_entry_policy_retention_rows"
    ] == 4.0
    assert agent.last_train_metrics[
        "healthy_entry_policy_retention_loss"
    ] > 0.0


def test_recovery_training_does_not_anchor_negative_pnl_entry_rows() -> None:
    agent = _agent(311, policy_retention_loss_weight=1.0)
    agent.retain_policy(apply_to_all_management_rows=True)
    recovery = tuple(
        Transition(
            **{
                **item.__dict__,
                "recovery_active": True,
            }
        )
        for item in _ordinary_v21_batch()[0]
    )

    agent.train_batch(
        (recovery, recovery),
        retain_nonnegative_entry_policy=True,
    )

    assert agent.last_train_metrics[
        "healthy_entry_policy_retention_rows"
    ] == 0.0
    assert agent.last_train_metrics[
        "healthy_entry_policy_retention_loss"
    ] == 0.0


def test_recovery_training_does_not_restore_v21_entry_policy_after_breakeven(
) -> None:
    agent = _agent(313, policy_retention_loss_weight=1.0)
    agent.retain_policy(apply_to_all_management_rows=True)
    with torch.no_grad():
        agent.online.output.bias.view(len(Action), agent.atoms)[
            int(Action.ENTER_SHORT_1)
        ].add_(3.0)
    recovered = tuple(
        Transition(
            **{
                **item.__dict__,
                "recovery_active": False,
                "recovery_latched": True,
            }
        )
        for item in _ordinary_v21_batch()[0]
    )

    agent.train_batch(
        (recovered, recovered),
        retain_nonnegative_entry_policy=True,
    )

    assert agent.last_train_metrics[
        "healthy_entry_policy_retention_rows"
    ] == 0.0
    assert agent.last_train_metrics[
        "healthy_entry_policy_retention_loss"
    ] == 0.0
