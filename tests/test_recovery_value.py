from __future__ import annotations

import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent
from propevolve.decision import Action
from propevolve.replay import Transition
from propevolve.recovery import (
    RecoveryHandoffPolicy,
    RecoveryBranchResult,
    RecoveryValueStore,
    RecoveryValueTarget,
    build_recovery_value_target,
    recovery_action_margin,
    recovery_action_values,
    recovery_value_kl,
    select_recovery_target_prefix,
)


class _RecordingPolicy:
    def __init__(self, action: Action, name: str) -> None:
        self.action = action
        self.name = name
        self.calls: list[tuple[float, object | None, float]] = []

    def select_action(
        self,
        observation,
        *,
        hidden,
        valid_actions,
        epsilon,
        return_action_values=False,
    ):
        marker = 0 if hidden is None else int(hidden)
        self.calls.append((float(observation[0]), hidden, float(epsilon)))
        values = np.arange(len(Action), dtype=np.float32)
        return self.action, marker + 1, values if return_action_values else None


def test_recovery_handoff_uses_recovery_below_zero_and_v21_at_breakeven() -> None:
    recovery = _RecordingPolicy(Action.WAIT, "recovery")
    v21 = _RecordingPolicy(Action.ENTER_LONG_1, "v21")
    policy = RecoveryHandoffPolicy(recovery, normal_policy=v21)
    valid = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)

    first, _, first_state = policy.select_action(
        np.array([-2_000.0], np.float32),
        valid_actions=valid,
        realized_pnl=-2_000.0,
        recovery_epsilon=0.4,
    )
    second, _, second_state = policy.select_action(
        np.array([0.0], np.float32),
        valid_actions=valid,
        realized_pnl=0.0,
        recovery_epsilon=0.4,
    )

    assert (first, first_state) == (Action.WAIT, "recovery")
    assert (second, second_state) == (Action.ENTER_LONG_1, "normal")
    assert recovery.calls == [(-2_000.0, None, 0.4)]
    # V21 is reconstructed from the causal prefix, then acts greedily at $0.
    assert v21.calls == [(-2_000.0, None, 0.0), (0.0, 1, 0.0)]


def test_recovery_handoff_reconstructs_each_policy_when_pnl_crosses_zero_again() -> None:
    recovery = _RecordingPolicy(Action.ENTER_SHORT_1, "recovery")
    v21 = _RecordingPolicy(Action.ENTER_LONG_1, "v21")
    policy = RecoveryHandoffPolicy(recovery, normal_policy=v21)
    valid = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)

    states = []
    for observation, pnl in ((-100.0, -100.0), (10.0, 10.0), (-50.0, -50.0)):
        _, _, state = policy.select_action(
            np.array([observation], np.float32),
            valid_actions=valid,
            realized_pnl=pnl,
            recovery_epsilon=0.2,
        )
        states.append(state)

    assert states == ["recovery", "normal", "recovery"]
    # Re-entering recovery rebuilds its state from both preceding observations.
    assert recovery.calls == [
        (-100.0, None, 0.2),
        (-100.0, None, 0.0),
        (10.0, 1, 0.0),
        (-50.0, 2, 0.2),
    ]
    assert v21.calls == [(-100.0, None, 0.0), (10.0, 1, 0.0)]


def test_recovery_handoff_reset_discards_the_prior_episode_prefix() -> None:
    recovery = _RecordingPolicy(Action.WAIT, "recovery")
    v21 = _RecordingPolicy(Action.ENTER_LONG_1, "v21")
    policy = RecoveryHandoffPolicy(recovery, normal_policy=v21)
    valid = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)

    policy.select_action(
        np.array([-1.0], np.float32),
        valid_actions=valid,
        realized_pnl=-1.0,
        recovery_epsilon=0.1,
    )
    policy.reset()
    policy.select_action(
        np.array([0.0], np.float32),
        valid_actions=valid,
        realized_pnl=0.0,
        recovery_epsilon=0.1,
    )

    assert v21.calls == [(0.0, None, 0.0)]


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


def test_recovery_target_replays_causal_prefix_and_preserves_recurrent_boundary(
) -> None:
    outcomes = {
        Action.WAIT: ("timeout", -2_400.0, False),
        Action.ENTER_LONG_1: ("timeout", 0.0, True),
        Action.ENTER_SHORT_1: ("blow", -3_000.0, False),
    }

    class PrefixEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, options):
            self.steps = 0
            return np.array([1.0, 0.0], np.float32), {
                "ticker": options["ticker"],
                "start": options["start"],
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "realized_pnl": -2_000.0,
                "equity_pnl": -2_000.0,
            }

        def step(self, action):
            self.steps += 1
            if self.steps == 1:
                assert Action(action) is Action.WAIT
                return np.array([2.0, 0.0], np.float32), 0.0, False, False, {
                    "outcome": None,
                    "valid_actions": (
                        Action.WAIT,
                        Action.ENTER_LONG_1,
                        Action.ENTER_SHORT_1,
                    ),
                    "realized_pnl": -2_000.0,
                    "equity_pnl": -2_000.0,
                }
            outcome, pnl, recovered = outcomes[Action(action)]
            return np.array([3.0, 0.0], np.float32), 0.0, True, False, {
                "outcome": outcome,
                "valid_actions": (),
                "realized_pnl": pnl,
                "equity_pnl": pnl,
                "recovery_status": (
                    "recovered" if recovered else "not_recovered"
                ),
            }

    prefix = (
        Transition(
            observation=np.array([1.0, 0.0], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([2.0, 0.0], np.float32),
            terminated=False,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            next_valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            recurrent_reset=True,
            recovery_active=True,
        ),
        Transition(
            observation=np.array([2.0, 0.0], np.float32),
            action=Action.ENTER_SHORT_1,
            reward=0.0,
            next_observation=np.array([3.0, 0.0], np.float32),
            terminated=True,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            next_valid_actions=(),
            recovery_active=True,
            paired_a_plus_side=Action.ENTER_SHORT_1,
            paired_a_plus_economic_win=False,
        ),
    )
    policy = _RecordingPolicy(Action.WAIT, "frozen")

    target = build_recovery_value_target(
        policy,
        PrefixEnvironment(),
        reset_options={
            "ticker": "NQ",
            "start": 17,
            "challenge_start_state": object(),
        },
        recurrent_horizon=64,
        start_pnl=-2_000.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="e" * 64,
        causal_prefix=prefix,
    )

    np.testing.assert_array_equal(
        target.recurrent_observations,
        np.array([[1.0, 0.0], [2.0, 0.0]], np.float32),
    )
    assert target.recurrent_resets == (True, False)
    assert target.action_values == pytest.approx((-0.2, 1.0, -1.0))
    assert target.anchor_action is Action.ENTER_SHORT_1
    assert target.anchor_economic_success is False
    assert policy.calls == [
        (1.0, None, 0.0),
        (2.0, 1, 0.0),
    ] * 3


def test_recovery_target_keeps_fallback_wait_anchor_metadata_atomic() -> None:
    """A WAIT fallback may carry an Entry label without being an Entry anchor."""
    outcomes = {
        Action.WAIT: ("timeout", -1_900.0, False),
        Action.ENTER_LONG_1: ("timeout", 0.0, True),
        Action.ENTER_SHORT_1: ("blow", -3_000.0, False),
    }

    class PrefixEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, options):
            self.steps = 0
            return np.array([1.0, 0.0], np.float32), {
                "ticker": options["ticker"],
                "start": options["start"],
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "realized_pnl": -2_000.0,
                "equity_pnl": -2_000.0,
            }

        def step(self, action):
            self.steps += 1
            if self.steps == 1:
                return np.array([2.0, 0.0], np.float32), 0.0, False, False, {
                    "outcome": None,
                    "valid_actions": (
                        Action.WAIT,
                        Action.ENTER_LONG_1,
                        Action.ENTER_SHORT_1,
                    ),
                    "realized_pnl": -2_000.0,
                    "equity_pnl": -2_000.0,
                }
            outcome, pnl, recovered = outcomes[Action(action)]
            return np.array([3.0, 0.0], np.float32), 0.0, True, False, {
                "outcome": outcome,
                "valid_actions": (),
                "realized_pnl": pnl,
                "equity_pnl": pnl,
                "recovery_status": (
                    "recovered" if recovered else "not_recovered"
                ),
            }

    valid_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    prefix = (
        Transition(
            observation=np.array([1.0, 0.0], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([2.0, 0.0], np.float32),
            terminated=False,
            valid_actions=valid_actions,
            next_valid_actions=valid_actions,
            recurrent_reset=True,
            recovery_active=True,
        ),
        Transition(
            observation=np.array([2.0, 0.0], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([3.0, 0.0], np.float32),
            terminated=False,
            valid_actions=valid_actions,
            next_valid_actions=valid_actions,
            recovery_active=True,
            paired_a_plus_side=Action.ENTER_LONG_1,
            paired_a_plus_economic_win=True,
        ),
    )

    target = build_recovery_value_target(
        _RecordingPolicy(Action.WAIT, "frozen"),
        PrefixEnvironment(),
        reset_options={
            "ticker": "NQ",
            "start": 17,
            "challenge_start_state": object(),
        },
        recurrent_horizon=64,
        start_pnl=-2_000.0,
        recovery_success_pnl=0.0,
        source_role="training",
        source_identity_sha256="9" * 64,
        causal_prefix=prefix,
    )

    assert target.anchor_action is None
    assert target.anchor_economic_success is None


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


def test_recovery_value_store_loads_legacy_v22_checkpoint_state() -> None:
    store = RecoveryValueStore(capacity=2, seed=23)
    store.add(RecoveryValueTarget(
        observation=np.array([1.0], np.float32),
        action_values=(0.0, 1.0, -1.0),
        source_identity_sha256="9" * 64,
    ))
    legacy_state = store.state_dict()
    legacy_state.pop("balanced_sample_count")
    for payload in legacy_state["targets"]:
        payload.pop("recurrent_observations")
        payload.pop("recurrent_resets")
        payload.pop("anchor_action")
        payload.pop("anchor_economic_success")

    restored = RecoveryValueStore(capacity=2, seed=23)
    restored.load_state_dict(legacy_state)

    assert restored.sample().identity_sha256 == store.sample().identity_sha256


def test_recovery_target_prefix_selects_the_requested_economic_boundary() -> None:
    valid = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    transitions = tuple(
        Transition(
            observation=np.array([float(index), 0.0], np.float32),
            action=action,
            reward=0.0,
            next_observation=np.array([float(index + 1), 0.0], np.float32),
            terminated=False,
            valid_actions=valid,
            next_valid_actions=valid,
            recurrent_reset=index == 0,
            recovery_active=True,
            regime_selectivity_headroom_fraction=headroom,
            paired_a_plus_side=action if action is not Action.WAIT else None,
            paired_a_plus_economic_win=economic_win,
        )
        for index, (action, economic_win, headroom) in enumerate((
            (Action.WAIT, None, 0.8),
            (Action.ENTER_LONG_1, True, 0.7),
            (Action.ENTER_SHORT_1, False, 0.2),
        ))
    )

    winning = select_recovery_target_prefix(transitions, prefer_success=True)
    failed = select_recovery_target_prefix(transitions, prefer_success=False)

    assert winning is not None and winning[-1].action is Action.ENTER_LONG_1
    assert failed is not None and failed[-1].action is Action.ENTER_SHORT_1
    assert len(winning) == 2
    assert len(failed) == 3


def test_recovery_value_store_balances_side_and_economic_boundary() -> None:
    store = RecoveryValueStore(capacity=8, seed=7)
    for index, (side, success) in enumerate((
        (Action.ENTER_LONG_1, False),
        (Action.ENTER_LONG_1, True),
        (Action.ENTER_SHORT_1, False),
        (Action.ENTER_SHORT_1, True),
    )):
        store.add(RecoveryValueTarget(
            observation=np.array([float(index)], np.float32),
            action_values=(0.0, 1.0, -1.0),
            source_identity_sha256=f"{index + 10:064x}",
            anchor_action=side,
            anchor_economic_success=success,
        ))

    sampled = {
        (target.anchor_action, target.anchor_economic_success)
        for target in (store.sample_balanced() for _ in range(4))
    }

    assert sampled == {
        (Action.ENTER_LONG_1, False),
        (Action.ENTER_LONG_1, True),
        (Action.ENTER_SHORT_1, False),
        (Action.ENTER_SHORT_1, True),
    }


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


def test_recovery_handoff_reconstructs_exact_v21_q_values_at_breakeven() -> None:
    direct_v21 = _agent(97)
    routed_v21 = _agent(97)
    recovery = _agent(101)
    handoff = RecoveryHandoffPolicy(recovery, normal_policy=routed_v21)
    valid = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    observations = (
        np.array([-0.3, 0.1], np.float32),
        np.array([-0.1, 0.4], np.float32),
        np.array([0.0, 0.8], np.float32),
    )
    direct_hidden = None
    direct_action = None
    direct_values = None
    for observation in observations:
        direct_action, direct_hidden, direct_values = direct_v21.select_action(
            observation,
            hidden=direct_hidden,
            valid_actions=valid,
            epsilon=0.0,
            return_action_values=True,
        )

    for observation, pnl in zip(observations, (-2_000.0, -500.0, 0.0), strict=True):
        routed_action, routed_values, state = handoff.select_action(
            observation,
            valid_actions=valid,
            realized_pnl=pnl,
            recovery_epsilon=0.0,
            return_action_values=True,
        )

    assert state == "normal"
    assert routed_action is direct_action
    np.testing.assert_allclose(routed_values, direct_values, rtol=0, atol=0)


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


def test_recovery_action_margin_requires_the_economic_winner_to_rank_first(
) -> None:
    q_values = torch.tensor([0.0, 0.1, 0.0], requires_grad=True)

    loss = recovery_action_margin(
        q_values,
        torch.tensor([-1.0, 1.0, 0.0]),
        margin=0.25,
    )

    assert loss.item() == pytest.approx(0.15)
    loss.backward()
    assert q_values.grad is not None
    assert q_values.grad[1] < 0.0
    assert q_values.grad[0] > 0.0


def test_recovery_action_margin_does_not_invent_a_winner_for_economic_ties(
) -> None:
    loss = recovery_action_margin(
        torch.zeros(3),
        torch.tensor([1.0, 1.0, -1.0]),
        margin=0.25,
    )

    assert loss.item() == 0.0


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
        recurrent_observations=np.array(
            [[1.0, 0.0], [0.0, 1.0]], np.float32
        ),
        recurrent_resets=(True, False),
    )

    control.train_batch(batch)
    candidate.train_batch(
        batch,
        recovery_target=target,
        recovery_value_loss_weight=0.25,
        recovery_value_temperature=1.0,
        recovery_action_margin=0.25,
    )

    assert candidate.optimizer_updates == control.optimizer_updates == 1
    assert candidate.last_train_metrics["recovery_value_rows"] == 1.0
    assert candidate.last_train_metrics["recovery_value_loss"] > 0.0
    assert candidate.last_train_metrics["recovery_action_margin_loss"] > 0.0
    assert candidate.last_train_metrics["recovery_recurrent_rows"] == 2.0
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
