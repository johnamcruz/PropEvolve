from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.agent import (
    RecurrentC51Agent,
    RecurrentC51Network,
    centered_entry_search_target,
    resolve_device,
)
from propevolve.decision import Action
from propevolve.replay import Transition


def _agent(observation_dim: int, **overrides) -> RecurrentC51Agent:
    settings = {
        "hidden_dim": 8,
        "atoms": 11,
        "value_min": -3.0,
        "value_max": 3.0,
        "gamma": 0.997,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "gradient_clip": 10.0,
        "target_sync_updates": 250,
        "device": "cpu",
        "seed": 0,
    }
    settings.update(overrides)
    return RecurrentC51Agent(observation_dim, **settings)


def test_network_emits_distribution_for_every_time_action_and_atom() -> None:
    network = RecurrentC51Network(observation_dim=12, action_count=5, atoms=21, hidden_dim=16)
    logits, hidden = network(torch.zeros(3, 5, 12))

    assert logits.shape == (3, 5, 5, 21)
    assert hidden.shape == (1, 3, 16)
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(3, 5, 5))


def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_compilation_failure_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", unavailable)
    agent = _agent(
        4,
        compile_model=True,
        compile_backend="inductor",
        compile_mode="default",
    )

    assert agent.compile_status == "fallback_eager"
    assert "compiler unavailable" in agent.compile_error


def test_lazy_compilation_failure_retries_the_update_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def compile_lazily(*args, **kwargs):
        def fail_when_called(*call_args, **call_kwargs):
            raise RuntimeError("lowering failed")

        return fail_when_called

    monkeypatch.setattr(torch, "compile", compile_lazily)
    agent = _agent(4, compile_model=True)

    selected, _, _ = agent.select_action(
        np.zeros(4, np.float32),
        hidden=None,
        valid_actions=(Action.WAIT,),
        epsilon=0.0,
    )

    assert selected == Action.WAIT
    assert agent.compile_status == "fallback_eager"
    assert "lowering failed" in agent.compile_error


def test_fp16_runtime_rejects_cpu_instead_of_silently_changing_precision() -> None:
    with pytest.raises(ValueError, match="mixed precision"):
        _agent(4, mixed_precision="fp16")


def test_mps_runtime_flags_are_applied_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.delenv("PYTORCH_MPS_PREFER_METAL", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)

    agent = _agent(
        4,
        device="mps",
        mps_prefer_metal=True,
        mps_fast_math=False,
    )

    assert agent.device.type == "mps"
    assert agent.runtime_environment == {
        "PYTORCH_MPS_PREFER_METAL": "1",
        "PYTORCH_MPS_FAST_MATH": "0",
    }


def test_agent_never_selects_an_action_rejected_by_external_mask() -> None:
    agent = _agent(4, seed=3)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        # Give ENTER_LONG_1 the largest unmasked value; it must still lose to WAIT.
        agent.online.output.bias.view(len(Action), agent.atoms)[Action.ENTER_LONG_1, -1] = 100

    selected, _, _ = agent.select_action(
        np.zeros(4, np.float32),
        hidden=None,
        valid_actions=(Action.WAIT,),
        epsilon=0.0,
    )

    assert selected == Action.WAIT


def test_agent_scores_long_and_short_hypotheses_in_the_same_decision() -> None:
    agent = _agent(4, seed=13)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        distributions = agent.online.output.bias.view(len(Action), agent.atoms)
        distributions[Action.ENTER_LONG_1, -1] = 20.0
        distributions[Action.ENTER_SHORT_1, 0] = 20.0

    selected, _, action_values = agent.select_action(
        np.zeros(4, np.float32),
        hidden=None,
        valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
        epsilon=0.0,
        return_action_values=True,
    )

    assert np.isfinite(action_values[[
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    ]]).all()
    assert action_values[Action.ENTER_LONG_1] > action_values[Action.WAIT]
    assert action_values[Action.WAIT] > action_values[Action.ENTER_SHORT_1]
    assert selected == Action.ENTER_LONG_1


def test_distributional_double_dqn_update_learns_from_recurrent_sequences() -> None:
    agent = _agent(2, seed=5)
    sequence = tuple(
        Transition(
            observation=np.array([i, 0], np.float32),
            action=Action.WAIT,
            reward=0.1 if i < 3 else 1.0,
            next_observation=np.array([i + 1, 0], np.float32),
            terminated=i == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
            next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
        )
        for i in range(4)
    )
    before = agent.online.output.weight.detach().clone()

    loss = agent.train_batch((sequence, sequence))

    assert np.isfinite(loss)
    assert loss > 0
    assert not torch.equal(before, agent.online.output.weight.detach())


def test_recurrent_double_dqn_targets_preserve_the_current_state_history() -> None:
    """The next-state target must be the one-step shift of one causal GRU trace."""
    agent = _agent(2, seed=41, learning_rate=1e-12)
    sequence = tuple(
        Transition(
            observation=np.array([float(index), float(index % 2)], np.float32),
            action=Action.HOLD,
            reward=0.05 * index,
            next_observation=np.array(
                [float(index + 1), float((index + 1) % 2)], np.float32
            ),
            terminated=index == 3,
            valid_actions=(Action.HOLD, Action.CLOSE),
            next_valid_actions=(Action.HOLD, Action.CLOSE),
        )
        for index in range(4)
    )
    observations = torch.as_tensor(
        np.stack([sequence[0].observation, *(item.next_observation for item in sequence)])
    ).view(1, 5, 2)
    actions = torch.full((1, 4), int(Action.HOLD), dtype=torch.long)
    rewards = torch.as_tensor([[item.reward for item in sequence]])
    terminated = torch.as_tensor([[item.terminated for item in sequence]])

    with torch.no_grad():
        online_logits, _ = agent.online(observations)
        target_logits, _ = agent.target(observations)
        current_logits = online_logits[:, :-1]
        online_next = online_logits[:, 1:]
        target_next = target_logits[:, 1:]
        online_q = (online_next.softmax(-1) * agent.support).sum(-1)
        valid = torch.zeros((1, 4, len(Action)), dtype=torch.bool)
        valid[..., int(Action.HOLD)] = True
        valid[..., int(Action.CLOSE)] = True
        next_actions = online_q.masked_fill(~valid, -torch.inf).argmax(-1)
        target_distribution = target_next.softmax(-1).gather(
            2,
            next_actions[..., None, None].expand(-1, -1, 1, agent.atoms),
        ).squeeze(2)
        projected = agent._project_distribution(
            target_distribution,
            rewards,
            terminated,
        )
        chosen = current_logits.gather(
            2,
            actions[..., None, None].expand(-1, -1, 1, agent.atoms),
        ).squeeze(2)
        expected_loss = -(
            projected * chosen.log_softmax(-1)
        ).sum(-1).mean().item()

    agent.train_batch((sequence,))

    assert agent.last_train_metrics["rl_loss"] == pytest.approx(
        expected_loss, rel=1e-6, abs=1e-6
    )
    assert agent.last_train_metrics["sampled_management_row_fraction"] == 1.0
    assert agent.last_train_metrics["sampled_management_close_fraction"] == 0.0
    assert np.isfinite(agent.last_train_metrics["management_hold_minus_close_q"])


def test_recurrent_double_dqn_propagates_configured_multi_step_returns() -> None:
    agent = _agent(2, seed=43, learning_rate=1e-12, n_step_return=3)
    sequence = tuple(
        Transition(
            observation=np.array([float(index), float(index % 2)], np.float32),
            action=Action.HOLD,
            reward=reward,
            next_observation=np.array(
                [float(index + 1), float((index + 1) % 2)], np.float32
            ),
            terminated=index == 3,
            valid_actions=(Action.HOLD, Action.CLOSE),
            next_valid_actions=(Action.HOLD, Action.CLOSE),
        )
        for index, reward in enumerate((0.1, 0.2, 0.4, 0.8))
    )
    observations = torch.as_tensor(
        np.stack([sequence[0].observation, *(item.next_observation for item in sequence)])
    ).view(1, 5, 2)
    rewards = torch.as_tensor([[item.reward for item in sequence]])
    gamma = agent.gamma
    returns = torch.stack((
        rewards[:, 0] + gamma * rewards[:, 1] + gamma**2 * rewards[:, 2],
        rewards[:, 1] + gamma * rewards[:, 2] + gamma**2 * rewards[:, 3],
    ), dim=1)
    terminated = torch.tensor([[False, True]])

    with torch.no_grad():
        online_logits, _ = agent.online(observations)
        target_logits, _ = agent.target(observations)
        current_logits = online_logits[:, :2]
        online_next = online_logits[:, 3:]
        target_next = target_logits[:, 3:]
        online_q = (online_next.softmax(-1) * agent.support).sum(-1)
        valid = torch.zeros((1, 2, len(Action)), dtype=torch.bool)
        valid[..., int(Action.HOLD)] = True
        valid[..., int(Action.CLOSE)] = True
        next_actions = online_q.masked_fill(~valid, -torch.inf).argmax(-1)
        next_distribution = target_next.softmax(-1).gather(
            2,
            next_actions[..., None, None].expand(-1, -1, 1, agent.atoms),
        ).squeeze(2)
        delta = (agent.value_max - agent.value_min) / (agent.atoms - 1)
        target_support = returns[..., None] + (
            gamma**3 * (~terminated).float()[..., None] * agent.support
        )
        target_support.clamp_(agent.value_min, agent.value_max)
        positions = (target_support - agent.value_min) / delta
        lower = positions.floor().long()
        upper = positions.ceil().long()
        projected = torch.zeros_like(next_distribution)
        projected.scatter_add_(
            -1, lower, next_distribution * (upper.float() - positions)
        )
        projected.scatter_add_(
            -1, upper, next_distribution * (positions - lower.float())
        )
        projected.scatter_add_(
            -1, lower, next_distribution * (lower == upper).float()
        )
        chosen = current_logits[:, :, int(Action.HOLD)]
        expected_loss = -(
            projected * chosen.log_softmax(-1)
        ).sum(-1).mean().item()

    agent.train_batch((sequence,))

    assert agent.last_train_metrics["rl_loss"] == pytest.approx(
        expected_loss, rel=1e-6, abs=1e-6
    )
    assert agent.last_train_metrics["n_step_return"] == 3.0


def test_eight_step_management_learning_prefers_a_delayed_winner_to_close() -> None:
    """Regression: delayed HOLD value must not collapse into immediate CLOSE."""
    agent = _agent(
        2,
        hidden_dim=16,
        atoms=51,
        value_min=-1.0,
        value_max=2.0,
        gamma=0.99,
        n_step_return=8,
        recurrent_burn_in=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_sync_updates=20,
        seed=7,
    )

    def management_sequence(
        initial_action: Action,
        delayed_reward: float,
    ) -> tuple[Transition, ...]:
        return tuple(
            Transition(
                observation=np.array([index / 16, 0.0], np.float32),
                action=initial_action if index == 8 else Action.HOLD,
                reward=delayed_reward if index == 15 else 0.0,
                next_observation=np.array([(index + 1) / 16, 0.0], np.float32),
                terminated=(
                    index == 15 if initial_action == Action.HOLD else index == 8
                ),
                valid_actions=(Action.HOLD, Action.CLOSE),
                next_valid_actions=(Action.HOLD, Action.CLOSE),
            )
            for index in range(16)
        )

    hold = management_sequence(Action.HOLD, delayed_reward=1.0)
    close = management_sequence(Action.CLOSE, delayed_reward=0.0)
    for _ in range(300):
        agent.train_batch((hold, close, hold, close))

    hidden = None
    for index in range(8):
        _, hidden, _ = agent.select_action(
            np.array([index / 16, 0.0], np.float32),
            hidden=hidden,
            valid_actions=(Action.HOLD,),
            epsilon=0.0,
        )
    selected, _, action_values = agent.select_action(
        np.array([0.5, 0.0], np.float32),
        hidden=hidden,
        valid_actions=(Action.HOLD, Action.CLOSE),
        epsilon=0.0,
        return_action_values=True,
    )

    assert selected == Action.HOLD
    assert action_values[Action.HOLD] - action_values[Action.CLOSE] > 0.5
    assert agent.last_train_metrics["sampled_hold_n_step_return"] > 0.9
    assert agent.last_train_metrics["sampled_close_n_step_return"] == 0.0


def test_recurrent_training_forgets_history_before_a_behavior_reset() -> None:
    """Learning state must match the hidden-state resets used during behavior."""
    left = _agent(
        2,
        hidden_dim=8,
        atoms=11,
        n_step_return=2,
        recurrent_burn_in=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=71,
    )
    right = _agent(
        2,
        hidden_dim=8,
        atoms=11,
        n_step_return=2,
        recurrent_burn_in=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=71,
    )

    def sequence(prefix: float) -> tuple[Transition, ...]:
        return tuple(
            Transition(
                observation=np.array([
                    prefix if index < 6 else index / 12,
                    1.0,
                ], np.float32),
                action=Action.HOLD,
                reward=0.25 if index == 9 else 0.0,
                next_observation=np.array([
                    prefix if index + 1 < 6 else (index + 1) / 12,
                    1.0,
                ], np.float32),
                terminated=False,
                valid_actions=(Action.HOLD, Action.CLOSE),
                next_valid_actions=(Action.HOLD, Action.CLOSE),
                recurrent_reset=index in {0, 6},
                next_recurrent_reset=index + 1 in {0, 6},
            )
            for index in range(12)
        )

    left_loss = left.train_batch((sequence(-100.0),))
    right_loss = right.train_batch((sequence(100.0),))

    assert left_loss == pytest.approx(right_loss, rel=1e-7, abs=1e-7)
    for left_parameter, right_parameter in zip(
        left.online.parameters(), right.online.parameters(), strict=True
    ):
        torch.testing.assert_close(left_parameter, right_parameter)


def test_pass_competence_anchor_prevents_management_policy_collapse() -> None:
    """Timeout replay must not erase management behavior that produced a pass."""
    anchored = _agent(
        2,
        hidden_dim=16,
        atoms=51,
        value_min=-1.0,
        value_max=2.0,
        gamma=0.99,
        n_step_return=8,
        recurrent_burn_in=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_update_mode="soft",
        target_soft_tau=0.005,
        policy_retention_loss_weight=10.0,
        seed=79,
    )
    control = _agent(
        2,
        hidden_dim=16,
        atoms=51,
        value_min=-1.0,
        value_max=2.0,
        gamma=0.99,
        n_step_return=8,
        recurrent_burn_in=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        target_update_mode="soft",
        target_soft_tau=0.005,
        policy_retention_loss_weight=0.0,
        seed=79,
    )

    def sequence(action: Action, reward: float, *, pass_episode: bool) -> tuple[Transition, ...]:
        return tuple(
            Transition(
                observation=np.array([index / 16, 1.0], np.float32),
                action=action if index == 8 else Action.HOLD,
                reward=reward if index == 15 else 0.0,
                next_observation=np.array([(index + 1) / 16, 1.0], np.float32),
                terminated=False,
                valid_actions=(Action.HOLD, Action.CLOSE),
                next_valid_actions=(Action.HOLD, Action.CLOSE),
                recurrent_reset=index == 0,
                competence_anchor=pass_episode,
            )
            for index in range(16)
        )

    passing_hold = sequence(Action.HOLD, 0.20, pass_episode=True)
    timeout_close = sequence(Action.CLOSE, 0.35, pass_episode=False)
    for _ in range(200):
        batch = (passing_hold,) * 4
        anchored.train_batch(batch)
        control.train_batch(batch)
    anchored.retain_policy()
    for _ in range(120):
        batch = (passing_hold, timeout_close, timeout_close, timeout_close)
        anchored.train_batch(batch)
        control.train_batch(batch)

    def decision(agent: RecurrentC51Agent) -> Action:
        hidden = None
        for index in range(8):
            _, hidden, _ = agent.select_action(
                np.array([index / 16, 1.0], np.float32),
                hidden=hidden,
                valid_actions=(Action.HOLD,),
                epsilon=0.0,
            )
        selected, _, _ = agent.select_action(
            np.array([0.5, 1.0], np.float32),
            hidden=hidden,
            valid_actions=(Action.HOLD, Action.CLOSE),
            epsilon=0.0,
        )
        return selected

    assert decision(control) == Action.CLOSE  # Fixture reproduces the collapse.
    assert decision(anchored) == Action.HOLD


def test_checkpoint_round_trip_preserves_recurrent_credit_horizon(
    tmp_path: Path,
) -> None:
    agent = _agent(2, n_step_return=8, recurrent_burn_in=64)

    checkpoint = agent.save(tmp_path / "credit-horizon.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")

    assert restored.n_step_return == 8
    assert restored.recurrent_burn_in == 64


def test_checkpoint_round_trip_preserves_training_competence_anchor(
    tmp_path: Path,
) -> None:
    agent = _agent(2, policy_retention_loss_weight=0.75)
    agent.retain_policy()
    assert agent.retention_anchor is not None
    expected = {
        key: value.detach().clone()
        for key, value in agent.retention_anchor.state_dict().items()
    }

    checkpoint = agent.save(tmp_path / "retention-anchor.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")

    assert restored.policy_retention_loss_weight == pytest.approx(0.75)
    assert restored.retention_anchor is not None
    for key, value in restored.retention_anchor.state_dict().items():
        torch.testing.assert_close(value, expected[key])
    restored.discard_retention_anchor()
    assert restored.retention_anchor is None


def test_soft_target_update_moves_gradually_after_every_learner_update() -> None:
    agent = _agent(
        2,
        seed=5,
        target_update_mode="soft",
        target_soft_tau=0.25,
    )
    sequence = tuple(
        Transition(
            observation=np.array([i, 0], np.float32),
            action=Action.WAIT,
            reward=0.1 if i < 3 else 1.0,
            next_observation=np.array([i + 1, 0], np.float32),
            terminated=i == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
            next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
        )
        for i in range(4)
    )
    target_before = agent.target.output.weight.detach().clone()

    agent.train_batch((sequence, sequence))

    online_after = agent.online.output.weight.detach()
    target_after = agent.target.output.weight.detach()
    torch.testing.assert_close(
        target_after,
        target_before.lerp(online_after, 0.25),
    )
    assert not torch.equal(target_after, online_after)


def test_hard_target_update_preserves_existing_interval_contract() -> None:
    agent = _agent(
        2,
        seed=5,
        target_sync_updates=1,
        target_update_mode="hard",
        target_soft_tau=1.0,
    )
    sequence = tuple(
        Transition(
            observation=np.array([i, 0], np.float32),
            action=Action.WAIT,
            reward=0.1 if i < 3 else 1.0,
            next_observation=np.array([i + 1, 0], np.float32),
            terminated=i == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
            next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
        )
        for i in range(4)
    )

    agent.train_batch((sequence, sequence))

    torch.testing.assert_close(
        agent.target.output.weight.detach(),
        agent.online.output.weight.detach(),
    )


def test_training_only_expansion_teacher_updates_shared_memory_and_is_discarded(
    tmp_path: Path,
) -> None:
    agent = _agent(2, seed=23, teacher_channels=4, teacher_loss_weight=0.2)
    sequence = tuple(
        Transition(
            observation=np.array([index, 0], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([index + 1, 0], np.float32),
            terminated=index == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
            next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
            teacher_target=np.array([0.9, 0.8, 0.1, 0.2], np.float32),
        )
        for index in range(4)
    )
    before = agent.online.input[1].weight.detach().clone()

    loss = agent.train_batch((sequence, sequence))
    assert agent.last_train_metrics["rl_loss"] > 0
    assert agent.last_train_metrics["teacher_loss"] > 0
    assert agent.last_train_metrics["total_loss"] == pytest.approx(loss)
    agent.discard_teacher()
    checkpoint = agent.save(tmp_path / "teacher-free.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")

    assert np.isfinite(loss)
    assert not torch.equal(before, agent.online.input[1].weight.detach())
    assert agent.teacher_channels == restored.teacher_channels == 0
    assert agent.online.teacher_output is None


def test_hidden_teacher_guidance_cannot_change_the_rl_update() -> None:
    plain = _agent(2, seed=83)
    taught = _agent(
        2,
        seed=83,
        teacher_channels=1,
        teacher_loss_weight=0.2,
    )
    sequence = tuple(
        Transition(
            observation=np.array([index, 1.0], np.float32),
            action=Action.HOLD,
            reward=0.1 if index == 3 else 0.0,
            next_observation=np.array([index + 1, 1.0], np.float32),
            terminated=False,
            valid_actions=(Action.HOLD, Action.CLOSE),
            next_valid_actions=(Action.HOLD, Action.CLOSE),
            teacher_target=np.array([0.9], np.float32),
        )
        for index in range(4)
    )

    plain.train_batch((sequence, sequence))
    taught.train_batch((sequence, sequence), teacher_weight_scale=0.0)

    assert taught.last_train_metrics["teacher_loss"] == 0.0
    for key, value in plain.online.state_dict().items():
        if not key.startswith("teacher_output."):
            torch.testing.assert_close(value, taught.online.state_dict()[key])


def test_teacher_channel_weights_keep_specialist_losses_independent() -> None:
    first = _agent(
        2,
        seed=31,
        teacher_channels=2,
        teacher_loss_weight=0.2,
        teacher_channel_loss_weights=(0.2, 0.0),
    )
    second = _agent(
        2,
        seed=31,
        teacher_channels=2,
        teacher_loss_weight=0.2,
        teacher_channel_loss_weights=(0.2, 0.0),
    )

    def sequence(second_target: float) -> tuple[Transition, ...]:
        return tuple(
            Transition(
                observation=np.array([index, 0], np.float32),
                action=Action.WAIT,
                reward=0.0,
                next_observation=np.array([index + 1, 0], np.float32),
                terminated=index == 3,
                valid_actions=(Action.WAIT,),
                next_valid_actions=(Action.WAIT,),
                teacher_target=np.array([0.8, second_target], np.float32),
            )
            for index in range(4)
        )

    loss_a = first.train_batch((sequence(0.0), sequence(0.0)))
    loss_b = second.train_batch((sequence(1.0), sequence(1.0)))

    assert loss_a == pytest.approx(loss_b)


def test_expansion_teacher_softly_guides_entry_values_without_training_management() -> None:
    agent = _agent(
        2,
        seed=29,
        teacher_channels=4,
        teacher_loss_weight=0.2,
        teacher_entry_search_loss_weight=0.3,
    )
    entry_sequence = tuple(
        Transition(
            observation=np.array([index, 0], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([index + 1, 0], np.float32),
            terminated=index == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
            next_valid_actions=(
                Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1
            ),
            teacher_target=np.array([0.9, 0.8, 0.1, 0.2], np.float32),
        )
        for index in range(4)
    )

    agent.train_batch((entry_sequence, entry_sequence))

    assert agent.last_train_metrics["entry_search_loss"] > 0

    management_sequence = tuple(
        Transition(
            observation=np.array([index, 0], np.float32),
            action=Action.HOLD,
            reward=0.0,
            next_observation=np.array([index + 1, 0], np.float32),
            terminated=index == 3,
            valid_actions=(Action.HOLD, Action.CLOSE),
            next_valid_actions=(Action.HOLD, Action.CLOSE),
            teacher_target=np.array([0.9, 0.8, 0.1, 0.2], np.float32),
        )
        for index in range(4)
    )

    agent.train_batch((management_sequence, management_sequence))

    assert agent.last_train_metrics["entry_search_loss"] == 0.0


def test_centered_entry_search_target_uses_training_base_rate_not_half() -> None:
    probabilities = torch.tensor([0.05, 0.10, 0.35], dtype=torch.float32)
    targets = centered_entry_search_target(
        probabilities,
        center=0.10,
        probability_epsilon=1e-6,
        teacher_temperature=1.0,
    )

    assert targets[0] < 0.5
    assert targets[1] == pytest.approx(0.5)
    assert targets[2] > 0.5
    assert torch.all(targets[1:] > targets[:-1])

    neutral_advantage = torch.tensor(0.0, requires_grad=True)
    high_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        neutral_advantage, targets[2]
    )
    high_loss.backward()
    assert neutral_advantage.grad is not None
    assert neutral_advantage.grad.item() < 0.0


def test_centered_entry_search_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="entry-search probability contract"):
        centered_entry_search_target(
            torch.tensor([0.2]),
            center=0.0,
            probability_epsilon=1e-6,
            teacher_temperature=1.0,
        )


def test_centered_entry_distillation_produces_teacher_free_greedy_entry() -> None:
    agent = _agent(
        2,
        seed=43,
        learning_rate=0.05,
        weight_decay=0.0,
        teacher_channels=4,
        teacher_loss_weight=1e-6,
        teacher_entry_search_loss_weight=10.0,
        teacher_entry_search_objective="centered_log_odds",
        teacher_entry_search_centers=(0.10, 0.10),
    )
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
    sequence = tuple(
        Transition(
            observation=np.zeros(2, np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.zeros(2, np.float32),
            terminated=index == 3,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            next_valid_actions=(
                () if index == 3 else (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            ),
            teacher_target=np.array([0.7, 0.7, 0.05, 0.05], np.float32),
        )
        for index in range(4)
    )

    for _ in range(20):
        agent.train_batch((sequence, sequence))
    agent.discard_teacher()
    action, _, values = agent.select_action(
        np.zeros(2, np.float32),
        hidden=None,
        valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        epsilon=0.0,
        return_action_values=True,
    )

    assert action == Action.ENTER_LONG_1
    assert values is not None
    assert values[int(Action.ENTER_LONG_1)] > values[int(Action.WAIT)]
    assert values[int(Action.ENTER_SHORT_1)] < values[int(Action.WAIT)]


@pytest.mark.parametrize("entry_age", range(1, 6))
@pytest.mark.parametrize(
    "entry_action",
    (Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
)
def test_post_launch_entry_action_targets_teach_exact_teacher_free_entry_timing(
    tmp_path: Path,
    entry_age: int,
    entry_action: Action,
) -> None:
    agent = _agent(
        5,
        seed=53 + entry_age + int(entry_action),
        hidden_dim=16,
        learning_rate=0.03,
        weight_decay=0.0,
        entry_action_loss_weight=20.0,
    )
    observations = tuple(np.eye(5, dtype=np.float32))
    sequence = tuple(
        Transition(
            observation=observations[age - 1],
            action=Action.WAIT,
            reward=0.0,
            next_observation=(
                observations[age]
                if age < 5
                else np.zeros(5, np.float32)
            ),
            terminated=age == 5,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            next_valid_actions=(
                ()
                if age == 5
                else (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            ),
            # Only decisions through the chosen executable age are labels.
            # Later decisions are unavailable/censored, never negative WAITs.
            entry_action_target=(
                entry_action if age == entry_age else Action.WAIT
                if age <= entry_age
                else None
            ),
        )
        for age in range(1, 6)
    )

    for _ in range(80):
        agent.train_batch((sequence, sequence))

    assert tuple(item.entry_action_target for item in sequence) == tuple(
        Action.WAIT if age < entry_age else entry_action
        if age == entry_age else None
        for age in range(1, 6)
    )
    assert agent.last_train_metrics["entry_action_supervised_rows"] == 2 * entry_age
    expected = tuple(
        Action.WAIT if age < entry_age else entry_action
        for age in range(1, entry_age + 1)
    )

    def greedy_trace(current: RecurrentC51Agent) -> tuple[Action, ...]:
        hidden = None
        actions = []
        for observation in observations[:entry_age]:
            action, hidden, _ = current.select_action(
                observation,
                hidden=hidden,
                valid_actions=(
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                epsilon=0.0,
            )
            actions.append(action)
        return tuple(actions)

    assert greedy_trace(agent) == expected
    agent.discard_teacher()
    assert agent.entry_action_loss_weight == 0.0
    assert greedy_trace(agent) == expected
    checkpoint = agent.save(tmp_path / "post-launch-action.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    assert greedy_trace(restored) == expected


def test_post_launch_entry_action_targets_have_no_effect_at_zero_auxiliary_scale() -> None:
    plain = _agent(5, seed=67)
    taught = _agent(
        5,
        seed=67,
        entry_action_loss_weight=1.0,
    )
    sequence = tuple(
        Transition(
            observation=np.eye(5, dtype=np.float32)[age - 1],
            action=Action.WAIT,
            reward=0.0,
            next_observation=(
                np.eye(5, dtype=np.float32)[age]
                if age < 5
                else np.zeros(5, np.float32)
            ),
            terminated=age == 5,
            valid_actions=(
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            ),
            next_valid_actions=(
                ()
                if age == 5
                else (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            ),
            entry_action_target=Action.ENTER_LONG_1,
        )
        for age in range(1, 6)
    )

    plain.train_batch((sequence, sequence))
    taught.train_batch((sequence, sequence), entry_action_weight_scale=0.0)

    assert taught.last_train_metrics["entry_action_loss"] == 0.0
    assert taught.last_train_metrics["entry_action_supervised_rows"] == 0.0
    assert taught.last_train_metrics["total_loss"] == pytest.approx(
        taught.last_train_metrics["rl_loss"]
    )
    for key, value in plain.online.state_dict().items():
        torch.testing.assert_close(value, taught.online.state_dict()[key])


def test_soft_regime_context_and_entry_actions_co_train_without_inference_dependency(
    tmp_path: Path,
) -> None:
    agent = _agent(
        7,
        seed=71,
        hidden_dim=24,
        learning_rate=0.03,
        weight_decay=0.0,
        teacher_channels=2,
        teacher_loss_weight=0.5,
        entry_action_loss_weight=20.0,
    )
    ages = np.eye(5, dtype=np.float32)

    def context(age: int, *, trending: bool) -> np.ndarray:
        regime = np.array([1.0, 0.0] if trending else [0.0, 1.0], np.float32)
        return np.concatenate((ages[age - 1], regime))

    def sequence(*, trending: bool) -> tuple[Transition, ...]:
        semantic_target = np.array(
            [0.9, 0.1] if trending else [0.1, 0.9], np.float32
        )
        return tuple(
            Transition(
                observation=context(age, trending=trending),
                action=Action.WAIT,
                reward=0.0,
                next_observation=(
                    context(age + 1, trending=trending)
                    if age < 5
                    else np.zeros(7, np.float32)
                ),
                terminated=age == 5,
                valid_actions=(
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                next_valid_actions=(
                    ()
                    if age == 5
                    else (
                        Action.WAIT,
                        Action.ENTER_LONG_1,
                        Action.ENTER_SHORT_1,
                    )
                ),
                teacher_target=semantic_target,
                entry_action_target=(
                    Action.WAIT
                    if age < 3
                    else (Action.ENTER_LONG_1 if trending else Action.WAIT)
                    if age == 3
                    else None
                ),
            )
            for age in range(1, 6)
        )

    trend = sequence(trending=True)
    chop = sequence(trending=False)
    np.testing.assert_array_equal(trend[2].observation[:5], chop[2].observation[:5])
    assert trend[2].entry_action_target == Action.ENTER_LONG_1
    assert chop[2].entry_action_target == Action.WAIT

    for _ in range(100):
        agent.train_batch((trend, chop))

    assert agent.last_train_metrics["teacher_loss"] > 0.0
    assert agent.last_train_metrics["entry_action_loss"] > 0.0
    assert agent.last_train_metrics["entry_action_supervised_rows"] == 6.0

    def greedy_trace(
        current: RecurrentC51Agent, *, trending: bool
    ) -> tuple[Action, ...]:
        hidden = None
        actions = []
        for age in range(1, 4):
            action, hidden, _ = current.select_action(
                context(age, trending=trending),
                hidden=hidden,
                valid_actions=(
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                epsilon=0.0,
            )
            actions.append(action)
        return tuple(actions)

    expected_trend = (Action.WAIT, Action.WAIT, Action.ENTER_LONG_1)
    expected_chop = (Action.WAIT, Action.WAIT, Action.WAIT)
    assert greedy_trace(agent, trending=True) == expected_trend
    assert greedy_trace(agent, trending=False) == expected_chop

    agent.discard_teacher()
    assert agent.teacher_channels == 0
    assert agent.entry_action_loss_weight == 0.0
    assert greedy_trace(agent, trending=True) == expected_trend
    assert greedy_trace(agent, trending=False) == expected_chop

    checkpoint = agent.save(tmp_path / "regime-confluence.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    assert greedy_trace(restored, trending=True) == expected_trend
    assert greedy_trace(restored, trending=False) == expected_chop


def test_exploration_still_reports_greedy_action_values_for_diagnostics() -> None:
    agent = _agent(2, seed=47)
    action, _, values = agent.select_action(
        np.zeros(2, np.float32),
        hidden=None,
        valid_actions=(
            Action.WAIT,
            Action.ENTER_LONG_1,
            Action.ENTER_SHORT_1,
        ),
        epsilon=1.0,
        return_action_values=True,
    )

    assert action in {
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    }
    assert values is not None
    assert np.isfinite(values[[
        int(Action.WAIT),
        int(Action.ENTER_LONG_1),
        int(Action.ENTER_SHORT_1),
    ]]).all()


def test_discard_teacher_and_checkpoint_round_trip_preserve_policy(
    tmp_path: Path,
) -> None:
    agent = _agent(
        3,
        seed=41,
        teacher_channels=4,
        teacher_loss_weight=0.2,
        teacher_entry_search_loss_weight=0.3,
        teacher_entry_search_objective="centered_log_odds",
        teacher_entry_search_centers=(0.10, 0.11),
    )
    observations = tuple(
        np.array([index / 10.0, 0.25, -0.5], np.float32)
        for index in range(6)
    )

    def trace(current: RecurrentC51Agent):
        hidden = None
        rows = []
        for observation in observations:
            action, hidden, values = current.select_action(
                observation,
                hidden=hidden,
                valid_actions=(
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                epsilon=0.0,
                return_action_values=True,
            )
            rows.append((action, values.copy(), hidden.detach().clone()))
        return rows

    before = trace(agent)
    agent.discard_teacher()
    after = trace(agent)
    checkpoint = agent.save(tmp_path / "teacher-free.pt", manifest={})
    restored, _ = RecurrentC51Agent.load(checkpoint, device="cpu")
    after_round_trip = trace(restored)

    for expected, actual, restored_actual in zip(
        before, after, after_round_trip, strict=True
    ):
        for observed in (actual, restored_actual):
            assert observed[0] == expected[0]
            np.testing.assert_array_equal(observed[1], expected[1])
            torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)


def test_zero_teacher_scale_skips_all_auxiliary_teacher_work() -> None:
    full = _agent(
        2,
        seed=37,
        teacher_channels=4,
        teacher_loss_weight=0.2,
        teacher_entry_search_loss_weight=0.3,
    )
    autonomous = _agent(
        2,
        seed=37,
        teacher_channels=4,
        teacher_loss_weight=0.2,
        teacher_entry_search_loss_weight=0.3,
    )
    sequence = tuple(
        Transition(
            observation=np.array([index, 0], np.float32),
            action=Action.WAIT,
            reward=0.0,
            next_observation=np.array([index + 1, 0], np.float32),
            terminated=index == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
            next_valid_actions=(
                Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1
            ),
            teacher_target=np.array([0.9, 0.8, 0.1, 0.2], np.float32),
        )
        for index in range(4)
    )

    full.train_batch((sequence, sequence), teacher_weight_scale=1.0)
    autonomous.train_batch((sequence, sequence), teacher_weight_scale=0.0)

    assert autonomous.last_train_metrics["teacher_loss"] == 0.0
    assert autonomous.last_train_metrics["entry_search_loss"] == 0.0
    assert autonomous.last_train_metrics["teacher_weight_scale"] == 0.0
    assert autonomous.last_train_metrics["total_loss"] == pytest.approx(
        autonomous.last_train_metrics["rl_loss"]
    )
    assert full.last_train_metrics["total_loss"] > full.last_train_metrics["rl_loss"]


def test_warm_start_preserves_policy_but_resets_training_state_and_teacher(
    tmp_path: Path,
) -> None:
    parent = _agent(2, seed=11)
    with torch.no_grad():
        parent.online.input[1].weight.fill_(0.25)
        parent.target.input[1].weight.fill_(0.5)
    parent._updates = 17
    checkpoint = parent.save(
        tmp_path / "parent.pt", manifest={"candidate_id": "parent-1"}
    )

    child, manifest = RecurrentC51Agent.warm_start(
        checkpoint,
        config={
            "observation_dim": 2,
            "hidden_dim": 8,
            "atoms": 11,
            "value_min": -3.0,
            "value_max": 3.0,
            "gamma": 0.997,
            "learning_rate": 5e-5,
            "weight_decay": 1e-5,
            "gradient_clip": 10.0,
            "target_sync_updates": 250,
            "device": "cpu",
            "seed": 29,
            "teacher_channels": 4,
            "teacher_loss_weight": 0.2,
        },
    )

    torch.testing.assert_close(
        child.online.input[1].weight,
        parent.online.input[1].weight,
    )
    torch.testing.assert_close(
        child.target.input[1].weight,
        parent.target.input[1].weight,
    )
    assert child.teacher_channels == 4
    assert child.online.teacher_output is not None
    assert child._updates == 0
    assert child.optimizer.state == {}
    assert child.learning_rate == 5e-5
    assert manifest == {"candidate_id": "parent-1"}


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable on this test host",
)
def test_mps_training_update_action_selection_and_checkpoint_resume(
    tmp_path: Path,
) -> None:
    agent = _agent(
        8,
        hidden_dim=16,
        atoms=21,
        device="mps",
        seed=17,
        target_sync_updates=1,
        n_step_return=2,
        recurrent_burn_in=1,
    )
    sequence = tuple(
        Transition(
            observation=np.full(8, index / 10, np.float32),
            action=Action.WAIT,
            reward=0.2 if index < 3 else 1.0,
            next_observation=np.full(8, (index + 1) / 10, np.float32),
            terminated=index == 3,
            valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
            next_valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
        )
        for index in range(4)
    )

    loss = agent.train_batch((sequence, sequence))
    selected, hidden, values = agent.select_action(
        np.zeros(8, np.float32),
        hidden=None,
        valid_actions=(Action.WAIT, Action.ENTER_LONG_1),
        epsilon=0.0,
        return_action_values=True,
    )
    checkpoint = agent.save(tmp_path / "agent.pt", manifest={"device": "mps"})
    restored, manifest = RecurrentC51Agent.load(checkpoint, device="mps")
    resumed_loss = restored.train_batch((sequence, sequence))

    assert np.isfinite(loss)
    assert np.isfinite(resumed_loss)
    assert selected in (Action.WAIT, Action.ENTER_LONG_1)
    assert hidden.device.type == "mps"
    assert np.isfinite(values[[int(Action.WAIT), int(Action.ENTER_LONG_1)]]).all()
    assert restored.device.type == "mps"
    assert restored.n_step_return == 2
    assert restored.recurrent_burn_in == 1
    assert manifest == {"device": "mps"}
