from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from propevolve.agent import RecurrentC51Agent, RecurrentC51Network, resolve_device
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
    assert manifest == {"device": "mps"}
