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
