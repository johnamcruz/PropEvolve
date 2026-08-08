from __future__ import annotations

import numpy as np

from propevolve.decision import Action
from propevolve.replay import BalancedSequenceReplay
from propevolve.training import evaluate_agent, train_agent


class Agent:
    def __init__(self) -> None:
        self.updates = 0

    def select_action(self, observation, *, hidden, valid_actions, epsilon):
        return Action.WAIT, None, np.zeros(len(Action), np.float32)

    def train_batch(self, sequences):
        self.updates += 1
        return 0.5


class Environment:
    def __init__(self) -> None:
        self.index = 0

    def reset(self):
        self.index = 0
        return np.array([0.0], np.float32), {"valid_actions": (Action.WAIT,)}

    def step(self, action):
        self.index += 1
        terminated = self.index == 4
        info = {
            "valid_actions": () if terminated else (Action.WAIT,),
            "outcome": "pass" if terminated else None,
            "ticker": "NQ",
            "primary_side": "flat",
        }
        return np.array([self.index], np.float32), 0.25, terminated, False, info


def test_training_collects_episodes_then_updates_from_balanced_replay() -> None:
    agent = Agent()
    replay = BalancedSequenceReplay(capacity_episodes=10, sequence_length=2, seed=1)

    result = train_agent(
        agent,
        Environment(),
        episodes=2,
        replay=replay,
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
    )

    assert result.passes == 2
    assert result.blows == result.timeouts == 0
    assert len(replay) == 2
    assert agent.updates == 2
    assert result.mean_loss == 0.5


def test_evaluation_never_updates_agent() -> None:
    agent = Agent()
    result = evaluate_agent(agent, Environment(), episodes=2, recurrent_horizon=2)
    assert result.passes == 2
    assert agent.updates == 0

