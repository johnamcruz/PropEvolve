from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.environment import MarketSeries
from propevolve.replay import BalancedSequenceReplay
from propevolve.training import assert_temporal_role, evaluate_agent, train_agent


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


class MultiMarketEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()
        self.episode_tickers = []
        self.ticker = ""

    def reset(self, *, options=None):
        self.index = 0
        self.ticker = options["ticker"]
        self.episode_tickers.append(self.ticker)
        return np.array([0.0], np.float32), {
            "ticker": self.ticker,
            "valid_actions": (Action.WAIT,),
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["ticker"] = self.ticker
        return observation, reward, terminated, truncated, info


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
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=1,
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


def test_one_shared_agent_trains_on_balanced_single_market_episodes() -> None:
    tickers = ("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN")
    environment = MultiMarketEnvironment()
    agent = Agent()
    replay = BalancedSequenceReplay(capacity_episodes=30, sequence_length=2, seed=7)

    result = train_agent(
        agent,
        environment,
        episodes=18,
        replay=replay,
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=tickers,
        ticker_seed=7,
    )

    assert Counter(environment.episode_tickers) == Counter({ticker: 2 for ticker in tickers})
    assert result.episodes == 18
    assert agent.updates == 18


def test_temporal_preflight_rejects_any_sealed_holdout_timestamp() -> None:
    timestamps = np.array([
        "2025-12-31T23:57:00", "2026-01-01T00:00:00"
    ], dtype="datetime64[ns]")
    close = np.array([100.0, 101.0], np.float32)
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=close,
        high=close,
        low=close,
        close=close,
        embeddings=np.zeros((2, 4), np.float32),
    )

    with pytest.raises(ValueError, match="temporal contract"):
        assert_temporal_role(
            {"NQ": market},
            role="selection",
            start="2025-01-01",
            end="2026-01-01",
            sealed_start="2026-01-01",
        )
