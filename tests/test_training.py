from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import propevolve.training as training_module
from propevolve.cache import build_embedding_cache
from propevolve.decision import Action
from propevolve.environment import ChallengeSpec, MarketSeries
from propevolve.replay import BalancedSequenceReplay
from propevolve.training import (
    HistoricalCandidateRunner,
    TrainingResult,
    TrainingProgress,
    assert_temporal_role,
    evaluate_agent,
    prop_safety_objective,
    train_agent,
)


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
            "trade_count": 2 if terminated else 0,
            "win_count": 1 if terminated else 0,
            "winning_r_sum": 2.5 if terminated else 0.0,
            "equity_pnl": 6_000.0 if terminated else 0.0,
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


def test_historical_candidate_flow_materializes_the_challenge_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReachedEnvironment(RuntimeError):
        pass

    captured = []

    class CapturingEnvironment:
        def __init__(self, markets, *, tick_values, round_trip_fees, spec, seed):
            captured.append(spec)
            raise ReachedEnvironment

    monkeypatch.setattr(
        training_module.AssetContract,
        "load",
        classmethod(lambda cls, path: object()),
    )
    monkeypatch.setattr(
        training_module,
        "load_markets",
        lambda **kwargs: {"NQ": object()},
    )
    monkeypatch.setattr(training_module, "assert_temporal_role", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        training_module,
        "HistoricalChallengeEnv",
        CapturingEnvironment,
    )
    config = {
        "_root": ".",
        "assets": "config/local-assets.json",
        "cache_root": "cache",
        "tickers": ("NQ",),
        "deployment_tickers": ("NQ",),
        "timeframe_minutes": 3,
        "temporal": {
            "train_start": "2021-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01",
        },
        "challenge": {
            "profit_target": 6000.0,
            "max_loss": 3000.0,
            "episode_days": 30,
            "bars_per_day": 480,
            "max_position_size": 1,
            "minimum_mll_headroom": 250.0,
            "trailing_mll_lock": True,
            "terminal_pass_reward": 250.0,
            "terminal_blow_reward": -1500.0,
            "terminal_timeout_reward": -2.0,
            "terminal_pass_speed_reward_per_day": 20.0,
            "reward_scale": 1000.0,
            "mll_proximity_penalty_coefficient": 0.0,
            "lead_giveback_penalty_coefficient": 0.0,
            "large_win_threshold_r": 2.0,
            "large_win_bonus_coefficient": 0.0,
        },
        "training": {"seed": 7},
        "point_values": {"NQ": 20.0},
        "round_trip_fees": {"NQ": 3.84},
    }

    with pytest.raises(ReachedEnvironment):
        HistoricalCandidateRunner().run(
            config,
            parent_candidate_ids=(),
            hypothesis="flow regression",
        )

    assert len(captured) == 1
    assert isinstance(captured[0], ChallengeSpec)


def test_historical_candidate_runs_the_complete_real_training_flow(
    tmp_path: Path,
) -> None:
    class TinyEncoder:
        checkpoint = tmp_path / "checkpoint"

        def encode(self, windows: np.ndarray) -> np.ndarray:
            return windows.mean(axis=2)

    data = tmp_path / "data"
    data.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"tiny-mask")
    adapter = checkpoint / "adapter_config.json"
    adapter.write_text("{}\n")
    source = data / "NQ_3min.csv"
    times = pd.to_datetime([
        "2024-12-31T23:39:00Z",
        "2024-12-31T23:42:00Z",
        "2024-12-31T23:45:00Z",
        "2024-12-31T23:48:00Z",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:03:00Z",
        "2025-01-01T00:06:00Z",
        "2025-01-01T00:09:00Z",
        "2025-01-01T00:12:00Z",
    ])
    close = np.arange(100, 109, dtype=float)
    pd.DataFrame({
        "datetime": times,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.arange(10, 19, dtype=float),
    }).to_csv(source, index=False)
    cache_root = tmp_path / "cache"
    build_embedding_cache(
        source=source,
        destination=cache_root / "NQ",
        ticker="NQ",
        encoder=TinyEncoder(),
        checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        research_end_exclusive="2026-01-01",
        context_length=2,
        stride=1,
        chunk_windows=4,
        timeframe_minutes=3,
    )
    assets_path = tmp_path / "assets.json"
    assets_path.write_text(json.dumps({
        "schema": "propevolve_local_assets_v1",
        "market_data": str(data),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "adapter_config_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "embedding_cache": None,
    }))
    config_path = tmp_path / "experiment.json"
    config_path.write_text("{}")
    config = {
        "_root": str(tmp_path),
        "_path": str(config_path),
        "assets": str(assets_path),
        "cache_root": str(cache_root),
        "output": str(tmp_path / "run"),
        "tickers": ("NQ",),
        "deployment_tickers": ("NQ",),
        "training_only_tickers": (),
        "timeframe_minutes": 3,
        "temporal": {
            "train_start": "2024-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01",
        },
        "challenge": {
            "profit_target": 6000.0,
            "max_loss": 3000.0,
            "episode_days": 1,
            "bars_per_day": 2,
            "max_position_size": 1,
            "minimum_mll_headroom": 250.0,
            "trailing_mll_lock": True,
            "terminal_pass_reward": 250.0,
            "terminal_blow_reward": -1500.0,
            "terminal_timeout_reward": -2.0,
            "terminal_pass_speed_reward_per_day": 20.0,
            "reward_scale": 1000.0,
        },
        "point_values": {"NQ": 20.0},
        "round_trip_fees": {"NQ": 3.84},
        "agent": {
            "hidden_dim": 8,
            "atoms": 11,
            "value_min": -3.0,
            "value_max": 3.0,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 10.0,
            "target_sync_updates": 2,
            "device": "cpu",
        },
        "training": {
            "episodes": 2,
            "minimum_environment_steps": 2,
            "validation_episodes": 1,
            "replay_capacity_episodes": 2,
            "replay_capacity_transitions": 8,
            "sequence_length": 1,
            "terminal_sequence_fraction": 0.5,
            "warmup_episodes": 1,
            "updates_per_episode": 1,
            "batch_sequences": 1,
            "recurrent_horizon": 2,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "seed": 7,
            "checkpoint_every_episodes": 1,
        },
    }

    candidate, evaluation = HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=(),
        hypothesis="complete flow regression",
    )

    assert candidate.model_path.is_file()
    assert evaluation.path.is_file()
    diagnostic_path = tmp_path / "run" / "training-diagnostics.jsonl"
    assert diagnostic_path.is_file()
    diagnostics = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert diagnostics
    assert diagnostics[-1]["schema"] == "propevolve_episode_diagnostic_v1"
    summary_path = tmp_path / "run" / "training-diagnostic-summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "propevolve_training_diagnostic_summary_v1"
    assert summary["source_sha256"] == hashlib.sha256(
        diagnostic_path.read_bytes()
    ).hexdigest()
    assert summary["overall"]["episodes"] == len(diagnostics)
    assert summary["by_ticker"]["NQ"]["episodes"] == len(diagnostics)
    assert evaluation.candidate_id == candidate.candidate_id
    assert evaluation.status in {"PASS", "FAIL", "REVISE"}
    assert set(evaluation.metrics) >= {
        "training.pass_rate",
        "selection.pass_rate",
        "selection.blow_rate",
        "selection.pass_minus_blow",
        "selection.timeout_mean_trade_count",
        "selection.timeout_trade_win_rate",
        "selection.timeout_average_win_r",
        "selection.timeout_mean_terminal_pnl",
    }


def test_training_collects_episodes_then_updates_from_balanced_replay(capsys) -> None:
    agent = Agent()
    replay = BalancedSequenceReplay(capacity_episodes=10, sequence_length=2, seed=1)
    diagnostics = []

    result = train_agent(
        agent,
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=replay,
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=1,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert result.passes == 2
    assert result.environment_steps == 8
    assert result.blows == result.timeouts == 0
    assert len(replay) == 2
    assert agent.updates == 2
    assert result.mean_loss == 0.5
    assert result.trade_win_rate == 0.5
    assert result.average_win_r == 2.5
    assert len(diagnostics) == 2
    assert diagnostics[-1]["schema"] == "propevolve_episode_diagnostic_v1"
    assert diagnostics[-1]["episode"] == 2
    assert diagnostics[-1]["outcome"] == "pass"
    assert diagnostics[-1]["expectancy_r"] == 0.0
    assert diagnostics[-1]["entry_epsilon"] == pytest.approx(0.135)
    assert diagnostics[-1]["management_epsilon"] == pytest.approx(0.135)
    assert diagnostics[-1]["updates"] == 1
    assert diagnostics[-1]["mean_training_loss"] == 0.5
    assert diagnostics[-1]["cumulative_pass_rate"] == 1.0
    assert diagnostics[-1]["cumulative_blow_rate"] == 0.0
    assert diagnostics[-1]["cumulative_average_balance"] == 6_000.0
    assert diagnostics[-1]["action_counts"]["WAIT"] == 4
    assert diagnostics[-1]["shadow_h50_complete_trades"] == 0
    assert (
        "winR=+0.000R balance=+6000.00 avg_balance=+6000.00 steps=4/8"
        in capsys.readouterr().out
    )


def test_training_uses_lower_exploration_for_position_management() -> None:
    class RecordingAgent(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.epsilons = []

        def select_action(self, observation, *, hidden, valid_actions, epsilon):
            self.epsilons.append((valid_actions, epsilon))
            return valid_actions[0], None, None

    class PositionEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT, Action.ENTER_LONG_1),
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 2
            return np.array([self.index], np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.HOLD, Action.CLOSE),
                "outcome": "timeout" if terminated else None,
                "ticker": "NQ",
                "primary_side": "long",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    agent = RecordingAgent()
    train_agent(
        agent,
        PositionEnvironment(),
        episodes=1,
        minimum_environment_steps=2,
        replay=BalancedSequenceReplay(capacity_episodes=2, sequence_length=1, seed=3),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        management_epsilon_start=0.05,
        management_epsilon_end=0.01,
        episode_tickers=None,
        ticker_seed=3,
    )

    assert agent.epsilons == [
        ((Action.WAIT, Action.ENTER_LONG_1), 0.25),
        ((Action.HOLD, Action.CLOSE), 0.05),
    ]


def test_training_resumes_from_an_episode_boundary() -> None:
    checkpoints: list[TrainingProgress] = []
    first = train_agent(
        Agent(),
        Environment(),
        episodes=1,
        minimum_environment_steps=4,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            capacity_transitions=16,
            sequence_length=2,
            seed=5,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        checkpoint_every_episodes=1,
        checkpoint_callback=checkpoints.append,
    )
    assert first.environment_steps == 4

    resumed = train_agent(
        Agent(),
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            capacity_transitions=16,
            sequence_length=2,
            seed=5,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        resume=checkpoints[-1],
    )

    assert resumed.episodes == 2
    assert resumed.environment_steps == 8
    assert resumed.passes == 2


def test_prop_safety_objective_hard_ranks_any_blow_below_zero_blow() -> None:
    common = dict(
        episodes=100,
        environment_steps=1000,
        passes=50,
        timeouts=50,
        trade_count=100,
        win_count=40,
        winning_r_sum=80.0,
        worst_pnl=-2_000.0,
        mean_terminal_pnl=2_000.0,
        mean_reward=0.0,
        mean_loss=1.0,
    )
    safe = TrainingResult(blows=0, **common)
    unsafe = TrainingResult(blows=1, **{**common, "passes": 99, "timeouts": 0})

    assert prop_safety_objective(
        unsafe, max_loss=3_000.0, profit_target=6_000.0
    ) < -1.0
    assert prop_safety_objective(
        safe, max_loss=3_000.0, profit_target=6_000.0
    ) >= 0.0


def test_evaluation_never_updates_agent() -> None:
    agent = Agent()
    result = evaluate_agent(agent, Environment(), episodes=2, recurrent_horizon=2)
    assert result.passes == 2
    assert agent.updates == 0


def test_evaluation_reports_pass_and_timeout_economics_separately(capsys) -> None:
    class OutcomeEnvironment:
        def __init__(self) -> None:
            self.episode = -1

        def reset(self):
            self.episode += 1
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,)
            }

        def step(self, action):
            outcome = ("pass", "timeout")[self.episode]
            info = {
                "valid_actions": (),
                "ticker": ("NQ", "SI")[self.episode],
                "outcome": outcome,
                "trade_count": (4, 10)[self.episode],
                "win_count": (2, 3)[self.episode],
                "winning_r_sum": (6.0, 3.0)[self.episode],
                "equity_pnl": (6_000.0, 1_500.0)[self.episode],
            }
            return np.array([1.0], np.float32), 1.0, True, False, info

    result = evaluate_agent(
        Agent(), OutcomeEnvironment(), episodes=2, recurrent_horizon=2
    )

    assert result.outcome("pass").mean_trade_count == 4.0
    assert result.outcome("pass").trade_win_rate == 0.5
    assert result.outcome("pass").average_win_r == 3.0
    assert result.outcome("timeout").mean_trade_count == 10.0
    assert result.outcome("timeout").trade_win_rate == 0.3
    assert result.outcome("timeout").average_win_r == 1.0
    assert result.outcome("timeout").mean_terminal_pnl == 1_500.0
    output = capsys.readouterr().out
    assert (
        "[validation] episode=1/2 ticker=NQ outcome=pass "
        "reward=+1.0000 trades=4 WR=50.0% winR=+3.000R pnl=+6000.00 "
        "cumulative_pass=1 cumulative_blow=0 cumulative_timeout=0"
    ) in output
    assert (
        "[validation] episode=2/2 ticker=SI outcome=timeout "
        "reward=+1.0000 trades=10 WR=30.0% winR=+1.000R pnl=+1500.00 "
        "cumulative_pass=1 cumulative_blow=0 cumulative_timeout=1"
    ) in output
    assert (
        "[validation] COMPLETE episodes=2 pass=1 blow=0 timeout=1 "
        "WR=35.7% winR=+1.800R mean_pnl=+3750.00"
    ) in output


def test_one_shared_agent_trains_on_balanced_single_market_episodes() -> None:
    tickers = ("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN")
    environment = MultiMarketEnvironment()
    agent = Agent()
    replay = BalancedSequenceReplay(capacity_episodes=30, sequence_length=2, seed=7)

    result = train_agent(
        agent,
        environment,
        episodes=18,
        minimum_environment_steps=72,
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
