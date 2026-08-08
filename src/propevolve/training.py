"""Historical training and temporal evaluation for the PropEvolve POC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np

from .assets import AssetContract
from .cache import load_market_series
from .decision import Action
from .environment import HistoricalChallengeEnv, MarketSeries
from .replay import BalancedSequenceReplay, Episode, Transition

if TYPE_CHECKING:
    from .agent import RecurrentC51Agent


@dataclass(frozen=True)
class TrainingResult:
    episodes: int
    passes: int
    blows: int
    timeouts: int
    mean_reward: float
    mean_loss: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_markets(
    *,
    asset_contract: AssetContract,
    cache_root: str | Path,
    tickers: tuple[str, ...],
    timeframe_minutes: int,
    start: str | None,
    end: str | None,
) -> dict[str, MarketSeries]:
    asset_contract.verify()
    root = Path(cache_root)
    markets = {}
    for ticker in tickers:
        markets[ticker] = load_market_series(
            Path(asset_contract.market_data) / f"{ticker}_{timeframe_minutes}min.csv",
            root / ticker,
            ticker=ticker,
            start=start,
            end=end,
        )
    return markets


def train_agent(
    agent: RecurrentC51Agent,
    environment: HistoricalChallengeEnv,
    *,
    episodes: int,
    replay: BalancedSequenceReplay,
    warmup_episodes: int = 8,
    updates_per_episode: int = 8,
    batch_sequences: int = 16,
    recurrent_horizon: int = 64,
    epsilon_start: float = 0.25,
    epsilon_end: float = 0.02,
) -> TrainingResult:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    rewards, losses = [], []
    for episode_index in range(episodes):
        observation, reset_info = environment.reset()
        valid = tuple(reset_info["valid_actions"])
        hidden = None
        transitions = []
        total_reward = 0.0
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (
            episode_index / max(1, episodes - 1)
        )
        terminal_info = reset_info
        step_index = 0
        while True:
            if step_index and step_index % recurrent_horizon == 0:
                hidden = None
            action, hidden, _ = agent.select_action(
                observation,
                hidden=hidden,
                valid_actions=valid,
                epsilon=epsilon,
            )
            next_observation, reward, terminated, _, info = environment.step(action)
            next_valid = tuple(info["valid_actions"])
            transitions.append(Transition(
                observation=observation,
                action=Action(action),
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                valid_actions=valid,
                next_valid_actions=next_valid,
            ))
            total_reward += reward
            observation, valid = next_observation, next_valid
            terminal_info = info
            step_index += 1
            if terminated:
                break
        outcome = str(terminal_info["outcome"])
        outcomes[outcome] += 1
        rewards.append(total_reward)
        if len(transitions) >= replay.sequence_length:
            replay.add(Episode(
                episode_id=f"historical-{episode_index}-{time.time_ns()}",
                ticker=str(terminal_info["ticker"]),
                outcome=outcome,
                primary_side=str(terminal_info["primary_side"]),
                ended_at_ns=time.time_ns(),
                transitions=tuple(transitions),
            ))
        if episode_index + 1 >= warmup_episodes and len(replay):
            for _ in range(updates_per_episode):
                losses.append(agent.train_batch(replay.sample(batch_sequences)))
        print(
            f"[train] episode={episode_index + 1}/{episodes} ticker={terminal_info['ticker']} "
            f"outcome={outcome} reward={total_reward:.4f} replay={len(replay)}",
            flush=True,
        )
    return TrainingResult(
        episodes=episodes,
        passes=outcomes["pass"],
        blows=outcomes["blow"],
        timeouts=outcomes["timeout"],
        mean_reward=float(np.mean(rewards)),
        mean_loss=float(np.mean(losses)) if losses else float("nan"),
    )


def evaluate_agent(
    agent: RecurrentC51Agent,
    environment: HistoricalChallengeEnv,
    *,
    episodes: int,
    recurrent_horizon: int = 64,
) -> TrainingResult:
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    rewards = []
    for _ in range(episodes):
        observation, info = environment.reset()
        valid = tuple(info["valid_actions"])
        hidden = None
        total = 0.0
        step_index = 0
        while True:
            if step_index and step_index % recurrent_horizon == 0:
                hidden = None
            action, hidden, _ = agent.select_action(
                observation, hidden=hidden, valid_actions=valid, epsilon=0.0
            )
            observation, reward, terminated, _, info = environment.step(action)
            valid = tuple(info["valid_actions"])
            total += reward
            step_index += 1
            if terminated:
                break
        outcomes[str(info["outcome"])] += 1
        rewards.append(total)
    return TrainingResult(
        episodes=episodes,
        passes=outcomes["pass"],
        blows=outcomes["blow"],
        timeouts=outcomes["timeout"],
        mean_reward=float(np.mean(rewards)),
        mean_loss=float("nan"),
    )


def write_run_report(
    path: str | Path,
    *,
    config_path: str | Path,
    assets: AssetContract,
    training: TrainingResult,
    validation: TrainingResult,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path).resolve(strict=True)
    payload = {
        "schema": "propevolve_historical_run_v1",
        "decision": "PROCEED" if validation.passes > validation.blows else "REVISE",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": assets.checkpoint_sha256,
        "training": asdict(training),
        "validation": asdict(validation),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path
