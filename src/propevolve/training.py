"""Historical training and temporal evaluation for the PropEvolve POC."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np

from .assets import AssetContract
from .cache import load_market_series
from .decision import Action
from .environment import HistoricalChallengeEnv, MarketSeries
from .evolution import (
    CandidateArchive,
    EvaluationGate,
    EvaluationStage,
    EvaluatorCascade,
)
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


class HistoricalCandidateRunner:
    """Train one immutable historical challenger and evaluate its first gate."""

    def run(
        self,
        config: dict,
        *,
        parent_candidate_ids: tuple[str, ...],
        hypothesis: str,
    ):
        from .agent import RecurrentC51Agent

        root = Path(config["_root"])
        assets = AssetContract.load(_resolve(root, config["assets"]))
        temporal = config["temporal"]
        cache_root = _resolve(root, config["cache_root"])
        train_markets = load_markets(
            asset_contract=assets,
            cache_root=cache_root,
            tickers=tuple(config["tickers"]),
            timeframe_minutes=int(config["timeframe_minutes"]),
            start=temporal["train_start"],
            end=temporal["train_end"],
        )
        validation_markets = load_markets(
            asset_contract=assets,
            cache_root=cache_root,
            tickers=tuple(config["deployment_tickers"]),
            timeframe_minutes=int(config["timeframe_minutes"]),
            start=temporal["validation_start"],
            end=temporal["validation_end"],
        )
        challenge = ChallengeSpec(**config["challenge"])
        seed = int(config["training"]["seed"])
        train_environment = HistoricalChallengeEnv(
            train_markets,
            tick_values=config["point_values"],
            round_trip_fees=config["round_trip_fees"],
            spec=challenge,
            seed=seed,
        )
        validation_environment = HistoricalChallengeEnv(
            validation_markets,
            tick_values={key: config["point_values"][key] for key in validation_markets},
            round_trip_fees={
                key: config["round_trip_fees"][key] for key in validation_markets
            },
            spec=challenge,
            seed=seed + 1,
        )
        observation_dim = next(iter(train_markets.values())).embeddings.shape[1] + 12
        agent = RecurrentC51Agent(
            observation_dim,
            seed=seed,
            **dict(config["agent"]),
        )
        training_config = config["training"]
        replay = BalancedSequenceReplay(
            capacity_episodes=int(training_config["replay_capacity_episodes"]),
            sequence_length=int(training_config["sequence_length"]),
            seed=seed,
        )
        training = train_agent(
            agent,
            train_environment,
            episodes=int(training_config["episodes"]),
            replay=replay,
            warmup_episodes=int(training_config["warmup_episodes"]),
            updates_per_episode=int(training_config["updates_per_episode"]),
            batch_sequences=int(training_config["batch_sequences"]),
            recurrent_horizon=int(training_config["recurrent_horizon"]),
            episode_tickers=tuple(config["tickers"]),
            ticker_seed=seed,
        )
        validation = evaluate_agent(
            agent,
            validation_environment,
            episodes=int(training_config["validation_episodes"]),
            recurrent_horizon=int(training_config["recurrent_horizon"]),
        )
        output = _resolve(root, config["output"])
        output.mkdir(parents=True, exist_ok=True)
        config_bytes = Path(config["_path"]).read_bytes()
        frozen_contract = {
            "checkpoint_sha256": assets.checkpoint_sha256,
            "experiment_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "training_tickers": list(config["tickers"]),
            "deployment_tickers": list(config["deployment_tickers"]),
            "training_only_tickers": list(config["training_only_tickers"]),
            "temporal": dict(temporal),
            "challenge": dict(config["challenge"]),
            "point_values": dict(config["point_values"]),
            "round_trip_fees": dict(config["round_trip_fees"]),
            "sealed_start": temporal["sealed_start"],
        }
        archive = CandidateArchive(output / "archive")
        recipe = {
            key: value for key, value in config.items() if not key.startswith("_")
        }
        with tempfile.TemporaryDirectory(prefix=".trained-", dir=output) as temporary:
            temporary_model = Path(temporary) / "model.pt"
            agent.save(temporary_model, manifest=frozen_contract)
            candidate = archive.register_candidate(
                temporary_model,
                contract=frozen_contract,
                recipe=recipe,
                parent_candidate_ids=parent_candidate_ids,
                hypothesis=hypothesis,
            )

        def training_metrics(_candidate):
            metrics = {
                "pass_rate": training.passes / training.episodes,
                "blow_rate": training.blows / training.episodes,
                "mean_reward": training.mean_reward,
            }
            if math.isfinite(training.mean_loss):
                metrics["mean_loss"] = training.mean_loss
            return metrics

        def selection_metrics(_candidate):
            pass_rate = validation.passes / validation.episodes
            blow_rate = validation.blows / validation.episodes
            return {
                "pass_rate": pass_rate,
                "blow_rate": blow_rate,
                "pass_minus_blow": pass_rate - blow_rate,
                "mean_reward": validation.mean_reward,
            }

        cascade = EvaluatorCascade(
            archive,
            {
                "schema": "propevolve_initial_historical_evaluator_v2",
                "selection_period": [
                    temporal["validation_start"], temporal["validation_end"]
                ],
                "sealed_start": temporal["sealed_start"],
                "decision_rule": "selection pass rate must exceed blow rate",
            },
            (
                EvaluationStage("training", training_metrics),
                EvaluationStage(
                    "selection",
                    selection_metrics,
                    gates=(EvaluationGate("pass_minus_blow", ">", 0.0),),
                ),
            ),
        )
        return candidate, cascade.evaluate(candidate.candidate_id)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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
    episode_tickers: tuple[str, ...] | None = None,
    ticker_seed: int = 0,
) -> TrainingResult:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    ticker_schedule = _balanced_ticker_schedule(
        episode_tickers,
        episodes=episodes,
        seed=ticker_seed,
    )
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    rewards, losses = [], []
    for episode_index in range(episodes):
        if ticker_schedule is None:
            observation, reset_info = environment.reset()
        else:
            observation, reset_info = environment.reset(
                options={"ticker": ticker_schedule[episode_index]}
            )
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


def _balanced_ticker_schedule(
    tickers: tuple[str, ...] | None,
    *,
    episodes: int,
    seed: int,
) -> tuple[str, ...] | None:
    if tickers is None:
        return None
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("episode tickers must be nonempty and unique")
    random = np.random.default_rng(seed)
    schedule = []
    while len(schedule) < episodes:
        cycle = list(tickers)
        random.shuffle(cycle)
        schedule.extend(cycle)
    return tuple(schedule[:episodes])


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
