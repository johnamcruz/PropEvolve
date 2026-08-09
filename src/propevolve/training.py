"""Historical training and temporal evaluation for the PropEvolve POC."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from .environment import ChallengeSpec, HistoricalChallengeEnv, MarketSeries
from .evolution import (
    CandidateArchive,
    EvaluationGate,
    EvaluationStage,
    EvaluatorCascade,
)
from .replay import BalancedSequenceReplay, Episode, Transition
from .teachers import (
    TeacherSignalCache,
    build_directional_oof_teacher_cache,
    load_teacher_bundle,
)

if TYPE_CHECKING:
    from .agent import RecurrentC51Agent


@dataclass(frozen=True)
class TrainingResult:
    episodes: int
    environment_steps: int
    passes: int
    blows: int
    timeouts: int
    trade_count: int
    win_count: int
    winning_r_sum: float
    worst_pnl: float
    mean_terminal_pnl: float
    mean_reward: float
    mean_loss: float

    @property
    def trade_win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count else 0.0

    @property
    def average_win_r(self) -> float:
        return self.winning_r_sum / self.win_count if self.win_count else 0.0


def prop_safety_objective(
    result: TrainingResult,
    *,
    max_loss: float,
    profit_target: float,
) -> float:
    """Rank zero-blow candidates by pass rate, then cushion and progress."""
    if result.blows:
        overage = max(0.0, -result.worst_pnl - max_loss) / max_loss
        return -1.0 - result.blows / result.episodes - overage
    margin = max(0.0, min(1.0, (max_loss + result.worst_pnl) / max_loss))
    progress = max(0.0, result.mean_terminal_pnl / profit_target)
    return result.passes / result.episodes + 0.05 * margin + 0.02 * progress


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
        teacher_config = config.get("teachers") or {}
        teachers_enabled = bool(teacher_config.get("enabled", False))
        teacher_channels = tuple(str(value) for value in teacher_config.get("channels", ()))
        if teachers_enabled:
            _ensure_teacher_caches(
                root=root,
                embedding_cache_root=cache_root,
                teacher_config=teacher_config,
                sealed_start=temporal["sealed_start"],
            )
        train_markets = load_markets(
            asset_contract=assets,
            cache_root=cache_root,
            tickers=tuple(config["tickers"]),
            timeframe_minutes=int(config["timeframe_minutes"]),
            start=temporal["train_start"],
            end=temporal["train_end"],
            teacher_cache_root=(
                _resolve(root, teacher_config["cache_root"])
                if teachers_enabled else None
            ),
            teacher_channels=teacher_channels,
            required_teacher_tickers=tuple(
                str(value) for value in teacher_config.get("required_tickers", ())
            ),
        )
        validation_markets = load_markets(
            asset_contract=assets,
            cache_root=cache_root,
            tickers=tuple(config["deployment_tickers"]),
            timeframe_minutes=int(config["timeframe_minutes"]),
            start=temporal["validation_start"],
            end=temporal["validation_end"],
        )
        assert_temporal_role(
            train_markets,
            role="training",
            start=temporal["train_start"],
            end=temporal["train_end"],
            sealed_start=temporal["sealed_start"],
        )
        assert_temporal_role(
            validation_markets,
            role="selection",
            start=temporal["validation_start"],
            end=temporal["validation_end"],
            sealed_start=temporal["sealed_start"],
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
            temporary_teacher_weights=(
                tuple(
                    float(teacher_config["loss_weights"][channel])
                    for channel in teacher_channels
                )
                if teachers_enabled else None
            ),
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
            minimum_environment_steps=int(
                training_config["minimum_environment_steps"]
            ),
            replay=replay,
            warmup_episodes=int(training_config["warmup_episodes"]),
            updates_per_episode=int(training_config["updates_per_episode"]),
            batch_sequences=int(training_config["batch_sequences"]),
            recurrent_horizon=int(training_config["recurrent_horizon"]),
            epsilon_start=float(training_config["epsilon_start"]),
            epsilon_end=float(training_config["epsilon_end"]),
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
            "embedding_cache_manifest_sha256": {
                ticker: hashlib.sha256(
                    (cache_root / ticker / "manifest.json").read_bytes()
                ).hexdigest()
                for ticker in sorted(set(config["tickers"]))
            },
            "experiment_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "training_tickers": list(config["tickers"]),
            "deployment_tickers": list(config["deployment_tickers"]),
            "training_only_tickers": list(config["training_only_tickers"]),
            "temporal": dict(temporal),
            "challenge": dict(config["challenge"]),
            "point_values": dict(config["point_values"]),
            "round_trip_fees": dict(config["round_trip_fees"]),
            "sealed_start": temporal["sealed_start"],
            "sealed_holdout_touched": False,
            "temporary_teachers": (
                {
                    "enabled": True,
                    "channels": list(teacher_channels),
                    "cache_manifest_sha256": {
                        ticker: hashlib.sha256(
                            (
                                _resolve(root, teacher_config["cache_root"])
                                / ticker
                                / "manifest.json"
                            ).read_bytes()
                        ).hexdigest()
                        for ticker in teacher_config.get("required_tickers", ())
                    },
                    "teacher_inputs_at_inference": False,
                    "temporary_heads_saved": False,
                }
                if teachers_enabled else {"enabled": False}
            ),
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
                "environment_steps": float(training.environment_steps),
                "trade_win_rate": training.trade_win_rate,
                "average_win_r": training.average_win_r,
                "worst_pnl": training.worst_pnl,
                "mean_terminal_pnl": training.mean_terminal_pnl,
                "safety_objective": prop_safety_objective(
                    training,
                    max_loss=challenge.max_loss,
                    profit_target=challenge.profit_target,
                ),
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
                "environment_steps": float(validation.environment_steps),
                "trade_win_rate": validation.trade_win_rate,
                "average_win_r": validation.average_win_r,
                "worst_pnl": validation.worst_pnl,
                "mean_terminal_pnl": validation.mean_terminal_pnl,
                "safety_objective": prop_safety_objective(
                    validation,
                    max_loss=challenge.max_loss,
                    profit_target=challenge.profit_target,
                ),
            }

        cascade = EvaluatorCascade(
            archive,
            {
                "schema": "propevolve_initial_historical_evaluator_v2",
                "selection_period": [
                    temporal["validation_start"], temporal["validation_end"]
                ],
                "sealed_start": temporal["sealed_start"],
                "sealed_holdout_touched": False,
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


def _ensure_teacher_caches(
    *,
    root: Path,
    embedding_cache_root: Path,
    teacher_config: dict,
    sealed_start: str,
) -> None:
    bundle = load_teacher_bundle(_resolve(root, teacher_config["bundle"]))
    cache_root = _resolve(root, teacher_config["cache_root"])
    required = tuple(str(value) for value in teacher_config["required_tickers"])
    if required != ("NQ",):
        raise ValueError(
            "the bundled paired Pivot OOF source currently supports exactly NQ"
        )
    destination = cache_root / "NQ"
    if destination.is_dir():
        TeacherSignalCache.load(
            destination,
            ticker="NQ",
            channels=teacher_config["channels"],
        )
        return
    bundle_root = Path(bundle["_root"])
    build_directional_oof_teacher_cache(
        embedding_cache_root=embedding_cache_root / "NQ",
        pivot_oof=bundle_root / bundle["pivot"]["oof_scores"],
        expansion_oof=bundle_root / bundle["expansion"]["oof_scores"],
        destination=destination,
        ticker="NQ",
        channels=teacher_config["channels"],
        research_end_exclusive=sealed_start,
    )


def load_markets(
    *,
    asset_contract: AssetContract,
    cache_root: str | Path,
    tickers: tuple[str, ...],
    timeframe_minutes: int,
    start: str | None,
    end: str | None,
    teacher_cache_root: str | Path | None = None,
    teacher_channels: tuple[str, ...] = (),
    required_teacher_tickers: tuple[str, ...] = (),
) -> dict[str, MarketSeries]:
    asset_contract.verify()
    root = Path(cache_root)
    markets = {}
    for ticker in tickers:
        market = load_market_series(
            Path(asset_contract.market_data) / f"{ticker}_{timeframe_minutes}min.csv",
            root / ticker,
            ticker=ticker,
            start=start,
            end=end,
        )
        if teacher_cache_root is not None:
            if ticker in required_teacher_tickers:
                teacher_path = Path(teacher_cache_root) / ticker
                if not teacher_path.is_dir():
                    raise ValueError(
                        f"required temporary teacher cache is missing for {ticker}"
                    )
                teacher = TeacherSignalCache.load(
                    teacher_path,
                    ticker=ticker,
                    channels=teacher_channels,
                )
                teacher_times = np.asarray(teacher.timestamps)
                rows = np.searchsorted(teacher_times, market.timestamps)
                if (
                    (rows >= len(teacher_times)).any()
                    or not np.array_equal(teacher_times[rows], market.timestamps)
                ):
                    raise ValueError(
                        f"temporary teacher timestamps do not exactly align for {ticker}"
                    )
                targets = np.asarray(teacher.probabilities[rows], dtype=np.float32)
                mask = np.asarray(teacher.availability[rows], dtype=np.bool_)
            else:
                targets = np.zeros((len(market.timestamps), len(teacher_channels)), np.float32)
                mask = np.zeros_like(targets, dtype=np.bool_)
            market = replace(
                market,
                teacher_targets=targets,
                teacher_mask=mask,
            )
        markets[ticker] = market
    return markets


def assert_temporal_role(
    markets: dict[str, MarketSeries],
    *,
    role: str,
    start: str,
    end: str,
    sealed_start: str,
) -> None:
    """Fail closed unless every causal decision belongs to its declared period."""
    lower = np.datetime64(start)
    upper = np.datetime64(end)
    sealed = np.datetime64(sealed_start)
    if upper > sealed:
        raise ValueError(f"{role} period crosses the sealed holdout")
    for ticker, market in markets.items():
        timestamps = np.asarray(market.timestamps)
        if len(timestamps) < 2:
            raise ValueError(f"{role} market {ticker} is empty")
        if (timestamps < lower).any() or (timestamps >= upper).any():
            raise ValueError(f"{role} market {ticker} violates its temporal contract")
        if (timestamps >= sealed).any():
            raise ValueError(f"{role} market {ticker} touches the sealed holdout")


def train_agent(
    agent: RecurrentC51Agent,
    environment: HistoricalChallengeEnv,
    *,
    episodes: int,
    minimum_environment_steps: int,
    replay: BalancedSequenceReplay,
    warmup_episodes: int,
    updates_per_episode: int,
    batch_sequences: int,
    recurrent_horizon: int,
    epsilon_start: float,
    epsilon_end: float,
    episode_tickers: tuple[str, ...] | None,
    ticker_seed: int,
) -> TrainingResult:
    if episodes < 1 or minimum_environment_steps < 1:
        raise ValueError("episode ceiling and minimum environment steps must be positive")
    ticker_schedule = _balanced_ticker_schedule(
        episode_tickers,
        episodes=episodes,
        seed=ticker_seed,
    )
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    rewards, losses = [], []
    terminal_pnls = []
    trade_count = win_count = 0
    winning_r_sum = 0.0
    environment_steps = 0
    completed_episodes = 0
    for episode_index in range(episodes):
        if ticker_schedule is None:
            observation, reset_info = environment.reset()
        else:
            observation, reset_info = environment.reset(
                options={"ticker": ticker_schedule[episode_index]}
            )
        valid = tuple(reset_info["valid_actions"])
        teacher_targets = reset_info.get("teacher_targets")
        teacher_mask = reset_info.get("teacher_mask")
        hidden = None
        transitions = []
        total_reward = 0.0
        # Decay exploration against actual market interaction, not the
        # emergency episode ceiling. Early pass/blow episodes vary greatly in
        # length, so episode-index decay would not represent equal experience.
        step_progress = min(1.0, environment_steps / minimum_environment_steps)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * step_progress
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
                teacher_targets=(
                    None
                    if teacher_targets is None
                    else np.asarray(teacher_targets, dtype=np.float32)
                ),
                teacher_mask=(
                    None
                    if teacher_mask is None
                    else np.asarray(teacher_mask, dtype=np.bool_)
                ),
            ))
            total_reward += reward
            observation, valid = next_observation, next_valid
            teacher_targets = info.get("teacher_targets")
            teacher_mask = info.get("teacher_mask")
            terminal_info = info
            step_index += 1
            environment_steps += 1
            if terminated:
                break
        outcome = str(terminal_info["outcome"])
        outcomes[outcome] += 1
        rewards.append(total_reward)
        terminal_pnls.append(float(terminal_info.get("equity_pnl", 0.0)))
        trade_count += int(terminal_info.get("trade_count", 0))
        win_count += int(terminal_info.get("win_count", 0))
        winning_r_sum += float(terminal_info.get("winning_r_sum", 0.0))
        completed_episodes += 1
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
            f"outcome={outcome} reward={total_reward:.4f} replay={len(replay)} "
            f"trades={int(terminal_info.get('trade_count', 0))} "
            f"WR={float(terminal_info.get('win_rate', 0.0)):.1%} "
            f"winR={float(terminal_info.get('avg_win_r', 0.0)):+.3f}R "
            f"steps={environment_steps:,}/{minimum_environment_steps:,}",
            flush=True,
        )
        if environment_steps >= minimum_environment_steps:
            break
    if environment_steps < minimum_environment_steps:
        raise RuntimeError(
            "episode safety ceiling reached before the minimum environment-step budget"
        )
    return TrainingResult(
        episodes=completed_episodes,
        environment_steps=environment_steps,
        passes=outcomes["pass"],
        blows=outcomes["blow"],
        timeouts=outcomes["timeout"],
        trade_count=trade_count,
        win_count=win_count,
        winning_r_sum=winning_r_sum,
        worst_pnl=float(np.min(terminal_pnls)),
        mean_terminal_pnl=float(np.mean(terminal_pnls)),
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
    recurrent_horizon: int,
) -> TrainingResult:
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    rewards = []
    terminal_pnls = []
    trade_count = win_count = 0
    winning_r_sum = 0.0
    environment_steps = 0
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
            environment_steps += 1
            if terminated:
                break
        outcomes[str(info["outcome"])] += 1
        rewards.append(total)
        terminal_pnls.append(float(info.get("equity_pnl", 0.0)))
        trade_count += int(info.get("trade_count", 0))
        win_count += int(info.get("win_count", 0))
        winning_r_sum += float(info.get("winning_r_sum", 0.0))
    return TrainingResult(
        episodes=episodes,
        environment_steps=environment_steps,
        passes=outcomes["pass"],
        blows=outcomes["blow"],
        timeouts=outcomes["timeout"],
        trade_count=trade_count,
        win_count=win_count,
        winning_r_sum=winning_r_sum,
        worst_pnl=float(np.min(terminal_pnls)),
        mean_terminal_pnl=float(np.mean(terminal_pnls)),
        mean_reward=float(np.mean(rewards)),
        mean_loss=float("nan"),
    )
