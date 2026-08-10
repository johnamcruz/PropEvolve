"""Historical training and temporal evaluation for the PropEvolve POC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
from typing import TYPE_CHECKING, Callable

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

if TYPE_CHECKING:
    from .agent import RecurrentC51Agent


@dataclass(frozen=True)
class OutcomeStatistics:
    outcome: str
    episodes: int
    trade_count: int
    win_count: int
    winning_r_sum: float
    terminal_pnl_sum: float
    reward_sum: float

    @property
    def mean_trade_count(self) -> float:
        return self.trade_count / self.episodes if self.episodes else 0.0

    @property
    def trade_win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count else 0.0

    @property
    def average_win_r(self) -> float:
        return self.winning_r_sum / self.win_count if self.win_count else 0.0

    @property
    def mean_terminal_pnl(self) -> float:
        return self.terminal_pnl_sum / self.episodes if self.episodes else 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.episodes if self.episodes else 0.0


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
    outcome_statistics: tuple[OutcomeStatistics, ...] = ()

    @property
    def trade_win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count else 0.0

    @property
    def average_win_r(self) -> float:
        return self.winning_r_sum / self.win_count if self.win_count else 0.0

    def outcome(self, name: str) -> OutcomeStatistics:
        for statistics in self.outcome_statistics:
            if statistics.outcome == name:
                return statistics
        raise KeyError(f"outcome statistics are unavailable for {name}")


def _outcome_metric_values(result: TrainingResult) -> dict[str, float]:
    metrics = {}
    for statistics in result.outcome_statistics:
        prefix = statistics.outcome
        metrics.update({
            f"{prefix}_mean_trade_count": statistics.mean_trade_count,
            f"{prefix}_trade_win_rate": statistics.trade_win_rate,
            f"{prefix}_average_win_r": statistics.average_win_r,
            f"{prefix}_mean_terminal_pnl": statistics.mean_terminal_pnl,
            f"{prefix}_mean_reward": statistics.mean_reward,
        })
    return metrics


@dataclass(frozen=True)
class TrainingProgress:
    """Cumulative state captured only after a complete training episode."""

    completed_episodes: int = 0
    environment_steps: int = 0
    passes: int = 0
    blows: int = 0
    timeouts: int = 0
    trade_count: int = 0
    win_count: int = 0
    winning_r_sum: float = 0.0
    worst_pnl: float = math.inf
    terminal_pnl_sum: float = 0.0
    terminal_pnl_count: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0

    def result(self) -> TrainingResult:
        if self.completed_episodes < 1 or self.terminal_pnl_count < 1:
            raise ValueError("training progress has no completed episodes")
        return TrainingResult(
            episodes=self.completed_episodes,
            environment_steps=self.environment_steps,
            passes=self.passes,
            blows=self.blows,
            timeouts=self.timeouts,
            trade_count=self.trade_count,
            win_count=self.win_count,
            winning_r_sum=self.winning_r_sum,
            worst_pnl=self.worst_pnl,
            mean_terminal_pnl=self.terminal_pnl_sum / self.terminal_pnl_count,
            mean_reward=self.reward_sum / self.reward_count,
            mean_loss=(
                self.loss_sum / self.loss_count
                if self.loss_count
                else float("nan")
            ),
        )


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
        teacher_config = config.get("teacher")
        teacher_targets = None
        if teacher_config is not None:
            from .expansion_teacher import CHANNELS, ExpansionTeacherTargets

            if tuple(teacher_config["channels"]) != CHANNELS:
                raise ValueError("Expansion teacher channel order drifted")
            teacher_targets = ExpansionTeacherTargets.load(
                _resolve(root, teacher_config["cache_root"]),
                train_markets,
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
        agent_settings = dict(config["agent"])
        if teacher_config is not None:
            agent_settings.update(
                teacher_channels=len(teacher_config["channels"]),
                teacher_loss_weight=float(teacher_config["loss_weight"]),
            )
        output = _resolve(root, config["output"])
        output.mkdir(parents=True, exist_ok=True)
        recovery_path = output / "training-recovery.pt"
        resume_identity = _training_resume_identity(config, cache_root, teacher_config)
        resume = None
        if recovery_path.is_file():
            loaded, manifest = RecurrentC51Agent.load(
                recovery_path, device=agent_settings["device"]
            )
            if manifest.get("resume_identity") != resume_identity:
                raise ValueError("training recovery identity drifted")
            resume = TrainingProgress(**manifest["progress"])
            train_environment.restore_rng_state(manifest["environment_rng_state"])
            agent = loaded
        else:
            agent = RecurrentC51Agent(
                observation_dim,
                seed=seed,
                **agent_settings,
            )
        training_config = config["training"]
        replay = BalancedSequenceReplay(
            capacity_episodes=int(training_config["replay_capacity_episodes"]),
            capacity_transitions=int(
                training_config["replay_capacity_transitions"]
            ),
            sequence_length=int(training_config["sequence_length"]),
            terminal_sequence_fraction=float(
                training_config["terminal_sequence_fraction"]
            ),
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
            resume=resume,
            checkpoint_every_episodes=int(
                training_config["checkpoint_every_episodes"]
            ),
            checkpoint_callback=lambda progress: _save_training_recovery(
                agent,
                recovery_path,
                resume_identity=resume_identity,
                progress=progress,
                environment_rng_state=train_environment.rng_state(),
            ),
            teacher_lookup=(
                teacher_targets.target if teacher_targets is not None else None
            ),
        )
        agent.discard_teacher()
        validation = evaluate_agent(
            agent,
            validation_environment,
            episodes=int(training_config["validation_episodes"]),
            recurrent_horizon=int(training_config["recurrent_horizon"]),
        )
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
            "teacher": (
                None
                if teacher_config is None
                else {
                    "kind": "expansion",
                    "training_only": True,
                    "cache_manifest_sha256": {
                        ticker: hashlib.sha256(
                            (
                                _resolve(root, teacher_config["cache_root"])
                                / ticker
                                / "manifest.json"
                            ).read_bytes()
                        ).hexdigest()
                        for ticker in sorted(config["tickers"])
                    },
                }
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
            metrics = {
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
            metrics.update(_outcome_metric_values(validation))
            return metrics

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


def _training_resume_identity(
    config: dict,
    cache_root: Path,
    teacher_config: dict | None,
) -> str:
    root = Path(config["_root"])
    digest = hashlib.sha256(Path(config["_path"]).read_bytes())
    for ticker in sorted(config["tickers"]):
        digest.update((cache_root / ticker / "manifest.json").read_bytes())
        if teacher_config is not None:
            digest.update(
                (
                    _resolve(root, teacher_config["cache_root"])
                    / ticker
                    / "manifest.json"
                ).read_bytes()
            )
    for module in (Path(__file__), Path(__file__).with_name("agent.py"), Path(__file__).with_name("replay.py"), Path(__file__).with_name("environment.py")):
        digest.update(module.read_bytes())
    return digest.hexdigest()


def _save_training_recovery(
    agent: RecurrentC51Agent,
    path: Path,
    *,
    resume_identity: str,
    progress: TrainingProgress,
    environment_rng_state: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    agent.save(
        temporary,
        manifest={
            "resume_identity": resume_identity,
            "progress": asdict(progress),
            "environment_rng_state": environment_rng_state,
            "replay_restored": False,
        },
    )
    os.replace(temporary, path)


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
    resume: TrainingProgress | None = None,
    checkpoint_every_episodes: int = 0,
    checkpoint_callback: Callable[[TrainingProgress], None] | None = None,
    teacher_lookup: Callable[[str, int], np.ndarray | None] | None = None,
) -> TrainingResult:
    if episodes < 1 or minimum_environment_steps < 1:
        raise ValueError("episode ceiling and minimum environment steps must be positive")
    ticker_schedule = _balanced_ticker_schedule(
        episode_tickers,
        episodes=episodes,
        seed=ticker_seed,
    )
    if checkpoint_every_episodes < 0:
        raise ValueError("checkpoint interval cannot be negative")
    if checkpoint_every_episodes and checkpoint_callback is None:
        raise ValueError("checkpoint callback is required when checkpointing is enabled")
    progress = resume or TrainingProgress()
    if progress.completed_episodes > episodes:
        raise ValueError("resume progress exceeds the episode ceiling")
    if progress.environment_steps > minimum_environment_steps:
        raise ValueError("resume progress exceeds the environment-step budget")
    for episode_index in range(progress.completed_episodes, episodes):
        if ticker_schedule is None:
            observation, reset_info = environment.reset()
        else:
            observation, reset_info = environment.reset(
                options={"ticker": ticker_schedule[episode_index]}
            )
        valid = tuple(reset_info["valid_actions"])
        decision_index = int(reset_info.get("start", 0))
        episode_ticker = str(reset_info.get("ticker", ""))
        hidden = None
        transitions = []
        total_reward = 0.0
        # Decay exploration against actual market interaction, not the
        # emergency episode ceiling. Early pass/blow episodes vary greatly in
        # length, so episode-index decay would not represent equal experience.
        step_progress = min(
            1.0, progress.environment_steps / minimum_environment_steps
        )
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
            teacher_target = (
                teacher_lookup(episode_ticker, decision_index)
                if teacher_lookup is not None
                else None
            )
            transitions.append(Transition(
                observation=observation,
                action=Action(action),
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                valid_actions=valid,
                next_valid_actions=next_valid,
                teacher_target=teacher_target,
            ))
            total_reward += reward
            observation, valid = next_observation, next_valid
            terminal_info = info
            decision_index = int(info.get("fill_index", decision_index + 1))
            step_index += 1
            episode_steps = step_index
            if terminated:
                break
        outcome = str(terminal_info["outcome"])
        if outcome not in {"pass", "blow", "timeout"}:
            raise ValueError(f"unknown terminal outcome: {outcome}")
        terminal_pnl = float(terminal_info.get("equity_pnl", 0.0))
        if len(transitions) >= replay.sequence_length:
            replay.add(Episode(
                episode_id=f"historical-{episode_index}-{time.time_ns()}",
                ticker=str(terminal_info["ticker"]),
                outcome=outcome,
                primary_side=str(terminal_info["primary_side"]),
                ended_at_ns=time.time_ns(),
                transitions=tuple(transitions),
            ))
        episode_losses = []
        if len(replay) >= warmup_episodes:
            for _ in range(updates_per_episode):
                episode_losses.append(
                    agent.train_batch(replay.sample(batch_sequences))
                )
        progress = TrainingProgress(
            completed_episodes=episode_index + 1,
            environment_steps=progress.environment_steps + episode_steps,
            passes=progress.passes + int(outcome == "pass"),
            blows=progress.blows + int(outcome == "blow"),
            timeouts=progress.timeouts + int(outcome == "timeout"),
            trade_count=(
                progress.trade_count + int(terminal_info.get("trade_count", 0))
            ),
            win_count=progress.win_count + int(terminal_info.get("win_count", 0)),
            winning_r_sum=(
                progress.winning_r_sum
                + float(terminal_info.get("winning_r_sum", 0.0))
            ),
            worst_pnl=min(progress.worst_pnl, terminal_pnl),
            terminal_pnl_sum=progress.terminal_pnl_sum + terminal_pnl,
            terminal_pnl_count=progress.terminal_pnl_count + 1,
            reward_sum=progress.reward_sum + total_reward,
            reward_count=progress.reward_count + 1,
            loss_sum=progress.loss_sum + sum(episode_losses),
            loss_count=progress.loss_count + len(episode_losses),
        )
        print(
            f"[train] episode={episode_index + 1}/{episodes} ticker={terminal_info['ticker']} "
            f"outcome={outcome} reward={total_reward:.4f} replay={len(replay)} "
            f"trades={int(terminal_info.get('trade_count', 0))} "
            f"WR={float(terminal_info.get('win_rate', 0.0)):.1%} "
            f"winR={float(terminal_info.get('avg_win_r', 0.0)):+.3f}R "
            f"steps={progress.environment_steps:,}/{minimum_environment_steps:,}",
            flush=True,
        )
        if (
            checkpoint_every_episodes
            and progress.completed_episodes % checkpoint_every_episodes == 0
        ):
            assert checkpoint_callback is not None
            checkpoint_callback(progress)
        if progress.environment_steps >= minimum_environment_steps:
            break
    if progress.environment_steps < minimum_environment_steps:
        raise RuntimeError(
            "episode safety ceiling reached before the minimum environment-step budget"
        )
    return progress.result()


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
    by_outcome = {
        outcome: {
            "episodes": 0,
            "trade_count": 0,
            "win_count": 0,
            "winning_r_sum": 0.0,
            "terminal_pnl_sum": 0.0,
            "reward_sum": 0.0,
        }
        for outcome in outcomes
    }
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
        outcome = str(info["outcome"])
        outcomes[outcome] += 1
        rewards.append(total)
        terminal_pnl = float(info.get("equity_pnl", 0.0))
        episode_trades = int(info.get("trade_count", 0))
        episode_wins = int(info.get("win_count", 0))
        episode_winning_r = float(info.get("winning_r_sum", 0.0))
        terminal_pnls.append(terminal_pnl)
        trade_count += episode_trades
        win_count += episode_wins
        winning_r_sum += episode_winning_r
        outcome_values = by_outcome[outcome]
        outcome_values["episodes"] += 1
        outcome_values["trade_count"] += episode_trades
        outcome_values["win_count"] += episode_wins
        outcome_values["winning_r_sum"] += episode_winning_r
        outcome_values["terminal_pnl_sum"] += terminal_pnl
        outcome_values["reward_sum"] += total
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
        outcome_statistics=tuple(
            OutcomeStatistics(outcome=outcome, **values)
            for outcome, values in by_outcome.items()
            if values["episodes"]
        ),
    )
