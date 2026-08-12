"""Historical training and temporal evaluation for the PropEvolve POC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from .assets import AssetContract
from .cache import load_market_series
from .config import (
    DEFAULT_RUNTIME,
    agent_runtime_settings,
    configure_runtime_environment,
)
from .decision import Action
from .environment import ChallengeSpec, HistoricalChallengeEnv, MarketSeries
from .evolution import (
    CandidateArchive,
    EvaluationGate,
    EvaluationStage,
    EvaluatorCascade,
)
from .observation import TradeManagementObservationSpec
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
    mfe_sum: float = 0.0
    mae_sum: float = 0.0
    retention_eligible_count: int = 0
    retention_capture_sum: float = 0.0
    retention_gap_sum: float = 0.0
    retention_round_trip_count: int = 0
    two_r_eligible_count: int = 0
    two_r_capture_sum: float = 0.0
    two_r_round_trip_count: int = 0

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

    @property
    def average_mfe_r(self) -> float:
        return self.mfe_sum / self.trade_count if self.trade_count else 0.0

    @property
    def average_mae_r(self) -> float:
        return self.mae_sum / self.trade_count if self.trade_count else 0.0

    @property
    def mfe_capture_ratio(self) -> float:
        return (
            self.retention_capture_sum / self.retention_eligible_count
            if self.retention_eligible_count else 0.0
        )

    @property
    def gave_it_all_back_rate(self) -> float:
        return (
            self.retention_round_trip_count / self.retention_eligible_count
            if self.retention_eligible_count else 0.0
        )

    @property
    def two_r_mfe_capture_ratio(self) -> float:
        return (
            self.two_r_capture_sum / self.two_r_eligible_count
            if self.two_r_eligible_count else 0.0
        )


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
    trade_r_sum: float = 0.0
    outcome_statistics: tuple[OutcomeStatistics, ...] = ()
    mfe_sum: float = 0.0
    mae_sum: float = 0.0
    retention_eligible_count: int = 0
    retention_capture_sum: float = 0.0
    retention_gap_sum: float = 0.0
    retention_round_trip_count: int = 0
    two_r_eligible_count: int = 0
    two_r_capture_sum: float = 0.0
    two_r_round_trip_count: int = 0
    near_blow_timeout_count: int = 0
    short_circuited: bool = False
    short_circuit_reason: str | None = None

    @property
    def trade_win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count else 0.0

    @property
    def average_win_r(self) -> float:
        return self.winning_r_sum / self.win_count if self.win_count else 0.0

    @property
    def expectancy_r(self) -> float:
        return self.trade_r_sum / self.trade_count if self.trade_count else 0.0

    @property
    def average_mfe_r(self) -> float:
        return self.mfe_sum / self.trade_count if self.trade_count else 0.0

    @property
    def average_mae_r(self) -> float:
        return self.mae_sum / self.trade_count if self.trade_count else 0.0

    @property
    def mfe_capture_ratio(self) -> float:
        return (
            self.retention_capture_sum / self.retention_eligible_count
            if self.retention_eligible_count else 0.0
        )

    @property
    def mfe_realized_gap_r(self) -> float:
        return (
            self.retention_gap_sum / self.retention_eligible_count
            if self.retention_eligible_count else 0.0
        )

    @property
    def gave_it_all_back_rate(self) -> float:
        return (
            self.retention_round_trip_count / self.retention_eligible_count
            if self.retention_eligible_count else 0.0
        )

    @property
    def two_r_mfe_capture_ratio(self) -> float:
        return (
            self.two_r_capture_sum / self.two_r_eligible_count
            if self.two_r_eligible_count else 0.0
        )

    @property
    def two_r_gave_it_all_back_rate(self) -> float:
        return (
            self.two_r_round_trip_count / self.two_r_eligible_count
            if self.two_r_eligible_count else 0.0
        )

    @property
    def near_blow_timeout_rate(self) -> float:
        return (
            self.near_blow_timeout_count / self.timeouts
            if self.timeouts else 0.0
        )

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
            f"{prefix}_average_mfe_r": statistics.average_mfe_r,
            f"{prefix}_average_mae_r": statistics.average_mae_r,
            f"{prefix}_mfe_capture_ratio": statistics.mfe_capture_ratio,
            f"{prefix}_gave_it_all_back_rate": statistics.gave_it_all_back_rate,
            f"{prefix}_two_r_mfe_capture_ratio": statistics.two_r_mfe_capture_ratio,
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
    trade_r_sum: float = 0.0
    worst_pnl: float = math.inf
    terminal_pnl_sum: float = 0.0
    terminal_pnl_count: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0
    mfe_sum: float = 0.0
    mae_sum: float = 0.0
    retention_eligible_count: int = 0
    retention_capture_sum: float = 0.0
    retention_gap_sum: float = 0.0
    retention_round_trip_count: int = 0
    two_r_eligible_count: int = 0
    two_r_capture_sum: float = 0.0
    two_r_round_trip_count: int = 0
    near_blow_timeout_count: int = 0
    recent_outcomes: tuple[str, ...] = ()
    recent_average_hold_bars: tuple[float, ...] = ()
    recent_voluntary_close_rates: tuple[float, ...] = ()
    short_circuit_reason: str | None = None

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
            trade_r_sum=self.trade_r_sum,
            worst_pnl=self.worst_pnl,
            mean_terminal_pnl=self.terminal_pnl_sum / self.terminal_pnl_count,
            mean_reward=self.reward_sum / self.reward_count,
            mean_loss=(
                self.loss_sum / self.loss_count
                if self.loss_count
                else float("nan")
            ),
            mfe_sum=self.mfe_sum,
            mae_sum=self.mae_sum,
            retention_eligible_count=self.retention_eligible_count,
            retention_capture_sum=self.retention_capture_sum,
            retention_gap_sum=self.retention_gap_sum,
            retention_round_trip_count=self.retention_round_trip_count,
            two_r_eligible_count=self.two_r_eligible_count,
            two_r_capture_sum=self.two_r_capture_sum,
            two_r_round_trip_count=self.two_r_round_trip_count,
            near_blow_timeout_count=self.near_blow_timeout_count,
            short_circuited=self.short_circuit_reason is not None,
            short_circuit_reason=self.short_circuit_reason,
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
    return (
        result.passes / result.episodes
        + 0.05 * margin
        + 0.02 * progress
        - 0.5 * result.near_blow_timeout_rate
    )


class HistoricalCandidateRunner:
    """Train one immutable historical challenger and evaluate its first gate."""

    def run(
        self,
        config: dict,
        *,
        parent_candidate_ids: tuple[str, ...],
        hypothesis: str,
    ):
        configure_runtime_environment(config.get("runtime", DEFAULT_RUNTIME))
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
        teacher_specs = tuple(config.get("teachers", ()))
        teacher_targets = None
        if teacher_specs:
            from .teachers import load_teacher_targets

            teacher_targets = load_teacher_targets(
                teacher_specs,
                root=root,
                markets=train_markets,
            )
        challenge = ChallengeSpec(**config["challenge"])
        observation_spec = TradeManagementObservationSpec.from_config(
            config.get("observation")
        )
        near_blow_loss_threshold = (
            float(
                config.get("campaign", {}).get(
                    "near_blow_loss_fraction", 0.75
                )
            )
            * challenge.max_loss
        )
        seed = int(config["training"]["seed"])
        train_environment = HistoricalChallengeEnv(
            train_markets,
            tick_values=config["point_values"],
            round_trip_fees=config["round_trip_fees"],
            spec=challenge,
            observation_spec=observation_spec,
            seed=seed,
        )
        validation_environment = HistoricalChallengeEnv(
            validation_markets,
            tick_values={key: config["point_values"][key] for key in validation_markets},
            round_trip_fees={
                key: config["round_trip_fees"][key] for key in validation_markets
            },
            spec=challenge,
            observation_spec=observation_spec,
            seed=seed + 1,
        )
        observation_dim = train_environment.observation_dim
        if validation_environment.observation_dim != observation_dim:
            raise ValueError("training and selection observation widths differ")
        agent_settings = dict(config["agent"])
        agent_settings.update(
            agent_runtime_settings(config.get("runtime", DEFAULT_RUNTIME))
        )
        if teacher_targets is not None:
            agent_settings.update(
                teacher_channels=len(teacher_targets.channels),
                teacher_loss_weight=sum(
                    float(spec["loss_weight"]) for spec in teacher_specs
                ),
                teacher_channel_loss_weights=teacher_targets.channel_loss_weights,
                teacher_entry_search_loss_weight=(
                    teacher_targets.entry_search_loss_weight
                ),
            )
        output = _resolve(root, config["output"])
        output.mkdir(parents=True, exist_ok=True)
        recovery_path = output / "training-recovery.pt"
        retained_policy_path = output / "retained-pass-policy.pt"
        diagnostics_path = output / "training-diagnostics.jsonl"
        resume_identity = _training_resume_identity(config, cache_root, teacher_specs)
        resume = None
        replay_state = None
        if recovery_path.is_file():
            loaded, manifest = RecurrentC51Agent.load(
                recovery_path, device=agent_settings["device"]
            )
            if manifest.get("resume_identity") != resume_identity:
                raise ValueError("training recovery identity drifted")
            resume = TrainingProgress(**manifest["progress"])
            train_environment.restore_rng_state(manifest["environment_rng_state"])
            replay_state = manifest.get("replay_state")
            if not manifest.get("replay_restored", False) or replay_state is None:
                raise ValueError("training recovery is missing replay state")
            agent = loaded
        else:
            if diagnostics_path.exists():
                raise ValueError("training diagnostics exist without resumable recovery")
            warm_start = config.get("_warm_start_model")
            if warm_start is None:
                agent = RecurrentC51Agent(
                    observation_dim,
                    seed=seed,
                    **agent_settings,
                )
            else:
                warm_path = Path(str(warm_start["model_path"])).resolve(strict=True)
                if _path_sha256(warm_path) != str(warm_start["model_sha256"]):
                    raise ValueError("warm-start model identity drifted")
                agent, parent_contract = RecurrentC51Agent.warm_start(
                    warm_path,
                    config={
                        "observation_dim": observation_dim,
                        "seed": seed,
                        **agent_settings,
                    },
                )
                expected_parent = {
                    "checkpoint_sha256": assets.checkpoint_sha256,
                    "training_tickers": list(config["tickers"]),
                    "deployment_tickers": list(config["deployment_tickers"]),
                    "training_only_tickers": list(config["training_only_tickers"]),
                    "temporal": dict(temporal),
                }
                if any(
                    parent_contract.get(field) != expected
                    for field, expected in expected_parent.items()
                ):
                    raise ValueError("warm-start causal contract drifted")
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
            safety_sequence_fraction=float(
                training_config.get("safety_sequence_fraction", 0.0)
            ),
            entry_opportunity_sequence_fraction=float(
                training_config.get("entry_opportunity_sequence_fraction", 0.0)
            ),
            seed=seed,
        )
        if replay_state is not None:
            replay.load_state_dict(replay_state)
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
            prefetch_batches=int(training_config.get("prefetch_batches", 0)),
            recurrent_horizon=int(training_config["recurrent_horizon"]),
            epsilon_start=float(training_config["epsilon_start"]),
            epsilon_end=float(training_config["epsilon_end"]),
            management_epsilon_start=float(
                training_config.get(
                    "management_epsilon_start", training_config["epsilon_start"]
                )
            ),
            management_epsilon_end=float(
                training_config.get(
                    "management_epsilon_end", training_config["epsilon_end"]
                )
            ),
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
                replay_state=replay.state_dict(),
            ),
            retention_checkpoint_callback=lambda evidence: _save_retained_policy(
                agent,
                retained_policy_path,
                resume_identity=resume_identity,
                evidence=evidence,
            ),
            teacher_lookup=(
                teacher_targets.target if teacher_targets is not None else None
            ),
            teacher_channels=(
                teacher_targets.channels if teacher_targets is not None else None
            ),
            teacher_loss_end_scale=float(
                training_config.get("teacher_loss_end_scale", 1.0)
            ),
            teacher_guidance_dropout_start=float(
                training_config.get("teacher_guidance_dropout_start", 0.0)
            ),
            teacher_guidance_dropout_end=float(
                training_config.get("teacher_guidance_dropout_end", 0.0)
            ),
            short_circuit_minimum_environment_steps=(
                int(training_config["short_circuit"]["minimum_environment_steps"])
                if training_config.get("short_circuit") is not None
                else None
            ),
            short_circuit_minimum_passes=(
                int(training_config["short_circuit"]["minimum_passes"])
                if training_config.get("short_circuit") is not None
                else 0
            ),
            short_circuit_maximum_blow_rate=(
                float(training_config["short_circuit"]["maximum_blow_rate"])
                if training_config.get("short_circuit") is not None
                else 1.0
            ),
            collapse_window_episodes=int(
                training_config.get("short_circuit", {})
                .get("collapse", {})
                .get("window_episodes", 0)
            ),
            collapse_minimum_prior_passes=int(
                training_config.get("short_circuit", {})
                .get("collapse", {})
                .get("minimum_prior_passes", 0)
            ),
            collapse_maximum_recent_passes=int(
                training_config.get("short_circuit", {})
                .get("collapse", {})
                .get("maximum_recent_passes", 0)
            ),
            collapse_maximum_average_hold_bars=float(
                training_config.get("short_circuit", {})
                .get("collapse", {})
                .get("maximum_average_hold_bars", math.inf)
            ),
            collapse_minimum_voluntary_close_rate=float(
                training_config.get("short_circuit", {})
                .get("collapse", {})
                .get("minimum_voluntary_close_rate", 1.0)
            ),
            episode_diagnostic_callback=lambda payload: _append_jsonl(
                diagnostics_path, payload
            ),
            near_blow_loss_threshold=near_blow_loss_threshold,
        )
        _write_training_diagnostic_summary(
            diagnostics_path,
            output / "training-diagnostic-summary.json",
        )
        retained_policy_restored = False
        if training.short_circuited and retained_policy_path.is_file():
            retained_agent, retention_manifest = RecurrentC51Agent.load(
                retained_policy_path,
                device=agent_settings["device"],
            )
            if retention_manifest.get("resume_identity") != resume_identity:
                raise ValueError("retained pass policy identity drifted")
            agent = retained_agent
            retained_policy_restored = True
        agent.discard_retention_anchor()
        agent.discard_teacher()
        validation = None
        if not training.short_circuited:
            validation = evaluate_agent(
                agent,
                validation_environment,
                episodes=int(training_config["validation_episodes"]),
                recurrent_horizon=int(training_config["recurrent_horizon"]),
                near_blow_loss_threshold=near_blow_loss_threshold,
                stop_on_first_blow=bool(
                    config.get("_validation_stop_on_blow", False)
                ),
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
            "warm_start_parent": (
                None
                if config.get("_warm_start_model") is None
                else {
                    "candidate_id": config["_warm_start_model"]["candidate_id"],
                    "model_sha256": config["_warm_start_model"]["model_sha256"],
                }
            ),
            "teachers": [
                {
                    "kind": spec["kind"],
                    "training_only": True,
                    "cache_manifest_sha256": {
                        ticker: hashlib.sha256(
                            (
                                _resolve(root, spec["cache_root"])
                                / ticker
                                / "manifest.json"
                            ).read_bytes()
                        ).hexdigest()
                        for ticker in sorted(config["tickers"])
                    },
                }
                for spec in teacher_specs
            ],
            "retained_pass_policy_restored": retained_policy_restored,
        }
        archive_output = _resolve(
            root, str(config.get("_archive_output", config["output"]))
        )
        archive = CandidateArchive(archive_output / "archive")
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
                "expectancy_r": training.expectancy_r,
                "worst_pnl": training.worst_pnl,
                "mean_terminal_pnl": training.mean_terminal_pnl,
                "safety_objective": prop_safety_objective(
                    training,
                    max_loss=challenge.max_loss,
                    profit_target=challenge.profit_target,
                ),
                "average_mfe_r": training.average_mfe_r,
                "average_mae_r": training.average_mae_r,
                "mfe_capture_ratio": training.mfe_capture_ratio,
                "mfe_realized_gap_r": training.mfe_realized_gap_r,
                "gave_it_all_back_rate": training.gave_it_all_back_rate,
                "two_r_mfe_capture_ratio": training.two_r_mfe_capture_ratio,
                "two_r_gave_it_all_back_rate": (
                    training.two_r_gave_it_all_back_rate
                ),
                "near_blow_timeout_count": float(
                    training.near_blow_timeout_count
                ),
                "near_blow_timeout_rate": training.near_blow_timeout_rate,
                "short_circuited": float(training.short_circuited),
                "retained_pass_policy_restored": float(
                    retained_policy_restored
                ),
            }
            if math.isfinite(training.mean_loss):
                metrics["mean_loss"] = training.mean_loss
            return metrics

        def selection_metrics(_candidate):
            assert validation is not None
            pass_rate = validation.passes / validation.episodes
            blow_rate = validation.blows / validation.episodes
            requested_validation_episodes = int(
                training_config["validation_episodes"]
            )
            metrics = {
                "pass_rate": pass_rate,
                "blow_rate": blow_rate,
                "pass_minus_blow": pass_rate - blow_rate,
                "evaluated_episodes": float(validation.episodes),
                "requested_episodes": float(requested_validation_episodes),
                "short_circuited": float(
                    validation.episodes < requested_validation_episodes
                ),
                "mean_reward": validation.mean_reward,
                "environment_steps": float(validation.environment_steps),
                "trade_win_rate": validation.trade_win_rate,
                "average_win_r": validation.average_win_r,
                "expectancy_r": validation.expectancy_r,
                "worst_pnl": validation.worst_pnl,
                "mean_terminal_pnl": validation.mean_terminal_pnl,
                "safety_objective": prop_safety_objective(
                    validation,
                    max_loss=challenge.max_loss,
                    profit_target=challenge.profit_target,
                ),
                "average_mfe_r": validation.average_mfe_r,
                "average_mae_r": validation.average_mae_r,
                "mfe_capture_ratio": validation.mfe_capture_ratio,
                "mfe_realized_gap_r": validation.mfe_realized_gap_r,
                "gave_it_all_back_rate": validation.gave_it_all_back_rate,
                "two_r_mfe_capture_ratio": validation.two_r_mfe_capture_ratio,
                "two_r_gave_it_all_back_rate": (
                    validation.two_r_gave_it_all_back_rate
                ),
                "near_blow_timeout_count": float(
                    validation.near_blow_timeout_count
                ),
                "near_blow_timeout_rate": validation.near_blow_timeout_rate,
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
                EvaluationStage(
                    "training",
                    training_metrics,
                    gates=(EvaluationGate("short_circuited", "==", 0.0),),
                ),
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
    teacher_specs: tuple[dict, ...],
) -> str:
    root = Path(config["_root"])
    digest = hashlib.sha256(json.dumps(
        {
            key: value
            for key, value in config.items()
            if not str(key).startswith("_")
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode())
    if config.get("_warm_start_model") is not None:
        digest.update(json.dumps(
            {
                "candidate_id": config["_warm_start_model"]["candidate_id"],
                "model_sha256": config["_warm_start_model"]["model_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode())
    for ticker in sorted(config["tickers"]):
        digest.update((cache_root / ticker / "manifest.json").read_bytes())
        for teacher_spec in teacher_specs:
            digest.update(
                (
                    _resolve(root, teacher_spec["cache_root"])
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
    replay_state: dict[str, object],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    agent.save(
        temporary,
        manifest={
            "resume_identity": resume_identity,
            "progress": asdict(progress),
            "environment_rng_state": environment_rng_state,
            "replay_state": replay_state,
            "replay_restored": True,
        },
    )
    os.replace(temporary, path)


def _save_retained_policy(
    agent: RecurrentC51Agent,
    path: Path,
    *,
    resume_identity: str,
    evidence: dict[str, object],
) -> None:
    """Preserve every pass policy immutably and atomically advance latest alias."""
    episode = int(evidence.get("episode", 0))
    ticker = str(evidence.get("ticker", ""))
    if episode < 1 or not ticker.isalnum():
        raise ValueError("retained pass evidence identity is invalid")
    archive = path.parent / "retained-pass-policies"
    archive.mkdir(parents=True, exist_ok=True)
    retained = archive / f"episode-{episode:06d}-{ticker}.pt"
    if retained.exists():
        raise ValueError("retained pass checkpoint already exists")
    temporary = retained.with_suffix(retained.suffix + ".tmp")
    agent.save(
        temporary,
        manifest={
            "resume_identity": resume_identity,
            "retention_evidence": dict(evidence),
        },
    )
    os.replace(temporary, retained)
    alias_temporary = path.with_suffix(path.suffix + ".tmp")
    shutil.copyfile(retained, alias_temporary)
    os.replace(alias_temporary, path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _diagnostic_aggregate(rows: list[dict]) -> dict[str, object]:
    episodes = len(rows)
    trades = sum(int(row.get("trade_count", 0)) for row in rows)
    wins = sum(
        float(row.get("trade_count", 0)) * float(row.get("win_rate", 0.0))
        for row in rows
    )
    activated = sum(
        float(row.get("trade_count", 0))
        * float(row.get("ratchet_activation_rate", 0.0))
        for row in rows
    )

    def weighted(field: str, weights: list[float]) -> float:
        total = sum(weights)
        return (
            sum(float(row.get(field, 0.0) or 0.0) * weight for row, weight in zip(rows, weights))
            / total
            if total else 0.0
        )

    trade_weights = [float(row.get("trade_count", 0)) for row in rows]
    win_weights = [
        float(row.get("trade_count", 0)) * float(row.get("win_rate", 0.0))
        for row in rows
    ]
    loss_weights = [
        float(row.get("trade_count", 0)) * (1.0 - float(row.get("win_rate", 0.0)))
        for row in rows
    ]
    activation_weights = [
        float(row.get("trade_count", 0))
        * float(row.get("ratchet_activation_rate", 0.0))
        for row in rows
    ]
    update_weights = [float(row.get("updates", 0)) for row in rows]
    teacher_weights = [float(row.get("teacher_scored_entries", 0)) for row in rows]
    teacher_channel_names = sorted({
        str(channel)
        for row in rows
        for channel in (row.get("selected_teacher_channel_means") or {})
    })
    result: dict[str, object] = {
        "episodes": episodes,
        "passes": sum(row.get("outcome") == "pass" for row in rows),
        "blows": sum(row.get("outcome") == "blow" for row in rows),
        "timeouts": sum(row.get("outcome") == "timeout" for row in rows),
        "near_blow_timeout_count": sum(
            bool(row.get("near_blow_timeout", False)) for row in rows
        ),
        "pass_rate": (
            sum(row.get("outcome") == "pass" for row in rows) / episodes
            if episodes else 0.0
        ),
        "blow_rate": (
            sum(row.get("outcome") == "blow" for row in rows) / episodes
            if episodes else 0.0
        ),
        "trades": trades,
        "trade_win_rate": wins / trades if trades else 0.0,
        "average_win_r": weighted("avg_win_r", win_weights),
        "average_loss_r": weighted("avg_loss_r", loss_weights),
        "expectancy_r": weighted("expectancy_r", trade_weights),
        "average_mfe_r": weighted("avg_mfe_r", trade_weights),
        "average_mae_r": weighted("avg_mae_r", trade_weights),
        "retention_eligible_count": int(sum(
            float(row.get("retention_eligible_count", 0)) for row in rows
        )),
        "mfe_capture_ratio": weighted(
            "mfe_capture_ratio",
            [float(row.get("retention_eligible_count", 0)) for row in rows],
        ),
        "mfe_realized_gap_r": weighted(
            "mfe_realized_gap_r",
            [float(row.get("retention_eligible_count", 0)) for row in rows],
        ),
        "gave_it_all_back_rate": weighted(
            "gave_it_all_back_rate",
            [float(row.get("retention_eligible_count", 0)) for row in rows],
        ),
        "two_r_eligible_count": int(sum(
            float(row.get("two_r_eligible_count", 0)) for row in rows
        )),
        "two_r_mfe_capture_ratio": weighted(
            "two_r_mfe_capture_ratio",
            [float(row.get("two_r_eligible_count", 0)) for row in rows],
        ),
        "two_r_gave_it_all_back_rate": weighted(
            "two_r_gave_it_all_back_rate",
            [float(row.get("two_r_eligible_count", 0)) for row in rows],
        ),
        "ratchet_activation_rate": activated / trades if trades else 0.0,
        "activated_average_realized_r": weighted(
            "activated_avg_realized_r", activation_weights
        ),
        "average_hold_bars": weighted("avg_hold_bars", trade_weights),
        "voluntary_close_count": sum(
            int(row.get("voluntary_close_count", 0)) for row in rows
        ),
        "initial_stop_count": sum(
            int(row.get("initial_stop_count", 0)) for row in rows
        ),
        "ratchet_stop_count": sum(
            int(row.get("ratchet_stop_count", 0)) for row in rows
        ),
        "terminal_liquidation_count": sum(
            int(row.get("terminal_liquidation_count", 0)) for row in rows
        ),
        "environment_steps": max(
            (int(row.get("environment_steps", 0)) for row in rows), default=0
        ),
        "latest_entry_epsilon": (
            float(rows[-1].get("entry_epsilon", 0.0)) if rows else 0.0
        ),
        "latest_management_epsilon": (
            float(rows[-1].get("management_epsilon", 0.0)) if rows else 0.0
        ),
        "mean_training_loss": weighted("mean_training_loss", update_weights),
        "mean_rl_loss": weighted("mean_rl_loss", update_weights),
        "mean_teacher_loss": weighted("mean_teacher_loss", update_weights),
        "mean_gradient_norm": weighted("mean_gradient_norm", update_weights),
        "sampled_management_row_fraction": weighted(
            "mean_sampled_management_row_fraction", update_weights
        ),
        "sampled_hold_reward": weighted(
            "mean_sampled_hold_reward", update_weights
        ),
        "sampled_close_reward": weighted(
            "mean_sampled_close_reward", update_weights
        ),
        "sampled_hold_n_step_return": weighted(
            "mean_sampled_hold_n_step_return", update_weights
        ),
        "sampled_close_n_step_return": weighted(
            "mean_sampled_close_n_step_return", update_weights
        ),
        "sampled_hold_td_loss": weighted(
            "mean_sampled_hold_td_loss", update_weights
        ),
        "sampled_close_td_loss": weighted(
            "mean_sampled_close_td_loss", update_weights
        ),
        "management_hold_minus_close_q": weighted(
            "mean_management_hold_minus_close_q", update_weights
        ),
        "sampled_management_close_fraction": weighted(
            "mean_sampled_management_close_fraction", update_weights
        ),
        "sampled_recurrent_reset_fraction": weighted(
            "mean_sampled_recurrent_reset_fraction", update_weights
        ),
        "sampled_burn_in_reset_coverage": weighted(
            "mean_sampled_burn_in_reset_coverage", update_weights
        ),
        "sampled_recurrent_reset_pattern_count": weighted(
            "mean_sampled_recurrent_reset_pattern_count", update_weights
        ),
        "policy_retention_loss": weighted(
            "mean_policy_retention_loss", update_weights
        ),
        "teacher_scored_entries": int(sum(teacher_weights)),
        "selected_side_attempt_probability_mean": weighted(
            "selected_side_attempt_probability_mean", teacher_weights
        ),
        "selected_side_clean_retained_probability_mean": weighted(
            "selected_side_clean_retained_probability_mean", teacher_weights
        ),
        "selected_teacher_channel_means": {
            channel: (
                sum(
                    float((row.get("selected_teacher_channel_means") or {}).get(
                        channel, 0.0
                    )) * weight
                    for row, weight in zip(rows, teacher_weights)
                ) / sum(
                    weight
                    for row, weight in zip(rows, teacher_weights)
                    if channel in (row.get("selected_teacher_channel_means") or {})
                )
            )
            for channel in teacher_channel_names
            if sum(
                weight
                for row, weight in zip(rows, teacher_weights)
                if channel in (row.get("selected_teacher_channel_means") or {})
            ) > 0
        },
        "short_circuited": bool(
            rows and rows[-1].get("training_short_circuited", False)
        ),
        "short_circuit_reason": (
            rows[-1].get("training_short_circuit_reason") if rows else None
        ),
        "action_counts": {
            action.name: sum(
                int(row.get("action_counts", {}).get(action.name, 0)) for row in rows
            )
            for action in Action
        },
    }
    timeouts = int(result["timeouts"])
    result["near_blow_timeout_rate"] = (
        int(result["near_blow_timeout_count"]) / timeouts if timeouts else 0.0
    )
    for horizon in (5, 10, 20, 50):
        prefix = f"shadow_h{horizon}"
        horizon_weights = [
            float(row.get(f"{prefix}_complete_trades", 0)) for row in rows
        ]
        result[prefix] = {
            "complete_trades": int(sum(horizon_weights)),
            "average_mfe_r": weighted(f"{prefix}_avg_mfe_r", horizon_weights),
            "average_mae_r": weighted(f"{prefix}_avg_mae_r", horizon_weights),
            "hit_2r_before_1r_rate": weighted(
                f"{prefix}_2r_before_1r_rate", horizon_weights
            ),
            "hit_3r_before_1r_rate": weighted(
                f"{prefix}_3r_before_1r_rate", horizon_weights
            ),
        }
    return result


def _write_training_diagnostic_summary(source: Path, destination: Path) -> None:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    by_ticker = {
        ticker: _diagnostic_aggregate([
            row for row in rows if str(row.get("ticker")) == ticker
        ])
        for ticker in sorted({str(row.get("ticker")) for row in rows})
    }
    by_outcome = {
        outcome: _diagnostic_aggregate([
            row for row in rows if str(row.get("outcome")) == outcome
        ])
        for outcome in sorted({str(row.get("outcome")) for row in rows})
    }
    payload = {
        "schema": "propevolve_training_diagnostic_summary_v1",
        "source": source.name,
        "source_sha256": _path_sha256(source),
        "overall": _diagnostic_aggregate(rows),
        "recent_20": _diagnostic_aggregate(rows[-20:]),
        "by_ticker": by_ticker,
        "by_outcome": by_outcome,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


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
        print(
            f"[market-load] START ticker={ticker} period=[{start},{end})",
            flush=True,
        )
        markets[ticker] = load_market_series(
            Path(asset_contract.market_data) / f"{ticker}_{timeframe_minutes}min.csv",
            root / ticker,
            ticker=ticker,
            start=start,
            end=end,
        )
        print(
            f"[market-load] COMPLETE ticker={ticker} "
            f"rows={len(markets[ticker].timestamps):,}",
            flush=True,
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


def _teacher_guidance_is_visible(
    *,
    seed: int,
    episode_index: int,
    ticker: str,
    decision_index: int,
    dropout_probability: float,
) -> bool:
    """Return a resume-stable teacher mask without mutable RNG state."""
    if dropout_probability <= 0:
        return True
    if dropout_probability >= 1:
        return False
    identity = f"{seed}:{episode_index}:{ticker}:{decision_index}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
    return draw >= dropout_probability


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
    management_epsilon_start: float | None = None,
    management_epsilon_end: float | None = None,
    episode_tickers: tuple[str, ...] | None,
    ticker_seed: int,
    prefetch_batches: int = 0,
    resume: TrainingProgress | None = None,
    checkpoint_every_episodes: int = 0,
    checkpoint_callback: Callable[[TrainingProgress], None] | None = None,
    retention_checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    teacher_lookup: Callable[[str, int], np.ndarray | None] | None = None,
    teacher_channels: tuple[str, ...] | None = None,
    teacher_loss_end_scale: float = 1.0,
    teacher_guidance_dropout_start: float = 0.0,
    teacher_guidance_dropout_end: float = 0.0,
    episode_diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
    near_blow_loss_threshold: float | None = None,
    short_circuit_minimum_environment_steps: int | None = None,
    short_circuit_minimum_passes: int = 0,
    short_circuit_maximum_blow_rate: float = 1.0,
    collapse_window_episodes: int = 0,
    collapse_minimum_prior_passes: int = 0,
    collapse_maximum_recent_passes: int = 0,
    collapse_maximum_average_hold_bars: float = math.inf,
    collapse_minimum_voluntary_close_rate: float = 1.0,
) -> TrainingResult:
    if episodes < 1 or minimum_environment_steps < 1:
        raise ValueError("episode ceiling and minimum environment steps must be positive")
    if isinstance(prefetch_batches, bool) or not 0 <= prefetch_batches <= 2:
        raise ValueError("replay prefetch must be between zero and two")
    management_epsilon_start = (
        epsilon_start
        if management_epsilon_start is None
        else float(management_epsilon_start)
    )
    management_epsilon_end = (
        epsilon_end
        if management_epsilon_end is None
        else float(management_epsilon_end)
    )
    if not 0 <= management_epsilon_end <= management_epsilon_start <= 1:
        raise ValueError("management epsilon schedule is invalid")
    if near_blow_loss_threshold is not None and near_blow_loss_threshold <= 0:
        raise ValueError("near-blow loss threshold must be positive")
    if (
        short_circuit_minimum_environment_steps is not None
        and (
            isinstance(short_circuit_minimum_environment_steps, bool)
            or not 1
            <= short_circuit_minimum_environment_steps
            <= minimum_environment_steps
        )
    ):
        raise ValueError("training short-circuit step boundary is invalid")
    if (
        isinstance(short_circuit_minimum_passes, bool)
        or short_circuit_minimum_passes < 0
        or isinstance(short_circuit_maximum_blow_rate, bool)
        or not 0 <= short_circuit_maximum_blow_rate <= 1
    ):
        raise ValueError("training short-circuit outcome boundary is invalid")
    if collapse_window_episodes and (
        isinstance(collapse_window_episodes, bool)
        or collapse_window_episodes < 2
        or isinstance(collapse_minimum_prior_passes, bool)
        or collapse_minimum_prior_passes < 1
        or isinstance(collapse_maximum_recent_passes, bool)
        or not 0 <= collapse_maximum_recent_passes < collapse_window_episodes
        or isinstance(collapse_maximum_average_hold_bars, bool)
        or not np.isfinite(collapse_maximum_average_hold_bars)
        or collapse_maximum_average_hold_bars <= 0
        or isinstance(collapse_minimum_voluntary_close_rate, bool)
        or not 0 <= collapse_minimum_voluntary_close_rate <= 1
    ):
        raise ValueError("training collapse detector boundary is invalid")
    if teacher_channels is not None and (
        len(set(teacher_channels)) != len(teacher_channels)
        or not all(isinstance(channel, str) and channel for channel in teacher_channels)
    ):
        raise ValueError("teacher diagnostic channels are invalid")
    if not 0 <= teacher_loss_end_scale <= 1:
        raise ValueError("teacher loss end scale must be between zero and one")
    if not (
        0
        <= teacher_guidance_dropout_start
        <= teacher_guidance_dropout_end
        <= 1
    ):
        raise ValueError("teacher guidance dropout schedule is invalid")
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
    if progress.short_circuit_reason is not None:
        return progress.result()
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
        action_counts = {action: 0 for action in Action}
        selected_entry_teacher_targets: list[tuple[float, float]] = []
        selected_teacher_targets: list[np.ndarray] = []
        total_reward = 0.0
        # Decay exploration against actual market interaction, not the
        # emergency episode ceiling. Early pass/blow episodes vary greatly in
        # length, so episode-index decay would not represent equal experience.
        step_progress = min(
            1.0, progress.environment_steps / minimum_environment_steps
        )
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * step_progress
        management_epsilon = (
            management_epsilon_start
            + (management_epsilon_end - management_epsilon_start) * step_progress
        )
        teacher_weight_scale = 1.0 + (
            teacher_loss_end_scale - 1.0
        ) * step_progress
        teacher_guidance_dropout_probability = (
            teacher_guidance_dropout_start
            + (
                teacher_guidance_dropout_end
                - teacher_guidance_dropout_start
            )
            * step_progress
        )
        terminal_info = reset_info
        step_index = 0
        while True:
            if step_index and step_index % recurrent_horizon == 0:
                hidden = None
            action_epsilon = (
                management_epsilon
                if set(valid) == {Action.HOLD, Action.CLOSE}
                else epsilon
            )
            action, hidden, _ = agent.select_action(
                observation,
                hidden=hidden,
                valid_actions=valid,
                epsilon=action_epsilon,
            )
            action_counts[Action(action)] += 1
            next_observation, reward, terminated, _, info = environment.step(action)
            next_valid = tuple(info["valid_actions"])
            teacher_target = (
                teacher_lookup(episode_ticker, decision_index)
                if teacher_lookup is not None
                else None
            )
            if teacher_target is not None and not _teacher_guidance_is_visible(
                seed=ticker_seed,
                episode_index=episode_index,
                ticker=episode_ticker,
                decision_index=decision_index,
                dropout_probability=teacher_guidance_dropout_probability,
            ):
                teacher_target = None
            entry_opportunity_priority = 0.0
            if teacher_target is not None:
                entry_opportunity_priority = max(
                    float(teacher_target[0]) * float(teacher_target[1]),
                    float(teacher_target[2]) * float(teacher_target[3]),
                )
            if teacher_target is not None and action in {
                Action.ENTER_LONG_1, Action.ENTER_SHORT_1
            }:
                if (
                    teacher_channels is not None
                    and np.asarray(teacher_target).size != len(teacher_channels)
                ):
                    raise ValueError(
                        "teacher diagnostic channel width does not match target"
                    )
                offset = 0 if action == Action.ENTER_LONG_1 else 2
                selected_entry_teacher_targets.append((
                    float(teacher_target[offset]),
                    float(teacher_target[offset + 1]),
                ))
                selected_teacher_targets.append(
                    np.asarray(teacher_target, dtype=np.float32).reshape(-1)
                )
            transitions.append(Transition(
                observation=observation,
                action=Action(action),
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                valid_actions=valid,
                next_valid_actions=next_valid,
                recurrent_reset=(
                    step_index == 0 or step_index % recurrent_horizon == 0
                ),
                next_recurrent_reset=(
                    not terminated
                    and (step_index + 1) % recurrent_horizon == 0
                ),
                teacher_target=teacher_target,
                safety_priority=float(
                    info.get("mll_proximity_penalty", 0.0)
                ),
                entry_opportunity_priority=entry_opportunity_priority,
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
        if outcome == "pass" and retention_checkpoint_callback is not None:
            # Preserve the exact policy that produced the pass before replay
            # updates can alter it. This is a rollback anchor, not promotion
            # evidence; chronological teacher-free selection remains required.
            retain_policy = getattr(agent, "retain_policy", None)
            if retain_policy is not None:
                retain_policy()
            retention_checkpoint_callback({
                "episode": episode_index + 1,
                "ticker": str(terminal_info["ticker"]),
                "outcome": outcome,
                "terminal_pnl": terminal_pnl,
            })
        near_blow_timeout = bool(
            outcome == "timeout"
            and near_blow_loss_threshold is not None
            and terminal_pnl <= -near_blow_loss_threshold
        )
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
        episode_rl_losses = []
        episode_teacher_losses = []
        episode_entry_search_losses = []
        learner_diagnostics: dict[str, list[float]] = {
            key: []
            for key in (
                "gradient_norm",
                "sampled_management_row_fraction",
                "sampled_hold_reward",
                "sampled_close_reward",
                "sampled_hold_n_step_return",
                "sampled_close_n_step_return",
                "sampled_hold_td_loss",
                "sampled_close_td_loss",
                "management_hold_minus_close_q",
                "sampled_management_close_fraction",
                "sampled_recurrent_reset_fraction",
                "sampled_burn_in_reset_coverage",
                "sampled_recurrent_reset_pattern_count",
                "policy_retention_loss",
            )
        }
        if len(replay) >= warmup_episodes:
            def train_replay_batch(batch: Sequence[Sequence[Transition]]) -> None:
                episode_losses.append(agent.train_batch(
                    batch,
                    teacher_weight_scale=teacher_weight_scale,
                ))
                train_metrics = getattr(agent, "last_train_metrics", {})
                if "rl_loss" in train_metrics:
                    episode_rl_losses.append(float(train_metrics["rl_loss"]))
                if "teacher_loss" in train_metrics:
                    episode_teacher_losses.append(
                        float(train_metrics["teacher_loss"])
                    )
                if "entry_search_loss" in train_metrics:
                    episode_entry_search_losses.append(
                        float(train_metrics["entry_search_loss"])
                    )
                for key in learner_diagnostics:
                    if key in train_metrics:
                        learner_diagnostics[key].append(float(train_metrics[key]))

            if prefetch_batches:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending: deque[Future] = deque()
                    for _ in range(min(prefetch_batches, updates_per_episode)):
                        pending.append(executor.submit(replay.sample, batch_sequences))
                    for update_index in range(updates_per_episode):
                        batch = pending.popleft().result()
                        remaining = updates_per_episode - len(pending) - update_index - 1
                        if remaining > 0:
                            pending.append(
                                executor.submit(replay.sample, batch_sequences)
                            )
                        train_replay_batch(batch)
            else:
                for _ in range(updates_per_episode):
                    train_replay_batch(replay.sample(batch_sequences))
        trade_count = int(terminal_info.get("trade_count", 0))
        recent_outcomes = ()
        recent_hold_bars = ()
        recent_close_rates = ()
        if collapse_window_episodes:
            recent_outcomes = (
                *tuple(progress.recent_outcomes),
                outcome,
            )[-collapse_window_episodes:]
            recent_hold_bars = (
                *tuple(progress.recent_average_hold_bars),
                float(terminal_info.get("avg_hold_bars", 0.0)),
            )[-collapse_window_episodes:]
            recent_close_rates = (
                *tuple(progress.recent_voluntary_close_rates),
                (
                    int(terminal_info.get("voluntary_close_count", 0))
                    / trade_count if trade_count else 0.0
                ),
            )[-collapse_window_episodes:]
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
            trade_r_sum=(
                progress.trade_r_sum
                + float(terminal_info.get("expectancy_r", 0.0))
                * int(terminal_info.get("trade_count", 0))
            ),
            worst_pnl=min(progress.worst_pnl, terminal_pnl),
            terminal_pnl_sum=progress.terminal_pnl_sum + terminal_pnl,
            terminal_pnl_count=progress.terminal_pnl_count + 1,
            reward_sum=progress.reward_sum + total_reward,
            reward_count=progress.reward_count + 1,
            loss_sum=progress.loss_sum + sum(episode_losses),
            loss_count=progress.loss_count + len(episode_losses),
            mfe_sum=(
                progress.mfe_sum
                + float(terminal_info.get("avg_mfe_r", 0.0))
                * int(terminal_info.get("trade_count", 0))
            ),
            mae_sum=(
                progress.mae_sum
                + float(terminal_info.get("avg_mae_r", 0.0))
                * int(terminal_info.get("trade_count", 0))
            ),
            retention_eligible_count=(
                progress.retention_eligible_count
                + int(terminal_info.get("retention_eligible_count", 0))
            ),
            retention_capture_sum=(
                progress.retention_capture_sum
                + float(terminal_info.get("mfe_capture_ratio", 0.0))
                * int(terminal_info.get("retention_eligible_count", 0))
            ),
            retention_gap_sum=(
                progress.retention_gap_sum
                + float(terminal_info.get("mfe_realized_gap_r", 0.0))
                * int(terminal_info.get("retention_eligible_count", 0))
            ),
            retention_round_trip_count=(
                progress.retention_round_trip_count
                + round(
                    float(terminal_info.get("gave_it_all_back_rate", 0.0))
                    * int(terminal_info.get("retention_eligible_count", 0))
                )
            ),
            two_r_eligible_count=(
                progress.two_r_eligible_count
                + int(terminal_info.get("two_r_eligible_count", 0))
            ),
            two_r_capture_sum=(
                progress.two_r_capture_sum
                + float(terminal_info.get("two_r_mfe_capture_ratio", 0.0))
                * int(terminal_info.get("two_r_eligible_count", 0))
            ),
            two_r_round_trip_count=(
                progress.two_r_round_trip_count
                + round(
                    float(terminal_info.get("two_r_gave_it_all_back_rate", 0.0))
                    * int(terminal_info.get("two_r_eligible_count", 0))
                )
            ),
            near_blow_timeout_count=(
                progress.near_blow_timeout_count + int(near_blow_timeout)
            ),
            recent_outcomes=recent_outcomes,
            recent_average_hold_bars=recent_hold_bars,
            recent_voluntary_close_rates=recent_close_rates,
        )
        reasons = []
        if (
            short_circuit_minimum_environment_steps is not None
            and progress.environment_steps >= short_circuit_minimum_environment_steps
        ):
            if progress.passes < short_circuit_minimum_passes:
                reasons.append(
                    f"passes {progress.passes} < {short_circuit_minimum_passes}"
                )
            blow_rate = progress.blows / progress.completed_episodes
            if blow_rate > short_circuit_maximum_blow_rate:
                reasons.append(
                    f"blow rate {blow_rate:.6f} > "
                    f"{short_circuit_maximum_blow_rate:.6f}"
                )
        if (
            collapse_window_episodes
            and len(progress.recent_outcomes) == collapse_window_episodes
        ):
            recent_passes = sum(
                recent_outcome == "pass"
                for recent_outcome in progress.recent_outcomes
            )
            prior_passes = progress.passes - recent_passes
            recent_hold = float(np.mean(progress.recent_average_hold_bars))
            recent_close_rate = float(
                np.mean(progress.recent_voluntary_close_rates)
            )
            if (
                prior_passes >= collapse_minimum_prior_passes
                and recent_passes <= collapse_maximum_recent_passes
                and recent_hold <= collapse_maximum_average_hold_bars
                and recent_close_rate >= collapse_minimum_voluntary_close_rate
            ):
                reasons.append(
                    "policy collapse: "
                    f"prior passes {prior_passes}; "
                    f"recent passes {recent_passes}/"
                    f"{collapse_window_episodes}; "
                    f"recent average hold {recent_hold:.6f} <= "
                    f"{collapse_maximum_average_hold_bars:.6f}; "
                    f"recent voluntary-close rate {recent_close_rate:.6f} >= "
                    f"{collapse_minimum_voluntary_close_rate:.6f}"
                )
        if reasons:
            progress = replace(
                progress,
                short_circuit_reason="; ".join(reasons),
            )
        cumulative_average_balance = (
            progress.terminal_pnl_sum / progress.terminal_pnl_count
        )
        if episode_diagnostic_callback is not None:
            diagnostic = {
                "schema": "propevolve_episode_diagnostic_v1",
                "episode": progress.completed_episodes,
                "ticker": str(terminal_info["ticker"]),
                "outcome": outcome,
                "reward": total_reward,
                "environment_steps": progress.environment_steps,
                "trade_count": int(terminal_info.get("trade_count", 0)),
                "win_rate": float(terminal_info.get("win_rate", 0.0)),
                "avg_win_r": float(terminal_info.get("avg_win_r", 0.0)),
                "avg_loss_r": float(terminal_info.get("avg_loss_r", 0.0)),
                "expectancy_r": float(terminal_info.get("expectancy_r", 0.0)),
                "avg_mfe_r": float(terminal_info.get("avg_mfe_r", 0.0)),
                "avg_mae_r": float(terminal_info.get("avg_mae_r", 0.0)),
                "retention_eligible_count": int(
                    terminal_info.get("retention_eligible_count", 0)
                ),
                "mfe_capture_ratio": float(
                    terminal_info.get("mfe_capture_ratio", 0.0)
                ),
                "mfe_realized_gap_r": float(
                    terminal_info.get("mfe_realized_gap_r", 0.0)
                ),
                "gave_it_all_back_rate": float(
                    terminal_info.get("gave_it_all_back_rate", 0.0)
                ),
                "two_r_eligible_count": int(
                    terminal_info.get("two_r_eligible_count", 0)
                ),
                "two_r_mfe_capture_ratio": float(
                    terminal_info.get("two_r_mfe_capture_ratio", 0.0)
                ),
                "two_r_gave_it_all_back_rate": float(
                    terminal_info.get("two_r_gave_it_all_back_rate", 0.0)
                ),
                "ratchet_activation_rate": float(
                    terminal_info.get("ratchet_activation_rate", 0.0)
                ),
                "activated_avg_realized_r": float(
                    terminal_info.get("activated_avg_realized_r", 0.0)
                ),
                "avg_hold_bars": float(terminal_info.get("avg_hold_bars", 0.0)),
                "voluntary_close_count": int(
                    terminal_info.get("voluntary_close_count", 0)
                ),
                "initial_stop_count": int(
                    terminal_info.get("initial_stop_count", 0)
                ),
                "ratchet_stop_count": int(
                    terminal_info.get("ratchet_stop_count", 0)
                ),
                "terminal_liquidation_count": int(
                    terminal_info.get("terminal_liquidation_count", 0)
                ),
                "largest_realized_trade": terminal_info.get(
                    "largest_realized_trade"
                ),
                "largest_mfe_trade": terminal_info.get("largest_mfe_trade"),
                "terminal_pnl": terminal_pnl,
                "near_blow_timeout": near_blow_timeout,
                "primary_side": str(terminal_info.get("primary_side", "flat")),
                "entry_epsilon": epsilon,
                "management_epsilon": management_epsilon,
                "teacher_weight_scale": teacher_weight_scale,
                "teacher_guidance_dropout_probability": (
                    teacher_guidance_dropout_probability
                ),
                "n_step_return": int(getattr(agent, "n_step_return", 1)),
                "recurrent_burn_in": int(
                    getattr(agent, "recurrent_burn_in", 0)
                ),
                "updates": len(episode_losses),
                "mean_training_loss": (
                    float(np.mean(episode_losses)) if episode_losses else None
                ),
                "mean_rl_loss": (
                    float(np.mean(episode_rl_losses))
                    if episode_rl_losses else None
                ),
                "mean_teacher_loss": (
                    float(np.mean(episode_teacher_losses))
                    if episode_teacher_losses else None
                ),
                "mean_entry_search_loss": (
                    float(np.mean(episode_entry_search_losses))
                    if episode_entry_search_losses else None
                ),
                **{
                    f"mean_{key}": (
                        float(np.mean(values)) if values else None
                    )
                    for key, values in learner_diagnostics.items()
                },
                "cumulative_passes": progress.passes,
                "cumulative_blows": progress.blows,
                "cumulative_timeouts": progress.timeouts,
                "cumulative_pass_rate": progress.passes / progress.completed_episodes,
                "cumulative_blow_rate": progress.blows / progress.completed_episodes,
                "cumulative_average_balance": cumulative_average_balance,
                "action_counts": {
                    action.name: action_counts[action] for action in Action
                },
                "teacher_scored_entries": len(selected_entry_teacher_targets),
                "selected_side_attempt_probability_mean": (
                    float(np.mean([value[0] for value in selected_entry_teacher_targets]))
                    if selected_entry_teacher_targets else None
                ),
                "selected_side_clean_retained_probability_mean": (
                    float(np.mean([value[1] for value in selected_entry_teacher_targets]))
                    if selected_entry_teacher_targets else None
                ),
                "selected_teacher_channel_means": (
                    {
                        channel: float(value)
                        for channel, value in zip(
                            teacher_channels,
                            np.mean(np.stack(selected_teacher_targets), axis=0),
                        )
                    }
                    if teacher_channels is not None and selected_teacher_targets
                    else None
                ),
                "training_short_circuited": (
                    progress.short_circuit_reason is not None
                ),
                "training_short_circuit_reason": progress.short_circuit_reason,
            }
            for horizon in (5, 10, 20, 50):
                prefix = f"shadow_h{horizon}"
                for suffix in (
                    "complete_trades",
                    "avg_mfe_r",
                    "avg_mae_r",
                    "2r_before_1r_rate",
                    "3r_before_1r_rate",
                ):
                    diagnostic[f"{prefix}_{suffix}"] = terminal_info.get(
                        f"{prefix}_{suffix}", 0
                    )
            episode_diagnostic_callback(diagnostic)
        print(
            f"[train] episode={episode_index + 1}/{episodes} ticker={terminal_info['ticker']} "
            f"outcome={outcome} reward={total_reward:.4f} replay={len(replay)} "
            f"trades={int(terminal_info.get('trade_count', 0))} "
            f"WR={float(terminal_info.get('win_rate', 0.0)):.1%} "
            f"winR={float(terminal_info.get('avg_win_r', 0.0)):+.3f}R "
            f"balance={terminal_pnl:+.2f} "
            f"avg_balance={cumulative_average_balance:+.2f} "
            f"steps={progress.environment_steps:,}/{minimum_environment_steps:,}",
            flush=True,
        )
        if (
            checkpoint_every_episodes
            and (
                progress.completed_episodes % checkpoint_every_episodes == 0
                or progress.short_circuit_reason is not None
                or progress.environment_steps >= minimum_environment_steps
            )
        ):
            assert checkpoint_callback is not None
            checkpoint_callback(progress)
        if (
            progress.short_circuit_reason is not None
            or progress.environment_steps >= minimum_environment_steps
        ):
            break
    if (
        progress.short_circuit_reason is None
        and progress.environment_steps < minimum_environment_steps
    ):
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
    near_blow_loss_threshold: float | None = None,
    stop_on_first_blow: bool = False,
) -> TrainingResult:
    if near_blow_loss_threshold is not None and near_blow_loss_threshold <= 0:
        raise ValueError("near-blow loss threshold must be positive")
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    rewards = []
    terminal_pnls = []
    trade_count = win_count = 0
    winning_r_sum = 0.0
    trade_r_sum = 0.0
    mfe_sum = mae_sum = 0.0
    retention_eligible_count = retention_round_trip_count = 0
    retention_capture_sum = retention_gap_sum = 0.0
    two_r_eligible_count = two_r_round_trip_count = 0
    two_r_capture_sum = 0.0
    near_blow_timeout_count = 0
    environment_steps = 0
    by_outcome = {
        outcome: {
            "episodes": 0,
            "trade_count": 0,
            "win_count": 0,
            "winning_r_sum": 0.0,
            "terminal_pnl_sum": 0.0,
            "reward_sum": 0.0,
            "mfe_sum": 0.0,
            "mae_sum": 0.0,
            "retention_eligible_count": 0,
            "retention_capture_sum": 0.0,
            "retention_gap_sum": 0.0,
            "retention_round_trip_count": 0,
            "two_r_eligible_count": 0,
            "two_r_capture_sum": 0.0,
            "two_r_round_trip_count": 0,
        }
        for outcome in outcomes
    }
    evaluated_episodes = 0
    for episode_index in range(episodes):
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
        near_blow_timeout = bool(
            outcome == "timeout"
            and near_blow_loss_threshold is not None
            and terminal_pnl <= -near_blow_loss_threshold
        )
        near_blow_timeout_count += int(near_blow_timeout)
        episode_trades = int(info.get("trade_count", 0))
        episode_wins = int(info.get("win_count", 0))
        episode_winning_r = float(info.get("winning_r_sum", 0.0))
        episode_trade_r = float(info.get("expectancy_r", 0.0)) * episode_trades
        episode_mfe_sum = float(info.get("avg_mfe_r", 0.0)) * episode_trades
        episode_mae_sum = float(info.get("avg_mae_r", 0.0)) * episode_trades
        episode_retention_count = int(info.get("retention_eligible_count", 0))
        episode_retention_capture = (
            float(info.get("mfe_capture_ratio", 0.0)) * episode_retention_count
        )
        episode_retention_gap = (
            float(info.get("mfe_realized_gap_r", 0.0)) * episode_retention_count
        )
        episode_round_trips = round(
            float(info.get("gave_it_all_back_rate", 0.0))
            * episode_retention_count
        )
        episode_two_r_count = int(info.get("two_r_eligible_count", 0))
        episode_two_r_capture = (
            float(info.get("two_r_mfe_capture_ratio", 0.0)) * episode_two_r_count
        )
        episode_two_r_round_trips = round(
            float(info.get("two_r_gave_it_all_back_rate", 0.0))
            * episode_two_r_count
        )
        terminal_pnls.append(terminal_pnl)
        trade_count += episode_trades
        win_count += episode_wins
        winning_r_sum += episode_winning_r
        trade_r_sum += episode_trade_r
        mfe_sum += episode_mfe_sum
        mae_sum += episode_mae_sum
        retention_eligible_count += episode_retention_count
        retention_capture_sum += episode_retention_capture
        retention_gap_sum += episode_retention_gap
        retention_round_trip_count += episode_round_trips
        two_r_eligible_count += episode_two_r_count
        two_r_capture_sum += episode_two_r_capture
        two_r_round_trip_count += episode_two_r_round_trips
        outcome_values = by_outcome[outcome]
        outcome_values["episodes"] += 1
        outcome_values["trade_count"] += episode_trades
        outcome_values["win_count"] += episode_wins
        outcome_values["winning_r_sum"] += episode_winning_r
        outcome_values["terminal_pnl_sum"] += terminal_pnl
        outcome_values["reward_sum"] += total
        outcome_values["mfe_sum"] += episode_mfe_sum
        outcome_values["mae_sum"] += episode_mae_sum
        outcome_values["retention_eligible_count"] += episode_retention_count
        outcome_values["retention_capture_sum"] += episode_retention_capture
        outcome_values["retention_gap_sum"] += episode_retention_gap
        outcome_values["retention_round_trip_count"] += episode_round_trips
        outcome_values["two_r_eligible_count"] += episode_two_r_count
        outcome_values["two_r_capture_sum"] += episode_two_r_capture
        outcome_values["two_r_round_trip_count"] += episode_two_r_round_trips
        episode_win_rate = (
            episode_wins / episode_trades if episode_trades else 0.0
        )
        episode_average_win_r = (
            episode_winning_r / episode_wins if episode_wins else 0.0
        )
        print(
            f"[validation] episode={episode_index + 1}/{episodes} "
            f"ticker={info.get('ticker', '?')} outcome={outcome} "
            f"reward={total:+.4f} trades={episode_trades} "
            f"WR={episode_win_rate:.1%} winR={episode_average_win_r:+.3f}R "
            f"pnl={terminal_pnl:+.2f} "
            f"cumulative_pass={outcomes['pass']} "
            f"cumulative_blow={outcomes['blow']} "
            f"cumulative_timeout={outcomes['timeout']}",
            flush=True,
        )
        evaluated_episodes = episode_index + 1
        if stop_on_first_blow and outcome == "blow":
            print(
                "[validation] SHORT_CIRCUIT reason=zero_blow_gate "
                f"episode={evaluated_episodes}/{episodes}",
                flush=True,
            )
            break
    result = TrainingResult(
        episodes=evaluated_episodes,
        environment_steps=environment_steps,
        passes=outcomes["pass"],
        blows=outcomes["blow"],
        timeouts=outcomes["timeout"],
        trade_count=trade_count,
        win_count=win_count,
        winning_r_sum=winning_r_sum,
        trade_r_sum=trade_r_sum,
        worst_pnl=float(np.min(terminal_pnls)),
        mean_terminal_pnl=float(np.mean(terminal_pnls)),
        mean_reward=float(np.mean(rewards)),
        mean_loss=float("nan"),
        outcome_statistics=tuple(
            OutcomeStatistics(outcome=outcome, **values)
            for outcome, values in by_outcome.items()
            if values["episodes"]
        ),
        mfe_sum=mfe_sum,
        mae_sum=mae_sum,
        retention_eligible_count=retention_eligible_count,
        retention_capture_sum=retention_capture_sum,
        retention_gap_sum=retention_gap_sum,
        retention_round_trip_count=retention_round_trip_count,
        two_r_eligible_count=two_r_eligible_count,
        two_r_capture_sum=two_r_capture_sum,
        two_r_round_trip_count=two_r_round_trip_count,
        near_blow_timeout_count=near_blow_timeout_count,
    )
    episode_display = (
        str(episodes)
        if evaluated_episodes == episodes
        else f"{evaluated_episodes}/{episodes}"
    )
    print(
        f"[validation] COMPLETE episodes={episode_display} "
        f"pass={result.passes} blow={result.blows} timeout={result.timeouts} "
        f"near_blow_timeout={result.near_blow_timeout_count} "
        f"({result.near_blow_timeout_rate:.1%}) "
        f"WR={result.trade_win_rate:.1%} winR={result.average_win_r:+.3f}R "
        f"mean_pnl={result.mean_terminal_pnl:+.2f}",
        flush=True,
    )
    return result
