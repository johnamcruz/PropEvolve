"""Historical training and temporal evaluation for the PropEvolve POC."""

from __future__ import annotations

from collections.abc import Mapping
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np
import torch

from .assets import AssetContract
from .balance_aware_regime_selectivity import (
    ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
    BalanceAwareRegimeSelectivity,
    PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
    PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
    EXPANSION_CHANNELS,
    REGIME_STATE_NAMES,
    REGIME_TEACHER_CHANNELS,
)
from .cache import load_market_series
from .config import (
    agent_runtime_settings,
    configure_runtime_environment,
    materialize_effective_config,
)
from .decision import Action, PositionSide
from .environment import (
    ChallengeSpec,
    ChallengeStartState,
    HistoricalChallengeEnv,
    MarketSeries,
)
from .episode_coverage import FullDataEpisodeCoverageSpec
from .entry_timing_audit import audit_entry_timing_episode
from .evolution import (
    CandidateArchive,
    EvaluationGate,
    EvaluationStage,
    EvaluatorCascade,
)
from .final_regime_probe import (
    SAMPLES_PER_ACTION as FINAL_REGIME_PROBE_SAMPLES_PER_ACTION,
    evaluate_final_regime_probe,
)
from .observation import TradeManagementObservationSpec
from .replay import (
    BalancedSequenceReplay,
    Episode,
    ReplayCheckpointStore,
    Transition,
)
from .recovery import (
    RecoveryHandoffPolicy,
    RecoveryValueStore,
    build_recovery_value_target,
    select_recovery_target_prefix,
)
from .teachers import agent_teacher_settings
from .training_health import (
    TrainingHealthDetector,
    TrainingHealthMonitor,
    TrainingPolicyHealthSpec,
)

if TYPE_CHECKING:
    from .agent import RecurrentC51Agent


_ENTRY_ACTION_ORDER = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1")
_REGIME_SELECTIVITY_STRATA = (
    "positive_long_short",
    "positive_long",
    "positive_short",
    "dominant_chop",
    "nonchop",
    "low_headroom_le_0_25",
    "mid_headroom_gt_0_25_lt_0_75",
    "safe_headroom_ge_0_75",
)
_REGIME_ACTION_NAMES = ("wait", "long", "short")
_REGIME_CONFUSION_FIELDS = tuple(
    f"target_{target}_predicted_{prediction}_rows"
    for target in _REGIME_ACTION_NAMES
    for prediction in _REGIME_ACTION_NAMES
)
_REGIME_SELECTIVITY_ADDITIVE_FIELDS = (
    "rows",
    "target_wait_probability_sum",
    "model_wait_probability_sum",
    "wait_absolute_error_sum",
    "target_action_probability_sum",
    "model_target_action_probability_sum",
    "target_action_absolute_error_sum",
    "greedy_wait_rows",
    "declared_side_probability_sum",
    "greedy_entry_rows",
    "correct_rows",
    *_REGIME_CONFUSION_FIELDS,
)
_REGIME_CHANNEL_ADDITIVE_FIELDS = (
    "rows",
    "target_probability_sum",
    "model_probability_sum",
    "absolute_error_sum",
    "squared_error_sum",
)
_ENTRY_BALANCE_ACTION_NAMES = ("wait", "long", "short")
_ENTRY_BALANCE_ADDITIVE_FIELDS = (
    "rows",
    "weighted_mass",
    "unweighted_ce_sum",
    "weighted_ce_sum",
)
_REGIME_ENTRY_CONFLICT_FIELDS = (
    "rows",
    "target_wait_probability_sum",
    "target_declared_side_probability_sum",
    "model_wait_probability_sum",
    "soft_wait_disagreement_rows",
)
_PERSISTENT_REGIME_SELECTIVITY_STRATA = (
    "exact_wait",
    "persistent_dead_chop",
    "transition_ready",
    "failed_setup_confluence",
    "failed_long_confluence",
    "failed_short_confluence",
)
_PERSISTENT_REGIME_SELECTIVITY_ADDITIVE_FIELDS = (
    "rows",
    "weight_sum",
    "model_wait_probability_sum",
)
_TRANSITION_POSITIVE_SIDE_FIELDS = (
    "long",
    "short",
)
_REGIME_ASSOCIATION_COHORTS = (
    "dead_wait",
    "transition_positive_long",
    "transition_positive_short",
)
_REGIME_ASSOCIATION_ADDITIVE_FIELDS = (
    "rows",
    "model_wait_probability_sum",
)
_PAIRED_A_PLUS_SIDES = ("long", "short")
_PAIRED_A_PLUS_REGIMES = REGIME_STATE_NAMES
_PAIRED_A_PLUS_GROUP_FIELDS = (
    "pair_count",
    "pair_mass",
    "loss_sum",
    "good_advantage_sum",
    "bad_advantage_sum",
)


def _entry_action_balance(
    entry_action_targets,
    entry_supervision_spec: Mapping[str, object] | None,
) -> tuple[tuple[float, float, float], Mapping[str, object] | None]:
    """Resolve one authenticated training-only class-balance contract."""

    if entry_action_targets is None or entry_supervision_spec is None:
        return (1.0, 1.0, 1.0), None
    if entry_supervision_spec.get("action_class_balance") is None:
        return (1.0, 1.0, 1.0), None
    receipt = entry_action_targets.balance_receipt()
    weights = receipt["class_weights"]
    return (
        tuple(float(weights[action]) for action in _ENTRY_ACTION_ORDER),
        receipt,
    )


def _entry_supervision_frozen_contract(
    entry_action_targets,
    balance_receipt: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Archive target and derived-weight identities together."""

    if entry_action_targets is None:
        return None
    return {
        "training_only": True,
        "manifest": _plain_contract_value(entry_action_targets.manifest),
        "balance_receipt": _plain_contract_value(balance_receipt),
    }


def _assert_recovery_entry_balance(
    agent,
    agent_settings: Mapping[str, object],
) -> None:
    """Fail closed when recovery weights differ from rebuilt target lineage."""

    expected = tuple(
        agent_settings.get("entry_action_class_weights", (1.0, 1.0, 1.0))
    )
    expected_reduction = str(agent_settings.get(
        "entry_action_loss_reduction", "population_weighted_mean_v1"
    ))
    expected_margin = float(agent_settings.get("entry_action_margin", 0.0))
    expected_gradient_conflict_mode = str(
        agent_settings.get("auxiliary_gradient_conflict_mode", "none")
    )
    if (
        agent.entry_action_class_weights != expected
        or getattr(
            agent,
            "entry_action_loss_reduction",
            "population_weighted_mean_v1",
        )
        != expected_reduction
        or getattr(agent, "entry_action_margin", 0.0) != expected_margin
        or getattr(agent, "auxiliary_gradient_conflict_mode", "none")
        != expected_gradient_conflict_mode
    ):
        raise ValueError("training recovery entry balance drifted")


def _assert_recovery_regime_selectivity(
    agent,
    agent_settings: Mapping[str, object],
) -> None:
    """Fail closed when resumed Stage 2A learning semantics drift."""
    expected_semantics = agent_settings.get("regime_selectivity_semantics")
    if expected_semantics is None:
        return
    expected_emphasis = float(
        agent_settings["regime_selectivity_persistent_chop_negative_emphasis"]
    )
    expected_chop_margin = float(
        agent_settings.get("regime_selectivity_chop_wait_margin", 0.0)
    )
    expected_failed_margin = float(
        agent_settings.get("regime_selectivity_failed_confluence_margin", 0.0)
    )
    expected_paired_margin = float(
        agent_settings.get("regime_selectivity_paired_a_plus_margin", 0.0)
    )
    expected_winner_loss_weight = float(agent_settings.get(
        "regime_selectivity_paired_a_plus_winner_loss_weight", 1.0
    ))
    undeclared_chop_default = (
        0.0
        if "regime_selectivity_chop_wait_margin" not in agent_settings
        else math.nan
    )
    undeclared_failed_default = (
        0.0
        if "regime_selectivity_failed_confluence_margin" not in agent_settings
        else math.nan
    )
    if (
        getattr(agent, "regime_selectivity_semantics", None)
        != expected_semantics
        or not math.isclose(
            float(getattr(
                agent,
                "regime_selectivity_failed_confluence_margin",
                undeclared_failed_default,
            )),
            expected_failed_margin,
        )
        or not math.isclose(
            float(getattr(
                agent,
                "regime_selectivity_persistent_chop_negative_emphasis",
                math.nan,
            )),
            expected_emphasis,
        )
        or not math.isclose(
            float(getattr(
                agent,
                "regime_selectivity_chop_wait_margin",
                undeclared_chop_default,
            )),
            expected_chop_margin,
        )
        or not math.isclose(
            float(getattr(
                agent,
                "regime_selectivity_paired_a_plus_margin",
                0.0,
            )),
            expected_paired_margin,
        )
        or not math.isclose(
            float(getattr(
                agent,
                "regime_selectivity_paired_a_plus_winner_loss_weight",
                1.0,
            )),
            expected_winner_loss_weight,
        )
        or getattr(agent, "regime_selectivity_side_balance", None)
        != agent_settings.get("regime_selectivity_side_balance")
    ):
        raise ValueError("training recovery Regime learning identity drifted")


def _regime_selectivity_agent_settings(
    specification: Mapping[str, object] | None,
) -> dict[str, object]:
    """Project one validated Stage 2A recipe onto the learner interface."""
    if specification is None:
        return {}
    side_balance = specification.get("side_balance")
    settings = {
        "regime_selectivity_loss_weight": float(specification["loss_weight"]),
        "regime_selectivity_expansion_centers": (
            float(specification["expansion_long_center"]),
            float(specification["expansion_short_center"]),
        ),
        "regime_selectivity_probability_epsilon": float(
            specification["probability_epsilon"]
        ),
        "regime_selectivity_headroom_pressure": float(
            specification["headroom_pressure"]
        ),
        "regime_selectivity_dominant_chop_pressure": float(
            specification["dominant_chop_pressure"]
        ),
        "regime_selectivity_q_temperature": float(
            specification["q_temperature"]
        ),
        "regime_selectivity_semantics": str(
            specification.get("semantics", "static_state_v1")
        ),
        "regime_selectivity_persistent_chop_negative_emphasis": float(
            specification.get("persistent_chop_negative_emphasis", 0.0)
        ),
        "regime_selectivity_side_balance": (
            "none"
            if side_balance is None
            else str(side_balance["schema"])
        ),
    }
    if "chop_wait_margin" in specification:
        settings.update({
            "regime_selectivity_chop_wait_margin": float(
                specification["chop_wait_margin"]
            ),
            "regime_selectivity_failed_confluence_margin": float(
                specification["failed_confluence_margin"]
            ),
        })
    if "paired_a_plus_margin" in specification:
        settings["regime_selectivity_paired_a_plus_margin"] = float(
            specification["paired_a_plus_margin"]
        )
    if "paired_a_plus_winner_loss_weight" in specification:
        settings[
            "regime_selectivity_paired_a_plus_winner_loss_weight"
        ] = float(specification["paired_a_plus_winner_loss_weight"])
    return settings


def _regime_selectivity_replay_settings(
    specification: Mapping[str, object] | None,
) -> dict[str, object]:
    """Project the frozen side sampler identity onto replay construction."""
    if specification is None:
        return {}
    side_balance = specification.get("side_balance")
    return {
        "entry_opportunity_side_balance": (
            "none"
            if side_balance is None
            else str(side_balance["schema"])
        )
    }


def _paired_a_plus_transition_evidence(
    *,
    teacher_target: np.ndarray | None,
    teacher_channels: Sequence[str] | None,
    entry_action_target: Action | None,
    metadata: object | None,
) -> tuple[np.ndarray | None, Action | None, bool | None]:
    """Bind one exact economic winner/failure to continuous teacher context."""
    if metadata is None or entry_action_target is None:
        return None, None, None
    context_channels = (*EXPANSION_CHANNELS, *REGIME_TEACHER_CHANNELS)
    if (
        teacher_target is None
        or teacher_channels is None
        or tuple(teacher_channels[: len(context_channels)]) != context_channels
    ):
        raise ValueError("paired recurrent A+ row lacks teacher context")
    side_action = {
        "long": Action.ENTER_LONG_1,
        "short": Action.ENTER_SHORT_1,
    }.get(getattr(metadata, "side", None))
    economic_win = getattr(metadata, "economic_win", None)
    economic_good = getattr(metadata, "economic_good", None)
    is_winner = (
        side_action is not None
        and economic_win is True
        and economic_good is True
        and entry_action_target == side_action
    )
    is_failure = (
        side_action is not None
        and economic_win is False
        and entry_action_target == Action.WAIT
    )
    if (
        getattr(metadata, "available", None) is not True
        or getattr(metadata, "censored", None) is not False
        or not (is_winner or is_failure)
    ):
        return None, None, None
    context = np.asarray(teacher_target, dtype=np.float32).reshape(-1)[
        : len(context_channels)
    ]
    if (
        context.shape != (len(context_channels),)
        or not np.isfinite(context).all()
        or (context < 0.0).any()
        or (context > 1.0).any()
    ):
        raise ValueError("paired recurrent A+ context is invalid")
    return context.copy(), side_action, bool(is_winner)


def _with_regime_wait_replay_priorities(
    transitions: Sequence[Transition],
    compiler: BalanceAwareRegimeSelectivity,
) -> tuple[Transition, ...]:
    """Attach loss-identical hard-WAIT priority once per completed episode."""
    eligible = [
        index
        for index, transition in enumerate(transitions)
        if transition.teacher_target is not None
        and transition.entry_action_target is not None
    ]
    if not eligible:
        return tuple(transitions)
    teachers = np.stack([
        np.asarray(transitions[index].teacher_target, dtype=np.float32)
        for index in eligible
    ])
    targets = np.asarray([
        int(Action(transitions[index].entry_action_target))
        for index in eligible
    ], dtype=np.int64)
    import torch

    with torch.no_grad():
        priorities = compiler.exact_wait_replay_priorities(
            torch.from_numpy(teachers),
            torch.from_numpy(targets),
        ).cpu().numpy()
    annotated = list(transitions)
    for index, priority in zip(eligible, priorities, strict=True):
        annotated[index] = replace(
            annotated[index],
            regime_wait_priority=float(priority),
        )
    return tuple(annotated)


def _regime_selectivity_probe_settings(
    specification: Mapping[str, object],
) -> dict[str, object]:
    """Project the authenticated learner identity onto policy probes."""
    return {
        "regime_selectivity_semantics": str(specification["semantics"]),
        "regime_selectivity_expansion_centers": (
            float(specification["expansion_long_center"]),
            float(specification["expansion_short_center"]),
        ),
    }


def _bounded_regime_selectivity_headroom(value: object) -> float:
    """Map finite account headroom onto the selectivity pressure domain."""
    headroom = float(value)
    if not np.isfinite(headroom) or headroom < 0.0:
        raise ValueError("decision-time MLL headroom fraction is invalid")
    return min(headroom, 1.0)


def _regime_selectivity_frozen_contract(
    specification: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Archive the complete validated training-only Stage 2A identity."""
    if specification is None:
        return None
    if specification.get("training_only") is not True:
        raise ValueError("Regime selectivity must remain training-only")
    return _plain_contract_value(specification)


def _regime_selectivity_episode_diagnostic(
    update_metrics: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    """Reduce optimizer updates by additive row statistics, never mean-of-means."""
    result: dict[str, dict[str, float | int]] = {}
    for stratum in _REGIME_SELECTIVITY_STRATA:
        prefix = f"regime_selectivity_{stratum}_"
        additive = {
            field: float(sum(update_metrics.get(prefix + field, ())))
            for field in _REGIME_SELECTIVITY_ADDITIVE_FIELDS
        }
        rows = int(round(additive["rows"]))
        result[stratum] = {
            "rows": rows,
            "target_wait_probability_sum": additive[
                "target_wait_probability_sum"
            ],
            "target_wait_probability_mean": (
                additive["target_wait_probability_sum"] / rows
                if rows else 0.0
            ),
            "model_wait_probability_sum": additive[
                "model_wait_probability_sum"
            ],
            "model_wait_probability_mean": (
                additive["model_wait_probability_sum"] / rows
                if rows else 0.0
            ),
            "wait_absolute_error_sum": additive["wait_absolute_error_sum"],
            "wait_mean_absolute_error": (
                additive["wait_absolute_error_sum"] / rows if rows else 0.0
            ),
            "target_action_probability_sum": additive[
                "target_action_probability_sum"
            ],
            "target_action_probability_mean": (
                additive["target_action_probability_sum"] / rows
                if rows else 0.0
            ),
            "model_target_action_probability_sum": additive[
                "model_target_action_probability_sum"
            ],
            "model_target_action_probability_mean": (
                additive["model_target_action_probability_sum"] / rows
                if rows else 0.0
            ),
            "target_action_absolute_error_sum": additive[
                "target_action_absolute_error_sum"
            ],
            "target_action_mean_absolute_error": (
                additive["target_action_absolute_error_sum"] / rows
                if rows else 0.0
            ),
            "greedy_wait_rows": int(round(additive["greedy_wait_rows"])),
            "greedy_wait_rate": (
                additive["greedy_wait_rows"] / rows if rows else 0.0
            ),
            "declared_side_probability_sum": additive[
                "declared_side_probability_sum"
            ],
            "declared_side_probability_mean": (
                additive["declared_side_probability_sum"] / rows
                if rows else 0.0
            ),
            "greedy_entry_rows": int(round(additive["greedy_entry_rows"])),
            "greedy_entry_rate": (
                additive["greedy_entry_rows"] / rows if rows else 0.0
            ),
            "correct_rows": int(round(additive["correct_rows"])),
            "accuracy": additive["correct_rows"] / rows if rows else 0.0,
            "confusion": {
                target: {
                    prediction: int(round(additive[
                        f"target_{target}_predicted_{prediction}_rows"
                    ]))
                    for prediction in _REGIME_ACTION_NAMES
                }
                for target in _REGIME_ACTION_NAMES
            },
        }
    return result


def _regime_selectivity_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate episode diagnostics by exact sampled positive-row counts."""
    update_metrics: dict[str, list[float]] = {
        f"regime_selectivity_{stratum}_{field}": []
        for stratum in _REGIME_SELECTIVITY_STRATA
        for field in _REGIME_SELECTIVITY_ADDITIVE_FIELDS
    }
    for row in rows:
        diagnostics = row.get("regime_selectivity")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        for stratum in _REGIME_SELECTIVITY_STRATA:
            values = diagnostics.get(stratum)
            values = values if isinstance(values, Mapping) else {}
            for field in _REGIME_SELECTIVITY_ADDITIVE_FIELDS:
                if field in _REGIME_CONFUSION_FIELDS:
                    target, prediction = field.removeprefix(
                        "target_"
                    ).removesuffix("_rows").split("_predicted_", 1)
                    confusion = values.get("confusion")
                    confusion = (
                        confusion if isinstance(confusion, Mapping) else {}
                    )
                    target_values = confusion.get(target)
                    target_values = (
                        target_values
                        if isinstance(target_values, Mapping) else {}
                    )
                    value = target_values.get(prediction, 0.0)
                else:
                    value = values.get(field, 0.0)
                update_metrics[
                    f"regime_selectivity_{stratum}_{field}"
                ].append(float(value or 0.0))
    return _regime_selectivity_episode_diagnostic(update_metrics)


def _regime_channel_episode_diagnostic(
    update_metrics: Mapping[str, Sequence[float]],
    channel_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Reduce named teacher-head errors with one additive update ledger."""
    result: dict[str, dict[str, float | int]] = {}
    for channel in channel_names:
        prefix = f"regime_teacher_channel_{channel}_"
        additive = {
            field: float(sum(update_metrics.get(prefix + field, ())))
            for field in _REGIME_CHANNEL_ADDITIVE_FIELDS
        }
        rows = int(round(additive["rows"]))
        target_mean = (
            additive["target_probability_sum"] / rows if rows else 0.0
        )
        model_mean = (
            additive["model_probability_sum"] / rows if rows else 0.0
        )
        result[channel] = {
            "rows": rows,
            **{
                field: additive[field]
                for field in _REGIME_CHANNEL_ADDITIVE_FIELDS
                if field != "rows"
            },
            "target_probability_mean": target_mean,
            "model_probability_mean": model_mean,
            "mean_error": model_mean - target_mean,
            "mean_absolute_error": (
                additive["absolute_error_sum"] / rows if rows else 0.0
            ),
            "root_mean_squared_error": (
                math.sqrt(additive["squared_error_sum"] / rows)
                if rows else 0.0
            ),
        }
    return result


def _regime_channel_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    channel_names = sorted({
        str(channel)
        for row in rows
        for channel in (
            row.get("regime_teacher_channels", {}).keys()
            if isinstance(row.get("regime_teacher_channels"), Mapping)
            else ()
        )
    })
    update_metrics: dict[str, list[float]] = {
        f"regime_teacher_channel_{channel}_{field}": []
        for channel in channel_names
        for field in _REGIME_CHANNEL_ADDITIVE_FIELDS
    }
    for row in rows:
        diagnostics = row.get("regime_teacher_channels")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        for channel in channel_names:
            values = diagnostics.get(channel)
            values = values if isinstance(values, Mapping) else {}
            for field in _REGIME_CHANNEL_ADDITIVE_FIELDS:
                update_metrics[
                    f"regime_teacher_channel_{channel}_{field}"
                ].append(float(values.get(field, 0.0) or 0.0))
    return _regime_channel_episode_diagnostic(update_metrics, channel_names)


def _entry_balance_diagnostic(
    update_metrics: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    additive_by_action: dict[str, dict[str, float]] = {}
    for action in _ENTRY_BALANCE_ACTION_NAMES:
        prefix = f"entry_balance_{action}_"
        additive_by_action[action] = {
            field: float(sum(update_metrics.get(prefix + field, ())))
            for field in _ENTRY_BALANCE_ADDITIVE_FIELDS
        }
    total_mass = sum(
        values["weighted_mass"] for values in additive_by_action.values()
    )
    total_weighted_ce = sum(
        values["weighted_ce_sum"] for values in additive_by_action.values()
    )
    result: dict[str, dict[str, float | int]] = {}
    for action, additive in additive_by_action.items():
        rows = int(round(additive["rows"]))
        weights = tuple(update_metrics.get(
            f"entry_balance_{action}_configured_weight", ()
        ))
        configured_weight = float(weights[-1]) if weights else 0.0
        result[action] = {
            "rows": rows,
            "configured_weight": configured_weight,
            "weighted_mass": additive["weighted_mass"],
            "weighted_mass_fraction": (
                additive["weighted_mass"] / total_mass if total_mass else 0.0
            ),
            "unweighted_ce_sum": additive["unweighted_ce_sum"],
            "unweighted_ce_mean": (
                additive["unweighted_ce_sum"] / rows if rows else 0.0
            ),
            "weighted_ce_sum": additive["weighted_ce_sum"],
            "weighted_loss_contribution": (
                additive["weighted_ce_sum"] / total_mass
                if total_mass else 0.0
            ),
            "weighted_ce_fraction": (
                additive["weighted_ce_sum"] / total_weighted_ce
                if total_weighted_ce else 0.0
            ),
        }
    return result


def _entry_balance_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    update_metrics: dict[str, list[float]] = {
        f"entry_balance_{action}_{field}": []
        for action in _ENTRY_BALANCE_ACTION_NAMES
        for field in (
            *_ENTRY_BALANCE_ADDITIVE_FIELDS,
            "configured_weight",
        )
    }
    for row in rows:
        diagnostics = row.get("entry_action_balance")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        for action in _ENTRY_BALANCE_ACTION_NAMES:
            values = diagnostics.get(action)
            values = values if isinstance(values, Mapping) else {}
            for field in (
                *_ENTRY_BALANCE_ADDITIVE_FIELDS,
                "configured_weight",
            ):
                update_metrics[f"entry_balance_{action}_{field}"].append(
                    float(values.get(field, 0.0) or 0.0)
                )
    return _entry_balance_diagnostic(update_metrics)


def _regime_entry_conflict_diagnostic(
    update_metrics: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for side in ("long", "short"):
        prefix = f"regime_entry_conflict_{side}_"
        additive = {
            field: float(sum(update_metrics.get(prefix + field, ())))
            for field in _REGIME_ENTRY_CONFLICT_FIELDS
        }
        rows = int(round(additive["rows"]))
        result[side] = {
            "rows": rows,
            **{
                field: additive[field]
                for field in _REGIME_ENTRY_CONFLICT_FIELDS
                if field != "rows"
            },
            "target_wait_probability_mean": (
                additive["target_wait_probability_sum"] / rows
                if rows else 0.0
            ),
            "target_declared_side_probability_mean": (
                additive["target_declared_side_probability_sum"] / rows
                if rows else 0.0
            ),
            "model_wait_probability_mean": (
                additive["model_wait_probability_sum"] / rows
                if rows else 0.0
            ),
            "soft_wait_disagreement_rate": (
                additive["soft_wait_disagreement_rows"] / rows
                if rows else 0.0
            ),
        }
    return result


def _regime_entry_conflict_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    update_metrics: dict[str, list[float]] = {
        f"regime_entry_conflict_{side}_{field}": []
        for side in ("long", "short")
        for field in _REGIME_ENTRY_CONFLICT_FIELDS
    }
    for row in rows:
        diagnostics = row.get("regime_entry_conflict")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        for side in ("long", "short"):
            values = diagnostics.get(side)
            values = values if isinstance(values, Mapping) else {}
            for field in _REGIME_ENTRY_CONFLICT_FIELDS:
                update_metrics[f"regime_entry_conflict_{side}_{field}"].append(
                    float(values.get(field, 0.0) or 0.0)
                )
    return _regime_entry_conflict_diagnostic(update_metrics)


def _regime_trade_economics(
    receipts: Sequence[Mapping[str, object]],
    visible_context: Mapping[int, Mapping[str, object]],
    *,
    episode_outcome: str,
) -> dict[str, object]:
    """Join closed trades to teacher context already visible at entry time."""
    grouped: dict[tuple[str, str, str, str], dict[str, float | int]] = {}
    unattributed = {"long": 0, "short": 0}
    attributed = 0
    for receipt in receipts:
        side = str(receipt.get("side", "unknown"))
        source_index = int(receipt["source_decision_index"])
        context = visible_context.get(source_index)
        if context is None:
            unattributed[side] = unattributed.get(side, 0) + 1
            continue
        attributed += 1
        teacher = context["teacher"]
        channels = context["channels"]
        assert isinstance(teacher, np.ndarray)
        assert isinstance(channels, tuple)
        channel_index = {channel: index for index, channel in enumerate(channels)}
        chop = float(teacher[channel_index["chop_no_trend_probability"]])
        chop_end_transition = float(
            teacher[channel_index["chop_end_transition_probability"]]
        )
        expansion_trend = float(
            teacher[channel_index["expansion_trend_probability"]]
        )
        static_regime = (
            "dominant_chop"
            if chop > max(chop_end_transition, expansion_trend)
            else "nonchop"
        )
        headroom = float(context["headroom_fraction"])
        headroom_stratum = (
            "low_headroom_le_0_25"
            if headroom <= 0.25 else
            "safe_headroom_ge_0_75"
            if headroom >= 0.75 else
            "mid_headroom_gt_0_25_lt_0_75"
        )
        key = (side, static_regime, headroom_stratum, episode_outcome)
        values = grouped.setdefault(key, {
            "trades": 0,
            "wins": 0,
            "realized_r_sum": 0.0,
            "mfe_r_sum": 0.0,
            "mae_r_sum": 0.0,
            "initial_stop_count": 0,
            "regime_channel_probability_sums": {
                channel: 0.0
                for channel in REGIME_TEACHER_CHANNELS
                if channel in channel_index
            },
        })
        realized_r = float(receipt.get("realized_r", 0.0))
        values["trades"] = int(values["trades"]) + 1
        values["wins"] = int(values["wins"]) + int(realized_r > 0.0)
        values["realized_r_sum"] = float(values["realized_r_sum"]) + realized_r
        values["mfe_r_sum"] = float(values["mfe_r_sum"]) + float(
            receipt.get("mfe_r", 0.0)
        )
        values["mae_r_sum"] = float(values["mae_r_sum"]) + float(
            receipt.get("mae_r", 0.0)
        )
        values["initial_stop_count"] = int(values["initial_stop_count"]) + int(
            receipt.get("exit_reason") == "initial_stop"
        )
        channel_sums = values["regime_channel_probability_sums"]
        assert isinstance(channel_sums, dict)
        for channel in channel_sums:
            channel_sums[channel] += float(teacher[channel_index[channel]])
    groups = []
    for key in sorted(grouped):
        side, static_regime, headroom_stratum, outcome = key
        values = grouped[key]
        trades = int(values["trades"])
        wins = int(values["wins"])
        groups.append({
            "side": side,
            "static_regime": static_regime,
            "headroom_stratum": headroom_stratum,
            "episode_outcome": outcome,
            "trades": trades,
            "wins": wins,
            "win_rate": wins / trades if trades else 0.0,
            "realized_r_sum": float(values["realized_r_sum"]),
            "realized_r_mean": float(values["realized_r_sum"]) / trades,
            "mfe_r_sum": float(values["mfe_r_sum"]),
            "mfe_r_mean": float(values["mfe_r_sum"]) / trades,
            "mae_r_sum": float(values["mae_r_sum"]),
            "mae_r_mean": float(values["mae_r_sum"]) / trades,
            "initial_stop_count": int(values["initial_stop_count"]),
            "regime_channel_probability_sums": dict(
                values["regime_channel_probability_sums"]
            ),
            "regime_channel_probability_means": {
                channel: float(total) / trades
                for channel, total in dict(
                    values["regime_channel_probability_sums"]
                ).items()
            },
        })
    total = len(receipts)
    return {
        "total_trades": total,
        "attributed_trades": attributed,
        "unattributed_trades": total - attributed,
        "attribution_coverage": attributed / total if total else 0.0,
        "unattributed_by_side": unattributed,
        "groups": groups,
    }


def _validation_closed_trade_economics(
    receipts: Sequence[Mapping[str, object]],
    *,
    reported_trade_count: int,
    episode_outcome: str,
) -> dict[str, object]:
    """Aggregate policy-only validation economics without teacher lookups."""
    if reported_trade_count < 0:
        raise ValueError("reported validation trade count must be nonnegative")
    if len(receipts) > reported_trade_count:
        raise ValueError("closed trade receipts exceed reported trade count")
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for receipt in receipts:
        side = str(receipt.get("side", "unknown"))
        if side not in {"long", "short"}:
            side = "unknown"
        raw_headroom = receipt.get("entry_mll_headroom")
        if (
            raw_headroom is None
            or isinstance(raw_headroom, bool)
            or not np.isfinite(float(raw_headroom))
        ):
            headroom_stratum = "unavailable"
        else:
            headroom = float(raw_headroom)
            headroom_stratum = (
                "critical_le_300"
                if headroom <= 300.0 else
                "safe_ge_500"
                if headroom >= 500.0 else
                "constrained_gt_300_lt_500"
            )
        key = (side, episode_outcome, headroom_stratum)
        values = grouped.setdefault(key, {
            "trades": 0,
            "wins": 0,
            "realized_r_sum": 0.0,
            "mfe_r_sum": 0.0,
            "mae_r_sum": 0.0,
            "hold_bars_sum": 0,
            "exit_reason_counts": {},
        })
        realized_r = float(receipt.get("realized_r", 0.0))
        values["trades"] = int(values["trades"]) + 1
        values["wins"] = int(values["wins"]) + int(realized_r > 0.0)
        values["realized_r_sum"] = (
            float(values["realized_r_sum"]) + realized_r
        )
        values["mfe_r_sum"] = float(values["mfe_r_sum"]) + float(
            receipt.get("mfe_r", 0.0)
        )
        values["mae_r_sum"] = float(values["mae_r_sum"]) + float(
            receipt.get("mae_r", 0.0)
        )
        values["hold_bars_sum"] = int(values["hold_bars_sum"]) + int(
            receipt.get("hold_bars", 0)
        )
        exit_reasons = values["exit_reason_counts"]
        assert isinstance(exit_reasons, dict)
        exit_reason = str(receipt.get("exit_reason", "unknown"))
        exit_reasons[exit_reason] = int(exit_reasons.get(exit_reason, 0)) + 1
    groups = []
    for side, outcome, headroom_stratum in sorted(grouped):
        values = grouped[(side, outcome, headroom_stratum)]
        trades = int(values["trades"])
        wins = int(values["wins"])
        groups.append({
            "side": side,
            "episode_outcome": outcome,
            "entry_headroom_stratum": headroom_stratum,
            "trades": trades,
            "wins": wins,
            "win_rate": wins / trades,
            "realized_r_sum": float(values["realized_r_sum"]),
            "realized_r_mean": float(values["realized_r_sum"]) / trades,
            "mfe_r_sum": float(values["mfe_r_sum"]),
            "mfe_r_mean": float(values["mfe_r_sum"]) / trades,
            "mae_r_sum": float(values["mae_r_sum"]),
            "mae_r_mean": float(values["mae_r_sum"]) / trades,
            "hold_bars_sum": int(values["hold_bars_sum"]),
            "hold_bars_mean": float(values["hold_bars_sum"]) / trades,
            "exit_reason_counts": dict(values["exit_reason_counts"]),
        })
    receipt_count = len(receipts)
    return {
        "reported_trade_count": reported_trade_count,
        "receipt_trade_count": receipt_count,
        "unattributed_trade_count": reported_trade_count - receipt_count,
        "receipt_coverage": (
            receipt_count / reported_trade_count
            if reported_trade_count else 0.0
        ),
        "groups": groups,
    }


def _regime_trade_economics_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    total = attributed = unattributed = 0
    unattributed_by_side: dict[str, int] = {"long": 0, "short": 0}
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        raw = row.get("regime_trade_economics")
        evidence = raw if isinstance(raw, Mapping) else {}
        total += int(evidence.get("total_trades", 0))
        attributed += int(evidence.get("attributed_trades", 0))
        unattributed += int(evidence.get("unattributed_trades", 0))
        side_counts = evidence.get("unattributed_by_side")
        side_counts = side_counts if isinstance(side_counts, Mapping) else {}
        for side, count in side_counts.items():
            unattributed_by_side[str(side)] = (
                unattributed_by_side.get(str(side), 0) + int(count)
            )
        raw_groups = evidence.get("groups")
        groups = raw_groups if isinstance(raw_groups, list) else []
        for item in groups:
            if not isinstance(item, Mapping):
                continue
            key = (
                str(item["side"]),
                str(item["static_regime"]),
                str(item["headroom_stratum"]),
                str(item["episode_outcome"]),
            )
            values = grouped.setdefault(key, {
                "trades": 0,
                "wins": 0,
                "realized_r_sum": 0.0,
                "mfe_r_sum": 0.0,
                "mae_r_sum": 0.0,
                "initial_stop_count": 0,
                "regime_channel_probability_sums": {},
            })
            for field in ("trades", "wins", "initial_stop_count"):
                values[field] = int(values[field]) + int(item.get(field, 0))
            for field in ("realized_r_sum", "mfe_r_sum", "mae_r_sum"):
                values[field] = float(values[field]) + float(item.get(field, 0.0))
            channel_values = item.get("regime_channel_probability_sums")
            channel_values = (
                channel_values if isinstance(channel_values, Mapping) else {}
            )
            channel_sums = values["regime_channel_probability_sums"]
            assert isinstance(channel_sums, dict)
            for channel, value in channel_values.items():
                channel_sums[str(channel)] = (
                    channel_sums.get(str(channel), 0.0) + float(value)
                )
    groups = []
    for key in sorted(grouped):
        side, static_regime, headroom_stratum, outcome = key
        values = grouped[key]
        trades = int(values["trades"])
        wins = int(values["wins"])
        channel_sums = dict(values["regime_channel_probability_sums"])
        groups.append({
            "side": side,
            "static_regime": static_regime,
            "headroom_stratum": headroom_stratum,
            "episode_outcome": outcome,
            "trades": trades,
            "wins": wins,
            "win_rate": wins / trades if trades else 0.0,
            "realized_r_sum": float(values["realized_r_sum"]),
            "realized_r_mean": float(values["realized_r_sum"]) / trades,
            "mfe_r_sum": float(values["mfe_r_sum"]),
            "mfe_r_mean": float(values["mfe_r_sum"]) / trades,
            "mae_r_sum": float(values["mae_r_sum"]),
            "mae_r_mean": float(values["mae_r_sum"]) / trades,
            "initial_stop_count": int(values["initial_stop_count"]),
            "regime_channel_probability_sums": channel_sums,
            "regime_channel_probability_means": {
                channel: value / trades
                for channel, value in channel_sums.items()
            },
        })
    return {
        "total_trades": total,
        "attributed_trades": attributed,
        "unattributed_trades": unattributed,
        "attribution_coverage": attributed / total if total else 0.0,
        "unattributed_by_side": unattributed_by_side,
        "groups": groups,
    }


def _paired_a_plus_diagnostic(
    update_metrics: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Reduce learned A+ contrasts by side and three-state Regime."""
    raw_losses = tuple(update_metrics.get(
        "regime_selectivity_paired_a_plus_loss", ()
    ))
    loss_sum = float(sum(raw_losses)) + float(sum(update_metrics.get(
        "regime_selectivity_paired_a_plus_loss_sum", ()
    )))
    update_count = float(len(raw_losses)) + float(sum(update_metrics.get(
        "regime_selectivity_paired_a_plus_update_count", ()
    )))
    pair_mass = float(sum(update_metrics.get(
        "regime_selectivity_paired_a_plus_pair_mass", ()
    )))
    good_sum = float(sum(update_metrics.get(
        "regime_selectivity_paired_a_plus_good_advantage_sum", ()
    )))
    bad_sum = float(sum(update_metrics.get(
        "regime_selectivity_paired_a_plus_bad_advantage_sum", ()
    )))
    groups: dict[str, dict[str, float]] = {}
    for side in _PAIRED_A_PLUS_SIDES:
        for regime in _PAIRED_A_PLUS_REGIMES:
            name = f"{side}_{regime}"
            prefix = f"regime_selectivity_paired_a_plus_{name}_"
            values = {
                field: float(sum(update_metrics.get(prefix + field, ())))
                for field in _PAIRED_A_PLUS_GROUP_FIELDS
            }
            mass = values["pair_mass"]
            groups[name] = {
                **values,
                "loss_mean": values["loss_sum"] / mass if mass else 0.0,
                "good_advantage_mean": (
                    values["good_advantage_sum"] / mass if mass else 0.0
                ),
                "bad_advantage_mean": (
                    values["bad_advantage_sum"] / mass if mass else 0.0
                ),
            }
    sides: dict[str, dict[str, float]] = {}
    for side in _PAIRED_A_PLUS_SIDES:
        side_groups = [
            groups[f"{side}_{regime}"] for regime in _PAIRED_A_PLUS_REGIMES
        ]
        side_mass = sum(group["pair_mass"] for group in side_groups)
        side_good_sum = sum(
            group["good_advantage_sum"] for group in side_groups
        )
        side_bad_sum = sum(
            group["bad_advantage_sum"] for group in side_groups
        )
        direct_prefix = f"regime_selectivity_paired_a_plus_{side}_"
        direct_pair_mass = float(sum(update_metrics.get(
            direct_prefix + "pair_mass", ()
        )))
        direct = {
            field: float(sum(update_metrics.get(direct_prefix + field, ())))
            for field in (
                "pair_count",
                "pair_mass",
                "loss_sum",
                "good_advantage_sum",
                "bad_advantage_sum",
                "winner_population_weight_sum",
                "failure_population_weight_sum",
            )
        }
        if direct_pair_mass:
            side_mass = direct_pair_mass
            side_good_sum = direct["good_advantage_sum"]
            side_bad_sum = direct["bad_advantage_sum"]
        sides[side] = {
            "pair_count": (
                direct["pair_count"]
                if direct_pair_mass
                else sum(group["pair_count"] for group in side_groups)
            ),
            "pair_mass": side_mass,
            "loss_sum": (
                direct["loss_sum"]
                if direct_pair_mass
                else sum(group["loss_sum"] for group in side_groups)
            ),
            "good_advantage_sum": side_good_sum,
            "good_advantage_mean": (
                side_good_sum / side_mass if side_mass else 0.0
            ),
            "bad_advantage_sum": side_bad_sum,
            "bad_advantage_mean": (
                side_bad_sum / side_mass if side_mass else 0.0
            ),
            "winner_population_weight_mean": (
                direct["winner_population_weight_sum"] / side_mass
                if direct_pair_mass else 1.0
            ),
            "failure_population_weight_mean": (
                direct["failure_population_weight_sum"] / side_mass
                if direct_pair_mass else 1.0
            ),
        }
    return {
        "loss_sum": loss_sum,
        "loss_mean": loss_sum / update_count if update_count else 0.0,
        "update_count": update_count,
        "active_groups": float(sum(update_metrics.get(
            "regime_selectivity_paired_a_plus_active_groups", ()
        ))),
        "pair_count": float(sum(update_metrics.get(
            "regime_selectivity_paired_a_plus_pair_count", ()
        ))),
        "pair_mass": pair_mass,
        "good_advantage_sum": good_sum,
        "good_advantage_mean": good_sum / pair_mass if pair_mass else 0.0,
        "bad_advantage_sum": bad_sum,
        "bad_advantage_mean": bad_sum / pair_mass if pair_mass else 0.0,
        "sides": sides,
        "groups": groups,
    }


def _persistent_regime_selectivity_diagnostic(
    update_metrics: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Reduce transition-aware WAIT evidence from additive optimizer metrics."""
    result: dict[str, dict[str, float]] = {}
    for stratum in _PERSISTENT_REGIME_SELECTIVITY_STRATA:
        values = {
            field: float(sum(update_metrics.get(
                f"regime_selectivity_{stratum}_{field}", ()
            )))
            for field in _PERSISTENT_REGIME_SELECTIVITY_ADDITIVE_FIELDS
        }
        rows = values["rows"]
        result[stratum] = {
            **values,
            "weight_mean": values["weight_sum"] / rows if rows else 0.0,
            "model_wait_probability_mean": (
                values["model_wait_probability_sum"] / rows
                if rows else 0.0
            ),
        }
    for side in _TRANSITION_POSITIVE_SIDE_FIELDS:
        prefix = f"regime_selectivity_transition_positive_{side}_"
        rows = float(sum(update_metrics.get(prefix + "rows", ())))
        probability_sum = float(sum(update_metrics.get(
            prefix + "declared_side_probability_sum", ()
        )))
        result[f"transition_positive_{side}"] = {
            "rows": rows,
            "declared_side_probability_sum": probability_sum,
            "declared_side_probability_mean": (
                probability_sum / rows if rows else 0.0
            ),
        }
    raw_losses = tuple(update_metrics.get(
        "regime_selectivity_association_loss", ()
    ))
    loss_sum = float(sum(raw_losses)) + float(sum(update_metrics.get(
        "regime_selectivity_association_loss_sum", ()
    )))
    update_count = float(len(raw_losses)) + float(sum(update_metrics.get(
        "regime_selectivity_association_update_count", ()
    )))
    association = {
        "loss_sum": loss_sum,
        "loss_mean": loss_sum / update_count if update_count else 0.0,
        "update_count": update_count,
        "active_updates": float(sum(update_metrics.get(
            "regime_selectivity_association_active", ()
        ))) + float(sum(update_metrics.get(
            "regime_selectivity_association_active_updates", ()
        ))),
        "skipped_updates": float(sum(update_metrics.get(
            "regime_selectivity_association_skipped", ()
        ))) + float(sum(update_metrics.get(
            "regime_selectivity_association_skipped_updates", ()
        ))),
    }
    for cohort in _REGIME_ASSOCIATION_COHORTS:
        prefix = f"regime_selectivity_association_{cohort}_"
        rows = float(sum(update_metrics.get(prefix + "rows", ())))
        probability_sum = float(sum(update_metrics.get(
            prefix + "model_wait_probability_sum", ()
        )))
        association.update({
            f"{cohort}_rows": rows,
            f"{cohort}_model_wait_probability_sum": probability_sum,
            f"{cohort}_model_wait_probability_mean": (
                probability_sum / rows if rows else 0.0
            ),
        })
    transition_positive_means = [
        association[
            f"transition_positive_{side}_model_wait_probability_mean"
        ]
        for side in _TRANSITION_POSITIVE_SIDE_FIELDS
        if association[f"transition_positive_{side}_rows"] > 0.0
    ]
    association["dead_wait_minus_transition_positive_model_wait"] = (
        association["dead_wait_model_wait_probability_mean"]
        - sum(transition_positive_means) / len(transition_positive_means)
        if association["dead_wait_rows"] > 0.0 and transition_positive_means
        else 0.0
    )
    result["association"] = association
    side_losses = tuple(update_metrics.get(
        "regime_selectivity_side_conditioned_loss", ()
    ))
    active_sides = tuple(update_metrics.get(
        "regime_selectivity_side_conditioned_active_sides", ()
    ))
    side_loss_sum = float(sum(side_losses)) + float(sum(update_metrics.get(
        "regime_selectivity_side_conditioned_loss_sum", ()
    )))
    side_update_count = float(len(side_losses)) + float(sum(update_metrics.get(
        "regime_selectivity_side_conditioned_update_count", ()
    )))
    active_sides_sum = float(sum(active_sides)) + float(sum(update_metrics.get(
        "regime_selectivity_side_conditioned_active_sides_sum", ()
    )))
    both_sides_active_updates = float(sum(
        float(value) >= 2.0 for value in active_sides
    )) + float(sum(update_metrics.get(
        "regime_selectivity_side_conditioned_both_sides_active_updates", ()
    )))
    result["side_conditioned"] = {
        "loss_sum": side_loss_sum,
        "loss_mean": (
            side_loss_sum / side_update_count if side_update_count else 0.0
        ),
        "update_count": side_update_count,
        "active_sides_sum": active_sides_sum,
        "both_sides_active_updates": both_sides_active_updates,
    }
    result["paired_a_plus"] = _paired_a_plus_diagnostic(update_metrics)
    return result


def _persistent_regime_selectivity_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate continuous persistent-chop evidence without thresholds."""
    update_metrics: dict[str, list[float]] = {
        f"regime_selectivity_{stratum}_{field}": []
        for stratum in _PERSISTENT_REGIME_SELECTIVITY_STRATA
        for field in _PERSISTENT_REGIME_SELECTIVITY_ADDITIVE_FIELDS
    }
    for side in _TRANSITION_POSITIVE_SIDE_FIELDS:
        update_metrics.update({
            f"regime_selectivity_transition_positive_{side}_rows": [],
            f"regime_selectivity_transition_positive_{side}_"
            "declared_side_probability_sum": [],
        })
    update_metrics.update({
        f"regime_selectivity_association_{field}": []
        for field in (
            "loss_sum",
            "update_count",
            "active_updates",
            "skipped_updates",
            *(
                f"{cohort}_{field}"
                for cohort in _REGIME_ASSOCIATION_COHORTS
                for field in _REGIME_ASSOCIATION_ADDITIVE_FIELDS
            ),
        )
    })
    update_metrics.update({
        f"regime_selectivity_side_conditioned_{field}": []
        for field in (
            "loss_sum",
            "update_count",
            "active_sides_sum",
            "both_sides_active_updates",
        )
    })
    update_metrics.update({
        f"regime_selectivity_paired_a_plus_{field}": []
        for field in (
            "loss_sum",
            "update_count",
            "active_groups",
            "pair_count",
            "pair_mass",
            "good_advantage_sum",
            "bad_advantage_sum",
        )
    })
    update_metrics.update({
        f"regime_selectivity_paired_a_plus_{side}_{regime}_{field}": []
        for side in _PAIRED_A_PLUS_SIDES
        for regime in _PAIRED_A_PLUS_REGIMES
        for field in _PAIRED_A_PLUS_GROUP_FIELDS
    })
    update_metrics.update({
        f"regime_selectivity_paired_a_plus_{side}_{field}": []
        for side in _PAIRED_A_PLUS_SIDES
        for field in (
            "pair_count",
            "pair_mass",
            "loss_sum",
            "good_advantage_sum",
            "bad_advantage_sum",
            "winner_population_weight_sum",
            "failure_population_weight_sum",
        )
    })
    for row in rows:
        diagnostics = row.get("persistent_regime_selectivity")
        diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
        for stratum in _PERSISTENT_REGIME_SELECTIVITY_STRATA:
            values = diagnostics.get(stratum)
            values = values if isinstance(values, Mapping) else {}
            for field in _PERSISTENT_REGIME_SELECTIVITY_ADDITIVE_FIELDS:
                update_metrics[
                    f"regime_selectivity_{stratum}_{field}"
                ].append(float(values.get(field, 0.0) or 0.0))
        for side in _TRANSITION_POSITIVE_SIDE_FIELDS:
            values = diagnostics.get(f"transition_positive_{side}")
            values = values if isinstance(values, Mapping) else {}
            for field in ("rows", "declared_side_probability_sum"):
                update_metrics[
                    f"regime_selectivity_transition_positive_{side}_{field}"
                ].append(float(values.get(field, 0.0) or 0.0))
        association = diagnostics.get("association")
        association = association if isinstance(association, Mapping) else {}
        for field in (
            "loss_sum",
            "update_count",
            "active_updates",
            "skipped_updates",
            *(
                f"{cohort}_{field}"
                for cohort in _REGIME_ASSOCIATION_COHORTS
                for field in _REGIME_ASSOCIATION_ADDITIVE_FIELDS
            ),
        ):
            update_metrics[f"regime_selectivity_association_{field}"].append(
                float(association.get(field, 0.0) or 0.0)
            )
        side_conditioned = diagnostics.get("side_conditioned")
        side_conditioned = (
            side_conditioned
            if isinstance(side_conditioned, Mapping)
            else {}
        )
        for field in (
            "loss_sum",
            "update_count",
            "active_sides_sum",
            "both_sides_active_updates",
        ):
            update_metrics[
                f"regime_selectivity_side_conditioned_{field}"
            ].append(float(side_conditioned.get(field, 0.0) or 0.0))
        paired = diagnostics.get("paired_a_plus")
        paired = paired if isinstance(paired, Mapping) else {}
        for field in (
            "loss_sum",
            "update_count",
            "active_groups",
            "pair_count",
            "pair_mass",
            "good_advantage_sum",
            "bad_advantage_sum",
        ):
            update_metrics[
                f"regime_selectivity_paired_a_plus_{field}"
            ].append(float(paired.get(field, 0.0) or 0.0))
        groups = paired.get("groups")
        groups = groups if isinstance(groups, Mapping) else {}
        sides = paired.get("sides")
        sides = sides if isinstance(sides, Mapping) else {}
        for side in _PAIRED_A_PLUS_SIDES:
            values = sides.get(side)
            values = values if isinstance(values, Mapping) else {}
            pair_mass = float(values.get("pair_mass", 0.0) or 0.0)
            direct = {
                field: float(values.get(field, 0.0) or 0.0)
                for field in (
                    "pair_count",
                    "pair_mass",
                    "loss_sum",
                    "good_advantage_sum",
                    "bad_advantage_sum",
                )
            }
            direct["winner_population_weight_sum"] = pair_mass * float(
                values.get("winner_population_weight_mean", 0.0) or 0.0
            )
            direct["failure_population_weight_sum"] = pair_mass * float(
                values.get("failure_population_weight_mean", 0.0) or 0.0
            )
            for field, value in direct.items():
                update_metrics[
                    f"regime_selectivity_paired_a_plus_{side}_{field}"
                ].append(value)
        for side in _PAIRED_A_PLUS_SIDES:
            for regime in _PAIRED_A_PLUS_REGIMES:
                name = f"{side}_{regime}"
                values = groups.get(name)
                values = values if isinstance(values, Mapping) else {}
                for field in _PAIRED_A_PLUS_GROUP_FIELDS:
                    update_metrics[
                        "regime_selectivity_paired_a_plus_"
                        f"{name}_{field}"
                    ].append(float(values.get(field, 0.0) or 0.0))
    return _persistent_regime_selectivity_diagnostic(update_metrics)


def _regime_selectivity_evaluation_metrics(
    diagnostics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, float]:
    """Expose row evidence and mechanism deltas to campaign selection gates."""
    metrics = {
        f"regime_selectivity_{stratum}_{field}": float(
            diagnostics.get(stratum, {}).get(field, 0.0)
        )
        for stratum in _REGIME_SELECTIVITY_STRATA
        for field in (
            "rows",
            "target_wait_probability_mean",
            "model_wait_probability_mean",
            "wait_mean_absolute_error",
            "target_action_probability_mean",
            "model_target_action_probability_mean",
            "target_action_mean_absolute_error",
            "greedy_wait_rate",
            "declared_side_probability_sum",
            "declared_side_probability_mean",
            "greedy_entry_rows",
            "greedy_entry_rate",
            "correct_rows",
            "accuracy",
        )
    }
    for stratum in _REGIME_SELECTIVITY_STRATA:
        confusion = diagnostics.get(stratum, {}).get("confusion", {})
        for target in _REGIME_ACTION_NAMES:
            target_values = confusion.get(target, {})
            for prediction in _REGIME_ACTION_NAMES:
                metrics[
                    f"regime_selectivity_{stratum}_target_{target}_"
                    f"predicted_{prediction}_rows"
                ] = float(target_values.get(prediction, 0.0))
    metrics["regime_selectivity_chop_minus_nonchop_target_wait"] = (
        metrics[
            "regime_selectivity_dominant_chop_target_wait_probability_mean"
        ]
        - metrics[
            "regime_selectivity_nonchop_target_wait_probability_mean"
        ]
    )
    metrics["regime_selectivity_chop_minus_nonchop_model_wait"] = (
        metrics[
            "regime_selectivity_dominant_chop_model_wait_probability_mean"
        ]
        - metrics[
            "regime_selectivity_nonchop_model_wait_probability_mean"
        ]
    )
    metrics["regime_selectivity_low_minus_safe_target_wait"] = (
        metrics[
            "regime_selectivity_low_headroom_le_0_25_"
            "target_wait_probability_mean"
        ]
        - metrics[
            "regime_selectivity_safe_headroom_ge_0_75_"
            "target_wait_probability_mean"
        ]
    )
    return metrics


def _persistent_regime_selectivity_evaluation_metrics(
    diagnostics: Mapping[str, object],
) -> dict[str, float]:
    metrics = {
        f"regime_selectivity_{stratum}_{field}": float(
            diagnostics.get(stratum, {}).get(field, 0.0)
        )
        for stratum in _PERSISTENT_REGIME_SELECTIVITY_STRATA
        for field in (
            "rows",
            "weight_sum",
            "weight_mean",
            "model_wait_probability_sum",
            "model_wait_probability_mean",
        )
    }
    for side in _TRANSITION_POSITIVE_SIDE_FIELDS:
        for field in (
            "rows",
            "declared_side_probability_sum",
            "declared_side_probability_mean",
        ):
            metrics[
                f"regime_selectivity_transition_positive_{side}_{field}"
            ] = float(
                diagnostics.get(f"transition_positive_{side}", {}).get(
                    field, 0.0
                )
            )
    metrics[
        "regime_selectivity_dead_wait_minus_"
        "transition_ready_wait_model_wait"
    ] = (
        metrics[
            "regime_selectivity_persistent_dead_chop_"
            "model_wait_probability_mean"
        ]
        - metrics[
            "regime_selectivity_transition_ready_"
            "model_wait_probability_mean"
        ]
    )
    association = diagnostics.get("association", {})
    for field in (
        "loss_sum",
        "loss_mean",
        "update_count",
        "active_updates",
        "skipped_updates",
        *(
            f"{cohort}_{field}"
            for cohort in _REGIME_ASSOCIATION_COHORTS
            for field in (
                *_REGIME_ASSOCIATION_ADDITIVE_FIELDS,
                "model_wait_probability_mean",
            )
        ),
    ):
        metrics[f"regime_selectivity_association_{field}"] = float(
            association.get(field, 0.0)
        )
    metrics["regime_selectivity_association_loss"] = metrics[
        "regime_selectivity_association_loss_mean"
    ]
    metrics[
        "regime_selectivity_dead_wait_minus_"
        "transition_positive_model_wait"
    ] = float(association.get(
        "dead_wait_minus_transition_positive_model_wait", 0.0
    ))
    side_conditioned = diagnostics.get("side_conditioned", {})
    side_conditioned = (
        side_conditioned if isinstance(side_conditioned, Mapping) else {}
    )
    for field in (
        "loss_sum",
        "loss_mean",
        "update_count",
        "active_sides_sum",
        "both_sides_active_updates",
    ):
        metrics[f"regime_selectivity_side_conditioned_{field}"] = float(
            side_conditioned.get(field, 0.0)
        )
    paired = diagnostics.get("paired_a_plus", {})
    paired = paired if isinstance(paired, Mapping) else {}
    for field in (
        "loss_sum",
        "loss_mean",
        "update_count",
        "active_groups",
        "pair_count",
        "pair_mass",
        "good_advantage_sum",
        "good_advantage_mean",
        "bad_advantage_sum",
        "bad_advantage_mean",
    ):
        metrics[f"regime_selectivity_paired_a_plus_{field}"] = float(
            paired.get(field, 0.0)
        )
    groups = paired.get("groups", {})
    groups = groups if isinstance(groups, Mapping) else {}
    sides = paired.get("sides", {})
    sides = sides if isinstance(sides, Mapping) else {}
    for side in _PAIRED_A_PLUS_SIDES:
        values = sides.get(side, {})
        values = values if isinstance(values, Mapping) else {}
        for field in (
            "pair_count",
            "pair_mass",
            "loss_sum",
            "good_advantage_sum",
            "good_advantage_mean",
            "bad_advantage_sum",
            "bad_advantage_mean",
        ):
            metrics[
                f"regime_selectivity_paired_a_plus_{side}_{field}"
            ] = float(values.get(field, 0.0))
    for side in _PAIRED_A_PLUS_SIDES:
        for regime in _PAIRED_A_PLUS_REGIMES:
            name = f"{side}_{regime}"
            values = groups.get(name, {})
            values = values if isinstance(values, Mapping) else {}
            for field in (*_PAIRED_A_PLUS_GROUP_FIELDS, "loss_mean",
                          "good_advantage_mean", "bad_advantage_mean"):
                metrics[
                    f"regime_selectivity_paired_a_plus_{name}_{field}"
                ] = float(values.get(field, 0.0))
    return metrics


def _regime_observability_evaluation_metrics(
    overall: Mapping[str, object],
    by_guidance_phase: Mapping[str, object],
) -> dict[str, float]:
    """Flatten bounded Regime learning diagnostics for candidate reasoning."""
    metrics: dict[str, float] = {}
    channels = overall.get("regime_teacher_channels")
    channels = channels if isinstance(channels, Mapping) else {}
    for channel, raw_values in channels.items():
        values = raw_values if isinstance(raw_values, Mapping) else {}
        for field in (
            "rows",
            "target_probability_mean",
            "model_probability_mean",
            "mean_error",
            "mean_absolute_error",
            "root_mean_squared_error",
        ):
            metrics[f"regime_teacher_channel_{channel}_{field}"] = float(
                values.get(field, 0.0)
            )
    balance = overall.get("entry_action_balance")
    balance = balance if isinstance(balance, Mapping) else {}
    for action in _ENTRY_BALANCE_ACTION_NAMES:
        raw_values = balance.get(action)
        values = raw_values if isinstance(raw_values, Mapping) else {}
        for field in (
            "rows",
            "configured_weight",
            "weighted_mass",
            "weighted_mass_fraction",
            "unweighted_ce_mean",
            "weighted_loss_contribution",
            "weighted_ce_fraction",
        ):
            metrics[f"entry_balance_{action}_{field}"] = float(
                values.get(field, 0.0)
            )
    conflict = overall.get("regime_entry_conflict")
    conflict = conflict if isinstance(conflict, Mapping) else {}
    for side in ("long", "short"):
        raw_values = conflict.get(side)
        values = raw_values if isinstance(raw_values, Mapping) else {}
        for field in (
            "rows",
            "target_wait_probability_mean",
            "target_declared_side_probability_mean",
            "model_wait_probability_mean",
            "soft_wait_disagreement_rows",
            "soft_wait_disagreement_rate",
        ):
            metrics[f"regime_entry_conflict_{side}_{field}"] = float(
                values.get(field, 0.0)
            )
    for phase, raw_phase in by_guidance_phase.items():
        phase_values = raw_phase if isinstance(raw_phase, Mapping) else {}
        metrics[f"regime_guidance_phase_{phase}_episodes"] = float(
            phase_values.get("episodes", 0.0)
        )
        metrics[f"regime_guidance_phase_{phase}_pass_rate"] = float(
            phase_values.get("pass_rate", 0.0)
        )
        metrics[f"regime_guidance_phase_{phase}_trade_win_rate"] = float(
            phase_values.get("trade_win_rate", 0.0)
        )
        metrics[f"regime_guidance_phase_{phase}_mean_terminal_pnl"] = float(
            phase_values.get("mean_terminal_pnl", 0.0)
        )
        metrics[
            f"regime_guidance_phase_{phase}_visible_fraction"
        ] = float(phase_values.get("teacher_guidance_visible_fraction", 0.0))
    trade_economics = overall.get("regime_trade_economics")
    trade_economics = (
        trade_economics if isinstance(trade_economics, Mapping) else {}
    )
    for field in (
        "total_trades",
        "attributed_trades",
        "unattributed_trades",
        "attribution_coverage",
    ):
        metrics[f"regime_trade_economics_{field}"] = float(
            trade_economics.get(field, 0.0)
        )
    raw_groups = trade_economics.get("groups")
    groups = raw_groups if isinstance(raw_groups, list) else []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        prefix = "regime_trade_economics_" + "_".join((
            str(group.get("side", "unknown")),
            str(group.get("static_regime", "unknown")),
            str(group.get("headroom_stratum", "unknown")),
            str(group.get("episode_outcome", "unknown")),
        ))
        for field in (
            "trades",
            "win_rate",
            "realized_r_mean",
            "mfe_r_mean",
            "mae_r_mean",
            "initial_stop_count",
        ):
            metrics[f"{prefix}_{field}"] = float(group.get(field, 0.0))
        channel_means = group.get("regime_channel_probability_means")
        channel_means = (
            channel_means if isinstance(channel_means, Mapping) else {}
        )
        for channel, value in channel_means.items():
            metrics[f"{prefix}_{channel}_mean"] = float(value)
    return metrics


def _training_evaluation_gates(
    *,
    regime_selectivity_active: bool,
    regime_selectivity_semantics: str = "static_state_v1",
    entry_action_loss_reduction: str = "population_weighted_mean_v1",
    entry_action_supervision_active: bool = False,
    chop_wait_margin_active: bool = False,
) -> tuple[EvaluationGate, ...]:
    gates = [EvaluationGate("short_circuited", "==", 0.0)]
    if entry_action_loss_reduction not in {
        "population_weighted_mean_v1",
        "equal_present_class_mean_v1",
    }:
        raise ValueError("entry action loss reduction gate is invalid")
    if (
        entry_action_supervision_active
        and entry_action_loss_reduction == "equal_present_class_mean_v1"
    ):
        for action in _ENTRY_BALANCE_ACTION_NAMES:
            gates.extend((
                EvaluationGate(f"entry_balance_{action}_rows", ">", 0.0),
                EvaluationGate(
                    f"entry_balance_{action}_weighted_mass_fraction",
                    ">=",
                    0.32,
                ),
                EvaluationGate(
                    f"entry_balance_{action}_weighted_mass_fraction",
                    "<=",
                    0.34,
                ),
            ))
    if regime_selectivity_active:
        gates.extend(
            (
                EvaluationGate(
                    "sampled_entry_action_long_rows", ">", 0.0
                ),
                EvaluationGate(
                    "sampled_entry_action_short_rows", ">", 0.0
                ),
                EvaluationGate(
                    "sampled_entry_action_long_recall", ">", 0.0
                ),
                EvaluationGate(
                    "sampled_entry_action_short_recall", ">", 0.0
                ),
                EvaluationGate(
                    "regime_selectivity_positive_long_rows", ">", 0.0
                ),
                EvaluationGate(
                    "regime_selectivity_positive_short_rows", ">", 0.0
                ),
                EvaluationGate(
                    "regime_selectivity_positive_long_"
                    "declared_side_probability_sum",
                    ">",
                    0.0,
                ),
                EvaluationGate(
                    "regime_selectivity_positive_short_"
                    "declared_side_probability_sum",
                    ">",
                    0.0,
                ),
                EvaluationGate(
                    "final_regime_probe_wait_rows",
                    "==",
                    float(FINAL_REGIME_PROBE_SAMPLES_PER_ACTION),
                ),
                EvaluationGate(
                    "final_regime_probe_long_rows",
                    "==",
                    float(FINAL_REGIME_PROBE_SAMPLES_PER_ACTION),
                ),
                EvaluationGate(
                    "final_regime_probe_short_rows",
                    "==",
                    float(FINAL_REGIME_PROBE_SAMPLES_PER_ACTION),
                ),
                EvaluationGate("final_regime_probe_wait_recall", ">=", 0.5),
                EvaluationGate("final_regime_probe_long_recall", ">=", 0.4),
                EvaluationGate("final_regime_probe_short_recall", ">=", 0.4),
            )
        )
        if regime_selectivity_semantics == "static_state_v1":
            gates.extend(
                EvaluationGate(metric, ">", 0.0)
                for metric in (
                    "regime_selectivity_dominant_chop_rows",
                    "regime_selectivity_nonchop_rows",
                    "final_regime_probe_dominant_chop_rows",
                    "final_regime_probe_nonchop_rows",
                )
            )
            # The matched class-reduction screen measures whether balanced
            # optimizer mass restores deployable WAIT/Long/Short behavior. Its
            # frozen hypothesis explicitly does not claim that the legacy
            # static Regime score has economic chop separation, so retain that
            # contrast as diagnostic evidence without selecting on noise.
            if entry_action_loss_reduction == "population_weighted_mean_v1":
                gates.append(EvaluationGate(
                    "final_regime_probe_chop_minus_nonchop_wait", ">", 0.0
                ))
        elif regime_selectivity_semantics in {
            "persistent_chop_negative_weight_v1",
            "persistent_chop_association_v2",
            "expansion_regime_confluence_v3",
            "side_conditioned_expansion_regime_confluence_v4",
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        }:
            gates.extend((
                EvaluationGate("latest_teacher_weight_scale", "==", 0.0),
                EvaluationGate(
                    "latest_entry_action_weight_scale",
                    "==",
                    1.0 if entry_action_supervision_active else 0.0,
                ),
            ))
            for side in ("long", "short"):
                gates.extend((
                    EvaluationGate(
                        f"regime_entry_conflict_{side}_rows", ">", 0.0
                    ),
                    EvaluationGate(
                        "regime_entry_conflict_"
                        f"{side}_target_wait_probability_mean",
                        "==",
                        0.0,
                    ),
                    EvaluationGate(
                        "regime_entry_conflict_"
                        f"{side}_target_declared_side_probability_mean",
                        "==",
                        1.0,
                    ),
                    EvaluationGate(
                        "regime_entry_conflict_"
                        f"{side}_soft_wait_disagreement_rows",
                        "==",
                        0.0,
                    ),
                ))
            gates.append(EvaluationGate(
                "regime_selectivity_exact_wait_weight_mean", ">", 1.0
            ))
            gates.extend(
                EvaluationGate(metric, ">", 0.0)
                for metric in (
                    "regime_selectivity_exact_wait_rows",
                    "regime_selectivity_persistent_dead_chop_weight_sum",
                    "regime_selectivity_transition_ready_weight_sum",
                    "regime_selectivity_transition_positive_long_rows",
                    "regime_selectivity_transition_positive_short_rows",
                    "regime_selectivity_transition_positive_long_"
                    "declared_side_probability_sum",
                    "regime_selectivity_transition_positive_short_"
                    "declared_side_probability_sum",
                    "final_regime_probe_persistent_dead_wait_mass",
                    "final_regime_probe_transition_ready_wait_mass",
                    "final_regime_probe_transition_positive_long_mass",
                    "final_regime_probe_transition_positive_short_mass",
                )
            )
            if regime_selectivity_semantics in {
                "persistent_chop_association_v2",
                "expansion_regime_confluence_v3",
                "side_conditioned_expansion_regime_confluence_v4",
                ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }:
                gates.extend(
                    EvaluationGate(metric, ">", 0.0)
                    for metric in (
                        "final_regime_probe_dead_wait_minus_transition_positive_wait",
                        "final_regime_probe_transition_positive_long_response",
                        "final_regime_probe_transition_positive_short_response",
                    )
                )
            if regime_selectivity_semantics in {
                "expansion_regime_confluence_v3",
                "side_conditioned_expansion_regime_confluence_v4",
                ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }:
                gates.extend((
                    EvaluationGate(
                        "regime_selectivity_failed_setup_confluence_rows",
                        ">",
                        0.0,
                    ),
                    EvaluationGate(
                        "final_regime_probe_failed_setup_confluence_mass",
                        ">",
                        0.0,
                    ),
                ))
            if regime_selectivity_semantics in {
                "side_conditioned_expansion_regime_confluence_v4",
                ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }:
                gates.extend((
                    EvaluationGate(
                        "regime_selectivity_failed_long_confluence_rows",
                        ">",
                        0.0,
                    ),
                    EvaluationGate(
                        "regime_selectivity_failed_short_confluence_rows",
                        ">",
                        0.0,
                    ),
                    EvaluationGate(
                        "final_regime_probe_failed_long_confluence_mass",
                        ">",
                        0.0,
                    ),
                    EvaluationGate(
                        "final_regime_probe_failed_short_confluence_mass",
                        ">",
                        0.0,
                    ),
                    EvaluationGate(
                        "regime_selectivity_side_conditioned_"
                        "both_sides_active_updates",
                        ">",
                        0.0,
                    ),
                ))
                if chop_wait_margin_active:
                    gates.extend((
                        EvaluationGate(
                            "regime_selectivity_persistent_dead_chop_rows",
                            ">=",
                            32.0,
                        ),
                        EvaluationGate(
                            "regime_selectivity_failed_short_confluence_rows",
                            ">=",
                            32.0,
                        ),
                        EvaluationGate(
                            "final_regime_probe_dominant_chop_wait_rows",
                            ">",
                            0.0,
                        ),
                        EvaluationGate(
                            "final_regime_probe_dominant_chop_greedy_entry_rows",
                            "==",
                            0.0,
                        ),
                    ))
            if regime_selectivity_semantics in {
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }:
                gates.extend((
                    EvaluationGate(
                        "regime_selectivity_paired_a_plus_pair_mass", ">", 0.0
                    ),
                    EvaluationGate(
                        "regime_selectivity_paired_a_plus_long_pair_mass",
                        ">",
                        0.0,
                    ),
                    EvaluationGate(
                        "regime_selectivity_paired_a_plus_short_pair_mass",
                        ">",
                        0.0,
                    ),
                ))
        else:
            raise ValueError("Regime selectivity gate semantics are invalid")
    return tuple(gates)


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
    flat_decision_count: int = 0
    greedy_entry_count: int = 0
    long_entry_count: int = 0
    short_entry_count: int = 0
    best_entry_advantage_sum: float = 0.0
    entry_advantage_probe_count: int = 0
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

    @property
    def greedy_entry_rate(self) -> float:
        return (
            self.greedy_entry_count / self.flat_decision_count
            if self.flat_decision_count else 0.0
        )

    @property
    def mean_best_entry_advantage(self) -> float:
        return (
            self.best_entry_advantage_sum / self.entry_advantage_probe_count
            if self.entry_advantage_probe_count else 0.0
        )

    def outcome(self, name: str) -> OutcomeStatistics:
        for statistics in self.outcome_statistics:
            if statistics.outcome == name:
                return statistics
        raise KeyError(f"outcome statistics are unavailable for {name}")


@dataclass(frozen=True)
class RecoveryCurriculumSettings:
    """Frozen additive Stage-2 recovery-value supervision contract."""

    schedule_seed: int
    recovery_value_loss_weight: float
    recovery_value_temperature: float
    recovery_value_store_capacity: int
    target_every_episodes: int
    supervision_start_pnls: tuple[float, ...]
    retain_nonnegative_entry_policy: bool
    start_state: ChallengeStartState
    recovery_action_margin: float = 0.0
    recovery_success_replay_update_period: int = 0
    recovery_success_replay_max_examples: int = 8
    recovery_success_replay_path: str | None = None
    recovery_success_replay_sha256: str | None = None
    healthy_pass_replay_update_period: int = 0
    healthy_pass_replay_max_examples: int = 8
    healthy_pass_replay_path: str | None = None
    healthy_pass_replay_sha256: str | None = None
    post_recovery_contrast_replay_update_period: int = 0
    post_recovery_contrast_replay_max_examples: int = 8
    post_recovery_contrast_replay_path: str | None = None
    post_recovery_contrast_replay_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schedule_seed, bool) or not isinstance(
            self.schedule_seed, int
        ):
            raise ValueError("recovery schedule seed must be an integer")
        if (
            isinstance(self.recovery_value_loss_weight, bool)
            or not 0.0 < self.recovery_value_loss_weight <= 1.0
            or isinstance(self.recovery_value_temperature, bool)
            or not np.isfinite(self.recovery_value_temperature)
            or self.recovery_value_temperature <= 0.0
            or isinstance(self.recovery_action_margin, bool)
            or not np.isfinite(self.recovery_action_margin)
            or self.recovery_action_margin < 0.0
            or isinstance(self.recovery_value_store_capacity, bool)
            or self.recovery_value_store_capacity < 1
            or isinstance(self.target_every_episodes, bool)
            or self.target_every_episodes < 1
            or type(self.retain_nonnegative_entry_policy) is not bool
            or isinstance(self.recovery_success_replay_update_period, bool)
            or self.recovery_success_replay_update_period < 0
            or isinstance(self.recovery_success_replay_max_examples, bool)
            or self.recovery_success_replay_max_examples < 1
            or isinstance(self.healthy_pass_replay_update_period, bool)
            or self.healthy_pass_replay_update_period < 0
            or isinstance(self.healthy_pass_replay_max_examples, bool)
            or self.healthy_pass_replay_max_examples < 1
            or isinstance(
                self.post_recovery_contrast_replay_update_period, bool
            )
            or self.post_recovery_contrast_replay_update_period < 0
            or isinstance(
                self.post_recovery_contrast_replay_max_examples, bool
            )
            or self.post_recovery_contrast_replay_max_examples < 1
        ):
            raise ValueError("recovery action-value supervision is invalid")
        replay_contracts = (
            (
                "recovery pass",
                self.recovery_success_replay_update_period,
                self.recovery_success_replay_path,
                self.recovery_success_replay_sha256,
            ),
            (
                "healthy pass",
                self.healthy_pass_replay_update_period,
                self.healthy_pass_replay_path,
                self.healthy_pass_replay_sha256,
            ),
            (
                "post-recovery contrast",
                self.post_recovery_contrast_replay_update_period,
                self.post_recovery_contrast_replay_path,
                self.post_recovery_contrast_replay_sha256,
            ),
        )
        for name, update_period, path, sha256 in replay_contracts:
            replay_identity = (path, sha256)
            if update_period == 0:
                if any(value is not None for value in replay_identity):
                    raise ValueError(f"{name} replay schedule is disabled")
            elif (
                not all(
                    isinstance(value, str) and value
                    for value in replay_identity
                )
                or len(str(sha256)) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(sha256)
                )
            ):
                raise ValueError(f"{name} replay identity is invalid")
        if (
            not self.supervision_start_pnls
            or any(
                not np.isfinite(value)
                or not self.start_state.mll_floor_pnl < value < 0.0
                for value in self.supervision_start_pnls
            )
            or len(set(self.supervision_start_pnls))
            != len(self.supervision_start_pnls)
            or self.start_state.realized_pnl not in self.supervision_start_pnls
        ):
            raise ValueError("recovery supervision start PnLs are invalid")
        if (
            not math.isclose(
                self.start_state.realized_pnl,
                self.start_state.equity_pnl,
            )
            or not math.isclose(
                self.start_state.realized_pnl,
                self.start_state.session_pnl,
            )
            or not self.start_state.mll_floor_pnl
            < self.start_state.realized_pnl
            < self.start_state.recovery_success_pnl
            or self.start_state.peak_equity_pnl
            < self.start_state.equity_pnl
            or self.start_state.passmark_locked
            or self.start_state.position_side != PositionSide.FLAT
            or self.start_state.position_size != 0
            or self.start_state.trading_days_elapsed < 0
        ):
            raise ValueError("Stage-2 recovery start contract drifted")

    def supervision_start_state(self, episode_index: int) -> ChallengeStartState:
        """Return one resume-stable negative-PnL training-only branch state."""
        if isinstance(episode_index, bool) or episode_index < 0:
            raise ValueError("recovery supervision episode index is invalid")
        offset = self.schedule_seed % len(self.supervision_start_pnls)
        pnl = self.supervision_start_pnls[
            (offset + episode_index) % len(self.supervision_start_pnls)
        ]
        return replace(
            self.start_state,
            realized_pnl=pnl,
            equity_pnl=pnl,
            session_pnl=pnl,
        )


@dataclass(frozen=True)
class BalanceOutcomeContrastSettings:
    """Sparse pass-versus-near-blow recurrent comparison schedule."""

    update_period: int
    max_examples: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.update_period, bool)
            or self.update_period < 1
            or isinstance(self.max_examples, bool)
            or self.max_examples < 1
        ):
            raise ValueError("balance outcome contrast settings are invalid")


@dataclass(frozen=True)
class BalanceCurriculumSettings:
    """Resume-stable ordinary challenge starts for one continuous policy."""

    schedule_seed: int
    start_pnls: tuple[float, ...]
    mll_floor_pnl: float
    pass_replay_update_period: int = 0
    pass_replay_max_examples: int = 8
    pass_replay_path: str | None = None
    pass_replay_sha256: str | None = None
    pass_replay_output: str | None = None
    outcome_contrast: BalanceOutcomeContrastSettings | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.schedule_seed, bool)
            or not isinstance(self.schedule_seed, int)
            or not math.isfinite(self.mll_floor_pnl)
            or self.mll_floor_pnl >= 0.0
            or len(self.start_pnls) < 1
            or any(
                not math.isfinite(value)
                or not self.mll_floor_pnl < value <= 0.0
                for value in self.start_pnls
            )
            or len(set(self.start_pnls)) != len(self.start_pnls)
            or isinstance(self.pass_replay_update_period, bool)
            or self.pass_replay_update_period < 0
            or isinstance(self.pass_replay_max_examples, bool)
            or self.pass_replay_max_examples < 1
        ):
            raise ValueError("balance curriculum settings are invalid")
        identity = (
            self.pass_replay_path,
            self.pass_replay_sha256,
            self.pass_replay_output,
        )
        if self.pass_replay_update_period == 0:
            if any(value is not None for value in identity):
                raise ValueError("balance pass replay schedule is disabled")
        elif (
            not all(isinstance(value, str) and value for value in identity)
            or len(str(self.pass_replay_sha256)) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(self.pass_replay_sha256)
            )
            or Path(str(self.pass_replay_output)).is_absolute()
            or Path(str(self.pass_replay_output)).name
            != str(self.pass_replay_output)
        ):
            raise ValueError("balance pass replay identity is invalid")

    def start_state(self, episode_index: int) -> ChallengeStartState:
        if isinstance(episode_index, bool) or episode_index < 0:
            raise ValueError("balance curriculum episode index is invalid")
        cycle_index, cycle_offset = divmod(
            episode_index, len(self.start_pnls)
        )
        cycle = list(self.start_pnls)
        np.random.default_rng(
            self.schedule_seed + cycle_index
        ).shuffle(cycle)
        pnl = float(cycle[cycle_offset])
        return ChallengeStartState(
            realized_pnl=pnl,
            equity_pnl=pnl,
            peak_equity_pnl=0.0,
            mll_floor_pnl=self.mll_floor_pnl,
            passmark_locked=False,
            position_side=PositionSide.FLAT,
            position_size=0,
            session_pnl=pnl,
            trading_days_elapsed=1,
            recovery_success_pnl=None,
        )


def _balance_curriculum_from_config(
    value: Mapping[str, object] | None,
    *,
    max_loss: float,
) -> tuple[BalanceCurriculumSettings | None, int]:
    if value is None:
        return None, 0
    required = {"schedule_seed", "start_pnls", "validation_episodes"}
    allowed = required | {"pass_replay", "outcome_contrast_replay"}
    if (
        not isinstance(value, Mapping)
        or not required.issubset(value)
        or not set(value).issubset(allowed)
    ):
        raise ValueError("balance curriculum fields are invalid")
    start_pnls = value["start_pnls"]
    if not isinstance(start_pnls, (list, tuple)):
        raise ValueError("balance curriculum starting PnLs are invalid")
    pass_replay = value.get("pass_replay")
    if pass_replay is not None and not isinstance(pass_replay, Mapping):
        raise ValueError("balance pass replay fields are invalid")
    outcome_contrast = value.get("outcome_contrast_replay")
    if outcome_contrast is not None and not isinstance(
        outcome_contrast, Mapping
    ):
        raise ValueError("balance outcome contrast fields are invalid")
    settings = BalanceCurriculumSettings(
        schedule_seed=int(value["schedule_seed"]),
        start_pnls=tuple(float(item) for item in start_pnls),
        mll_floor_pnl=-float(max_loss),
        pass_replay_update_period=(
            0 if pass_replay is None else int(pass_replay["update_period"])
        ),
        pass_replay_max_examples=(
            8 if pass_replay is None else int(pass_replay["max_examples"])
        ),
        pass_replay_path=(
            None if pass_replay is None else str(pass_replay["path"])
        ),
        pass_replay_sha256=(
            None if pass_replay is None else str(pass_replay["sha256"])
        ),
        pass_replay_output=(
            None if pass_replay is None else str(pass_replay["output"])
        ),
        outcome_contrast=(
            None
            if outcome_contrast is None
            else BalanceOutcomeContrastSettings(
                update_period=int(outcome_contrast["update_period"]),
                max_examples=int(outcome_contrast["max_examples"]),
            )
        ),
    )
    validation_episodes = int(value["validation_episodes"])
    if validation_episodes < 1:
        raise ValueError("balance curriculum validation budget is invalid")
    return settings, validation_episodes


@dataclass(frozen=True)
class RecoveryStressResult:
    episodes: int
    recovered: int
    not_recovered: int
    retained: int
    relapsed: int
    recovered_then_blown: int
    passes: int
    timeouts: int
    blows: int
    mean_terminal_pnl: float
    mean_wait_decisions: float
    entries_used: int
    environment_steps: int

    @property
    def recovery_success_rate(self) -> float:
        return self.recovered / self.episodes

    @property
    def blow_rate(self) -> float:
        return self.blows / self.episodes

    @property
    def retained_recovery_rate(self) -> float:
        return 0.0 if self.recovered == 0 else self.retained / self.recovered

    @property
    def relapse_rate(self) -> float:
        return 0.0 if self.recovered == 0 else self.relapsed / self.recovered


def _recovery_curriculum_from_config(
    value: Mapping[str, object] | None,
) -> tuple[RecoveryCurriculumSettings | None, int]:
    if value is None:
        return None, 0
    required = {
        "schedule_seed",
        "recovery_success_pnl",
        "action_value_supervision",
        "start_state",
        "stress_evaluation_episodes",
    }
    if set(value) != required:
        raise ValueError("recovery curriculum fields are invalid")
    start = value["start_state"]
    if not isinstance(start, Mapping):
        raise ValueError("recovery curriculum start state is invalid")
    start_required = {
        "realized_pnl",
        "equity_pnl",
        "peak_equity_pnl",
        "mll_floor_pnl",
        "passmark_locked",
        "position_side",
        "position_size",
        "session_pnl",
        "trading_days_elapsed",
    }
    if set(start) != start_required:
        raise ValueError("recovery curriculum start-state fields are invalid")
    supervision = value["action_value_supervision"]
    supervision_required = {
        "loss_weight",
        "temperature",
        "store_capacity",
        "target_every_episodes",
        "start_pnls",
        "retain_nonnegative_entry_policy",
    }
    supervision_optional = {
        "action_margin",
        "success_replay",
        "healthy_pass_replay",
        "post_recovery_contrast_replay",
    }
    if (
        not isinstance(supervision, Mapping)
        or not supervision_required.issubset(supervision)
        or set(supervision) - supervision_required - supervision_optional
    ):
        raise ValueError("recovery action-value supervision is invalid")
    success_replay = supervision.get("success_replay")
    healthy_pass_replay = supervision.get("healthy_pass_replay")
    post_recovery_contrast_replay = supervision.get(
        "post_recovery_contrast_replay"
    )
    if success_replay is not None and (
        not isinstance(success_replay, Mapping)
        or set(success_replay)
        != {"path", "sha256", "update_period", "max_examples"}
    ):
        raise ValueError("recovery pass replay identity is invalid")
    if healthy_pass_replay is not None and (
        not isinstance(healthy_pass_replay, Mapping)
        or set(healthy_pass_replay)
        != {"path", "sha256", "update_period", "max_examples"}
    ):
        raise ValueError("healthy pass replay identity is invalid")
    if post_recovery_contrast_replay is not None and (
        not isinstance(post_recovery_contrast_replay, Mapping)
        or set(post_recovery_contrast_replay)
        != {"path", "sha256", "update_period", "max_examples"}
    ):
        raise ValueError("post-recovery contrast replay identity is invalid")
    integer_fields = (
        value["schedule_seed"],
        start["position_side"],
        start["position_size"],
        start["trading_days_elapsed"],
        value["stress_evaluation_episodes"],
        supervision["store_capacity"],
        supervision["target_every_episodes"],
        *(
            ()
            if success_replay is None
            else (success_replay["update_period"],)
        ),
        *(
            ()
            if success_replay is None
            else (success_replay["max_examples"],)
        ),
        *(
            ()
            if healthy_pass_replay is None
            else (healthy_pass_replay["update_period"],)
        ),
        *(
            ()
            if healthy_pass_replay is None
            else (healthy_pass_replay["max_examples"],)
        ),
        *(
            ()
            if post_recovery_contrast_replay is None
            else (post_recovery_contrast_replay["update_period"],)
        ),
        *(
            ()
            if post_recovery_contrast_replay is None
            else (post_recovery_contrast_replay["max_examples"],)
        ),
    )
    numeric_fields = (
        value["recovery_success_pnl"],
        start["realized_pnl"],
        start["equity_pnl"],
        start["peak_equity_pnl"],
        start["mll_floor_pnl"],
        start["session_pnl"],
        supervision["loss_weight"],
        supervision["temperature"],
        supervision.get("action_margin", 0.0),
    )
    if (
        any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in integer_fields
        )
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in numeric_fields
        )
        or not isinstance(start["passmark_locked"], bool)
        or not isinstance(supervision["start_pnls"], (list, tuple))
        or not isinstance(supervision["retain_nonnegative_entry_policy"], bool)
    ):
        raise ValueError("recovery curriculum scalar types are invalid")
    settings = RecoveryCurriculumSettings(
        schedule_seed=int(value["schedule_seed"]),
        recovery_value_loss_weight=float(supervision["loss_weight"]),
        recovery_value_temperature=float(supervision["temperature"]),
        recovery_value_store_capacity=int(supervision["store_capacity"]),
        target_every_episodes=int(supervision["target_every_episodes"]),
        supervision_start_pnls=tuple(
            float(item) for item in supervision["start_pnls"]
        ),
        retain_nonnegative_entry_policy=bool(
            supervision["retain_nonnegative_entry_policy"]
        ),
        recovery_action_margin=float(supervision.get("action_margin", 0.0)),
        recovery_success_replay_update_period=(
            0
            if success_replay is None
            else int(success_replay["update_period"])
        ),
        recovery_success_replay_max_examples=(
            8 if success_replay is None else int(success_replay["max_examples"])
        ),
        recovery_success_replay_path=(
            None if success_replay is None else str(success_replay["path"])
        ),
        recovery_success_replay_sha256=(
            None if success_replay is None else str(success_replay["sha256"])
        ),
        healthy_pass_replay_update_period=(
            0
            if healthy_pass_replay is None
            else int(healthy_pass_replay["update_period"])
        ),
        healthy_pass_replay_max_examples=(
            8
            if healthy_pass_replay is None
            else int(healthy_pass_replay["max_examples"])
        ),
        healthy_pass_replay_path=(
            None
            if healthy_pass_replay is None
            else str(healthy_pass_replay["path"])
        ),
        healthy_pass_replay_sha256=(
            None
            if healthy_pass_replay is None
            else str(healthy_pass_replay["sha256"])
        ),
        post_recovery_contrast_replay_update_period=(
            0
            if post_recovery_contrast_replay is None
            else int(post_recovery_contrast_replay["update_period"])
        ),
        post_recovery_contrast_replay_max_examples=(
            8
            if post_recovery_contrast_replay is None
            else int(post_recovery_contrast_replay["max_examples"])
        ),
        post_recovery_contrast_replay_path=(
            None
            if post_recovery_contrast_replay is None
            else str(post_recovery_contrast_replay["path"])
        ),
        post_recovery_contrast_replay_sha256=(
            None
            if post_recovery_contrast_replay is None
            else str(post_recovery_contrast_replay["sha256"])
        ),
        start_state=ChallengeStartState(
            realized_pnl=float(start["realized_pnl"]),
            equity_pnl=float(start["equity_pnl"]),
            peak_equity_pnl=float(start["peak_equity_pnl"]),
            mll_floor_pnl=float(start["mll_floor_pnl"]),
            passmark_locked=bool(start["passmark_locked"]),
            position_side=PositionSide(int(start["position_side"])),
            position_size=int(start["position_size"]),
            session_pnl=float(start["session_pnl"]),
            trading_days_elapsed=int(start["trading_days_elapsed"]),
            recovery_success_pnl=float(value["recovery_success_pnl"]),
        ),
    )
    stress_episodes = int(value["stress_evaluation_episodes"])
    if stress_episodes < 0:
        raise ValueError("recovery stress episode budget is invalid")
    return settings, stress_episodes


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
        config = materialize_effective_config(config)
        configure_runtime_environment(config["runtime"])
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
        teacher_specs = tuple(config["teachers"])
        teacher_targets = None
        if teacher_specs:
            from .teachers import load_teacher_targets

            teacher_targets = load_teacher_targets(
                teacher_specs,
                root=root,
                markets=train_markets,
            )
        entry_supervision_spec = config["entry_supervision"]
        entry_action_targets = None
        entry_action_balance_receipt = None
        if entry_supervision_spec is not None:
            from .entry_supervision import build_entry_action_targets

            # Economic future truth is built only for the authenticated training
            # slice.  It is never built for, or available to, validation.
            entry_action_targets = build_entry_action_targets(
                train_markets,
                entry_supervision_spec,
                point_values=config["point_values"],
                round_trip_fees=config["round_trip_fees"],
                training_end_exclusive=temporal["train_end"],
            )
            entry_action_class_weights, entry_action_balance_receipt = (
                _entry_action_balance(
                    entry_action_targets,
                    entry_supervision_spec,
                )
            )
            if entry_action_balance_receipt is not None:
                balance_weights = entry_action_balance_receipt["class_weights"]
                print(
                    "[entry-supervision] BALANCE "
                    f"counts={dict(entry_action_balance_receipt['target_counts'])} "
                    f"weights={dict(balance_weights)} "
                    f"identity={entry_action_balance_receipt['identity_sha256']}",
                    flush=True,
                )
        challenge = ChallengeSpec(**config["challenge"])
        observation_spec = TradeManagementObservationSpec.from_config(
            config["observation"]
        )
        near_blow_loss_threshold = (
            float(config["campaign"]["near_blow_loss_fraction"])
            * challenge.max_loss
        )
        training_config = config["training"]
        seed = int(training_config["seed"])
        episode_coverage_spec = (
            FullDataEpisodeCoverageSpec.from_config(
                training_config["episode_coverage"]
            )
            if training_config["episode_coverage"] is not None
            else None
        )
        train_environment = HistoricalChallengeEnv(
            train_markets,
            tick_values=config["point_values"],
            round_trip_fees=config["round_trip_fees"],
            spec=challenge,
            observation_spec=observation_spec,
            seed=seed,
            episode_coverage=episode_coverage_spec,
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
            agent_runtime_settings(config["runtime"])
        )
        if teacher_targets is not None:
            agent_settings.update(agent_teacher_settings(teacher_specs))
        regime_selectivity_spec = config["regime_selectivity"]
        if regime_selectivity_spec is not None:
            from .teachers.expansion import verify_expansion_entry_center_receipt

            expansion_spec = next(
                item for item in teacher_specs if item["kind"] == "expansion"
            )
            verify_expansion_entry_center_receipt(
                {
                    **expansion_spec,
                    "entry_search_center_receipt": regime_selectivity_spec[
                        "expansion_center_receipt"
                    ],
                    "entry_search_center_receipt_sha256": regime_selectivity_spec[
                        "expansion_center_receipt_sha256"
                    ],
                    "entry_search_long_center": regime_selectivity_spec[
                        "expansion_long_center"
                    ],
                    "entry_search_short_center": regime_selectivity_spec[
                        "expansion_short_center"
                    ],
                },
                root=root,
                expected_tickers=tuple(config["tickers"]),
            )
            agent_settings.update(
                _regime_selectivity_agent_settings(regime_selectivity_spec)
            )
        if entry_supervision_spec is not None:
            agent_settings["entry_action_loss_weight"] = float(
                entry_supervision_spec["loss_weight"]
            )
        if entry_action_balance_receipt is not None:
            agent_settings["entry_action_class_weights"] = (
                entry_action_class_weights
            )
        output = _resolve(root, config["output"])
        output.mkdir(parents=True, exist_ok=True)
        recovery_path = output / "training-recovery.pt"
        retained_policy_path = output / "retained-pass-policy.pt"
        diagnostics_path = output / "training-diagnostics.jsonl"
        policy_health_path = output / "training-policy-health.jsonl"
        policy_health_probe_path = output / "training-policy-health-probe.pkl"
        validation_diagnostics_path = output / "validation-diagnostics.jsonl"
        balance_validation_diagnostics_path = (
            output / "balance-validation-diagnostics.jsonl"
        )
        resume_identity = _training_resume_identity(config, cache_root, teacher_specs)
        active_short_circuit = training_config["short_circuit"]
        policy_health_config = (
            active_short_circuit.get("policy_health")
            if active_short_circuit is not None
            else None
        )
        resume = None
        replay_state = None
        replay_checkpoint = None
        balance_pass_replay_sampler_state = None
        recovery_value_store_state = None
        recovery_success_replay_sampler_state = None
        healthy_pass_replay_sampler_state = None
        post_recovery_contrast_replay_sampler_state = None
        if recovery_path.is_file():
            loaded, manifest = RecurrentC51Agent.load(
                recovery_path, device=agent_settings["device"]
            )
            if manifest.get("resume_identity") != resume_identity:
                raise ValueError("training recovery identity drifted")
            resume = TrainingProgress(**manifest["progress"])
            _truncate_episode_jsonl(
                diagnostics_path,
                completed_episodes=resume.completed_episodes,
                episode_field="episode",
                required=True,
            )
            _truncate_episode_jsonl(
                policy_health_path,
                completed_episodes=resume.completed_episodes,
                episode_field="completed_episodes",
                required=policy_health_config is not None,
            )
            _reconcile_policy_health_probe_corpus(
                policy_health_probe_path,
                resume_identity=resume_identity,
                completed_episodes=resume.completed_episodes,
                checkpoint_contract=manifest.get(
                    "policy_health_probe_corpus"
                ),
            )
            def load_retained_manifest(candidate_path: Path) -> Mapping[str, object]:
                _, retained_manifest = RecurrentC51Agent.load(
                    candidate_path,
                    device="cpu",
                )
                return retained_manifest

            _reconcile_retained_pass_policies(
                retained_policy_path,
                resume_identity=resume_identity,
                completed_episodes=resume.completed_episodes,
                manifest_loader=load_retained_manifest,
            )
            train_environment.restore_rng_state(manifest["environment_rng_state"])
            replay_state = manifest.get("replay_state")
            replay_checkpoint = manifest.get("replay_checkpoint")
            if (
                not manifest.get("replay_restored", False)
                or (replay_state is None) == (replay_checkpoint is None)
            ):
                raise ValueError("training recovery is missing replay state")
            balance_pass_replay_sampler_state = manifest.get(
                "balance_pass_replay_sampler_state"
            )
            recovery_value_store_state = manifest.get(
                "recovery_value_store_state"
            )
            recovery_success_replay_sampler_state = manifest.get(
                "recovery_success_replay_sampler_state"
            )
            healthy_pass_replay_sampler_state = manifest.get(
                "healthy_pass_replay_sampler_state"
            )
            post_recovery_contrast_replay_sampler_state = manifest.get(
                "post_recovery_contrast_replay_sampler_state"
            )
            if any(
                manifest.get(key) is not None
                for key in (
                    "recovery_success_replay_state",
                    "healthy_pass_replay_state",
                    "post_recovery_contrast_replay_state",
                )
            ):
                raise ValueError(
                    "training recovery embeds an obsolete replay library"
                )
            _assert_recovery_entry_balance(loaded, agent_settings)
            _assert_recovery_regime_selectivity(loaded, agent_settings)
            agent = loaded
        else:
            if (
                diagnostics_path.exists()
                or policy_health_path.exists()
                or policy_health_probe_path.exists()
                or retained_policy_path.exists()
                or (output / "retained-pass-policies").exists()
            ):
                raise ValueError("training artifacts exist without resumable recovery")
            warm_start = config["_warm_start_model"]
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
        recovery_curriculum, recovery_stress_episodes = (
            _recovery_curriculum_from_config(config["recovery_curriculum"])
        )
        balance_curriculum, balance_validation_episodes = (
            _balance_curriculum_from_config(
                config["balance_curriculum"],
                max_loss=challenge.max_loss,
            )
        )
        balance_pass_replay_output_path = (
            None
            if balance_curriculum is None
            or balance_curriculum.pass_replay_output is None
            else output / balance_curriculum.pass_replay_output
        )
        if recovery_curriculum is not None and balance_curriculum is not None:
            raise ValueError(
                "balance and recovery curricula are mutually exclusive"
            )
        recovery_value_policy = None
        recovery_value_environment = None
        recovery_value_store = None
        if recovery_curriculum is not None:
            warm_start = config["_warm_start_model"]
            if warm_start is None:
                raise ValueError(
                    "Stage-2 recovery requires the frozen V21 parent policy"
                )
            warm_path = Path(str(warm_start["model_path"])).resolve(strict=True)
            if _path_sha256(warm_path) != str(warm_start["model_sha256"]):
                raise ValueError("recovery supervisor identity drifted")
            recovery_value_policy, _ = RecurrentC51Agent.load(
                warm_path,
                device=agent_settings["device"],
            )
            assert_teacher_free = getattr(
                recovery_value_policy, "assert_teacher_free", None
            )
            if assert_teacher_free is not None:
                assert_teacher_free()
            recovery_value_environment = HistoricalChallengeEnv(
                train_markets,
                tick_values=config["point_values"],
                round_trip_fees=config["round_trip_fees"],
                spec=challenge,
                observation_spec=observation_spec,
                seed=recovery_curriculum.schedule_seed,
            )
            recovery_value_store = RecoveryValueStore(
                capacity=recovery_curriculum.recovery_value_store_capacity,
                seed=recovery_curriculum.schedule_seed,
            )
            if recovery_value_store_state is not None:
                recovery_value_store.load_state_dict(
                    recovery_value_store_state
                )
        elif recovery_value_store_state is not None:
            raise ValueError(
                "V21 recovery checkpoint unexpectedly contains Stage-2B state"
            )
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
                training_config["safety_sequence_fraction"]
            ),
            entry_opportunity_sequence_fraction=float(
                training_config["entry_opportunity_sequence_fraction"]
            ),
            regime_wait_sequence_fraction=float(
                training_config["regime_wait_sequence_fraction"]
            ),
            regime_wait_sequence_update_period=int(
                training_config["regime_wait_sequence_update_period"]
            ),
            **_regime_selectivity_replay_settings(regime_selectivity_spec),
            recurrent_burn_in=int(agent.recurrent_burn_in),
            n_step_return=int(agent.n_step_return),
            seed=seed,
        )
        replay_checkpoint_store = ReplayCheckpointStore(
            output / "training-replay"
        )
        if replay_checkpoint is not None:
            replay_checkpoint_store.restore(replay, replay_checkpoint)
        elif replay_state is not None:
            replay.load_state_dict(replay_state)
        balance_pass_replay = None
        if (
            balance_curriculum is not None
            and balance_curriculum.pass_replay_path is not None
        ):
            balance_pass_replay = BalancedSequenceReplay(
                capacity_episodes=(
                    balance_curriculum.pass_replay_max_examples
                ),
                capacity_transitions=None,
                sequence_length=int(training_config["sequence_length"]),
                recurrent_burn_in=int(agent.recurrent_burn_in),
                n_step_return=int(agent.n_step_return),
                seed=balance_curriculum.schedule_seed + 3,
            )
            _load_balance_pass_replay_source(
                _resolve(root, balance_curriculum.pass_replay_path),
                expected_sha256=str(
                    balance_curriculum.pass_replay_sha256
                ),
                replay=balance_pass_replay,
                max_examples=balance_curriculum.pass_replay_max_examples,
            )
            balance_pass_replay.absorb_recent_passes(
                replay,
                max_examples=balance_curriculum.pass_replay_max_examples,
            )
            if balance_pass_replay_sampler_state is not None:
                balance_pass_replay.load_sampler_state_dict(
                    balance_pass_replay_sampler_state
                )
            elif resume is not None:
                raise ValueError(
                    "training recovery is missing balance pass sampler state"
                )
        elif balance_pass_replay_sampler_state is not None:
            raise ValueError(
                "checkpoint contains disabled balance pass sampler state"
            )
        recovery_success_replay = None
        if (
            recovery_curriculum is not None
            and recovery_curriculum.recovery_success_replay_path is not None
        ):
            recovery_success_replay = BalancedSequenceReplay(
                capacity_episodes=int(
                    training_config["replay_capacity_episodes"]
                ),
                capacity_transitions=int(
                    training_config["replay_capacity_transitions"]
                ),
                sequence_length=int(training_config["sequence_length"]),
                terminal_sequence_fraction=float(
                    training_config["terminal_sequence_fraction"]
                ),
                safety_sequence_fraction=float(
                    training_config["safety_sequence_fraction"]
                ),
                entry_opportunity_sequence_fraction=float(
                    training_config["entry_opportunity_sequence_fraction"]
                ),
                regime_wait_sequence_fraction=float(
                    training_config["regime_wait_sequence_fraction"]
                ),
                regime_wait_sequence_update_period=int(
                    training_config["regime_wait_sequence_update_period"]
                ),
                **_regime_selectivity_replay_settings(
                    regime_selectivity_spec
                ),
                recurrent_burn_in=int(agent.recurrent_burn_in),
                n_step_return=int(agent.n_step_return),
                seed=recovery_curriculum.schedule_seed,
            )
            _load_recovery_success_replay_artifact(
                _resolve(
                    root,
                    recovery_curriculum.recovery_success_replay_path,
                ),
                expected_sha256=str(
                    recovery_curriculum.recovery_success_replay_sha256
                ),
                replay=recovery_success_replay,
            )
            if recovery_success_replay_sampler_state is not None:
                recovery_success_replay.load_sampler_state_dict(
                    recovery_success_replay_sampler_state
                )
            elif resume is not None:
                raise ValueError(
                    "training recovery is missing recovery pass sampler state"
                )
        elif recovery_success_replay_sampler_state is not None:
            raise ValueError(
                "checkpoint contains disabled recovery pass sampler state"
            )
        healthy_pass_replay = None
        if (
            recovery_curriculum is not None
            and recovery_curriculum.healthy_pass_replay_path is not None
        ):
            healthy_pass_replay = BalancedSequenceReplay(
                capacity_episodes=int(
                    training_config["replay_capacity_episodes"]
                ),
                capacity_transitions=int(
                    training_config["replay_capacity_transitions"]
                ),
                sequence_length=int(training_config["sequence_length"]),
                terminal_sequence_fraction=float(
                    training_config["terminal_sequence_fraction"]
                ),
                safety_sequence_fraction=float(
                    training_config["safety_sequence_fraction"]
                ),
                entry_opportunity_sequence_fraction=float(
                    training_config["entry_opportunity_sequence_fraction"]
                ),
                regime_wait_sequence_fraction=float(
                    training_config["regime_wait_sequence_fraction"]
                ),
                regime_wait_sequence_update_period=int(
                    training_config["regime_wait_sequence_update_period"]
                ),
                **_regime_selectivity_replay_settings(
                    regime_selectivity_spec
                ),
                recurrent_burn_in=int(agent.recurrent_burn_in),
                n_step_return=int(agent.n_step_return),
                seed=recovery_curriculum.schedule_seed + 1,
            )
            _load_healthy_pass_replay_artifact(
                _resolve(
                    root,
                    recovery_curriculum.healthy_pass_replay_path,
                ),
                expected_sha256=str(
                    recovery_curriculum.healthy_pass_replay_sha256
                ),
                replay=healthy_pass_replay,
            )
            if healthy_pass_replay_sampler_state is not None:
                healthy_pass_replay.load_sampler_state_dict(
                    healthy_pass_replay_sampler_state
                )
            elif resume is not None:
                raise ValueError(
                    "training recovery is missing healthy pass sampler state"
                )
        elif healthy_pass_replay_sampler_state is not None:
            raise ValueError(
                "checkpoint contains disabled healthy pass sampler state"
            )
        post_recovery_contrast_replay = None
        if (
            recovery_curriculum is not None
            and recovery_curriculum.post_recovery_contrast_replay_path
            is not None
        ):
            post_recovery_contrast_replay = BalancedSequenceReplay(
                capacity_episodes=int(
                    training_config["replay_capacity_episodes"]
                ),
                capacity_transitions=int(
                    training_config["replay_capacity_transitions"]
                ),
                sequence_length=int(training_config["sequence_length"]),
                terminal_sequence_fraction=float(
                    training_config["terminal_sequence_fraction"]
                ),
                safety_sequence_fraction=float(
                    training_config["safety_sequence_fraction"]
                ),
                entry_opportunity_sequence_fraction=float(
                    training_config["entry_opportunity_sequence_fraction"]
                ),
                regime_wait_sequence_fraction=float(
                    training_config["regime_wait_sequence_fraction"]
                ),
                regime_wait_sequence_update_period=int(
                    training_config["regime_wait_sequence_update_period"]
                ),
                **_regime_selectivity_replay_settings(
                    regime_selectivity_spec
                ),
                recurrent_burn_in=int(agent.recurrent_burn_in),
                n_step_return=int(agent.n_step_return),
                seed=recovery_curriculum.schedule_seed + 2,
            )
            _load_post_recovery_contrast_replay_artifact(
                _resolve(
                    root,
                    recovery_curriculum.post_recovery_contrast_replay_path,
                ),
                expected_sha256=str(
                    recovery_curriculum.post_recovery_contrast_replay_sha256
                ),
                replay=post_recovery_contrast_replay,
            )
            if post_recovery_contrast_replay_sampler_state is not None:
                post_recovery_contrast_replay.load_sampler_state_dict(
                    post_recovery_contrast_replay_sampler_state
                )
            elif resume is not None:
                raise ValueError(
                    "training recovery is missing post-recovery contrast "
                    "sampler state"
                )
        elif post_recovery_contrast_replay_sampler_state is not None:
            raise ValueError(
                "checkpoint contains disabled post-recovery contrast sampler "
                "state"
            )
        policy_health_monitor = None
        if policy_health_config is not None:
            if teacher_targets is None or regime_selectivity_spec is None:
                raise ValueError(
                    "training policy-health probe requires Regime target lineage"
                )
            policy_health_spec = TrainingPolicyHealthSpec.from_config(
                policy_health_config
            )
            fixed_health_samples = None

            def policy_health_probe(
                completed_episodes: int,
            ) -> Mapping[str, float]:
                nonlocal fixed_health_samples
                if fixed_health_samples is None:
                    if policy_health_probe_path.is_file():
                        payload = _load_policy_health_probe_corpus(
                            policy_health_probe_path,
                            resume_identity=resume_identity,
                        )
                        if int(payload["completed_episodes"]) > completed_episodes:
                            raise ValueError(
                                "training policy-health probe corpus drifted"
                            )
                        fixed_health_samples = payload["samples"]
                    else:
                        fixed_health_samples = replay.final_regime_probe_sequences(
                            samples_per_action=(
                                FINAL_REGIME_PROBE_SAMPLES_PER_ACTION
                            ),
                        )
                        temporary = policy_health_probe_path.with_suffix(
                            policy_health_probe_path.suffix + ".tmp"
                        )
                        with temporary.open("wb") as stream:
                            pickle.dump({
                                "schema": (
                                    "propevolve_training_policy_health_probe_corpus_v1"
                                ),
                                "resume_identity": resume_identity,
                                "completed_episodes": completed_episodes,
                                "samples": fixed_health_samples,
                            }, stream, protocol=pickle.HIGHEST_PROTOCOL)
                        os.replace(temporary, policy_health_probe_path)
                return evaluate_final_regime_probe(
                    agent,
                    fixed_health_samples,
                    teacher_channel_names=teacher_targets.channels,
                    q_temperature=float(
                        regime_selectivity_spec["q_temperature"]
                    ),
                    **_regime_selectivity_probe_settings(
                        regime_selectivity_spec
                    ),
                    source_period=(
                        str(temporal["train_start"]),
                        str(temporal["train_end"]),
                    ),
                ).metrics

            policy_health_monitor = TrainingHealthMonitor(
                TrainingHealthDetector(policy_health_spec),
                probe=policy_health_probe,
                receipt_callback=lambda payload: _append_jsonl(
                    policy_health_path, payload
                ),
                initial_entry_weighted_masses=(
                    _load_policy_health_entry_weighted_masses(
                        policy_health_path,
                        completed_episodes=resume.completed_episodes,
                    )
                    if resume is not None
                    else None
                ),
                minimum_entry_mass_completed_episodes=int(
                    training_config["short_circuit"].get(
                        "minimum_completed_episodes",
                        1,
                    )
                ),
            )
        def save_training_recovery(progress: TrainingProgress) -> None:
            replay_descriptor = replay_checkpoint_store.persist(replay)
            _save_training_recovery(
                agent,
                recovery_path,
                resume_identity=resume_identity,
                progress=progress,
                environment_rng_state=train_environment.rng_state(),
                replay_checkpoint=replay_descriptor,
                balance_pass_replay_sampler_state=(
                    None
                    if balance_pass_replay is None
                    else balance_pass_replay.sampler_state_dict()
                ),
                recovery_value_store_state=(
                    None
                    if recovery_value_store is None
                    else recovery_value_store.state_dict()
                ),
                recovery_success_replay_sampler_state=(
                    None
                    if recovery_success_replay is None
                    else recovery_success_replay.sampler_state_dict()
                ),
                healthy_pass_replay_sampler_state=(
                    None
                    if healthy_pass_replay is None
                    else healthy_pass_replay.sampler_state_dict()
                ),
                post_recovery_contrast_replay_sampler_state=(
                    None
                    if post_recovery_contrast_replay is None
                    else post_recovery_contrast_replay.sampler_state_dict()
                ),
                policy_health_probe_path=(
                    policy_health_probe_path
                    if policy_health_config is not None
                    else None
                ),
            )
            replay_checkpoint_store.prune(replay_descriptor)

        training = train_agent(
            agent,
            train_environment,
            episodes=int(training_config["episodes"]),
            minimum_environment_steps=int(
                training_config["minimum_environment_steps"]
            ),
            budget_mode=str(training_config["budget_mode"]),
            replay=replay,
            warmup_episodes=int(training_config["warmup_episodes"]),
            updates_per_episode=int(training_config["updates_per_episode"]),
            batch_sequences=int(training_config["batch_sequences"]),
            prefetch_batches=int(training_config["prefetch_batches"]),
            recurrent_horizon=int(training_config["recurrent_horizon"]),
            greedy_diagnostic_interval_steps=int(
                training_config["greedy_diagnostic_interval_steps"]
            ),
            epsilon_start=float(training_config["epsilon_start"]),
            epsilon_end=float(training_config["epsilon_end"]),
            management_epsilon_start=float(
                training_config["management_epsilon_start"]
            ),
            management_epsilon_end=float(
                training_config["management_epsilon_end"]
            ),
            episode_tickers=tuple(config["tickers"]),
            ticker_seed=seed,
            resume=resume,
            checkpoint_every_episodes=int(
                training_config["checkpoint_every_episodes"]
            ),
            checkpoint_callback=save_training_recovery,
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
            entry_action_lookup=(
                entry_action_targets.target
                if entry_action_targets is not None
                else None
            ),
            entry_action_metadata_lookup=(
                entry_action_targets.metadata
                if entry_action_targets is not None
                else None
            ),
            teacher_loss_end_scale=float(
                training_config["teacher_loss_end_scale"]
            ),
            teacher_guidance_dropout_start=float(
                training_config["teacher_guidance_dropout_start"]
            ),
            teacher_guidance_dropout_end=float(
                training_config["teacher_guidance_dropout_end"]
            ),
            teacher_autonomy_start_fraction=float(
                training_config["teacher_autonomy_start_fraction"]
            ),
            entry_supervision_autonomy_start_fraction=float(
                training_config["entry_supervision_autonomy_start_fraction"]
            ),
            short_circuit_minimum_environment_steps=(
                int(active_short_circuit["minimum_environment_steps"])
                if (
                    active_short_circuit is not None
                    and training_config["budget_mode"] == "environment_steps"
                )
                else None
            ),
            short_circuit_minimum_episodes=(
                int(
                    active_short_circuit[
                        "minimum_completed_episodes"
                    ]
                )
                if (
                    active_short_circuit is not None
                    and training_config["budget_mode"] == "episodes"
                )
                else None
            ),
            short_circuit_minimum_passes=(
                int(active_short_circuit["minimum_passes"])
                if active_short_circuit is not None
                else 0
            ),
            short_circuit_maximum_blow_rate=(
                float(active_short_circuit["maximum_blow_rate"])
                if active_short_circuit is not None
                else 1.0
            ),
            collapse_window_episodes=int(
                (active_short_circuit or {})
                .get("collapse", {})
                .get("window_episodes", 0)
            ),
            collapse_minimum_prior_passes=int(
                (active_short_circuit or {})
                .get("collapse", {})
                .get("minimum_prior_passes", 0)
            ),
            collapse_maximum_recent_passes=int(
                (active_short_circuit or {})
                .get("collapse", {})
                .get("maximum_recent_passes", 0)
            ),
            collapse_maximum_average_hold_bars=float(
                (active_short_circuit or {})
                .get("collapse", {})
                .get("maximum_average_hold_bars", math.inf)
            ),
            collapse_minimum_voluntary_close_rate=float(
                (active_short_circuit or {})
                .get("collapse", {})
                .get("minimum_voluntary_close_rate", 1.0)
            ),
            episode_diagnostic_callback=lambda payload: _append_jsonl(
                diagnostics_path, payload
            ),
            training_health_callback=policy_health_monitor,
            near_blow_loss_threshold=near_blow_loss_threshold,
            balance_curriculum=balance_curriculum,
            balance_pass_replay=balance_pass_replay,
            balance_pass_replay_callback=(
                None
                if balance_pass_replay_output_path is None
                else lambda state: _save_balance_pass_replay_artifact(
                    balance_pass_replay_output_path,
                    replay_state=state,
                    resume_identity=resume_identity,
                )
            ),
            recovery_curriculum=recovery_curriculum,
            recovery_value_policy=recovery_value_policy,
            recovery_value_environment=recovery_value_environment,
            recovery_value_store=recovery_value_store,
            recovery_value_source_identity_sha256=(
                resume_identity if recovery_curriculum is not None else None
            ),
            recovery_success_replay=recovery_success_replay,
            healthy_pass_replay=healthy_pass_replay,
            post_recovery_contrast_replay=post_recovery_contrast_replay,
        )
        episode_coverage_receipt = None
        episode_coverage_path = output / "episode-coverage-receipt.json"
        if episode_coverage_spec is not None:
            episode_coverage_receipt = train_environment.episode_coverage_receipt(
                require_complete=not training.short_circuited,
            )
            temporary = episode_coverage_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(
                episode_coverage_receipt,
                indent=2,
                sort_keys=True,
            ) + "\n")
            os.replace(temporary, episode_coverage_path)
        diagnostic_summary_path = output / "training-diagnostic-summary.json"
        _write_training_diagnostic_summary(
            diagnostics_path,
            diagnostic_summary_path,
        )
        diagnostic_summary = json.loads(diagnostic_summary_path.read_text())
        retained_policy_restored = False
        if training.short_circuited and retained_policy_path.is_file():
            retained_agent, retention_manifest = RecurrentC51Agent.load(
                retained_policy_path,
                device=agent_settings["device"],
            )
            if retention_manifest.get("resume_identity") != resume_identity:
                raise ValueError("retained pass policy identity drifted")
            retained_episode, _ = _retained_pass_identity(
                retention_manifest,
                resume_identity=resume_identity,
            )
            if retained_episode > training.episodes:
                raise ValueError(
                    "retained pass policy exceeds the durable training boundary"
                )
            agent = retained_agent
            retained_policy_restored = True
        agent.discard_retention_anchor()
        agent.discard_teacher()
        final_regime_probe = None
        if regime_selectivity_spec is not None:
            if teacher_targets is None:
                raise ValueError("final Regime probe requires training label lineage")
            try:
                final_regime_probe_samples = replay.final_regime_probe_sequences(
                    samples_per_action=FINAL_REGIME_PROBE_SAMPLES_PER_ACTION,
                )
            except ValueError as error:
                if (
                    not training.short_circuited
                    or str(error)
                    != "final Regime probe lacks exact balanced authentic rows"
                ):
                    raise
            else:
                final_regime_probe = evaluate_final_regime_probe(
                    agent,
                    final_regime_probe_samples,
                    teacher_channel_names=teacher_targets.channels,
                    q_temperature=float(regime_selectivity_spec["q_temperature"]),
                    **_regime_selectivity_probe_settings(
                        regime_selectivity_spec
                    ),
                    source_period=(
                        str(temporal["train_start"]),
                        str(temporal["train_end"]),
                    ),
                )
                probe_path = output / "final-regime-probe.json"
                probe_temporary = probe_path.with_suffix(".json.tmp")
                probe_temporary.write_text(
                    json.dumps(
                        final_regime_probe.as_dict(),
                        indent=2,
                        sort_keys=True,
                    ) + "\n"
                )
                os.replace(probe_temporary, probe_path)
        validation = None
        balance_validation = None
        recovery_stress = None
        if not training.short_circuited:
            _preserve_partial_validation_diagnostics(
                validation_diagnostics_path
            )
            validation = evaluate_agent(
                agent,
                validation_environment,
                episodes=int(training_config["validation_episodes"]),
                recurrent_horizon=int(training_config["recurrent_horizon"]),
                near_blow_loss_threshold=near_blow_loss_threshold,
                stop_on_first_blow=bool(
                    config["_validation_stop_on_blow"]
                ),
                no_trade_patience_episodes=int(
                    training_config["validation_no_trade_patience_episodes"]
                ),
                greedy_diagnostic_interval_steps=int(
                    training_config["greedy_diagnostic_interval_steps"]
                ),
                episode_diagnostic_callback=lambda payload: _append_jsonl(
                    validation_diagnostics_path, payload
                ),
                normal_policy=recovery_value_policy,
            )
            if recovery_stress_episodes:
                assert recovery_curriculum is not None
                recovery_stress = evaluate_recovery_stress(
                    agent,
                    validation_environment,
                    episodes=recovery_stress_episodes,
                    recurrent_horizon=int(training_config["recurrent_horizon"]),
                    settings=recovery_curriculum,
                    episode_tickers=tuple(config["deployment_tickers"]),
                    normal_policy=recovery_value_policy,
                )
            if balance_validation_episodes:
                assert balance_curriculum is not None
                _preserve_partial_validation_diagnostics(
                    balance_validation_diagnostics_path
                )
                balance_validation = evaluate_agent(
                    agent,
                    validation_environment,
                    episodes=balance_validation_episodes,
                    recurrent_horizon=int(training_config["recurrent_horizon"]),
                    near_blow_loss_threshold=near_blow_loss_threshold,
                    greedy_diagnostic_interval_steps=int(
                        training_config["greedy_diagnostic_interval_steps"]
                    ),
                    episode_diagnostic_callback=lambda payload: _append_jsonl(
                        balance_validation_diagnostics_path, payload
                    ),
                    balance_curriculum=balance_curriculum,
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
            "training_resume_identity": resume_identity,
            "entry_action_loss_reduction": str(
                agent.entry_action_loss_reduction
            ),
            "auxiliary_gradient_conflict_mode": str(
                config["agent"]["auxiliary_gradient_conflict_mode"]
            ),
            "runtime_source_modules_sha256": (
                _runtime_source_modules_sha256(config)
            ),
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
                if config["_warm_start_model"] is None
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
            "entry_supervision": _entry_supervision_frozen_contract(
                entry_action_targets,
                entry_action_balance_receipt,
            ),
            "regime_selectivity": _regime_selectivity_frozen_contract(
                regime_selectivity_spec
            ),
            "final_regime_probe": (
                None
                if final_regime_probe is None
                else {
                    "schema": final_regime_probe.schema,
                    "source_period": list(final_regime_probe.source_period),
                    "sample_identity_sha256": (
                        final_regime_probe.sample_identity_sha256
                    ),
                    "path": str(output / "final-regime-probe.json"),
                    "file_sha256": _path_sha256(
                        output / "final-regime-probe.json"
                    ),
                }
            ),
            "training_policy_health": (
                None
                if policy_health_config is None
                else {
                    "schema": "propevolve_training_policy_health_v1",
                    "path": str(policy_health_path),
                    "file_sha256": _path_sha256(policy_health_path),
                    "fixed_probe_corpus": (
                        None
                        if not policy_health_probe_path.is_file()
                        else {
                            "path": str(policy_health_probe_path),
                            "file_sha256": _path_sha256(
                                policy_health_probe_path
                            ),
                        }
                    ),
                }
            ),
            "episode_coverage": (
                None
                if episode_coverage_receipt is None
                else {
                    "receipt": _plain_contract_value(
                        episode_coverage_receipt
                    ),
                    "path": str(episode_coverage_path),
                    "file_sha256": _path_sha256(episode_coverage_path),
                }
            ),
            "validation_diagnostics": (
                None
                if validation is None
                else {
                    "schema": "propevolve_validation_episode_diagnostic_v1",
                    "path": str(validation_diagnostics_path),
                    "file_sha256": _path_sha256(validation_diagnostics_path),
                }
            ),
            "recovery_curriculum": _plain_contract_value(
                config["recovery_curriculum"]
            ),
            "balance_curriculum": _plain_contract_value(
                config["balance_curriculum"]
            ),
            "balance_validation_diagnostics": (
                None
                if balance_validation is None
                else {
                    "schema": "propevolve_validation_episode_diagnostic_v1",
                    "path": str(balance_validation_diagnostics_path),
                    "file_sha256": _path_sha256(
                        balance_validation_diagnostics_path
                    ),
                }
            ),
            "retained_pass_policy_restored": retained_policy_restored,
        }
        archive_output = _resolve(root, str(config["_archive_output"]))
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
                "episodes": float(training.episodes),
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
            if episode_coverage_receipt is not None:
                coverage_markets = episode_coverage_receipt["markets"]
                metrics.update({
                    "episode_coverage_complete": float(
                        bool(episode_coverage_receipt["complete"])
                    ),
                    "minimum_market_episode_coverage": min(
                        float(item["coverage_fraction"])
                        for item in coverage_markets.values()
                    ),
                })
            if final_regime_probe is not None:
                metrics.update(final_regime_probe.metrics)
                overall_diagnostics = diagnostic_summary["overall"]
                metrics["latest_teacher_weight_scale"] = float(
                    overall_diagnostics["latest_teacher_weight_scale"]
                )
                metrics["latest_entry_action_weight_scale"] = float(
                    overall_diagnostics["latest_entry_action_weight_scale"]
                )
                metrics.update(_regime_selectivity_evaluation_metrics(
                    overall_diagnostics["regime_selectivity"]
                ))
                metrics.update(
                    _persistent_regime_selectivity_evaluation_metrics(
                        overall_diagnostics["persistent_regime_selectivity"]
                    )
                )
                metrics.update(_regime_observability_evaluation_metrics(
                    overall_diagnostics,
                    diagnostic_summary.get("by_guidance_phase", {}),
                ))
                for side in ("ENTER_LONG_1", "ENTER_SHORT_1"):
                    metric_side = "long" if side.endswith("LONG_1") else "short"
                    metrics[
                        f"sampled_entry_action_{metric_side}_rows"
                    ] = float(
                        overall_diagnostics[
                            "sampled_entry_action_target_counts"
                        ][side]
                    )
                    metrics[
                        f"sampled_entry_action_{metric_side}_recall"
                    ] = float(
                        overall_diagnostics["sampled_entry_action_recall"][side]
                    )
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
                "trade_count": float(validation.trade_count),
                "flat_decision_count": float(validation.flat_decision_count),
                "greedy_entry_count": float(validation.greedy_entry_count),
                "long_entry_count": float(validation.long_entry_count),
                "short_entry_count": float(validation.short_entry_count),
                "long_entry_share": (
                    validation.long_entry_count / validation.greedy_entry_count
                    if validation.greedy_entry_count else 0.0
                ),
                "short_entry_share": (
                    validation.short_entry_count / validation.greedy_entry_count
                    if validation.greedy_entry_count else 0.0
                ),
                "entry_advantage_probe_count": float(
                    validation.entry_advantage_probe_count
                ),
                "greedy_entry_rate": validation.greedy_entry_rate,
                "mean_best_entry_advantage": (
                    validation.mean_best_entry_advantage
                ),
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

        def recovery_stress_metrics(_candidate):
            assert recovery_stress is not None
            return {
                "episodes": float(recovery_stress.episodes),
                "recovered": float(recovery_stress.recovered),
                "recovery_success_rate": recovery_stress.recovery_success_rate,
                "not_recovered": float(recovery_stress.not_recovered),
                "not_recovered_rate": (
                    recovery_stress.not_recovered / recovery_stress.episodes
                ),
                "retained": float(recovery_stress.retained),
                "retained_recovery_rate": (
                    recovery_stress.retained_recovery_rate
                ),
                "relapsed": float(recovery_stress.relapsed),
                "relapse_rate": recovery_stress.relapse_rate,
                "recovered_then_blown": float(
                    recovery_stress.recovered_then_blown
                ),
                "passes": float(recovery_stress.passes),
                "timeouts": float(recovery_stress.timeouts),
                "blows": float(recovery_stress.blows),
                "blow_rate": recovery_stress.blow_rate,
                "mean_terminal_pnl": recovery_stress.mean_terminal_pnl,
                "mean_wait_decisions": recovery_stress.mean_wait_decisions,
                "entries_used": float(recovery_stress.entries_used),
                "outcome_accounted": float(
                    recovery_stress.passes
                    + recovery_stress.timeouts
                    + recovery_stress.blows
                    == recovery_stress.episodes
                ),
                "environment_steps": float(recovery_stress.environment_steps),
            }

        def balance_stress_metrics(_candidate):
            assert balance_validation is not None
            pass_rate = balance_validation.passes / balance_validation.episodes
            blow_rate = balance_validation.blows / balance_validation.episodes
            return {
                "episodes": float(balance_validation.episodes),
                "passes": float(balance_validation.passes),
                "timeouts": float(balance_validation.timeouts),
                "blows": float(balance_validation.blows),
                "pass_rate": pass_rate,
                "blow_rate": blow_rate,
                "pass_minus_blow": pass_rate - blow_rate,
                "mean_terminal_pnl": balance_validation.mean_terminal_pnl,
                "worst_pnl": balance_validation.worst_pnl,
                "near_blow_timeout_rate": (
                    balance_validation.near_blow_timeout_rate
                ),
                "trade_win_rate": balance_validation.trade_win_rate,
                "average_win_r": balance_validation.average_win_r,
                "long_entry_count": float(balance_validation.long_entry_count),
                "short_entry_count": float(
                    balance_validation.short_entry_count
                ),
                "outcome_accounted": float(
                    balance_validation.passes
                    + balance_validation.timeouts
                    + balance_validation.blows
                    == balance_validation.episodes
                ),
                "environment_steps": float(
                    balance_validation.environment_steps
                ),
            }

        stages = [
            EvaluationStage(
                "training",
                training_metrics,
                gates=_training_evaluation_gates(
                    regime_selectivity_active=(
                        regime_selectivity_spec is not None
                    ),
                    regime_selectivity_semantics=(
                        "static_state_v1"
                        if regime_selectivity_spec is None
                        else str(regime_selectivity_spec["semantics"])
                    ),
                    entry_action_loss_reduction=str(
                        agent.entry_action_loss_reduction
                    ),
                    entry_action_supervision_active=(
                        entry_supervision_spec is not None
                    ),
                    chop_wait_margin_active=(
                        regime_selectivity_spec is not None
                        and (
                            float(regime_selectivity_spec.get(
                                "chop_wait_margin", 0.0
                            )) > 0.0
                            or float(regime_selectivity_spec.get(
                                "failed_confluence_margin", 0.0
                            )) > 0.0
                        )
                    ),
                ),
            ),
            EvaluationStage(
                "selection",
                selection_metrics,
                gates=_selection_evaluation_gates(
                    require_both_entry_sides=(
                        regime_selectivity_spec is not None
                        and regime_selectivity_spec.get("side_balance")
                        is not None
                    )
                ),
            ),
        ]
        if recovery_stress is not None:
            stages.append(EvaluationStage(
                "recovery_stress",
                recovery_stress_metrics,
                gates=_recovery_stress_integrity_gates(),
            ))
        if balance_validation is not None:
            stages.append(EvaluationStage(
                "balance_stress",
                balance_stress_metrics,
                gates=_recovery_stress_integrity_gates(),
            ))
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
            tuple(stages),
        )
        return candidate, cascade.evaluate(candidate.candidate_id)


def _selection_evaluation_gates(
    *, require_both_entry_sides: bool = False,
) -> tuple[EvaluationGate, ...]:
    """Reject incomplete selection even when its partial economics look positive."""
    gates = [
        EvaluationGate("short_circuited", "==", 0.0),
        EvaluationGate("pass_minus_blow", ">", 0.0),
    ]
    if require_both_entry_sides:
        gates.extend((
            EvaluationGate("long_entry_count", ">", 0.0),
            EvaluationGate("short_entry_count", ">", 0.0),
        ))
    return tuple(gates)


def _recovery_stress_integrity_gates() -> tuple[EvaluationGate, ...]:
    """Gate evaluator integrity; campaign requirements own economic acceptance."""
    return (EvaluationGate("outcome_accounted", "==", 1.0),)


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
    if config["_warm_start_model"] is not None:
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
    source_identity = _runtime_source_modules_sha256(config)
    digest.update(json.dumps(
        source_identity, sort_keys=True, separators=(",", ":")
    ).encode())
    return digest.hexdigest()


def _runtime_source_modules_sha256(config: Mapping) -> dict[str, str]:
    """Content-address the complete package source used by a candidate run."""
    del config  # The whole package is deliberately bound, not a conditional subset.
    package_root = Path(__file__).parent
    return {
        module.relative_to(package_root).as_posix(): hashlib.sha256(
            module.read_bytes()
        ).hexdigest()
        for module in sorted(package_root.rglob("*.py"))
    }


def _save_training_recovery(
    agent: RecurrentC51Agent,
    path: Path,
    *,
    resume_identity: str,
    progress: TrainingProgress,
    environment_rng_state: dict,
    replay_checkpoint: dict[str, object] | None = None,
    replay_state: dict[str, object] | None = None,
    balance_pass_replay_sampler_state: dict[str, object] | None = None,
    recovery_value_store_state: dict[str, object] | None = None,
    recovery_success_replay_sampler_state: dict[str, object] | None = None,
    healthy_pass_replay_sampler_state: dict[str, object] | None = None,
    post_recovery_contrast_replay_sampler_state: dict[str, object] | None = None,
    policy_health_probe_path: Path | None = None,
) -> None:
    if (replay_checkpoint is None) == (replay_state is None):
        raise ValueError("training recovery requires exactly one replay source")
    policy_health_probe_corpus = None
    if policy_health_probe_path is not None and policy_health_probe_path.exists():
        payload = _load_policy_health_probe_corpus(
            policy_health_probe_path,
            resume_identity=resume_identity,
        )
        corpus_episode = int(payload["completed_episodes"])
        if corpus_episode > progress.completed_episodes:
            raise ValueError(
                "policy-health probe corpus exceeds the recovery checkpoint"
            )
        policy_health_probe_corpus = {
            "completed_episodes": corpus_episode,
            "file_sha256": _path_sha256(policy_health_probe_path),
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    replay_manifest = (
        {"replay_checkpoint": replay_checkpoint}
        if replay_checkpoint is not None
        else {"replay_state": replay_state}
    )
    agent.save(
        temporary,
        manifest={
            "resume_identity": resume_identity,
            "progress": asdict(progress),
            "environment_rng_state": environment_rng_state,
            **replay_manifest,
            "replay_restored": True,
            "balance_pass_replay_sampler_state": (
                balance_pass_replay_sampler_state
            ),
            "recovery_value_store_state": recovery_value_store_state,
            "recovery_success_replay_sampler_state": (
                recovery_success_replay_sampler_state
            ),
            "healthy_pass_replay_sampler_state": (
                healthy_pass_replay_sampler_state
            ),
            "post_recovery_contrast_replay_sampler_state": (
                post_recovery_contrast_replay_sampler_state
            ),
            "policy_health_probe_corpus": policy_health_probe_corpus,
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


def _retained_pass_identity(
    manifest: Mapping[str, object],
    *,
    resume_identity: str,
) -> tuple[int, str]:
    if manifest.get("resume_identity") != resume_identity:
        raise ValueError("retained pass policy identity drifted")
    evidence = manifest.get("retention_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("retained pass evidence is malformed")
    episode = evidence.get("episode")
    ticker = evidence.get("ticker")
    if (
        isinstance(episode, bool)
        or not isinstance(episode, int)
        or episode < 1
        or not isinstance(ticker, str)
        or not ticker.isalnum()
        or evidence.get("outcome") != "pass"
    ):
        raise ValueError("retained pass evidence identity is invalid")
    return episode, ticker


def _reconcile_retained_pass_policies(
    path: Path,
    *,
    resume_identity: str,
    completed_episodes: int,
    manifest_loader: Callable[[Path], Mapping[str, object]],
) -> None:
    """Keep the retained-pass alias at or behind the durable recovery point."""
    if completed_episodes < 0 or not callable(manifest_loader):
        raise ValueError("retained pass recovery contract is invalid")
    archive = path.parent / "retained-pass-policies"
    if not archive.exists() and not path.exists():
        return
    if archive.exists() and not archive.is_dir():
        raise ValueError("retained pass archive is malformed")
    archive.mkdir(parents=True, exist_ok=True)
    partial = archive / "partial"

    def preserve(source: Path, *, label: str) -> None:
        partial.mkdir(parents=True, exist_ok=True)
        source_sha256 = _path_sha256(source)
        destination = partial / (
            f"{label}.after-episode-{completed_episodes:06d}."
            f"{source_sha256}{source.suffix}"
        )
        if destination.exists():
            if (
                not destination.is_file()
                or _path_sha256(destination) != source_sha256
            ):
                raise ValueError("retained pass partial evidence drifted")
            source.unlink()
            return
        os.replace(source, destination)

    durable: list[tuple[int, str, Path]] = []
    for retained in sorted(archive.glob("*.pt")):
        parts = retained.stem.split("-", 2)
        if (
            len(parts) != 3
            or parts[0] != "episode"
            or len(parts[1]) != 6
            or not parts[1].isdigit()
            or not parts[2].isalnum()
        ):
            raise ValueError("retained pass checkpoint name is malformed")
        episode, ticker = _retained_pass_identity(
            manifest_loader(retained),
            resume_identity=resume_identity,
        )
        if episode != int(parts[1]) or ticker != parts[2]:
            raise ValueError("retained pass filename drifted from its evidence")
        if episode > completed_episodes:
            preserve(retained, label=retained.stem)
        else:
            durable.append((episode, ticker, retained))

    alias_identity: tuple[int, str] | None = None
    if path.exists():
        if not path.is_file():
            raise ValueError("retained pass alias is malformed")
        alias_identity = _retained_pass_identity(
            manifest_loader(path),
            resume_identity=resume_identity,
        )
        alias_episode, alias_ticker = alias_identity
        if alias_episode <= completed_episodes and not any(
            (episode, ticker) == alias_identity
            for episode, ticker, _ in durable
        ):
            recovered = archive / (
                f"episode-{alias_episode:06d}-{alias_ticker}.pt"
            )
            shutil.copyfile(path, recovered)
            durable.append((alias_episode, alias_ticker, recovered))
        if alias_episode > completed_episodes:
            preserve(path, label="latest-alias")

    if durable:
        _, _, latest = max(durable, key=lambda item: (item[0], item[1]))
        temporary = path.with_suffix(path.suffix + ".recovery.tmp")
        shutil.copyfile(latest, temporary)
        os.replace(temporary, path)
    elif path.exists():
        # This can only be an alias without a durable archive counterpart.
        preserve(path, label="latest-alias")


def _load_policy_health_probe_corpus(
    path: Path,
    *,
    resume_identity: str,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError("training policy-health probe corpus is unavailable")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if (
        not isinstance(payload, dict)
        or set(payload) != {
            "schema",
            "resume_identity",
            "completed_episodes",
            "samples",
        }
        or payload.get("schema")
        != "propevolve_training_policy_health_probe_corpus_v1"
        or payload.get("resume_identity") != resume_identity
        or isinstance(payload.get("completed_episodes"), bool)
        or not isinstance(payload.get("completed_episodes"), int)
        or int(payload["completed_episodes"]) < 1
        or not isinstance(payload.get("samples"), tuple)
        or not payload["samples"]
    ):
        raise ValueError("training policy-health probe corpus drifted")
    return payload


def _reconcile_policy_health_probe_corpus(
    path: Path,
    *,
    resume_identity: str,
    completed_episodes: int,
    checkpoint_contract: Mapping[str, object] | None,
) -> None:
    """Bind a fixed health corpus to the exact durable recovery checkpoint."""
    if completed_episodes < 0:
        raise ValueError("policy-health probe recovery boundary is invalid")
    if checkpoint_contract is not None and (
        not isinstance(checkpoint_contract, Mapping)
        or set(checkpoint_contract) != {"completed_episodes", "file_sha256"}
        or isinstance(checkpoint_contract.get("completed_episodes"), bool)
        or not isinstance(checkpoint_contract.get("completed_episodes"), int)
        or not isinstance(checkpoint_contract.get("file_sha256"), str)
        or len(str(checkpoint_contract["file_sha256"])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(checkpoint_contract["file_sha256"])
        )
    ):
        raise ValueError("policy-health probe checkpoint contract is invalid")
    if not path.exists():
        if checkpoint_contract is not None:
            raise ValueError("policy-health probe checkpoint corpus is missing")
        return
    payload = _load_policy_health_probe_corpus(
        path,
        resume_identity=resume_identity,
    )
    corpus_episode = int(payload["completed_episodes"])
    corpus_sha256 = _path_sha256(path)
    if checkpoint_contract is not None:
        if (
            int(checkpoint_contract["completed_episodes"]) != corpus_episode
            or str(checkpoint_contract["file_sha256"]) != corpus_sha256
            or corpus_episode > completed_episodes
        ):
            raise ValueError("policy-health probe checkpoint identity drifted")
        return
    if corpus_episode <= completed_episodes:
        raise ValueError(
            "durable recovery does not authenticate its policy-health probe corpus"
        )
    preserved = path.with_name(
        f"{path.stem}.partial-episode-{corpus_episode:06d}-"
        f"{corpus_sha256}{path.suffix}"
    )
    if preserved.exists():
        if not preserved.is_file() or _path_sha256(preserved) != corpus_sha256:
            raise ValueError("preserved policy-health probe evidence drifted")
        path.unlink()
    else:
        os.replace(path, preserved)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        stream.write("\n")


def _load_policy_health_entry_weighted_masses(
    path: Path,
    *,
    completed_episodes: int,
) -> dict[str, float]:
    if completed_episodes < 1 or not path.is_file():
        raise ValueError("policy-health cumulative entry mass is missing")
    lines = [line for line in path.read_text().splitlines() if line]
    try:
        receipt = json.loads(lines[-1])
        identity = receipt.pop("identity_sha256")
        masses = receipt["entry_weighted_masses"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("policy-health cumulative entry mass is malformed") from error
    expected_identity = hashlib.sha256(json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    if (
        identity != expected_identity
        or receipt.get("completed_episodes") != completed_episodes
        or not isinstance(masses, Mapping)
        or set(masses) != {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}
    ):
        raise ValueError("policy-health cumulative entry mass is malformed")
    result = {str(action): float(value) for action, value in masses.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("policy-health cumulative entry mass is malformed")
    return result


def _truncate_episode_jsonl(
    path: Path,
    *,
    completed_episodes: int,
    episode_field: str,
    required: bool,
) -> None:
    """Reconcile evidence written after the last durable episode checkpoint."""
    if (
        completed_episodes < 0
        or not episode_field
        or not isinstance(required, bool)
    ):
        raise ValueError("episode evidence recovery contract is invalid")
    if not path.exists():
        if required and completed_episodes:
            raise ValueError("required episode evidence stream is missing")
        return
    if not path.is_file():
        raise ValueError("episode evidence recovery contract is invalid")
    kept: list[str] = []
    previous = 0
    dropped = False
    lines = path.read_text().splitlines()
    last_nonempty = next(
        (index for index in range(len(lines) - 1, -1, -1) if lines[index].strip()),
        -1,
    )
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line)
            episode = payload[episode_field]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            if index == last_nonempty and previous == completed_episodes:
                dropped = True
                break
            raise ValueError("episode evidence stream is malformed") from error
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode != previous + 1
        ):
            raise ValueError("episode evidence order is invalid")
        previous = episode
        if episode <= completed_episodes:
            kept.append(line)
        else:
            dropped = True
    if previous < completed_episodes:
        raise ValueError("episode evidence stream lags durable recovery")
    if not dropped:
        return
    temporary = path.with_suffix(path.suffix + ".recovery.tmp")
    temporary.write_text("".join(f"{line}\n" for line in kept))
    os.replace(temporary, path)


def _preserve_partial_validation_diagnostics(path: Path) -> Path | None:
    """Atomically rotate exactly one prior validation stream before restart."""
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("validation diagnostics path is not a file")
    content_sha256 = _path_sha256(path)
    preserved = path.with_name(
        f"{path.stem}.partial-{content_sha256}{path.suffix}"
    )
    if preserved.exists() and not preserved.is_file():
        raise ValueError("preserved validation diagnostics path is not a file")
    os.replace(path, preserved)
    return preserved


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
    guidance_eligible = sum(
        int(row.get("teacher_guidance_eligible_decisions", 0)) for row in rows
    )
    guidance_visible = sum(
        int(row.get("teacher_guidance_visible_decisions", 0)) for row in rows
    )
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
        "latest_teacher_weight_scale": (
            float(rows[-1].get("teacher_weight_scale", 0.0)) if rows else 0.0
        ),
        "latest_entry_action_weight_scale": (
            float(rows[-1].get("entry_action_weight_scale", 0.0))
            if rows else 0.0
        ),
        "latest_teacher_schedule_progress": (
            float(rows[-1].get("teacher_schedule_progress", 0.0))
            if rows else 0.0
        ),
        "latest_entry_action_schedule_progress": (
            float(rows[-1].get("entry_action_schedule_progress", 0.0))
            if rows else 0.0
        ),
        "mean_teacher_weight_scale": weighted(
            "teacher_weight_scale", update_weights
        ),
        "mean_entry_action_weight_scale": weighted(
            "entry_action_weight_scale", update_weights
        ),
        "guidance_phase": (
            str(rows[0].get("guidance_phase", "unknown"))
            if rows
            and len({str(row.get("guidance_phase", "unknown")) for row in rows}) == 1
            else "mixed"
        ),
        "teacher_guidance_eligible_decisions": guidance_eligible,
        "teacher_guidance_visible_decisions": guidance_visible,
        "teacher_guidance_visible_fraction": (
            guidance_visible / guidance_eligible if guidance_eligible else 0.0
        ),
        "mean_training_loss": weighted("mean_training_loss", update_weights),
        "mean_rl_loss": weighted("mean_rl_loss", update_weights),
        "mean_teacher_loss": weighted("mean_teacher_loss", update_weights),
        "mean_entry_action_loss": weighted(
            "mean_entry_action_loss", update_weights
        ),
        "mean_entry_action_supervised_rows": weighted(
            "mean_entry_action_supervised_rows", update_weights
        ),
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
        "healthy_entry_policy_retention_loss": weighted(
            "mean_healthy_entry_policy_retention_loss", update_weights
        ),
        "healthy_entry_policy_retention_rows": weighted(
            "mean_healthy_entry_policy_retention_rows", update_weights
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
        "entry_action_target_counts": {
            action.name: sum(
                int(row.get("entry_action_target_counts", {}).get(action.name, 0))
                for row in rows
            )
            for action in (
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            )
        },
        "regime_selectivity": _regime_selectivity_summary(rows),
        "regime_teacher_channels": _regime_channel_summary(rows),
        "entry_action_balance": _entry_balance_summary(rows),
        "regime_entry_conflict": _regime_entry_conflict_summary(rows),
        "regime_trade_economics": _regime_trade_economics_summary(rows),
        "persistent_regime_selectivity": (
            _persistent_regime_selectivity_summary(rows)
        ),
    }
    sampled_target_counts = {
        action: sum(
            int(row.get("sampled_entry_action_target_counts", {}).get(action, 0))
            for row in rows
        )
        for action in _ENTRY_ACTION_ORDER
    }
    sampled_prediction_counts = {
        action: sum(
            int(
                row.get("sampled_entry_action_prediction_counts", {}).get(
                    action, 0
                )
            )
            for row in rows
        )
        for action in _ENTRY_ACTION_ORDER
    }
    sampled_correct_counts = {
        action: sum(
            int(row.get("sampled_entry_action_correct_counts", {}).get(action, 0))
            for row in rows
        )
        for action in _ENTRY_ACTION_ORDER
    }
    result["sampled_entry_action_target_counts"] = sampled_target_counts
    result["sampled_entry_action_prediction_counts"] = sampled_prediction_counts
    result["sampled_entry_action_correct_counts"] = sampled_correct_counts
    result["sampled_entry_action_recall"] = {
        action: (
            sampled_correct_counts[action] / sampled_target_counts[action]
            if sampled_target_counts[action] else 0.0
        )
        for action in _ENTRY_ACTION_ORDER
    }
    result["sampled_entry_action_precision"] = {
        action: (
            sampled_correct_counts[action] / sampled_prediction_counts[action]
            if sampled_prediction_counts[action] else 0.0
        )
        for action in _ENTRY_ACTION_ORDER
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
    by_guidance_phase = {
        phase: _diagnostic_aggregate([
            row for row in rows if str(row.get("guidance_phase")) == phase
        ])
        for phase in sorted({str(row.get("guidance_phase")) for row in rows})
    }
    payload = {
        "schema": "propevolve_training_diagnostic_summary_v1",
        "source": source.name,
        "source_sha256": _path_sha256(source),
        "overall": _diagnostic_aggregate(rows),
        "recent_20": _diagnostic_aggregate(rows[-20:]),
        "by_ticker": by_ticker,
        "by_outcome": by_outcome,
        "by_guidance_phase": by_guidance_phase,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_recovery_success_replay_artifact(
    path: Path,
    *,
    expected_sha256: str,
    replay: BalancedSequenceReplay,
) -> None:
    """Authenticate and restore a training-only V22 recovery-pass library."""
    _load_authenticated_pass_replay_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_schema="propevolve_recovery_success_replay_v1",
        replay=replay,
        sample_kind="recovery",
    )


def _load_balance_pass_replay_artifact(
    path: Path,
    *,
    expected_sha256: str,
    replay: BalancedSequenceReplay,
) -> None:
    """Authenticate complete passes for the single-policy balance curriculum."""
    _load_authenticated_pass_replay_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_schema="propevolve_balance_pass_replay_v1",
        replay=replay,
        sample_kind="balance",
    )


def _balance_pass_replay_source_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise ValueError("balance pass replay source does not exist")
    artifacts = tuple(sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix == ".pt"
    ))
    if not artifacts:
        raise ValueError("balance pass replay directory is empty")
    return artifacts


def _balance_pass_replay_source_sha256(path: Path) -> str:
    artifacts = _balance_pass_replay_source_paths(path)
    if len(artifacts) == 1 and artifacts[0] == path:
        return _path_sha256(path)
    digest = hashlib.sha256()
    for artifact in artifacts:
        encoded_name = artifact.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(_path_sha256(artifact)))
    return digest.hexdigest()


def _load_balance_pass_replay_source(
    path: Path,
    *,
    expected_sha256: str,
    replay: BalancedSequenceReplay,
    max_examples: int,
) -> None:
    artifacts = _balance_pass_replay_source_paths(path)
    if _balance_pass_replay_source_sha256(path) != expected_sha256:
        raise ValueError("balance pass replay source identity drifted")
    if len(artifacts) == 1 and artifacts[0] == path:
        _load_balance_pass_replay_artifact(
            path,
            expected_sha256=expected_sha256,
            replay=replay,
        )
        return
    for index, artifact in enumerate(artifacts):
        candidate = BalancedSequenceReplay(
            capacity_episodes=max_examples,
            capacity_transitions=None,
            sequence_length=replay.sequence_length,
            recurrent_burn_in=replay.recurrent_burn_in,
            n_step_return=replay.n_step_return,
            seed=index,
        )
        _load_balance_pass_replay_artifact(
            artifact,
            expected_sha256=_path_sha256(artifact),
            replay=candidate,
        )
        replay.absorb_recent_passes(
            candidate,
            max_examples=max_examples,
        )


def _save_balance_pass_replay_artifact(
    path: Path,
    *,
    replay_state: Mapping[str, object],
    resume_identity: str,
) -> None:
    """Atomically persist the bounded, training-only balance-pass library."""
    if not resume_identity:
        raise ValueError("balance pass replay source identity is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "propevolve_balance_pass_replay_v1",
            "source_checkpoints": [{
                "causal_identity_sha256": resume_identity,
                "resume_identity": resume_identity,
            }],
            "replay_state": dict(replay_state),
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_healthy_pass_replay_artifact(
    path: Path,
    *,
    expected_sha256: str,
    replay: BalancedSequenceReplay,
) -> None:
    """Authenticate and restore V21 healthy pass-policy rehearsal rows."""
    _load_authenticated_pass_replay_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_schema="propevolve_healthy_pass_replay_v1",
        replay=replay,
        sample_kind="healthy",
    )


def _load_post_recovery_contrast_replay_artifact(
    path: Path,
    *,
    expected_sha256: str,
    replay: BalancedSequenceReplay,
) -> None:
    """Authenticate retained-versus-giveback recurrent training pairs."""
    _load_authenticated_pass_replay_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_schema="propevolve_post_recovery_contrast_replay_v1",
        replay=replay,
        sample_kind="post_recovery_contrast",
    )


def _load_authenticated_pass_replay_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
    replay: BalancedSequenceReplay,
    sample_kind: str,
) -> None:
    if sample_kind not in {
        "balance",
        "recovery",
        "healthy",
        "post_recovery_contrast",
    }:
        raise ValueError("training replay sample kind is invalid")
    if _path_sha256(path) != expected_sha256:
        raise ValueError("pass replay identity drifted")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != expected_schema:
        raise ValueError("pass replay schema is invalid")
    sources = payload.get("source_checkpoints")
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(source, Mapping)
            or set(source)
            != {"causal_identity_sha256", "resume_identity"}
            or not isinstance(source["causal_identity_sha256"], str)
            or len(source["causal_identity_sha256"]) != 64
            or not isinstance(source["resume_identity"], str)
            or not source["resume_identity"]
            for source in sources
        )
    ):
        raise ValueError("pass replay source identity is invalid")
    state = payload.get("replay_state")
    if not isinstance(state, Mapping):
        raise ValueError("pass replay state is invalid")
    if sample_kind == "balance" and any(
        not isinstance(episode, Mapping)
        or episode.get("outcome") != "pass"
        for episode in state.get("episodes", ())
    ):
        raise ValueError("balance pass replay contains a non-pass episode")
    replay.load_state_dict(state)
    if sample_kind == "balance":
        sample = replay.sample_balance_pass_entry_sequences(1)
        requirement = "economic winner entry"
    elif sample_kind == "healthy":
        sample = replay.sample_healthy_pass_sequences(1)
        requirement = "healthy policy row"
    elif sample_kind == "recovery":
        sample = replay.sample_successful_recovery_sequences(1)
        requirement = "successful boundary"
    else:
        sample = replay.sample_post_recovery_contrast_pairs(1)
        requirement = "retained-versus-giveback pair"
    if not sample:
        article = "an" if requirement.startswith("economic") else "a"
        raise ValueError(f"pass replay lacks {article} {requirement}")
    replay.load_state_dict(state)


def _plain_contract_value(value):
    """Convert immutable runtime receipts into JSON/pickle-safe containers."""
    if isinstance(value, Mapping):
        return {
            str(key): _plain_contract_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_plain_contract_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_contract_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    budget_mode: str = "environment_steps",
    replay: BalancedSequenceReplay,
    warmup_episodes: int,
    updates_per_episode: int,
    batch_sequences: int,
    recurrent_horizon: int,
    greedy_diagnostic_interval_steps: int = 256,
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
    entry_action_lookup: Callable[[str, int], Action | None] | None = None,
    entry_action_metadata_lookup: Callable[[str, int], object | None] | None = None,
    teacher_loss_end_scale: float = 1.0,
    teacher_guidance_dropout_start: float = 0.0,
    teacher_guidance_dropout_end: float = 0.0,
    teacher_autonomy_start_fraction: float = 1.0,
    entry_supervision_autonomy_start_fraction: float | None = None,
    episode_diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
    training_health_callback: Callable[
        [TrainingProgress, dict[str, object]], str | None
    ] | None = None,
    near_blow_loss_threshold: float | None = None,
    short_circuit_minimum_environment_steps: int | None = None,
    short_circuit_minimum_episodes: int | None = None,
    short_circuit_minimum_passes: int = 0,
    short_circuit_maximum_blow_rate: float = 1.0,
    collapse_window_episodes: int = 0,
    collapse_minimum_prior_passes: int = 0,
    collapse_maximum_recent_passes: int = 0,
    collapse_maximum_average_hold_bars: float = math.inf,
    collapse_minimum_voluntary_close_rate: float = 1.0,
    balance_curriculum: BalanceCurriculumSettings | None = None,
    balance_pass_replay: BalancedSequenceReplay | None = None,
    balance_pass_replay_callback: (
        Callable[[dict[str, object]], None] | None
    ) = None,
    recovery_curriculum: RecoveryCurriculumSettings | None = None,
    recovery_value_policy: RecurrentC51Agent | None = None,
    recovery_value_environment: HistoricalChallengeEnv | None = None,
    recovery_value_store: RecoveryValueStore | None = None,
    recovery_value_source_identity_sha256: str | None = None,
    recovery_success_replay: BalancedSequenceReplay | None = None,
    healthy_pass_replay: BalancedSequenceReplay | None = None,
    post_recovery_contrast_replay: BalancedSequenceReplay | None = None,
) -> TrainingResult:
    if episodes < 1 or minimum_environment_steps < 1:
        raise ValueError("episode ceiling and minimum environment steps must be positive")
    if budget_mode not in {"environment_steps", "episodes"}:
        raise ValueError("training budget mode is invalid")
    if (
        replay.recurrent_burn_in != int(getattr(agent, "recurrent_burn_in", 0))
        or replay.n_step_return != int(getattr(agent, "n_step_return", 1))
    ):
        raise ValueError("replay recurrent learning contract drifted from the agent")
    if (
        isinstance(greedy_diagnostic_interval_steps, bool)
        or greedy_diagnostic_interval_steps < 1
    ):
        raise ValueError("greedy diagnostic interval must be positive")
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
        short_circuit_minimum_episodes is not None
        and (
            isinstance(short_circuit_minimum_episodes, bool)
            or not 1 <= short_circuit_minimum_episodes <= episodes
        )
    ):
        raise ValueError("training short-circuit episode boundary is invalid")
    if (
        budget_mode == "environment_steps"
        and short_circuit_minimum_episodes is not None
    ):
        raise ValueError(
            "episode short-circuit boundary requires episode budget mode"
        )
    if (
        budget_mode == "episodes"
        and short_circuit_minimum_environment_steps is not None
    ):
        raise ValueError(
            "step short-circuit boundary requires environment-step budget mode"
        )
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
    if balance_curriculum is not None and recovery_curriculum is not None:
        raise ValueError("balance and recovery curricula are mutually exclusive")
    if balance_curriculum is None:
        if balance_pass_replay is not None:
            raise ValueError("balance pass replay requires a curriculum")
    else:
        pass_replay_enabled = (
            balance_curriculum.pass_replay_update_period > 0
        )
        if pass_replay_enabled != (balance_pass_replay is not None):
            raise ValueError("balance pass replay contract is incomplete")
        if balance_pass_replay is not None and (
            balance_pass_replay.sequence_length != replay.sequence_length
            or balance_pass_replay.recurrent_burn_in
            != replay.recurrent_burn_in
            or balance_pass_replay.n_step_return != replay.n_step_return
        ):
            raise ValueError("balance pass replay recurrent contract drifted")
        if balance_pass_replay is not None:
            balance_pass_replay.absorb_recent_passes(
                replay,
                max_examples=balance_curriculum.pass_replay_max_examples,
            )
        if (
            balance_curriculum.outcome_contrast is not None
            and near_blow_loss_threshold is None
        ):
            raise ValueError(
                "balance outcome contrast requires a near-blow boundary"
            )
    recovery_components = (
        recovery_value_policy,
        recovery_value_environment,
        recovery_value_store,
        recovery_value_source_identity_sha256,
    )
    if recovery_curriculum is None:
        if (
            any(component is not None for component in recovery_components)
            or recovery_success_replay is not None
            or healthy_pass_replay is not None
            or post_recovery_contrast_replay is not None
        ):
            raise ValueError("recovery-value components require a curriculum")
    elif (
        any(component is None for component in recovery_components)
        or len(str(recovery_value_source_identity_sha256)) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(recovery_value_source_identity_sha256)
        )
    ):
        raise ValueError("recovery-value supervision contract is incomplete")
    if recovery_curriculum is not None:
        pass_replay_enabled = (
            recovery_curriculum.recovery_success_replay_update_period > 0
        )
        if pass_replay_enabled != (recovery_success_replay is not None):
            raise ValueError("recovery pass replay contract is incomplete")
        if recovery_success_replay is not None and (
            recovery_success_replay.sequence_length != replay.sequence_length
            or recovery_success_replay.recurrent_burn_in
            != replay.recurrent_burn_in
            or recovery_success_replay.n_step_return != replay.n_step_return
        ):
            raise ValueError("recovery pass replay recurrent contract drifted")
        healthy_replay_enabled = (
            recovery_curriculum.healthy_pass_replay_update_period > 0
        )
        if healthy_replay_enabled != (healthy_pass_replay is not None):
            raise ValueError("healthy pass replay contract is incomplete")
        if healthy_pass_replay is not None and (
            healthy_pass_replay.sequence_length != replay.sequence_length
            or healthy_pass_replay.recurrent_burn_in
            != replay.recurrent_burn_in
            or healthy_pass_replay.n_step_return != replay.n_step_return
        ):
            raise ValueError("healthy pass replay recurrent contract drifted")
        contrast_replay_enabled = (
            recovery_curriculum.post_recovery_contrast_replay_update_period > 0
        )
        if contrast_replay_enabled != (post_recovery_contrast_replay is not None):
            raise ValueError(
                "post-recovery contrast replay contract is incomplete"
            )
        if post_recovery_contrast_replay is not None and (
            post_recovery_contrast_replay.sequence_length
            != replay.sequence_length
            or post_recovery_contrast_replay.recurrent_burn_in
            != replay.recurrent_burn_in
            or post_recovery_contrast_replay.n_step_return
            != replay.n_step_return
        ):
            raise ValueError(
                "post-recovery contrast replay recurrent contract drifted"
            )
        if recovery_success_replay is not None:
            recovery_success_replay.absorb_recent_successful_recoveries(
                replay,
                max_examples=(
                    recovery_curriculum.recovery_success_replay_max_examples
                ),
            )
        if healthy_pass_replay is not None:
            if recovery_success_replay is not None:
                healthy_pass_replay.absorb_recent_healthy_passes(
                    recovery_success_replay,
                    max_examples=(
                        recovery_curriculum.healthy_pass_replay_max_examples
                    ),
                )
            healthy_pass_replay.absorb_recent_healthy_passes(
                replay,
                max_examples=(
                    recovery_curriculum.healthy_pass_replay_max_examples
                ),
            )
        if post_recovery_contrast_replay is not None:
            post_recovery_contrast_replay.absorb_recent_post_recovery_contrasts(
                replay,
                max_examples=(
                    recovery_curriculum
                    .post_recovery_contrast_replay_max_examples
                ),
            )
    if not 0 <= teacher_loss_end_scale <= 1:
        raise ValueError("teacher loss end scale must be between zero and one")
    if not (
        0
        <= teacher_guidance_dropout_start
        <= teacher_guidance_dropout_end
        <= 1
    ):
        raise ValueError("teacher guidance dropout schedule is invalid")
    if not 0 < teacher_autonomy_start_fraction <= 1:
        raise ValueError("teacher autonomy start fraction is invalid")
    if entry_supervision_autonomy_start_fraction is None:
        entry_supervision_autonomy_start_fraction = (
            teacher_autonomy_start_fraction
        )
    if (
        isinstance(entry_supervision_autonomy_start_fraction, bool)
        or not isinstance(
            entry_supervision_autonomy_start_fraction,
            (int, float),
        )
        or not teacher_autonomy_start_fraction
        <= float(entry_supervision_autonomy_start_fraction)
        <= 1.0
    ):
        raise ValueError(
            "entry supervision autonomy start fraction is invalid"
        )
    entry_supervision_autonomy_start_fraction = float(
        entry_supervision_autonomy_start_fraction
    )
    paired_recurrent_a_plus = (
        getattr(agent, "regime_selectivity_semantics", None)
        == PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
    )
    paired_context_channels = (*EXPANSION_CHANNELS, *REGIME_TEACHER_CHANNELS)
    if paired_recurrent_a_plus and (
        teacher_channels is None
        or tuple(teacher_channels[: len(paired_context_channels)])
        != paired_context_channels
        or entry_action_lookup is None
        or entry_action_metadata_lookup is None
        or replay.entry_opportunity_side_balance
        != "paired_recurrent_long_short_v1"
    ):
        raise ValueError(
            "paired recurrent A+ training evidence contract is incomplete"
        )
    ticker_schedule = _balanced_ticker_schedule(
        episode_tickers,
        episodes=episodes,
        seed=ticker_seed,
    )
    if checkpoint_every_episodes < 0:
        raise ValueError("checkpoint interval cannot be negative")
    if checkpoint_every_episodes and checkpoint_callback is None:
        raise ValueError("checkpoint callback is required when checkpointing is enabled")
    if training_health_callback is not None and checkpoint_callback is None:
        raise ValueError(
            "checkpoint callback is required for training health stops"
        )
    progress = resume or TrainingProgress()
    if progress.completed_episodes > episodes:
        raise ValueError("resume progress exceeds the episode ceiling")
    if (
        budget_mode == "environment_steps"
        and progress.environment_steps > minimum_environment_steps
    ):
        raise ValueError("resume progress exceeds the environment-step budget")
    if progress.short_circuit_reason is not None:
        return progress.result()
    for episode_index in range(progress.completed_episodes, episodes):
        reset_options = {}
        if ticker_schedule is not None:
            reset_options["ticker"] = ticker_schedule[episode_index]
        if recovery_curriculum is not None:
            reset_options["challenge_start_state"] = (
                recovery_curriculum.start_state
            )
        elif balance_curriculum is not None:
            reset_options["challenge_start_state"] = (
                balance_curriculum.start_state(episode_index)
            )
        if not reset_options:
            observation, reset_info = environment.reset()
        else:
            observation, reset_info = environment.reset(
                options=reset_options
            )
        valid = tuple(reset_info["valid_actions"])
        decision_index = int(reset_info.get("start", 0))
        episode_start_index = decision_index
        episode_end_index = reset_info.get("end")
        if budget_mode == "episodes":
            if (
                isinstance(episode_end_index, bool)
                or not isinstance(episode_end_index, (int, np.integer))
                or int(episode_end_index) <= episode_start_index
            ):
                raise ValueError(
                    "episode-budget schedules require causal start/end boundaries"
                )
            episode_end_index = int(episode_end_index)
        episode_ticker = str(reset_info.get("ticker", ""))
        hidden = None
        handoff_policy = (
            None
            if recovery_curriculum is None
            else RecoveryHandoffPolicy(
                agent,
                normal_policy=recovery_value_policy,
            )
        )
        policy_state_decisions = {"recovery": 0, "normal": 0}
        policy_state_handoffs = {
            "recovery_to_normal": 0,
            "normal_to_recovery": 0,
        }
        previous_policy_state: str | None = None
        transitions = []
        action_counts = {action: 0 for action in Action}
        greedy_flat_action_counts = {action: 0 for action in Action}
        greedy_flat_probe_count = 0
        greedy_flat_entry_advantages: list[float] = []
        selected_entry_teacher_targets: list[tuple[float, float]] = []
        selected_teacher_targets: list[np.ndarray] = []
        visible_regime_entry_context: dict[int, dict[str, object]] = {}
        teacher_guidance_eligible_decisions = 0
        teacher_guidance_visible_decisions = 0
        entry_action_target_counts = {
            Action.WAIT: 0,
            Action.ENTER_LONG_1: 0,
            Action.ENTER_SHORT_1: 0,
        }
        entry_timing_flat_actions: dict[int, Action] = {}
        entry_timing_metadata: dict[int, object] = {}
        recovery_entries_used = 0
        total_reward = 0.0
        # Legacy recipes decay against market interaction. Explicit episode
        # budgets instead use the declared challenge count and causal position
        # within each challenge, independent of variable episode length.
        step_progress = (
            episode_index / episodes
            if budget_mode == "episodes"
            else min(1.0, progress.environment_steps / minimum_environment_steps)
        )
        teacher_schedule_progress = min(
            1.0, step_progress / teacher_autonomy_start_fraction
        )
        entry_action_schedule_progress = min(
            1.0,
            step_progress / entry_supervision_autonomy_start_fraction,
        )
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * step_progress
        management_epsilon = (
            management_epsilon_start
            + (management_epsilon_end - management_epsilon_start) * step_progress
        )
        teacher_weight_scale = 1.0 + (
            teacher_loss_end_scale - 1.0
        ) * teacher_schedule_progress
        entry_action_weight_scale = 1.0
        teacher_guidance_dropout_probability = teacher_guidance_dropout_start
        terminal_info = reset_info
        step_index = 0
        while True:
            if budget_mode == "episodes":
                assert isinstance(episode_end_index, int)
                within_episode_progress = min(
                    1.0,
                    max(
                        0.0,
                        (decision_index - episode_start_index)
                        / (episode_end_index - episode_start_index),
                    ),
                )
                decision_progress = min(
                    1.0,
                    (episode_index + within_episode_progress) / episodes,
                )
                epsilon = epsilon_start + (
                    epsilon_end - epsilon_start
                ) * decision_progress
                management_epsilon = management_epsilon_start + (
                    management_epsilon_end - management_epsilon_start
                ) * decision_progress
                teacher_schedule_progress = min(
                    1.0,
                    decision_progress / teacher_autonomy_start_fraction,
                )
                entry_action_schedule_progress = min(
                    1.0,
                    decision_progress
                    / entry_supervision_autonomy_start_fraction,
                )
                teacher_weight_scale = 1.0 + (
                    teacher_loss_end_scale - 1.0
                ) * teacher_schedule_progress
                entry_action_weight_scale = 1.0
                teacher_guidance_dropout_probability = (
                    teacher_guidance_dropout_start
                    + (
                        teacher_guidance_dropout_end
                        - teacher_guidance_dropout_start
                    )
                    * teacher_schedule_progress
                )
            if step_index and step_index % recurrent_horizon == 0:
                hidden = None
                if handoff_policy is not None:
                    handoff_policy.reset()
            action_epsilon = (
                management_epsilon
                if set(valid) == {Action.HOLD, Action.CLOSE}
                else epsilon
            )
            flat_actions = {
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            }
            diagnostic_probe = (
                flat_actions.issubset(valid)
                and (progress.environment_steps + step_index)
                % greedy_diagnostic_interval_steps == 0
            )
            if handoff_policy is None:
                action, hidden, action_values = agent.select_action(
                    observation,
                    hidden=hidden,
                    valid_actions=valid,
                    epsilon=action_epsilon,
                    return_action_values=diagnostic_probe,
                )
            else:
                action, action_values, policy_state = handoff_policy.select_action(
                    observation,
                    valid_actions=valid,
                    realized_pnl=float(terminal_info["realized_pnl"]),
                    recovery_epsilon=action_epsilon,
                    return_action_values=diagnostic_probe,
                )
                policy_state_decisions[policy_state] += 1
                if (
                    previous_policy_state is not None
                    and policy_state != previous_policy_state
                ):
                    policy_state_handoffs[
                        f"{previous_policy_state}_to_{policy_state}"
                    ] += 1
                previous_policy_state = policy_state
            action_counts[Action(action)] += 1
            if flat_actions.issubset(valid):
                entry_timing_flat_actions[decision_index] = Action(action)
            if diagnostic_probe:
                assert action_values is not None
                values = np.asarray(action_values, dtype=np.float64)
                greedy_action = max(valid, key=lambda item: values[int(item)])
                greedy_flat_action_counts[Action(greedy_action)] += 1
                greedy_flat_probe_count += 1
                greedy_flat_entry_advantages.append(
                    max(
                        values[int(Action.ENTER_LONG_1)],
                        values[int(Action.ENTER_SHORT_1)],
                    ) - values[int(Action.WAIT)]
                )
            next_observation, reward, terminated, _, info = environment.step(action)
            recovery_entries_used += int(info.get("recovery_entry_used", False))
            next_valid = tuple(info["valid_actions"])
            if budget_mode == "environment_steps":
                decision_progress = min(
                    1.0,
                    (progress.environment_steps + step_index)
                    / minimum_environment_steps,
                )
                decision_teacher_progress = min(
                    1.0, decision_progress / teacher_autonomy_start_fraction
                )
                teacher_guidance_dropout_probability = (
                    teacher_guidance_dropout_start
                    + (
                        teacher_guidance_dropout_end
                        - teacher_guidance_dropout_start
                    )
                    * decision_teacher_progress
                )
            teacher_visible = _teacher_guidance_is_visible(
                seed=ticker_seed,
                episode_index=episode_index,
                ticker=episode_ticker,
                decision_index=decision_index,
                dropout_probability=teacher_guidance_dropout_probability,
            )
            guidance_eligible = bool(
                teacher_lookup is not None
                and teacher_weight_scale > 0.0
            )
            teacher_guidance_eligible_decisions += int(guidance_eligible)
            teacher_guidance_visible_decisions += int(
                guidance_eligible and teacher_visible
            )
            teacher_target = (
                teacher_lookup(episode_ticker, decision_index)
                if teacher_lookup is not None
                else None
            )
            entry_action_target = (
                entry_action_lookup(episode_ticker, decision_index)
                if (
                    entry_action_lookup is not None
                    and flat_actions.issubset(valid)
                )
                else None
            )
            entry_action_metadata = (
                entry_action_metadata_lookup(episode_ticker, decision_index)
                if entry_action_metadata_lookup is not None
                else None
            )
            if entry_action_metadata is not None:
                entry_timing_metadata[decision_index] = entry_action_metadata
            regime_selectivity_headroom_fraction = None
            if (
                float(
                    getattr(agent, "regime_selectivity_loss_weight", 0.0)
                )
                > 0.0
                and teacher_target is not None
                and entry_action_target is not None
                and flat_actions.issubset(valid)
            ):
                if "mll_headroom_fraction" not in terminal_info:
                    raise ValueError(
                        "balance-aware Regime selectivity requires "
                        "decision-time MLL headroom"
                    )
                # Profits can move account headroom above one full MLL budget.
                # Selectivity pressure is already defined on [0, 1], so retain
                # the exact zero-risk meaning without rejecting valid episodes.
                regime_selectivity_headroom_fraction = (
                    _bounded_regime_selectivity_headroom(
                        terminal_info["mll_headroom_fraction"]
                    )
                )
            if (
                teacher_target is not None
                and teacher_channels is not None
                and flat_actions.issubset(valid)
                and "mll_headroom_fraction" in terminal_info
                and all(
                    channel in teacher_channels
                    for channel in (
                        "chop_no_trend_probability",
                        "chop_end_transition_probability",
                        "expansion_trend_probability",
                    )
                )
            ):
                visible_regime_entry_context[decision_index] = {
                    "teacher": np.asarray(
                        teacher_target, dtype=np.float32
                    ).reshape(-1).copy(),
                    "channels": tuple(teacher_channels),
                    "headroom_fraction": _bounded_regime_selectivity_headroom(
                        terminal_info["mll_headroom_fraction"]
                    ),
                }
            if entry_action_target is not None:
                entry_action_target = Action(entry_action_target)
                if entry_action_target not in entry_action_target_counts:
                    raise ValueError("entry action target is not a flat action")
                entry_action_target_counts[entry_action_target] += 1
            entry_opportunity_priority = 0.0
            if teacher_target is not None:
                entry_opportunity_priority = max(
                    float(teacher_target[0]) * float(teacher_target[1]),
                    float(teacher_target[2]) * float(teacher_target[3]),
                )
            if entry_action_target in {
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            }:
                entry_opportunity_priority = max(entry_opportunity_priority, 1.0)
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
            paired_a_plus_context = None
            paired_a_plus_side = None
            paired_a_plus_economic_win = None
            if paired_recurrent_a_plus and entry_action_target is not None:
                assert entry_action_metadata_lookup is not None
                (
                    paired_a_plus_context,
                    paired_a_plus_side,
                    paired_a_plus_economic_win,
                ) = _paired_a_plus_transition_evidence(
                    teacher_target=teacher_target,
                    teacher_channels=teacher_channels,
                    entry_action_target=entry_action_target,
                    metadata=entry_action_metadata,
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
                teacher_imitation_visible=teacher_visible,
                entry_action_target=entry_action_target,
                regime_selectivity_headroom_fraction=(
                    regime_selectivity_headroom_fraction
                ),
                recovery_active=(
                    recovery_curriculum is not None
                    and float(terminal_info["realized_pnl"]) < 0.0
                ),
                safety_priority=float(
                    info.get("mll_proximity_penalty", 0.0)
                ),
                entry_opportunity_priority=entry_opportunity_priority,
                source_decision_index=decision_index,
                paired_a_plus_context=paired_a_plus_context,
                paired_a_plus_side=paired_a_plus_side,
                paired_a_plus_economic_win=paired_a_plus_economic_win,
            ))
            total_reward += reward
            observation, valid = next_observation, next_valid
            terminal_info = info
            decision_index = int(info.get("fill_index", decision_index + 1))
            step_index += 1
            episode_steps = step_index
            if terminated:
                break
        update_progress = (
            (episode_index + 1) / episodes
            if budget_mode == "episodes"
            else min(
                1.0,
                (progress.environment_steps + episode_steps)
                / minimum_environment_steps,
            )
        )
        teacher_schedule_progress = min(
            1.0, update_progress / teacher_autonomy_start_fraction
        )
        entry_action_schedule_progress = min(
            1.0,
            update_progress / entry_supervision_autonomy_start_fraction,
        )
        teacher_weight_scale = 1.0 + (
            teacher_loss_end_scale - 1.0
        ) * teacher_schedule_progress
        entry_action_weight_scale = 1.0
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
        closed_trade_receipts = getattr(
            environment, "closed_trade_receipts", None
        )
        regime_trade_economics = _regime_trade_economics(
            () if closed_trade_receipts is None else closed_trade_receipts(),
            visible_regime_entry_context,
            episode_outcome=outcome,
        )
        replay_transitions = tuple(transitions)
        recovery_value_target_generated = False
        recovery_value_target_added = False
        recovery_value_target_discriminative = False
        recovery_value_target_side: str | None = None
        recovery_value_target_economic_success: bool | None = None
        if (
            recovery_curriculum is not None
            and episode_index % recovery_curriculum.target_every_episodes == 0
        ):
            assert recovery_value_policy is not None
            assert recovery_value_environment is not None
            assert recovery_value_store is not None
            assert recovery_value_source_identity_sha256 is not None
            causal_prefix = select_recovery_target_prefix(
                replay_transitions,
                recovery_succeeded=(
                    bool(terminal_info.get("recovery_success", False))
                    and terminal_pnl >= (
                        recovery_curriculum.start_state.recovery_success_pnl
                    )
                ),
            )
            if causal_prefix is not None:
                target_policy_identity = hashlib.sha256(
                    (
                        f"{recovery_value_source_identity_sha256}\0"
                        f"recovery-updates={getattr(agent, 'optimizer_updates', 0)}"
                    ).encode("ascii")
                ).hexdigest()
                recovery_target = build_recovery_value_target(
                    agent,
                    recovery_value_environment,
                    normal_policy=recovery_value_policy,
                    reset_options={
                        "ticker": episode_ticker,
                        "start": episode_start_index,
                        "challenge_start_state": (
                            recovery_curriculum.start_state
                        ),
                    },
                    recurrent_horizon=recurrent_horizon,
                    start_pnl=(
                        recovery_curriculum.start_state.realized_pnl
                    ),
                    recovery_success_pnl=(
                        recovery_curriculum.start_state.recovery_success_pnl
                    ),
                    source_role="training",
                    source_identity_sha256=(
                        target_policy_identity
                    ),
                    causal_prefix=causal_prefix,
                )
                recovery_value_target_generated = True
                recovery_value_target_discriminative = (
                    recovery_target.is_discriminative
                )
                recovery_value_target_added = recovery_value_store.add(
                    recovery_target
                )
                recovery_value_target_side = (
                    None
                    if recovery_target.anchor_action is None
                    else recovery_target.anchor_action.name
                )
                recovery_value_target_economic_success = (
                    recovery_target.anchor_economic_success
                )
        if (
            replay.regime_wait_sequence_fraction > 0.0
            or replay.regime_wait_sequence_update_period > 0
        ):
            compiler = getattr(agent, "regime_selectivity", None)
            if compiler is None:
                raise ValueError(
                    "Regime WAIT replay requires authenticated selectivity"
                )
            replay_transitions = _with_regime_wait_replay_priorities(
                replay_transitions,
                compiler,
            )
        completed_episode = Episode(
            episode_id=f"historical-{episode_index}-{time.time_ns()}",
            ticker=str(terminal_info["ticker"]),
            outcome=outcome,
            primary_side=str(terminal_info["primary_side"]),
            ended_at_ns=time.time_ns(),
            transitions=replay_transitions,
            terminal_pnl=terminal_pnl,
        )
        replay.add(completed_episode)
        balance_pass_replay_promoted_passes = 0
        if outcome == "pass" and balance_pass_replay is not None:
            assert balance_curriculum is not None
            balance_pass_replay_promoted_passes = (
                balance_pass_replay.absorb_recent_passes(
                    replay,
                    max_examples=(
                        balance_curriculum.pass_replay_max_examples
                    ),
                )
            )
            if (
                balance_pass_replay_promoted_passes > 0
                and balance_pass_replay_callback is not None
            ):
                balance_pass_replay_callback(
                    balance_pass_replay.state_dict()
                )
        recovery_success_replay_promoted_episodes = 0
        recovery_success_replay_promoted_passes = 0
        healthy_pass_replay_promoted_passes = 0
        post_recovery_contrast_replay_promoted_episodes = 0
        if (
            outcome in {"pass", "timeout"}
            and recovery_success_replay is not None
        ):
            recovery_success_replay_promoted_episodes = (
                recovery_success_replay.absorb_recent_successful_recoveries(
                    replay,
                    max_examples=(
                        recovery_curriculum
                        .recovery_success_replay_max_examples
                    ),
                )
            )
            if outcome == "pass":
                recovery_success_replay_promoted_passes = (
                    recovery_success_replay_promoted_episodes
                )
        if outcome == "pass" and healthy_pass_replay is not None:
            healthy_pass_replay_promoted_passes = (
                healthy_pass_replay.absorb_recent_healthy_passes(
                    replay,
                    max_examples=(
                        recovery_curriculum.healthy_pass_replay_max_examples
                    ),
                )
            )
        if post_recovery_contrast_replay is not None:
            post_recovery_contrast_replay_promoted_episodes = (
                post_recovery_contrast_replay
                .absorb_recent_post_recovery_contrasts(
                    replay,
                    max_examples=(
                        recovery_curriculum
                        .post_recovery_contrast_replay_max_examples
                    ),
                )
            )
        episode_losses = []
        episode_rl_losses = []
        episode_teacher_losses = []
        episode_entry_search_losses = []
        episode_entry_action_losses = []
        episode_entry_action_rows = []
        learner_diagnostics: dict[str, list[float]] = {
            key: []
            for key in (
                "gradient_norm",
                "gradient_conflict_primary_norm",
                "gradient_conflict_safety_norm",
                "gradient_conflict_opportunity_norm",
                "gradient_conflict_pre_projection_cosine",
                "gradient_conflict_post_projection_cosine",
                "gradient_conflict_projected",
                "economic_boundary_backtracks",
                "economic_boundary_count",
                "economic_boundary_active_constraint_count",
                "economic_boundary_initial_min_margin_delta",
                "economic_boundary_final_min_margin_delta",
                "economic_boundary_final_min_required_headroom",
                "economic_boundary_long_winner_min_margin_delta",
                "economic_boundary_short_winner_min_margin_delta",
                "economic_boundary_failed_long_min_margin_delta",
                "economic_boundary_failed_short_min_margin_delta",
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
                "healthy_entry_policy_retention_loss",
                "healthy_entry_policy_retention_rows",
                "entry_action_target_wait_rows",
                "entry_action_target_long_rows",
                "entry_action_target_short_rows",
                "entry_action_prediction_wait_rows",
                "entry_action_prediction_long_rows",
                "entry_action_prediction_short_rows",
                "entry_action_correct_wait_rows",
                "entry_action_correct_long_rows",
                "entry_action_correct_short_rows",
                "regime_selectivity_loss",
                "regime_selectivity_supervised_rows",
                "regime_selectivity_target_wait_mean",
                "regime_selectivity_low_headroom_rows",
                "regime_selectivity_low_headroom_wait_mean",
                "regime_selectivity_dominant_chop_rows",
                "regime_selectivity_dominant_chop_wait_mean",
                "regime_selectivity_association_loss",
                "regime_selectivity_association_active",
                "regime_selectivity_association_skipped",
                "regime_selectivity_side_conditioned_loss",
                "regime_selectivity_side_conditioned_active_sides",
                "regime_selectivity_paired_a_plus_loss",
                "regime_selectivity_paired_a_plus_active_groups",
                "regime_selectivity_paired_a_plus_pair_count",
                "regime_selectivity_paired_a_plus_pair_mass",
                "regime_selectivity_paired_a_plus_good_advantage_sum",
                "regime_selectivity_paired_a_plus_bad_advantage_sum",
                "regime_selectivity_dead_wait_minus_"
                "transition_positive_model_wait",
                "recovery_value_loss",
                "recovery_value_rows",
                "recovery_value_top1_concurrence",
                "recovery_wait_minus_long_q",
                "recovery_wait_minus_short_q",
                "recovery_action_margin_loss",
                "recovery_recurrent_rows",
                "challenge_return_self_imitation_rows",
                "challenge_return_self_imitation_bonus_sum",
                "challenge_return_self_imitation_added_clip_rows",
                *(
                    f"challenge_return_self_imitation_{action}_{field}"
                    for action in ("wait", "long", "short")
                    for field in ("rows", "bonus_sum")
                ),
                *(
                    f"regime_selectivity_association_{cohort}_{field}"
                    for cohort in _REGIME_ASSOCIATION_COHORTS
                    for field in _REGIME_ASSOCIATION_ADDITIVE_FIELDS
                ),
                *(
                    "regime_selectivity_paired_a_plus_"
                    f"{side}_{field}"
                    for side in _PAIRED_A_PLUS_SIDES
                    for field in (
                        "pair_count",
                        "pair_mass",
                        "loss_sum",
                        "good_advantage_sum",
                        "bad_advantage_sum",
                        "winner_population_weight_sum",
                        "failure_population_weight_sum",
                    )
                ),
                *(
                    "regime_selectivity_paired_a_plus_"
                    f"{side}_{regime}_{field}"
                    for side in _PAIRED_A_PLUS_SIDES
                    for regime in _PAIRED_A_PLUS_REGIMES
                    for field in _PAIRED_A_PLUS_GROUP_FIELDS
                ),
                *(
                    f"regime_selectivity_{stratum}_{field}"
                    for stratum in _PERSISTENT_REGIME_SELECTIVITY_STRATA
                    for field in _PERSISTENT_REGIME_SELECTIVITY_ADDITIVE_FIELDS
                ),
                *(
                    f"regime_selectivity_transition_positive_{side}_{field}"
                    for side in _TRANSITION_POSITIVE_SIDE_FIELDS
                    for field in (
                        "rows",
                        "declared_side_probability_sum",
                    )
                ),
                *(
                    f"regime_selectivity_{stratum}_{field}"
                    for stratum in _REGIME_SELECTIVITY_STRATA
                    for field in _REGIME_SELECTIVITY_ADDITIVE_FIELDS
                ),
                *(
                    f"regime_teacher_channel_{channel}_{field}"
                    for channel in REGIME_TEACHER_CHANNELS
                    if teacher_channels is not None
                    and channel in teacher_channels
                    for field in _REGIME_CHANNEL_ADDITIVE_FIELDS
                ),
                *(
                    f"entry_balance_{action}_{field}"
                    for action in _ENTRY_BALANCE_ACTION_NAMES
                    for field in (
                        *_ENTRY_BALANCE_ADDITIVE_FIELDS,
                        "configured_weight",
                    )
                ),
                *(
                    f"regime_entry_conflict_{side}_{field}"
                    for side in ("long", "short")
                    for field in _REGIME_ENTRY_CONFLICT_FIELDS
                ),
            )
        }
        for class_index, action in enumerate(_ENTRY_BALANCE_ACTION_NAMES):
            learner_diagnostics[
                f"entry_balance_{action}_configured_weight"
            ].append(float(
                getattr(agent, "entry_action_class_weights", (1.0, 1.0, 1.0))[
                    class_index
                ]
            ))
        balance_pass_replay_sequences = 0
        balance_outcome_contrast_pairs = 0
        recovery_success_replay_sequences = 0
        healthy_pass_replay_sequences = 0
        post_recovery_contrast_pairs = 0
        if len(replay) >= warmup_episodes:
            def train_replay_batch(
                batch: Sequence[Sequence[Transition]],
                update_index: int,
            ) -> None:
                nonlocal balance_pass_replay_sequences
                nonlocal balance_outcome_contrast_pairs
                nonlocal recovery_success_replay_sequences
                nonlocal healthy_pass_replay_sequences
                nonlocal post_recovery_contrast_pairs
                if (
                    balance_curriculum is not None
                    and balance_pass_replay is not None
                    and (update_index + 1)
                    % balance_curriculum.pass_replay_update_period
                    == 0
                ):
                    pass_sequences = (
                        balance_pass_replay
                        .sample_balance_pass_entry_sequences(
                            1,
                            max_examples=(
                                balance_curriculum.pass_replay_max_examples
                            ),
                        )
                    )
                    if not pass_sequences:
                        raise ValueError(
                            "balance pass replay lacks an economic winner entry"
                        )
                    batch = tuple(batch) + pass_sequences
                    balance_pass_replay_sequences += len(pass_sequences)
                if (
                    balance_curriculum is not None
                    and balance_curriculum.outcome_contrast is not None
                    and (update_index + 1)
                    % balance_curriculum.outcome_contrast.update_period
                    == 0
                ):
                    existing_pair_ids = [
                        int(transition.paired_a_plus_pair_id)
                        for sequence in batch
                        for transition in sequence
                        if transition.paired_a_plus_pair_id is not None
                    ]
                    assert near_blow_loss_threshold is not None
                    contrast_sequences = (
                        replay.sample_balance_outcome_contrast_pairs(
                            1,
                            near_blow_pnl=-near_blow_loss_threshold,
                            max_examples=(
                                balance_curriculum
                                .outcome_contrast.max_examples
                            ),
                            pair_id_start=(
                                max(existing_pair_ids, default=-1) + 1
                            ),
                            challenge_return_discount=(
                                float(getattr(
                                    agent,
                                    "challenge_return_discount",
                                    1.0,
                                ))
                                if float(getattr(
                                    agent,
                                    "challenge_return_self_imitation_weight",
                                    0.0,
                                )) > 0.0
                                else None
                            ),
                        )
                    )
                    if contrast_sequences:
                        batch = tuple(batch) + contrast_sequences
                        balance_outcome_contrast_pairs += (
                            len(contrast_sequences) // 2
                        )
                if (
                    recovery_curriculum is not None
                    and recovery_success_replay is not None
                    and (update_index + 1)
                    % recovery_curriculum.recovery_success_replay_update_period
                    == 0
                ):
                    recovery_sequences = (
                        recovery_success_replay
                        .sample_successful_recovery_sequences(
                            1,
                            max_examples=(
                                recovery_curriculum
                                .recovery_success_replay_max_examples
                            ),
                        )
                    )
                    if recovery_sequences:
                        batch = tuple(batch) + recovery_sequences
                        recovery_success_replay_sequences += len(
                            recovery_sequences
                        )
                if (
                    recovery_curriculum is not None
                    and healthy_pass_replay is not None
                    and (update_index + 1)
                    % recovery_curriculum.healthy_pass_replay_update_period
                    == 0
                ):
                    healthy_sequences = (
                        healthy_pass_replay.sample_healthy_pass_sequences(
                            1,
                            max_examples=(
                                recovery_curriculum
                                .healthy_pass_replay_max_examples
                            ),
                        )
                    )
                    if healthy_sequences:
                        batch = tuple(batch) + healthy_sequences
                        healthy_pass_replay_sequences += len(
                            healthy_sequences
                        )
                if (
                    recovery_curriculum is not None
                    and post_recovery_contrast_replay is not None
                    and (update_index + 1)
                    % (
                        recovery_curriculum
                        .post_recovery_contrast_replay_update_period
                    )
                    == 0
                ):
                    existing_pair_ids = [
                        int(transition.paired_a_plus_pair_id)
                        for sequence in batch
                        for transition in sequence
                        if transition.paired_a_plus_pair_id is not None
                    ]
                    contrast_sequences = (
                        post_recovery_contrast_replay
                        .sample_post_recovery_contrast_pairs(
                            1,
                            max_examples=(
                                recovery_curriculum
                                .post_recovery_contrast_replay_max_examples
                            ),
                            pair_id_start=(
                                max(existing_pair_ids, default=-1) + 1
                            ),
                        )
                    )
                    if contrast_sequences:
                        batch = tuple(batch) + contrast_sequences
                        post_recovery_contrast_pairs += (
                            len(contrast_sequences) // 2
                        )
                recovery_target = (
                    recovery_value_store.sample_balanced()
                    if recovery_value_store is not None
                    and len(recovery_value_store) > 0
                    else None
                )
                train_kwargs: dict[str, object] = {
                    "teacher_weight_scale": teacher_weight_scale,
                    "entry_action_weight_scale": entry_action_weight_scale,
                }
                if recovery_curriculum is not None:
                    train_kwargs.update({
                        "recovery_target": recovery_target,
                        "recovery_value_loss_weight": (
                            recovery_curriculum.recovery_value_loss_weight
                        ),
                        "recovery_value_temperature": (
                            recovery_curriculum.recovery_value_temperature
                        ),
                        "recovery_action_margin": (
                            recovery_curriculum.recovery_action_margin
                        ),
                        "retain_nonnegative_entry_policy": (
                            recovery_curriculum.retain_nonnegative_entry_policy
                        ),
                    })
                episode_losses.append(agent.train_batch(batch, **train_kwargs))
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
                if "entry_action_loss" in train_metrics:
                    episode_entry_action_losses.append(
                        float(train_metrics["entry_action_loss"])
                    )
                if "entry_action_supervised_rows" in train_metrics:
                    episode_entry_action_rows.append(
                        float(train_metrics["entry_action_supervised_rows"])
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
                        train_replay_batch(batch, update_index)
            else:
                for update_index in range(updates_per_episode):
                    train_replay_batch(
                        replay.sample(batch_sequences),
                        update_index,
                    )
        trade_count = int(terminal_info.get("trade_count", 0))
        recent_outcomes = progress.recent_outcomes if collapse_window_episodes else ()
        recent_hold_bars = (
            progress.recent_average_hold_bars if collapse_window_episodes else ()
        )
        recent_close_rates = (
            progress.recent_voluntary_close_rates if collapse_window_episodes else ()
        )
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
        outcome_short_circuit_boundary_reached = (
            budget_mode == "environment_steps"
            and short_circuit_minimum_environment_steps is not None
            and progress.environment_steps >= short_circuit_minimum_environment_steps
        ) or (
            budget_mode == "episodes"
            and short_circuit_minimum_episodes is not None
            and progress.completed_episodes >= short_circuit_minimum_episodes
        )
        if outcome_short_circuit_boundary_reached:
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
        collapse_short_circuit_boundary_reached = (
            (
                budget_mode == "environment_steps"
                and (
                    short_circuit_minimum_environment_steps is None
                    or progress.environment_steps
                    >= short_circuit_minimum_environment_steps
                )
            )
            or (
                budget_mode == "episodes"
                and (
                    short_circuit_minimum_episodes is None
                    or progress.completed_episodes
                    >= short_circuit_minimum_episodes
                )
            )
        )
        if (
            collapse_window_episodes
            and collapse_short_circuit_boundary_reached
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
        if (
            episode_diagnostic_callback is not None
            or training_health_callback is not None
        ):
            challenge_return_rows = int(round(sum(
                learner_diagnostics[
                    "challenge_return_self_imitation_rows"
                ]
            )))
            challenge_return_bonus_sum = float(sum(
                learner_diagnostics[
                    "challenge_return_self_imitation_bonus_sum"
                ]
            ))
            challenge_return_actions = {}
            for action_name, metric_name in (
                ("WAIT", "wait"),
                ("ENTER_LONG_1", "long"),
                ("ENTER_SHORT_1", "short"),
            ):
                rows = int(round(sum(learner_diagnostics[
                    f"challenge_return_self_imitation_{metric_name}_rows"
                ])))
                bonus_sum = float(sum(learner_diagnostics[
                    "challenge_return_self_imitation_"
                    f"{metric_name}_bonus_sum"
                ]))
                challenge_return_actions[action_name] = {
                    "rows": rows,
                    "bonus_sum": bonus_sum,
                    "bonus_mean": (
                        bonus_sum / rows if rows else None
                    ),
                }
            diagnostic = {
                "schema": "propevolve_episode_diagnostic_v1",
                "episode": progress.completed_episodes,
                "ticker": str(terminal_info["ticker"]),
                "outcome": outcome,
                "episode_kind": (
                    "recovery"
                    if recovery_curriculum is not None
                    else "balance_curriculum"
                    if balance_curriculum is not None
                    else "ordinary"
                ),
                "starting_realized_pnl": float(
                    reset_info.get("realized_pnl", 0.0)
                ),
                "policy_state_decisions": policy_state_decisions,
                "policy_state_handoffs": policy_state_handoffs,
                "reward": total_reward,
                "environment_steps": progress.environment_steps,
                "budget_mode": budget_mode,
                "budget_progress": update_progress,
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
                "recovery_entry_used": bool(recovery_entries_used),
                "recovery_entries_used": recovery_entries_used,
                "recovery_trade_closed": bool(
                    terminal_info.get("recovery_trade_closed", False)
                ),
                "recovery_success": bool(
                    terminal_info.get("recovery_success", False)
                ),
                "recovery_retained": bool(
                    terminal_info.get("recovery_retained", False)
                ),
                "recovery_relapsed": bool(
                    terminal_info.get("recovery_relapsed", False)
                ),
                "recovery_relapse_count": int(
                    terminal_info.get("recovery_relapse_count", 0)
                ),
                "first_recovery_index": terminal_info.get(
                    "first_recovery_index"
                ),
                "first_recovery_relapse_index": terminal_info.get(
                    "first_recovery_relapse_index"
                ),
                "post_recovery_min_realized_pnl": terminal_info.get(
                    "post_recovery_min_realized_pnl"
                ),
                "recovered_then_blown": bool(
                    terminal_info.get("recovered_then_blown", False)
                ),
                "recovery_wait_decisions": int(
                    terminal_info.get("recovery_wait_decisions", 0)
                ),
                "balance_pass_replay_sequences": (
                    balance_pass_replay_sequences
                ),
                "balance_outcome_contrast_pairs": (
                    balance_outcome_contrast_pairs
                ),
                "challenge_return_self_imitation": {
                    "rows": challenge_return_rows,
                    "bonus_sum": challenge_return_bonus_sum,
                    "bonus_mean": (
                        challenge_return_bonus_sum / challenge_return_rows
                        if challenge_return_rows else None
                    ),
                    "added_clip_rows": int(round(sum(
                        learner_diagnostics[
                            "challenge_return_self_imitation_added_clip_rows"
                        ]
                    ))),
                    "actions": challenge_return_actions,
                },
                "balance_pass_replay_promoted_passes": (
                    balance_pass_replay_promoted_passes
                ),
                "recovery_success_replay_sequences": (
                    recovery_success_replay_sequences
                ),
                "recovery_success_replay_promoted_passes": (
                    recovery_success_replay_promoted_passes
                ),
                "recovery_success_replay_promoted_episodes": (
                    recovery_success_replay_promoted_episodes
                ),
                "healthy_pass_replay_sequences": (
                    healthy_pass_replay_sequences
                ),
                "healthy_pass_replay_promoted_passes": (
                    healthy_pass_replay_promoted_passes
                ),
                "post_recovery_contrast_pairs": (
                    post_recovery_contrast_pairs
                ),
                "post_recovery_contrast_replay_promoted_episodes": (
                    post_recovery_contrast_replay_promoted_episodes
                ),
                "recovery_value_target_added": recovery_value_target_added,
                "recovery_value_target_generated": (
                    recovery_value_target_generated
                ),
                "recovery_value_target_discriminative": (
                    recovery_value_target_discriminative
                ),
                "recovery_value_target_side": recovery_value_target_side,
                "recovery_value_target_economic_success": (
                    recovery_value_target_economic_success
                ),
                "recovery_value_store_size": (
                    0
                    if recovery_value_store is None
                    else len(recovery_value_store)
                ),
                "near_blow_timeout": near_blow_timeout,
                "regime_trade_economics": regime_trade_economics,
                "primary_side": str(terminal_info.get("primary_side", "flat")),
                "entry_epsilon": epsilon,
                "management_epsilon": management_epsilon,
                "teacher_weight_scale": teacher_weight_scale,
                "entry_action_weight_scale": entry_action_weight_scale,
                "teacher_schedule_progress": teacher_schedule_progress,
                "entry_action_schedule_progress": (
                    entry_action_schedule_progress
                ),
                "guidance_phase": (
                    "autonomy" if teacher_weight_scale <= 0.0 else "guidance"
                ),
                "teacher_guidance_eligible_decisions": (
                    teacher_guidance_eligible_decisions
                ),
                "teacher_guidance_visible_decisions": (
                    teacher_guidance_visible_decisions
                ),
                "teacher_guidance_visible_fraction": (
                    teacher_guidance_visible_decisions
                    / teacher_guidance_eligible_decisions
                    if teacher_guidance_eligible_decisions else 0.0
                ),
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
                "mean_entry_action_loss": (
                    float(np.mean(episode_entry_action_losses))
                    if episode_entry_action_losses else None
                ),
                "mean_entry_action_supervised_rows": (
                    float(np.mean(episode_entry_action_rows))
                    if episode_entry_action_rows else None
                ),
                "entry_action_target_counts": {
                    action.name: entry_action_target_counts[action]
                    for action in entry_action_target_counts
                },
                "entry_timing_audit": (
                    audit_entry_timing_episode(
                        flat_actions=entry_timing_flat_actions,
                        metadata_by_decision=entry_timing_metadata,
                    )
                    if entry_action_metadata_lookup is not None
                    else None
                ),
                "sampled_entry_action_target_counts": {
                    action: int(round(sum(learner_diagnostics[key])))
                    for action, key in {
                        "WAIT": "entry_action_target_wait_rows",
                        "ENTER_LONG_1": "entry_action_target_long_rows",
                        "ENTER_SHORT_1": "entry_action_target_short_rows",
                    }.items()
                },
                "sampled_entry_action_prediction_counts": {
                    action: int(round(sum(learner_diagnostics[key])))
                    for action, key in {
                        "WAIT": "entry_action_prediction_wait_rows",
                        "ENTER_LONG_1": "entry_action_prediction_long_rows",
                        "ENTER_SHORT_1": "entry_action_prediction_short_rows",
                    }.items()
                },
                "sampled_entry_action_correct_counts": {
                    action: int(round(sum(learner_diagnostics[key])))
                    for action, key in {
                        "WAIT": "entry_action_correct_wait_rows",
                        "ENTER_LONG_1": "entry_action_correct_long_rows",
                        "ENTER_SHORT_1": "entry_action_correct_short_rows",
                    }.items()
                },
                "sampled_entry_action_recall": {
                    action: (
                        sum(learner_diagnostics[correct_key])
                        / sum(learner_diagnostics[target_key])
                        if sum(learner_diagnostics[target_key]) > 0.0
                        else None
                    )
                    for action, target_key, correct_key in (
                        ("WAIT", "entry_action_target_wait_rows", "entry_action_correct_wait_rows"),
                        ("ENTER_LONG_1", "entry_action_target_long_rows", "entry_action_correct_long_rows"),
                        ("ENTER_SHORT_1", "entry_action_target_short_rows", "entry_action_correct_short_rows"),
                    )
                },
                "sampled_entry_action_precision": {
                    action: (
                        sum(learner_diagnostics[correct_key])
                        / sum(learner_diagnostics[prediction_key])
                        if sum(learner_diagnostics[prediction_key]) > 0.0
                        else None
                    )
                    for action, prediction_key, correct_key in (
                        ("WAIT", "entry_action_prediction_wait_rows", "entry_action_correct_wait_rows"),
                        ("ENTER_LONG_1", "entry_action_prediction_long_rows", "entry_action_correct_long_rows"),
                        ("ENTER_SHORT_1", "entry_action_prediction_short_rows", "entry_action_correct_short_rows"),
                    )
                },
                "regime_selectivity": (
                    _regime_selectivity_episode_diagnostic(learner_diagnostics)
                ),
                "regime_teacher_channels": _regime_channel_episode_diagnostic(
                    learner_diagnostics,
                    tuple(
                        channel
                        for channel in REGIME_TEACHER_CHANNELS
                        if teacher_channels is not None
                        and channel in teacher_channels
                    ),
                ),
                "entry_action_balance": _entry_balance_diagnostic(
                    learner_diagnostics
                ),
                "regime_entry_conflict": _regime_entry_conflict_diagnostic(
                    learner_diagnostics
                ),
                "persistent_regime_selectivity": (
                    _persistent_regime_selectivity_diagnostic(
                        learner_diagnostics
                    )
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
                "greedy_flat_action_counts": {
                    action.name: greedy_flat_action_counts[action]
                    for action in Action
                },
                "greedy_flat_entry_rate": (
                    (
                        greedy_flat_action_counts[Action.ENTER_LONG_1]
                        + greedy_flat_action_counts[Action.ENTER_SHORT_1]
                    ) / greedy_flat_probe_count
                    if greedy_flat_probe_count else 0.0
                ),
                "greedy_flat_probe_count": greedy_flat_probe_count,
                "mean_greedy_best_entry_advantage": (
                    float(np.mean(greedy_flat_entry_advantages))
                    if greedy_flat_entry_advantages else None
                ),
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
            if training_health_callback is not None:
                health_reason = training_health_callback(progress, diagnostic)
                if health_reason is not None and (
                    not isinstance(health_reason, str) or not health_reason.strip()
                ):
                    raise ValueError(
                        "training health callback must return a nonempty reason"
                    )
                if health_reason is not None:
                    reasons = [
                        value
                        for value in (
                            progress.short_circuit_reason,
                            health_reason.strip(),
                        )
                        if value
                    ]
                    progress = replace(
                        progress,
                        short_circuit_reason="; ".join(dict.fromkeys(reasons)),
                    )
                    diagnostic["training_short_circuited"] = True
                    diagnostic["training_short_circuit_reason"] = (
                        progress.short_circuit_reason
                    )
            if episode_diagnostic_callback is not None:
                episode_diagnostic_callback(diagnostic)
        budget_status = (
            f"steps={progress.environment_steps:,}/{minimum_environment_steps:,}"
            if budget_mode == "environment_steps"
            else f"steps={progress.environment_steps:,}"
        )
        print(
            f"[train] episode={episode_index + 1}/{episodes} ticker={terminal_info['ticker']} "
            f"outcome={outcome} reward={total_reward:.4f} replay={len(replay)} "
            f"trades={int(terminal_info.get('trade_count', 0))} "
            f"WR={float(terminal_info.get('win_rate', 0.0)):.1%} "
            f"winR={float(terminal_info.get('avg_win_r', 0.0)):+.3f}R "
            f"balance={terminal_pnl:+.2f} "
            f"avg_balance={cumulative_average_balance:+.2f} {budget_status}",
            flush=True,
        )
        periodic_checkpoint_due = bool(
            checkpoint_every_episodes
            and (
                progress.completed_episodes % checkpoint_every_episodes == 0
                or (
                    budget_mode == "environment_steps"
                    and progress.environment_steps >= minimum_environment_steps
                )
                or (
                    budget_mode == "episodes"
                    and progress.completed_episodes >= episodes
                )
            )
        )
        if checkpoint_callback is not None and (
            periodic_checkpoint_due or progress.short_circuit_reason is not None
        ):
            checkpoint_callback(progress)
        if (
            progress.short_circuit_reason is not None
            or (
                budget_mode == "environment_steps"
                and progress.environment_steps >= minimum_environment_steps
            )
            or (
                budget_mode == "episodes"
                and progress.completed_episodes >= episodes
            )
        ):
            break
    if (
        progress.short_circuit_reason is None
        and budget_mode == "environment_steps"
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
    no_trade_patience_episodes: int = 0,
    greedy_diagnostic_interval_steps: int = 256,
    episode_diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
    normal_policy: RecurrentC51Agent | None = None,
    balance_curriculum: BalanceCurriculumSettings | None = None,
) -> TrainingResult:
    assert_teacher_free = getattr(agent, "assert_teacher_free", None)
    if assert_teacher_free is not None:
        assert_teacher_free()
    handoff_policy = (
        None
        if normal_policy is None
        else RecoveryHandoffPolicy(agent, normal_policy=normal_policy)
    )
    if handoff_policy is not None:
        handoff_policy.assert_teacher_free()
    if handoff_policy is not None and balance_curriculum is not None:
        raise ValueError("balance evaluation cannot use a policy handoff")
    if near_blow_loss_threshold is not None and near_blow_loss_threshold <= 0:
        raise ValueError("near-blow loss threshold must be positive")
    if (
        isinstance(no_trade_patience_episodes, bool)
        or not isinstance(no_trade_patience_episodes, int)
        or not 0 <= no_trade_patience_episodes <= episodes
    ):
        raise ValueError("validation no-trade patience is invalid")
    if (
        isinstance(greedy_diagnostic_interval_steps, bool)
        or greedy_diagnostic_interval_steps < 1
    ):
        raise ValueError("greedy diagnostic interval must be positive")
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
    flat_decision_count = greedy_entry_count = 0
    long_entry_count = short_entry_count = 0
    entry_advantage_probe_count = 0
    best_entry_advantage_sum = 0.0
    consecutive_zero_trade_episodes = 0
    validation_short_circuit_reason = None
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
        if balance_curriculum is None:
            observation, info = environment.reset()
        else:
            observation, info = environment.reset(options={
                "challenge_start_state": balance_curriculum.start_state(
                    episode_index
                ),
            })
        starting_realized_pnl = float(info.get("realized_pnl", 0.0))
        valid = tuple(info["valid_actions"])
        hidden = None
        if handoff_policy is not None:
            handoff_policy.reset()
        total = 0.0
        step_index = 0
        episode_flat_action_counts = {
            Action.WAIT: 0,
            Action.ENTER_LONG_1: 0,
            Action.ENTER_SHORT_1: 0,
        }
        episode_headroom = {
            name: {"flat_decisions": 0, "entries": 0}
            for name in (
                "le_0_25",
                "between_0_25_and_0_75",
                "ge_0_75",
                "unavailable",
            )
        }
        episode_best_entry_margins: list[float] = []
        episode_long_margins: list[float] = []
        episode_short_margins: list[float] = []
        episode_policy_state_decisions = {"recovery": 0, "normal": 0}
        while True:
            if step_index and step_index % recurrent_horizon == 0:
                hidden = None
                if handoff_policy is not None:
                    handoff_policy.reset()
            flat_actions = {
                Action.WAIT,
                Action.ENTER_LONG_1,
                Action.ENTER_SHORT_1,
            }
            is_flat_decision = flat_actions.issubset(valid)
            diagnostic_probe = (
                is_flat_decision
                and environment_steps % greedy_diagnostic_interval_steps == 0
            )
            if handoff_policy is None:
                action, hidden, action_values = agent.select_action(
                    observation,
                    hidden=hidden,
                    valid_actions=valid,
                    epsilon=0.0,
                    return_action_values=diagnostic_probe,
                )
            else:
                action, action_values, policy_state = handoff_policy.select_action(
                    observation,
                    valid_actions=valid,
                    realized_pnl=float(info["realized_pnl"]),
                    recovery_epsilon=0.0,
                    return_action_values=diagnostic_probe,
                )
                episode_policy_state_decisions[policy_state] += 1
            if is_flat_decision:
                flat_decision_count += 1
                episode_flat_action_counts[Action(action)] += 1
                greedy_entry_count += int(
                    action in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
                )
                long_entry_count += int(action == Action.ENTER_LONG_1)
                short_entry_count += int(action == Action.ENTER_SHORT_1)
                headroom_value = info.get("mll_headroom_fraction")
                if (
                    headroom_value is None
                    or isinstance(headroom_value, bool)
                    or not np.isfinite(float(headroom_value))
                ):
                    headroom_bucket = "unavailable"
                elif float(headroom_value) <= 0.25:
                    headroom_bucket = "le_0_25"
                elif float(headroom_value) >= 0.75:
                    headroom_bucket = "ge_0_75"
                else:
                    headroom_bucket = "between_0_25_and_0_75"
                episode_headroom[headroom_bucket]["flat_decisions"] += 1
                episode_headroom[headroom_bucket]["entries"] += int(
                    action in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}
                )
            if diagnostic_probe:
                assert action_values is not None
                values = np.asarray(action_values, dtype=np.float64)
                wait_value = values[int(Action.WAIT)]
                long_margin = values[int(Action.ENTER_LONG_1)] - wait_value
                short_margin = values[int(Action.ENTER_SHORT_1)] - wait_value
                best_margin = max(long_margin, short_margin)
                best_entry_advantage_sum += best_margin
                entry_advantage_probe_count += 1
                episode_best_entry_margins.append(float(best_margin))
                episode_long_margins.append(float(long_margin))
                episode_short_margins.append(float(short_margin))
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
        if episode_diagnostic_callback is not None:
            episode_flat_decisions = sum(episode_flat_action_counts.values())
            episode_entries = (
                episode_flat_action_counts[Action.ENTER_LONG_1]
                + episode_flat_action_counts[Action.ENTER_SHORT_1]
            )
            margin_rows = len(episode_best_entry_margins)
            closed_trade_receipts = getattr(
                environment, "closed_trade_receipts", None
            )
            closed_trade_economics = _validation_closed_trade_economics(
                (
                    ()
                    if closed_trade_receipts is None
                    else closed_trade_receipts()
                ),
                reported_trade_count=episode_trades,
                episode_outcome=outcome,
            )
            episode_diagnostic_callback({
                "schema": "propevolve_validation_episode_diagnostic_v1",
                "episode": episode_index + 1,
                "ticker": str(info.get("ticker", "?")),
                "outcome": outcome,
                "starting_realized_pnl": starting_realized_pnl,
                "reward": total,
                "terminal_pnl": terminal_pnl,
                "near_blow_timeout": near_blow_timeout,
                "trade_count": episode_trades,
                "win_rate": episode_win_rate,
                "average_win_r": episode_average_win_r,
                "expectancy_r": float(info.get("expectancy_r", 0.0)),
                "average_mfe_r": float(info.get("avg_mfe_r", 0.0)),
                "average_mae_r": float(info.get("avg_mae_r", 0.0)),
                "environment_steps": step_index,
                "flat_greedy_action_counts": {
                    action.name: episode_flat_action_counts[action]
                    for action in (
                        Action.WAIT,
                        Action.ENTER_LONG_1,
                        Action.ENTER_SHORT_1,
                    )
                },
                "flat_entry_rate": (
                    episode_entries / episode_flat_decisions
                    if episode_flat_decisions else 0.0
                ),
                "entry_counts": {
                    "ENTER_LONG_1": episode_flat_action_counts[
                        Action.ENTER_LONG_1
                    ],
                    "ENTER_SHORT_1": episode_flat_action_counts[
                        Action.ENTER_SHORT_1
                    ],
                },
                "policy_state_decisions": episode_policy_state_decisions,
                "headroom": episode_headroom,
                "closed_trade_economics": closed_trade_economics,
                "sampled_q_margins": {
                    "rows": margin_rows,
                    "best_entry_minus_wait_mean": (
                        float(np.mean(episode_best_entry_margins))
                        if margin_rows else 0.0
                    ),
                    "long_minus_wait_mean": (
                        float(np.mean(episode_long_margins))
                        if margin_rows else 0.0
                    ),
                    "short_minus_wait_mean": (
                        float(np.mean(episode_short_margins))
                        if margin_rows else 0.0
                    ),
                    "best_entry_minus_wait_min": (
                        float(np.min(episode_best_entry_margins))
                        if margin_rows else 0.0
                    ),
                    "best_entry_minus_wait_max": (
                        float(np.max(episode_best_entry_margins))
                        if margin_rows else 0.0
                    ),
                },
            })
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
            validation_short_circuit_reason = "zero_blow_gate"
            print(
                "[validation] SHORT_CIRCUIT reason=zero_blow_gate "
                f"episode={evaluated_episodes}/{episodes}",
                flush=True,
            )
            break
        consecutive_zero_trade_episodes = (
            consecutive_zero_trade_episodes + 1
            if episode_trades == 0 else 0
        )
        if (
            no_trade_patience_episodes
            and consecutive_zero_trade_episodes >= no_trade_patience_episodes
        ):
            validation_short_circuit_reason = (
                "universal_wait: "
                f"{no_trade_patience_episodes} consecutive zero-trade episodes"
            )
            print(
                "[validation] SHORT_CIRCUIT reason=universal_wait "
                f"episodes={no_trade_patience_episodes} "
                f"evaluated={evaluated_episodes}/{episodes}",
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
        flat_decision_count=flat_decision_count,
        greedy_entry_count=greedy_entry_count,
        long_entry_count=long_entry_count,
        short_entry_count=short_entry_count,
        best_entry_advantage_sum=best_entry_advantage_sum,
        entry_advantage_probe_count=entry_advantage_probe_count,
        short_circuited=validation_short_circuit_reason is not None,
        short_circuit_reason=validation_short_circuit_reason,
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
        f"mean_pnl={result.mean_terminal_pnl:+.2f} "
        f"greedy_entry={result.greedy_entry_rate:.1%} "
        f"entry_adv={result.mean_best_entry_advantage:+.4f}",
        flush=True,
    )
    return result


def evaluate_recovery_stress(
    agent: RecurrentC51Agent,
    environment: HistoricalChallengeEnv,
    *,
    episodes: int,
    recurrent_horizon: int,
    settings: RecoveryCurriculumSettings,
    episode_tickers: tuple[str, ...] | None = None,
    normal_policy: RecurrentC51Agent | None = None,
) -> RecoveryStressResult:
    """Run fixed-window recovery starts with normal challenge outcomes."""
    if episodes < 1 or recurrent_horizon < 1:
        raise ValueError("recovery stress budget must be positive")
    ticker_schedule = _balanced_ticker_schedule(
        episode_tickers,
        episodes=episodes,
        seed=settings.schedule_seed,
    )
    outcomes = {"pass": 0, "blow": 0, "timeout": 0}
    statuses = {"recovered": 0, "not_recovered": 0}
    retained = 0
    relapsed = 0
    recovered_then_blown = 0
    terminal_pnls: list[float] = []
    wait_decisions: list[int] = []
    entries_used = 0
    environment_steps = 0
    handoff_policy = (
        None
        if normal_policy is None
        else RecoveryHandoffPolicy(agent, normal_policy=normal_policy)
    )
    if handoff_policy is not None:
        handoff_policy.assert_teacher_free()
    for episode_index in range(episodes):
        options: dict[str, object] = {
            "challenge_start_state": settings.start_state,
        }
        if ticker_schedule is not None:
            options["ticker"] = ticker_schedule[episode_index]
        observation, info = environment.reset(options=options)
        valid = tuple(info["valid_actions"])
        hidden = None
        if handoff_policy is not None:
            handoff_policy.reset()
        step_index = 0
        while True:
            if step_index and step_index % recurrent_horizon == 0:
                hidden = None
                if handoff_policy is not None:
                    handoff_policy.reset()
            if handoff_policy is None:
                action, hidden, _ = agent.select_action(
                    observation,
                    hidden=hidden,
                    valid_actions=valid,
                    epsilon=0.0,
                )
            else:
                action, _, _ = handoff_policy.select_action(
                    observation,
                    valid_actions=valid,
                    realized_pnl=float(info["realized_pnl"]),
                    recovery_epsilon=0.0,
                )
            observation, _, terminated, _, info = environment.step(action)
            valid = tuple(info["valid_actions"])
            step_index += 1
            environment_steps += 1
            if terminated:
                break
        outcome = str(info.get("outcome"))
        status = str(info.get("recovery_status"))
        if outcome not in outcomes:
            raise ValueError(f"unknown recovery stress outcome: {outcome}")
        if status not in statuses:
            raise ValueError(f"unknown recovery status: {status}")
        outcomes[outcome] += 1
        statuses[status] += 1
        if status == "recovered":
            episode_relapsed = bool(info.get("recovery_relapsed", False))
            if episode_relapsed:
                relapsed += 1
            else:
                retained += 1
            if outcome == "blow":
                recovered_then_blown += 1
        entries_used += int(info.get("trade_count", 0))
        terminal_pnls.append(float(info["equity_pnl"]))
        wait_decisions.append(int(info.get("recovery_wait_decisions", 0)))
    result = RecoveryStressResult(
        episodes=episodes,
        recovered=statuses["recovered"],
        not_recovered=statuses["not_recovered"],
        retained=retained,
        relapsed=relapsed,
        recovered_then_blown=recovered_then_blown,
        passes=outcomes["pass"],
        timeouts=outcomes["timeout"],
        blows=outcomes["blow"],
        mean_terminal_pnl=float(np.mean(terminal_pnls)),
        mean_wait_decisions=float(np.mean(wait_decisions)),
        entries_used=entries_used,
        environment_steps=environment_steps,
    )
    print(
        "[recovery-stress] COMPLETE "
        f"episodes={episodes} recovered={result.recovered} "
        f"not_recovered={result.not_recovered} "
        f"retained={result.retained} relapsed={result.relapsed} "
        f"pass={result.passes} timeout={result.timeouts} blow={result.blows} "
        f"success_rate={result.recovery_success_rate:.1%} "
        f"mean_pnl={result.mean_terminal_pnl:+.2f}",
        flush=True,
    )
    return result
