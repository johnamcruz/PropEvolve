"""Fail-closed experiment configuration for historical training."""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path


DEFAULT_RUNTIME = {
    "mixed_precision": "off",
    "compile_model": False,
    "compile_backend": "inductor",
    "compile_mode": "default",
    "mps_prefer_metal": False,
    "mps_fast_math": False,
    "benchmark_max_relative_loss_drift": 0.05,
}
AGENT_RUNTIME_FIELDS = (
    "mixed_precision",
    "compile_model",
    "compile_backend",
    "compile_mode",
    "mps_prefer_metal",
    "mps_fast_math",
)
ENTRY_ACTION_LOSS_REDUCTIONS = {
    "population_weighted_mean_v1",
    "equal_present_class_mean_v1",
}

LEGACY_REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS = (
    "regime_selectivity.schema",
    "regime_selectivity.training_only",
    "regime_selectivity.target_source",
    "regime_selectivity.action_order",
    "regime_selectivity.formula",
    "regime_selectivity.expansion_center_receipt",
    "regime_selectivity.expansion_center_receipt_sha256",
    "regime_selectivity.expansion_long_center",
    "regime_selectivity.expansion_short_center",
    "regime_selectivity.probability_epsilon",
)
REGIME_SELECTIVITY_CORE_FROZEN_IDENTITY_PATHS = (
    *(
        path
        for path in LEGACY_REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS
        if path != "regime_selectivity.formula"
    ),
    "regime_selectivity.side_balance.schema",
    "regime_selectivity.side_balance.action_order",
    "regime_selectivity.headroom_pressure",
    "regime_selectivity.dominant_chop_pressure",
)
REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS = (
    "regime_selectivity.formula",
    "regime_selectivity.semantics",
    "regime_selectivity.persistent_chop_negative_emphasis",
)
REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS = (
    *REGIME_SELECTIVITY_CORE_FROZEN_IDENTITY_PATHS,
    *REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS,
)
RECOVERY_CURRICULUM_FROZEN_PATHS = (
    "recovery_curriculum.schedule_seed",
    "recovery_curriculum.stress_evaluation_episodes",
    "recovery_curriculum.recovery_success_pnl",
    "recovery_curriculum.start_state",
    "recovery_curriculum.action_value_supervision",
)
RECOVERY_CURRICULUM_REVISION_PATHS: tuple[str, ...] = ()
TRAINING_POLICY_HEALTH_FROZEN_PATH = "training.short_circuit.policy_health"


def _validate_training_policy_health(
    policy_health: object,
    *,
    training_episodes: int,
) -> None:
    """Validate the serialized fail-fast detector against its typed runtime spec."""
    required = {
        "schema",
        "minimum_completed_episodes",
        "probe_interval_episodes",
        "minimum_probe_recall",
        "entry_mass_fraction",
        "require_zero_positive_entry_soft_wait_veto",
        "economic_futility",
    }
    recall_actions = {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}
    economic_fields = {
        "minimum_completed_episodes",
        "maximum_near_blow_timeout_rate",
        "maximum_mean_terminal_pnl",
        "maximum_expectancy_r",
        "minimum_failed_conditions",
    }
    optional = {"require_positive_persistent_regime_association"}
    if (
        not isinstance(policy_health, dict)
        or not required <= set(policy_health)
        or set(policy_health) - required - optional
    ):
        raise ValueError("training short circuit policy health contract is invalid")
    recalls = policy_health["minimum_probe_recall"]
    entry_mass = policy_health["entry_mass_fraction"]
    economic = policy_health["economic_futility"]
    if (
        policy_health["schema"] != "propevolve_training_policy_health_v1"
        or not isinstance(recalls, dict)
        or set(recalls) != recall_actions
        or not isinstance(entry_mass, dict)
        or set(entry_mass) != {"minimum", "maximum"}
        or not isinstance(economic, dict)
        or set(economic) != economic_fields
        or not isinstance(
            policy_health["require_zero_positive_entry_soft_wait_veto"], bool
        )
        or not isinstance(
            policy_health.get(
                "require_positive_persistent_regime_association", False
            ),
            bool,
        )
    ):
        raise ValueError("training short circuit policy health contract is invalid")
    integer_values = (
        policy_health["minimum_completed_episodes"],
        policy_health["probe_interval_episodes"],
        economic["minimum_completed_episodes"],
        economic["minimum_failed_conditions"],
    )
    numeric_values = (
        *recalls.values(),
        entry_mass["minimum"],
        entry_mass["maximum"],
        economic["maximum_near_blow_timeout_rate"],
        economic["maximum_mean_terminal_pnl"],
        economic["maximum_expectancy_r"],
    )
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in integer_values
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        )
        or policy_health["minimum_completed_episodes"] > training_episodes
        or policy_health["probe_interval_episodes"] > training_episodes
        or economic["minimum_completed_episodes"] > training_episodes
    ):
        raise ValueError("training short circuit policy health contract is invalid")
    try:
        from .training_health import TrainingPolicyHealthSpec

        TrainingPolicyHealthSpec(
            minimum_completed_episodes=int(
                policy_health["minimum_completed_episodes"]
            ),
            probe_interval_episodes=int(policy_health["probe_interval_episodes"]),
            minimum_wait_recall=float(recalls["WAIT"]),
            minimum_long_recall=float(recalls["ENTER_LONG_1"]),
            minimum_short_recall=float(recalls["ENTER_SHORT_1"]),
            minimum_entry_mass_fraction=float(entry_mass["minimum"]),
            maximum_entry_mass_fraction=float(entry_mass["maximum"]),
            require_zero_positive_entry_soft_wait_veto=bool(
                policy_health["require_zero_positive_entry_soft_wait_veto"]
            ),
            require_positive_persistent_regime_association=bool(
                policy_health.get(
                    "require_positive_persistent_regime_association", False
                )
            ),
            economic_futility_minimum_completed_episodes=int(
                economic["minimum_completed_episodes"]
            ),
            economic_futility_maximum_near_blow_timeout_rate=float(
                economic["maximum_near_blow_timeout_rate"]
            ),
            economic_futility_maximum_mean_terminal_pnl=float(
                economic["maximum_mean_terminal_pnl"]
            ),
            economic_futility_maximum_expectancy_r=float(
                economic["maximum_expectancy_r"]
            ),
            economic_futility_minimum_failed_conditions=int(
                economic["minimum_failed_conditions"]
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training short circuit policy health contract is invalid"
        ) from error


def agent_runtime_settings(runtime: dict) -> dict:
    """Project the serialized runtime contract onto agent constructor fields."""
    return {field: runtime[field] for field in AGENT_RUNTIME_FIELDS}


def configure_runtime_environment(runtime: dict) -> dict[str, str]:
    """Set process-wide MPS flags before torch or MPS initialization."""
    environment = {
        "PYTORCH_MPS_PREFER_METAL": (
            "1" if bool(runtime["mps_prefer_metal"]) else "0"
        ),
        "PYTORCH_MPS_FAST_MATH": "1" if bool(runtime["mps_fast_math"]) else "0",
    }
    os.environ.update(environment)
    return environment


REQUIRED_RECIPE_FIELDS = {
    "cache": {
        "format",
        "encoder_identity_sha256",
        "context_length",
        "stride",
        "chunk_windows",
        "batch_series",
        "fast_group_attention",
        "device",
    },
    "challenge": {
        "profit_target",
        "max_loss",
        "episode_days",
        "bars_per_day",
        "max_position_size",
        "minimum_mll_headroom",
        "trailing_mll_lock",
        "terminal_pass_reward",
        "terminal_blow_reward",
        "terminal_timeout_reward",
        "terminal_pass_speed_reward_per_day",
        "reward_scale",
    },
    "agent": {
        "hidden_dim",
        "atoms",
        "value_min",
        "value_max",
        "gamma",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "target_sync_updates",
        "device",
    },
    "runtime": {
        "mixed_precision",
        "compile_model",
        "compile_backend",
        "compile_mode",
        "mps_prefer_metal",
        "mps_fast_math",
        "benchmark_max_relative_loss_drift",
    },
    "training": {
        "episodes",
        "validation_episodes",
        "replay_capacity_episodes",
        "replay_capacity_transitions",
        "sequence_length",
        "warmup_episodes",
        "updates_per_episode",
        "batch_sequences",
        "recurrent_horizon",
        "epsilon_start",
        "epsilon_end",
        "seed",
        "checkpoint_every_episodes",
        "prefetch_batches",
    },
}


def _require_recipe_fields(payload: dict) -> None:
    for section, required in REQUIRED_RECIPE_FIELDS.items():
        values = payload.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"{section} recipe section is required")
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"{section} recipe is missing fields {missing}")
    training = payload["training"]
    if (
        training.get("budget_mode", "environment_steps")
        == "environment_steps"
        and "minimum_environment_steps" not in training
    ):
        raise ValueError(
            "training recipe is missing fields ['minimum_environment_steps']"
        )


def _workspace_root(payload: dict, *, config_path: Path) -> Path:
    """Resolve the repository root declared by recipe content."""
    configured = payload.get("workspace_root")
    if not isinstance(configured, str) or not configured.strip():
        raise ValueError("workspace_root recipe field is required")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("workspace_root does not exist") from error


def _recipe_path_exists(payload: dict, path: str) -> bool:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _validate_entry_supervision(payload: dict, challenge: dict) -> None:
    """Validate the fully frozen, training-only post-launch action contract."""
    specification = payload.get("entry_supervision")
    if specification is None:
        return
    if isinstance(specification, dict):
        # Preserve the frozen unweighted v8 negative control while every new
        # recipe declares its balancing contract explicitly.
        specification.setdefault("action_class_balance", None)
    required = {
        "schema",
        "training_only",
        "decision_count",
        "fill_offsets",
        "execution",
        "risk_dollars",
        "launch",
        "continuation",
        "target_r",
        "stop_r",
        "horizon_bars",
        "collision",
        "loss_weight",
        "action_class_balance",
    }
    phase_fields = {"favorable_r", "adverse_r", "horizon_bars"}
    if (
        not isinstance(specification, dict)
        or set(specification) != required
        or specification.get("schema") != "post_launch_entry_v1"
        or specification.get("training_only") is not True
        or specification.get("execution") != "next_bar_open"
        or specification.get("collision") != "stop_first"
        or isinstance(specification.get("decision_count"), bool)
        or specification.get("decision_count") != 5
        or specification.get("fill_offsets") != [1, 2, 3, 4, 5]
    ):
        raise ValueError("entry supervision contract is invalid")
    balance = specification["action_class_balance"]
    if balance is not None and (
        not isinstance(balance, dict)
        or set(balance) != {"schema", "action_order"}
        or balance.get("schema") != "inverse_frequency_v1"
        or balance.get("action_order")
        != ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
    ):
        raise ValueError("entry supervision class balance contract is invalid")
    for phase in ("launch", "continuation"):
        values = specification.get(phase)
        if (
            not isinstance(values, dict)
            or set(values) != phase_fields
            or any(isinstance(values[field], bool) for field in phase_fields)
            or float(values["favorable_r"]) != 0.5
            or float(values["adverse_r"]) != 0.25
            or int(values["horizon_bars"]) != 3
            or float(values["horizon_bars"]) != 3.0
        ):
            raise ValueError("entry supervision phase contract is invalid")
    numeric = {
        field: specification[field]
        for field in (
            "risk_dollars",
            "target_r",
            "stop_r",
            "horizon_bars",
            "loss_weight",
        )
    }
    if (
        any(isinstance(value, bool) or not isinstance(value, (int, float))
            for value in numeric.values())
        or float(numeric["risk_dollars"]) != 300.0
        or float(numeric["risk_dollars"])
        != float(challenge.get("per_trade_risk_dollars", -1.0))
        or float(numeric["target_r"]) != 2.0
        or float(numeric["stop_r"]) != 1.0
        or int(numeric["horizon_bars"]) != 150
        or float(numeric["horizon_bars"]) != 150.0
        or float(numeric["loss_weight"]) <= 0.0
    ):
        raise ValueError("entry supervision economics are invalid")
    training = payload.get("training")
    if (
        not isinstance(training, dict)
        or float(training.get("teacher_loss_end_scale", -1.0)) != 0.0
        or float(training.get("teacher_autonomy_start_fraction", -1.0)) != 0.8
    ):
        raise ValueError(
            "entry supervision requires semantic-teacher autonomy for the "
            "final twenty percent"
        )
    entry_schedule_field = "entry_supervision_autonomy_start_fraction"
    entry_schedule_path = f"training.{entry_schedule_field}"
    explicitly_declared_entry_schedule = entry_schedule_field in training
    teacher_autonomy_start_fraction = float(
        training["teacher_autonomy_start_fraction"]
    )
    training.setdefault(
        entry_schedule_field,
        teacher_autonomy_start_fraction,
    )
    entry_autonomy_start_fraction = training[entry_schedule_field]
    if (
        isinstance(entry_autonomy_start_fraction, bool)
        or not isinstance(entry_autonomy_start_fraction, (int, float))
        or not teacher_autonomy_start_fraction
        <= float(entry_autonomy_start_fraction)
        <= 1.0
    ):
        raise ValueError(
            "entry supervision autonomy start fraction must be between "
            "the teacher boundary and one"
        )
    if explicitly_declared_entry_schedule and entry_schedule_path not in (
        payload.get("evolution", {}).get("frozen_paths", ())
    ):
        raise ValueError("entry supervision schedule must be frozen")


def _validate_recovery_curriculum(payload: dict, challenge: dict) -> None:
    """Authenticate the optional learned Stage-2 recovery contract."""
    curriculum = payload.get("recovery_curriculum")
    if curriculum is None:
        return
    required = {
        "schedule_seed",
        "stress_evaluation_episodes",
        "recovery_success_pnl",
        "start_state",
        "action_value_supervision",
    }
    start_fields = {
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
    supervision_fields = {
        "loss_weight",
        "temperature",
        "store_capacity",
        "target_every_episodes",
        "start_pnls",
        "retain_nonnegative_entry_policy",
    }
    optional_supervision_fields = {"success_replay", "healthy_pass_replay"}
    if not isinstance(curriculum, dict) or set(curriculum) != required:
        raise ValueError("recovery curriculum contract is invalid")
    start = curriculum["start_state"]
    supervision = curriculum["action_value_supervision"]
    if (
        not isinstance(start, dict)
        or set(start) != start_fields
        or not isinstance(supervision, dict)
        or not supervision_fields.issubset(supervision)
        or set(supervision) - supervision_fields - optional_supervision_fields
    ):
        raise ValueError("recovery curriculum contract is invalid")
    success_replay = supervision.get("success_replay")
    if success_replay is not None and (
        not isinstance(success_replay, dict)
        or set(success_replay)
        != {"path", "sha256", "update_period", "max_examples"}
        or not isinstance(success_replay["path"], str)
        or not success_replay["path"]
        or not isinstance(success_replay["sha256"], str)
        or len(success_replay["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in success_replay["sha256"]
        )
        or isinstance(success_replay["update_period"], bool)
        or not isinstance(success_replay["update_period"], int)
        or success_replay["update_period"] < 1
        or isinstance(success_replay["max_examples"], bool)
        or not isinstance(success_replay["max_examples"], int)
        or success_replay["max_examples"] < 1
    ):
        raise ValueError("recovery pass replay contract is invalid")
    healthy_pass_replay = supervision.get("healthy_pass_replay")
    if healthy_pass_replay is not None and (
        not isinstance(healthy_pass_replay, dict)
        or set(healthy_pass_replay)
        != {"path", "sha256", "update_period", "max_examples"}
        or not isinstance(healthy_pass_replay["path"], str)
        or not healthy_pass_replay["path"]
        or not isinstance(healthy_pass_replay["sha256"], str)
        or len(healthy_pass_replay["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in healthy_pass_replay["sha256"]
        )
        or isinstance(healthy_pass_replay["update_period"], bool)
        or not isinstance(healthy_pass_replay["update_period"], int)
        or healthy_pass_replay["update_period"] < 1
        or isinstance(healthy_pass_replay["max_examples"], bool)
        or not isinstance(healthy_pass_replay["max_examples"], int)
        or healthy_pass_replay["max_examples"] < 1
    ):
        raise ValueError("healthy pass replay contract is invalid")
    integer_values = (
        curriculum["schedule_seed"],
        curriculum["stress_evaluation_episodes"],
        start["position_side"],
        start["position_size"],
        start["trading_days_elapsed"],
        supervision["store_capacity"],
        supervision["target_every_episodes"],
    )
    numeric_values = (
        curriculum["recovery_success_pnl"],
        start["realized_pnl"],
        start["equity_pnl"],
        start["peak_equity_pnl"],
        start["mll_floor_pnl"],
        start["session_pnl"],
        supervision["loss_weight"],
        supervision["temperature"],
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        )
        or not isinstance(start["passmark_locked"], bool)
        or not isinstance(supervision["start_pnls"], list)
        or not isinstance(supervision["retain_nonnegative_entry_policy"], bool)
    ):
        raise ValueError("recovery curriculum scalar types are invalid")
    stress_episodes = int(curriculum["stress_evaluation_episodes"])
    if not 1 <= stress_episodes <= 200:
        raise ValueError("recovery curriculum budget is invalid")
    if (
        not 0.0 < float(supervision["loss_weight"]) <= 1.0
        or float(supervision["temperature"]) <= 0.0
        or int(supervision["store_capacity"]) < 1
        or int(supervision["target_every_episodes"]) < 1
    ):
        raise ValueError("recovery action-value supervision is invalid")
    start_pnls = supervision["start_pnls"]
    if (
        not start_pnls
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not float(start["mll_floor_pnl"]) < float(value) < 0.0
            for value in start_pnls
        )
        or len({float(value) for value in start_pnls}) != len(start_pnls)
        or float(start["realized_pnl"])
        not in {float(value) for value in start_pnls}
    ):
        raise ValueError("recovery supervision start PnLs are invalid")
    exact_start = {
        "realized_pnl": -2_000.0,
        "equity_pnl": -2_000.0,
        "peak_equity_pnl": 0.0,
        "mll_floor_pnl": -3_000.0,
        "passmark_locked": False,
        "position_side": 0,
        "position_size": 0,
        "session_pnl": -2_000.0,
        "trading_days_elapsed": 1,
    }
    if any(
        isinstance(expected, float)
        and not math.isclose(float(start[field]), expected)
        or not isinstance(expected, float)
        and start[field] != expected
        for field, expected in exact_start.items()
    ) or not math.isclose(float(curriculum["recovery_success_pnl"]), 0.0):
        raise ValueError("Stage-2 recovery start contract drifted")
    if (
        not math.isclose(float(challenge["max_loss"]), 3_000.0)
        or not math.isclose(float(challenge["minimum_mll_headroom"]), 500.0)
        or not math.isclose(
            float(challenge.get("per_trade_risk_dollars", math.nan)), 300.0
        )
    ):
        raise ValueError("Stage-2 recovery challenge economics drifted")


def _validate_regime_selectivity(payload: dict, *, root: Path) -> None:
    """Authenticate the optional Stage 2A training-only selectivity contract."""
    specification = payload.get("regime_selectivity")
    if specification is None:
        return
    from .balance_aware_regime_selectivity import (
        ACTION_ORDER,
        ALL_DOMINANT_CHOP_MARGIN_FORMULA,
        ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        CHOP_MARGIN_EXPANSION_REGIME_CONFLUENCE_FORMULA,
        EXPANSION_REGIME_CONFLUENCE_FORMULA,
        EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        FORMULA,
        PERSISTENT_CHOP_ASSOCIATION_FORMULA,
        PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        PAIRED_A_PLUS_CONTRASTIVE_FORMULA,
        PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA,
        PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        SCHEMA,
        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA,
        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        STATIC_STATE_SEMANTICS,
        TARGET_SOURCE,
    )
    from .teachers.expansion import (
        CHANNELS as EXPANSION_CHANNELS,
        ENTRY_CENTER_FORMULA,
        ENTRY_CENTER_RECEIPT_SCHEMA,
    )

    legacy_required = {
        "schema",
        "training_only",
        "target_source",
        "action_order",
        "formula",
        "loss_weight",
        "expansion_center_receipt",
        "expansion_center_receipt_sha256",
        "expansion_long_center",
        "expansion_short_center",
        "probability_epsilon",
        "headroom_pressure",
        "dominant_chop_pressure",
        "q_temperature",
    }
    required = {
        *legacy_required,
        "semantics",
        "persistent_chop_negative_emphasis",
        "side_balance",
    }
    margin_fields = {"chop_wait_margin", "failed_confluence_margin"}
    paired_fields = {"paired_a_plus_margin"}
    valid_required_sets = (
        required,
        required | margin_fields,
        required | margin_fields | paired_fields,
    )
    legacy_contract = (
        isinstance(specification, dict)
        and set(specification) == legacy_required
    )
    if legacy_contract:
        specification["semantics"] = STATIC_STATE_SEMANTICS
        specification["persistent_chop_negative_emphasis"] = 0.0
        specification["side_balance"] = None
    semantics = (
        specification.get("semantics")
        if isinstance(specification, dict)
        else None
    )
    side_balance = (
        specification.get("side_balance")
        if isinstance(specification, dict)
        else None
    )
    paired_margin_present = (
        isinstance(specification, dict)
        and "paired_a_plus_margin" in specification
    )
    expected_formulas = {
        STATIC_STATE_SEMANTICS: FORMULA,
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS: (
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA
        ),
        PERSISTENT_CHOP_ASSOCIATION_SEMANTICS: (
            PERSISTENT_CHOP_ASSOCIATION_FORMULA
        ),
        EXPANSION_REGIME_CONFLUENCE_SEMANTICS: (
            EXPANSION_REGIME_CONFLUENCE_FORMULA
        ),
        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS: (
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA
        ),
        ALL_DOMINANT_CHOP_MARGIN_SEMANTICS: (
            ALL_DOMINANT_CHOP_MARGIN_FORMULA
        ),
        PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS: (
            PAIRED_A_PLUS_CONTRASTIVE_FORMULA
        ),
        PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS: (
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA
        ),
    }
    expected_formula = expected_formulas.get(semantics)
    if (
        isinstance(specification, dict)
        and margin_fields <= set(specification)
        and semantics == SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS
    ):
        expected_formula = CHOP_MARGIN_EXPANSION_REGIME_CONFLUENCE_FORMULA
    short_circuit = payload.get("training", {}).get("short_circuit") or {}
    policy_health = short_circuit.get("policy_health", {})
    require_positive_association = (
        policy_health.get(
            "require_positive_persistent_regime_association", False
        )
        if isinstance(policy_health, dict)
        else False
    )
    association_declared = (
        semantics in {
            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        }
        or (
            isinstance(specification, dict)
            and specification.get("formula") in {
                PERSISTENT_CHOP_ASSOCIATION_FORMULA,
                EXPANSION_REGIME_CONFLUENCE_FORMULA,
                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA,
                ALL_DOMINANT_CHOP_MARGIN_FORMULA,
                PAIRED_A_PLUS_CONTRASTIVE_FORMULA,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA,
            }
        )
        or require_positive_association is True
    )
    if association_declared and (
        semantics not in {
            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        }
        or not isinstance(specification, dict)
        or set(specification) not in valid_required_sets
        or specification.get("formula") != expected_formula
    ):
        raise ValueError("persistent-chop association contract is invalid")
    teachers = tuple(payload.get("teachers") or ())
    numeric = tuple(
        specification.get(field)
        for field in (
            "loss_weight",
            "expansion_long_center",
            "expansion_short_center",
            "probability_epsilon",
            "headroom_pressure",
            "dominant_chop_pressure",
            "q_temperature",
            "persistent_chop_negative_emphasis",
            "chop_wait_margin",
            "failed_confluence_margin",
            "paired_a_plus_margin",
        )
    ) if (
        isinstance(specification, dict)
        and margin_fields | paired_fields <= set(specification)
    ) else (
        tuple(
            specification.get(field)
            for field in (
                "loss_weight",
                "expansion_long_center",
                "expansion_short_center",
                "probability_epsilon",
                "headroom_pressure",
                "dominant_chop_pressure",
                "q_temperature",
                "persistent_chop_negative_emphasis",
                "chop_wait_margin",
                "failed_confluence_margin",
            )
        ) if isinstance(specification, dict) and margin_fields <= set(specification) else (
            tuple(
                specification.get(field)
                for field in (
                    "loss_weight",
                    "expansion_long_center",
                    "expansion_short_center",
                    "probability_epsilon",
                    "headroom_pressure",
                    "dominant_chop_pressure",
                    "q_temperature",
                    "persistent_chop_negative_emphasis",
                )
            ) if isinstance(specification, dict) else ()
        )
    )
    if (
        not isinstance(specification, dict)
        or set(specification) not in valid_required_sets
        or specification.get("schema") != SCHEMA
        or specification.get("training_only") is not True
        or specification.get("target_source") != TARGET_SOURCE
        or tuple(specification.get("action_order", ())) != ACTION_ORDER
        or semantics not in {
            STATIC_STATE_SEMANTICS,
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
            PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
            EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        }
        or specification.get("formula") != expected_formula
        or (
            semantics
            in {
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }
            and set(specification) != required | margin_fields | paired_fields
        )
        or (
            semantics
            not in {
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }
            and paired_margin_present
        )
        or (
            side_balance is not None
            and (
                not isinstance(side_balance, dict)
                or set(side_balance) != {"schema", "action_order"}
                or side_balance.get("schema")
                not in {
                    "equal_long_short_v1",
                    "paired_recurrent_long_short_v1",
                }
                or side_balance.get("action_order")
                != ["ENTER_LONG_1", "ENTER_SHORT_1"]
            )
        )
        or (not legacy_contract and side_balance is None)
        or payload.get("entry_supervision") is None
        or len(teachers) < 2
        or teachers[0].get("kind") != "expansion"
        or not any(teacher.get("kind") == "regime" for teacher in teachers)
        or tuple(teachers[0].get("channels", ())) != EXPANSION_CHANNELS
        or len(numeric) not in {8, 10, 11}
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in numeric
        )
        or any(not math.isfinite(float(value)) for value in numeric)
        or float(specification["loss_weight"]) <= 0.0
        or not 0.0 < float(specification["probability_epsilon"]) < 0.5
        or any(
            not float(specification["probability_epsilon"])
            < float(specification[field])
            < 1.0 - float(specification["probability_epsilon"])
            for field in ("expansion_long_center", "expansion_short_center")
        )
        or float(specification["headroom_pressure"]) < 0.0
        or float(specification["dominant_chop_pressure"]) < 0.0
        or float(specification.get("chop_wait_margin", 0.0)) < 0.0
        or float(specification.get("failed_confluence_margin", 0.0)) < 0.0
        or float(specification.get("paired_a_plus_margin", 0.0)) < 0.0
        or (
            semantics
            in {
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }
            and float(specification["paired_a_plus_margin"]) <= 0.0
        )
        or float(specification["q_temperature"]) <= 0.0
        or (
            semantics == STATIC_STATE_SEMANTICS
            and float(specification["persistent_chop_negative_emphasis"]) != 0.0
        )
        or (
            semantics in {
                PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                PERSISTENT_CHOP_ASSOCIATION_SEMANTICS,
                EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
                ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
                PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
                PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
            }
            and float(specification["persistent_chop_negative_emphasis"]) <= 0.0
        )
    ):
        raise ValueError("balance-aware Regime selectivity contract is invalid")
    if (
        semantics == PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
    ) != (
        isinstance(side_balance, dict)
        and side_balance.get("schema") == "paired_recurrent_long_short_v1"
    ):
        raise ValueError(
            "paired recurrent A+ semantics require paired recurrent replay"
        )
    training = payload.get("training", {})
    paired_entry_slots = round(
        int(training.get("batch_sequences", 0))
        * float(training.get("entry_opportunity_sequence_fraction", 0.0))
    )
    if (
        semantics == PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
        and (paired_entry_slots < 2 or paired_entry_slots % 2)
    ):
        raise ValueError(
            "paired recurrent A+ replay requires an even positive pair stratum"
        )
    if (
        (
            float(training.get("regime_wait_sequence_fraction", 0.0)) > 0.0
            or int(training.get("regime_wait_sequence_update_period", 0)) > 0
        )
        and semantics
        not in {
            SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
            PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
            PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        }
    ):
        raise ValueError(
            "Regime WAIT replay requires side-conditioned confluence semantics"
        )
    receipt_path = (
        root / str(specification["expansion_center_receipt"])
    ).resolve(strict=True)
    if not receipt_path.is_relative_to(root):
        raise ValueError("Regime selectivity receipt must be repository-local")
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    if digest != specification["expansion_center_receipt_sha256"]:
        raise ValueError("Regime selectivity receipt identity drifted")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != ENTRY_CENTER_RECEIPT_SCHEMA
        or receipt.get("formula") != ENTRY_CENTER_FORMULA
        or tuple(receipt.get("tickers", ())) != tuple(payload.get("tickers", ()))
        or tuple(receipt.get("channels", ())) != EXPANSION_CHANNELS
        or receipt.get("training_end_exclusive") != payload["temporal"]["train_end"]
        or float(receipt.get("long_center", -1.0))
        != float(specification["expansion_long_center"])
        or float(receipt.get("short_center", -1.0))
        != float(specification["expansion_short_center"])
    ):
        raise ValueError("Regime selectivity receipt contract drifted")


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get("schema") != "propevolve_historical_training_v1":
        raise ValueError("unsupported PropEvolve experiment schema")
    root = _workspace_root(payload, config_path=path)
    # Schema-v1 recipes created before runtime tuning preserve their original
    # eager FP32 behavior. Current repository recipes serialize these fields.
    payload.setdefault("runtime", dict(DEFAULT_RUNTIME))
    if isinstance(payload.get("training"), dict):
        payload["training"].setdefault("prefetch_batches", 0)
    if isinstance(payload.get("agent"), dict):
        # Schema-v1 recipes predate gradual target-network updates. Preserve
        # their exact hard-sync behavior while allowing new recipes to declare
        # the safer update contract explicitly.
        payload["agent"].setdefault("target_update_mode", "hard")
        payload["agent"].setdefault("target_soft_tau", 1.0)
    _require_recipe_fields(payload)
    try:
        from .observation import TradeManagementObservationSpec

        TradeManagementObservationSpec.from_config(payload.get("observation"))
    except (TypeError, ValueError) as error:
        raise ValueError("observation contract is invalid") from error
    if "timeframe_minutes" not in payload:
        raise ValueError("timeframe_minutes recipe field is required")
    tickers = tuple(str(value) for value in payload.get("tickers", ()))
    deployment = tuple(str(value) for value in payload.get("deployment_tickers", ()))
    training_only = tuple(str(value) for value in payload.get("training_only_tickers", ()))
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be a nonempty unique population")
    if not deployment or not set(deployment) <= set(tickers):
        raise ValueError("deployment tickers must be a nonempty training subset")
    if set(training_only) & set(deployment):
        raise ValueError("training-only tickers cannot be deployed")
    if not set(training_only) <= set(tickers):
        raise ValueError("training-only tickers must belong to the training population")
    for field in ("point_values", "round_trip_fees"):
        values = payload.get(field) or {}
        if set(values) != set(tickers) or any(float(value) <= 0 for value in values.values()):
            raise ValueError(f"{field} must positively cover the exact ticker population")
    challenge = payload["challenge"]
    # Schema-v1 receipts predate configurable reward shaping. Normalize their
    # behavior explicitly while new recipes serialize every setting in JSON.
    challenge.setdefault("mll_proximity_penalty_coefficient", 0.0)
    challenge.setdefault("lead_giveback_penalty_coefficient", 0.0)
    challenge.setdefault("large_win_threshold_r", 2.0)
    challenge.setdefault("large_win_bonus_coefficient", 0.0)
    challenge.setdefault("ratchet_lock_floor_r", 0.0)
    if challenge["max_position_size"] != 1:
        raise ValueError("the initial PropEvolve recipe supports one contract")
    if not isinstance(challenge["trailing_mll_lock"], bool):
        raise ValueError("challenge trailing_mll_lock must be boolean")
    if any(
        float(challenge[field]) < 0
        for field in (
            "mll_proximity_penalty_coefficient",
            "lead_giveback_penalty_coefficient",
            "large_win_threshold_r",
            "large_win_bonus_coefficient",
            "ratchet_lock_floor_r",
        )
    ):
        raise ValueError("challenge reward-shaping settings must be nonnegative")
    ratchet_fields = (
        "per_trade_risk_dollars",
        "ratchet_activation_r",
        "ratchet_giveback_r",
    )
    present_ratchet_fields = tuple(
        field for field in ratchet_fields if field in challenge
    )
    if present_ratchet_fields and len(present_ratchet_fields) != len(ratchet_fields):
        raise ValueError("trade risk and ratchet fields must be declared together")
    if present_ratchet_fields and (
        float(challenge["per_trade_risk_dollars"]) <= 0
        or float(challenge["ratchet_giveback_r"]) <= 0
        or float(challenge["ratchet_activation_r"])
        <= float(challenge["ratchet_giveback_r"])
    ):
        raise ValueError("trade risk and ratchet fields are invalid")
    _validate_entry_supervision(payload, challenge)
    _validate_recovery_curriculum(payload, challenge)
    cache = payload["cache"]
    if cache["format"] not in {"native", "ffm_frozen_representation_v2"}:
        raise ValueError("cache format must be native or ffm_frozen_representation_v2")
    if not str(cache["encoder_identity_sha256"]).strip():
        raise ValueError("cache encoder identity must be declared")
    agent = payload["agent"]
    explicitly_declared_entry_reduction = (
        "entry_action_loss_reduction" in agent
    )
    agent.setdefault("n_step_return", 1)
    agent.setdefault("recurrent_burn_in", 0)
    agent.setdefault("policy_retention_loss_weight", 0.0)
    agent.setdefault(
        "entry_action_loss_reduction", "population_weighted_mean_v1"
    )
    agent.setdefault("entry_action_margin", 0.0)
    if agent["entry_action_loss_reduction"] not in ENTRY_ACTION_LOSS_REDUCTIONS:
        raise ValueError("entry action loss reduction is invalid")
    if (
        isinstance(agent["entry_action_margin"], bool)
        or not isinstance(agent["entry_action_margin"], (int, float))
        or not math.isfinite(float(agent["entry_action_margin"]))
        or float(agent["entry_action_margin"]) < 0.0
    ):
        raise ValueError("entry action margin is invalid")
    if (
        "agent.entry_action_loss_reduction"
        in payload.get("evolution", {}).get("frozen_paths", ())
        and not explicitly_declared_entry_reduction
    ):
        raise ValueError("entry action loss reduction is missing")
    if agent["device"] not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("agent device must be auto, cuda, mps, or cpu")
    if (
        agent["target_update_mode"] not in {"hard", "soft"}
        or isinstance(agent["target_soft_tau"], bool)
        or not isinstance(agent["target_soft_tau"], (int, float))
        or not 0 < float(agent["target_soft_tau"]) <= 1
        or (
            agent["target_update_mode"] == "hard"
            and float(agent["target_soft_tau"]) != 1.0
        )
    ):
        raise ValueError("agent target update contract is invalid")
    if (
        isinstance(agent["policy_retention_loss_weight"], bool)
        or not isinstance(agent["policy_retention_loss_weight"], (int, float))
        or float(agent["policy_retention_loss_weight"]) < 0
    ):
        raise ValueError("agent policy retention loss weight must be nonnegative")
    runtime = payload["runtime"]
    if runtime["mixed_precision"] not in {"off", "fp16"}:
        raise ValueError("runtime mixed precision must be off or fp16")
    if not isinstance(runtime["compile_model"], bool):
        raise ValueError("runtime compile_model must be boolean")
    if not isinstance(runtime["mps_prefer_metal"], bool):
        raise ValueError("runtime mps_prefer_metal must be boolean")
    if not isinstance(runtime["mps_fast_math"], bool):
        raise ValueError("runtime mps_fast_math must be boolean")
    if not str(runtime["compile_backend"]).strip():
        raise ValueError("runtime compile backend is required")
    if not str(runtime["compile_mode"]).strip():
        raise ValueError("runtime compile mode is required")
    if (
        isinstance(runtime["benchmark_max_relative_loss_drift"], bool)
        or not isinstance(
            runtime["benchmark_max_relative_loss_drift"], (int, float)
        )
        or not 0 <= float(runtime["benchmark_max_relative_loss_drift"]) <= 1
    ):
        raise ValueError("runtime benchmark loss drift must be between zero and one")
    training = payload["training"]
    training.setdefault("short_circuit", None)
    training.setdefault("terminal_sequence_fraction", 0.0)
    training.setdefault("safety_sequence_fraction", 0.0)
    training.setdefault("entry_opportunity_sequence_fraction", 0.0)
    training.setdefault("regime_wait_sequence_fraction", 0.0)
    training.setdefault("regime_wait_sequence_update_period", 0)
    training.setdefault("management_epsilon_start", training["epsilon_start"])
    training.setdefault("management_epsilon_end", training["epsilon_end"])
    training.setdefault("validation_no_trade_patience_episodes", 0)
    training.setdefault("greedy_diagnostic_interval_steps", 256)
    training_budget_mode = training.get("budget_mode", "environment_steps")
    if training_budget_mode not in {"environment_steps", "episodes"}:
        raise ValueError("training budget mode is invalid")
    if (
        isinstance(training["episodes"], bool)
        or not isinstance(training["episodes"], int)
        or training["episodes"] < 1
        or isinstance(training["validation_episodes"], bool)
        or not isinstance(training["validation_episodes"], int)
        or training["validation_episodes"] < 1
    ):
        raise ValueError("training budget contract is invalid")
    if training_budget_mode == "environment_steps":
        minimum_environment_steps = training.get("minimum_environment_steps")
        if (
            isinstance(minimum_environment_steps, bool)
            or not isinstance(minimum_environment_steps, int)
            or minimum_environment_steps < 1
        ):
            raise ValueError("training environment-step budget is invalid")
    elif "minimum_environment_steps" in training:
        raise ValueError(
            "episode training budget cannot declare minimum environment steps"
        )
    if (
        isinstance(training["prefetch_batches"], bool)
        or not isinstance(training["prefetch_batches"], int)
        or not 0 <= training["prefetch_batches"] <= 2
    ):
        raise ValueError("training replay prefetch must be between zero and two")
    if (
        isinstance(training["replay_capacity_transitions"], bool)
        or int(training["replay_capacity_transitions"])
        < int(training["sequence_length"])
    ):
        raise ValueError("training replay transition capacity is invalid")
    if (
        isinstance(agent["n_step_return"], bool)
        or not isinstance(agent["n_step_return"], int)
        or not 1 <= int(agent["n_step_return"]) <= int(training["sequence_length"])
    ):
        raise ValueError("agent n-step return must fit the replay sequence")
    if (
        isinstance(agent["recurrent_burn_in"], bool)
        or not isinstance(agent["recurrent_burn_in"], int)
        or int(agent["recurrent_burn_in"]) < 0
        or int(agent["recurrent_burn_in"]) + int(agent["n_step_return"])
        > int(training["sequence_length"])
    ):
        raise ValueError(
            "agent recurrent burn-in plus n-step return must fit the replay sequence"
        )
    if not 0 <= float(training["epsilon_end"]) <= float(training["epsilon_start"]) <= 1:
        raise ValueError("training epsilon schedule is invalid")
    if not (
        0
        <= float(training["management_epsilon_end"])
        <= float(training["management_epsilon_start"])
        <= 1
    ):
        raise ValueError("training management epsilon schedule is invalid")
    if (
        not 0 <= float(training["terminal_sequence_fraction"]) <= 1
        or not 0 <= float(training["safety_sequence_fraction"]) <= 1
        or not 0 <= float(training["entry_opportunity_sequence_fraction"]) <= 1
        or not 0 <= float(training["regime_wait_sequence_fraction"]) <= 1
        or float(training["terminal_sequence_fraction"])
        + float(training["safety_sequence_fraction"]) > 1
        or float(training["terminal_sequence_fraction"])
        + float(training["safety_sequence_fraction"])
        + float(training["entry_opportunity_sequence_fraction"])
        + float(training["regime_wait_sequence_fraction"]) > 1
    ):
        raise ValueError("training replay sequence fractions are invalid")
    regime_wait_update_period = training["regime_wait_sequence_update_period"]
    if (
        isinstance(regime_wait_update_period, bool)
        or not isinstance(regime_wait_update_period, int)
        or regime_wait_update_period < 0
        or (
            regime_wait_update_period > 0
            and float(training["regime_wait_sequence_fraction"]) > 0.0
        )
        or (
            regime_wait_update_period > 0
            and round(
                int(training["batch_sequences"])
                * float(training["terminal_sequence_fraction"])
            ) < 1
        )
    ):
        raise ValueError("training Regime WAIT update schedule is invalid")
    if (
        (
            float(training["regime_wait_sequence_fraction"]) > 0.0
            or regime_wait_update_period > 0
        )
        and payload.get("regime_selectivity") is None
    ):
        raise ValueError(
            "Regime WAIT replay requires authenticated selectivity"
        )
    if (
        isinstance(training["checkpoint_every_episodes"], bool)
        or int(training["checkpoint_every_episodes"]) < 1
    ):
        raise ValueError("training checkpoint interval must be positive")
    if (
        isinstance(training["validation_no_trade_patience_episodes"], bool)
        or not isinstance(
            training["validation_no_trade_patience_episodes"], int
        )
        or not 0
        <= training["validation_no_trade_patience_episodes"]
        <= int(training["validation_episodes"])
    ):
        raise ValueError("validation no-trade patience is invalid")
    if (
        isinstance(training["greedy_diagnostic_interval_steps"], bool)
        or not isinstance(training["greedy_diagnostic_interval_steps"], int)
        or training["greedy_diagnostic_interval_steps"] < 1
    ):
        raise ValueError("greedy diagnostic interval is invalid")
    short_circuit = training.get("short_circuit")
    if short_circuit is not None:
        boundary_field = (
            "minimum_completed_episodes"
            if training_budget_mode == "episodes"
            else "minimum_environment_steps"
        )
        required_short_circuit = {
            boundary_field,
            "minimum_passes",
            "maximum_blow_rate",
        }
        if (
            not isinstance(short_circuit, dict)
            or not required_short_circuit <= set(short_circuit)
            or set(short_circuit) - required_short_circuit
            - {"collapse", "policy_health"}
            or isinstance(short_circuit[boundary_field], bool)
            or not isinstance(short_circuit[boundary_field], int)
            or not 1
            <= short_circuit[boundary_field]
            <= int(
                training[
                    "episodes"
                    if training_budget_mode == "episodes"
                    else "minimum_environment_steps"
                ]
            )
            or isinstance(short_circuit["minimum_passes"], bool)
            or not isinstance(short_circuit["minimum_passes"], int)
            or short_circuit["minimum_passes"] < 0
            or isinstance(short_circuit["maximum_blow_rate"], bool)
            or not isinstance(short_circuit["maximum_blow_rate"], (int, float))
            or not 0 <= float(short_circuit["maximum_blow_rate"]) <= 1
        ):
            raise ValueError("training short circuit contract is invalid")
        if "policy_health" in short_circuit:
            if training_budget_mode != "episodes":
                raise ValueError(
                    "training short circuit policy health requires episode budget mode"
                )
            _validate_training_policy_health(
                short_circuit["policy_health"],
                training_episodes=int(training["episodes"]),
            )
        collapse = short_circuit.get("collapse")
        if collapse is not None and (
            not isinstance(collapse, dict)
            or set(collapse) != {
                "window_episodes",
                "minimum_prior_passes",
                "maximum_recent_passes",
                "maximum_average_hold_bars",
                "minimum_voluntary_close_rate",
            }
            or isinstance(collapse["window_episodes"], bool)
            or not isinstance(collapse["window_episodes"], int)
            or collapse["window_episodes"] < 2
            or isinstance(collapse["minimum_prior_passes"], bool)
            or not isinstance(collapse["minimum_prior_passes"], int)
            or collapse["minimum_prior_passes"] < 1
            or isinstance(collapse["maximum_recent_passes"], bool)
            or not isinstance(collapse["maximum_recent_passes"], int)
            or not 0
            <= collapse["maximum_recent_passes"]
            < collapse["window_episodes"]
            or isinstance(collapse["maximum_average_hold_bars"], bool)
            or not isinstance(
                collapse["maximum_average_hold_bars"], (int, float)
            )
            or not math.isfinite(
                float(collapse["maximum_average_hold_bars"])
            )
            or collapse["maximum_average_hold_bars"] <= 0
            or isinstance(collapse["minimum_voluntary_close_rate"], bool)
            or not isinstance(
                collapse["minimum_voluntary_close_rate"], (int, float)
            )
            or not 0 <= collapse["minimum_voluntary_close_rate"] <= 1
        ):
            raise ValueError("training collapse detector contract is invalid")
    teacher = payload.get("teacher")
    teachers = payload.get("teachers")
    if teacher is not None and teachers is not None:
        raise ValueError("declare teacher or teachers, not both")
    if teacher is not None:
        teachers = [teacher]
    if teachers is not None:
        from .teachers.expansion import CHANNELS as EXPANSION_CHANNELS
        from .teachers.regime import CHANNELS as REGIME_CHANNELS
        from .teachers.trend import CHANNELS as TREND_CHANNELS

        if (
            not isinstance(teachers, list)
            or not teachers
            or len({item.get("kind") for item in teachers if isinstance(item, dict)})
            != len(teachers)
        ):
            raise ValueError("training-only teacher collection is invalid")
        expected_channels = {
            "expansion": EXPANSION_CHANNELS,
            "regime": REGIME_CHANNELS,
            "trend": TREND_CHANNELS,
        }
        for index, item in enumerate(teachers):
            if not isinstance(item, dict):
                raise ValueError("training-only teacher contract is invalid")
            item.setdefault("entry_search_loss_weight", 0.0)
            kind = item.get("kind")
            if kind == "expansion":
                item.setdefault("entry_search_objective", "raw_probability")
                item.setdefault("entry_search_long_center", 0.5)
                item.setdefault("entry_search_short_center", 0.5)
                item.setdefault("entry_search_probability_epsilon", 1e-6)
                item.setdefault("entry_search_teacher_temperature", 1.0)
                item.setdefault("entry_search_q_temperature", 1.0)
                item.setdefault("entry_search_center_receipt", "")
                item.setdefault("entry_search_center_receipt_sha256", "")
            required_fields = {
                "kind", "cache_root", "channels", "loss_weight",
                "entry_search_loss_weight",
            }
            if kind == "expansion":
                required_fields.update({
                    "entry_search_objective",
                    "entry_search_long_center",
                    "entry_search_short_center",
                    "entry_search_probability_epsilon",
                    "entry_search_teacher_temperature",
                    "entry_search_q_temperature",
                    "entry_search_center_receipt",
                    "entry_search_center_receipt_sha256",
                })
            epsilon = float(item.get("entry_search_probability_epsilon", 1e-6))
            centered = item.get("entry_search_objective") == "centered_log_odds"
            center_receipt = str(item.get("entry_search_center_receipt", ""))
            center_receipt_sha256 = str(
                item.get("entry_search_center_receipt_sha256", "")
            )
            if (
                set(item) != required_fields
                or kind not in expected_channels
                or tuple(item.get("channels", ())) != expected_channels[kind]
                or float(item.get("loss_weight", 0.0)) <= 0
                or float(item.get("entry_search_loss_weight", 0.0)) < 0
                or (kind != "expansion" and float(item["entry_search_loss_weight"]) != 0)
                or not str(item.get("cache_root", "")).strip()
            ):
                label = str(kind).title()
                raise ValueError(f"training-only {label} teacher contract is invalid")
            if kind == "expansion" and (
                item["entry_search_objective"] not in {
                    "raw_probability", "centered_log_odds"
                }
                or not 0 < epsilon < 0.5
                or any(
                    not epsilon < float(item[field]) < 1.0 - epsilon
                    for field in (
                        "entry_search_long_center",
                        "entry_search_short_center",
                    )
                )
                or float(item["entry_search_teacher_temperature"]) <= 0
                or float(item["entry_search_q_temperature"]) <= 0
                or (
                    centered
                    and (
                        float(item["entry_search_loss_weight"]) <= 0
                        or not center_receipt
                        or len(center_receipt_sha256) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in center_receipt_sha256
                        )
                    )
                )
                or (not centered and (center_receipt or center_receipt_sha256))
            ):
                raise ValueError(
                    "training-only Expansion entry-search contract is invalid"
                )
            if float(item["entry_search_loss_weight"]) > 0 and index != 0:
                raise ValueError("entry-guiding Expansion teacher must be first")
        training.setdefault("teacher_loss_end_scale", 1.0)
        training.setdefault("teacher_guidance_dropout_start", 0.0)
        training.setdefault("teacher_guidance_dropout_end", 0.0)
        training.setdefault("teacher_autonomy_start_fraction", 1.0)
        if (
            isinstance(training["teacher_loss_end_scale"], bool)
            or not 0 <= float(training["teacher_loss_end_scale"]) <= 1
        ):
            raise ValueError("teacher loss end scale must be between zero and one")
        if isinstance(training["teacher_guidance_dropout_start"], bool) or isinstance(
            training["teacher_guidance_dropout_end"], bool
        ):
            raise ValueError("teacher guidance dropout schedule is invalid")
        dropout_start = float(training["teacher_guidance_dropout_start"])
        dropout_end = float(training["teacher_guidance_dropout_end"])
        if not 0 <= dropout_start <= dropout_end <= 1:
            raise ValueError("teacher guidance dropout schedule is invalid")
        if (
            isinstance(training["teacher_autonomy_start_fraction"], bool)
            or not 0 < float(training["teacher_autonomy_start_fraction"]) <= 1
        ):
            raise ValueError("teacher autonomy start fraction is invalid")
        payload["teachers"] = tuple(teachers)
    _validate_regime_selectivity(payload, root=root)
    temporal = payload.get("temporal") or {}
    ordered = [
        temporal.get("train_start"), temporal.get("train_end"),
        temporal.get("validation_start"), temporal.get("validation_end"),
        temporal.get("sealed_start"),
    ]
    if any(value is None for value in ordered):
        raise ValueError("temporal contract is incomplete")
    if not (
        ordered[0] < ordered[1]
        and ordered[1] <= ordered[2]
        and ordered[2] < ordered[3]
        and ordered[3] <= ordered[4]
    ):
        raise ValueError("temporal train, validation, and sealed periods overlap")
    sealed_confirmation = payload.get("sealed_confirmation")
    if sealed_confirmation is not None:
        required_confirmation = {
            "start",
            "end",
            "episode_sessions",
            "window_mode",
            "tickers",
            "minimum_episodes_per_ticker",
            "teacher_free",
            "minimum_pass_rate",
            "minimum_per_market_pass_rate",
            "maximum_blow_rate",
            "minimum_expectancy_r",
        }
        if (
            not isinstance(sealed_confirmation, dict)
            or set(sealed_confirmation) != required_confirmation
        ):
            raise ValueError("sealed confirmation contract is invalid")
        try:
            sealed_start = date.fromisoformat(str(sealed_confirmation["start"]))
            sealed_end = date.fromisoformat(str(sealed_confirmation["end"]))
        except ValueError as error:
            raise ValueError("sealed confirmation dates are invalid") from error
        confirmation_tickers = tuple(
            str(value) for value in sealed_confirmation["tickers"]
        )
        numeric_rates = (
            sealed_confirmation["minimum_pass_rate"],
            sealed_confirmation["minimum_per_market_pass_rate"],
            sealed_confirmation["maximum_blow_rate"],
        )
        if (
            sealed_confirmation["start"] != temporal["sealed_start"]
            or sealed_start >= sealed_end
            or sealed_confirmation["window_mode"] != "non_overlapping"
            or sealed_confirmation["teacher_free"] is not True
            or confirmation_tickers != tickers
            or isinstance(sealed_confirmation["episode_sessions"], bool)
            or int(sealed_confirmation["episode_sessions"])
            != int(challenge["episode_days"])
            or isinstance(sealed_confirmation["minimum_episodes_per_ticker"], bool)
            or int(sealed_confirmation["minimum_episodes_per_ticker"]) < 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in numeric_rates
            )
            or isinstance(sealed_confirmation["minimum_expectancy_r"], bool)
            or not isinstance(
                sealed_confirmation["minimum_expectancy_r"], (int, float)
            )
        ):
            raise ValueError(
                "sealed confirmation must be teacher-free, non-overlapping, "
                "and aligned with the frozen challenge and ticker contract"
            )
        sealed_confirmation["tickers"] = confirmation_tickers
        payload["sealed_confirmation"] = sealed_confirmation
    evolution = payload.get("evolution") or {}
    if not str(evolution.get("hypothesis", "")).strip():
        raise ValueError("evolution hypothesis is required")
    allowed = tuple(str(value) for value in evolution.get("allowed_revision_paths", ()))
    frozen = tuple(str(value) for value in evolution.get("frozen_paths", ()))
    campaign_declaration = payload.get("campaign")
    stages = (
        campaign_declaration.get("budget_stages")
        if isinstance(campaign_declaration, dict)
        else None
    )
    explicitly_nonrevisable = (
        isinstance(campaign_declaration, dict)
        and campaign_declaration.get("max_revisions_per_stage") == 0
        and not isinstance(
            campaign_declaration.get("max_revisions_per_stage"), bool
        )
        and isinstance(stages, list)
        and bool(stages)
        and all(
            isinstance(stage, dict)
            and stage.get("allow_revisions") is False
            and stage.get("revision_paths") == []
            for stage in stages
        )
    )
    if (not allowed and not explicitly_nonrevisable) or not frozen:
        raise ValueError("evolution revision and frozen paths must be declared")
    if any(
        path == locked or path.startswith(locked + ".")
        for path in allowed
        for locked in frozen
    ):
        raise ValueError("evolution allowlist overlaps the frozen contract")
    if payload.get("entry_supervision") is not None and "entry_supervision" not in frozen:
        raise ValueError("entry supervision must be frozen for the campaign")
    if (
        isinstance(training.get("short_circuit"), dict)
        and training["short_circuit"].get("policy_health") is not None
        and TRAINING_POLICY_HEALTH_FROZEN_PATH not in frozen
    ):
        raise ValueError("training short circuit policy health must be frozen")
    if payload.get("regime_selectivity") is not None:
        side_balance = payload["regime_selectivity"].get("side_balance")
        if side_balance is None:
            valid_regime_identity = all(
                path in frozen
                for path in LEGACY_REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS
            )
        else:
            semantics_paths = set(REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS)
            revisable_semantics = semantics_paths & set(allowed)
            valid_regime_identity = all(
                path in frozen
                for path in REGIME_SELECTIVITY_CORE_FROZEN_IDENTITY_PATHS
            ) and (
                all(path in frozen for path in semantics_paths)
                or revisable_semantics == semantics_paths
            )
            if revisable_semantics and revisable_semantics != semantics_paths:
                valid_regime_identity = False
            margin_paths = {
                "regime_selectivity.chop_wait_margin",
                "regime_selectivity.failed_confluence_margin",
            }
            if margin_fields := (
                {"chop_wait_margin", "failed_confluence_margin"}
                & set(payload["regime_selectivity"])
            ):
                valid_regime_identity = (
                    valid_regime_identity
                    and margin_fields
                    == {"chop_wait_margin", "failed_confluence_margin"}
                    and margin_paths <= set(frozen)
                )
            paired_margin_frozen = (
                "paired_a_plus_margin" in payload["regime_selectivity"]
            )
            if paired_margin_frozen:
                valid_regime_identity = (
                    valid_regime_identity
                    and "regime_selectivity.paired_a_plus_margin" in frozen
                )
        if not valid_regime_identity:
            raise ValueError(
                "Regime selectivity identity must be frozen for the campaign"
            )
        if (
            float(training.get("regime_wait_sequence_fraction", 0.0)) > 0.0
            and "training.regime_wait_sequence_fraction" not in frozen
        ):
            raise ValueError(
                "Regime WAIT replay identity must be frozen for the campaign"
            )
        if (
            int(training.get("regime_wait_sequence_update_period", 0)) > 0
            and "training.regime_wait_sequence_update_period" not in frozen
        ):
            raise ValueError(
                "Regime WAIT replay identity must be frozen for the campaign"
            )
    if payload.get("recovery_curriculum") is not None:
        if not all(path in frozen for path in RECOVERY_CURRICULUM_FROZEN_PATHS):
            raise ValueError(
                "recovery curriculum account identity must be frozen"
            )
        recovery_allowed = {
            path for path in allowed if path.startswith("recovery_curriculum.")
        }
        if recovery_allowed != set(RECOVERY_CURRICULUM_REVISION_PATHS):
            raise ValueError(
                "recovery curriculum revision surface is invalid"
            )
    revision_bounds = evolution.get("revision_bounds") or {}
    if not isinstance(revision_bounds, dict):
        raise ValueError("evolution revision bounds must be an object")
    for revision_path, bounds in revision_bounds.items():
        if revision_path not in allowed:
            raise ValueError(
                f"revision bound path is not allowlisted: {revision_path}"
            )
        if not isinstance(bounds, dict) or set(bounds) != {"minimum", "maximum"}:
            raise ValueError(f"revision bounds are invalid for {revision_path}")
        minimum, maximum = bounds["minimum"], bounds["maximum"]
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or float(minimum) > float(maximum)
        ):
            raise ValueError(f"revision bounds are invalid for {revision_path}")
    evolution["allowed_revision_paths"] = allowed
    evolution["frozen_paths"] = frozen
    missing_frozen_paths = tuple(
        path for path in frozen if not _recipe_path_exists(payload, path)
    )
    if missing_frozen_paths:
        raise ValueError(
            "frozen recipe path does not exist: "
            + ", ".join(missing_frozen_paths)
        )
    evolution["parent_candidate_ids"] = tuple(
        str(value) for value in evolution.get("parent_candidate_ids", ())
    )
    base_parent = evolution.get("base_parent")
    if base_parent is not None:
        required_parent_fields = {
            "archive_root", "candidate_id", "evaluation_id", "model_sha256"
        }
        if (
            not isinstance(base_parent, dict)
            or set(base_parent) != required_parent_fields
            or any(
                not str(base_parent[field]).strip()
                for field in required_parent_fields
            )
        ):
            raise ValueError("external base parent contract is invalid")
        parent_ids = evolution["parent_candidate_ids"]
        if len(parent_ids) != 1:
            raise ValueError(
                "external base parent requires exactly one parent candidate"
            )
        if parent_ids[0] != str(base_parent["candidate_id"]):
            raise ValueError(
                "external base parent disagrees with parent candidate identity"
            )
        evolution["base_parent"] = {
            field: str(base_parent[field]) for field in required_parent_fields
        }
    payload["evolution"] = evolution
    campaign = payload.get("campaign") or {}
    if not str(campaign.get("state_root", "")).strip():
        raise ValueError("campaign state_root is required")
    max_revisions = campaign.get("max_revisions_per_stage")
    if isinstance(max_revisions, bool) or not isinstance(max_revisions, int):
        raise ValueError("campaign max revisions must be an integer")
    if max_revisions < 0:
        raise ValueError("campaign max revisions must be nonnegative")
    reasoning = campaign.get("reasoning") or {}
    if reasoning.get("provider") not in {"codex", "manual"}:
        raise ValueError("campaign reasoning provider must be codex or manual")
    proposer = reasoning.get("proposer", "standard")
    if proposer not in {"standard", "gepa_reflective"}:
        raise ValueError(
            "campaign reasoning proposer must be standard or gepa_reflective"
        )
    reasoning["proposer"] = proposer
    campaign["reasoning"] = reasoning
    near_blow_loss_fraction = campaign.get("near_blow_loss_fraction", 0.75)
    if (
        isinstance(near_blow_loss_fraction, bool)
        or not isinstance(near_blow_loss_fraction, (int, float))
        or not 0 < float(near_blow_loss_fraction) <= 1
    ):
        raise ValueError("campaign near-blow loss fraction must be in (0, 1]")
    campaign["near_blow_loss_fraction"] = float(near_blow_loss_fraction)
    requirements = campaign.get("selection_requirements") or ()
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("campaign selection requirements must be nonempty")
    for requirement in requirements:
        if (
            not isinstance(requirement, dict)
            or requirement.get("operator") not in {">", ">=", "<", "<=", "=="}
            or not str(requirement.get("metric", "")).strip()
            or isinstance(requirement.get("value"), bool)
            or not isinstance(requirement.get("value"), (int, float))
        ):
            raise ValueError("campaign selection requirement is invalid")
    budget_stages = campaign.get("budget_stages")
    if budget_stages is None:
        if training_budget_mode == "environment_steps":
            budget_stages = [{
                "name": "historical_candidate",
                "minimum_environment_steps": int(
                    training["minimum_environment_steps"]
                ),
                "selection_requirements": requirements,
            }]
        else:
            stage_short_circuit = training.get("short_circuit")
            stage = {
                "name": "historical_candidate",
                "budget_mode": "episodes",
                "training_episodes": int(training["episodes"]),
                "validation_episodes": int(training["validation_episodes"]),
                "selection_requirements": requirements,
            }
            if stage_short_circuit:
                stage["short_circuit_minimum_episodes"] = int(
                    stage_short_circuit["minimum_completed_episodes"]
                )
            budget_stages = [stage]
    if not isinstance(budget_stages, list) or not budget_stages:
        raise ValueError("campaign budget stages must be a nonempty array")
    names = []
    budgets = []
    for stage_index, stage in enumerate(budget_stages):
        if not isinstance(stage, dict):
            raise ValueError("campaign budget stage contract is invalid")
        required_stage_fields = {"name", "selection_requirements"}
        optional_stage_fields = {
            "budget_mode", "seed", "seeds", "max_parallel",
            "allow_revisions",
            "parent_improvement_requirements", "warm_start_parent",
            "parent_improvement_any_requirements", "curriculum_override",
            "parent_retention_requirements", "revision_paths",
            "minimum_environment_steps", "training_episodes",
            "validation_episodes", "short_circuit_minimum_episodes",
            "episode_coverage",
        }
        if (
            not required_stage_fields <= set(stage)
            or set(stage) - required_stage_fields - optional_stage_fields
        ):
            raise ValueError("campaign budget stage contract is invalid")
        name = str(stage["name"])
        budget_mode = stage.get("budget_mode", "environment_steps")
        stage_requirements = stage["selection_requirements"]
        if (
            not name
            or budget_mode not in {"environment_steps", "episodes"}
            or budget_mode != training_budget_mode
            or not isinstance(stage_requirements, list)
            or not stage_requirements
        ):
            raise ValueError("campaign budget stage contract is invalid")
        if budget_mode == "environment_steps":
            if (
                "training_episodes" in stage
                or "validation_episodes" in stage
                or "short_circuit_minimum_episodes" in stage
                or isinstance(stage.get("minimum_environment_steps"), bool)
                or not isinstance(stage.get("minimum_environment_steps"), int)
                or stage["minimum_environment_steps"] < 1
            ):
                raise ValueError("campaign step budget stage contract is invalid")
            budget = stage["minimum_environment_steps"]
        else:
            stage_short_circuit = stage.get(
                "short_circuit_minimum_episodes"
            )
            if (
                "minimum_environment_steps" in stage
                or isinstance(stage.get("training_episodes"), bool)
                or not isinstance(stage.get("training_episodes"), int)
                or stage["training_episodes"] < 1
                or stage["training_episodes"] > int(training["episodes"])
                or isinstance(stage.get("validation_episodes"), bool)
                or not isinstance(stage.get("validation_episodes"), int)
                or stage["validation_episodes"] < 1
                or (
                    short_circuit is None
                    and "short_circuit_minimum_episodes" in stage
                )
                or (
                    short_circuit is not None
                    and (
                        isinstance(stage_short_circuit, bool)
                        or not isinstance(stage_short_circuit, int)
                        or not 1
                        <= stage_short_circuit
                        <= stage["training_episodes"]
                    )
                )
            ):
                raise ValueError("campaign episode budget stage contract is invalid")
            budget = stage["training_episodes"]
            policy_health = (
                short_circuit.get("policy_health")
                if isinstance(short_circuit, dict)
                else None
            )
            if policy_health is not None and any(
                int(boundary) > budget
                for boundary in (
                    policy_health["minimum_completed_episodes"],
                    policy_health["probe_interval_episodes"],
                    policy_health["economic_futility"][
                        "minimum_completed_episodes"
                    ],
                )
            ):
                raise ValueError(
                    "campaign episode budget cannot reach policy health boundary"
                )
        episode_coverage = stage.get("episode_coverage")
        if episode_coverage is not None:
            required_coverage_gates = ({
                "metric": "training.minimum_market_episode_coverage",
                "operator": "==",
                "value": 1.0,
            }, {
                "metric": "training.episode_coverage_complete",
                "operator": "==",
                "value": 1.0,
            })
            if (
                budget_mode != "episodes"
                or stage_index != len(budget_stages) - 1
                or not isinstance(episode_coverage, dict)
                or set(episode_coverage) != {"schema", "episode_budget"}
                or episode_coverage.get("schema")
                != "full_data_episode_coverage_v1"
                or isinstance(episode_coverage.get("episode_budget"), bool)
                or not isinstance(episode_coverage.get("episode_budget"), int)
                or episode_coverage["episode_budget"]
                != stage["training_episodes"]
                or any(
                    gate not in stage_requirements
                    for gate in required_coverage_gates
                )
            ):
                raise ValueError(
                    "campaign episode coverage contract is invalid"
                )
        seed = stage.get("seed")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise ValueError("campaign budget stage seed is invalid")
        seeds = stage.get("seeds")
        max_parallel = stage.get("max_parallel")
        if seed is not None and seeds is not None:
            raise ValueError("campaign budget stage cannot set seed and seeds")
        if seeds is not None:
            if (
                not isinstance(seeds, list)
                or not seeds
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in seeds
                )
                or len(set(seeds)) != len(seeds)
                or isinstance(max_parallel, bool)
                or not isinstance(max_parallel, int)
                or max_parallel < 1
                or max_parallel > len(seeds)
                or stage.get("allow_revisions", True) is not False
            ):
                raise ValueError("campaign multi-seed stage contract is invalid")
            stage["seeds"] = tuple(seeds)
        elif max_parallel is not None:
            raise ValueError("campaign max_parallel requires seeds")
        if not isinstance(stage.get("allow_revisions", True), bool):
            raise ValueError("campaign budget stage revision policy is invalid")
        if not isinstance(stage.get("warm_start_parent", False), bool):
            raise ValueError("campaign warm-start policy is invalid")
        curriculum_override = stage.get("curriculum_override", {})
        if not isinstance(curriculum_override, dict) or any(
            not str(path).strip() for path in curriculum_override
        ):
            raise ValueError("campaign curriculum override is invalid")
        if any(
            path not in payload["evolution"]["allowed_revision_paths"]
            for path in curriculum_override
        ):
            raise ValueError("campaign curriculum override is not allowlisted")
        semantics_override_paths = set(curriculum_override) & set(
            REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS
        )
        if semantics_override_paths:
            from .balance_aware_regime_selectivity import (
                PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
                PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
                STATIC_STATE_SEMANTICS,
            )

            if semantics_override_paths != set(
                REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS
            ):
                raise ValueError(
                    "Regime semantics curriculum override must be atomic"
                )
            override_semantics = curriculum_override[
                "regime_selectivity.semantics"
            ]
            override_formula = curriculum_override[
                "regime_selectivity.formula"
            ]
            override_emphasis = curriculum_override[
                "regime_selectivity.persistent_chop_negative_emphasis"
            ]
            valid_static = (
                override_semantics == STATIC_STATE_SEMANTICS
                and override_formula
                == payload["regime_selectivity"]["formula"]
                and float(override_emphasis) == 0.0
            )
            valid_persistent = (
                override_semantics
                == PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
                and override_formula == PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA
                and isinstance(override_emphasis, (int, float))
                and not isinstance(override_emphasis, bool)
                and math.isfinite(float(override_emphasis))
                and float(override_emphasis) > 0.0
            )
            if not (valid_static or valid_persistent):
                raise ValueError("Regime semantics curriculum override is invalid")
        for override_path, value in curriculum_override.items():
            bounds = revision_bounds.get(override_path)
            if bounds is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not float(bounds["minimum"])
                <= float(value)
                <= float(bounds["maximum"])
            ):
                raise ValueError("campaign curriculum override exceeds its bounds")
        revision_paths = stage.get(
            "revision_paths", payload["evolution"]["allowed_revision_paths"]
        )
        if (
            not isinstance(revision_paths, (list, tuple))
            or len(set(revision_paths)) != len(revision_paths)
            or any(
                path not in payload["evolution"]["allowed_revision_paths"]
                for path in revision_paths
            )
        ):
            raise ValueError("campaign stage revision paths are invalid")
        stage["revision_paths"] = tuple(revision_paths)
        for requirement in stage_requirements:
            if (
                not isinstance(requirement, dict)
                or requirement.get("operator") not in {">", ">=", "<", "<=", "=="}
                or not str(requirement.get("metric", "")).strip()
                or isinstance(requirement.get("value"), bool)
                or not isinstance(requirement.get("value"), (int, float))
            ):
                raise ValueError("campaign budget stage requirement is invalid")
        parent_requirements = stage.get("parent_improvement_requirements", [])
        if not isinstance(parent_requirements, list):
            raise ValueError("campaign parent-improvement requirements must be an array")
        for requirement in parent_requirements:
            if (
                not isinstance(requirement, dict)
                or set(requirement) != {"metric", "direction", "minimum_delta"}
                or requirement.get("direction") not in {"maximize", "minimize"}
                or not str(requirement.get("metric", "")).strip()
                or isinstance(requirement.get("minimum_delta"), bool)
                or not isinstance(requirement.get("minimum_delta"), (int, float))
                or float(requirement["minimum_delta"]) < 0
            ):
                raise ValueError(
                    "campaign parent-improvement requirement is invalid"
                )
        parent_any_requirements = stage.get(
            "parent_improvement_any_requirements", []
        )
        if not isinstance(parent_any_requirements, list) or (
            parent_any_requirements and len(parent_any_requirements) < 2
        ):
            raise ValueError(
                "campaign any-parent-improvement requirements must contain "
                "at least two alternatives"
            )
        for requirement in parent_any_requirements:
            if (
                not isinstance(requirement, dict)
                or set(requirement) != {"metric", "direction", "minimum_delta"}
                or requirement.get("direction") not in {"maximize", "minimize"}
                or not str(requirement.get("metric", "")).strip()
                or isinstance(requirement.get("minimum_delta"), bool)
                or not isinstance(requirement.get("minimum_delta"), (int, float))
                or float(requirement["minimum_delta"]) < 0
            ):
                raise ValueError(
                    "campaign any-parent-improvement requirement is invalid"
                )
        parent_retention_requirements = stage.get(
            "parent_retention_requirements", []
        )
        if not isinstance(parent_retention_requirements, list):
            raise ValueError(
                "campaign parent-retention requirements must be a list"
            )
        for requirement in parent_retention_requirements:
            requirement_keys = (
                frozenset(requirement)
                if isinstance(requirement, dict)
                else frozenset()
            )
            if (
                not isinstance(requirement, dict)
                or requirement_keys not in {
                    frozenset({"metric", "maximum_regression"}),
                    frozenset({"metric", "direction", "maximum_regression"}),
                }
                or requirement.get("direction", "minimize") not in {
                    "maximize", "minimize"
                }
                or not str(requirement.get("metric", "")).strip()
                or isinstance(requirement.get("maximum_regression"), bool)
                or not isinstance(
                    requirement.get("maximum_regression"), (int, float)
                )
                or not math.isfinite(float(requirement["maximum_regression"]))
                or float(requirement["maximum_regression"]) < 0
            ):
                raise ValueError(
                    "campaign parent-retention requirement is invalid"
                )
        names.append(name)
        budgets.append(budget)
    if len(set(names)) != len(names) or budgets != sorted(budgets):
        raise ValueError("campaign budget stages must have unique names and increasing budgets")
    if (
        base_parent is None
        and evolution["parent_candidate_ids"]
        and bool(budget_stages[0].get("warm_start_parent", False))
    ):
        raise ValueError(
            "first-stage external warm start requires a base parent contract"
        )
    if base_parent is not None and not bool(
        budget_stages[0].get("warm_start_parent", False)
    ):
        raise ValueError(
            "external base parent requires first-stage warm start"
        )
    campaign["budget_stages"] = tuple(budget_stages)
    finalization = campaign.get("finalization")
    if finalization is not None:
        if not isinstance(finalization, dict):
            raise ValueError("campaign finalization must be an object")
        required = {
            "registry_root", "export_root", "minimum_seed_count", "ranking"
        }
        if set(finalization) != required:
            raise ValueError("campaign finalization contract is invalid")
        minimum_seed_count = finalization["minimum_seed_count"]
        ranking = finalization["ranking"]
        if (
            not str(finalization["registry_root"]).strip()
            or not str(finalization["export_root"]).strip()
            or isinstance(minimum_seed_count, bool)
            or not isinstance(minimum_seed_count, int)
            or minimum_seed_count < 1
            or not isinstance(ranking, list)
            or not ranking
        ):
            raise ValueError("campaign finalization contract is invalid")
        seed_count = sum(len(stage.get("seeds", ())) for stage in budget_stages)
        if minimum_seed_count > seed_count:
            raise ValueError("campaign finalization requires more seeds than declared")
        for rule in ranking:
            if (
                not isinstance(rule, dict)
                or set(rule) != {"metric", "direction"}
                or not str(rule["metric"]).strip()
                or rule["direction"] not in {"minimize", "maximize"}
            ):
                raise ValueError("campaign finalization ranking is invalid")
    diagnostics = campaign.get("diagnostic_targets", [])
    if not isinstance(diagnostics, list):
        raise ValueError("campaign diagnostic targets must be an array")
    for target in diagnostics:
        if (
            not isinstance(target, dict)
            or target.get("operator") not in {">", ">=", "<", "<=", "=="}
            or not str(target.get("metric", "")).strip()
            or isinstance(target.get("value"), bool)
            or not isinstance(target.get("value"), (int, float))
        ):
            raise ValueError("campaign diagnostic target is invalid")
    niches = campaign.get("niches") or ()
    if not isinstance(niches, list) or not niches:
        raise ValueError("campaign niches must be nonempty")
    payload["campaign"] = campaign
    payload["tickers"] = tickers
    payload["deployment_tickers"] = deployment
    payload["training_only_tickers"] = training_only
    payload["_path"] = str(path)
    payload["_root"] = str(root)
    return payload
