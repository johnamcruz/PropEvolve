"""Fail-closed experiment configuration for historical training."""

from __future__ import annotations

from datetime import date
import json
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
        "minimum_environment_steps",
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


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != "propevolve_historical_training_v1":
        raise ValueError("unsupported PropEvolve experiment schema")
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
    cache = payload["cache"]
    if cache["format"] not in {"native", "ffm_frozen_representation_v2"}:
        raise ValueError("cache format must be native or ffm_frozen_representation_v2")
    if not str(cache["encoder_identity_sha256"]).strip():
        raise ValueError("cache encoder identity must be declared")
    agent = payload["agent"]
    agent.setdefault("n_step_return", 1)
    agent.setdefault("recurrent_burn_in", 0)
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
    training.setdefault("terminal_sequence_fraction", 0.0)
    training.setdefault("safety_sequence_fraction", 0.0)
    training.setdefault("entry_opportunity_sequence_fraction", 0.0)
    training.setdefault("management_epsilon_start", training["epsilon_start"])
    training.setdefault("management_epsilon_end", training["epsilon_end"])
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
        or float(training["terminal_sequence_fraction"])
        + float(training["safety_sequence_fraction"]) > 1
        or float(training["terminal_sequence_fraction"])
        + float(training["safety_sequence_fraction"])
        + float(training["entry_opportunity_sequence_fraction"]) > 1
    ):
        raise ValueError("training replay sequence fractions are invalid")
    if (
        isinstance(training["checkpoint_every_episodes"], bool)
        or int(training["checkpoint_every_episodes"]) < 1
    ):
        raise ValueError("training checkpoint interval must be positive")
    short_circuit = training.get("short_circuit")
    if short_circuit is not None:
        required_short_circuit = {
            "minimum_environment_steps",
            "minimum_passes",
            "maximum_blow_rate",
        }
        if (
            not isinstance(short_circuit, dict)
            or frozenset(short_circuit) not in {
                frozenset(required_short_circuit),
                frozenset((*required_short_circuit, "collapse")),
            }
            or isinstance(short_circuit["minimum_environment_steps"], bool)
            or not isinstance(short_circuit["minimum_environment_steps"], int)
            or not 1
            <= short_circuit["minimum_environment_steps"]
            <= int(training["minimum_environment_steps"])
            or isinstance(short_circuit["minimum_passes"], bool)
            or not isinstance(short_circuit["minimum_passes"], int)
            or short_circuit["minimum_passes"] < 0
            or isinstance(short_circuit["maximum_blow_rate"], bool)
            or not isinstance(short_circuit["maximum_blow_rate"], (int, float))
            or not 0 <= float(short_circuit["maximum_blow_rate"]) <= 1
        ):
            raise ValueError("training short circuit contract is invalid")
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
            if (
                set(item) != {
                    "kind", "cache_root", "channels", "loss_weight",
                    "entry_search_loss_weight",
                }
                or kind not in expected_channels
                or tuple(item.get("channels", ())) != expected_channels[kind]
                or float(item.get("loss_weight", 0.0)) <= 0
                or float(item.get("entry_search_loss_weight", 0.0)) < 0
                or (kind != "expansion" and float(item["entry_search_loss_weight"]) != 0)
                or not str(item.get("cache_root", "")).strip()
            ):
                label = str(kind).title()
                raise ValueError(f"training-only {label} teacher contract is invalid")
            if float(item["entry_search_loss_weight"]) > 0 and index != 0:
                raise ValueError("entry-guiding Expansion teacher must be first")
        training.setdefault("teacher_loss_end_scale", 1.0)
        training.setdefault("teacher_guidance_dropout_start", 0.0)
        training.setdefault("teacher_guidance_dropout_end", 0.0)
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
        payload["teachers"] = tuple(teachers)
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
    if not allowed or not frozen:
        raise ValueError("evolution revision and frozen paths must be declared")
    if any(
        path == locked or path.startswith(locked + ".")
        for path in allowed
        for locked in frozen
    ):
        raise ValueError("evolution allowlist overlaps the frozen contract")
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
    evolution["parent_candidate_ids"] = tuple(
        str(value) for value in evolution.get("parent_candidate_ids", ())
    )
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
        budget_stages = [{
            "name": "historical_candidate",
            "minimum_environment_steps": int(
                training["minimum_environment_steps"]
            ),
            "selection_requirements": requirements,
        }]
    if not isinstance(budget_stages, list) or not budget_stages:
        raise ValueError("campaign budget stages must be a nonempty array")
    names = []
    budgets = []
    for stage in budget_stages:
        required_stage_fields = {
            "name", "minimum_environment_steps", "selection_requirements"
        }
        optional_stage_fields = {
            "seed", "seeds", "max_parallel", "allow_revisions",
            "parent_improvement_requirements", "warm_start_parent",
            "curriculum_override", "revision_paths",
        }
        if (
            not isinstance(stage, dict)
            or not required_stage_fields <= set(stage)
            or set(stage) - required_stage_fields - optional_stage_fields
        ):
            raise ValueError("campaign budget stage contract is invalid")
        name = str(stage["name"])
        budget = stage["minimum_environment_steps"]
        stage_requirements = stage["selection_requirements"]
        if (
            not name
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget < 1
            or not isinstance(stage_requirements, list)
            or not stage_requirements
        ):
            raise ValueError("campaign budget stage contract is invalid")
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
        names.append(name)
        budgets.append(budget)
    if len(set(names)) != len(names) or budgets != sorted(budgets):
        raise ValueError("campaign budget stages must have unique names and increasing budgets")
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
    payload["_path"] = str(path.resolve())
    payload["_root"] = str(
        path.parent.parent.resolve()
        if path.parent.name == "config"
        else path.parent.resolve()
    )
    return payload
