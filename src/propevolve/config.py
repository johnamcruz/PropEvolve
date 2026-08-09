"""Fail-closed experiment configuration for historical training."""

from __future__ import annotations

import json
from pathlib import Path


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
    "training": {
        "episodes",
        "minimum_environment_steps",
        "validation_episodes",
        "replay_capacity_episodes",
        "sequence_length",
        "warmup_episodes",
        "updates_per_episode",
        "batch_sequences",
        "recurrent_horizon",
        "epsilon_start",
        "epsilon_end",
        "seed",
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
    _require_recipe_fields(payload)
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
    if agent["device"] not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("agent device must be auto, cuda, mps, or cpu")
    training = payload["training"]
    training.setdefault("terminal_sequence_fraction", 0.0)
    if not 0 <= float(training["epsilon_end"]) <= float(training["epsilon_start"]) <= 1:
        raise ValueError("training epsilon schedule is invalid")
    if not 0 <= float(training["terminal_sequence_fraction"]) <= 1:
        raise ValueError("training terminal sequence fraction must be between zero and one")
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
