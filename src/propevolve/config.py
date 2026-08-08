"""Fail-closed experiment configuration for historical training."""

from __future__ import annotations

import json
from pathlib import Path


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("schema") != "propevolve_historical_training_v1":
        raise ValueError("unsupported PropEvolve experiment schema")
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
    if max_revisions < 1:
        raise ValueError("campaign max revisions must be positive")
    reasoning = campaign.get("reasoning") or {}
    if reasoning.get("provider") not in {"codex", "manual"}:
        raise ValueError("campaign reasoning provider must be codex or manual")
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
