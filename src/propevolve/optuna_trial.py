"""Direct Optuna trial execution over the production learner and validator."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import load_experiment_config
from .evolution import CandidateArchive


OPTUNA_TRIAL_RESULT_SCHEMA = "propevolve_optuna_trial_result_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_optuna_trial(
    config_path: str | Path,
    *,
    result_path: str | Path,
) -> dict:
    """Run one selection trial without campaign promotion or revision gates."""
    config_file = Path(config_path).resolve(strict=True)
    config = load_experiment_config(config_file)
    stages = tuple(config["campaign"]["budget_stages"])
    if len(stages) != 1:
        raise ValueError("Optuna trial requires exactly one compiled budget stage")
    stage = stages[0]
    if str(stage.get("budget_mode", "environment_steps")) != "episodes":
        raise ValueError("Optuna trial requires an episode budget")

    training = dict(config["training"])
    training["budget_mode"] = "episodes"
    training["episodes"] = int(stage["training_episodes"])
    training["validation_episodes"] = int(stage["validation_episodes"])
    stage_short_circuit = stage.get("short_circuit_minimum_episodes")
    if stage_short_circuit is None:
        training["short_circuit"] = None
    else:
        short_circuit = dict(training["short_circuit"])
        short_circuit.pop("minimum_environment_steps", None)
        short_circuit["minimum_completed_episodes"] = int(stage_short_circuit)
        training["short_circuit"] = short_circuit
    if stage.get("episode_coverage") is None:
        training.pop("episode_coverage", None)
    else:
        training["episode_coverage"] = dict(stage["episode_coverage"])
    config["training"] = training

    root = Path(config["_root"])
    output = _resolve(root, str(config["output"]))
    config["_archive_output"] = str(output)
    config["_validation_stop_on_blow"] = any(
        requirement.get("metric") == "selection.blow_rate"
        and requirement.get("operator") == "=="
        and float(requirement.get("value")) == 0.0
        for requirement in stage["selection_requirements"]
    )

    archive = CandidateArchive(output / "archive")
    base_parent = config["evolution"].get("base_parent")
    parents = tuple(config["evolution"]["parent_candidate_ids"])
    if base_parent is not None:
        from .orchestration import _assert_parent_causal_contract

        parent = archive.register_external_parent(
            _resolve(root, str(base_parent["archive_root"])),
            candidate_id=str(base_parent["candidate_id"]),
            evaluation_id=str(base_parent["evaluation_id"]),
            model_sha256=str(base_parent["model_sha256"]),
        )
        _assert_parent_causal_contract(parent, config)
        if bool(stage.get("warm_start_parent", False)):
            if parents != (parent.candidate_id,):
                raise ValueError(
                    "Optuna warm start requires the declared parent candidate"
                )
            config["_warm_start_model"] = {
                "candidate_id": parent.candidate_id,
                "model_path": str(parent.model_path),
                "model_sha256": parent.manifest["model_sha256"],
            }

    from .training import HistoricalCandidateRunner

    candidate, evaluation = HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=parents,
        hypothesis=str(config["evolution"]["hypothesis"]),
        collect_all_evidence=True,
    )
    metrics = {str(name): float(value) for name, value in evaluation.metrics.items()}
    if not metrics or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Optuna trial metrics must be present and finite")
    payload = {
        "schema": OPTUNA_TRIAL_RESULT_SCHEMA,
        "config_path": str(config_file),
        "config_sha256": _sha256(config_file),
        "candidate_id": candidate.candidate_id,
        "evaluation_id": evaluation.evaluation_id,
        "evaluation_status": evaluation.status,
        "metrics": metrics,
    }
    _write_result(Path(result_path), payload)
    return payload


__all__ = ["OPTUNA_TRIAL_RESULT_SCHEMA", "run_optuna_trial"]
