"""JSON-driven Optuna selection over direct PropEvolve training trials."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Mapping, Sequence

import optuna


OPTUNA_SWEEP_SCHEMA = "propevolve_optuna_sweep_v2"
RESULT_SCHEMA = "propevolve_optuna_sweep_result_v2"
_CONSTRAINT_OPERATORS = frozenset({"==", ">=", ">", "<="})


def _set_path(payload: dict, dotted_path: str, value: object) -> None:
    parts = dotted_path.split(".")
    target = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"Optuna path does not resolve: {dotted_path}")
        target = child
    if parts[-1] not in target:
        raise ValueError(f"Optuna path does not exist: {dotted_path}")
    target[parts[-1]] = value


def _get_path(payload: Mapping, dotted_path: str) -> object:
    target: object = payload
    for part in dotted_path.split("."):
        if not isinstance(target, Mapping) or part not in target:
            raise ValueError(f"Optuna path does not resolve: {dotted_path}")
        target = target[part]
    return target


def _write_exact(path: Path, payload: Mapping) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"existing Optuna artifact drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text() != rendered:
                raise ValueError(f"existing Optuna artifact drifted: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _load_trial_result(config_path: Path, result_path: Path) -> dict | None:
    if not result_path.is_file():
        return None
    from .optuna_trial import OPTUNA_TRIAL_RESULT_SCHEMA

    payload = json.loads(result_path.read_text())
    if (
        payload.get("schema") != OPTUNA_TRIAL_RESULT_SCHEMA
        or payload.get("config_sha256") != _sha256(config_path.read_bytes())
        or not isinstance(payload.get("metrics"), Mapping)
    ):
        raise ValueError("Optuna trial result identity drifted")
    return payload


def _default_trial_runner(
    config_path: Path,
    *,
    run_id: str,
    result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    worker_environment = os.environ.copy()
    worker_environment.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    })
    with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "propevolve.cli",
                "optuna-trial",
                "--config",
                str(config_path),
                "--result",
                str(result_path),
            ],
            stdout=stdout,
            stderr=stderr,
            env=worker_environment,
            check=False,
        )
    result = _load_trial_result(config_path, result_path)
    if result is None:
        raise RuntimeError(
            f"Optuna trial exited {process.returncode} without a result"
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Optuna trial {run_id} exited unexpectedly with "
            f"{process.returncode}"
        )
    return result


def _cell_metrics(state, stage_name: str) -> dict[str, float]:
    receipts = tuple(
        receipt for receipt in state.receipts if receipt.stage == stage_name
    )
    if len(receipts) != 1:
        raise ValueError(
            f"terminal injected Optuna state must have one {stage_name!r} receipt"
        )
    raw = receipts[0].outputs.get("metrics")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("terminal injected Optuna metrics are missing")
    metrics = {str(key): float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("terminal injected Optuna metrics must be finite")
    return metrics


def _trial_metrics(result: object, stage_name: str) -> tuple[str, dict[str, float]]:
    if isinstance(result, Mapping):
        raw = result.get("metrics")
        status = str(result.get("evaluation_status", "PASS"))
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("terminal Optuna trial metrics are missing")
        metrics = {str(key): float(value) for key, value in raw.items()}
    else:
        # Preserve the injected-runner seam used by inexpensive orchestration
        # tests while the production path always uses a direct trial receipt.
        status = str(getattr(getattr(result, "phase", None), "value", "PASS"))
        metrics = _cell_metrics(result, stage_name)
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("terminal Optuna trial metrics must be finite")
    return status, metrics


@dataclass(frozen=True)
class ObjectiveTerm:
    metric: str
    weight: float
    floor: float | None = None
    cap: float | None = None


@dataclass(frozen=True)
class SweepStage:
    name: str
    training_episodes: int
    validation_episodes: int
    start_pnls: tuple[float, ...]
    balance_validation_episodes: int
    short_circuit: dict | None


@dataclass(frozen=True)
class PromotionContract:
    enabled: bool
    top_k: int
    finalist_top_k: int
    seeds: tuple[int, ...]
    seed_paths: tuple[str, ...]
    required_feasible_seeds: int
    acceptance: dict[str, tuple[str, float]]


@dataclass(frozen=True)
class OptunaSweep:
    path: Path
    name: str
    base_config_path: Path
    base_config: dict
    base_config_sha256: str
    study_root: str
    sampler: str
    seed: int
    n_trials: int
    n_jobs: int
    n_startup_trials: int
    multivariate: bool
    group: bool
    screening_artifact_retention: str
    maximum_retained_feasible_trials: int | None
    objective_terms: tuple[ObjectiveTerm, ...]
    constraints: dict[str, tuple[str, float]]
    search_space: dict[str, dict]
    stages: dict[str, SweepStage]
    promotion: PromotionContract
    frozen_paths: tuple[str, ...]
    allowed_search_prefixes: tuple[str, ...]
    external_parent_economic_search_paths: tuple[str, ...]

    @property
    def screening_stage(self) -> str:
        return self.stages["screening"].name


@dataclass(frozen=True)
class OptunaSweepResult:
    name: str
    status: str
    best_trial_number: int | None
    completed_trials: int
    pruned_trials: int
    failed_trials: int
    interrupted_trials: int
    study_path: Path
    result_path: Path
    promoted_trial_numbers: tuple[int, ...] = ()
    winner_trial_number: int | None = None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _study_storage(path: Path) -> optuna.storages.JournalStorage:
    """Use Optuna's file-locked journal for concurrent local workers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(path))
    )


def _constraint_contract(
    raw: object,
    *,
    label: str,
) -> dict[str, tuple[str, float]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Optuna {label} constraints are required")
    constraints: dict[str, tuple[str, float]] = {}
    for metric, rule in raw.items():
        if (
            not isinstance(metric, str)
            or not metric
            or not isinstance(rule, dict)
            or set(rule) != {"operator", "value"}
            or rule["operator"] not in _CONSTRAINT_OPERATORS
            or isinstance(rule["value"], bool)
            or not isinstance(rule["value"], (int, float))
            or not math.isfinite(float(rule["value"]))
        ):
            raise ValueError(f"Optuna {label} constraint is invalid: {metric}")
        constraints[metric] = (str(rule["operator"]), float(rule["value"]))
    return constraints


def _objective_contract(raw: object) -> tuple[ObjectiveTerm, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Optuna objective terms are required")
    terms: list[ObjectiveTerm] = []
    metrics: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Optuna objective term is invalid")
        if not {"metric", "weight"} <= set(item) <= {
            "metric", "weight", "floor", "cap"
        }:
            raise ValueError("Optuna objective term is invalid")
        metric = item["metric"]
        weight = item["weight"]
        if (
            not isinstance(metric, str)
            or not metric
            or metric in metrics
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) == 0.0
        ):
            raise ValueError("Optuna objective term is invalid")
        floor = item.get("floor")
        cap = item.get("cap")
        for bound in (floor, cap):
            if bound is not None and (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
            ):
                raise ValueError("Optuna objective bound is invalid")
        if floor is not None and cap is not None and float(floor) > float(cap):
            raise ValueError("Optuna objective bounds are inverted")
        terms.append(ObjectiveTerm(
            metric=metric,
            weight=float(weight),
            floor=None if floor is None else float(floor),
            cap=None if cap is None else float(cap),
        ))
        metrics.add(metric)
    return tuple(terms)


def _path_is_frozen(path: str, frozen_paths: Sequence[str]) -> bool:
    return any(
        path == frozen or path.startswith(f"{frozen}.")
        for frozen in frozen_paths
    )


def _validate_search_assignment(
    *,
    path: str,
    base_config: Mapping,
    frozen_paths: Sequence[str],
    allowed_prefixes: Sequence[str],
    external_parent_economic_search_paths: Sequence[str],
) -> None:
    from .orchestration import (
        _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS,
        _EXTERNAL_PARENT_ECONOMIC_FIELDS,
        _EXTERNAL_PARENT_TRAINING_ONLY_RECIPE_PATHS,
    )

    _get_path(base_config, path)
    if _path_is_frozen(path, frozen_paths):
        raise ValueError(f"Optuna search assignment changes frozen path: {path}")
    if not any(path.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(f"Optuna search assignment is outside allowed paths: {path}")
    root, _, leaf = path.partition(".")
    changes_parent_contract = (
        root in _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS
        or (root == "challenge" and leaf in _EXTERNAL_PARENT_ECONOMIC_FIELDS)
    )
    if (
        changes_parent_contract
        and path not in _EXTERNAL_PARENT_TRAINING_ONLY_RECIPE_PATHS
        and path not in external_parent_economic_search_paths
    ):
        raise ValueError(
            f"Optuna search space changes external parent contract: {path}"
        )


def _search_space_contract(
    raw: object,
    *,
    base_config: Mapping,
    frozen_paths: Sequence[str],
    allowed_prefixes: Sequence[str],
    external_parent_economic_search_paths: Sequence[str],
) -> dict[str, dict]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Optuna search space must be nonempty")
    search_space: dict[str, dict] = {}
    assignment_paths: set[str] = set()
    for name, specification in raw.items():
        if not isinstance(name, str) or not name or not isinstance(specification, dict):
            raise ValueError(f"Optuna search dimension is invalid: {name}")
        kind = specification.get("type")
        if kind in {"float", "int"}:
            if not {"type", "low", "high"} <= set(specification) <= {
                "type", "low", "high", "log", "step"
            }:
                raise ValueError(f"Optuna search dimension is invalid: {name}")
            low = specification["low"]
            high = specification["high"]
            if (
                isinstance(low, bool)
                or isinstance(high, bool)
                or not isinstance(low, (int, float))
                or not isinstance(high, (int, float))
                or not math.isfinite(float(low))
                or not math.isfinite(float(high))
                or float(low) >= float(high)
            ):
                raise ValueError(f"Optuna bounds are invalid: {name}")
            if kind == "int" and (not isinstance(low, int) or not isinstance(high, int)):
                raise ValueError(f"Optuna integer bounds are invalid: {name}")
            if "step" in specification:
                step = specification["step"]
                if (
                    isinstance(step, bool)
                    or not isinstance(step, (int, float))
                    or float(step) <= 0.0
                    or not math.isfinite(float(step))
                ):
                    raise ValueError(f"Optuna step is invalid: {name}")
            if bool(specification.get("log", False)) and "step" in specification:
                raise ValueError(f"Optuna log dimension cannot declare a step: {name}")
            _validate_search_assignment(
                path=name,
                base_config=base_config,
                frozen_paths=frozen_paths,
                allowed_prefixes=allowed_prefixes,
                external_parent_economic_search_paths=(
                    external_parent_economic_search_paths
                ),
            )
            assignment_paths.add(name)
        elif kind == "categorical":
            if set(specification) != {"type", "choices"}:
                raise ValueError(f"Optuna search dimension is invalid: {name}")
            if not isinstance(specification["choices"], list) or not specification["choices"]:
                raise ValueError(f"Optuna choices are invalid: {name}")
            _validate_search_assignment(
                path=name,
                base_config=base_config,
                frozen_paths=frozen_paths,
                allowed_prefixes=allowed_prefixes,
                external_parent_economic_search_paths=(
                    external_parent_economic_search_paths
                ),
            )
            assignment_paths.add(name)
        elif kind == "categorical_mapping":
            if set(specification) != {"type", "choices"}:
                raise ValueError(f"Optuna search dimension is invalid: {name}")
            choices = specification["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Optuna choices are invalid: {name}")
            labels: set[str] = set()
            expected_paths: set[str] | None = None
            for choice in choices:
                if (
                    not isinstance(choice, dict)
                    or set(choice) != {"name", "values"}
                    or not isinstance(choice["name"], str)
                    or not choice["name"]
                    or choice["name"] in labels
                    or not isinstance(choice["values"], dict)
                    or not choice["values"]
                ):
                    raise ValueError(f"Optuna mapping choice is invalid: {name}")
                labels.add(choice["name"])
                paths = set(choice["values"])
                if expected_paths is None:
                    expected_paths = paths
                elif paths != expected_paths:
                    raise ValueError(
                        f"Optuna mapping choices assign different paths: {name}"
                    )
                for path in paths:
                    _validate_search_assignment(
                        path=path,
                        base_config=base_config,
                        frozen_paths=frozen_paths,
                        allowed_prefixes=allowed_prefixes,
                        external_parent_economic_search_paths=(
                            external_parent_economic_search_paths
                        ),
                    )
                    if path in assignment_paths:
                        raise ValueError(
                            f"Optuna path is assigned by multiple dimensions: {path}"
                        )
            assert expected_paths is not None
            assignment_paths.update(expected_paths)
        else:
            raise ValueError(f"Optuna search dimension is invalid: {name}")
        search_space[name] = deepcopy(specification)
    missing_economic_paths = (
        set(external_parent_economic_search_paths) - assignment_paths
    )
    if missing_economic_paths:
        raise ValueError(
            "Optuna external parent economic search path is not assigned: "
            f"{sorted(missing_economic_paths)[0]}"
        )
    return search_space


def _stage_contract(raw: object, *, base_config: Mapping, label: str) -> SweepStage:
    if (
        not isinstance(raw, dict)
        or set(raw) != {
            "name",
            "training_episodes",
            "validation_episodes",
            "start_pnls",
            "balance_validation_episodes",
            "short_circuit",
        }
        or not isinstance(raw["name"], str)
        or not raw["name"]
    ):
        raise ValueError(f"Optuna {label} stage is invalid")
    training_episodes = raw["training_episodes"]
    validation_episodes = raw["validation_episodes"]
    start_pnls = raw["start_pnls"]
    balance_validation_episodes = raw["balance_validation_episodes"]
    if (
        isinstance(training_episodes, bool)
        or not isinstance(training_episodes, int)
        or training_episodes < 1
        or isinstance(validation_episodes, bool)
        or not isinstance(validation_episodes, int)
        or validation_episodes < 1
        or not isinstance(start_pnls, list)
        or not start_pnls
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -float(base_config["challenge"]["max_loss"])
            < float(value) <= 0.0
            for value in start_pnls
        )
        or len({float(value) for value in start_pnls}) != len(start_pnls)
        or isinstance(balance_validation_episodes, bool)
        or not isinstance(balance_validation_episodes, int)
        or balance_validation_episodes < 0
    ):
        raise ValueError(f"Optuna {label} budget is invalid")
    stages = tuple(
        stage
        for stage in base_config["campaign"]["budget_stages"]
        if stage["name"] == raw["name"]
    )
    if len(stages) != 1:
        raise ValueError(f"Optuna {label} stage must resolve exactly once")
    short_circuit = raw["short_circuit"]
    if short_circuit is not None:
        if (
            not isinstance(short_circuit, dict)
            or not {
                "minimum_completed_episodes",
                "minimum_passes",
                "maximum_blow_rate",
            } <= set(short_circuit)
            or isinstance(short_circuit["minimum_completed_episodes"], bool)
            or not isinstance(short_circuit["minimum_completed_episodes"], int)
            or not 1
            <= short_circuit["minimum_completed_episodes"]
            <= training_episodes
            or isinstance(short_circuit["minimum_passes"], bool)
            or not isinstance(short_circuit["minimum_passes"], int)
            or short_circuit["minimum_passes"] < 0
            or isinstance(short_circuit["maximum_blow_rate"], bool)
            or not isinstance(short_circuit["maximum_blow_rate"], (int, float))
            or not 0.0 <= float(short_circuit["maximum_blow_rate"]) <= 1.0
        ):
            raise ValueError(f"Optuna {label} short circuit is invalid")
    return SweepStage(
        raw["name"],
        training_episodes,
        validation_episodes,
        tuple(float(value) for value in start_pnls),
        balance_validation_episodes,
        deepcopy(short_circuit),
    )


def load_optuna_sweep(path: str | Path) -> OptunaSweep:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if set(payload) != {
        "schema", "name", "base_config", "study_root", "study", "objective",
        "search_space", "stages", "promotion", "frozen", "artifacts",
    } or payload["schema"] != OPTUNA_SWEEP_SCHEMA:
        raise ValueError("Optuna sweep contract is invalid")
    base_path = (path.parent / str(payload["base_config"])).resolve()
    base_bytes = base_path.read_bytes()
    base_config = json.loads(base_bytes)

    study = payload["study"]
    required_study_fields = {
        "sampler", "seed", "n_trials", "n_jobs", "n_startup_trials",
    }
    optional_study_fields = {"multivariate", "group"}
    if (
        not isinstance(study, dict)
        or not required_study_fields <= set(study)
        or not set(study) <= required_study_fields | optional_study_fields
    ):
        raise ValueError("Optuna study contract is invalid")
    n_jobs = int(study["n_jobs"])
    if study["sampler"] != "tpe" or not 1 <= n_jobs <= 3:
        raise ValueError(
            "PropEvolve requires constrained TPE with one to three isolated jobs"
        )
    n_trials = int(study["n_trials"])
    n_startup = int(study["n_startup_trials"])
    multivariate = study.get("multivariate", False)
    group = study.get("group", False)
    if not isinstance(multivariate, bool) or not isinstance(group, bool):
        raise ValueError("Optuna TPE mode is invalid")
    if group and not multivariate:
        raise ValueError("Optuna TPE group requires multivariate sampling")
    if n_trials < 1 or not 1 <= n_startup <= n_trials:
        raise ValueError("Optuna trial budget is invalid")

    artifacts = payload["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or "screening_retention" not in artifacts
        or not set(artifacts) <= {"screening_retention", "maximum_retained_feasible_trials"}
        or artifacts["screening_retention"] not in {"compact", "keep"}
    ):
        raise ValueError("Optuna artifact retention contract is invalid")
    retained_limit = artifacts.get("maximum_retained_feasible_trials")
    if retained_limit is not None and (
        isinstance(retained_limit, bool) or not isinstance(retained_limit, int)
        or retained_limit < 1 or artifacts["screening_retention"] != "compact"
    ):
        raise ValueError("Optuna retained feasible trial limit is invalid")

    frozen = payload["frozen"]
    if (
        not isinstance(frozen, dict)
        or not {
            "teacher_free_selection", "paths", "allowed_search_prefixes"
        } <= set(frozen)
        or not set(frozen) <= {
            "teacher_free_selection", "paths", "allowed_search_prefixes",
            "external_parent_economic_search_paths",
        }
        or frozen["teacher_free_selection"] is not True
        or not isinstance(frozen["paths"], list)
        or not frozen["paths"]
        or not isinstance(frozen["allowed_search_prefixes"], list)
        or not frozen["allowed_search_prefixes"]
    ):
        raise ValueError("Optuna frozen contract is invalid")
    frozen_paths = tuple(str(item) for item in frozen["paths"])
    allowed_prefixes = tuple(str(item) for item in frozen["allowed_search_prefixes"])
    economic_search_raw = frozen.get(
        "external_parent_economic_search_paths", []
    )
    if (
        not isinstance(economic_search_raw, list)
        or any(not isinstance(item, str) for item in economic_search_raw)
    ):
        raise ValueError("Optuna external parent economic paths are invalid")
    economic_search_paths = tuple(str(item) for item in economic_search_raw)
    from .orchestration import _EXTERNAL_PARENT_ECONOMIC_FIELDS

    valid_economic_search_paths = {
        f"challenge.{field}" for field in _EXTERNAL_PARENT_ECONOMIC_FIELDS
    }
    if (
        len(set(economic_search_paths)) != len(economic_search_paths)
        or not set(economic_search_paths) <= valid_economic_search_paths
    ):
        raise ValueError("Optuna external parent economic paths are invalid")
    if len(set(frozen_paths)) != len(frozen_paths) or any(
        not item for item in (*frozen_paths, *allowed_prefixes)
    ):
        raise ValueError("Optuna frozen contract is invalid")
    for frozen_path in frozen_paths:
        _get_path(base_config, frozen_path)
    sealed = base_config.get("sealed_confirmation")
    if not isinstance(sealed, Mapping) or sealed.get("teacher_free") is not True:
        raise ValueError("Optuna selection base must remain teacher-free")

    objective = payload["objective"]
    if not isinstance(objective, dict) or set(objective) != {"terms", "constraints"}:
        raise ValueError("Optuna objective contract is invalid")
    objective_terms = _objective_contract(objective["terms"])
    constraints = _constraint_contract(objective["constraints"], label="screening")
    search_space = _search_space_contract(
        payload["search_space"],
        base_config=base_config,
        frozen_paths=frozen_paths,
        allowed_prefixes=allowed_prefixes,
        external_parent_economic_search_paths=economic_search_paths,
    )

    stages_raw = payload["stages"]
    if not isinstance(stages_raw, dict) or set(stages_raw) != {
        "screening", "confirmation", "multi_seed"
    }:
        raise ValueError("Optuna stages contract is invalid")
    stages = {
        label: _stage_contract(value, base_config=base_config, label=label)
        for label, value in stages_raw.items()
    }

    promotion_raw = payload["promotion"]
    if not isinstance(promotion_raw, dict) or set(promotion_raw) != {
        "enabled", "top_k", "finalist_top_k", "seeds", "seed_paths",
        "required_feasible_seeds", "acceptance",
    }:
        raise ValueError("Optuna promotion contract is invalid")
    enabled = promotion_raw["enabled"]
    top_k = promotion_raw["top_k"]
    finalist_top_k = promotion_raw["finalist_top_k"]
    seeds = promotion_raw["seeds"]
    seed_paths = promotion_raw["seed_paths"]
    required_feasible_seeds = promotion_raw["required_feasible_seeds"]
    if (
        not isinstance(enabled, bool)
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= n_trials
        or isinstance(finalist_top_k, bool)
        or not isinstance(finalist_top_k, int)
        or not 1 <= finalist_top_k <= top_k
        or not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or not isinstance(seed_paths, list)
        or not seed_paths
        or isinstance(required_feasible_seeds, bool)
        or not isinstance(required_feasible_seeds, int)
        or not 1 <= required_feasible_seeds <= len(seeds)
    ):
        raise ValueError("Optuna promotion contract is invalid")
    seed_paths_tuple = tuple(str(item) for item in seed_paths)
    for seed_path in seed_paths_tuple:
        current = _get_path(base_config, seed_path)
        if isinstance(current, bool) or not isinstance(current, int):
            raise ValueError(f"Optuna seed path is not integer-valued: {seed_path}")
        if _path_is_frozen(seed_path, frozen_paths):
            raise ValueError(f"Optuna seed path is frozen: {seed_path}")
    promotion = PromotionContract(
        enabled=enabled,
        top_k=top_k,
        finalist_top_k=finalist_top_k,
        seeds=tuple(seeds),
        seed_paths=seed_paths_tuple,
        required_feasible_seeds=required_feasible_seeds,
        acceptance=_constraint_contract(
            promotion_raw["acceptance"], label="promotion"
        ),
    )
    sweep = OptunaSweep(
        path=path,
        name=str(payload["name"]),
        base_config_path=base_path,
        base_config=base_config,
        base_config_sha256=_sha256(base_bytes),
        study_root=str(payload["study_root"]),
        sampler="tpe",
        seed=int(study["seed"]),
        n_trials=n_trials,
        n_jobs=n_jobs,
        n_startup_trials=n_startup,
        multivariate=multivariate,
        group=group,
        screening_artifact_retention=str(artifacts["screening_retention"]),
        maximum_retained_feasible_trials=retained_limit,
        objective_terms=objective_terms,
        constraints=constraints,
        search_space=search_space,
        stages=stages,
        promotion=promotion,
        frozen_paths=frozen_paths,
        allowed_search_prefixes=allowed_prefixes,
        external_parent_economic_search_paths=economic_search_paths,
    )
    _baseline_parameters(sweep)
    return sweep


def _constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    values = trial.user_attrs.get("constraint")
    if not isinstance(values, (list, tuple)):
        return (1.0,)
    return tuple(float(value) for value in values)


def _mapping_choice(specification: Mapping, label: object) -> Mapping[str, object]:
    choices = tuple(
        choice for choice in specification["choices"] if choice["name"] == label
    )
    if len(choices) != 1:
        raise ValueError(f"Optuna categorical mapping choice is invalid: {label}")
    return choices[0]["values"]


def _sample(trial: optuna.Trial, sweep: OptunaSweep) -> dict[str, object]:
    sampled: dict[str, object] = {}
    for name, specification in sweep.search_space.items():
        kind = specification["type"]
        if kind == "categorical":
            value = trial.suggest_categorical(name, specification["choices"])
        elif kind == "categorical_mapping":
            value = trial.suggest_categorical(
                name, [choice["name"] for choice in specification["choices"]]
            )
        elif kind == "int":
            value = trial.suggest_int(
                name,
                int(specification["low"]),
                int(specification["high"]),
                step=int(specification.get("step", 1)),
                log=bool(specification.get("log", False)),
            )
        else:
            kwargs = {"log": bool(specification.get("log", False))}
            if "step" in specification:
                kwargs["step"] = float(specification["step"])
            value = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                **kwargs,
            )
        sampled[name] = value
    return sampled


def _apply_parameters(config: dict, sweep: OptunaSweep, parameters: Mapping) -> None:
    for name, value in parameters.items():
        specification = sweep.search_space[name]
        if specification["type"] == "categorical_mapping":
            for path, mapped_value in _mapping_choice(specification, value).items():
                _set_path(config, path, mapped_value)
        else:
            _set_path(config, name, value)


def _baseline_parameters(sweep: OptunaSweep) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for name, specification in sweep.search_space.items():
        kind = specification["type"]
        if kind != "categorical_mapping":
            value = _get_path(sweep.base_config, name)
            if kind in {"float", "int"} and not (
                float(specification["low"])
                <= float(value)
                <= float(specification["high"])
            ):
                raise ValueError(
                    f"Optuna dimension excludes exact baseline control: {name}"
                )
            if kind == "int" and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(
                    f"Optuna integer dimension has non-integer baseline: {name}"
                )
            if kind == "categorical" and value not in specification["choices"]:
                raise ValueError(
                    f"Optuna categorical dimension excludes baseline: {name}"
                )
            parameters[name] = value
            continue
        matches = [
            choice["name"]
            for choice in specification["choices"]
            if all(
                _get_path(sweep.base_config, path) == expected
                for path, expected in choice["values"].items()
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Optuna mapping dimension lacks one exact baseline choice: {name}"
            )
        parameters[name] = matches[0]
    return parameters


def _workspace_root(sweep: OptunaSweep) -> Path:
    configured = Path(str(sweep.base_config["workspace_root"])).expanduser()
    if not configured.is_absolute():
        configured = sweep.base_config_path.parent / configured
    return configured.resolve(strict=True)


def _compile_trial_config(
    sweep: OptunaSweep,
    *,
    parameters: Mapping[str, object],
    stage: SweepStage,
    run_root: Path,
    seed: int | None = None,
) -> dict:
    config = deepcopy(sweep.base_config)
    frozen_before = {
        path: deepcopy(_get_path(config, path)) for path in sweep.frozen_paths
    }
    _apply_parameters(config, sweep, parameters)
    base_economic_overrides = set(
        sweep.base_config["evolution"].get(
            "external_parent_economic_overrides", ()
        )
    )
    economic_overrides = set(
        config["evolution"].get("external_parent_economic_overrides", ())
    )
    for path in sweep.external_parent_economic_search_paths:
        _, _, field = path.partition(".")
        if _get_path(config, path) != _get_path(sweep.base_config, path):
            economic_overrides.add(field)
        elif field not in base_economic_overrides:
            economic_overrides.discard(field)
    if economic_overrides:
        config["evolution"]["external_parent_economic_overrides"] = sorted(
            economic_overrides
        )
    else:
        config["evolution"].pop("external_parent_economic_overrides", None)
    # Generated configs may live under an artifact directory. Preserve the
    # base recipe's repository identity instead of reinterpreting its relative
    # workspace root from the generated config's parent directory.
    config["workspace_root"] = str(_workspace_root(sweep))
    matching_stages = [
        item
        for item in config["campaign"]["budget_stages"]
        if item["name"] == stage.name
    ]
    if len(matching_stages) != 1:
        raise ValueError("Optuna stage must resolve exactly once")
    selected_stage = deepcopy(matching_stages[0])
    selected_stage["training_episodes"] = stage.training_episodes
    selected_stage["validation_episodes"] = stage.validation_episodes
    config["balance_curriculum"]["start_pnls"] = list(stage.start_pnls)
    config["balance_curriculum"]["validation_episodes"] = (
        stage.balance_validation_episodes
    )
    if stage.balance_validation_episodes == 0:
        selected_stage["selection_requirements"] = [
            requirement
            for requirement in selected_stage["selection_requirements"]
            if not str(requirement["metric"]).startswith("balance_stress.")
        ]
        config["campaign"]["selection_requirements"] = [
            requirement
            for requirement in config["campaign"]["selection_requirements"]
            if not str(requirement["metric"]).startswith("balance_stress.")
        ]
    config["training"]["short_circuit"] = deepcopy(stage.short_circuit)
    if stage.short_circuit is None:
        selected_stage.pop("short_circuit_minimum_episodes", None)
    else:
        selected_stage["short_circuit_minimum_episodes"] = int(
            stage.short_circuit["minimum_completed_episodes"]
        )
    config["output"] = str(run_root)
    config["campaign"]["state_root"] = str(run_root / "ml-loop-state")
    config["campaign"]["budget_stages"] = [selected_stage]
    config["campaign"]["max_revisions_per_stage"] = 0
    config["evolution"]["allowed_revision_paths"] = []
    config["evolution"]["revision_bounds"] = {}
    config["evolution"]["hypothesis"] = (
        "JSON-driven constrained Optuna trial over declared economic, safety, "
        "and replay learning parameters with frozen causal mechanics."
    )
    if seed is not None:
        for path in sweep.promotion.seed_paths:
            _set_path(config, path, seed)
    for path, expected in frozen_before.items():
        if _get_path(config, path) != expected:
            raise ValueError(f"Optuna compilation changed frozen path: {path}")
    return config


def _constraint_value(actual: float, operator: str, expected: float) -> float:
    if operator == "==":
        return abs(actual - expected)
    if operator == ">=":
        return expected - actual
    if operator == ">":
        return expected - actual + 1e-12
    if operator == "<=":
        return actual - expected
    raise ValueError(f"unsupported constraint operator: {operator}")


def _objective_value(
    metrics: Mapping[str, float], terms: Sequence[ObjectiveTerm]
) -> float:
    total = 0.0
    for term in terms:
        value = float(metrics[term.metric])
        if term.floor is not None:
            value = max(value, term.floor)
        if term.cap is not None:
            value = min(value, term.cap)
        total += term.weight * value
    return total


def _print_trial_result(
    *,
    trial_number: int,
    state: str,
    feasible: bool,
    objective: float | None,
    metrics: Mapping[str, float],
) -> None:
    objective_text = "null" if objective is None else f"{objective:.1f}"
    pass_rate = metrics.get("selection.pass_rate")
    blow_rate = metrics.get("selection.blow_rate")
    pass_rate_text = "n/a" if pass_rate is None else f"{100.0 * pass_rate:g}%"
    blow_rate_text = "n/a" if blow_rate is None else f"{100.0 * blow_rate:g}%"
    print(
        f"[optuna-result] trial={trial_number} state={state} "
        f"feasible={str(feasible).lower()} objective={objective_text} "
        f"pass_rate={pass_rate_text} blow_rate={blow_rate_text}",
        flush=True,
    )


def _required_metrics(sweep: OptunaSweep) -> frozenset[str]:
    return frozenset(
        [term.metric for term in sweep.objective_terms]
        + list(sweep.constraints)
        + list(sweep.promotion.acceptance)
    )


def _constraint_vector(
    metrics: Mapping[str, float],
    constraints: Mapping[str, tuple[str, float]],
) -> list[float]:
    missing = sorted(set(constraints) - set(metrics))
    if missing:
        raise ValueError(f"Optuna trial is missing constraint metrics: {missing}")
    return [
        _constraint_value(float(metrics[metric]), operator, expected)
        for metric, (operator, expected) in constraints.items()
    ]


def _is_feasible(
    metrics: Mapping[str, float],
    constraints: Mapping[str, tuple[str, float]],
) -> bool:
    return all(value <= 0.0 for value in _constraint_vector(metrics, constraints))


def _prune_training_short_circuit(
    trial: optuna.Trial,
    metrics: Mapping[str, float],
    *,
    constraint_count: int,
) -> None:
    if float(metrics.get("training.short_circuited", 0.0)) != 1.0:
        return
    trial.set_user_attr("constraint", [1.0] * constraint_count)
    raise optuna.TrialPruned("training short circuit rejected trial")


def _best_feasible(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    eligible = [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and all(value <= 0.0 for value in _constraints_func(trial))
    ]
    return max(eligible, key=lambda trial: float(trial.value)) if eligible else None


def _top_feasible(
    study: optuna.Study, count: int, *,
    trials: Sequence[optuna.trial.FrozenTrial] | None = None,
) -> tuple[optuna.trial.FrozenTrial, ...]:
    eligible = [
        trial
        for trial in (study.trials if trials is None else trials)
        if trial.state is optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and all(value <= 0.0 for value in _constraints_func(trial))
    ]
    return tuple(
        sorted(eligible, key=lambda trial: float(trial.value), reverse=True)[:count]
    )


def _reconcile_interrupted_trials(study: optuna.Study) -> frozenset[int]:
    reconciled = set(int(value) for value in study.user_attrs.get(
        "reconciled_interrupted_trials", ()
    ))
    retryable = tuple(
        trial for trial in study.trials
        if trial.number not in reconciled
        and (
            trial.state is optuna.trial.TrialState.RUNNING
            or trial.state is optuna.trial.TrialState.FAIL
            and trial.user_attrs.get("stopped_study_on_executor_error") is True
        )
    )
    for trial in retryable:
        if trial.state is optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        reconciled.add(trial.number)
        if trial.params:
            retry_attrs = dict(trial.user_attrs)
            retry_attrs["retry_of_trial"] = trial.number
            retry_attrs.setdefault("artifact_trial_number", trial.number)
            study.enqueue_trial(
                trial.params,
                user_attrs=retry_attrs,
            )
    study.set_user_attr("reconciled_interrupted_trials", sorted(reconciled))
    return frozenset(reconciled)


def _compact_screening_trial(
    *,
    artifacts: Path,
    trial: optuna.trial.FrozenTrial,
) -> None:
    raw_root = trial.user_attrs.get("trial_artifact_root")
    if not isinstance(raw_root, str) or not raw_root:
        return
    trial_root = Path(raw_root)
    screening_root = (artifacts / "screening").resolve()
    try:
        resolved = trial_root.resolve(strict=True)
    except FileNotFoundError:
        return
    artifact_number = int(
        trial.user_attrs.get("artifact_trial_number", trial.number)
    )
    if (
        resolved.parent != screening_root
        or resolved.name != f"trial-{artifact_number:03d}"
    ):
        raise ValueError("Optuna cleanup target escaped the screening root")

    evidence_root = artifacts / "screening-evidence" / f"trial-{trial.number:03d}"
    preserved: list[str] = []
    removed_bytes = 0
    removed_files = 0
    for path in resolved.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            removed_bytes += path.stat().st_size
            removed_files += 1
            if path.suffix in {".json", ".jsonl"}:
                relative = path.relative_to(resolved)
                destination = evidence_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                preserved.append(str(relative))
    shutil.rmtree(resolved)
    receipt = {
        "schema": "propevolve_optuna_trial_cleanup_v1",
        "trial_number": trial.number,
        "trial_state": trial.state.name,
        "source_root": str(resolved),
        "removed_bytes": removed_bytes,
        "removed_files": removed_files,
        "preserved_evidence": sorted(preserved),
    }
    _write_exact(
        artifacts / "cleanup" / f"trial-{trial.number:03d}.json",
        receipt,
    )


def _trial_is_rejected(trial: optuna.trial.FrozenTrial) -> bool:
    if trial.state in {
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }:
        return True
    return (
        trial.state is optuna.trial.TrialState.COMPLETE
        and any(value > 0.0 for value in _constraints_func(trial))
    )


def _clean_code_commit(repository_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "-C", str(repository_root), "status", "--porcelain"], text=True
    )
    if status:
        raise ValueError("Optuna study requires a clean PropEvolve checkout")
    return subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _run_compiled_trial(
    *,
    config: dict,
    config_path: Path,
    run_id: str,
    stage_name: str,
    artifact_root: Path,
    runner,
    state_loader,
    config_validator,
) -> tuple[str, dict[str, float]]:
    _write_exact(config_path, config)
    if config_validator is None:
        from .config import load_experiment_config

        validated_config = load_experiment_config(config_path)
    else:
        validation_result = config_validator(config_path)
        validated_config = (
            validation_result
            if isinstance(validation_result, Mapping)
            else config
        )
    result_path = Path(str(validated_config["output"])) / "optuna-trial-result.json"
    existing = _load_trial_result(config_path, result_path)
    if existing is not None:
        result = existing
    elif runner is None:
        result = _default_trial_runner(
            config_path,
            run_id=run_id,
            result_path=result_path,
            stdout_path=artifact_root / "logs" / f"{run_id}.stdout.log",
            stderr_path=artifact_root / "logs" / f"{run_id}.stderr.log",
        )
    else:
        result = runner(config_path, run_id=run_id)
    return _trial_metrics(result, stage_name)


def _run_promotion(
    *,
    sweep: OptunaSweep,
    study: optuna.Study,
    artifacts: Path,
    configs: Path,
    runner,
    state_loader,
    config_validator,
) -> dict[str, object]:
    promoted = _top_feasible(study, sweep.promotion.top_k)
    if not promoted:
        return {
            "promotion_status": "FAILED_GATE",
            "promoted_trial_numbers": [],
            "confirmation": [],
            "winner_trial_number": None,
            "winner_feasible_seeds": 0,
            "multi_seed": [],
        }

    def optimize_fixed_study(
        fixed_study: optuna.Study,
        *,
        target_trials: int,
        objective,
    ) -> tuple[optuna.trial.FrozenTrial, ...]:
        reconciled = _reconcile_interrupted_trials(fixed_study)
        terminal = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
        completed = sum(
            trial.state in terminal and trial.number not in reconciled
            for trial in fixed_study.trials
        )
        remaining = max(0, target_trials - completed)
        if remaining:
            fixed_study.optimize(
                objective,
                n_trials=remaining,
                n_jobs=min(sweep.n_jobs, remaining),
                show_progress_bar=False,
            )
        return tuple(
            trial for trial in fixed_study.trials
            if trial.number not in reconciled and trial.state in terminal
        )

    confirmation_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=sweep.seed,
            constraints_func=_constraints_func,
            multivariate=sweep.multivariate,
            group=sweep.group,
        ),
        study_name=f"{sweep.name}-confirmation",
        storage=_study_storage(
            artifacts / "confirmation.study.journal.log"
        ),
        load_if_exists=True,
    )
    promoted_numbers = [trial.number for trial in promoted]
    expected_sources = confirmation_study.user_attrs.get(
        "screening_trial_numbers"
    )
    if expected_sources is None:
        confirmation_study.set_user_attr(
            "screening_trial_numbers", promoted_numbers
        )
        for source in promoted:
            confirmation_study.enqueue_trial(
                source.params,
                user_attrs={"screening_trial_number": source.number},
            )
    elif list(expected_sources) != promoted_numbers:
        raise ValueError("Optuna confirmation candidates drifted")

    def confirmation_objective(trial: optuna.Trial) -> float:
        parameters = _sample(trial, sweep)
        source_number = int(trial.user_attrs["screening_trial_number"])
        run_root = artifacts / "confirmation" / f"trial-{source_number:03d}"
        config = _compile_trial_config(
            sweep,
            parameters=parameters,
            stage=sweep.stages["confirmation"],
            run_root=run_root,
        )
        run_id = f"{sweep.name}-confirm-trial-{source_number:03d}"
        try:
            _, metrics = _run_compiled_trial(
                config=config,
                config_path=configs / f".optuna-{run_id}.json",
                run_id=run_id,
                stage_name=sweep.stages["confirmation"].name,
                artifact_root=artifacts,
                runner=runner,
                state_loader=state_loader,
                config_validator=config_validator,
            )
        except Exception:
            trial.set_user_attr("stopped_study_on_executor_error", True)
            trial.study.stop()
            raise
        for metric, value in metrics.items():
            trial.set_user_attr(metric, float(value))
        _prune_training_short_circuit(
            trial,
            metrics,
            constraint_count=len(sweep.promotion.acceptance),
        )
        constraints = _constraint_vector(
            metrics, sweep.promotion.acceptance
        )
        trial.set_user_attr("constraint", constraints)
        return _objective_value(metrics, sweep.objective_terms)

    confirmation_trials = optimize_fixed_study(
        confirmation_study,
        target_trials=len(promoted),
        objective=confirmation_objective,
    )
    confirmations = []
    for trial in confirmation_trials:
        if trial.state is not optuna.trial.TrialState.COMPLETE:
            continue
        metrics = {
            metric: float(trial.user_attrs[metric])
            for metric in _required_metrics(sweep)
            if metric in trial.user_attrs
        }
        confirmations.append({
            "trial_number": int(trial.user_attrs["screening_trial_number"]),
            "objective": float(trial.value),
            "feasible": all(
                value <= 0.0 for value in _constraints_func(trial)
            ),
            "metrics": metrics,
            "parameters": dict(trial.params),
        })
    finalists = tuple(sorted(
        (item for item in confirmations if item["feasible"]),
        key=lambda item: float(item["objective"]),
        reverse=True,
    )[:sweep.promotion.finalist_top_k])

    multiseed_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=sweep.seed + 1,
            constraints_func=_constraints_func,
            multivariate=sweep.multivariate,
            group=sweep.group,
        ),
        study_name=f"{sweep.name}-multi-seed",
        storage=_study_storage(
            artifacts / "multi-seed.study.journal.log"
        ),
        load_if_exists=True,
    )
    desired_runs = [
        (int(finalist["trial_number"]), int(seed), finalist["parameters"])
        for finalist in finalists
        for seed in sweep.promotion.seeds
    ]
    expected_runs = [[source, seed] for source, seed, _ in desired_runs]
    existing_runs = multiseed_study.user_attrs.get("requested_runs")
    if existing_runs is None:
        multiseed_study.set_user_attr("requested_runs", expected_runs)
        for source, seed, parameters in desired_runs:
            multiseed_study.enqueue_trial(
                parameters,
                user_attrs={
                    "screening_trial_number": source,
                    "seed": seed,
                },
            )
    elif list(existing_runs) != expected_runs:
        raise ValueError("Optuna multi-seed candidates drifted")

    def multiseed_objective(trial: optuna.Trial) -> float:
        parameters = _sample(trial, sweep)
        source_number = int(trial.user_attrs["screening_trial_number"])
        seed = int(trial.user_attrs["seed"])
        run_root = (
            artifacts / "multi-seed" / f"trial-{source_number:03d}"
            / f"seed-{seed}"
        )
        config = _compile_trial_config(
            sweep,
            parameters=parameters,
            stage=sweep.stages["multi_seed"],
            run_root=run_root,
            seed=seed,
        )
        run_id = f"{sweep.name}-multiseed-trial-{source_number:03d}-seed-{seed}"
        try:
            _, metrics = _run_compiled_trial(
                config=config,
                config_path=configs / f".optuna-{run_id}.json",
                run_id=run_id,
                stage_name=sweep.stages["multi_seed"].name,
                artifact_root=artifacts,
                runner=runner,
                state_loader=state_loader,
                config_validator=config_validator,
            )
        except Exception:
            trial.set_user_attr("stopped_study_on_executor_error", True)
            trial.study.stop()
            raise
        for metric, value in metrics.items():
            trial.set_user_attr(metric, float(value))
        _prune_training_short_circuit(
            trial,
            metrics,
            constraint_count=len(sweep.promotion.acceptance),
        )
        constraints = _constraint_vector(
            metrics, sweep.promotion.acceptance
        )
        trial.set_user_attr("constraint", constraints)
        if float(metrics.get("selection.blow_rate", 0.0)) > 0.0:
            trial.set_user_attr("stopped_study_on_blow", True)
            trial.study.stop()
        return _objective_value(metrics, sweep.objective_terms)

    multiseed_trials = optimize_fixed_study(
        multiseed_study,
        target_trials=len(desired_runs),
        objective=multiseed_objective,
    )
    multi_seed: list[dict[str, object]] = []
    winner: dict[str, object] | None = None
    for finalist in finalists:
        trial_number = int(finalist["trial_number"])
        seed_results: list[dict[str, object]] = []
        short_circuit_reason: str | None = None
        for trial in multiseed_trials:
            if (
                trial.state is not optuna.trial.TrialState.COMPLETE
                or int(trial.user_attrs["screening_trial_number"])
                != trial_number
            ):
                continue
            metrics = {
                metric: float(trial.user_attrs[metric])
                for metric in _required_metrics(sweep)
                if metric in trial.user_attrs
            }
            seed_results.append({
                "seed": int(trial.user_attrs["seed"]),
                "objective": float(trial.value),
                "feasible": all(
                    value <= 0.0 for value in _constraints_func(trial)
                ),
                "metrics": metrics,
            })
            if float(metrics.get("selection.blow_rate", 0.0)) > 0.0:
                short_circuit_reason = "zero_blow_gate"
        feasible_count = sum(bool(item["feasible"]) for item in seed_results)
        mean_objective = (
            sum(float(item["objective"]) for item in seed_results)
            / len(seed_results)
            if seed_results else float("-inf")
        )
        candidate = {
            "trial_number": trial_number,
            "parameters": finalist["parameters"],
            "feasible_seeds": feasible_count,
            "required_feasible_seeds": sweep.promotion.required_feasible_seeds,
            "mean_objective": mean_objective,
            "short_circuit_reason": short_circuit_reason,
            "evaluated_seeds": len(seed_results),
            "requested_seeds": len(sweep.promotion.seeds),
            "seeds": seed_results,
        }
        multi_seed.append(candidate)
        if feasible_count >= sweep.promotion.required_feasible_seeds and (
            winner is None or mean_objective > float(winner["mean_objective"])
        ):
            winner = candidate
    return {
        "promotion_status": "COMPLETE" if winner is not None else "FAILED_GATE",
        "promoted_trial_numbers": promoted_numbers,
        "confirmation": confirmations,
        "winner_trial_number": None if winner is None else int(winner["trial_number"]),
        "winner_feasible_seeds": 0 if winner is None else int(winner["feasible_seeds"]),
        "multi_seed": multi_seed,
    }


def run_optuna_sweep(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    config_root: str | Path | None = None,
    n_trials: int | None = None,
    runner=None,
    state_loader=None,
    config_validator=None,
    code_commit: str | None = None,
) -> OptunaSweepResult:
    """Run/resume screening with bounded isolated trial subprocesses."""
    sweep = load_optuna_sweep(path)
    repository_root = _workspace_root(sweep)
    artifacts = Path(artifact_root) if artifact_root is not None else (
        repository_root / sweep.study_root
    )
    configs = Path(config_root) if config_root is not None else (
        artifacts / "configs"
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    target_trials = sweep.n_trials if n_trials is None else int(n_trials)
    if target_trials < 1:
        raise ValueError("Optuna target trial count must be positive")
    storage_path = artifacts / "study.journal.log"
    sweep_sha256 = _sha256(sweep.path.read_bytes())
    active_code_commit = code_commit or _clean_code_commit(repository_root)
    sampler = optuna.samplers.TPESampler(
        seed=sweep.seed,
        n_startup_trials=min(sweep.n_startup_trials, target_trials),
        constraints_func=_constraints_func,
        multivariate=sweep.multivariate,
        group=sweep.group,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=sweep.name,
        storage=_study_storage(storage_path),
        load_if_exists=True,
    )
    authority = {
        "base_config_sha256": sweep.base_config_sha256,
        "code_commit": active_code_commit,
        "sweep_config_sha256": sweep_sha256,
        "target_trials": target_trials,
    }
    existing_authority = {name: study.user_attrs.get(name) for name in authority}
    if any(value is not None for value in existing_authority.values()):
        if existing_authority != authority:
            raise ValueError("Optuna study authority drifted")
    else:
        for name, value in authority.items():
            study.set_user_attr(name, value)
    reconciled = _reconcile_interrupted_trials(study)
    if not study.trials:
        study.enqueue_trial(
            _baseline_parameters(sweep), user_attrs={"baseline_control": True}
        )
    def objective(trial: optuna.Trial) -> float:
        artifact_trial = int(
            trial.user_attrs.get("artifact_trial_number", trial.number)
        )
        parameters = _sample(trial, sweep)
        trial_root = artifacts / "screening" / f"trial-{artifact_trial:03d}"
        config = _compile_trial_config(
            sweep,
            parameters=parameters,
            stage=sweep.stages["screening"],
            run_root=trial_root,
        )
        config_path = configs / (
            f".optuna-{sweep.name}-screen-trial-{artifact_trial:03d}.json"
        )
        run_id = f"{sweep.name}-screen-trial-{artifact_trial:03d}"
        print(
            f"[optuna] trial={trial.number}/{target_trials - 1} START "
            f"artifact_trial={artifact_trial} "
            f"params={json.dumps(parameters, sort_keys=True)}",
            flush=True,
        )
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("trial_artifact_root", str(trial_root))
        trial.set_user_attr("artifact_trial_number", artifact_trial)
        try:
            trial_status, metrics = _run_compiled_trial(
                config=config,
                config_path=config_path,
                run_id=run_id,
                stage_name=sweep.stages["screening"].name,
                artifact_root=artifacts,
                runner=runner,
                state_loader=state_loader,
                config_validator=config_validator,
            )
        except Exception:
            trial.set_user_attr("stopped_study_on_executor_error", True)
            trial.study.stop()
            raise
        trial.set_user_attr("evaluation_status", trial_status)
        for metric, value in metrics.items():
            trial.set_user_attr(metric, float(value))
        if float(metrics.get("training.short_circuited", 0.0)) == 1.0:
            _print_trial_result(
                trial_number=trial.number,
                state="PRUNED",
                feasible=False,
                objective=None,
                metrics=metrics,
            )
            _prune_training_short_circuit(
                trial,
                metrics,
                constraint_count=len(sweep.constraints),
            )
        if float(metrics.get("selection.short_circuited", 0.0)) == 1.0:
            trial.set_user_attr("constraint", [1.0] * len(sweep.constraints))
            _print_trial_result(
                trial_number=trial.number,
                state="COMPLETE",
                feasible=False,
                objective=-1_000_000.0,
                metrics=metrics,
            )
            return -1_000_000.0
        missing = sorted(_required_metrics(sweep) - set(metrics))
        if missing:
            raise ValueError(f"Optuna trial is missing objective metrics: {missing}")
        constraints = _constraint_vector(metrics, sweep.constraints)
        trial.set_user_attr("constraint", constraints)
        value = _objective_value(metrics, sweep.objective_terms)
        _print_trial_result(
            trial_number=trial.number,
            state="COMPLETE",
            feasible=all(item <= 0.0 for item in constraints),
            objective=value,
            metrics=metrics,
        )
        return value

    def compact_recorded_trial(trial: optuna.trial.FrozenTrial) -> None:
        if (
            sweep.screening_artifact_retention == "compact"
            and _trial_is_rejected(trial)
        ):
            _compact_screening_trial(artifacts=artifacts, trial=trial)

    cleanup_lock = threading.Lock()

    def compact_terminal_trial(
        _study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        # Optuna invokes callbacks from concurrent objective threads. Serialize
        # artifact cleanup, never training, and retain the current best family.
        with cleanup_lock:
            compact_recorded_trial(trial)
            limit = sweep.maximum_retained_feasible_trials
            if sweep.screening_artifact_retention == "compact" and limit is not None:
                # Training can finish while cleanup holds its own lock. Rank
                # and delete from one snapshot; a newly completed winner must
                # never be deleted using a keep set calculated before it existed.
                recorded = _study.get_trials(deepcopy=True)
                keep = {item.number for item in _top_feasible(
                    _study, limit, trials=recorded,
                )}
                for completed in recorded:
                    if (completed.state is optuna.trial.TrialState.COMPLETE
                            and completed.number not in keep):
                        _compact_screening_trial(artifacts=artifacts, trial=completed)

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    effective_terminal = tuple(
        trial for trial in study.trials
        if trial.state in terminal_states and trial.number not in reconciled
    )
    remaining = max(0, target_trials - len(effective_terminal))
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=sweep.n_jobs,
            callbacks=[compact_terminal_trial],
        )
    best = _best_feasible(study)
    effective_trials = [trial for trial in study.trials if trial.number not in reconciled]
    states = [trial.state for trial in effective_trials]
    terminal_count = sum(state in terminal_states for state in states)
    screening_status = (
        "COMPLETE" if terminal_count >= target_trials and best is not None
        else "FAILED_GATE" if terminal_count >= target_trials
        else "INCOMPLETE"
    )
    promotion_payload: dict[str, object] = {
        "promotion_status": "DISABLED",
        "promoted_trial_numbers": [],
        "confirmation": [],
        "winner_trial_number": None,
        "winner_feasible_seeds": 0,
        "multi_seed": [],
    }
    if screening_status == "COMPLETE" and sweep.promotion.enabled and n_trials is None:
        promotion_payload = _run_promotion(
            sweep=sweep,
            study=study,
            artifacts=artifacts,
            configs=configs,
            runner=runner,
            state_loader=state_loader,
            config_validator=config_validator,
        )
    status = screening_status
    if sweep.promotion.enabled and n_trials is None:
        status = str(promotion_payload["promotion_status"])
    result_payload = {
        "schema": RESULT_SCHEMA,
        "study": sweep.name,
        "status": status,
        "screening_status": screening_status,
        "base_config_sha256": sweep.base_config_sha256,
        "sweep_config_sha256": sweep_sha256,
        "target_trials": target_trials,
        "completed_trials": sum(
            state is optuna.trial.TrialState.COMPLETE for state in states
        ),
        "pruned_trials": sum(
            state is optuna.trial.TrialState.PRUNED for state in states
        ),
        "failed_trials": sum(state is optuna.trial.TrialState.FAIL for state in states),
        "interrupted_trials": len(reconciled),
        "best_trial_number": None if best is None else best.number,
        "best_value": None if best is None else best.value,
        "best_parameters": None if best is None else best.params,
        "best_metrics": None if best is None else {
            metric: best.user_attrs.get(metric)
            for metric in sorted(_required_metrics(sweep))
        },
        **promotion_payload,
    }
    result_path = artifacts / "study.result.json"
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    return OptunaSweepResult(
        name=sweep.name,
        status=status,
        best_trial_number=None if best is None else best.number,
        completed_trials=int(result_payload["completed_trials"]),
        pruned_trials=int(result_payload["pruned_trials"]),
        failed_trials=int(result_payload["failed_trials"]),
        interrupted_trials=int(result_payload["interrupted_trials"]),
        study_path=storage_path,
        result_path=result_path,
        promoted_trial_numbers=tuple(promotion_payload["promoted_trial_numbers"]),
        winner_trial_number=promotion_payload["winner_trial_number"],
    )


__all__ = [
    "OPTUNA_SWEEP_SCHEMA",
    "ObjectiveTerm",
    "OptunaSweep",
    "OptunaSweepResult",
    "PromotionContract",
    "SweepStage",
    "load_optuna_sweep",
    "run_optuna_sweep",
]
