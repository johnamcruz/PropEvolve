"""JSON-driven constrained Optuna selection over PropEvolve campaigns."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

import optuna


OPTUNA_SWEEP_SCHEMA = "propevolve_optuna_sweep_v2"
RESULT_SCHEMA = "propevolve_optuna_sweep_result_v2"
_CONSTRAINT_OPERATORS = frozenset({"==", ">=", ">", "<="})
_TERMINAL_PHASES = frozenset({"COMPLETE", "FAILED_GATE", "STOPPED", "BLOCKED"})


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


def _default_state_loader(config_path: Path, run_id: str):
    from ml_training_loop.stores import JsonRunStore

    from .config import load_experiment_config

    config = load_experiment_config(config_path)
    state_root = Path(config["_root"]) / str(config["campaign"]["state_root"])
    return JsonRunStore(state_root).load(run_id)


def _default_runner(
    config_path: Path,
    *,
    run_id: str,
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
                "evolve",
                "--config",
                str(config_path),
                "--run-id",
                run_id,
            ],
            stdout=stdout,
            stderr=stderr,
            env=worker_environment,
            check=False,
        )
    state = _default_state_loader(config_path, run_id)
    if state is None:
        raise RuntimeError(
            f"Optuna campaign exited {process.returncode} without campaign state"
        )
    if process.returncode not in {0, 2}:
        raise RuntimeError(
            f"Optuna campaign {run_id} exited unexpectedly with "
            f"{process.returncode}"
        )
    return state


def _cell_metrics(state, stage_name: str) -> dict[str, float]:
    receipts = tuple(
        receipt for receipt in state.receipts if receipt.stage == stage_name
    )
    if len(receipts) != 1:
        raise ValueError(
            f"terminal Optuna campaign must have one {stage_name!r} receipt"
        )
    raw = receipts[0].outputs.get("metrics")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("terminal Optuna campaign metrics are missing")
    metrics = {str(key): float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("terminal Optuna campaign metrics must be finite")
    return metrics


def _plan_identity(config: Mapping) -> str:
    from .orchestration import _plan

    return _plan(config).identity


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
    objective_terms: tuple[ObjectiveTerm, ...]
    constraints: dict[str, tuple[str, float]]
    search_space: dict[str, dict]
    stages: dict[str, SweepStage]
    promotion: PromotionContract
    frozen_paths: tuple[str, ...]
    allowed_search_prefixes: tuple[str, ...]

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
) -> None:
    from .orchestration import (
        _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS,
        _EXTERNAL_PARENT_ECONOMIC_FIELDS,
    )

    _get_path(base_config, path)
    if _path_is_frozen(path, frozen_paths):
        raise ValueError(f"Optuna search assignment changes frozen path: {path}")
    if not any(path.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(f"Optuna search assignment is outside allowed paths: {path}")
    root, _, leaf = path.partition(".")
    if root in _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS or (
        root == "challenge" and leaf in _EXTERNAL_PARENT_ECONOMIC_FIELDS
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
        "search_space", "stages", "promotion", "frozen",
    } or payload["schema"] != OPTUNA_SWEEP_SCHEMA:
        raise ValueError("Optuna sweep contract is invalid")
    base_path = (path.parent / str(payload["base_config"])).resolve()
    base_bytes = base_path.read_bytes()
    base_config = json.loads(base_bytes)

    study = payload["study"]
    if not isinstance(study, dict) or set(study) != {
        "sampler", "seed", "n_trials", "n_jobs", "n_startup_trials",
    }:
        raise ValueError("Optuna study contract is invalid")
    n_jobs = int(study["n_jobs"])
    if study["sampler"] != "tpe" or not 1 <= n_jobs <= 3:
        raise ValueError(
            "PropEvolve requires constrained TPE with one to three isolated jobs"
        )
    n_trials = int(study["n_trials"])
    n_startup = int(study["n_startup_trials"])
    if n_trials < 1 or not 1 <= n_startup <= n_trials:
        raise ValueError("Optuna trial budget is invalid")

    frozen = payload["frozen"]
    if (
        not isinstance(frozen, dict)
        or set(frozen) != {
            "teacher_free_selection", "paths", "allowed_search_prefixes"
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
        objective_terms=objective_terms,
        constraints=constraints,
        search_space=search_space,
        stages=stages,
        promotion=promotion,
        frozen_paths=frozen_paths,
        allowed_search_prefixes=allowed_prefixes,
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


def _compile_campaign(
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
        raise ValueError(f"Optuna campaign is missing constraint metrics: {missing}")
    return [
        _constraint_value(float(metrics[metric]), operator, expected)
        for metric, (operator, expected) in constraints.items()
    ]


def _is_feasible(
    metrics: Mapping[str, float],
    constraints: Mapping[str, tuple[str, float]],
) -> bool:
    return all(value <= 0.0 for value in _constraint_vector(metrics, constraints))


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
    study: optuna.Study, count: int
) -> tuple[optuna.trial.FrozenTrial, ...]:
    eligible = [
        trial
        for trial in study.trials
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
    running = tuple(
        trial for trial in study.trials
        if trial.state is optuna.trial.TrialState.RUNNING
    )
    for trial in running:
        campaign_trial = int(
            trial.user_attrs.get("resume_campaign_trial", trial.number)
        )
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        reconciled.add(trial.number)
        if trial.params:
            study.enqueue_trial(
                trial.params,
                user_attrs={"resume_campaign_trial": campaign_trial},
            )
    study.set_user_attr("reconciled_interrupted_trials", sorted(reconciled))
    return frozenset(reconciled)


def _clean_code_commit(repository_root: Path) -> str:
    status = subprocess.check_output(
        ["git", "-C", str(repository_root), "status", "--porcelain"], text=True
    )
    if status:
        raise ValueError("Optuna study requires a clean PropEvolve checkout")
    return subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _run_compiled_campaign(
    *,
    config: dict,
    config_path: Path,
    run_id: str,
    stage_name: str,
    artifact_root: Path,
    runner,
    state_loader,
    config_validator,
) -> tuple[object, dict[str, float]]:
    _write_exact(config_path, config)
    if config_validator is None:
        from .config import load_experiment_config

        load_experiment_config(config_path)
    else:
        config_validator(config_path)
    expected_plan_identity = _plan_identity(config)
    state = state_loader(config_path, run_id)
    if state is None or state.phase.value not in _TERMINAL_PHASES:
        if runner is None:
            state = _default_runner(
                config_path,
                run_id=run_id,
                stdout_path=artifact_root / "logs" / f"{run_id}.stdout.log",
                stderr_path=artifact_root / "logs" / f"{run_id}.stderr.log",
            )
        else:
            state = runner(config_path, run_id=run_id)
    if state is None or state.phase.value not in _TERMINAL_PHASES:
        raise RuntimeError(f"Optuna campaign is not terminal: {run_id}")
    if state.plan_identity != expected_plan_identity:
        raise ValueError(f"Optuna campaign plan identity drifted: {run_id}")
    if state.phase.value == "BLOCKED":
        raise RuntimeError(f"Optuna campaign blocked: {run_id}")
    return state, _cell_metrics(state, stage_name)


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
    def run_confirmation(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
        parameters = dict(trial.params)
        run_root = artifacts / "confirmation" / f"trial-{trial.number:03d}"
        config = _compile_campaign(
            sweep,
            parameters=parameters,
            stage=sweep.stages["confirmation"],
            run_root=run_root,
        )
        run_id = f"{sweep.name}-confirm-trial-{trial.number:03d}"
        _, metrics = _run_compiled_campaign(
            config=config,
            config_path=configs / f".optuna-{run_id}.json",
            run_id=run_id,
            stage_name=sweep.stages["confirmation"].name,
            artifact_root=artifacts,
            runner=runner,
            state_loader=state_loader,
            config_validator=config_validator,
        )
        return {
            "trial_number": trial.number,
            "objective": _objective_value(metrics, sweep.objective_terms),
            "feasible": _is_feasible(metrics, sweep.promotion.acceptance),
            "metrics": metrics,
            "parameters": parameters,
        }

    with ThreadPoolExecutor(
        max_workers=min(sweep.n_jobs, len(promoted)),
    ) as executor:
        confirmations = list(executor.map(run_confirmation, promoted))
    finalists = tuple(sorted(
        (item for item in confirmations if item["feasible"]),
        key=lambda item: float(item["objective"]),
        reverse=True,
    )[:sweep.promotion.finalist_top_k])
    multi_seed: list[dict[str, object]] = []
    winner: dict[str, object] | None = None
    for finalist in finalists:
        trial_number = int(finalist["trial_number"])
        seed_results: list[dict[str, object]] = []
        short_circuit_reason: str | None = None
        for seed in sweep.promotion.seeds:
            run_root = (
                artifacts / "multi-seed" / f"trial-{trial_number:03d}"
                / f"seed-{seed}"
            )
            config = _compile_campaign(
                sweep,
                parameters=finalist["parameters"],
                stage=sweep.stages["multi_seed"],
                run_root=run_root,
                seed=seed,
            )
            run_id = (
                f"{sweep.name}-multiseed-trial-{trial_number:03d}-seed-{seed}"
            )
            _, metrics = _run_compiled_campaign(
                config=config,
                config_path=configs / f".optuna-{run_id}.json",
                run_id=run_id,
                stage_name=sweep.stages["multi_seed"].name,
                artifact_root=artifacts,
                runner=runner,
                state_loader=state_loader,
                config_validator=config_validator,
            )
            seed_results.append({
                "seed": seed,
                "objective": _objective_value(metrics, sweep.objective_terms),
                "feasible": _is_feasible(metrics, sweep.promotion.acceptance),
                "metrics": metrics,
            })
            if float(metrics.get("selection.blow_rate", 0.0)) > 0.0:
                short_circuit_reason = "zero_blow_gate"
                break
        feasible_count = sum(bool(item["feasible"]) for item in seed_results)
        mean_objective = sum(
            float(item["objective"]) for item in seed_results
        ) / len(seed_results)
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
        "promoted_trial_numbers": [trial.number for trial in promoted],
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
        sweep.base_config_path.parent
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    target_trials = sweep.n_trials if n_trials is None else int(n_trials)
    if target_trials < 1:
        raise ValueError("Optuna target trial count must be positive")
    storage_path = artifacts / "study.db"
    sweep_sha256 = _sha256(sweep.path.read_bytes())
    active_code_commit = code_commit or _clean_code_commit(repository_root)
    sampler = optuna.samplers.TPESampler(
        seed=sweep.seed,
        n_startup_trials=min(sweep.n_startup_trials, target_trials),
        constraints_func=_constraints_func,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=sweep.name,
        storage=f"sqlite:///{storage_path}",
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
    reserved_baseline: optuna.Trial | None = None
    if not study.trials:
        study.enqueue_trial(
            _baseline_parameters(sweep), user_attrs={"baseline_control": True}
        )
        # Reserve the queued baseline before concurrent sampling. SQLite-backed
        # Optuna workers must never race to claim the same WAITING control.
        reserved_baseline = study.ask()
    load_state = state_loader or _default_state_loader

    def objective(trial: optuna.Trial) -> float:
        campaign_trial = int(
            trial.user_attrs.get("resume_campaign_trial", trial.number)
        )
        parameters = _sample(trial, sweep)
        trial_root = artifacts / "screening" / f"trial-{campaign_trial:03d}"
        config = _compile_campaign(
            sweep,
            parameters=parameters,
            stage=sweep.stages["screening"],
            run_root=trial_root,
        )
        config_path = configs / (
            f".optuna-{sweep.name}-screen-trial-{campaign_trial:03d}.json"
        )
        run_id = f"{sweep.name}-screen-trial-{campaign_trial:03d}"
        print(
            f"[optuna] trial={trial.number}/{target_trials - 1} START "
            f"campaign_trial={campaign_trial} "
            f"params={json.dumps(parameters, sort_keys=True)}",
            flush=True,
        )
        state, metrics = _run_compiled_campaign(
            config=config,
            config_path=config_path,
            run_id=run_id,
            stage_name=sweep.stages["screening"].name,
            artifact_root=artifacts,
            runner=runner,
            state_loader=load_state,
            config_validator=config_validator,
        )
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("campaign_phase", state.phase.value)
        for metric, value in metrics.items():
            trial.set_user_attr(metric, float(value))
        if float(metrics.get("training.short_circuited", 0.0)) == 1.0:
            trial.set_user_attr("constraint", [1.0] * (len(sweep.constraints) + 1))
            raise optuna.TrialPruned("training short circuit rejected trial")
        if float(metrics.get("selection.short_circuited", 0.0)) == 1.0:
            trial.set_user_attr("constraint", [1.0] * (len(sweep.constraints) + 1))
            return -1_000_000.0
        missing = sorted(_required_metrics(sweep) - set(metrics))
        if missing:
            raise ValueError(f"Optuna campaign is missing objective metrics: {missing}")
        constraints = [
            0.0 if state.phase.value == "COMPLETE" else 1.0,
            *_constraint_vector(metrics, sweep.constraints),
        ]
        trial.set_user_attr("constraint", constraints)
        value = _objective_value(metrics, sweep.objective_terms)
        selected_metrics = {
            metric: metrics[metric] for metric in sorted(_required_metrics(sweep))
        }
        print(
            f"[optuna] trial={trial.number} COMPLETE phase={state.phase.value} "
            f"objective={value:.6f} "
            f"feasible={all(item <= 0.0 for item in constraints)} "
            f"metrics={json.dumps(selected_metrics, sort_keys=True)}",
            flush=True,
        )
        return value

    def finish_reserved_trial(trial: optuna.Trial) -> None:
        try:
            value = objective(trial)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except Exception:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        else:
            study.tell(trial, value)

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
    if reserved_baseline is not None:
        peer_trials = min(remaining - 1, max(0, sweep.n_jobs - 1))
        if peer_trials:
            with ThreadPoolExecutor(max_workers=2) as executor:
                baseline_future = executor.submit(
                    finish_reserved_trial, reserved_baseline
                )
                peer_future = executor.submit(
                    study.optimize,
                    objective,
                    n_trials=peer_trials,
                    n_jobs=peer_trials,
                )
                baseline_future.result()
                peer_future.result()
        else:
            finish_reserved_trial(reserved_baseline)
        remaining -= 1 + peer_trials
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=sweep.n_jobs)
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
            state_loader=load_state,
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
