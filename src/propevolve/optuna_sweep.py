"""Constrained Optuna search over the existing PropEvolve campaign runner."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import optuna

OPTUNA_SWEEP_SCHEMA = "propevolve_optuna_sweep_v1"
OBJECTIVE_FORMULA = (
    "100*pass_rate + 8*min(average_win_r,8)"
)
SEARCHABLE_PATHS = frozenset({
    "agent.learning_rate",
    "agent.policy_retention_loss_weight",
    "challenge.large_win_bonus_coefficient",
    "regime_selectivity.loss_weight",
    "regime_selectivity.persistent_chop_negative_emphasis",
    "training.teacher_guidance_dropout_end",
})
RISK_INVARIANTS = (
    "challenge.max_loss",
    "challenge.max_position_size",
    "challenge.minimum_mll_headroom",
    "challenge.per_trade_risk_dollars",
    "challenge.trailing_mll_lock",
)
FEASIBILITY_CONSTRAINTS = {
    "selection.blow_rate": ("==", 0.0),
    "selection.average_win_r": (">=", 3.0),
    "selection.pass_rate": (">=", 0.2),
    "selection.expectancy_r": (">=", 0.0),
    "selection.trade_win_rate": (">=", 0.38),
    "selection.two_r_mfe_capture_ratio": (">=", 0.7),
    "selection.long_entry_count": (">", 0.0),
    "selection.short_entry_count": (">", 0.0),
    "selection.near_blow_timeout_rate": ("<=", 0.6263636363636363),
    "selection.short_circuited": ("==", 0.0),
}


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


def _write_exact(path: Path, payload: Mapping) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise ValueError(f"existing Optuna artifact drifted: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


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
            check=False,
        )
    state = _default_state_loader(config_path, run_id)
    if state is None:
        raise RuntimeError(
            f"Optuna trial exited {process.returncode} without campaign state"
        )
    if process.returncode not in {0, 2}:
        raise RuntimeError(
            f"Optuna trial {run_id} exited unexpectedly with {process.returncode}"
        )
    return state


def _cell_metrics(state, screening_stage: str) -> dict[str, float]:
    receipts = tuple(
        receipt for receipt in state.receipts if receipt.stage == screening_stage
    )
    if len(receipts) != 1:
        raise ValueError(
            f"terminal Optuna trial must have one {screening_stage!r} receipt"
        )
    raw = receipts[0].outputs.get("metrics")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("terminal Optuna trial metrics are missing")
    metrics = {str(key): float(value) for key, value in raw.items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("terminal Optuna trial metrics must be finite")
    return metrics


def _plan_identity(config: Mapping) -> str:
    from .orchestration import _plan

    return _plan(config).identity


@dataclass(frozen=True)
class OptunaSweep:
    path: Path
    name: str
    base_config_path: Path
    base_config: dict
    base_config_sha256: str
    screening_stage: str
    study_root: str
    sampler: str
    seed: int
    n_trials: int
    n_jobs: int
    n_startup_trials: int
    objective_formula: str
    constraints: dict[str, tuple[str, float]]
    search_space: dict[str, dict]


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_optuna_sweep(path: str | Path) -> OptunaSweep:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    required = {
        "schema",
        "name",
        "base_config",
        "screening_stage",
        "study_root",
        "study",
        "objective",
        "search_space",
        "frozen",
    }
    if set(payload) != required or payload["schema"] != OPTUNA_SWEEP_SCHEMA:
        raise ValueError("Optuna sweep contract is invalid")
    study = payload["study"]
    if set(study) != {
        "sampler",
        "seed",
        "n_trials",
        "n_jobs",
        "n_startup_trials",
    }:
        raise ValueError("Optuna study contract is invalid")
    if study["sampler"] != "tpe" or int(study["n_jobs"]) != 1:
        raise ValueError("Stage 2A requires constrained TPE with one MPS job")
    n_trials = int(study["n_trials"])
    n_startup = int(study["n_startup_trials"])
    if n_trials < 1 or not 1 <= n_startup <= n_trials:
        raise ValueError("Optuna trial budget is invalid")
    frozen = payload["frozen"]
    if (
        frozen.get("mps_workers") != 1
        or frozen.get("teacher_free_selection") is not True
        or frozen.get("sealed_confirmation_start") != "2026-01-01"
        or tuple(frozen.get("risk_invariants", ())) != RISK_INVARIANTS
    ):
        raise ValueError("Optuna frozen safety contract is invalid")
    objective = payload["objective"]
    if (
        set(objective) != {"formula", "constraints"}
        or objective["formula"] != OBJECTIVE_FORMULA
    ):
        raise ValueError("Optuna objective contract is invalid")
    constraints = {}
    raw_constraints = objective.get("constraints")
    if not isinstance(raw_constraints, dict) or not raw_constraints:
        raise ValueError("Optuna feasibility constraints are required")
    for metric, rule in raw_constraints.items():
        if (
            not isinstance(rule, dict)
            or set(rule) != {"operator", "value"}
            or rule["operator"] not in {"==", ">=", ">", "<="}
            or isinstance(rule["value"], bool)
            or not isinstance(rule["value"], (int, float))
            or not math.isfinite(float(rule["value"]))
        ):
            raise ValueError(f"Optuna constraint is invalid: {metric}")
        constraints[str(metric)] = (str(rule["operator"]), float(rule["value"]))
    if constraints != FEASIBILITY_CONSTRAINTS:
        raise ValueError("Optuna feasibility constraints drifted")
    search_space = payload["search_space"]
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("Optuna search space must be nonempty")
    from .orchestration import (
        _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS,
        _EXTERNAL_PARENT_ECONOMIC_FIELDS,
    )

    parent_bound = []
    for name in search_space:
        root, _, leaf = str(name).partition(".")
        if root in _EXTERNAL_PARENT_CAUSAL_RECIPE_PATHS or (
            root == "challenge" and leaf in _EXTERNAL_PARENT_ECONOMIC_FIELDS
        ):
            parent_bound.append(str(name))
    if parent_bound:
        raise ValueError(
            "Optuna search space changes external parent contract: "
            + ", ".join(sorted(parent_bound))
        )
    if set(search_space) != SEARCHABLE_PATHS:
        raise ValueError("Optuna search space drifted")
    for name, specification in search_space.items():
        if not isinstance(specification, dict):
            raise ValueError(f"Optuna search dimension is invalid: {name}")
        kind = specification.get("type")
        allowed = {"type", "choices"} if kind == "categorical" else {
            "type", "low", "high", "log"
        }
        required_dimension = (
            {"type", "choices"}
            if kind == "categorical"
            else {"type", "low", "high"}
        )
        if (
            not required_dimension <= set(specification) <= allowed
            or kind not in {"categorical", "float"}
        ):
            raise ValueError(f"Optuna search dimension is invalid: {name}")
        if kind == "categorical":
            if (
                not isinstance(specification["choices"], list)
                or not specification["choices"]
            ):
                raise ValueError(f"Optuna choices are invalid: {name}")
        else:
            low = float(specification["low"])
            high = float(specification["high"])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"Optuna bounds are invalid: {name}")
    base_path = (path.parent / str(payload["base_config"])).resolve()
    base_bytes = base_path.read_bytes()
    return OptunaSweep(
        path=path,
        name=str(payload["name"]),
        base_config_path=base_path,
        base_config=json.loads(base_bytes),
        base_config_sha256=_sha256(base_bytes),
        screening_stage=str(payload["screening_stage"]),
        study_root=str(payload["study_root"]),
        sampler="tpe",
        seed=int(study["seed"]),
        n_trials=n_trials,
        n_jobs=1,
        n_startup_trials=n_startup,
        objective_formula=str(payload["objective"]["formula"]),
        constraints=constraints,
        search_space=search_space,
    )


def _constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    values = trial.user_attrs.get("constraint")
    if not isinstance(values, (list, tuple)):
        return (1.0,)
    return tuple(float(value) for value in values)


def _sample(trial: optuna.Trial, sweep: OptunaSweep) -> dict[str, object]:
    sampled = {}
    for name, specification in sweep.search_space.items():
        if specification["type"] == "categorical":
            value = trial.suggest_categorical(name, specification["choices"])
        else:
            value = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=bool(specification.get("log", False)),
            )
        sampled[name] = value
    return sampled


def _compile_trial(
    sweep: OptunaSweep,
    trial: optuna.Trial,
    artifact_root: Path,
) -> tuple[dict, dict[str, object]]:
    parameters = _sample(trial, sweep)
    config = deepcopy(sweep.base_config)
    for path, value in parameters.items():
        _set_path(config, path, value)
    stages = [
        stage
        for stage in config["campaign"]["budget_stages"]
        if stage["name"] == sweep.screening_stage
    ]
    if len(stages) != 1:
        raise ValueError("Optuna screening stage must resolve exactly once")
    trial_root = artifact_root / "trials" / f"trial-{trial.number:03d}"
    config["output"] = str(trial_root)
    config["campaign"]["state_root"] = str(trial_root / "ml-loop-state")
    config["campaign"]["budget_stages"] = deepcopy(stages)
    config["campaign"]["max_revisions_per_stage"] = 0
    config["evolution"]["allowed_revision_paths"] = []
    config["evolution"]["revision_bounds"] = {}
    config["evolution"]["hypothesis"] = (
        "Constrained TPE Stage 2A trial for zero blow, at least 3R average "
        "winners, lower near-blow incidence, Expansion participation, both "
        "directions, and teacher-free pass-rate improvement."
    )
    return config, parameters


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


def _objective_value(metrics: Mapping[str, float]) -> float:
    return (
        100.0 * float(metrics["selection.pass_rate"])
        + 8.0 * min(float(metrics["selection.average_win_r"]), 8.0)
    )


def _best_feasible(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    eligible = [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and all(value <= 0.0 for value in _constraints_func(trial))
    ]
    return max(eligible, key=lambda trial: float(trial.value)) if eligible else None


def _reconcile_interrupted_trials(study: optuna.Study) -> frozenset[int]:
    reconciled = set(int(value) for value in study.user_attrs.get(
        "reconciled_interrupted_trials", ()
    ))
    running = tuple(
        trial
        for trial in study.trials
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
        ["git", "-C", str(repository_root), "status", "--porcelain"],
        text=True,
    )
    if status:
        raise ValueError("Optuna study requires a clean PropEvolve checkout")
    return subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


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
    """Run or resume a constrained TPE study through one physical MPS slot."""
    sweep = load_optuna_sweep(path)
    repository_root = sweep.base_config_path.parent.parent
    artifacts = Path(artifact_root) if artifact_root is not None else (
        repository_root / sweep.study_root
    )
    configs = (
        Path(config_root)
        if config_root is not None
        else sweep.base_config_path.parent
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
    existing_authority = {
        name: study.user_attrs.get(name) for name in authority
    }
    if any(value is not None for value in existing_authority.values()):
        if existing_authority != authority:
            raise ValueError("Optuna study authority drifted")
    else:
        for name, value in authority.items():
            study.set_user_attr(name, value)
    reconciled = _reconcile_interrupted_trials(study)
    load_state = state_loader or _default_state_loader

    def objective(trial: optuna.Trial) -> float:
        campaign_trial = int(
            trial.user_attrs.get("resume_campaign_trial", trial.number)
        )
        config, parameters = _compile_trial(sweep, trial, artifacts)
        if campaign_trial != trial.number:
            trial_root = artifacts / "trials" / f"trial-{campaign_trial:03d}"
            config["output"] = str(trial_root)
            config["campaign"]["state_root"] = str(
                trial_root / "ml-loop-state"
            )
        config_path = configs / (
            f".optuna-{sweep.name}-trial-{campaign_trial:03d}.json"
        )
        _write_exact(config_path, config)
        if config_validator is None:
            from .config import load_experiment_config

            load_experiment_config(config_path)
        else:
            config_validator(config_path)
        run_id = f"{sweep.name}-trial-{campaign_trial:03d}"
        expected_plan_identity = _plan_identity(config)
        print(
            f"[optuna] trial={trial.number}/{target_trials - 1} START "
            f"campaign_trial={campaign_trial} "
            f"params={json.dumps(parameters, sort_keys=True)}",
            flush=True,
        )
        state = load_state(config_path, run_id)
        if state is None or state.phase.value not in {
            "COMPLETE", "FAILED_GATE", "STOPPED", "BLOCKED"
        }:
            if runner is None:
                state = _default_runner(
                    config_path,
                    run_id=run_id,
                    stdout_path=(
                        artifacts / "logs" / f"trial-{trial.number:03d}.stdout.log"
                    ),
                    stderr_path=(
                        artifacts / "logs" / f"trial-{trial.number:03d}.stderr.log"
                    ),
                )
            else:
                state = runner(config_path, run_id=run_id)
        if state is None or state.phase.value not in {
            "COMPLETE", "FAILED_GATE", "STOPPED", "BLOCKED"
        }:
            raise RuntimeError(f"Optuna trial is not terminal: {trial.number}")
        if state.plan_identity != expected_plan_identity:
            raise ValueError(f"Optuna trial plan identity drifted: {trial.number}")
        if state.phase.value == "BLOCKED":
            study.stop()
            raise RuntimeError(f"Optuna trial blocked; study stopped: {trial.number}")
        metrics = _cell_metrics(state, sweep.screening_stage)
        trial.set_user_attr("parameters", parameters)
        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("campaign_phase", state.phase.value)
        for metric, value in metrics.items():
            trial.set_user_attr(metric, float(value))
        if float(metrics.get("training.short_circuited", 0.0)) == 1.0:
            trial.set_user_attr("constraint", [1.0] * (len(sweep.constraints) + 1))
            raise optuna.TrialPruned("existing training short circuit rejected trial")
        if float(metrics.get("selection.short_circuited", 0.0)) == 1.0:
            trial.set_user_attr("constraint", [1.0] * (len(sweep.constraints) + 1))
            print(
                f"[optuna] trial={trial.number} INFEASIBLE "
                "selection_short_circuit=true",
                flush=True,
            )
            return -1_000_000.0
        missing = sorted(set(sweep.constraints) - set(metrics))
        if missing:
            raise ValueError(f"Optuna trial is missing constraint metrics: {missing}")
        constraints = [0.0 if state.phase.value == "COMPLETE" else 1.0, *[
            _constraint_value(float(metrics[metric]), operator, expected)
            for metric, (operator, expected) in sweep.constraints.items()
        ]]
        trial.set_user_attr("constraint", constraints)
        value = _objective_value(metrics)
        print(
            f"[optuna] trial={trial.number} COMPLETE phase={state.phase.value} "
            f"objective={value:.6f} pass={metrics['selection.pass_rate']:.3f} "
            f"blow={metrics['selection.blow_rate']:.3f} "
            f"winR={metrics['selection.average_win_r']:.3f} "
            f"near_blow={metrics['selection.near_blow_timeout_rate']:.3f} "
            f"feasible={all(item <= 0.0 for item in constraints)}",
            flush=True,
        )
        return value

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    effective_terminal = tuple(
        trial
        for trial in study.trials
        if trial.state in terminal_states and trial.number not in reconciled
    )
    remaining = max(0, target_trials - len(effective_terminal))
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)
    best = _best_feasible(study)
    effective_trials = [
        trial for trial in study.trials if trial.number not in reconciled
    ]
    states = [trial.state for trial in effective_trials]
    terminal_count = sum(state in terminal_states for state in states)
    status = (
        "COMPLETE"
        if terminal_count >= target_trials and best is not None
        else "FAILED_GATE"
        if terminal_count >= target_trials
        else "INCOMPLETE"
    )
    result_payload = {
        "schema": "propevolve_optuna_sweep_result_v1",
        "study": sweep.name,
        "status": status,
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
            metric: best.user_attrs.get(metric) for metric in sweep.constraints
        },
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
    )


__all__ = [
    "OPTUNA_SWEEP_SCHEMA",
    "OptunaSweep",
    "OptunaSweepResult",
    "load_optuna_sweep",
    "run_optuna_sweep",
]
