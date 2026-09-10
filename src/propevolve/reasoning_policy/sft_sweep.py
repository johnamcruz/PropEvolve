"""Config-driven Optuna selection for reasoning-policy action SFT.

The sweep owns optimization only. It reuses an immutable prepared MLX view and
never invokes the prop campaign, RL stage, or sealed economic evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import optuna

from .model_config import read_recipe


SCHEMA = "propevolve_reasoning_sft_sweep_v1"


@dataclass(frozen=True)
class SFTSweep:
    path: Path
    name: str
    root: Path
    base_sft_config: Path
    view: Path
    study_root: Path
    study: dict
    trial: dict
    search_space: dict
    selection: dict
    temporal_selection: dict
    artifacts: dict
    identity: str


def _finite_number(value, *, minimum=None):
    valid = (not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(float(value)))
    if minimum is not None:
        valid = valid and float(value) >= minimum
    return valid


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _set_path(payload: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = payload
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"SFT sweep path does not resolve: {dotted}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError(f"SFT sweep path does not exist: {dotted}")
    if isinstance(node[parts[-1]], bool) or type(value) is not type(node[parts[-1]]):
        # Int choices are valid for existing floats, but booleans never are.
        if not (_finite_number(value) and _finite_number(node[parts[-1]])):
            raise ValueError(f"SFT sweep choice changes value type: {dotted}")
    node[parts[-1]] = value


def _get_path(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"SFT sweep path does not resolve: {dotted}")
        node = node[part]
    return node


def load_sft_sweep(path: str | Path) -> SFTSweep:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    required = {
        "schema", "name", "workspace_root", "base_sft_config", "view",
        "study_root", "study", "trial", "search_space", "selection",
        "temporal_selection", "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != SCHEMA:
        raise ValueError("reasoning SFT sweep contract is invalid")
    root = _resolve(path.parent, payload["workspace_root"])
    base = _resolve(root, payload["base_sft_config"])
    view = _resolve(root, payload["view"])
    study_root = _resolve(root, payload["study_root"])
    if not base.is_file() or not (view / "view_manifest.json").is_file():
        raise ValueError("reasoning SFT sweep resources are missing")

    study = payload["study"]
    study_keys = {
        "seed", "n_trials", "n_jobs", "n_startup_trials", "multivariate", "pruner",
        "convergence_patience_trials", "convergence_min_improvement",
    }
    if (not isinstance(study, dict) or not study_keys.issubset(study)
            or set(study) - study_keys - {"sampler"}
            or study.get("sampler", "tpe") not in {"tpe", "grid"}
            or type(study["seed"]) is not int or type(study["n_trials"]) is not int
            or type(study["n_jobs"]) is not int or study["n_jobs"] < 1
            or type(study["n_startup_trials"]) is not int
            or not 1 <= study["n_startup_trials"] <= study["n_trials"]
            or type(study["multivariate"]) is not bool
            or type(study["convergence_patience_trials"]) is not int
            or study["convergence_patience_trials"] < 1
            or not _finite_number(study["convergence_min_improvement"], minimum=0)):
        raise ValueError("reasoning SFT study settings are invalid")
    pruner = study["pruner"]
    if (not isinstance(pruner, dict) or set(pruner) != {
            "kind", "n_startup_trials", "n_warmup_evaluations",
            "interval_evaluations"}
            or pruner["kind"] != "median"
            or any(type(pruner[name]) is not int or pruner[name] < 0
                   for name in ("n_startup_trials", "n_warmup_evaluations"))
            or type(pruner["interval_evaluations"]) is not int
            or pruner["interval_evaluations"] < 1):
        raise ValueError("reasoning SFT pruner settings are invalid")
    trial = payload["trial"]
    if (not isinstance(trial, dict) or set(trial) != {
            "epochs", "evaluation_every_epochs", "save_every_epochs",
            "patience_evaluations"}
            or any(type(trial[name]) is not int or trial[name] < 1 for name in trial)):
        raise ValueError("reasoning SFT trial settings are invalid")

    search = payload["search_space"]
    if not isinstance(search, dict) or not search:
        raise ValueError("reasoning SFT search space is empty")
    base_payload = read_recipe(base)
    assigned = set()
    for name, dimension in search.items():
        if (not isinstance(name, str) or not name or not isinstance(dimension, dict)
                or set(dimension) != {"path", "choices"}
                or not isinstance(dimension["path"], str)
                or dimension["path"] in assigned
                or not isinstance(dimension["choices"], list)
                or len(dimension["choices"]) < 2):
            raise ValueError("reasoning SFT search dimension is invalid")
        assigned.add(dimension["path"])
        for choice in dimension["choices"]:
            candidate = deepcopy(base_payload)
            _set_path(candidate, dimension["path"], choice)
    if study.get("sampler", "tpe") == "grid":
        combinations = math.prod(len(dimension["choices"])
                                 for dimension in search.values())
        if study["n_trials"] != combinations:
            raise ValueError("grid study budget must equal its configured combinations")

    selection = payload["selection"]
    action_contract = {
        "required_actions", "minimum_action_advantage", "minimum_macro_accuracy"}
    task_contract = {
        "required_boundaries", "minimum_boundary_advantage", "minimum_macro_accuracy"}
    contract = set(selection) if isinstance(selection, dict) else set()
    required_key = "required_actions" if contract == action_contract else "required_boundaries"
    minimum_key = ("minimum_action_advantage" if contract == action_contract
                   else "minimum_boundary_advantage")
    if (contract not in {frozenset(action_contract), frozenset(task_contract)}
            or not isinstance(selection[required_key], list)
            or len(selection[required_key]) < 3
            or len(set(selection[required_key])) != len(selection[required_key])
            or not _finite_number(selection[minimum_key])
            or not _finite_number(selection["minimum_macro_accuracy"], minimum=0)
            or selection["minimum_macro_accuracy"] > 1):
        raise ValueError("reasoning SFT selection contract is invalid")

    temporal = payload["temporal_selection"]
    if (not isinstance(temporal, dict) or set(temporal) != {
            "train", "validation", "untouched_evaluation_start_ns",
            "final_confirmation_start_ns"}
            or any(not isinstance(temporal[name], list) or len(temporal[name]) != 2
                   for name in ("train", "validation"))):
        raise ValueError("reasoning SFT temporal contract is invalid")
    train, valid = temporal["train"], temporal["validation"]
    unseen, final = (temporal["untouched_evaluation_start_ns"],
                     temporal["final_confirmation_start_ns"])
    if (any(type(value) is not int for value in (*train, *valid, unseen, final))
            or not train[0] < train[1] <= valid[0] < valid[1] <= unseen < final):
        raise ValueError("reasoning SFT validation touches untouched evaluation data")
    receipt = json.loads((view / "view_manifest.json").read_text())
    source = receipt.get("source_manifest", {})
    if (source.get("splits") != {"train": train, "valid": valid}
            or source.get("sealed_start_ns") != final):
        raise ValueError("prepared view temporal roles differ from SFT selection contract")
    counts = source.get("counts")
    if (not isinstance(counts, dict) or any(type(counts.get(role)) is not int
            or counts[role] < 1 for role in ("train", "valid"))):
        raise ValueError("prepared view counts are invalid")

    artifacts = payload["artifacts"]
    if (not isinstance(artifacts, dict) or set(artifacts) != {
            "delete_failed_trials", "maximum_retained_trials"}
            or type(artifacts["delete_failed_trials"]) is not bool
            or type(artifacts["maximum_retained_trials"]) is not int
            or artifacts["maximum_retained_trials"] < 1):
        raise ValueError("reasoning SFT artifact contract is invalid")
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    return SFTSweep(path, payload["name"], root, base, view, study_root,
                    study, trial, search, selection, temporal, artifacts, identity)


def materialize_trial_config(sweep: SFTSweep, parameters: dict, *,
                             trial_number: int, output: Path) -> dict:
    if set(parameters) != set(sweep.search_space):
        raise ValueError("SFT trial parameters differ from configured search space")
    from .mlx_sft import read_sft_config
    config = deepcopy(read_sft_config(sweep.base_sft_config, root=sweep.root))
    for name, value in parameters.items():
        dimension = sweep.search_space[name]
        if value not in dimension["choices"]:
            raise ValueError(f"SFT trial choice is outside configured search: {name}")
        _set_path(config, dimension["path"], value)
    receipt = json.loads((sweep.view / "view_manifest.json").read_text())
    counts = receipt["source_manifest"]["counts"]
    train_batches = counts["train"] // config["batch_size"]
    valid_batches = counts["valid"] // config["batch_size"]
    if (train_batches < 1 or valid_batches * config["batch_size"] != counts["valid"]):
        raise ValueError("SFT sweep corpus does not form complete batches")
    requested_iterations = sweep.trial["epochs"] * train_batches
    accumulation = config["grad_accumulation_steps"]
    config["iters"] = math.ceil(requested_iterations / accumulation) * accumulation
    config["steps_per_eval"] = sweep.trial["evaluation_every_epochs"] * train_batches
    config["save_every"] = sweep.trial["save_every_epochs"] * train_batches
    config["val_batches"] = valid_batches
    hierarchical = "required_boundaries" in sweep.selection
    config["early_stopping"] = {
        **config["early_stopping"], "enabled": True,
        "patience_evaluations": sweep.trial["patience_evaluations"],
        "restore_best": True,
        "monitor": "worst_task_advantage" if hierarchical else "worst_action_advantage",
        "mode": "max",
    }
    trial_root = sweep.study_root / "trials" / f"trial-{trial_number:03d}"
    config["adapter_path"] = str(trial_root / "adapter")
    config["validation_metrics_path"] = str(trial_root / "validation-metrics.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def selection_objective(selection: dict, *, required_actions: tuple[str, ...] | None = None,
                        required_boundaries: tuple[str, ...] | None = None):
    report = selection.get("best_report")
    if (required_actions is None) == (required_boundaries is None):
        raise ValueError("SFT selection requires exactly one boundary family")
    required = required_actions if required_actions is not None else required_boundaries
    source_key = "per_action" if required_actions is not None else "per_task"
    macro_key = "macro_accuracy" if required_actions is not None else "task_macro_accuracy"
    boundaries = None if not isinstance(report, dict) else report.get(source_key)
    if not isinstance(boundaries, dict) or set(required) - set(boundaries):
        family = "action" if required_actions is not None else "task"
        raise ValueError(f"SFT selection is missing a required {family} boundary")
    margins = {}
    for name in required:
        value = boundaries[name].get("mean_target_advantage")
        if not _finite_number(value):
            raise ValueError("SFT decision boundary is non-finite")
        margins[name] = float(value)
    macro = report.get(macro_key)
    if not _finite_number(macro, minimum=0) or macro > 1:
        raise ValueError("SFT macro accuracy is invalid")
    return min(margins.values()), margins, float(macro)


def _default_trial_runner(config_path: Path, view: Path, trial_number: int,
                          trial: optuna.Trial) -> dict:
    config = json.loads(config_path.read_text())
    trial_root = Path(config["adapter_path"]).parent
    trial_root.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config["validation_metrics_path"])
    monitor = config["early_stopping"]["monitor"]
    with (trial_root / "train.log").open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "propevolve.reasoning_policy.mlx_sft",
             "--config", str(config_path), "--view", str(view), "--train"],
            cwd=config.get("workspace_root"), stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        reported = 0
        while process.poll() is None:
            if metrics_path.is_file():
                rows = metrics_path.read_text().splitlines()
                for raw in rows[reported:]:
                    report = json.loads(raw)
                    trial.report(float(report[monitor]), step=reported)
                    reported += 1
                    if trial.should_prune():
                        process.terminate()
                        process.wait(timeout=30)
                        raise optuna.TrialPruned(
                            f"worst decision margin pruned after evaluation {reported}")
            time.sleep(1)
        returncode = process.returncode
    if returncode:
        raise RuntimeError(f"reasoning SFT trial {trial_number} exited {returncode}")
    return json.loads((Path(config["adapter_path"]) / "training_selection.json").read_text())


def _storage(path: Path):
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(path)))


def run_sft_sweep(path: str | Path, *, target_trials: int | None = None,
                  trial_runner=None) -> dict:
    sweep = load_sft_sweep(path)
    configured_trials = sweep.study["n_trials"]
    target = configured_trials if target_trials is None else int(target_trials)
    if target < 1 or target > configured_trials:
        raise ValueError("target trials must be within the JSON-configured study budget")
    sweep.study_root.mkdir(parents=True, exist_ok=True)
    sampler = (optuna.samplers.GridSampler(
        {name: dimension["choices"] for name, dimension in sweep.search_space.items()},
        seed=sweep.study["seed"])
        if sweep.study.get("sampler", "tpe") == "grid" else
        optuna.samplers.TPESampler(
            seed=sweep.study["seed"],
            n_startup_trials=min(sweep.study["n_startup_trials"], target),
            multivariate=sweep.study["multivariate"],
        ))
    study = optuna.create_study(
        study_name=sweep.name, direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=sweep.study["pruner"]["n_startup_trials"],
            n_warmup_steps=sweep.study["pruner"]["n_warmup_evaluations"],
            interval_steps=sweep.study["pruner"]["interval_evaluations"],
        ),
        storage=_storage(sweep.study_root / "study.journal.log"),
        load_if_exists=True,
    )
    authority = study.user_attrs.get("sweep_identity")
    if authority is not None and authority != sweep.identity:
        raise ValueError("reasoning SFT sweep changed while resuming its study")
    if authority is None:
        study.set_user_attr("sweep_identity", sweep.identity)
    runner = _default_trial_runner if trial_runner is None else trial_runner
    if not study.trials and sweep.study.get("sampler", "tpe") != "grid":
        base = read_recipe(sweep.base_sft_config)
        baseline = {name: _get_path(base, dimension["path"])
                    for name, dimension in sweep.search_space.items()}
        if all(value in sweep.search_space[name]["choices"]
               for name, value in baseline.items()):
            study.enqueue_trial(baseline, user_attrs={"baseline_control": True})

    def objective(trial):
        parameters = {name: trial.suggest_categorical(name, dimension["choices"])
                      for name, dimension in sweep.search_space.items()}
        trial_root = sweep.study_root / "trials" / f"trial-{trial.number:03d}"
        config_path = trial_root / "config.json"
        materialize_trial_config(sweep, parameters, trial_number=trial.number,
                                 output=config_path)
        trial.set_user_attr("config_path", str(config_path))
        try:
            selection = (runner(config_path, sweep.view, trial.number, trial)
                         if trial_runner is None else
                         runner(config_path, sweep.view, trial.number))
            kwargs = ({"required_actions": tuple(sweep.selection["required_actions"])}
                      if "required_actions" in sweep.selection else
                      {"required_boundaries": tuple(sweep.selection["required_boundaries"])})
            value, margins, macro = selection_objective(selection, **kwargs)
        except Exception:
            if sweep.artifacts["delete_failed_trials"] and trial_root.exists():
                shutil.rmtree(trial_root)
            raise
        minimum = sweep.selection.get(
            "minimum_action_advantage", sweep.selection.get("minimum_boundary_advantage"))
        mastered = (value >= minimum
                    and macro >= sweep.selection["minimum_macro_accuracy"])
        # Preserve the v1 result key for existing readers while exposing the
        # more precise task/action-neutral name to new diagnostics.
        trial.set_user_attr("action_margins", margins)
        trial.set_user_attr("decision_margins", margins)
        trial.set_user_attr("macro_accuracy", macro)
        trial.set_user_attr("mastered", mastered)
        print(f"[reasoning-sft-optuna] trial={trial.number} COMPLETE "
              f"objective={value:+.6f} mastered={str(mastered).lower()} "
              f"margins={json.dumps(margins, sort_keys=True)}", flush=True)
        return value

    terminal = [item for item in study.trials if item.state in {
        optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL}]
    remaining = max(0, target - len(terminal))

    def convergence_and_retention(active_study, completed_trial):
        complete = [item for item in active_study.trials
                    if item.state is optuna.trial.TrialState.COMPLETE]
        keep_count = sweep.artifacts["maximum_retained_trials"]
        keep = {item.number for item in sorted(
            complete, key=lambda item: item.value, reverse=True)[:keep_count]}
        for item in complete:
            if item.number not in keep:
                adapter = (sweep.study_root / "trials" /
                           f"trial-{item.number:03d}" / "adapter")
                if adapter.exists():
                    shutil.rmtree(adapter)
        patience = sweep.study["convergence_patience_trials"]
        if len(complete) <= patience:
            return
        ordered = sorted(complete, key=lambda item: item.number)
        earlier = ordered[:-patience]
        recent = ordered[-patience:]
        prior_best = max(item.value for item in earlier)
        recent_best = max(item.value for item in recent)
        if recent_best < prior_best + sweep.study["convergence_min_improvement"]:
            active_study.set_user_attr("converged_at_trial", completed_trial.number)
            active_study.stop()

    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=sweep.study["n_jobs"],
                       callbacks=[convergence_and_retention])
    terminal = [item for item in study.trials if item.state in {
        optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL}]
    complete = [item for item in terminal if item.state is optuna.trial.TrialState.COMPLETE]
    best = max(complete, key=lambda item: item.value) if complete else None
    result = {
        "schema": "propevolve_reasoning_sft_sweep_result_v1",
        "study": sweep.name,
        "configured_trials": configured_trials,
        "target_trials": target,
        "terminal_trials": len(terminal),
        "completed_trials": len(complete),
        "failed_trials": sum(item.state is optuna.trial.TrialState.FAIL for item in terminal),
        "best_trial_number": None if best is None else best.number,
        "best_objective": None if best is None else best.value,
        "best_parameters": None if best is None else best.params,
        "best_action_margins": None if best is None else best.user_attrs.get("action_margins"),
        "mastered": False if best is None else bool(best.user_attrs.get("mastered")),
    }
    destination = sweep.study_root / "study.result.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-trials", type=int)
    args = parser.parse_args(argv)
    result = run_sft_sweep(args.config, target_trials=args.target_trials)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
