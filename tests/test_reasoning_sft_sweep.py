from __future__ import annotations

import json
from pathlib import Path

import pytest

from propevolve.reasoning_policy.sft_sweep import (
    load_sft_sweep,
    materialize_trial_config,
    run_sft_sweep,
    selection_objective,
)
from propevolve.reasoning_policy.mlx_sft import read_sft_config


def _base_recipe(path: Path) -> Path:
    source = Path("config/reasoning/policy_sft.json").resolve()
    payload = json.loads(source.read_text())
    payload.pop("inherits", None)
    # Tests exercise sweep orchestration, not model loading.
    parent = Path("config/reasoning/sft.json").resolve()
    merged = json.loads(parent.read_text())
    merged.update(payload)
    defaults = json.loads(Path("config/reasoning/defaults.json").read_text())
    defaults.update(merged)
    defaults["data"] = str(path.parent / "dataset")
    defaults["resume_adapter_file"] = None
    path.write_text(json.dumps(defaults))
    return path


def _sweep(tmp_path: Path, *, trials: int = 5) -> Path:
    view = tmp_path / "view"
    view.mkdir()
    (view / "view_manifest.json").write_text(json.dumps({
        "source_manifest": {
            "counts": {"train": 18, "valid": 9},
            "splits": {"train": [1, 100], "valid": [100, 200]},
            "sealed_start_ns": 300,
        }
    }))
    payload = {
        "schema": "propevolve_reasoning_sft_sweep_v1",
        "name": "action-mastery-test",
        "workspace_root": str(tmp_path),
        "base_sft_config": str(_base_recipe(tmp_path / "base.json")),
        "view": str(view),
        "study_root": "study",
        "study": {
            "seed": 41, "n_trials": trials, "n_jobs": 1,
            "n_startup_trials": 2,
            "multivariate": True, "convergence_patience_trials": 4,
            "convergence_min_improvement": 0.01,
            "pruner": {"kind": "median", "n_startup_trials": 2,
                       "n_warmup_evaluations": 2, "interval_evaluations": 1},
        },
        "trial": {
            "epochs": 2, "evaluation_every_epochs": 1,
            "save_every_epochs": 1, "patience_evaluations": 2,
        },
        "search_space": {
            "learning_rate": {"path": "learning_rate", "choices": [1e-5, 3e-5]},
            "soft_weight": {"path": "action_supervision.soft_target_weight",
                            "choices": [0.25, 1.0]},
            "ranking_weight": {"path": "action_supervision.ranking_weight",
                               "choices": [1.0, 2.0]},
            "margin": {"path": "action_supervision.margin", "choices": [0.25, 0.4]},
        },
        "selection": {
            "required_actions": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
            "minimum_action_advantage": 0.25,
            "minimum_macro_accuracy": 1.0,
        },
        "temporal_selection": {
            "train": [1, 100], "validation": [100, 200],
            "untouched_evaluation_start_ns": 200,
            "final_confirmation_start_ns": 300,
        },
        "artifacts": {"delete_failed_trials": True,
                      "maximum_retained_trials": 2},
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(payload))
    return path


def _selection(long: float, short: float, wait: float) -> dict:
    values = {"ENTER_LONG_1": long, "ENTER_SHORT_1": short, "WAIT": wait}
    return {
        "best_iteration": 18,
        "best_report": {
            "worst_action_advantage": min(values.values()),
            "macro_accuracy": 1.0,
            "per_action": {
                name: {"mean_target_advantage": value, "accuracy": 1.0}
                for name, value in values.items()
            },
        },
    }


def test_sweep_materializes_epoch_budget_without_touching_holdouts(tmp_path):
    sweep = load_sft_sweep(_sweep(tmp_path))
    output = tmp_path / "trial.json"
    config = materialize_trial_config(sweep, {
        "learning_rate": 3e-5, "soft_weight": 0.25,
        "ranking_weight": 2.0, "margin": 0.4,
    }, trial_number=7, output=output)
    assert config["iters"] == 36
    assert config["steps_per_eval"] == 18
    assert config["val_batches"] == 9
    assert config["early_stopping"]["monitor"] == "worst_action_advantage"
    assert config["early_stopping"]["mode"] == "max"
    assert config["validation_metrics_path"].endswith(
        "trials/trial-007/validation-metrics.jsonl")
    assert read_sft_config(output)["early_stopping"] == config["early_stopping"]
    assert config["adapter_path"].endswith("trials/trial-007/adapter")
    assert sweep.temporal_selection["untouched_evaluation_start_ns"] == 200


def test_objective_fails_closed_if_any_action_boundary_is_missing_or_negative():
    assert selection_objective(_selection(.4, .3, .5),
        required_actions=("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"))[0] == .3
    with pytest.raises(ValueError, match="required action"):
        incomplete = _selection(.4, .3, .5)
        del incomplete["best_report"]["per_action"]["WAIT"]
        selection_objective(incomplete,
            required_actions=("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"))


def test_optuna_sweep_resumes_and_ranks_the_worst_action_not_average(tmp_path):
    path = _sweep(tmp_path, trials=5)
    calls = []
    evidence = [
        _selection(.8, -.2, .8),
        _selection(.31, .32, .30),
        _selection(.6, .2, .6),
        _selection(.4, .4, .4),
        _selection(.5, .5, .5),
    ]

    def runner(config_path, view, trial_number):
        calls.append((json.loads(Path(config_path).read_text()), Path(view), trial_number))
        return evidence[trial_number]

    first = run_sft_sweep(path, target_trials=3, trial_runner=runner)
    assert first["terminal_trials"] == 3
    assert first["best_trial_number"] == 1
    resumed = run_sft_sweep(path, target_trials=5, trial_runner=runner)
    assert resumed["terminal_trials"] == 5
    assert resumed["best_trial_number"] == 4
    assert len(calls) == 5
    assert all(call[1] == tmp_path / "view" for call in calls)
    assert json.loads((tmp_path / "study" / "study.result.json").read_text()) == resumed


def test_grid_sweep_evaluates_each_configured_pair_once(tmp_path):
    path = _sweep(tmp_path, trials=4)
    payload = json.loads(path.read_text())
    payload["study"]["sampler"] = "grid"
    payload["study"]["n_startup_trials"] = 1
    payload["study"]["convergence_patience_trials"] = 4
    payload["search_space"] = {
        "lora_learning_rate": {
            "path": "component_learning_rates.lora", "choices": [1e-6, 3e-6]},
        "projector_learning_rate": {
            "path": "component_learning_rates.projector", "choices": [3e-6, 1e-5]},
    }
    base = json.loads(Path(payload["base_sft_config"]).read_text())
    base["component_learning_rates"] = {"lora": 3e-6, "projector": 3e-6}
    Path(payload["base_sft_config"]).write_text(json.dumps(base))
    path.write_text(json.dumps(payload))
    observed = []

    def runner(config_path, view, trial_number):
        config = json.loads(Path(config_path).read_text())
        observed.append(tuple(config["component_learning_rates"].values()))
        return _selection(.3, .3, .3)

    result = run_sft_sweep(path, trial_runner=runner)
    assert result["terminal_trials"] == 4
    assert len(observed) == len(set(observed)) == 4


def test_sweep_rounds_epoch_budget_to_complete_optimizer_group(tmp_path):
    path = _sweep(tmp_path, trials=4)
    payload = json.loads(path.read_text())
    manifest = Path(payload["view"]) / "view_manifest.json"
    receipt = json.loads(manifest.read_text())
    receipt["source_manifest"]["counts"]["train"] = 19
    manifest.write_text(json.dumps(receipt))
    output = tmp_path / "trial.json"

    config = materialize_trial_config(load_sft_sweep(path), {
        "learning_rate": 3e-5, "soft_weight": 0.25,
        "ranking_weight": 2.0, "margin": 0.4,
    }, trial_number=7, output=output)

    assert config["iters"] == 39
    assert config["iters"] % config["grad_accumulation_steps"] == 0
    assert config["steps_per_eval"] == 19


def test_sweep_rejects_validation_that_touches_unseen_evaluation(tmp_path):
    path = _sweep(tmp_path)
    payload = json.loads(path.read_text())
    payload["temporal_selection"]["untouched_evaluation_start_ns"] = 150
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="untouched evaluation"):
        load_sft_sweep(path)
