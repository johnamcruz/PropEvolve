from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from ml_training_loop import Phase, RunState, StageReceipt
import pytest

from propevolve.config import load_experiment_config
from propevolve.orchestration import _plan
import propevolve.optuna_engine as optuna_engine
from propevolve.optuna_sweep import load_optuna_sweep, run_optuna_sweep
from propevolve.training import _trend_start_confluence_agent_settings
from tests.recipe_fixtures import active_sweep_recipe


ACTIVE_CONTRACT = active_sweep_recipe().resolve()
_ACTIVE_PAYLOAD = json.loads(ACTIVE_CONTRACT.read_text())
BASE_CONFIG = (
    ACTIVE_CONTRACT.parent / _ACTIVE_PAYLOAD["base_config"]
).resolve()
STAGE = str(_ACTIVE_PAYLOAD["stages"]["screening"]["name"])
SCREENING_SHORT_CIRCUIT = {
    "minimum_completed_episodes": 50,
    "minimum_passes": 1,
    "maximum_blow_rate": 0.30,
    "collapse": {
        "window_episodes": 5,
        "minimum_prior_passes": 2,
        "maximum_recent_passes": 0,
        "maximum_average_hold_bars": 4,
        "minimum_voluntary_close_rate": 0.8,
    },
}


def test_exact_artifact_write_is_safe_for_concurrent_identical_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".optuna-trial.json"
    payload = {"trial": 0, "episodes": 50}
    barrier = threading.Barrier(3, timeout=5.0)
    original_exists = Path.exists

    def synchronized_exists(candidate: Path) -> bool:
        if candidate == path:
            barrier.wait()
            return False
        return original_exists(candidate)

    monkeypatch.setattr(Path, "exists", synchronized_exists)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(optuna_engine._write_exact, path, payload)
            for _ in range(3)
        ]
        for future in futures:
            future.result()

    assert json.loads(path.read_text()) == payload


def _payload(
    *,
    promotion_enabled: bool = False,
    n_trials: int = 3,
    n_jobs: int = 1,
) -> dict:
    return {
        "schema": "propevolve_optuna_sweep_v2",
        "name": "stage2_learning_contract_tpe_test",
        "base_config": str(BASE_CONFIG),
        "study_root": "runs/stage2_learning_contract_tpe_test",
        "study": {
            "sampler": "tpe",
            "seed": 314159,
            "n_trials": n_trials,
            "n_jobs": n_jobs,
            "n_startup_trials": min(2, n_trials),
        },
        "artifacts": {"screening_retention": "compact"},
        "objective": {
            "terms": [
                {"metric": "selection.pass_rate", "weight": 100.0},
                {
                    "metric": "selection.average_win_r",
                    "weight": 8.0,
                    "cap": 8.0,
                },
                {
                    "metric": "selection.near_blow_timeout_rate",
                    "weight": -20.0,
                },
            ],
            "constraints": {
                "selection.blow_rate": {"operator": "==", "value": 0.0},
                "selection.expectancy_r": {"operator": ">=", "value": 0.0},
                "selection.two_r_mfe_capture_ratio": {
                    "operator": ">=",
                    "value": 0.7,
                },
                "selection.long_entry_count": {
                    "operator": ">",
                    "value": 0.0,
                },
                "selection.short_entry_count": {
                    "operator": ">",
                    "value": 0.0,
                },
                "selection.short_circuited": {
                    "operator": "==",
                    "value": 0.0,
                },
            },
        },
        "search_space": {
            "challenge.large_win_bonus_coefficient": {
                "type": "float",
                "low": 0.05,
                "high": 0.30,
            },
            "regime_selectivity.loss_weight": {
                "type": "float",
                "low": 0.10,
                "high": 0.80,
                "log": True,
            },
            "balance_curriculum.outcome_contrast_replay.update_period": {
                "type": "int",
                "low": 1,
                "high": 8,
                "step": 1,
            },
            "replay_mix": {
                "type": "categorical_mapping",
                "choices": [
                    {
                        "name": "active_control",
                        "values": {
                            "training.terminal_sequence_fraction": 0.375,
                            "training.safety_sequence_fraction": 0.375,
                            "training.entry_opportunity_sequence_fraction": 0.25,
                        },
                    },
                    {
                        "name": "opportunity_leaning",
                        "values": {
                            "training.terminal_sequence_fraction": 0.4375,
                            "training.safety_sequence_fraction": 0.1875,
                            "training.entry_opportunity_sequence_fraction": 0.375,
                        },
                    },
                ],
            },
        },
        "stages": {
            "screening": {
                "name": STAGE,
                "training_episodes": 50,
                "validation_episodes": 50,
                "start_pnls": [0.0],
                "balance_validation_episodes": 0,
                "short_circuit": SCREENING_SHORT_CIRCUIT,
            },
            "confirmation": {
                "name": STAGE,
                "training_episodes": 200,
                "validation_episodes": 200,
                "start_pnls": [0.0],
                "balance_validation_episodes": 0,
                "short_circuit": None,
            },
            "multi_seed": {
                "name": STAGE,
                "training_episodes": 200,
                "validation_episodes": 200,
                "start_pnls": [0.0],
                "balance_validation_episodes": 0,
                "short_circuit": None,
            },
        },
        "promotion": {
            "enabled": promotion_enabled,
            "top_k": min(2, n_trials),
            "finalist_top_k": 1,
            "seeds": [11, 22],
            "seed_paths": ["training.seed"],
            "required_feasible_seeds": 2,
            "acceptance": {
                "selection.pass_rate": {"operator": ">=", "value": 0.60},
                "selection.blow_rate": {"operator": "==", "value": 0.0},
                "selection.near_blow_timeout_rate": {
                    "operator": "<=",
                    "value": 0.25,
                },
                "selection.two_r_mfe_capture_ratio": {
                    "operator": ">=",
                    "value": 0.70,
                },
                "selection.long_entry_count": {
                    "operator": ">",
                    "value": 0.0,
                },
                "selection.short_entry_count": {
                    "operator": ">",
                    "value": 0.0,
                },
            },
        },
        "frozen": {
            "teacher_free_selection": True,
            "allowed_search_prefixes": [
                "agent.",
                "challenge.",
                "regime_selectivity.",
                "training.",
                "balance_curriculum.",
            ],
            "paths": [
                "teachers",
                "entry_supervision",
                "observation",
                "temporal",
                "sealed_confirmation",
                "agent.hidden_dim",
                "agent.atoms",
                "agent.value_min",
                "agent.value_max",
                "agent.gamma",
                "agent.n_step_return",
                "agent.recurrent_burn_in",
                "agent.learning_rate",
                "agent.policy_retention_loss_weight",
                "challenge.profit_target",
                "challenge.max_loss",
                "challenge.max_position_size",
                "challenge.minimum_mll_headroom",
                "challenge.per_trade_risk_dollars",
                "challenge.trailing_mll_lock",
            ],
        },
    }


def _contract(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(_payload(**kwargs)))
    return path


def _metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "selection.pass_rate": 0.25,
        "selection.blow_rate": 0.0,
        "selection.near_blow_timeout_rate": 0.20,
        "selection.average_win_r": 2.0,
        "selection.expectancy_r": 0.2,
        "selection.two_r_mfe_capture_ratio": 0.75,
        "selection.long_entry_count": 10.0,
        "selection.short_entry_count": 10.0,
        "selection.short_circuited": 0.0,
        "training.short_circuited": 0.0,
        "training.sampled_entry_action_long_rows": 100.0,
        "training.sampled_entry_action_short_rows": 100.0,
        "training.regime_selectivity_paired_a_plus_long_pair_mass": 50.0,
        "training.regime_selectivity_paired_a_plus_short_pair_mass": 50.0,
        "training.final_regime_probe_transition_positive_long_response": 0.1,
        "training.final_regime_probe_transition_positive_short_response": 0.1,
    }
    metrics.update(overrides)
    return metrics


def _state(config_path: Path, run_id: str, metrics: dict[str, float]) -> RunState:
    config = json.loads(config_path.read_text())
    return RunState(
        run_id,
        _plan(config).identity,
        Phase.COMPLETE,
        receipts=(StageReceipt(
            STAGE,
            1,
            "complete",
            {"metrics": metrics},
        ),),
    )


def test_v2_contract_is_fully_json_driven(tmp_path: Path) -> None:
    sweep = load_optuna_sweep(_contract(tmp_path))

    assert set(sweep.search_space) == {
        "challenge.large_win_bonus_coefficient",
        "regime_selectivity.loss_weight",
        "balance_curriculum.outcome_contrast_replay.update_period",
        "replay_mix",
    }
    assert [term.metric for term in sweep.objective_terms] == [
        "selection.pass_rate",
        "selection.average_win_r",
        "selection.near_blow_timeout_rate",
    ]
    assert sweep.stages["screening"].training_episodes == 50
    assert sweep.stages["screening"].start_pnls == (0.0,)
    assert sweep.stages["screening"].balance_validation_episodes == 0
    assert sweep.stages["confirmation"].validation_episodes == 200
    assert sweep.promotion.seeds == (11, 22)
    assert "agent.learning_rate" in sweep.frozen_paths


def test_v2_runs_three_screening_trials_in_isolated_parallel_slots(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(3, timeout=5.0)
    lock = threading.Lock()
    active = 0
    peak_active = 0
    run_ids: list[str] = []

    def runner(config_path: Path, *, run_id: str):
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            run_ids.append(run_id)
        try:
            barrier.wait()
            return _state(config_path, run_id, _metrics())
        finally:
            with lock:
                active -= 1

    result = run_optuna_sweep(
        _contract(tmp_path, n_trials=4, n_jobs=3),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    assert peak_active == 3
    assert len(run_ids) == len(set(run_ids)) == 4


def test_v2_parallel_screening_uses_lock_safe_journal_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_storage: list[object] = []
    create_study = optuna_engine.optuna.create_study

    def recording_create_study(*args, **kwargs):
        observed_storage.append(kwargs.get("storage"))
        return create_study(*args, **kwargs)

    monkeypatch.setattr(
        optuna_engine.optuna,
        "create_study",
        recording_create_study,
    )
    run_optuna_sweep(
        _contract(tmp_path, n_trials=1, n_jobs=3),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=lambda config_path, *, run_id: _state(
            config_path,
            run_id,
            _metrics(),
        ),
        state_loader=lambda *_args: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert isinstance(
        observed_storage[0],
        optuna_engine.optuna.storages.JournalStorage,
    )


def test_v2_default_generated_configs_share_the_study_lifecycle(
    tmp_path: Path,
) -> None:
    """A fresh study must not collide with configs from an older study."""
    base_payload = json.loads(BASE_CONFIG.read_text())
    base_payload["workspace_root"] = str(BASE_CONFIG.parent.parent)
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_payload))
    sweep_payload = _payload(n_trials=1, n_jobs=1)
    sweep_payload["base_config"] = base_path.name
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep_payload))
    artifact_root = tmp_path / "study"
    observed_configs: list[Path] = []

    def runner(config_path: Path, *, run_id: str):
        observed_configs.append(config_path)
        return {"evaluation_status": "PASS", "metrics": _metrics()}

    result = run_optuna_sweep(
        sweep_path,
        artifact_root=artifact_root,
        runner=runner,
        state_loader=lambda *_args: None,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    assert len(observed_configs) == 1
    assert observed_configs[0].parent == artifact_root / "configs"


def test_v2_uses_optuna_to_refill_worker_after_each_terminal_trial(
    tmp_path: Path,
) -> None:
    initial_barrier = threading.Barrier(3, timeout=5.0)
    release_slow_trials = threading.Event()
    refill_started = threading.Event()
    lock = threading.Lock()
    starts = 0

    def runner(config_path: Path, *, run_id: str):
        nonlocal starts
        with lock:
            ordinal = starts
            starts += 1
        if ordinal < 3:
            initial_barrier.wait()
            if ordinal == 0:
                return {
                    "evaluation_status": "PASS",
                    "metrics": _metrics(),
                }
            assert release_slow_trials.wait(timeout=5.0), (
                "Optuna did not refill a freed worker slot"
            )
        else:
            refill_started.set()
            release_slow_trials.set()
        return {
            "evaluation_status": "PASS",
            "metrics": _metrics(),
        }

    result = run_optuna_sweep(
        _contract(tmp_path, n_trials=5, n_jobs=3),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda *_args: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    assert starts == 5
    assert refill_started.is_set()
    assert result.completed_trials == 5
    assert result.failed_trials == 0


def test_v2_resumes_interrupted_trial_with_same_params_and_artifact_identity(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, n_trials=2, n_jobs=1)
    sweep = load_optuna_sweep(contract)
    study_root = tmp_path / "study"
    study_root.mkdir()
    study = optuna_engine.optuna.create_study(
        direction="maximize",
        sampler=optuna_engine.optuna.samplers.TPESampler(
            seed=sweep.seed,
            constraints_func=optuna_engine._constraints_func,
        ),
        study_name=sweep.name,
        storage=optuna_engine._study_storage(
            study_root / "study.journal.log"
        ),
    )
    authority = {
        "base_config_sha256": sweep.base_config_sha256,
        "code_commit": "test-commit",
        "sweep_config_sha256": optuna_engine._sha256(contract.read_bytes()),
        "target_trials": 2,
    }
    for name, value in authority.items():
        study.set_user_attr(name, value)
    baseline = optuna_engine._baseline_parameters(sweep)
    study.enqueue_trial(baseline, user_attrs={"baseline_control": True})
    interrupted = study.ask()
    sampled = optuna_engine._sample(interrupted, sweep)
    interrupted.set_user_attr("artifact_trial_number", interrupted.number)

    observed: list[tuple[str, dict[str, object]]] = []

    def runner(config_path: Path, *, run_id: str):
        observed.append((run_id, json.loads(config_path.read_text())))
        return {"evaluation_status": "PASS", "metrics": _metrics()}

    result = run_optuna_sweep(
        contract,
        artifact_root=study_root,
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda *_args: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    resumed = optuna_engine.optuna.load_study(
        study_name=sweep.name,
        storage=optuna_engine._study_storage(result.study_path),
    )
    assert result.interrupted_trials == 1
    assert result.completed_trials == 2
    assert result.failed_trials == 0
    assert len(observed) == 2
    assert observed[0][0].endswith("screen-trial-000")
    retry = next(
        trial for trial in resumed.trials
        if trial.user_attrs.get("retry_of_trial") == interrupted.number
    )
    assert retry.params == sampled
    assert retry.state is optuna_engine.optuna.trial.TrialState.COMPLETE


def test_v2_retries_executor_failure_without_consuming_trial_target(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path, n_trials=2, n_jobs=1)
    study_root = tmp_path / "study"

    with pytest.raises(RuntimeError, match="resume infrastructure failed"):
        run_optuna_sweep(
            contract,
            artifact_root=study_root,
            config_root=tmp_path / "configs",
            runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("resume infrastructure failed")
            ),
            state_loader=lambda *_args: None,
            config_validator=lambda path: None,
            code_commit="test-commit",
        )

    observed: list[str] = []

    def working_runner(config_path: Path, *, run_id: str):
        observed.append(run_id)
        return {"evaluation_status": "PASS", "metrics": _metrics()}

    result = run_optuna_sweep(
        contract,
        artifact_root=study_root,
        config_root=tmp_path / "configs",
        runner=working_runner,
        state_loader=lambda *_args: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    study = optuna_engine.optuna.load_study(
        study_name="stage2_learning_contract_tpe_test",
        storage=optuna_engine._study_storage(result.study_path),
    )
    retried = next(
        trial for trial in study.trials
        if trial.user_attrs.get("retry_of_trial") == 0
    )
    assert result.completed_trials == 2
    assert result.failed_trials == 0
    assert result.interrupted_trials == 1
    assert len(observed) == 2
    assert observed[0].endswith("screen-trial-000")
    assert retried.state is optuna_engine.optuna.trial.TrialState.COMPLETE


def test_v2_reports_one_economic_summary_when_trial_completes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = _metrics(**{
        "selection.pass_rate": 0.60,
        "selection.blow_rate": 0.0,
        "selection.near_blow_timeout_rate": 0.10,
        "selection.average_win_r": 3.0,
    })

    run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=lambda config_path, *, run_id: _state(
            config_path, run_id, metrics
        ),
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    result_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[optuna-result]")
    ]
    assert result_lines == [
        "[optuna-result] trial=0 state=COMPLETE feasible=true "
        "objective=82.0 pass_rate=60% blow_rate=0%"
    ]


def test_v2_reports_one_economic_summary_when_trial_is_pruned(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = _metrics(**{
        "selection.pass_rate": 0.02,
        "selection.blow_rate": 0.20,
        "training.short_circuited": 1.0,
    })

    run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=lambda config_path, *, run_id: _state(
            config_path, run_id, metrics
        ),
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    result_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[optuna-result]")
    ]
    assert result_lines == [
        "[optuna-result] trial=0 state=PRUNED feasible=false "
        "objective=null pass_rate=2% blow_rate=20%"
    ]


def test_v2_prunes_before_validation_without_fabricating_selection_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=lambda _config_path, *, run_id: {
            "evaluation_status": "FAIL",
            "metrics": {"training.short_circuited": 1.0},
        },
        state_loader=lambda *_args: (_ for _ in ()).throw(
            AssertionError("Optuna must not load campaign state")
        ),
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.pruned_trials == 1
    assert result.failed_trials == 0
    result_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[optuna-result]")
    ]
    assert result_lines == [
        "[optuna-result] trial=0 state=PRUNED feasible=false "
        "objective=null pass_rate=n/a blow_rate=n/a"
    ]


def test_v2_compacts_rejected_trial_only_after_terminal_study_record(
    tmp_path: Path,
) -> None:
    study_root = tmp_path / "study"

    def runner(config_path: Path, *, run_id: str):
        config = json.loads(config_path.read_text())
        trial_root = Path(config["output"])
        (trial_root / "training-replay").mkdir(parents=True)
        (trial_root / "training-replay" / "episode.pkl").write_bytes(
            b"disposable-replay"
        )
        (trial_root / "training-diagnostic-summary.json").write_text(
            json.dumps({"run_id": run_id, "status": "short_circuited"})
        )
        return {
            "evaluation_status": "FAIL",
            "metrics": {"training.short_circuited": 1.0},
        }

    result = run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=study_root,
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda *_args: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.pruned_trials == 1
    assert not (study_root / "screening" / "trial-000").exists()
    assert (
        study_root
        / "screening-evidence"
        / "trial-000"
        / "training-diagnostic-summary.json"
    ).is_file()
    receipt = json.loads(
        (study_root / "cleanup" / "trial-000.json").read_text()
    )
    assert receipt["trial_state"] == "PRUNED"
    assert receipt["removed_bytes"] > 0
    study = optuna_engine.optuna.load_study(
        study_name="stage2_learning_contract_tpe_test",
        storage=optuna_engine._study_storage(result.study_path),
    )
    trial, = study.trials
    assert trial.state is optuna_engine.optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["training.short_circuited"] == 1.0


def test_v2_reports_penalized_result_when_validation_short_circuits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metrics = _metrics(**{
        "selection.pass_rate": 0.0,
        "selection.blow_rate": 1.0,
        "selection.short_circuited": 1.0,
    })

    run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=lambda config_path, *, run_id: _state(
            config_path, run_id, metrics
        ),
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    result_lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[optuna-result]")
    ]
    assert result_lines == [
        "[optuna-result] trial=0 state=COMPLETE feasible=false "
        "objective=-1000000.0 pass_rate=0% blow_rate=100%"
    ]


def test_v2_stops_study_after_systemic_trial_executor_failure(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def broken_runner(_config_path: Path, *, run_id: str):
        calls.append(run_id)
        raise RuntimeError("systemic trial executor failure")

    with pytest.raises(RuntimeError, match="systemic trial executor failure"):
        run_optuna_sweep(
            _contract(tmp_path, n_trials=5, n_jobs=1),
            artifact_root=tmp_path / "study",
            config_root=tmp_path / "configs",
            runner=broken_runner,
            state_loader=lambda *_args: None,
            config_validator=lambda path: None,
            code_commit="test-commit",
        )

    assert calls == ["stage2_learning_contract_tpe_test-screen-trial-000"]
    study = optuna_engine.optuna.load_study(
        study_name="stage2_learning_contract_tpe_test",
        storage=optuna_engine._study_storage(
            tmp_path / "study" / "study.journal.log"
        ),
    )
    trial, = study.trials
    assert trial.state is optuna_engine.optuna.trial.TrialState.FAIL
    assert trial.user_attrs["stopped_study_on_executor_error"] is True


def test_v2_default_worker_is_an_isolated_capped_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        result_path = Path(command[-1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            "schema": "propevolve_optuna_trial_result_v1",
            "config_sha256": optuna_engine._sha256(
                config_path.read_bytes()
            ),
            "evaluation_status": "FAIL",
            "metrics": _metrics(),
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(optuna_engine.subprocess, "run", fake_run)
    config_path = tmp_path / "trial.json"
    config_path.write_text("{}")
    result_path = tmp_path / "trial-result.json"

    actual_result = optuna_engine._default_trial_runner(
        config_path,
        run_id="isolated-trial-001",
        result_path=result_path,
        stdout_path=tmp_path / "trial.stdout.log",
        stderr_path=tmp_path / "trial.stderr.log",
    )

    assert actual_result["evaluation_status"] == "FAIL"
    assert captured["command"][1:7] == [
        "-u",
        "-m",
        "propevolve.cli",
        "optuna-trial",
        "--config",
        str(config_path),
    ]
    assert captured["command"][-2:] == ["--result", str(result_path)]
    environment = captured["environment"]
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["VECLIB_MAXIMUM_THREADS"] == "1"


@pytest.mark.parametrize("n_jobs", (0, 4))
def test_v2_rejects_parallelism_outside_the_bounded_mac_contract(
    tmp_path: Path,
    n_jobs: int,
) -> None:
    with pytest.raises(ValueError, match="one to three isolated jobs"):
        load_optuna_sweep(_contract(tmp_path, n_jobs=n_jobs))


def test_v2_rejects_any_search_assignment_under_a_frozen_path(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["search_space"]["agent.learning_rate"] = {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="frozen path"):
        load_optuna_sweep(path)


def test_v2_rejects_entry_label_search_even_when_training_weight_is_allowed(
    tmp_path: Path,
) -> None:
    payload = _ACTIVE_PAYLOAD.copy()
    payload["base_config"] = str(BASE_CONFIG)
    payload["search_space"] = {
        **payload["search_space"],
        "entry_supervision.target_r": {
            "type": "categorical",
            "choices": [2.0, 3.0],
        },
    }
    payload["frozen"] = {
        **payload["frozen"],
        "paths": [
            path
            for path in payload["frozen"]["paths"]
            if path != "entry_supervision.target_r"
        ],
    }
    path = tmp_path / "invalid-entry-label-search.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="Optuna search space changes external parent contract",
    ):
        load_optuna_sweep(path)


def test_v2_rejects_search_space_without_exact_baseline_control(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["search_space"]["challenge.large_win_bonus_coefficient"] = {
        "type": "float",
        "low": 0.20,
        "high": 0.30,
    }
    path = tmp_path / "missing-baseline.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="baseline"):
        load_optuna_sweep(path)


def test_v2_screening_applies_numeric_and_grouped_json_dimensions(
    tmp_path: Path,
) -> None:
    configs: list[dict] = []

    def runner(config_path: Path, *, run_id: str):
        config = json.loads(config_path.read_text())
        configs.append(config)
        return _state(config_path, run_id, _metrics())

    result = run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    assert len(configs) == 1
    config = configs[0]
    stage, = config["campaign"]["budget_stages"]
    assert stage["training_episodes"] == 50
    assert stage["validation_episodes"] == 50
    assert stage["short_circuit_minimum_episodes"] == 50
    assert config["training"]["short_circuit"] == SCREENING_SHORT_CIRCUIT
    assert config["balance_curriculum"]["start_pnls"] == [0.0]
    assert config["balance_curriculum"]["validation_episodes"] == 0
    assert all(
        not item["metric"].startswith("balance_stress.")
        for item in stage["selection_requirements"]
    )
    assert config["training"]["terminal_sequence_fraction"] == 0.4375
    assert config["training"]["safety_sequence_fraction"] == 0.1875
    assert config["training"]["entry_opportunity_sequence_fraction"] == 0.375
    base = json.loads(BASE_CONFIG.read_text())
    assert config["teachers"] == base["teachers"]
    assert config["entry_supervision"] == base["entry_supervision"]
    for path in (
        ("agent", "hidden_dim"),
        ("agent", "atoms"),
        ("agent", "recurrent_burn_in"),
        ("challenge", "profit_target"),
        ("challenge", "max_loss"),
        ("challenge", "minimum_mll_headroom"),
    ):
        root, leaf = path
        assert config[root][leaf] == base[root][leaf]


def test_active_sweep_compiles_through_real_config_validation(
    tmp_path: Path,
) -> None:
    sweep = load_optuna_sweep(ACTIVE_CONTRACT)

    def runner(config_path: Path, *, run_id: str):
        normalized = load_experiment_config(config_path)
        assert normalized["runtime"]["learner_backend"] == "mlx"
        assert normalized["agent"]["device"] == "mps"
        assert normalized["runtime"]["mixed_precision"] == "off"
        assert normalized["runtime"]["compile_model"] is False
        assert normalized["training"]["prefetch_batches"] == 0
        return RunState(
            run_id,
            _plan(normalized).identity,
            Phase.COMPLETE,
            receipts=(StageReceipt(
                STAGE,
                1,
                "complete",
                {"metrics": _metrics()},
            ),),
        )

    result = run_optuna_sweep(
        ACTIVE_CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        n_trials=1,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        code_commit="test-commit",
    )

    assert sweep.n_trials == 50
    assert sweep.n_jobs == 3
    assert sweep.base_config["runtime"]["learner_backend"] == "mlx"
    assert sweep.base_config["training"]["prefetch_batches"] == 0
    assert sweep.stages["screening"].short_circuit == SCREENING_SHORT_CIRCUIT
    assert sweep.stages["screening"].training_episodes == 50
    assert sweep.stages["screening"].validation_episodes == 50
    assert sweep.stages["screening"].start_pnls == (0.0,)
    assert sweep.stages["screening"].balance_validation_episodes == 0
    assert sweep.stages["confirmation"].short_circuit is None
    assert sweep.stages["multi_seed"].short_circuit is None
    assert result.status == "COMPLETE"


def test_active_sweep_every_search_value_compiles_through_real_validation(
    tmp_path: Path,
) -> None:
    sweep = load_optuna_sweep(ACTIVE_CONTRACT)
    baseline = optuna_engine._baseline_parameters(sweep)
    for name, specification in sweep.search_space.items():
        for index, value in enumerate(specification["choices"]):
            parameters = {**baseline, name: value}
            label = f"{name.replace('.', '-')}-{index}"
            config = optuna_engine._compile_trial_config(
                sweep,
                parameters=parameters,
                stage=sweep.stages["screening"],
                run_root=tmp_path / label,
            )
            path = tmp_path / f"{label}.json"
            optuna_engine._write_exact(path, config)
            normalized = load_experiment_config(path)
            assert optuna_engine._get_path(normalized, name) == value
            assert normalized["runtime"]["learner_backend"] == "mlx"
            assert normalized["agent"]["device"] == "mps"
            assert normalized["trend_start_confluence"]["enabled"] is True
            if name.startswith("trend_start_confluence."):
                agent_settings = _trend_start_confluence_agent_settings(
                    normalized["trend_start_confluence"]
                )
                agent_path = {
                    "trend_start_confluence.loss_weight": (
                        "trend_start_confluence_loss_weight"
                    ),
                    "trend_start_confluence.opportunity_loss_weight": (
                        "trend_start_confluence_opportunity_loss_weight"
                    ),
                    "trend_start_confluence.safety_loss_weight": (
                        "trend_start_confluence_safety_loss_weight"
                    ),
                    "trend_start_confluence.margin": (
                        "trend_start_confluence_margin"
                    ),
                    "trend_start_confluence.confirmation_lookback_bars": (
                        "trend_start_confluence_confirmation_lookback_bars"
                    ),
                }[name]
                assert agent_settings[agent_path] == value


def test_direct_trial_economics_are_scored_without_campaign_phase(
    tmp_path: Path,
) -> None:
    def runner(config_path: Path, *, run_id: str):
        load_experiment_config(config_path)
        return {
            "evaluation_status": "FAIL",
            "metrics": _metrics(**{"selection.pass_rate": 0.60}),
        }

    def reject_campaign_state(*_args, **_kwargs):
        raise AssertionError("direct Optuna selection must not load campaign state")

    result = run_optuna_sweep(
        _contract(tmp_path, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=reject_campaign_state,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    study = optuna_engine.optuna.load_study(
        study_name="stage2_learning_contract_tpe_test",
        storage=optuna_engine._study_storage(result.study_path),
    )
    trial, = study.trials
    assert trial.state is optuna_engine.optuna.trial.TrialState.COMPLETE
    assert trial.user_attrs["evaluation_status"] == "FAIL"
    assert "campaign_phase" not in trial.user_attrs
    assert trial.user_attrs["selection.pass_rate"] == pytest.approx(0.60)


def test_active_sweep_freezes_verified_learning_mechanics() -> None:
    sweep = load_optuna_sweep(ACTIVE_CONTRACT)

    required_frozen_paths = {
        # Authenticated causal inputs and labels, including the five-bar entry
        # window and next-bar execution contract.
        "teachers",
        "observation",
        "entry_supervision.schema",
        "entry_supervision.training_only",
        "entry_supervision.decision_count",
        "entry_supervision.fill_offsets",
        "entry_supervision.execution",
        "entry_supervision.risk_dollars",
        "entry_supervision.launch",
        "entry_supervision.continuation",
        "entry_supervision.target_r",
        "entry_supervision.stop_r",
        "entry_supervision.horizon_bars",
        "entry_supervision.collision",
        "entry_supervision.action_class_balance",
        "entry_supervision.loss_weight",
        "regime_selectivity.expansion_center_receipt",
        "regime_selectivity.expansion_center_receipt_sha256",
        "regime_selectivity.expansion_long_center",
        "regime_selectivity.expansion_short_center",
        "regime_selectivity.probability_epsilon",
        # C51/R2D2 architecture and recurrent replay semantics.
        "agent.hidden_dim",
        "agent.atoms",
        "agent.value_min",
        "agent.value_max",
        "agent.gamma",
        "agent.n_step_return",
        "agent.recurrent_burn_in",
        "agent.auxiliary_gradient_conflict_mode",
        "agent.challenge_return_self_imitation_weight",
        "regime_selectivity.semantics",
        "regime_selectivity.side_balance",
        "training.sequence_length",
        "training.recurrent_horizon",
        # Prop challenge, action, execution, and safety invariants.
        "challenge.profit_target",
        "challenge.max_loss",
        "challenge.episode_days",
        "challenge.max_position_size",
        "challenge.minimum_mll_headroom",
        "challenge.trailing_mll_lock",
        "challenge.per_trade_risk_dollars",
        "challenge.large_win_bonus_coefficient",
        "challenge.mll_proximity_penalty_coefficient",
        "challenge.lead_giveback_penalty_coefficient",
        "regime_selectivity.loss_weight",
        "regime_selectivity.persistent_chop_negative_emphasis",
        "trend_start_confluence.schema",
        "trend_start_confluence.training_only",
        "trend_start_confluence.target_source",
        "trend_start_confluence.score_formula",
        "trend_start_confluence.semantics",
        "trend_start_confluence.enabled",
        "balance_curriculum.outcome_contrast_replay.update_period",
        "balance_curriculum.outcome_contrast_replay.max_examples",
        "sealed_confirmation",
    }

    assert _ACTIVE_PAYLOAD["frozen"]["teacher_free_selection"] is True
    assert required_frozen_paths <= set(sweep.frozen_paths)
    assert "entry_supervision.opportunity_loss_multiplier" not in (
        sweep.search_space
    )
    assert "agent.entry_action_margin" not in sweep.search_space
    assert "trend_start_confluence.enabled" in sweep.frozen_paths
    for path in (
        "trend_start_confluence.loss_weight",
        "trend_start_confluence.opportunity_loss_weight",
        "trend_start_confluence.safety_loss_weight",
        "trend_start_confluence.margin",
        "trend_start_confluence.confirmation_lookback_bars",
    ):
        assert path not in sweep.frozen_paths


def test_active_sweep_inherits_trial15_empirical_control() -> None:
    sweep = load_optuna_sweep(ACTIVE_CONTRACT)
    base = sweep.base_config

    assert base["entry_supervision"]["loss_weight"] == 0.90
    assert base["agent"]["challenge_return_self_imitation_weight"] == 0.025
    assert base["challenge"]["large_win_bonus_coefficient"] == 0.125
    assert base["challenge"]["mll_proximity_penalty_coefficient"] == 0.00030
    assert base["challenge"]["lead_giveback_penalty_coefficient"] == 0.00010
    assert base["regime_selectivity"]["loss_weight"] == pytest.approx(
        0.3604153677515533
    )
    assert base["regime_selectivity"][
        "persistent_chop_negative_emphasis"
    ] == 2.25
    assert base["balance_curriculum"]["outcome_contrast_replay"] == {
        "update_period": 5,
        "max_examples": 8,
    }
    assert base["entry_supervision"]["opportunity_loss_multiplier"] == 1.5
    assert base["agent"]["entry_action_margin"] == 0.4
    assert base["regime_selectivity"][
        "paired_a_plus_winner_loss_weight"
    ] == 2.5
    assert base["training"]["paired_a_plus_population_weighting"] == (
        "population_proportional_v1"
    )
    assert base["training"]["paired_a_plus_control_candidates"] == 8
    assert base["training"][
        "paired_a_plus_violation_replay_update_period"
    ] == 4
    assert base["training"][
        "paired_a_plus_violation_candidate_pairs_per_side"
    ] == 16
    assert base["training"]["paired_a_plus_violation_pairs_per_side"] == 1
    assert base["trend_start_confluence"]["enabled"] is True
    assert base["trend_start_confluence"]["loss_weight"] == 1.0
    assert base["trend_start_confluence"]["opportunity_loss_weight"] == 1.0
    assert base["trend_start_confluence"]["safety_loss_weight"] == 1.0
    assert base["trend_start_confluence"][
        "confirmation_lookback_bars"
    ] == 50
    trend = next(
        teacher for teacher in base["teachers"] if teacher["kind"] == "trend"
    )
    assert trend["loss_weight"] == 0.0


def test_active_sweep_searches_only_trend_usage() -> None:
    sweep = load_optuna_sweep(ACTIVE_CONTRACT)
    assert sweep.base_config["trend_start_confluence"]["enabled"] is True
    assert set(sweep.search_space) == {
        "trend_start_confluence.loss_weight",
        "trend_start_confluence.opportunity_loss_weight",
        "trend_start_confluence.safety_loss_weight",
        "trend_start_confluence.margin",
        "trend_start_confluence.confirmation_lookback_bars",
    }
    assert sweep.search_space["trend_start_confluence.loss_weight"] == {
        "type": "categorical",
        "choices": [0.25, 0.5, 0.75, 1.0],
    }
    assert sweep.search_space[
        "trend_start_confluence.opportunity_loss_weight"
    ] == {
        "type": "categorical",
        "choices": [0.5, 1.0, 1.5],
    }
    assert sweep.search_space[
        "trend_start_confluence.safety_loss_weight"
    ] == {
        "type": "categorical",
        "choices": [0.5, 1.0, 1.5],
    }
    assert sweep.search_space[
        "trend_start_confluence.margin"
    ] == {"type": "categorical", "choices": [0.1, 0.25, 0.4]}
    assert sweep.search_space[
        "trend_start_confluence.confirmation_lookback_bars"
    ] == {"type": "categorical", "choices": [5, 10, 20, 50]}
    assert "trend_start_confluence.enabled" in sweep.frozen_paths
    for path in (
        "training.paired_a_plus_control_candidates",
        "training.paired_a_plus_violation_replay_update_period",
        "training.paired_a_plus_violation_candidate_pairs_per_side",
        "training.paired_a_plus_violation_pairs_per_side",
        "regime_selectivity.paired_a_plus_winner_loss_weight",
    ):
        assert path in sweep.frozen_paths


def test_v2_runs_top_k_confirmation_then_multiseed_winner_retrain(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, int, int, int]] = []
    short_circuits: list[dict | None] = []

    def runner(config_path: Path, *, run_id: str):
        config = json.loads(config_path.read_text())
        stage, = config["campaign"]["budget_stages"]
        seed = int(config["training"]["seed"])
        short_circuits.append(config["training"]["short_circuit"])
        calls.append((
            run_id,
            int(stage["training_episodes"]),
            int(stage["validation_episodes"]),
            int(config["balance_curriculum"]["validation_episodes"]),
            seed,
        ))
        if "screen-trial-000" in run_id:
            metrics = _metrics(**{"selection.pass_rate": 0.35})
        elif "screen-trial-001" in run_id:
            metrics = _metrics(**{"selection.pass_rate": 0.55})
        elif "screen-trial-002" in run_id:
            metrics = _metrics(**{"selection.pass_rate": 0.45})
        elif "confirm-trial-001" in run_id:
            metrics = _metrics(**{
                "selection.pass_rate": 0.65,
                "selection.near_blow_timeout_rate": 0.15,
            })
        elif "confirm-trial-002" in run_id:
            metrics = _metrics(**{
                "selection.pass_rate": 0.61,
                "selection.near_blow_timeout_rate": 0.20,
            })
        else:
            metrics = _metrics(**{
                "selection.pass_rate": 0.64,
                "selection.near_blow_timeout_rate": 0.16,
            })
        return _state(config_path, run_id, metrics)

    result = run_optuna_sweep(
        _contract(tmp_path, promotion_enabled=True, n_trials=3),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert len(calls) == 7
    assert [item[1:4] for item in calls[:3]] == [(50, 50, 0)] * 3
    assert [item[1:4] for item in calls[3:5]] == [(200, 200, 0)] * 2
    assert [item[1:4] for item in calls[5:]] == [(200, 200, 0)] * 2
    assert short_circuits[:3] == [SCREENING_SHORT_CIRCUIT] * 3
    assert short_circuits[3:] == [None] * 4
    assert [item[4] for item in calls[5:]] == [11, 22]
    payload = json.loads(result.result_path.read_text())
    assert payload["promoted_trial_numbers"] == [1, 2]
    assert payload["winner_trial_number"] == 1
    assert payload["winner_feasible_seeds"] == 2
    assert payload["promotion_status"] == "COMPLETE"


def test_v2_runs_three_confirmation_trials_in_parallel(
    tmp_path: Path,
) -> None:
    payload = _payload(promotion_enabled=True, n_trials=3, n_jobs=3)
    payload["promotion"]["top_k"] = 3
    contract = tmp_path / "parallel-confirmation.json"
    contract.write_text(json.dumps(payload))
    barrier = threading.Barrier(3, timeout=5.0)
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def runner(config_path: Path, *, run_id: str):
        nonlocal active, peak_active
        if "-confirm-" in run_id:
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                barrier.wait()
            finally:
                with lock:
                    active -= 1
        return _state(config_path, run_id, _metrics(**{
            "selection.pass_rate": 0.65,
            "selection.near_blow_timeout_rate": 0.15,
        }))

    result = run_optuna_sweep(
        contract,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    assert peak_active == 3


def test_v2_multi_seed_promotion_stops_after_first_teacher_free_blow(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(config_path: Path, *, run_id: str):
        calls.append(run_id)
        if "multiseed" in run_id:
            metrics = _metrics(**{
                "selection.pass_rate": 0.0,
                "selection.blow_rate": 1.0,
                "selection.short_circuited": 1.0,
            })
        else:
            metrics = _metrics(**{
                "selection.pass_rate": 0.65,
                "selection.near_blow_timeout_rate": 0.15,
            })
        return _state(config_path, run_id, metrics)

    result = run_optuna_sweep(
        _contract(tmp_path, promotion_enabled=True, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert calls == [
        f"stage2_learning_contract_tpe_test-screen-trial-000",
        f"stage2_learning_contract_tpe_test-confirm-trial-000",
        f"stage2_learning_contract_tpe_test-multiseed-trial-000-seed-11",
    ]
    assert result.status == "FAILED_GATE"
    payload = json.loads(result.result_path.read_text())
    candidate, = payload["multi_seed"]
    assert candidate["short_circuit_reason"] == "zero_blow_gate"
    assert candidate["evaluated_seeds"] == 1
    assert candidate["requested_seeds"] == 2


def test_v2_confirmation_prunes_training_short_circuit_without_validation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(config_path: Path, *, run_id: str):
        calls.append(run_id)
        if "-confirm-" in run_id:
            return {
                "evaluation_status": "FAIL",
                "metrics": {"training.short_circuited": 1.0},
            }
        return _state(config_path, run_id, _metrics(**{
            "selection.pass_rate": 0.65,
            "selection.near_blow_timeout_rate": 0.15,
        }))

    result = run_optuna_sweep(
        _contract(tmp_path, promotion_enabled=True, n_trials=1),
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        runner=runner,
        state_loader=lambda *_args: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert calls == [
        "stage2_learning_contract_tpe_test-screen-trial-000",
        "stage2_learning_contract_tpe_test-confirm-trial-000",
    ]
    assert result.status == "FAILED_GATE"
    confirmation = optuna_engine.optuna.load_study(
        study_name="stage2_learning_contract_tpe_test-confirmation",
        storage=optuna_engine._study_storage(
            tmp_path / "study" / "confirmation.study.journal.log"
        ),
    )
    trial, = confirmation.trials
    assert trial.state is optuna_engine.optuna.trial.TrialState.PRUNED
    assert trial.user_attrs["training.short_circuited"] == 1.0
