from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ml_training_loop import Phase, RunState, StageReceipt
import optuna
import pytest

from propevolve.orchestration import _plan
from propevolve.optuna_sweep import load_optuna_sweep, run_optuna_sweep


CONTRACT = Path("config/sweeps/stage2a_regime_selectivity_tpe_v2.json")
CONFLUENCE_CONTRACT = Path(
    "config/sweeps/stage2a_regime_selectivity_tpe_v3.json"
)


def _state(
    config_path: Path,
    run_id: str,
    *,
    index: int,
    overrides: dict[str, float] | None = None,
) -> RunState:
    config = json.loads(config_path.read_text())
    metrics = {
        "selection.blow_rate": 0.0,
        "selection.near_blow_timeout_rate": 0.30 - index / 100,
        "selection.pass_rate": 0.20 + index / 100,
        "selection.average_win_r": 3.0 + index / 10,
        "selection.expectancy_r": 0.1,
        "selection.trade_win_rate": 0.5,
        "selection.two_r_mfe_capture_ratio": 0.75,
        "selection.long_entry_count": 3.0,
        "selection.short_entry_count": 4.0,
        "selection.short_circuited": 0.0,
        "training.short_circuited": 0.0,
    }
    metrics.update(overrides or {})
    return RunState(
        run_id,
        _plan(config).identity,
        Phase.COMPLETE,
        receipts=(StageReceipt(
            "persistent_chop_association_100ep",
            1,
            "complete",
            {"metrics": metrics},
        ),),
    )


def test_stage2a_tpe_contract_searches_only_causal_learning_knobs() -> None:
    sweep = load_optuna_sweep(CONTRACT)

    assert sweep.n_trials == 24
    assert sweep.n_jobs == 1
    assert sweep.sampler == "tpe"
    assert set(sweep.search_space) == {
        "challenge.large_win_bonus_coefficient",
        "regime_selectivity.loss_weight",
        "regime_selectivity.persistent_chop_negative_emphasis",
        "training.teacher_guidance_dropout_end",
    }
    assert sweep.base_config["agent"]["learning_rate"] == 0.0001
    assert sweep.base_config["agent"]["policy_retention_loss_weight"] == 10
    assert sweep.constraints == {
        "selection.average_win_r": (">=", 3.0),
        "selection.blow_rate": ("==", 0.0),
        "selection.expectancy_r": (">=", 0.0),
        "selection.trade_win_rate": (">=", 0.38),
        "selection.long_entry_count": (">", 0.0),
        "selection.near_blow_timeout_rate": ("<=", 0.6263636363636363),
        "selection.pass_rate": (">=", 0.2),
        "selection.short_circuited": ("==", 0.0),
        "selection.short_entry_count": (">", 0.0),
        "selection.two_r_mfe_capture_ratio": (">=", 0.7),
    }


def test_stage2a_v3_tpe_searches_the_fixed_wait_confluence_mechanism() -> None:
    sweep = load_optuna_sweep(CONFLUENCE_CONTRACT)

    assert sweep.screening_stage == "expansion_regime_confluence_100ep"
    assert sweep.base_config["regime_selectivity"]["semantics"] == (
        "expansion_regime_confluence_v3"
    )
    assert set(sweep.search_space) == {
        "challenge.large_win_bonus_coefficient",
        "regime_selectivity.loss_weight",
        "regime_selectivity.persistent_chop_negative_emphasis",
        "training.teacher_guidance_dropout_end",
    }
    assert sweep.base_config["agent"]["learning_rate"] == 0.0001
    assert sweep.base_config["agent"]["policy_retention_loss_weight"] == 10
    assert sweep.base_config["entry_supervision"]["target_r"] == 2.0


@pytest.mark.parametrize(
    "path",
    ("entry_supervision.loss_weight", "challenge.ratchet_giveback_r"),
)
def test_tpe_rejects_external_parent_contract_search_dimensions(
    tmp_path: Path,
    path: str,
) -> None:
    payload = json.loads(CONTRACT.read_text())
    payload["base_config"] = str(
        Path("config/historical_mask_expansion_anchored_regime_stage2_v10.json")
        .resolve()
    )
    payload["search_space"][path] = {
        "type": "float",
        "low": 0.1,
        "high": 0.2,
    }
    contract = tmp_path / "invalid-parent-drift.json"
    contract.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="Optuna search space changes external parent contract",
    ):
        load_optuna_sweep(contract)


def test_tpe_runs_existing_campaign_trials_sequentially_and_resumes_sqlite(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def runner(config_path: Path, *, run_id: str):
        calls.append(run_id)
        return _state(config_path, run_id, index=len(calls))

    result = run_optuna_sweep(
        CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        n_trials=3,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.status == "COMPLETE"
    assert len(calls) == 3
    assert result.best_trial_number == 2
    study = optuna.load_study(
        study_name="stage2a_regime_selectivity_tpe_v2",
        storage=f"sqlite:///{tmp_path / 'study' / 'study.db'}",
    )
    assert len(study.trials) == 3
    assert study.trials[2].value == pytest.approx(100 * 0.23 + 8 * 3.3)
    assert all(
        all(value <= 0.0 for value in trial.user_attrs["constraint"])
        for trial in study.trials
    )

    resumed = run_optuna_sweep(
        CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        n_trials=3,
        runner=lambda config_path, run_id: (_ for _ in ()).throw(
            AssertionError("completed Optuna study must not rerun trials")
        ),
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )
    assert resumed.best_trial_number == 2


def test_tpe_evaluates_exact_frozen_baseline_before_sampled_trials(
    tmp_path: Path,
) -> None:
    configs: list[dict[str, object]] = []

    def runner(config_path: Path, *, run_id: str):
        configs.append(json.loads(config_path.read_text()))
        return _state(config_path, run_id, index=len(configs))

    run_optuna_sweep(
        CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        n_trials=2,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert configs[0]["challenge"]["large_win_bonus_coefficient"] == 0.1
    assert configs[0]["regime_selectivity"]["loss_weight"] == 0.3
    assert configs[0]["regime_selectivity"][
        "persistent_chop_negative_emphasis"
    ] == 2.0
    assert configs[0]["training"]["teacher_guidance_dropout_end"] == 0.5
    study = optuna.load_study(
        study_name="stage2a_regime_selectivity_tpe_v2",
        storage=f"sqlite:///{tmp_path / 'study' / 'study.db'}",
    )
    assert study.trials[0].user_attrs["baseline_control"] is True


def test_tpe_selects_only_zero_blow_three_r_teacher_free_trials(
    tmp_path: Path,
) -> None:
    calls = []
    outcomes = (
        {
            "selection.pass_rate": 0.80,
            "selection.blow_rate": 0.0,
            "selection.average_win_r": 4.0,
        },
        {
            "selection.pass_rate": 0.70,
            "selection.blow_rate": 0.0,
            "selection.average_win_r": 2.9,
        },
        {
            "selection.pass_rate": 0.40,
            "selection.blow_rate": 0.0,
            "selection.average_win_r": 3.2,
        },
    )

    def runner(config_path: Path, *, run_id: str):
        index = len(calls)
        calls.append(run_id)
        state = _state(
            config_path,
            run_id,
            index=index,
            overrides=outcomes[index],
        )
        return replace(state, phase=Phase.FAILED_GATE) if index == 0 else state

    result = run_optuna_sweep(
        CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        n_trials=3,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert result.best_trial_number == 2
    payload = json.loads(result.result_path.read_text())
    assert payload["best_metrics"]["selection.blow_rate"] == 0.0
    assert payload["best_metrics"]["selection.average_win_r"] == 3.2


def test_tpe_continues_after_sparse_validation_short_circuit(
    tmp_path: Path,
) -> None:
    calls = []

    def runner(config_path: Path, *, run_id: str):
        calls.append(run_id)
        if len(calls) == 1:
            config = json.loads(config_path.read_text())
            return RunState(
                run_id,
                _plan(config).identity,
                Phase.FAILED_GATE,
                receipts=(StageReceipt(
                    "persistent_chop_association_100ep",
                    1,
                    "complete",
                    {"metrics": {
                        "selection.blow_rate": 1.0,
                        "selection.short_circuited": 1.0,
                        "training.short_circuited": 0.0,
                    }},
                ),),
            )
        return _state(config_path, run_id, index=2)

    result = run_optuna_sweep(
        CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "config",
        n_trials=2,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert len(calls) == 2
    assert result.best_trial_number == 1
    study = optuna.load_study(
        study_name="stage2a_regime_selectivity_tpe_v2",
        storage=f"sqlite:///{tmp_path / 'study' / 'study.db'}",
    )
    assert study.trials[0].value == -1_000_000.0


def test_tpe_requeues_interrupted_trial_without_reducing_budget(
    tmp_path: Path,
) -> None:
    sweep = load_optuna_sweep(CONTRACT)
    study_root = tmp_path / "study"
    study_root.mkdir()
    storage = f"sqlite:///{study_root / 'study.db'}"
    study = optuna.create_study(
        study_name=sweep.name,
        storage=storage,
        direction="maximize",
    )
    authority = {
        "base_config_sha256": sweep.base_config_sha256,
        "code_commit": "test-commit",
        "sweep_config_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "target_trials": 3,
    }
    for name, value in authority.items():
        study.set_user_attr(name, value)
    fixed_parameters = {
        name: specification["low"]
        for name, specification in sweep.search_space.items()
    }
    study.enqueue_trial(
        fixed_parameters,
        user_attrs={"resume_campaign_trial": 7},
    )
    interrupted = study.ask()
    for name, specification in sweep.search_space.items():
        interrupted.suggest_float(
            name,
            specification["low"],
            specification["high"],
            log=specification.get("log", False),
        )
    calls = []

    def runner(config_path: Path, *, run_id: str):
        calls.append(run_id)
        return _state(config_path, run_id, index=len(calls))

    result = run_optuna_sweep(
        CONTRACT,
        artifact_root=study_root,
        config_root=tmp_path / "config",
        n_trials=3,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        config_validator=lambda path: None,
        code_commit="test-commit",
    )

    assert len(calls) == 3
    assert calls[0].endswith("trial-007")
    assert result.interrupted_trials == 1
    resumed = optuna.load_study(study_name=sweep.name, storage=storage)
    assert all(
        trial.state is not optuna.trial.TrialState.RUNNING
        for trial in resumed.trials
    )
    assert len(resumed.trials) == 4
