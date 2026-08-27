from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace

from ml_training_loop import Phase, RunState, StageReceipt
import pytest

from propevolve.orchestration import _plan
import propevolve.optuna_engine as optuna_engine
from propevolve.optuna_sweep import load_optuna_sweep, run_optuna_sweep
from tests.recipe_fixtures import active_sweep_recipe


ACTIVE_CONTRACT = active_sweep_recipe().resolve()
_ACTIVE_PAYLOAD = json.loads(ACTIVE_CONTRACT.read_text())
BASE_CONFIG = (
    ACTIVE_CONTRACT.parent / _ACTIVE_PAYLOAD["base_config"]
).resolve()
STAGE = str(_ACTIVE_PAYLOAD["stages"]["screening"]["name"])


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
                        "name": "v31_control",
                        "values": {
                            "training.terminal_sequence_fraction": 0.50,
                            "training.safety_sequence_fraction": 0.25,
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
                "training_episodes": 100,
                "validation_episodes": 100,
            },
            "confirmation": {
                "name": STAGE,
                "training_episodes": 200,
                "validation_episodes": 200,
            },
            "multi_seed": {
                "name": STAGE,
                "training_episodes": 200,
                "validation_episodes": 200,
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
    assert sweep.stages["screening"].training_episodes == 100
    assert sweep.stages["confirmation"].validation_episodes == 200
    assert sweep.promotion.seeds == (11, 22)
    assert "agent.learning_rate" in sweep.frozen_paths


def test_v2_runs_three_screening_campaigns_in_isolated_parallel_slots(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(3, timeout=5.0)
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def runner(config_path: Path, *, run_id: str):
        nonlocal active, peak_active
        if run_id.endswith("trial-000"):
            return _state(config_path, run_id, _metrics())
        with lock:
            active += 1
            peak_active = max(peak_active, active)
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


def test_v2_default_worker_is_an_isolated_capped_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected_state = object()

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(optuna_engine.subprocess, "run", fake_run)
    monkeypatch.setattr(
        optuna_engine,
        "_default_state_loader",
        lambda config_path, run_id: expected_state,
    )
    config_path = tmp_path / "trial.json"
    config_path.write_text("{}")

    actual_state = optuna_engine._default_runner(
        config_path,
        run_id="isolated-trial-001",
        stdout_path=tmp_path / "trial.stdout.log",
        stderr_path=tmp_path / "trial.stderr.log",
    )

    assert actual_state is expected_state
    assert captured["command"][1:7] == [
        "-u",
        "-m",
        "propevolve.cli",
        "evolve",
        "--config",
        str(config_path),
    ]
    assert captured["command"][-2:] == ["--run-id", "isolated-trial-001"]
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
    assert stage["training_episodes"] == 100
    assert stage["validation_episodes"] == 100
    assert config["training"]["terminal_sequence_fraction"] == 0.50
    assert config["training"]["safety_sequence_fraction"] == 0.25
    assert config["training"]["entry_opportunity_sequence_fraction"] == 0.25
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
        return _state(config_path, run_id, _metrics())

    result = run_optuna_sweep(
        ACTIVE_CONTRACT,
        artifact_root=tmp_path / "study",
        config_root=tmp_path / "configs",
        n_trials=1,
        runner=runner,
        state_loader=lambda config_path, run_id: None,
        code_commit="test-commit",
    )

    assert sweep.n_trials == 24
    assert sweep.n_jobs == 3
    assert result.status == "COMPLETE"


def test_active_sweep_freezes_verified_learning_mechanics() -> None:
    sweep = load_optuna_sweep(ACTIVE_CONTRACT)

    required_frozen_paths = {
        # Authenticated causal inputs and labels, including the five-bar entry
        # window and next-bar execution contract.
        "teachers",
        "observation",
        "entry_supervision",
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
        "sealed_confirmation",
    }

    assert _ACTIVE_PAYLOAD["frozen"]["teacher_free_selection"] is True
    assert required_frozen_paths <= set(sweep.frozen_paths)


def test_v2_runs_top_k_confirmation_then_multiseed_winner_retrain(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, int, int]] = []

    def runner(config_path: Path, *, run_id: str):
        config = json.loads(config_path.read_text())
        stage, = config["campaign"]["budget_stages"]
        seed = int(config["training"]["seed"])
        calls.append((
            run_id,
            int(stage["training_episodes"]),
            int(stage["validation_episodes"]),
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
    assert [item[1:3] for item in calls[:3]] == [(100, 100)] * 3
    assert [item[1:3] for item in calls[3:5]] == [(200, 200)] * 2
    assert [item[1:3] for item in calls[5:]] == [(200, 200)] * 2
    assert [item[3] for item in calls[5:]] == [11, 22]
    payload = json.loads(result.result_path.read_text())
    assert payload["promoted_trial_numbers"] == [1, 2]
    assert payload["winner_trial_number"] == 1
    assert payload["winner_feasible_seeds"] == 2
    assert payload["promotion_status"] == "COMPLETE"
