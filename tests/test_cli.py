from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

from ml_training_loop import Phase, RunState
from ml_training_loop.stores import JsonRunStore
import propevolve.orchestration
import propevolve.teachers.expansion
import propevolve.teachers.regime
import propevolve.teachers.trend
from propevolve.cli import main
from tests.recipe_fixtures import retained_sweep_recipe, stage2_recipe


_STAGE2_RECIPE = stage2_recipe(19, contains="paired_aplus_contrastive.json")
_SWEEP_RECIPE = retained_sweep_recipe()


def test_optuna_sweep_command_dispatches_constrained_tpe(
    monkeypatch,
    capsys,
) -> None:
    import propevolve.optuna_sweep

    calls = []

    def fake_sweep(config_path):
        calls.append(config_path)
        return SimpleNamespace(
            status="COMPLETE",
            best_trial_number=7,
            study_path=Path("runs/study/study.db"),
            result_path=Path("runs/study/study.result.json"),
        )

    monkeypatch.setattr(propevolve.optuna_sweep, "run_optuna_sweep", fake_sweep)

    code = main([
        "optuna-sweep",
        "--config",
        str(_SWEEP_RECIPE),
    ])

    assert code == 0
    assert calls == [str(_SWEEP_RECIPE)]
    assert json.loads(capsys.readouterr().out) == {
        "best_trial_number": 7,
        "result": "runs/study/study.result.json",
        "status": "COMPLETE",
        "study": "runs/study/study.db",
    }


def test_setup_assets_command_creates_links_without_copying(tmp_path: Path) -> None:
    data = tmp_path / "source-data"
    data.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text("{}")
    workspace = tmp_path / "workspace"

    code = main([
        "setup-assets",
        "--workspace", str(workspace),
        "--market-data", str(data),
        "--checkpoint", str(checkpoint),
    ])

    assert code == 0
    assert (workspace / "data/ohlcv").is_symlink()
    assert (workspace / "checkpoints/chronos2_mask_full").is_symlink()


def test_validate_config_command_accepts_promoted_recipe() -> None:
    assert main([
        "validate-config", "--config",
        str(_STAGE2_RECIPE),
    ]) == 0


def test_evolve_status_reads_durable_state_without_running_training(
    tmp_path: Path,
    capsys,
) -> None:
    payload = json.loads(_STAGE2_RECIPE.read_text())
    payload["campaign"]["state_root"] = "runs/status-test/ml-loop-state"
    config_path = tmp_path / "experiment.json"
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_source.read_bytes())
    config_path.write_text(json.dumps(payload))
    store = JsonRunStore(tmp_path / "runs/status-test/ml-loop-state")
    store.save(RunState("status-run", "plan", Phase.NEEDS_REASONING))

    code = main([
        "evolve-status",
        "--config", str(config_path),
        "--run-id", "status-run",
    ])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "NEEDS_REASONING"


def test_evolve_command_dispatches_the_shared_training_loop(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    def fake_campaign(config_path, *, run_id, recover_reasoning=False):
        calls.append((config_path, run_id, recover_reasoning))
        return RunState(run_id, "plan", Phase.COMPLETE)

    monkeypatch.setattr(
        propevolve.orchestration,
        "run_evolution_campaign",
        fake_campaign,
    )

    code = main([
        "evolve",
        "--config", str(_STAGE2_RECIPE),
        "--run-id", "fake-e2e",
    ])

    assert code == 0
    assert calls == [(
        str(_STAGE2_RECIPE),
        "fake-e2e",
        False,
    )]
    assert json.loads(capsys.readouterr().out)["phase"] == "COMPLETE"


def test_evolve_command_auto_versions_repeated_frozen_config_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "any-config-name.json"
    config = {
        "_root": str(tmp_path),
        "output": "runs/auto-version-test",
        "campaign": {
            "state_root": "runs/auto-version-test/ml-loop-state",
        },
    }
    monkeypatch.setattr(
        "propevolve.cli.load_experiment_config",
        lambda path: config,
    )
    state_root = tmp_path / str(config["campaign"]["state_root"])
    (state_root / "legacy-manual-r2-20260818").mkdir(parents=True)
    (state_root / "legacy-manual-r6-20260819").mkdir()
    calls = []

    def fake_campaign(config_path, *, run_id, recover_reasoning=False):
        calls.append((config_path, run_id, recover_reasoning))
        return RunState(run_id, "plan", Phase.COMPLETE)

    monkeypatch.setattr(
        propevolve.orchestration,
        "run_evolution_campaign",
        fake_campaign,
    )

    assert main(["evolve", "--config", str(config_path)]) == 0
    assert main(["evolve", "--config", str(config_path)]) == 0

    assert calls == [
        (str(config_path), "auto-version-test-r7", False),
        (str(config_path), "auto-version-test-r8", False),
    ]
    assert (state_root / "auto-version-test-r7").is_dir()
    assert (state_root / "auto-version-test-r8").is_dir()


def test_evolve_command_can_recover_the_last_reasoning_checkpoint(
    monkeypatch,
) -> None:
    calls = []

    def fake_campaign(config_path, *, run_id, recover_reasoning=False):
        calls.append((config_path, run_id, recover_reasoning))
        return RunState(run_id, "plan", Phase.COMPLETE)

    monkeypatch.setattr(
        propevolve.orchestration,
        "run_evolution_campaign",
        fake_campaign,
    )

    assert main([
        "evolve",
        "--config", str(_STAGE2_RECIPE),
        "--run-id", "recover-e2e",
        "--recover-reasoning",
    ]) == 0
    assert calls == [(
        str(_STAGE2_RECIPE),
        "recover-e2e",
        True,
    )]


def test_expansion_teacher_cache_command_dispatches_requested_tickers(
    monkeypatch,
) -> None:
    calls = []

    def fake_builder(config_path, *, requested_tickers):
        calls.append((config_path, requested_tickers))
        return ()

    monkeypatch.setattr(
        propevolve.teachers.expansion,
        "build_expansion_teacher_caches",
        fake_builder,
    )

    code = main([
        "build-expansion-teacher-cache",
        "--config", "config/expansion_teacher_cache_v1.json",
        "--ticker", "NQ",
        "--ticker", "ES",
    ])

    assert code == 0
    assert calls == [
        ("config/expansion_teacher_cache_v1.json", ("NQ", "ES"))
    ]


def test_regime_teacher_cache_command_dispatches_requested_tickers(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        propevolve.teachers.regime,
        "build_regime_teacher_caches",
        lambda config, *, requested_tickers: calls.append(
            (config, requested_tickers)
        ),
    )

    result = main([
        "build-regime-teacher-cache",
        "--config", "config/regime_teacher_cache_v1.json",
        "--ticker", "NQ",
        "--ticker", "ES",
    ])

    assert result == 0
    assert calls == [("config/regime_teacher_cache_v1.json", ("NQ", "ES"))]


def test_trend_teacher_cache_command_dispatches_requested_tickers(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        propevolve.teachers.trend,
        "build_trend_teacher_caches",
        lambda config, *, requested_tickers: calls.append(
            (config, requested_tickers)
        ),
    )

    result = main([
        "build-trend-teacher-cache",
        "--config", "config/trend_teacher_cache_v1.json",
        "--ticker", "NQ",
        "--ticker", "ES",
    ])

    assert result == 0
    assert calls == [("config/trend_teacher_cache_v1.json", ("NQ", "ES"))]
