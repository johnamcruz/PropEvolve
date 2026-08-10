from __future__ import annotations

from pathlib import Path
import json

from ml_training_loop import Phase, RunState
from ml_training_loop.stores import JsonRunStore
import propevolve.orchestration
import propevolve.expansion_teacher
from propevolve.cli import main


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
    assert main(["validate-config", "--config", "config/historical_mask_v1.json"]) == 0


def test_evolve_status_reads_durable_state_without_running_training(
    tmp_path: Path,
    capsys,
) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["campaign"]["state_root"] = "runs/status-test/ml-loop-state"
    config_path = tmp_path / "experiment.json"
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
        "--config", "config/historical_mask_v1.json",
        "--run-id", "fake-e2e",
    ])

    assert code == 0
    assert calls == [("config/historical_mask_v1.json", "fake-e2e", False)]
    assert json.loads(capsys.readouterr().out)["phase"] == "COMPLETE"


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
        "--config", "config/historical_mask_v1.json",
        "--run-id", "recover-e2e",
        "--recover-reasoning",
    ]) == 0
    assert calls == [("config/historical_mask_v1.json", "recover-e2e", True)]


def test_expansion_teacher_cache_command_dispatches_requested_tickers(
    monkeypatch,
) -> None:
    calls = []

    def fake_builder(config_path, *, requested_tickers):
        calls.append((config_path, requested_tickers))
        return ()

    monkeypatch.setattr(
        propevolve.expansion_teacher,
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
