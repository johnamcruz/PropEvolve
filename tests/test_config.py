from __future__ import annotations

import json
from pathlib import Path

import pytest

from propevolve.config import load_experiment_config


def test_config_locks_training_only_markets_out_of_deployment(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["tickers"] = ["NQ", "CL"]
    payload["deployment_tickers"] = ["NQ"]
    payload["training_only_tickers"] = ["CL"]
    payload["point_values"] = {"NQ": 20, "CL": 1000}
    payload["round_trip_fees"] = {"NQ": 3.84, "CL": 4.02}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["tickers"] == ("NQ", "CL")
    assert config["deployment_tickers"] == ("NQ",)


def test_config_rejects_hidden_agent_default(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    del payload["agent"]["target_sync_updates"]
    path = tmp_path / "missing-agent-setting.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="agent recipe is missing"):
        load_experiment_config(path)


def test_config_rejects_training_only_market_in_deployment(tmp_path: Path) -> None:
    source = Path("config/historical_mask_v1.json")
    payload = json.loads(source.read_text())
    payload["deployment_tickers"].append("CL")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="training-only"):
        load_experiment_config(path)


def test_config_rejects_revision_allowlist_that_overlaps_frozen_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["evolution"]["allowed_revision_paths"].append("temporal.train_start")
    path = tmp_path / "invalid-evolution.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="overlaps"):
        load_experiment_config(path)
