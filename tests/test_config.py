from __future__ import annotations

import json
from pathlib import Path

import pytest

from propevolve.config import load_experiment_config


def test_ratchet_experiment_recipe_is_complete_and_frozen() -> None:
    config = load_experiment_config("config/historical_mask_ratchet_v1.json")

    assert config["challenge"]["per_trade_risk_dollars"] == 200.0
    assert config["challenge"]["ratchet_activation_r"] == 2.0
    assert config["challenge"]["ratchet_giveback_r"] == 0.5
    assert config["challenge"]["daily_profit_lock_dollars"] == 3_000.0
    assert "challenge" in config["evolution"]["frozen_paths"]
    assert config["training"]["minimum_environment_steps"] == 5_000_000


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


def test_config_preserves_declared_trade_risk_and_ratchet_fields(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["challenge"].update({
        "per_trade_risk_dollars": 200.0,
        "ratchet_activation_r": 2.0,
        "ratchet_giveback_r": 0.5,
        "daily_profit_lock_dollars": 3_000.0,
    })
    path = tmp_path / "ratchet.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["challenge"]["per_trade_risk_dollars"] == 200.0
    assert config["challenge"]["ratchet_activation_r"] == 2.0
    assert config["challenge"]["ratchet_giveback_r"] == 0.5
    assert config["challenge"]["daily_profit_lock_dollars"] == 3_000.0


def test_config_rejects_partial_ratchet_contract(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["challenge"]["per_trade_risk_dollars"] = 200.0
    path = tmp_path / "partial-ratchet.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="declared together"):
        load_experiment_config(path)
