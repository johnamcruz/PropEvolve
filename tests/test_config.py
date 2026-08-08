from __future__ import annotations

import json
from pathlib import Path

import pytest

from propevolve.config import load_experiment_config


def test_config_locks_training_only_markets_out_of_deployment(tmp_path: Path) -> None:
    payload = {
        "schema": "propevolve_historical_training_v1",
        "assets": "assets.json",
        "tickers": ["NQ", "CL"],
        "deployment_tickers": ["NQ"],
        "training_only_tickers": ["CL"],
        "timeframe_minutes": 3,
        "cache_root": "cache",
        "cache": {},
        "challenge": {},
        "point_values": {"NQ": 20, "CL": 1000},
        "round_trip_fees": {"NQ": 3.78, "CL": 4.02},
        "temporal": {
            "train_start": "2021-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01"
        },
        "agent": {},
        "training": {},
        "evolution": {
            "hypothesis": "baseline",
            "allowed_revision_paths": ["agent.hidden_dim"],
            "frozen_paths": ["temporal", "challenge"],
        },
        "campaign": {
            "state_root": "runs/test/ml-loop-state",
            "max_revisions_per_stage": 2,
            "reasoning": {"provider": "manual"},
            "selection_requirements": [
                {"metric": "selection.pass_minus_blow", "operator": ">", "value": 0}
            ],
            "niches": [
                {"name": "pass", "metric": "selection.pass_rate", "maximize": True}
            ],
        },
        "output": "runs/test"
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["tickers"] == ("NQ", "CL")
    assert config["deployment_tickers"] == ("NQ",)


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
