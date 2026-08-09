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
    assert "challenge.profit_target" in config["evolution"]["frozen_paths"]
    assert "challenge.per_trade_risk_dollars" not in config["evolution"]["frozen_paths"]
    assert config["training"]["terminal_sequence_fraction"] == 0.0
    assert config["evolution"]["revision_bounds"]["challenge.per_trade_risk_dollars"] == {
        "minimum": 100.0,
        "maximum": 500.0,
    }
    assert config["training"]["minimum_environment_steps"] == 5_000_000
    assert config["training"]["validation_episodes"] == 200
    assert config["campaign"]["selection_requirements"] == [
        {"metric": "selection.pass_rate", "operator": ">=", "value": 0.5},
        {"metric": "selection.blow_rate", "operator": "==", "value": 0.0},
    ]
    assert config["campaign"]["diagnostic_targets"] == [
        {"metric": "selection.trade_win_rate", "operator": ">=", "value": 0.4},
        {"metric": "selection.average_win_r", "operator": ">=", "value": 2.0},
    ]


def test_safety_replay_recipe_exposes_only_bounded_training_reward_revisions() -> None:
    config = load_experiment_config("config/historical_mask_safety_replay_v1.json")

    assert config["campaign"]["reasoning"]["proposer"] == "standard"
    assert config["training"]["terminal_sequence_fraction"] == 0.5
    assert config["challenge"]["mll_proximity_penalty_coefficient"] == 0.0001
    for path in (
        "challenge.mll_proximity_penalty_coefficient",
        "challenge.lead_giveback_penalty_coefficient",
        "challenge.large_win_bonus_coefficient",
        "challenge.terminal_pass_reward",
        "challenge.terminal_blow_reward",
        "training.terminal_sequence_fraction",
    ):
        assert path in config["evolution"]["allowed_revision_paths"]
        assert path in config["evolution"]["revision_bounds"]
        assert path not in config["evolution"]["frozen_paths"]
    for path in (
        "challenge.profit_target",
        "challenge.max_loss",
        "temporal",
        "point_values",
        "round_trip_fees",
        "training.minimum_environment_steps",
    ):
        assert path in config["evolution"]["frozen_paths"]


def test_config_accepts_optional_gepa_reflective_reasoning_proposer(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["campaign"]["reasoning"]["proposer"] = "gepa_reflective"
    path = tmp_path / "gepa-reflective.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["campaign"]["reasoning"]["provider"] == "codex"
    assert config["campaign"]["reasoning"]["proposer"] == "gepa_reflective"


def test_config_rejects_unknown_reasoning_proposer(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["campaign"]["reasoning"]["proposer"] = "unbounded_search"
    path = tmp_path / "invalid-proposer.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="reasoning proposer"):
        load_experiment_config(path)


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
    })
    path = tmp_path / "ratchet.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["challenge"]["per_trade_risk_dollars"] == 200.0
    assert config["challenge"]["ratchet_activation_r"] == 2.0
    assert config["challenge"]["ratchet_giveback_r"] == 0.5


def test_config_rejects_partial_ratchet_contract(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["challenge"]["per_trade_risk_dollars"] = 200.0
    path = tmp_path / "partial-ratchet.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="declared together"):
        load_experiment_config(path)
