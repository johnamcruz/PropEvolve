from __future__ import annotations

import json
from pathlib import Path

import pytest

from propevolve.config import agent_runtime_settings, load_experiment_config


def test_runtime_performance_contract_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_curriculum_v8.json"
    ).read_text())
    payload["runtime"] = {
        "mixed_precision": "fp16",
        "compile_model": True,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "mps_prefer_metal": True,
        "mps_fast_math": False,
        "benchmark_max_relative_loss_drift": 0.05,
    }
    payload["training"]["prefetch_batches"] = 1
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["runtime"]["mixed_precision"] == "fp16"
    assert config["runtime"]["mps_prefer_metal"] is True
    assert config["training"]["prefetch_batches"] == 1
    assert set(agent_runtime_settings(config["runtime"])) == {
        "mixed_precision",
        "compile_model",
        "compile_backend",
        "compile_mode",
        "mps_prefer_metal",
        "mps_fast_math",
    }

    payload["runtime"]["mixed_precision"] = "fp8"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mixed precision"):
        load_experiment_config(path)

    payload["runtime"]["mixed_precision"] = "fp16"
    payload["training"]["prefetch_batches"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prefetch"):
        load_experiment_config(path)


def test_teacher_curriculum_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    source = Path("config/historical_mask_expansion_regime_curriculum_v8.json")
    payload = json.loads(source.read_text())
    path = tmp_path / "teacher-curriculum.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["training"]["teacher_loss_end_scale"] == 0.1
    assert config["training"]["teacher_guidance_dropout_start"] == 0.0
    assert config["training"]["teacher_guidance_dropout_end"] == 0.5

    payload["training"]["teacher_guidance_dropout_start"] = 0.75
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="teacher guidance dropout"):
        load_experiment_config(path)


def test_legacy_schema_v1_recipe_keeps_eager_fp32_runtime(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload.pop("runtime")
    payload["training"].pop("prefetch_batches")
    path = tmp_path / "legacy-v1.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["runtime"] == {
        "mixed_precision": "off",
        "compile_model": False,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "mps_prefer_metal": False,
        "mps_fast_math": False,
        "benchmark_max_relative_loss_drift": 0.05,
    }
    assert config["training"]["prefetch_batches"] == 0


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


def test_winner_retention_recipe_enables_economic_shaping_and_near_blow_gate() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_teacher_winner_retention_v5.json"
    )

    assert config["challenge"]["lead_giveback_penalty_coefficient"] > 0
    assert config["challenge"]["large_win_bonus_coefficient"] > 0
    assert config["challenge"]["minimum_mll_headroom"] == 500
    assert config["campaign"]["near_blow_loss_fraction"] == 0.75
    screen = config["campaign"]["budget_stages"][0]
    assert {
        "metric": "selection.near_blow_timeout_rate",
        "operator": "<=",
        "value": 0.05,
    } in screen["selection_requirements"]
    assert screen["parent_improvement_requirements"] == [{
        "metric": "selection.two_r_mfe_capture_ratio",
        "direction": "maximize",
        "minimum_delta": 0.0,
    }]
    assert [
        rule["metric"]
        for rule in config["campaign"]["finalization"]["ranking"]
    ] == [
        "selection.blow_rate",
        "selection.near_blow_timeout_rate",
        "selection.two_r_mfe_capture_ratio",
        "selection.pass_rate",
        "selection.expectancy_r",
    ]
    assert "campaign.near_blow_loss_fraction" in config["evolution"]["frozen_paths"]


def test_curriculum_recipe_teaches_four_priorities_by_warm_started_stage() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_teacher_curriculum_v6.json"
    )

    stages = config["campaign"]["budget_stages"]
    assert [stage["name"] for stage in stages] == [
        "safety_foundation_1m",
        "winner_retention_1m",
        "challenge_completion_2m",
        "confirmation_5m_multiseed",
    ]
    assert stages[0]["curriculum_override"] == {
        "challenge.lead_giveback_penalty_coefficient": 0,
        "challenge.large_win_bonus_coefficient": 0,
    }
    assert "challenge.large_win_bonus_coefficient" not in stages[0][
        "revision_paths"
    ]
    assert "challenge.lead_giveback_penalty_coefficient" not in stages[0][
        "revision_paths"
    ]
    assert stages[1]["curriculum_override"] == {
        "agent.learning_rate": 0.000075,
        "challenge.lead_giveback_penalty_coefficient": 0,
        "challenge.large_win_bonus_coefficient": 0.1,
    }
    assert "challenge.large_win_bonus_coefficient" in stages[1]["revision_paths"]
    assert "challenge.lead_giveback_penalty_coefficient" not in stages[1][
        "revision_paths"
    ]
    assert stages[2]["curriculum_override"] == {
        "agent.learning_rate": 0.00005,
        "challenge.lead_giveback_penalty_coefficient": 0.001,
    }
    assert "challenge.lead_giveback_penalty_coefficient" in stages[2][
        "revision_paths"
    ]
    assert config["challenge"]["mll_proximity_penalty_coefficient"] > 0
    assert config["evolution"]["revision_bounds"][
        "challenge.mll_proximity_penalty_coefficient"
    ]["minimum"] > 0
    assert config["training"]["safety_sequence_fraction"] == 0.25
    assert "training.safety_sequence_fraction" in stages[0]["revision_paths"]


def test_expansion_entry_search_recipe_is_training_only_and_matched() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_entry_search_curriculum_v7.json"
    )

    assert config["teacher"]["entry_search_loss_weight"] == 0.3
    assert config["training"]["entry_opportunity_sequence_fraction"] == 0.25
    assert "teacher" in config["evolution"]["frozen_paths"]
    assert "training.entry_opportunity_sequence_fraction" in config["evolution"][
        "frozen_paths"
    ]
    assert config["output"].endswith("historical_mask_expansion_entry_search_curriculum_v7")


def test_config_accepts_three_frozen_training_only_teachers(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_entry_search_curriculum_v7.json"
    ).read_text())
    expansion = payload.pop("teacher")
    payload["teachers"] = [
        expansion,
        {
            "kind": "regime",
            "cache_root": "cache/regime_teacher_9market_3min_pre2025_v1",
            "channels": [
                "structure_chop_probability",
                "structure_neutral_probability",
                "structure_trend_probability",
                "structure_chop_persistence_probability",
                "structure_trend_onset_probability",
                "structure_trend_persistence_probability",
                "structure_trend_weakening_probability",
                "structure_other_transition_probability",
                "kaufman_efficiency",
                "volatility_low_probability",
                "volatility_normal_probability",
                "volatility_high_probability",
                "volatility_low_persistence_probability",
                "volatility_expansion_onset_probability",
                "volatility_high_persistence_probability",
                "volatility_contraction_probability",
                "volatility_other_transition_probability",
                "volatility_percentile",
            ],
            "loss_weight": 0.1,
            "entry_search_loss_weight": 0.0,
        },
        {
            "kind": "trend",
            "cache_root": "cache/trend_teacher_9market_3min_pre2025_v1",
            "channels": [
                "long_launch_probability",
                "short_launch_probability",
                "long_conditional_quality",
                "short_conditional_quality",
            ],
            "loss_weight": 0.1,
            "entry_search_loss_weight": 0.0,
        },
    ]
    payload["evolution"]["frozen_paths"].remove("teacher")
    payload["evolution"]["frozen_paths"].append("teachers")
    path = tmp_path / "combined-teachers.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert [teacher["kind"] for teacher in config["teachers"]] == [
        "expansion", "regime", "trend"
    ]
    assert config.get("teacher") is None

    payload["teachers"][1]["channels"] = ["wrong"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Regime teacher contract"):
        load_experiment_config(path)


def test_expansion_regime_curriculum_is_teacher_free_at_selection() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_regime_curriculum_v8.json"
    )

    assert [teacher["kind"] for teacher in config["teachers"]] == [
        "expansion", "regime"
    ]
    assert config["teachers"][0]["entry_search_loss_weight"] == 0.3
    assert config["teachers"][1]["entry_search_loss_weight"] == 0.0
    assert "teachers" in config["evolution"]["frozen_paths"]
    assert config["sealed_confirmation"]["teacher_free"] is True


def test_expansion_regime_trend_curriculum_is_teacher_free_at_selection() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_curriculum_v9.json"
    )

    assert [teacher["kind"] for teacher in config["teachers"]] == [
        "expansion", "regime", "trend"
    ]
    assert config["teachers"][0]["entry_search_loss_weight"] == 0.3
    assert all(
        teacher["entry_search_loss_weight"] == 0.0
        for teacher in config["teachers"][1:]
    )
    assert "teachers" in config["evolution"]["frozen_paths"]
    assert config["sealed_confirmation"]["teacher_free"] is True


def test_expansion_entry_recipe_freezes_teacher_free_2026_confirmation() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_entry_search_curriculum_v7.json"
    )

    confirmation = config["sealed_confirmation"]
    assert confirmation["start"] == "2026-01-01"
    assert confirmation["end"] == "2027-01-01"
    assert confirmation["episode_sessions"] == 30
    assert confirmation["window_mode"] == "non_overlapping"
    assert confirmation["teacher_free"] is True
    assert confirmation["tickers"] == config["tickers"]
    assert confirmation["minimum_pass_rate"] == 0.5
    assert confirmation["maximum_blow_rate"] == 0.0
    assert "sealed_confirmation" in config["evolution"]["frozen_paths"]


def test_config_rejects_sealed_confirmation_that_can_use_teachers(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        Path(
            "config/historical_mask_expansion_entry_search_curriculum_v7.json"
        ).read_text()
    )
    payload["sealed_confirmation"]["teacher_free"] = False
    path = tmp_path / "teacher-leak.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="teacher-free"):
        load_experiment_config(path)


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
        "ratchet_lock_floor_r": 2.0,
    })
    path = tmp_path / "ratchet.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["challenge"]["per_trade_risk_dollars"] == 200.0
    assert config["challenge"]["ratchet_activation_r"] == 2.0
    assert config["challenge"]["ratchet_giveback_r"] == 0.5
    assert config["challenge"]["ratchet_lock_floor_r"] == 2.0


def test_config_rejects_partial_ratchet_contract(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["challenge"]["per_trade_risk_dollars"] = 200.0
    path = tmp_path / "partial-ratchet.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="declared together"):
        load_experiment_config(path)


def test_expansion_teacher_recipe_is_training_only_and_frozen() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_teacher_v1.json"
    )

    assert config["teacher"]["kind"] == "expansion"
    assert config["teacher"]["loss_weight"] == 0.2
    assert config["temporal"]["train_end"] == "2025-01-01"
    assert config["temporal"]["sealed_start"] == "2026-01-01"
    assert "teacher" in config["evolution"]["frozen_paths"]


def test_large_win_expansion_challenger_is_bounded_and_distinct() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_teacher_large_win_v1.json"
    )

    assert config["challenge"]["large_win_threshold_r"] == 2.0
    assert config["challenge"]["large_win_bonus_coefficient"] == 0.1
    assert config["training"]["minimum_environment_steps"] == 2_000_000
    assert config["output"] == "runs/historical_mask_expansion_teacher_large_win_v1"
    assert (
        config["campaign"]["state_root"]
        == "runs/historical_mask_expansion_teacher_large_win_v1/ml-loop-state"
    )
    assert config["temporal"] == {
        "train_start": "2021-01-01",
        "train_end": "2025-01-01",
        "validation_start": "2025-01-01",
        "validation_end": "2026-01-01",
        "sealed_start": "2026-01-01",
    }


def test_expansion_ratchet_floor_challenger_is_one_isolated_revision() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_teacher_ratchet_floor_v1.json"
    )

    assert config["challenge"]["ratchet_activation_r"] == 2.0
    assert config["challenge"]["ratchet_giveback_r"] == 0.5
    assert config["challenge"]["ratchet_lock_floor_r"] == 2.0
    assert config["challenge"]["large_win_bonus_coefficient"] == 0.0
    assert config["training"]["minimum_environment_steps"] == 2_000_000
    assert (
        config["output"]
        == "runs/historical_mask_expansion_teacher_ratchet_floor_v1"
    )


def test_management_exploration_challenger_preserves_the_matched_contract() -> None:
    baseline = load_experiment_config(
        "config/historical_mask_expansion_teacher_ratchet_floor_diagnostics_v2.json"
    )
    challenger = load_experiment_config(
        "config/historical_mask_expansion_teacher_management_exploration_v3.json"
    )

    for section in ("cache", "teacher", "challenge", "temporal", "agent"):
        assert challenger[section] == baseline[section]
    assert challenger["training"] == {
        **baseline["training"],
        "management_epsilon_start": 0.05,
        "management_epsilon_end": 0.01,
    }
    assert challenger["output"].endswith("management_exploration_v3")
    assert "training.management_epsilon_start" in challenger["evolution"][
        "allowed_revision_paths"
    ]
    assert "training.management_epsilon_end" in challenger["evolution"][
        "allowed_revision_paths"
    ]


def test_staged_budget_recipe_screens_confirms_then_freezes_eight_final_seeds() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_teacher_staged_budget_v4.json"
    )
    stages = config["campaign"]["budget_stages"]

    assert [stage["minimum_environment_steps"] for stage in stages] == [
        1_000_000,
        2_000_000,
        5_000_000,
    ]
    assert stages[-1]["seeds"] == (
        11111,
        22222,
        33333,
        44444,
        55555,
        66666,
        77777,
        88888,
    )
    assert stages[-1]["max_parallel"] == 3
    assert stages[-1]["allow_revisions"] is False
    assert config["campaign"]["finalization"]["minimum_seed_count"] == 8
    assert stages[-1]["selection_requirements"] == [
        {"metric": "selection.pass_rate", "operator": ">=", "value": 0.5},
        {"metric": "selection.blow_rate", "operator": "==", "value": 0},
        {"metric": "selection.trade_win_rate", "operator": ">=", "value": 0.4},
        {"metric": "selection.average_win_r", "operator": ">=", "value": 2.0},
    ]
