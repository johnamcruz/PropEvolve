from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from propevolve.config import (
    REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS,
    REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS,
    agent_runtime_settings,
    load_experiment_config,
)
from propevolve.balance_aware_regime_selectivity import (
    ACTION_ORDER as REGIME_SELECTIVITY_ACTION_ORDER,
    FORMULA as REGIME_SELECTIVITY_FORMULA,
    SCHEMA as REGIME_SELECTIVITY_SCHEMA,
    TARGET_SOURCE as REGIME_SELECTIVITY_TARGET_SOURCE,
)


def _stage2a_config(tmp_path: Path) -> Path:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_balanced_v8b.json"
    ).read_text())
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt = receipt_source.read_bytes()
    receipt_path = tmp_path / "config" / "receipts" / "centers.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt)
    receipt_payload = json.loads(receipt)
    payload["regime_selectivity"] = {
        "schema": REGIME_SELECTIVITY_SCHEMA,
        "training_only": True,
        "target_source": REGIME_SELECTIVITY_TARGET_SOURCE,
        "action_order": list(REGIME_SELECTIVITY_ACTION_ORDER),
        "formula": REGIME_SELECTIVITY_FORMULA,
        "semantics": "static_state_v1",
        "persistent_chop_negative_emphasis": 0.0,
        "side_balance": {
            "schema": "equal_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
        "loss_weight": 0.3,
        "expansion_center_receipt": "config/receipts/centers.json",
        "expansion_center_receipt_sha256": hashlib.sha256(receipt).hexdigest(),
        "expansion_long_center": receipt_payload["long_center"],
        "expansion_short_center": receipt_payload["short_center"],
        "probability_epsilon": 1e-6,
        "headroom_pressure": 1.0,
        "dominant_chop_pressure": 2.0,
        "q_temperature": 1.0,
    }
    payload["evolution"]["frozen_paths"].extend(
        REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS
    )
    path = tmp_path / "config" / "stage2a.json"
    path.write_text(json.dumps(payload))
    return path


def test_stage2a_regime_selectivity_is_authenticated_and_frozen(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)

    config = load_experiment_config(path)

    assert config["regime_selectivity"]["expansion_long_center"] == pytest.approx(
        0.10249102659218842
    )
    assert config["regime_selectivity"]["expansion_short_center"] == pytest.approx(
        0.10399580328775007
    )
    assert set(REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS) <= set(
        config["evolution"]["frozen_paths"]
    )
    assert config["regime_selectivity"]["side_balance"] == {
        "schema": "equal_long_short_v1",
        "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
    }

    payload = json.loads(path.read_text())
    payload["regime_selectivity"]["expansion_long_center"] = 0.5
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="receipt contract drifted"):
        load_experiment_config(path)


def test_stage2a_requires_every_selectivity_identity_field_to_be_frozen(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["evolution"]["frozen_paths"].remove(
        "regime_selectivity.expansion_center_receipt_sha256"
    )
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="identity must be frozen"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "field",
    (
        "regime_selectivity.headroom_pressure",
        "regime_selectivity.dominant_chop_pressure",
    ),
)
def test_stage2a_requires_static_pressure_identity_to_be_frozen(
    tmp_path: Path,
    field: str,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    frozen_paths = payload["evolution"]["frozen_paths"]
    if field in frozen_paths:
        frozen_paths.remove(field)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="identity must be frozen"):
        load_experiment_config(path)


def test_stage2a_transition_semantics_switch_is_atomic_and_stage_bound(
    tmp_path: Path,
) -> None:
    from propevolve.balance_aware_regime_selectivity import (
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    )

    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    for field in REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS:
        payload["evolution"]["frozen_paths"].remove(field)
        payload["evolution"]["allowed_revision_paths"].append(field)
    payload["evolution"]["revision_bounds"][
        "regime_selectivity.persistent_chop_negative_emphasis"
    ] = {"minimum": 0.25, "maximum": 2.0}
    transition = payload["campaign"]["budget_stages"][1]
    transition["curriculum_override"] = {
        "regime_selectivity.formula": PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        "regime_selectivity.semantics": (
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
        "regime_selectivity.persistent_chop_negative_emphasis": 1.0,
    }
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)
    assert config["campaign"]["budget_stages"][1][
        "curriculum_override"
    ] == transition["curriculum_override"]

    del payload["campaign"]["budget_stages"][1]["curriculum_override"][
        "regime_selectivity.formula"
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be atomic"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "field",
    (
        "loss_weight",
        "expansion_long_center",
        "expansion_short_center",
        "probability_epsilon",
        "headroom_pressure",
        "dominant_chop_pressure",
        "q_temperature",
    ),
)
def test_stage2a_rejects_nonfinite_selectivity_settings(
    tmp_path: Path,
    field: str,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"][field] = float("nan")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="selectivity contract is invalid"):
        load_experiment_config(path)


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


def test_n_step_return_is_explicit_and_must_fit_replay_sequence(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_curriculum_v8.json"
    ).read_text())
    payload["agent"]["n_step_return"] = 8
    payload["agent"]["recurrent_burn_in"] = 16
    path = tmp_path / "n-step.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["agent"]["n_step_return"] == 8
    assert config["agent"]["recurrent_burn_in"] == 16

    payload["agent"]["n_step_return"] = payload["training"]["sequence_length"] + 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="n-step return"):
        load_experiment_config(path)

    payload["agent"]["n_step_return"] = 8
    payload["agent"]["recurrent_burn_in"] = payload["training"]["sequence_length"] - 7
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="burn-in"):
        load_experiment_config(path)


def test_v8e_recipe_really_uses_the_declared_eight_step_return() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_profile_n_step_td_v8e.json"
    )

    assert config["agent"]["n_step_return"] == 8
    assert config["agent"]["recurrent_burn_in"] == 64
    assert config["agent"]["policy_retention_loss_weight"] == 10.0
    assert config["training"]["sequence_length"] == 96
    assert "agent.n_step_return" in config["evolution"]["frozen_paths"]
    assert "agent.recurrent_burn_in" in config["evolution"]["frozen_paths"]
    assert "agent.policy_retention_loss_weight" in config["evolution"][
        "frozen_paths"
    ]
    assert "training.sequence_length" in config["evolution"]["frozen_paths"]
    assert "training.recurrent_horizon" in config["evolution"]["frozen_paths"]
    assert "training.sequence_length" not in config["evolution"][
        "allowed_revision_paths"
    ]
    assert "training.recurrent_horizon" not in config["evolution"][
        "allowed_revision_paths"
    ]
    assert config["training"]["short_circuit"]["collapse"] == {
        "window_episodes": 5,
        "minimum_prior_passes": 2,
        "maximum_recent_passes": 0,
        "maximum_average_hold_bars": 4.0,
        "minimum_voluntary_close_rate": 0.8,
    }


def test_validation_no_trade_patience_is_json_configured_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_trend_profile_n_step_td_v8e.json"
    ).read_text())
    payload["training"]["validation_no_trade_patience_episodes"] = 5
    path = tmp_path / "validation-patience.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)
    assert config["training"]["validation_no_trade_patience_episodes"] == 5

    for invalid in (True, -1, payload["training"]["validation_episodes"] + 1):
        payload["training"]["validation_no_trade_patience_episodes"] = invalid
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="validation no-trade patience"):
            load_experiment_config(path)


def test_v8_uses_exact_post_launch_actions_for_entry_guidance() -> None:
    raw = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    ).read_text())
    assert "action_class_balance" not in raw["entry_supervision"]

    config = load_experiment_config(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    )
    expansion = config["teachers"][0]

    assert expansion["entry_search_loss_weight"] == 0.0
    assert expansion["entry_search_objective"] == "raw_probability"
    assert expansion["entry_search_long_center"] == 0.5
    assert expansion["entry_search_short_center"] == 0.5
    assert expansion["entry_search_center_receipt"] == ""
    assert expansion["entry_search_center_receipt_sha256"] == ""
    assert config["challenge"]["per_trade_risk_dollars"] == 300
    assert "challenge.per_trade_risk_dollars" in config["evolution"]["frozen_paths"]
    assert "challenge.per_trade_risk_dollars" not in config["evolution"]["allowed_revision_paths"]
    assert "challenge.per_trade_risk_dollars" not in config["evolution"]["revision_bounds"]
    assert all(
        "challenge.per_trade_risk_dollars" not in stage["revision_paths"]
        for stage in config["campaign"]["budget_stages"]
    )
    assert config["training"]["teacher_loss_end_scale"] == 0.0
    assert config["training"]["teacher_guidance_dropout_end"] == 1.0
    assert config["training"]["teacher_autonomy_start_fraction"] == 0.8
    assert config["training"]["validation_no_trade_patience_episodes"] == 5
    assert config["entry_supervision"] == {
        "schema": "post_launch_entry_v1",
        "training_only": True,
        "decision_count": 5,
        "fill_offsets": [1, 2, 3, 4, 5],
        "execution": "next_bar_open",
        "risk_dollars": 300,
        "launch": {
            "favorable_r": 0.5,
            "adverse_r": 0.25,
            "horizon_bars": 3,
        },
        "continuation": {
            "favorable_r": 0.5,
            "adverse_r": 0.25,
            "horizon_bars": 3,
        },
        "target_r": 2.0,
        "stop_r": 1.0,
        "horizon_bars": 150,
        "collision": "stop_first",
        "loss_weight": 0.3,
        "action_class_balance": None,
    }
    assert "entry_supervision" in config["evolution"]["frozen_paths"]


def test_v8b_is_a_fresh_inverse_frequency_balanced_entry_campaign() -> None:
    original = load_experiment_config(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    )
    balanced = load_experiment_config(
        "config/historical_mask_expansion_regime_post_launch_entry_balanced_v8b.json"
    )

    assert original["entry_supervision"]["action_class_balance"] is None
    assert balanced["entry_supervision"]["action_class_balance"] == {
        "schema": "inverse_frequency_v1",
        "action_order": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
    }
    assert balanced["output"] != original["output"]
    assert balanced["campaign"]["state_root"] != original["campaign"]["state_root"]
    assert "entry_supervision" in balanced["evolution"]["frozen_paths"]
    assert all(
        path != "entry_supervision.action_class_balance"
        and not path.startswith("entry_supervision.")
        for path in balanced["evolution"]["allowed_revision_paths"]
    )
    assert "entry_supervision.action_class_balance" not in balanced["evolution"][
        "revision_bounds"
    ]
    normalized_original = json.loads(json.dumps(original))
    normalized_balanced = json.loads(json.dumps(balanced))
    normalized_balanced["entry_supervision"]["action_class_balance"] = None
    normalized_balanced["evolution"]["hypothesis"] = normalized_original[
        "evolution"
    ]["hypothesis"]
    normalized_balanced["_path"] = normalized_original["_path"]
    for path in (
        ("output",),
        ("campaign", "state_root"),
        ("campaign", "finalization", "registry_root"),
        ("campaign", "finalization", "export_root"),
    ):
        left = normalized_original
        right = normalized_balanced
        for field in path[:-1]:
            left = left[field]
            right = right[field]
        right[path[-1]] = left[path[-1]]
    assert normalized_balanced == normalized_original


@pytest.mark.parametrize(
    "balance",
    (
        {
            "schema": "manual_v1",
            "action_order": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
        },
        {
            "schema": "inverse_frequency_v1",
            "action_order": ["WAIT", "ENTER_SHORT_1", "ENTER_LONG_1"],
        },
        {"schema": "inverse_frequency_v1"},
    ),
)
def test_entry_action_balance_contract_fails_closed(
    tmp_path: Path,
    balance: dict,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_balanced_v8b.json"
    ).read_text())
    payload["entry_supervision"]["action_class_balance"] = balance
    path = tmp_path / "invalid-entry-balance.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="class balance contract"):
        load_experiment_config(path)


def test_legacy_entry_action_recipes_keep_population_weighted_reduction() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_regime_post_launch_entry_balanced_v8b.json"
    )

    assert (
        config["agent"]["entry_action_loss_reduction"]
        == "population_weighted_mean_v1"
    )


def test_entry_action_loss_reduction_fails_closed_on_unknown_value(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_balanced_v8b.json"
    ).read_text())
    payload["agent"]["entry_action_loss_reduction"] = "unknown"
    path = tmp_path / "unknown-entry-action-reduction.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry action loss reduction is invalid"):
        load_experiment_config(path)


def test_stage2a_entry_balance_repair_is_one_matched_nonrevisable_screen() -> None:
    source = load_experiment_config(
        "config/historical_mask_expansion_regime_stage2a_learning_repair_v1.json"
    )
    repaired = load_experiment_config(
        "config/historical_mask_expansion_regime_stage2a_entry_balance_repair_v1.json"
    )

    assert (
        repaired["agent"]["entry_action_loss_reduction"]
        == "equal_present_class_mean_v1"
    )
    assert repaired["evolution"]["base_parent"] == source["evolution"][
        "base_parent"
    ]
    assert repaired["evolution"]["parent_candidate_ids"] == source[
        "evolution"
    ]["parent_candidate_ids"]
    assert "recovery_curriculum" not in repaired
    assert "agent.entry_action_loss_reduction" in repaired["evolution"][
        "frozen_paths"
    ]
    assert "agent.entry_action_loss_reduction" not in repaired["evolution"][
        "allowed_revision_paths"
    ]
    assert repaired["evolution"]["allowed_revision_paths"] == ()
    assert repaired["evolution"]["revision_bounds"] == {}
    assert {
        "regime_selectivity.loss_weight",
        "regime_selectivity.q_temperature",
        *REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS,
    } <= set(repaired["evolution"]["frozen_paths"])
    assert repaired["campaign"]["max_revisions_per_stage"] == 0
    assert len(repaired["campaign"]["budget_stages"]) == 1
    assert repaired["campaign"]["budget_stages"][0][
        "minimum_environment_steps"
    ] == 500_000
    assert repaired["campaign"]["budget_stages"][0]["allow_revisions"] is False
    assert repaired["campaign"]["budget_stages"][0]["revision_paths"] == ()

    expected = json.loads(json.dumps(source))
    actual = json.loads(json.dumps(repaired))
    expected["agent"]["entry_action_loss_reduction"] = (
        "equal_present_class_mean_v1"
    )
    expected["evolution"]["hypothesis"] = actual["evolution"]["hypothesis"]
    expected["evolution"]["frozen_paths"].append(
        "agent.entry_action_loss_reduction"
    )
    expected["evolution"]["frozen_paths"].extend([
        "regime_selectivity.loss_weight",
        "regime_selectivity.q_temperature",
        "regime_selectivity.formula",
        "regime_selectivity.semantics",
        "regime_selectivity.persistent_chop_negative_emphasis",
    ])
    expected["evolution"]["frozen_paths"] = sorted(
        expected["evolution"]["frozen_paths"]
    )
    actual["evolution"]["frozen_paths"] = sorted(
        actual["evolution"]["frozen_paths"]
    )
    expected["evolution"]["allowed_revision_paths"] = []
    expected["evolution"]["revision_bounds"] = {}
    expected["campaign"]["state_root"] = actual["campaign"]["state_root"]
    expected["campaign"]["max_revisions_per_stage"] = 0
    expected["campaign"]["budget_stages"] = [
        expected["campaign"]["budget_stages"][0]
    ]
    expected["campaign"]["budget_stages"][0]["name"] = actual["campaign"][
        "budget_stages"
    ][0]["name"]
    expected["campaign"]["budget_stages"][0][
        "selection_requirements"
    ] = actual["campaign"]["budget_stages"][0]["selection_requirements"]
    expected["campaign"]["budget_stages"][0]["allow_revisions"] = False
    expected["campaign"]["budget_stages"][0]["revision_paths"] = []
    expected["output"] = actual["output"]
    expected["_path"] = actual["_path"]

    assert actual == expected


def test_stage2_v4_is_one_frozen_two_boundary_matched_screen() -> None:
    from propevolve.balance_aware_regime_selectivity import (
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    )

    source = load_experiment_config(
        "config/historical_mask_expansion_regime_"
        "stage2a_entry_balance_repair_v1.json"
    )
    repaired = load_experiment_config(
        "config/historical_mask_expansion_regime_stage2_v4.json"
    )

    assert repaired["regime_selectivity"] == {
        **source["regime_selectivity"],
        "formula": PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        "semantics": PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        "persistent_chop_negative_emphasis": 1.0,
    }
    assert repaired["agent"]["entry_action_loss_reduction"] == (
        "equal_present_class_mean_v1"
    )
    assert repaired["evolution"]["base_parent"] == source["evolution"][
        "base_parent"
    ]
    assert repaired["evolution"]["parent_candidate_ids"] == source[
        "evolution"
    ]["parent_candidate_ids"]
    assert repaired["evolution"]["allowed_revision_paths"] == ()
    assert repaired["evolution"]["revision_bounds"] == {}
    assert repaired["campaign"]["max_revisions_per_stage"] == 0
    assert len(repaired["campaign"]["budget_stages"]) == 1
    assert repaired["campaign"]["budget_stages"][0][
        "minimum_environment_steps"
    ] == 500_000
    assert repaired["campaign"]["budget_stages"][0][
        "allow_revisions"
    ] is False
    assert repaired["campaign"]["budget_stages"][0]["revision_paths"] == ()
    assert repaired["campaign"]["budget_stages"][0][
        "curriculum_override"
    ] == {}
    assert repaired["training"][
        "entry_supervision_autonomy_start_fraction"
    ] == 0.95
    assert "training.entry_supervision_autonomy_start_fraction" in repaired[
        "evolution"
    ]["frozen_paths"]
    near_blow_gate = {
        "metric": "selection.near_blow_timeout_rate",
        "operator": "<=",
        "value": 0.6263636363636363,
    }
    assert repaired["campaign"]["budget_stages"][0][
        "selection_requirements"
    ] == [
        *source["campaign"]["budget_stages"][0][
            "selection_requirements"
        ],
        near_blow_gate,
    ]

    # Full-recipe equality is the matched-experiment guard.  Build the expected
    # child from the normalized loaded parent and permit only the declared
    # identity fields, atomic semantics switch, and entry-consolidation
    # boundary to differ.
    expected = json.loads(json.dumps(source))
    actual = json.loads(json.dumps(repaired))
    expected["regime_selectivity"].update({
        "formula": PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        "semantics": PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        "persistent_chop_negative_emphasis": 1.0,
    })
    expected["training"]["entry_supervision_autonomy_start_fraction"] = 0.95
    expected_frozen_paths = list(source["evolution"]["frozen_paths"])
    expected_frozen_paths.insert(
        expected_frozen_paths.index("training.greedy_diagnostic_interval_steps"),
        "training.entry_supervision_autonomy_start_fraction",
    )
    expected["evolution"]["frozen_paths"] = expected_frozen_paths
    expected["evolution"]["hypothesis"] = actual["evolution"]["hypothesis"]
    expected["campaign"]["state_root"] = actual["campaign"]["state_root"]
    expected["campaign"]["budget_stages"][0]["name"] = actual["campaign"][
        "budget_stages"
    ][0]["name"]
    expected["campaign"]["budget_stages"][0][
        "selection_requirements"
    ] = actual["campaign"]["budget_stages"][0]["selection_requirements"]
    expected["output"] = actual["output"]
    expected["_path"] = actual["_path"]

    assert actual == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("formula", REGIME_SELECTIVITY_FORMULA),
        ("semantics", "static_state_v1"),
        ("persistent_chop_negative_emphasis", 0.0),
    ),
)
def test_stage2_v4_persistent_chop_switch_cannot_partially_revert(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_stage2_v4.json"
    ).read_text())
    payload["regime_selectivity"][field] = value
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_source.read_bytes())
    path = tmp_path / "config" / "partial-stage2-v4.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="balance-aware Regime selectivity contract is invalid",
    ):
        load_experiment_config(path)


def test_fresh_entry_action_reduction_contract_rejects_missing_field(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_stage2a_entry_balance_repair_v1.json"
    ).read_text())
    del payload["agent"]["entry_action_loss_reduction"]
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_source.read_bytes())
    path = tmp_path / "config" / "missing-entry-action-reduction.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry action loss reduction is missing"):
        load_experiment_config(path)


def test_empty_revision_allowlist_requires_explicitly_nonrevisable_campaign(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_stage2a_entry_balance_repair_v1.json"
    ).read_text())
    payload["campaign"]["budget_stages"][0]["allow_revisions"] = True
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_source.read_bytes())
    path = tmp_path / "config" / "revisable-without-allowlist.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="revision and frozen paths"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda value: value.update({"training_only": False}), "contract"),
        (lambda value: value.update({"fill_offsets": [0, 1, 2, 3, 4]}), "contract"),
        (lambda value: value.update({"execution": "same_bar_close"}), "contract"),
        (lambda value: value.update({"risk_dollars": 500}), "economics"),
        (lambda value: value.update({"target_r": 3.0}), "economics"),
        (lambda value: value.update({"collision": "target_first"}), "contract"),
        (lambda value: value.update({"loss_weight": 0.0}), "economics"),
    ),
)
def test_entry_supervision_contract_fails_closed(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    ).read_text())
    mutate(payload["entry_supervision"])
    path = tmp_path / "invalid-entry-supervision.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"entry supervision {match}"):
        load_experiment_config(path)


def test_entry_supervision_must_remain_campaign_frozen(tmp_path: Path) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    ).read_text())
    payload["evolution"]["frozen_paths"].remove("entry_supervision")
    path = tmp_path / "unfrozen-entry-supervision.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry supervision must be frozen"):
        load_experiment_config(path)


def test_centered_entry_distillation_rejects_unauthenticated_centers(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    ).read_text())
    payload["teachers"][0]["entry_search_loss_weight"] = 0.3
    payload["teachers"][0]["entry_search_objective"] = "centered_log_odds"
    payload["teachers"][0]["entry_search_long_center"] = 0.1
    payload["teachers"][0]["entry_search_short_center"] = 0.1
    payload["teachers"][0]["entry_search_center_receipt"] = "receipt.json"
    payload["teachers"][0]["entry_search_center_receipt_sha256"] = ""
    path = tmp_path / "bad-center.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Expansion entry-search contract"):
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


def test_entry_supervision_schedule_defaults_to_legacy_teacher_boundary() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    )

    assert config["training"][
        "entry_supervision_autonomy_start_fraction"
    ] == config["training"]["teacher_autonomy_start_fraction"]


def test_distinct_entry_supervision_schedule_must_be_valid_and_frozen(
    tmp_path: Path,
) -> None:
    source = Path(
        "config/historical_mask_expansion_regime_post_launch_entry_v8.json"
    )
    payload = json.loads(source.read_text())
    schedule_path = "training.entry_supervision_autonomy_start_fraction"
    payload["training"]["entry_supervision_autonomy_start_fraction"] = 0.95
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt_source.read_bytes())
    path = tmp_path / "config" / "entry-consolidation.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry supervision schedule must be frozen"):
        load_experiment_config(path)

    payload["evolution"]["frozen_paths"].append(schedule_path)
    path.write_text(json.dumps(payload))
    config = load_experiment_config(path)
    assert config["training"][
        "entry_supervision_autonomy_start_fraction"
    ] == 0.95

    for invalid in (True, 0.79, 1.01):
        payload["training"][
            "entry_supervision_autonomy_start_fraction"
        ] = invalid
        path.write_text(json.dumps(payload))
        with pytest.raises(
            ValueError,
            match="entry supervision autonomy start fraction",
        ):
            load_experiment_config(path)


def test_training_short_circuit_is_explicit_and_fail_closed(tmp_path: Path) -> None:
    source = Path("config/historical_mask_expansion_regime_curriculum_v8.json")
    payload = json.loads(source.read_text())
    path = tmp_path / "short-circuit.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["training"]["short_circuit"] == {
        "minimum_environment_steps": 500_000,
        "minimum_passes": 1,
        "maximum_blow_rate": 0.1,
    }

    payload["training"]["short_circuit"]["maximum_blow_rate"] = 1.5
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training short circuit"):
        load_experiment_config(path)

    payload["training"]["short_circuit"]["maximum_blow_rate"] = 0.1
    payload["training"]["short_circuit"]["collapse"] = {
        "window_episodes": 1,
        "minimum_prior_passes": 2,
        "maximum_recent_passes": 0,
        "maximum_average_hold_bars": 4.0,
        "minimum_voluntary_close_rate": 0.8,
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="collapse detector"):
        load_experiment_config(path)

    payload["training"]["short_circuit"]["collapse"] = {
        "window_episodes": 5,
        "minimum_prior_passes": 2,
        "maximum_recent_passes": 0,
        "maximum_average_hold_bars": float("nan"),
        "minimum_voluntary_close_rate": 0.8,
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="collapse detector"):
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
    assert config["observation"] == {
        "management_state": "entry_risk_v1",
        "include_unrealized_r": True,
        "include_peak_favorable_r": True,
        "include_giveback_r": True,
        "include_hold_fraction": True,
        "include_ratchet_active": True,
        "include_protected_r": True,
        "r_scale": 10.0,
        "hold_horizon_bars": 120,
    }
    stages = config["campaign"]["budget_stages"]
    assert stages[0]["curriculum_override"][
        "challenge.large_win_bonus_coefficient"
    ] == 0.1
    assert "challenge.large_win_bonus_coefficient" in stages[0]["revision_paths"]
    expected_average_winner_r = (0.75, 1.25, 1.5, 2.0)
    for stage, expected in zip(stages, expected_average_winner_r):
        requirement = next(
            item
            for item in stage["selection_requirements"]
            if item["metric"] == "selection.average_win_r"
        )
        assert requirement == {
            "metric": "selection.average_win_r",
            "operator": ">=",
            "value": expected,
        }
    assert any(
        item["metric"] == "selection.average_win_r"
        for item in stages[1]["parent_improvement_requirements"]
    )


def test_trade_management_observation_contract_is_fail_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        Path(
            "config/historical_mask_expansion_regime_trend_curriculum_v9.json"
        ).read_text()
    )
    path = tmp_path / "invalid-observation.json"

    payload["observation"]["include_peak_favorable_r"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="observation contract is invalid"):
        load_experiment_config(path)

    payload["observation"]["include_peak_favorable_r"] = True
    payload["observation"]["r_scale"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="observation contract is invalid"):
        load_experiment_config(path)


def test_expansion_regime_trend_profile_is_a_matched_two_teacher_benchmark() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_profile_v8.json"
    )

    assert [teacher["kind"] for teacher in config["teachers"]] == [
        "expansion",
        "regime",
    ]
    assert config["observation"]["management_state"] == "entry_risk_v1"
    assert config["sealed_confirmation"]["teacher_free"] is True
    assert config["output"].endswith("expansion_regime_trend_profile_v8")
    assert all(
        any(
            requirement["metric"] == "selection.average_win_r"
            for requirement in stage["selection_requirements"]
        )
        for stage in config["campaign"]["budget_stages"]
    )


def test_target_sync_challenger_changes_only_the_diagnosed_learning_boundary() -> None:
    baseline = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_profile_v8.json"
    )
    challenger = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_profile_sync_v8b.json"
    )

    for section in (
        "cache",
        "cache_root",
        "teachers",
        "observation",
        "challenge",
        "temporal",
        "runtime",
        "sealed_confirmation",
    ):
        assert challenger[section] == baseline[section]
    assert challenger["agent"] == {
        **baseline["agent"],
        "target_sync_updates": 1_000,
    }
    assert challenger["training"] == {
        **baseline["training"],
        "short_circuit": {
            **baseline["training"]["short_circuit"],
            "collapse": {
                "window_episodes": 12,
                "minimum_prior_passes": 2,
                "maximum_recent_passes": 0,
                "maximum_average_hold_bars": 4.0,
                "minimum_voluntary_close_rate": 0.8,
            },
        },
    }
    assert challenger["campaign"]["budget_stages"] == baseline["campaign"][
        "budget_stages"
    ]
    assert challenger["campaign"]["selection_requirements"] == baseline[
        "campaign"
    ]["selection_requirements"]
    assert challenger["campaign"]["diagnostic_targets"] == baseline["campaign"][
        "diagnostic_targets"
    ]


def test_config_accepts_explicit_soft_target_update(tmp_path: Path) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["agent"]["target_update_mode"] = "soft"
    payload["agent"]["target_soft_tau"] = 0.005
    path = tmp_path / "soft-target.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["agent"]["target_update_mode"] == "soft"
    assert config["agent"]["target_soft_tau"] == 0.005


@pytest.mark.parametrize(
    ("mode", "tau"),
    (("periodic", 0.005), ("soft", 0.0), ("soft", 1.1)),
)
def test_config_rejects_invalid_target_update_contract(
    tmp_path: Path,
    mode: str,
    tau: float,
) -> None:
    payload = json.loads(Path("config/historical_mask_v1.json").read_text())
    payload["agent"]["target_update_mode"] = mode
    payload["agent"]["target_soft_tau"] = tau
    path = tmp_path / "invalid-target-update.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="target update"):
        load_experiment_config(path)


def test_soft_target_challenger_changes_only_target_update_mechanism() -> None:
    baseline = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_profile_sync_v8b.json"
    )
    challenger = load_experiment_config(
        "config/historical_mask_expansion_regime_trend_profile_soft_target_v8c.json"
    )

    for section in (
        "cache",
        "cache_root",
        "teachers",
        "observation",
        "challenge",
        "temporal",
        "runtime",
        "sealed_confirmation",
        "training",
    ):
        assert challenger[section] == baseline[section]
    assert challenger["agent"] == {
        **baseline["agent"],
        "target_update_mode": "soft",
        "target_soft_tau": 0.005,
    }
    assert challenger["campaign"]["budget_stages"] == baseline["campaign"][
        "budget_stages"
    ]
    assert challenger["campaign"]["selection_requirements"] == baseline[
        "campaign"
    ]["selection_requirements"]


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


def test_stage2_recovery_recipe_authenticates_one_shot_account_contract(
    tmp_path: Path,
) -> None:
    source = Path(
        "config/historical_mask_expansion_regime_stage2_selectivity_recovery_v1.json"
    )
    config = load_experiment_config(source)

    assert config["challenge"]["minimum_mll_headroom"] == 500
    assert config["challenge"]["per_trade_risk_dollars"] == 300
    assert config["agent"]["learning_rate"] == 0.0001
    assert "agent.learning_rate" in config["evolution"]["frozen_paths"]
    assert "agent.learning_rate" not in config["evolution"][
        "allowed_revision_paths"
    ]
    assert "agent.learning_rate" not in config["evolution"]["revision_bounds"]
    assert all(
        "agent.learning_rate" not in stage["revision_paths"]
        and "agent.learning_rate" not in stage.get("curriculum_override", {})
        for stage in config["campaign"]["budget_stages"]
    )
    assert config["campaign"]["budget_stages"][1][
        "parent_retention_requirements"
    ] == [{
        "metric": "selection.greedy_entry_rate",
        "maximum_regression": 0.02,
    }]
    assert config["recovery_curriculum"] == {
        "episode_fraction": 0.0,
        "schedule_seed": 271828,
        "stress_evaluation_episodes": 200,
        "start_state": {
            "realized_pnl": -2700,
            "equity_pnl": -2700,
            "peak_equity_pnl": 0,
            "mll_floor_pnl": -3000,
            "passmark_locked": False,
            "position_side": 0,
            "position_size": 0,
            "session_pnl": -2700,
            "trading_days_elapsed": 1,
        },
        "entry_permit": {
            "remaining_entries": 1,
            "exception_headroom": 300,
            "success_pnl": -2500,
        },
    }

    payload = json.loads(source.read_text())
    payload["recovery_curriculum"]["entry_permit"]["remaining_entries"] = 2
    path = tmp_path / "stage2-invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="start contract drifted"):
        load_experiment_config(path)


def test_stage2a_regime_repair_is_two_matched_stage1_warm_started_screens() -> None:
    from propevolve.balance_aware_regime_selectivity import (
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    )

    failed = load_experiment_config(
        "config/historical_mask_expansion_regime_stage2_selectivity_recovery_v1.json"
    )
    repaired = load_experiment_config(
        "config/historical_mask_expansion_regime_stage2a_learning_repair_v1.json"
    )

    assert repaired["output"] != failed["output"]
    assert repaired["campaign"]["state_root"] != failed["campaign"]["state_root"]
    assert repaired["evolution"]["base_parent"] == failed["evolution"][
        "base_parent"
    ]
    assert repaired["evolution"]["parent_candidate_ids"] == failed[
        "evolution"
    ]["parent_candidate_ids"]
    assert repaired["evolution"]["base_parent"] == {
        "archive_root": (
            "runs/historical_mask_expansion_regime_post_launch_entry_"
            "balanced_v8b/archive"
        ),
        "candidate_id": (
            "1bccc5f5e81e87527644f8547b69b26cf5bc1227688b96971a664a81e9f964a0"
        ),
        "evaluation_id": (
            "c49852955655b705e376e057dfe2bf58784481175363b970bab063d8c42f981b"
        ),
        "model_sha256": (
            "b445ce526eebafd3121981e9de720031d9710cd4e99c8dc49017d35e50d55584"
        ),
    }
    assert "recovery_curriculum" not in repaired
    for field in (
        "assets",
        "cache",
        "cache_root",
        "tickers",
        "deployment_tickers",
        "training_only_tickers",
        "timeframe_minutes",
        "temporal",
        "point_values",
        "round_trip_fees",
        "teachers",
        "entry_supervision",
        "observation",
        "runtime",
        "challenge",
        "agent",
    ):
        assert repaired[field] == failed[field]

    selectivity = repaired["regime_selectivity"]
    assert selectivity["semantics"] == "static_state_v1"
    assert selectivity["persistent_chop_negative_emphasis"] == 0.0
    assert selectivity["side_balance"] == {
        "schema": "equal_long_short_v1",
        "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
    }
    assert {
        "regime_selectivity.headroom_pressure",
        "regime_selectivity.dominant_chop_pressure",
    } <= set(repaired["evolution"]["frozen_paths"])
    stages = repaired["campaign"]["budget_stages"]
    assert repaired["campaign"]["max_revisions_per_stage"] == 3
    assert [stage["name"] for stage in stages] == [
        "regime_side_balance_500k",
        "persistent_chop_regime_500k",
    ]
    assert [stage["minimum_environment_steps"] for stage in stages] == [
        500_000,
        500_000,
    ]
    assert all(stage["warm_start_parent"] is True for stage in stages)
    assert stages[0]["curriculum_override"] == {}
    assert stages[1]["curriculum_override"] == {
        "regime_selectivity.formula": PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        "regime_selectivity.semantics": (
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
        "regime_selectivity.persistent_chop_negative_emphasis": 1.0,
    }
    assert stages[1]["parent_improvement_requirements"] == [{
        "metric": "selection.near_blow_timeout_rate",
        "direction": "minimize",
        "minimum_delta": 0.01,
    }]
    expected_parent_retention = [
        {
            "metric": "selection.pass_rate",
            "direction": "maximize",
            "maximum_regression": 0.0,
        },
        {
            "metric": "selection.average_win_r",
            "direction": "maximize",
            "maximum_regression": 0.0,
        },
        {
            "metric": "selection.expectancy_r",
            "direction": "maximize",
            "maximum_regression": 0.0,
        },
        {
            "metric": "selection.two_r_mfe_capture_ratio",
            "direction": "maximize",
            "maximum_regression": 0.0,
        },
        {
            "metric": "selection.near_blow_timeout_rate",
            "direction": "minimize",
            "maximum_regression": 0.0,
        },
    ]
    assert all(
        stage["parent_retention_requirements"] == expected_parent_retention
        for stage in stages
    )
    for stage in stages:
        assert {
            "metric": "selection.long_entry_count",
            "operator": ">",
            "value": 0,
        } in stage["selection_requirements"]
        assert {
            "metric": "selection.short_entry_count",
            "operator": ">",
            "value": 0,
        } in stage["selection_requirements"]


def test_stage2_recovery_stress_baseline_is_frozen_while_fraction_may_revise(
    tmp_path: Path,
) -> None:
    source = Path(
        "config/historical_mask_expansion_regime_stage2_selectivity_recovery_v1.json"
    )
    payload = json.loads(source.read_text())
    assert "recovery_curriculum.stress_evaluation_episodes" in payload[
        "evolution"
    ]["frozen_paths"]
    assert "recovery_curriculum.stress_evaluation_episodes" not in payload[
        "evolution"
    ]["allowed_revision_paths"]
    assert payload["recovery_curriculum"]["episode_fraction"] == 0.0
    assert payload["recovery_curriculum"]["stress_evaluation_episodes"] == 200

    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt = receipt_source.read_bytes()
    receipt_path = tmp_path / "config" / "receipts" / receipt_source.name
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt)
    payload["recovery_curriculum"]["stress_evaluation_episodes"] = 0
    path = tmp_path / "config" / "stage2-no-baseline.json"
    path.write_text(json.dumps(payload))
    config = load_experiment_config(path)
    assert config["recovery_curriculum"]["episode_fraction"] == 0.0
    assert config["recovery_curriculum"]["stress_evaluation_episodes"] == 0

    payload["recovery_curriculum"]["episode_fraction"] = 0.25
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires a frozen stress evaluation"):
        load_experiment_config(path)
