from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tests.recipe_fixtures import (
    paired_aplus_recipe,
    paired_recurrent_aplus_recipe,
)

from propevolve.config import (
    BALANCE_CURRICULUM_FROZEN_PATHS,
    BALANCE_OUTCOME_CONTRAST_FROZEN_PATH,
    DEFAULT_CONFIG_PATH,
    RECOVERY_CURRICULUM_FROZEN_PATHS,
    REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS,
    REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS,
    TRAINING_POLICY_HEALTH_FROZEN_PATH,
    agent_runtime_settings,
    load_experiment_config,
    materialize_effective_config,
)
from propevolve.balance_aware_regime_selectivity import (
    ACTION_ORDER as REGIME_SELECTIVITY_ACTION_ORDER,
    ALL_DOMINANT_CHOP_MARGIN_FORMULA,
    ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
    CHOP_MARGIN_EXPANSION_REGIME_CONFLUENCE_FORMULA,
    EXPANSION_REGIME_CONFLUENCE_FORMULA,
    EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    FORMULA as REGIME_SELECTIVITY_FORMULA,
    PAIRED_A_PLUS_CONTRASTIVE_FORMULA,
    PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
    PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA,
    PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
    SCHEMA as REGIME_SELECTIVITY_SCHEMA,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    TARGET_SOURCE as REGIME_SELECTIVITY_TARGET_SOURCE,
)


CURRENT_RECIPE = paired_aplus_recipe(100)
STAGE2_PAIRED_A_PLUS_RECIPE = CURRENT_RECIPE
STAGE2_PAIRED_A_PLUS_450_RECIPE = paired_aplus_recipe(450)
STAGE2_PAIRED_RECURRENT_A_PLUS_RECIPE = paired_recurrent_aplus_recipe(200)

STAGE2_V6_ASSOCIATION_SEMANTICS = "persistent_chop_association_v2"
STAGE2_V6_ASSOCIATION_FORMULA = (
    "equal_present_group_mean(exact_wait_weighted_ce,exact_long_ce,"
    "exact_short_ce,zero_margin_dead_vs_transition_positive_wait_rank)"
)


def _current_payload() -> dict:
    return json.loads(CURRENT_RECIPE.read_text())


def _generic_payload() -> dict:
    """Return a current-recipe fixture without Stage 2-specific selectivity."""
    payload = _current_payload()
    payload.pop("regime_selectivity")
    payload["training"].pop("regime_wait_sequence_update_period", None)
    payload["evolution"]["frozen_paths"] = [
        path for path in payload["evolution"]["frozen_paths"]
        if not path.startswith("regime_selectivity.")
        and path != "training.regime_wait_sequence_update_period"
    ]
    return payload


def _recovery_curriculum_payload() -> dict:
    payload = _generic_payload()
    payload["challenge"]["minimum_mll_headroom"] = 500.0
    payload["challenge"]["per_trade_risk_dollars"] = 300.0
    payload["recovery_curriculum"] = {
        "schedule_seed": 37,
        "stress_evaluation_episodes": 200,
        "recovery_success_pnl": 0.0,
        "start_state": {
            "realized_pnl": -2_000.0,
            "equity_pnl": -2_000.0,
            "peak_equity_pnl": 0.0,
            "mll_floor_pnl": -3_000.0,
            "passmark_locked": False,
            "position_side": 0,
            "position_size": 0,
            "session_pnl": -2_000.0,
            "trading_days_elapsed": 1,
        },
        "action_value_supervision": {
            "loss_weight": 0.25,
            "temperature": 1.0,
            "action_margin": 0.25,
            "store_capacity": 200,
            "target_every_episodes": 1,
            "start_pnls": [-2_500, -2_000, -1_500, -1_000, -500],
            "retain_nonnegative_entry_policy": True,
            "post_recovery_contrast_replay": {
                "path": "runs/local/retained-versus-giveback.pt",
                "sha256": "a" * 64,
                "update_period": 8,
                "max_examples": 8,
            },
        },
    }
    payload["evolution"]["frozen_paths"].extend(
        RECOVERY_CURRICULUM_FROZEN_PATHS
    )
    return payload


def test_effective_config_materialization_is_single_and_idempotent() -> None:
    source = {
        "challenge": {},
        "agent": {},
        "training": {"epsilon_start": 0.2, "epsilon_end": 0.05},
    }

    first = materialize_effective_config(source)
    second = materialize_effective_config(first)

    assert first == second
    assert source == {
        "challenge": {},
        "agent": {},
        "training": {"epsilon_start": 0.2, "epsilon_end": 0.05},
    }
    assert first["runtime"] == json.loads(DEFAULT_CONFIG_PATH.read_text())[
        "values"
    ]["runtime"]
    assert first["training"]["regime_wait_sequence_fraction"] == 0.0
    assert first["training"]["management_epsilon_start"] == 0.2
    assert first["agent"]["target_update_mode"] == "hard"
    assert first["agent"]["auxiliary_gradient_conflict_mode"] == "none"


def test_effective_config_defaults_are_loaded_from_json(
    tmp_path: Path,
) -> None:
    defaults = json.loads(DEFAULT_CONFIG_PATH.read_text())
    defaults["values"]["runtime"]["compile_mode"] = "reduce-overhead"
    defaults["values"]["training"]["minimum_environment_steps"] = 17
    defaults_path = tmp_path / "arbitrary-default-settings-name.json"
    defaults_path.write_text(json.dumps(defaults))

    effective = materialize_effective_config(
        {
            "output": "runs/example",
            "training": {
                "epsilon_start": 0.2,
                "epsilon_end": 0.05,
                "minimum_environment_steps": 99,
            },
        },
        defaults_path=defaults_path,
    )

    assert effective["runtime"]["compile_mode"] == "reduce-overhead"
    assert effective["training"]["minimum_environment_steps"] == 99
    assert effective["training"]["management_epsilon_start"] == 0.2
    assert effective["_archive_output"] == "runs/example"


def test_effective_config_rejects_invalid_default_document(
    tmp_path: Path,
) -> None:
    defaults_path = tmp_path / "defaults.json"
    defaults_path.write_text(json.dumps({"schema": "wrong", "values": {}}))

    with pytest.raises(ValueError, match="default config schema"):
        materialize_effective_config({}, defaults_path=defaults_path)


def test_effective_config_rematerializes_normalized_sequences() -> None:
    effective = materialize_effective_config({
        "training": {"epsilon_start": 0.2, "epsilon_end": 0.05},
        "teachers": ({"kind": "regime"},),
        "campaign": {
            "budget_stages": ({"name": "stage"},),
        },
        "evolution": {"allowed_revision_paths": []},
    })

    assert effective["teachers"] == [{
        "entry_search_loss_weight": 0.0,
        "kind": "regime",
    }]
    assert effective["campaign"]["budget_stages"] == [{
        "allow_revisions": True,
        "budget_mode": "environment_steps",
        "curriculum_override": {},
        "name": "stage",
        "parent_improvement_any_requirements": [],
        "parent_improvement_requirements": [],
        "parent_retention_requirements": [],
        "revision_paths": [],
        "warm_start_parent": False,
    }]


def _balance_curriculum_payload() -> dict:
    payload = _generic_payload()
    payload["balance_curriculum"] = {
        "schedule_seed": 37,
        "start_pnls": [-2_000],
        "validation_episodes": 200,
    }
    payload["evolution"]["frozen_paths"].extend(
        BALANCE_CURRICULUM_FROZEN_PATHS
    )
    return payload


def test_config_accepts_single_policy_balance_curriculum_by_values(
    tmp_path: Path,
) -> None:
    payload = _balance_curriculum_payload()
    path = tmp_path / "arbitrary-balance-curriculum.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["balance_curriculum"] == {
        "schedule_seed": 37,
        "start_pnls": (-2_000.0,),
        "validation_episodes": 200,
        "outcome_contrast_replay": None,
    }
    assert config.get("recovery_curriculum") is None


def test_config_accepts_optional_balance_outcome_contrast_replay(
    tmp_path: Path,
) -> None:
    payload = _balance_curriculum_payload()
    payload["balance_curriculum"]["outcome_contrast_replay"] = {
        "update_period": 8,
        "max_examples": 8,
    }
    payload["evolution"]["frozen_paths"].append(
        BALANCE_OUTCOME_CONTRAST_FROZEN_PATH
    )
    path = tmp_path / "v23-balance-outcome-contrast.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["balance_curriculum"]["outcome_contrast_replay"] == {
        "update_period": 8,
        "max_examples": 8,
    }


def test_config_accepts_authenticated_balance_pass_replay_by_values(
    tmp_path: Path,
) -> None:
    payload = _balance_curriculum_payload()
    payload["balance_curriculum"]["pass_replay"] = {
        "path": "runs/recovery-pass-replay/balance-passes.pt",
        "sha256": "a" * 64,
        "update_period": 8,
        "max_examples": 8,
        "output": "balance-pass-replay.pt",
    }
    payload["evolution"]["frozen_paths"].append(
        "balance_curriculum.pass_replay"
    )
    path = tmp_path / "arbitrary-unified-balance-recipe.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["balance_curriculum"]["pass_replay"] == {
        "path": "runs/recovery-pass-replay/balance-passes.pt",
        "sha256": "a" * 64,
        "update_period": 8,
        "max_examples": 8,
        "output": "balance-pass-replay.pt",
    }
    assert config.get("recovery_curriculum") is None


def test_config_rejects_balance_curriculum_with_recovery_mode(
    tmp_path: Path,
) -> None:
    payload = _recovery_curriculum_payload()
    payload["balance_curriculum"] = {
        "schedule_seed": 37,
        "start_pnls": [-2_000],
        "validation_episodes": 200,
    }
    payload["evolution"]["frozen_paths"].extend(
        BALANCE_CURRICULUM_FROZEN_PATHS
    )
    path = tmp_path / "two-policy-modes.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="mutually exclusive"):
        load_experiment_config(path)


def test_config_accepts_post_recovery_contrast_replay_by_values(
    tmp_path: Path,
) -> None:
    payload = _recovery_curriculum_payload()
    path = tmp_path / "arbitrary-runtime-name.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    replay = config["recovery_curriculum"]["action_value_supervision"][
        "post_recovery_contrast_replay"
    ]
    assert replay["update_period"] == 8
    assert replay["max_examples"] == 8
    assert config["recovery_curriculum"]["action_value_supervision"][
        "action_margin"
    ] == 0.25


def test_config_rejects_negative_recovery_action_margin(tmp_path: Path) -> None:
    payload = _recovery_curriculum_payload()
    payload["recovery_curriculum"]["action_value_supervision"][
        "action_margin"
    ] = -0.1
    path = tmp_path / "negative-recovery-margin.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="recovery action-value supervision",
    ):
        load_experiment_config(path)


def _episode_budget_payload() -> dict:
    payload = _generic_payload()
    payload["training"]["budget_mode"] = "episodes"
    payload["training"]["episodes"] = 500
    payload["training"].pop("minimum_environment_steps", None)
    payload["evolution"]["frozen_paths"] = [
        path for path in payload["evolution"]["frozen_paths"]
        if path != "training.minimum_environment_steps"
    ]
    short_circuit = payload["training"]["short_circuit"]
    short_circuit["minimum_completed_episodes"] = 18
    short_circuit.pop("minimum_environment_steps", None)
    requirements = payload["campaign"]["selection_requirements"]
    payload["campaign"]["budget_stages"] = [
        {
            "name": f"episode_{episodes}",
            "budget_mode": "episodes",
            "training_episodes": episodes,
            "validation_episodes": 200,
            "short_circuit_minimum_episodes": 18,
            "allow_revisions": False,
            "warm_start_parent": True,
            "revision_paths": [],
            "selection_requirements": list(requirements),
        }
        for episodes in (200, 300, 500)
    ]
    return payload


def _policy_health_payload(
    *,
    require_positive_persistent_regime_association: bool = False,
) -> dict:
    payload = {
        "schema": "propevolve_training_policy_health_v1",
        "minimum_completed_episodes": 45,
        "probe_interval_episodes": 45,
        "minimum_probe_recall": {
            "WAIT": 0.35,
            "ENTER_LONG_1": 0.30,
            "ENTER_SHORT_1": 0.30,
        },
        "entry_mass_fraction": {"minimum": 0.30, "maximum": 0.36},
        "require_zero_positive_entry_soft_wait_veto": True,
        "economic_futility": {
            "minimum_completed_episodes": 45,
            "maximum_near_blow_timeout_rate": 0.75,
            "maximum_mean_terminal_pnl": -1500.0,
            "maximum_expectancy_r": -0.15,
            "minimum_failed_conditions": 2,
        },
    }
    if require_positive_persistent_regime_association:
        payload["require_positive_persistent_regime_association"] = True
    return payload


def _stage2a_config(tmp_path: Path) -> Path:
    payload = _current_payload()
    receipt_source = Path(
        "config/receipts/expansion_entry_centers_9market_pre2025_v1.json"
    )
    receipt = receipt_source.read_bytes()
    receipt_path = tmp_path / "config" / "receipts" / "centers.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(receipt)
    receipt_payload = json.loads(receipt)
    payload["regime_selectivity"]["expansion_center_receipt"] = (
        "config/receipts/centers.json"
    )
    payload["regime_selectivity"]["expansion_center_receipt_sha256"] = (
        hashlib.sha256(receipt).hexdigest()
    )
    payload["regime_selectivity"]["expansion_long_center"] = receipt_payload[
        "long_center"
    ]
    payload["regime_selectivity"]["expansion_short_center"] = receipt_payload[
        "short_center"
    ]
    payload["evolution"]["frozen_paths"] = sorted(set(
        payload["evolution"]["frozen_paths"]
    ) | set(REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS))
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


def test_stage2a_accepts_versioned_expansion_regime_confluence_semantics(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"].update({
        "formula": EXPANSION_REGIME_CONFLUENCE_FORMULA,
        "semantics": EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    })
    for field in (
        "chop_wait_margin",
        "failed_confluence_margin",
        "paired_a_plus_margin",
    ):
        payload["regime_selectivity"].pop(field, None)
        frozen_path = f"regime_selectivity.{field}"
        if frozen_path in payload["evolution"]["frozen_paths"]:
            payload["evolution"]["frozen_paths"].remove(frozen_path)
    payload["training"]["regime_wait_sequence_update_period"] = 0
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["regime_selectivity"]["semantics"] == (
        EXPANSION_REGIME_CONFLUENCE_SEMANTICS
    )


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
    transition = payload["campaign"]["budget_stages"][0]
    payload["campaign"]["max_revisions_per_stage"] = 1
    transition["allow_revisions"] = True
    transition["revision_paths"] = list(
        REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS
    )
    transition["curriculum_override"] = {
        "regime_selectivity.formula": PERSISTENT_CHOP_NEGATIVE_WEIGHT_FORMULA,
        "regime_selectivity.semantics": (
            PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS
        ),
        "regime_selectivity.persistent_chop_negative_emphasis": 1.0,
    }
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)
    assert config["campaign"]["budget_stages"][0][
        "curriculum_override"
    ] == transition["curriculum_override"]

    del payload["campaign"]["budget_stages"][0]["curriculum_override"][
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
    payload = _generic_payload()
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
    payload = _generic_payload()
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


def test_validation_no_trade_patience_is_json_configured_and_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
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


def test_sparse_hard_wait_update_schedule_is_integer_frozen_and_selectivity_bound(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["training"].update({
        "terminal_sequence_fraction": 0.5,
        "safety_sequence_fraction": 0.25,
        "entry_opportunity_sequence_fraction": 0.25,
        "regime_wait_sequence_fraction": 0.0,
        "regime_wait_sequence_update_period": 8,
    })
    frozen = payload["evolution"]["frozen_paths"]
    frozen.append("training.regime_wait_sequence_update_period")
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)
    assert config["training"]["regime_wait_sequence_update_period"] == 8

    payload["evolution"]["frozen_paths"] = [
        field for field in payload["evolution"]["frozen_paths"]
        if field != "training.regime_wait_sequence_update_period"
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Regime WAIT replay identity"):
        load_experiment_config(path)


def test_all_dominant_chop_margin_semantics_are_frozen_and_loadable(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"].update({
        "semantics": ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
        "formula": ALL_DOMINANT_CHOP_MARGIN_FORMULA,
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
    })
    payload["regime_selectivity"].pop("paired_a_plus_margin", None)
    payload["evolution"]["frozen_paths"].remove(
        "regime_selectivity.paired_a_plus_margin"
    )
    payload["evolution"]["frozen_paths"] = sorted(set(
        payload["evolution"]["frozen_paths"]
    ) | {
        "regime_selectivity.chop_wait_margin",
        "regime_selectivity.failed_confluence_margin",
    })
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["regime_selectivity"]["semantics"] == (
        ALL_DOMINANT_CHOP_MARGIN_SEMANTICS
    )
    assert config["regime_selectivity"]["formula"] == (
        ALL_DOMINANT_CHOP_MARGIN_FORMULA
    )


def test_paired_a_plus_margin_is_configured_and_frozen_without_score_cutoffs(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"].update({
        "semantics": PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        "formula": PAIRED_A_PLUS_CONTRASTIVE_FORMULA,
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
        "paired_a_plus_margin": 0.25,
    })
    payload["evolution"]["frozen_paths"] = sorted(set(
        payload["evolution"]["frozen_paths"]
    ) | {
        "regime_selectivity.chop_wait_margin",
        "regime_selectivity.failed_confluence_margin",
        "regime_selectivity.paired_a_plus_margin",
    })
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    selectivity = config["regime_selectivity"]
    assert selectivity["paired_a_plus_margin"] == 0.25
    assert not any("threshold" in field for field in selectivity)


def test_paired_a_plus_recipe_freezes_the_training_contrast_contract() -> None:
    candidate = load_experiment_config(STAGE2_PAIRED_A_PLUS_RECIPE)

    selectivity = candidate["regime_selectivity"]
    assert selectivity["semantics"] == PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS
    assert selectivity["formula"] == PAIRED_A_PLUS_CONTRASTIVE_FORMULA
    assert selectivity["paired_a_plus_margin"] == 0.25
    assert "regime_selectivity.paired_a_plus_margin" in candidate[
        "evolution"
    ]["frozen_paths"]
    assert not any("threshold" in field for field in selectivity)
    stage = candidate["campaign"]["budget_stages"][0]
    assert stage["training_episodes"] == 100
    assert stage["validation_episodes"] == 200


def test_paired_recurrent_recipe_requires_its_explicit_replay_contract(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"].update({
        "semantics": PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        "formula": PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA,
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
        "paired_a_plus_margin": 0.25,
        "side_balance": {
            "schema": "paired_recurrent_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
    })
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["regime_selectivity"]["semantics"] == (
        PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
    )
    payload["regime_selectivity"]["side_balance"]["schema"] = (
        "equal_long_short_v1"
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="paired recurrent"):
        load_experiment_config(path)
    payload["regime_selectivity"]["side_balance"]["schema"] = (
        "paired_recurrent_long_short_v1"
    )
    payload["training"]["batch_sequences"] = 12
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="even positive pair stratum"):
        load_experiment_config(path)


def test_paired_recurrent_winner_weight_is_configured_and_frozen(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"].update({
        "semantics": PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        "formula": PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA,
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
        "paired_a_plus_margin": 0.25,
        "paired_a_plus_winner_loss_weight": 2.0,
        "side_balance": {
            "schema": "paired_recurrent_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
    })
    payload["evolution"]["frozen_paths"] = sorted(set(
        payload["evolution"]["frozen_paths"]
    ) | {
        "regime_selectivity.chop_wait_margin",
        "regime_selectivity.failed_confluence_margin",
        "regime_selectivity.paired_a_plus_margin",
        "regime_selectivity.paired_a_plus_winner_loss_weight",
    })
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["regime_selectivity"][
        "paired_a_plus_winner_loss_weight"
    ] == 2.0


def test_paired_recurrent_recipe_loading_is_independent_of_config_filename(
    tmp_path: Path,
) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["workspace_root"] = ".."
    payload["regime_selectivity"].update({
        "semantics": PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS,
        "formula": PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA,
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
        "paired_a_plus_margin": 0.25,
        "side_balance": {
            "schema": "paired_recurrent_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
    })
    path.write_text(json.dumps(payload))
    renamed = tmp_path / "runtime-selected" / "anything.json"
    renamed.parent.mkdir()
    renamed.write_bytes(path.read_bytes())

    original = load_experiment_config(path)
    selected_at_runtime = load_experiment_config(renamed)

    assert selected_at_runtime["regime_selectivity"] == original[
        "regime_selectivity"
    ]
    assert selected_at_runtime["campaign"] == original["campaign"]
    assert Path(selected_at_runtime["_path"]).name == renamed.name
    assert Path(selected_at_runtime["_root"]) == tmp_path.resolve()


def test_recipe_requires_explicit_workspace_root_key(tmp_path: Path) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload.pop("workspace_root")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="workspace_root recipe field is required"):
        load_experiment_config(path)


def test_v21_recipe_changes_only_paired_replay_and_uses_200_episode_stage() -> None:
    baseline = load_experiment_config(STAGE2_PAIRED_A_PLUS_RECIPE)
    candidate = load_experiment_config(STAGE2_PAIRED_RECURRENT_A_PLUS_RECIPE)

    assert candidate.get("recovery_curriculum") is None
    selectivity = candidate["regime_selectivity"]
    assert selectivity["semantics"] == (
        PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_SEMANTICS
    )
    assert selectivity["formula"] == PAIRED_RECURRENT_A_PLUS_CONTRASTIVE_FORMULA
    assert "softplus(action_margin-matched_economic_winner" in selectivity[
        "formula"
    ]
    assert "softplus(action_margin+matched_economic_failure" in selectivity[
        "formula"
    ]
    assert not selectivity["semantics"].endswith("_v7")
    assert selectivity["side_balance"] == {
        "schema": "paired_recurrent_long_short_v1",
        "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
    }
    assert selectivity["paired_a_plus_margin"] == 0.25
    stage = candidate["campaign"]["budget_stages"][0]
    assert stage["training_episodes"] == 200
    assert stage["validation_episodes"] == 200
    for key in ("agent", "challenge", "entry_supervision", "teachers", "temporal"):
        assert candidate[key] == baseline[key]


@pytest.mark.parametrize("declaration", ("missing", "null"))
def test_episode_recipe_parses_disabled_short_circuit_from_json(
    tmp_path: Path,
    declaration: str,
) -> None:
    payload = _episode_budget_payload()
    payload["training"]["episodes"] = 200
    payload["training"]["validation_episodes"] = 200
    if declaration == "missing":
        payload["training"].pop("short_circuit")
    else:
        payload["training"]["short_circuit"] = None
    payload["evolution"]["frozen_paths"] = [
        path
        for path in payload["evolution"]["frozen_paths"]
        if path != TRAINING_POLICY_HEALTH_FROZEN_PATH
    ]
    payload["campaign"]["budget_stages"] = [{
        "name": "complete_training_budget",
        "budget_mode": "episodes",
        "training_episodes": 200,
        "validation_episodes": 200,
        "allow_revisions": False,
        "warm_start_parent": True,
        "revision_paths": [],
        "selection_requirements": list(
            payload["campaign"]["selection_requirements"]
        ),
    }]
    path = tmp_path / "runtime-selected-recipe.json"
    path.write_text(json.dumps(payload))

    recipe = load_experiment_config(path)
    stage = recipe["campaign"]["budget_stages"][0]

    assert recipe["training"]["short_circuit"] is None
    assert stage["training_episodes"] == 200
    assert stage["validation_episodes"] == 200
    assert "short_circuit_minimum_episodes" not in stage


def test_config_rejects_frozen_path_removed_by_runtime_json(tmp_path: Path) -> None:
    payload = _episode_budget_payload()
    payload["training"]["short_circuit"] = None
    for stage in payload["campaign"]["budget_stages"]:
        stage.pop("short_circuit_minimum_episodes")
    path = tmp_path / "stale-frozen-path.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="frozen recipe path does not exist: "
        "training.short_circuit.policy_health",
    ):
        load_experiment_config(path)


def test_relative_only_paired_recurrent_v7_recipe_is_rejected(tmp_path: Path) -> None:
    path = _stage2a_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["regime_selectivity"]["semantics"] = (
        "paired_recurrent_a_plus_expansion_regime_contrastive_v7"
    )
    payload["regime_selectivity"]["formula"] = (
        "equal_present_group_mean(exact_wait_expansion_regime_confluence_"
        "weighted_ce,exact_long_ce,exact_short_ce,dead_vs_transition_positive_"
        "wait_rank,failed_long_vs_valid_long_wait_rank,failed_short_vs_valid_"
        "short_wait_rank)+membership_weighted_mean(all_action_dominant_chop_"
        "wait_margin,failed_long_wait_margin,failed_short_wait_margin)+equal_"
        "present_side_mean(softplus(pair_margin+matched_economic_failure_side_"
        "q_minus_wait-matched_economic_winner_side_q_minus_wait))"
    )
    payload["regime_selectivity"].update({
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
        "paired_a_plus_margin": 0.25,
        "side_balance": {
            "schema": "paired_recurrent_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
    })
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="semantics|contract"):
        load_experiment_config(path)


def test_paired_a_plus_450_recipe_scales_v19_curriculum_proportionally() -> None:
    baseline = json.loads(STAGE2_PAIRED_A_PLUS_RECIPE.read_text())
    candidate = json.loads(STAGE2_PAIRED_A_PLUS_450_RECIPE.read_text())

    stage = candidate["campaign"]["budget_stages"][0]
    assert stage["name"] == "paired_aplus_contrastive_450ep_proportional"
    assert stage["training_episodes"] == 450
    assert stage["validation_episodes"] == 200
    assert candidate["output"].endswith("_450ep_proportional")
    assert candidate["campaign"]["state_root"].endswith(
        "_450ep_proportional/ml-loop-state"
    )
    assert "teacher_schedule_episodes" not in candidate["training"]
    assert candidate["training"]["teacher_autonomy_start_fraction"] == 0.8
    assert candidate["training"][
        "entry_supervision_autonomy_start_fraction"
    ] == 0.95

    candidate["campaign"]["budget_stages"][0]["name"] = (
        baseline["campaign"]["budget_stages"][0]["name"]
    )
    candidate["campaign"]["budget_stages"][0]["training_episodes"] = 100
    candidate["campaign"]["state_root"] = baseline["campaign"]["state_root"]
    candidate["output"] = baseline["output"]
    assert candidate == baseline


def test_replay_fraction_contract_includes_hard_wait_quota(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload["training"].update({
        "terminal_sequence_fraction": 0.5,
        "safety_sequence_fraction": 0.25,
        "entry_opportunity_sequence_fraction": 0.25,
        "regime_wait_sequence_fraction": 0.125,
    })
    path = tmp_path / "invalid-replay-fractions.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="replay sequence fractions"):
        load_experiment_config(path)

    payload = _generic_payload()
    payload["training"].update({
        "terminal_sequence_fraction": 0.375,
        "safety_sequence_fraction": 0.25,
        "entry_opportunity_sequence_fraction": 0.25,
        "regime_wait_sequence_fraction": 0.125,
    })
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires authenticated selectivity"):
        load_experiment_config(path)


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
    payload = _generic_payload()
    payload["entry_supervision"]["action_class_balance"] = balance
    path = tmp_path / "invalid-entry-balance.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="class balance contract"):
        load_experiment_config(path)


def test_legacy_entry_action_recipes_keep_population_weighted_reduction(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["agent"].pop("entry_action_loss_reduction")
    payload["evolution"]["frozen_paths"].remove(
        "agent.entry_action_loss_reduction"
    )
    path = tmp_path / "legacy-entry-reduction.json"
    path.write_text(json.dumps(payload))

    assert (
        load_experiment_config(path)["agent"]["entry_action_loss_reduction"]
        == "population_weighted_mean_v1"
    )


def test_entry_action_loss_reduction_fails_closed_on_unknown_value(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["agent"]["entry_action_loss_reduction"] = "unknown"
    path = tmp_path / "unknown-entry-action-reduction.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry action loss reduction is invalid"):
        load_experiment_config(path)


def test_entry_action_margin_fails_closed_on_invalid_value(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload["agent"]["entry_action_margin"] = -0.1
    path = tmp_path / "invalid-entry-action-margin.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry action margin is invalid"):
        load_experiment_config(path)


def test_auxiliary_gradient_conflict_mode_fails_closed_on_unknown_value(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["agent"]["auxiliary_gradient_conflict_mode"] = "arbitrary"
    path = tmp_path / "invalid-gradient-conflict-mode.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="gradient conflict mode is invalid"):
        load_experiment_config(path)


def test_preserve_opportunity_gradient_conflict_mode_is_config_driven(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["agent"]["auxiliary_gradient_conflict_mode"] = (
        "pcgrad_preserve_opportunity_v2"
    )
    path = tmp_path / "preserve-opportunity-gradient-conflict.json"
    path.write_text(json.dumps(payload))

    loaded = load_experiment_config(path)

    assert (
        loaded["agent"]["auxiliary_gradient_conflict_mode"]
        == "pcgrad_preserve_opportunity_v2"
    )


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
    payload = json.loads(CURRENT_RECIPE.read_text())
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
        match="contract is invalid",
    ):
        load_experiment_config(path)


def test_fresh_entry_action_reduction_contract_rejects_missing_field(
    tmp_path: Path,
) -> None:
    payload = _current_payload()
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
    payload = _current_payload()
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
        (lambda value: value.update({"target_r": 0.0}), "economics"),
        (lambda value: value.update({"collision": "target_first"}), "contract"),
        (lambda value: value.update({"loss_weight": 0.0}), "economics"),
    ),
)
def test_entry_supervision_contract_fails_closed(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    payload = _generic_payload()
    mutate(payload["entry_supervision"])
    path = tmp_path / "invalid-entry-supervision.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"entry supervision {match}"):
        load_experiment_config(path)


def test_entry_supervision_accepts_alternate_relational_recipe(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["challenge"]["per_trade_risk_dollars"] = 450.0
    payload["entry_supervision"].update({
        "decision_count": 3,
        "fill_offsets": [1, 3, 5],
        "risk_dollars": 450.0,
        "launch": {
            "favorable_r": 0.6,
            "adverse_r": 0.3,
            "horizon_bars": 4,
        },
        "continuation": {
            "favorable_r": 0.75,
            "adverse_r": 0.35,
            "horizon_bars": 5,
        },
        "target_r": 3.0,
        "stop_r": 1.25,
        "horizon_bars": 180,
    })
    path = tmp_path / "alternate-entry-recipe.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["entry_supervision"]["fill_offsets"] == (1, 3, 5)
    assert config["entry_supervision"]["target_r"] == 3.0


def test_recovery_curriculum_accepts_challenge_relative_start(
    tmp_path: Path,
) -> None:
    payload = _recovery_curriculum_payload()
    payload["challenge"]["minimum_mll_headroom"] = 300.0
    payload["recovery_curriculum"]["start_state"].update({
        "realized_pnl": -1_500.0,
        "equity_pnl": -1_500.0,
        "session_pnl": -1_500.0,
    })
    path = tmp_path / "alternate-recovery-recipe.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["recovery_curriculum"]["start_state"]["realized_pnl"] == -1_500.0
    assert config["challenge"]["minimum_mll_headroom"] == 300.0


def test_entry_supervision_must_remain_campaign_frozen(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload["evolution"]["frozen_paths"].remove("entry_supervision")
    path = tmp_path / "unfrozen-entry-supervision.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="entry supervision must be frozen"):
        load_experiment_config(path)


def test_centered_entry_distillation_rejects_unauthenticated_centers(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
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
    payload = _generic_payload()
    payload.pop("entry_supervision")
    payload["evolution"]["frozen_paths"].remove("entry_supervision")
    payload["training"].update({
        "teacher_loss_end_scale": 0.1,
        "teacher_guidance_dropout_start": 0.0,
        "teacher_guidance_dropout_end": 0.5,
    })
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


def test_entry_supervision_schedule_defaults_to_legacy_teacher_boundary(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["training"].pop("entry_supervision_autonomy_start_fraction")
    payload["evolution"]["frozen_paths"].remove(
        "training.entry_supervision_autonomy_start_fraction"
    )
    path = tmp_path / "legacy-entry-schedule.json"
    path.write_text(json.dumps(payload))
    config = load_experiment_config(path)

    assert config["training"].get(
        "entry_supervision_autonomy_start_fraction"
    ) == config["training"]["teacher_autonomy_start_fraction"]


def test_distinct_entry_supervision_schedule_must_be_valid_and_frozen(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    schedule_path = "training.entry_supervision_autonomy_start_fraction"
    payload["evolution"]["frozen_paths"].remove(schedule_path)
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
    payload = _generic_payload()
    expected = dict(payload["training"]["short_circuit"])
    expected.pop("collapse")
    payload["training"]["short_circuit"].pop("collapse")
    path = tmp_path / "short-circuit.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["training"]["short_circuit"] == expected

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


def test_active_stage2a_uses_the_frozen_episode_budget() -> None:
    config = load_experiment_config(CURRENT_RECIPE)

    assert config["training"]["budget_mode"] == "episodes"
    assert config["training"]["episodes"] == 500
    stage = config["campaign"]["budget_stages"][0]
    assert stage["budget_mode"] == "episodes"
    assert stage["training_episodes"] == 100
    assert stage["short_circuit_minimum_episodes"] == 18


def test_episode_budget_declares_exact_training_and_short_circuit_boundaries(
    tmp_path: Path,
) -> None:
    payload = _episode_budget_payload()
    path = tmp_path / "episode-budget.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["training"]["short_circuit"][
        "minimum_completed_episodes"
    ] == 18
    assert [
        (stage["budget_mode"], stage["training_episodes"],
         stage["validation_episodes"], stage["short_circuit_minimum_episodes"])
        for stage in config["campaign"]["budget_stages"]
    ] == [
        ("episodes", 200, 200, 18),
        ("episodes", 300, 200, 18),
        ("episodes", 500, 200, 18),
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["training"].update(
                minimum_environment_steps=1
            ),
            "episode training budget",
        ),
        (
            lambda payload: payload["campaign"]["budget_stages"][0].update(
                minimum_environment_steps=1
            ),
            "episode budget stage",
        ),
        (
            lambda payload: payload["campaign"]["budget_stages"][0].update(
                short_circuit_minimum_episodes=201
            ),
            "episode budget stage",
        ),
        (
            lambda payload: payload["campaign"]["budget_stages"][1].update(
                budget_mode="environment_steps",
                minimum_environment_steps=300,
            ),
            "budget stage contract",
        ),
        (
            lambda payload: payload["training"]["short_circuit"].update(
                minimum_environment_steps=18
            ),
            "training short circuit",
        ),
    ),
)
def test_episode_budget_rejects_mixed_or_ambiguous_boundaries(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _episode_budget_payload()
    mutation(payload)
    path = tmp_path / "invalid-episode-budget.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_experiment_config(path)


def test_episode_stage_must_reach_each_frozen_policy_health_boundary(
    tmp_path: Path,
) -> None:
    payload = _episode_budget_payload()
    payload["campaign"]["budget_stages"][0]["training_episodes"] = 40
    payload["training"]["short_circuit"]["policy_health"] = (
        _policy_health_payload()
    )
    payload["evolution"]["frozen_paths"].append(
        TRAINING_POLICY_HEALTH_FROZEN_PATH
    )
    path = tmp_path / "unreachable-policy-health.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="policy health boundary"):
        load_experiment_config(path)


def test_episode_budget_policy_health_contract_is_exact_finite_and_frozen(
    tmp_path: Path,
) -> None:
    payload = _episode_budget_payload()
    payload["training"]["short_circuit"]["policy_health"] = (
        _policy_health_payload()
    )
    payload["evolution"]["frozen_paths"].append(
        "training.short_circuit.policy_health"
    )
    path = tmp_path / "policy-health.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["training"]["short_circuit"]["policy_health"] == (
        _policy_health_payload()
    )

    payload["training"]["short_circuit"]["policy_health"][
        "minimum_probe_recall"
    ]["ENTER_SHORT_1"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="policy health"):
        load_experiment_config(path)

    payload["training"]["short_circuit"]["policy_health"] = (
        _policy_health_payload()
    )
    payload["evolution"]["frozen_paths"] = [
        field for field in payload["evolution"]["frozen_paths"]
        if field not in {
            "training.short_circuit",
            "training.short_circuit.policy_health",
        }
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="policy health must be frozen"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda health: health.update(extra_threshold=1.0),
        lambda health: health.update(minimum_completed_episodes=True),
        lambda health: health["minimum_probe_recall"].pop("WAIT"),
        lambda health: health["entry_mass_fraction"].update(maximum=0.29),
        lambda health: health["economic_futility"].update(
            minimum_failed_conditions=4
        ),
    ),
)
def test_policy_health_rejects_unknown_missing_or_wrong_typed_thresholds(
    tmp_path: Path,
    mutation,
) -> None:
    payload = _episode_budget_payload()
    health = _policy_health_payload()
    mutation(health)
    payload["training"]["short_circuit"]["policy_health"] = health
    payload["evolution"]["frozen_paths"].append(
        TRAINING_POLICY_HEALTH_FROZEN_PATH
    )
    path = tmp_path / "invalid-policy-health.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="policy health"):
        load_experiment_config(path)


def test_final_episode_coverage_is_exact_and_economically_gated(
    tmp_path: Path,
) -> None:
    payload = _episode_budget_payload()
    final = payload["campaign"]["budget_stages"][-1]
    final["episode_coverage"] = {
        "schema": "full_data_episode_coverage_v1",
        "episode_budget": 500,
    }
    final["selection_requirements"].extend([
        {
            "metric": "training.minimum_market_episode_coverage",
            "operator": "==",
            "value": 1.0,
        },
        {
            "metric": "training.episode_coverage_complete",
            "operator": "==",
            "value": 1.0,
        },
    ])
    path = tmp_path / "full-coverage.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["campaign"]["budget_stages"][-1][
        "episode_coverage"
    ] == final["episode_coverage"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["campaign"]["budget_stages"][0].update(
            episode_coverage={
                "schema": "full_data_episode_coverage_v1",
                "episode_budget": 200,
            }
        ),
        lambda payload: payload["campaign"]["budget_stages"][-1].update(
            episode_coverage={
                "schema": "full_data_episode_coverage_v1",
                "episode_budget": 499,
            }
        ),
        lambda payload: payload["campaign"]["budget_stages"][-1].update(
            episode_coverage={
                "schema": "full_data_episode_coverage_v1",
                "episode_budget": 500,
            }
        ),
    ),
)
def test_episode_coverage_fails_closed_on_wrong_stage_budget_or_missing_gates(
    tmp_path: Path,
    mutation,
) -> None:
    payload = _episode_budget_payload()
    mutation(payload)
    path = tmp_path / "invalid-full-coverage.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="episode coverage contract"):
        load_experiment_config(path)


def test_legacy_schema_v1_recipe_keeps_eager_fp32_runtime(tmp_path: Path) -> None:
    payload = _generic_payload()
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


def test_config_accepts_three_frozen_training_only_teachers(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["teachers"] = [
        payload["teachers"][0],
        {
            "kind": "regime",
            "cache_root": (
                "cache/expansion_anchored_regime_teacher_9market_3min_pre2025_v1"
            ),
            "channels": [
                "chop_no_trend_probability",
                "chop_end_transition_probability",
                "expansion_trend_probability",
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
    assert "teachers" in payload["evolution"]["frozen_paths"]
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


def test_trade_management_observation_contract_is_fail_closed(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
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


def test_config_accepts_explicit_soft_target_update(tmp_path: Path) -> None:
    payload = _generic_payload()
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
    payload = _generic_payload()
    payload["agent"]["target_update_mode"] = mode
    payload["agent"]["target_soft_tau"] = tau
    path = tmp_path / "invalid-target-update.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="target update"):
        load_experiment_config(path)


def test_config_rejects_sealed_confirmation_that_can_use_teachers(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["sealed_confirmation"]["teacher_free"] = False
    path = tmp_path / "teacher-leak.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="teacher-free"):
        load_experiment_config(path)


def test_config_accepts_optional_gepa_reflective_reasoning_proposer(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["campaign"]["reasoning"]["proposer"] = "gepa_reflective"
    path = tmp_path / "gepa-reflective.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["campaign"]["reasoning"]["provider"] == "codex"
    assert config["campaign"]["reasoning"]["proposer"] == "gepa_reflective"


def test_config_rejects_unknown_reasoning_proposer(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload["campaign"]["reasoning"]["proposer"] = "unbounded_search"
    path = tmp_path / "invalid-proposer.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="reasoning proposer"):
        load_experiment_config(path)


def test_config_locks_training_only_markets_out_of_deployment(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload["tickers"] = ["NQ", "CL"]
    payload["deployment_tickers"] = ["NQ"]
    payload["training_only_tickers"] = ["CL"]
    payload["point_values"] = {"NQ": 20, "CL": 1000}
    payload["round_trip_fees"] = {"NQ": 3.84, "CL": 4.02}
    payload["sealed_confirmation"]["tickers"] = ["NQ", "CL"]
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload))

    config = load_experiment_config(path)

    assert config["tickers"] == ("NQ", "CL")
    assert config["deployment_tickers"] == ("NQ",)


def test_config_rejects_hidden_agent_default(tmp_path: Path) -> None:
    payload = _generic_payload()
    del payload["agent"]["target_sync_updates"]
    path = tmp_path / "missing-agent-setting.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="agent recipe is missing"):
        load_experiment_config(path)


def test_config_rejects_training_only_market_in_deployment(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload["deployment_tickers"].append("CL")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="training-only"):
        load_experiment_config(path)


def test_config_rejects_revision_allowlist_that_overlaps_frozen_contract(
    tmp_path: Path,
) -> None:
    payload = _generic_payload()
    payload["evolution"]["allowed_revision_paths"].append("temporal.train_start")
    path = tmp_path / "invalid-evolution.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="overlaps"):
        load_experiment_config(path)


def test_config_preserves_declared_trade_risk_and_ratchet_fields(tmp_path: Path) -> None:
    payload = _generic_payload()
    payload.pop("entry_supervision")
    payload["evolution"]["frozen_paths"].remove("entry_supervision")
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
    payload = _generic_payload()
    payload.pop("entry_supervision")
    payload["challenge"].pop("ratchet_activation_r")
    payload["challenge"].pop("ratchet_giveback_r")
    payload["challenge"]["per_trade_risk_dollars"] = 200.0
    path = tmp_path / "partial-ratchet.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="declared together"):
        load_experiment_config(path)


def test_current_expansion_anchored_regime_stage2_restarts_from_stage1() -> None:
    config = load_experiment_config(CURRENT_RECIPE)
    regime = next(
        teacher for teacher in config["teachers"]
        if teacher["kind"] == "regime"
    )

    assert regime["cache_root"] == (
        "cache/expansion_anchored_regime_teacher_9market_3min_pre2025_v1"
    )
    assert tuple(regime["channels"]) == (
        "chop_no_trend_probability",
        "chop_end_transition_probability",
        "expansion_trend_probability",
    )
    assert config["evolution"]["base_parent"] == {
        "archive_root": (
            "runs/historical_mask_expansion_regime_post_launch_entry_balanced_v8b/"
            "archive"
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
