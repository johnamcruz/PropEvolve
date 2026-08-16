from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from propevolve.config import (
    REGIME_SELECTIVITY_FROZEN_IDENTITY_PATHS,
    REGIME_SELECTIVITY_SEMANTICS_REVISION_PATHS,
    TRAINING_POLICY_HEALTH_FROZEN_PATH,
    agent_runtime_settings,
    load_experiment_config,
)
from propevolve.balance_aware_regime_selectivity import (
    ACTION_ORDER as REGIME_SELECTIVITY_ACTION_ORDER,
    EXPANSION_REGIME_CONFLUENCE_FORMULA,
    EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    FORMULA as REGIME_SELECTIVITY_FORMULA,
    SCHEMA as REGIME_SELECTIVITY_SCHEMA,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
    TARGET_SOURCE as REGIME_SELECTIVITY_TARGET_SOURCE,
)


CURRENT_RECIPE = Path(
    "config/historical_mask_expansion_anchored_regime_stage2_v9.json"
)
STAGE2_V4_RECIPE = Path(
    "config/historical_mask_expansion_regime_stage2_v4.json"
)
STAGE2_V5_RECIPE = Path(
    "config/historical_mask_expansion_regime_stage2_v5.json"
)
STAGE2_V6_RECIPE = Path(
    "config/historical_mask_expansion_regime_stage2_v6.json"
)
STAGE2_V7_RECIPE = Path(
    "config/historical_mask_expansion_regime_stage2_v7.json"
)

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
    payload["evolution"]["frozen_paths"] = [
        path for path in payload["evolution"]["frozen_paths"]
        if not path.startswith("regime_selectivity.")
    ]
    return payload


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


@pytest.mark.parametrize(
    "path",
    (STAGE2_V4_RECIPE, STAGE2_V5_RECIPE, STAGE2_V6_RECIPE, STAGE2_V7_RECIPE),
)
def test_retired_eighteen_channel_regime_recipes_fail_closed(path: Path) -> None:
    with pytest.raises(ValueError, match="Regime teacher contract"):
        load_experiment_config(path)


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
        "config/historical_mask_expansion_anchored_regime_stage2_v9.json"
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
    payload = _generic_payload()
    mutate(payload["entry_supervision"])
    path = tmp_path / "invalid-entry-supervision.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"entry supervision {match}"):
        load_experiment_config(path)


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


def test_expansion_anchored_regime_stage2_v9_restarts_from_stage1() -> None:
    config = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v9.json"
    )
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


def test_expansion_anchored_regime_stage2_v10_strengthens_chop_avoidance() -> None:
    baseline = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v9.json"
    )
    candidate = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v10.json"
    )

    assert candidate["regime_selectivity"][
        "persistent_chop_negative_emphasis"
    ] == 2.0
    assert candidate["training"]["teacher_guidance_dropout_end"] == 0.5
    assert candidate["evolution"]["base_parent"] == baseline["evolution"][
        "base_parent"
    ]

    ignored_paths = {
        ("_path",),
        ("campaign", "state_root"),
        ("evolution", "hypothesis"),
        ("output",),
        ("regime_selectivity", "persistent_chop_negative_emphasis"),
        ("training", "teacher_guidance_dropout_end"),
    }

    def flattened(payload, prefix=()):
        values = {}
        for key, value in payload.items():
            path = prefix + (key,)
            if path in ignored_paths:
                continue
            if isinstance(value, dict):
                values.update(flattened(value, path))
            else:
                values[path] = value
        return values

    assert flattened(candidate) == flattened(baseline)


def test_expansion_anchored_regime_stage2_v11_changes_only_wait_confluence() -> None:
    baseline = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v10.json"
    )
    candidate = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v11.json"
    )

    assert candidate["regime_selectivity"]["semantics"] == (
        EXPANSION_REGIME_CONFLUENCE_SEMANTICS
    )
    assert candidate["regime_selectivity"]["formula"] == (
        EXPANSION_REGIME_CONFLUENCE_FORMULA
    )
    assert candidate["evolution"]["base_parent"] == baseline["evolution"][
        "base_parent"
    ]
    assert candidate["teachers"] == baseline["teachers"]
    assert candidate["entry_supervision"] == baseline["entry_supervision"]
    assert candidate["agent"] == baseline["agent"]
    assert candidate["training"] == baseline["training"]
    assert candidate["entry_supervision"]["target_r"] == 2.0
    assert candidate["entry_supervision"]["stop_r"] == 1.0


def test_expansion_anchored_regime_stage2_v12_conditions_wait_by_side() -> None:
    baseline = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v11.json"
    )
    candidate = load_experiment_config(
        "config/historical_mask_expansion_anchored_regime_stage2_v12.json"
    )

    assert candidate["regime_selectivity"]["semantics"] == (
        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS
    )
    assert candidate["regime_selectivity"]["formula"] == (
        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_FORMULA
    )
    assert candidate["evolution"]["base_parent"] == baseline["evolution"][
        "base_parent"
    ]
    assert candidate["teachers"] == baseline["teachers"]
    assert candidate["entry_supervision"] == baseline["entry_supervision"]
    assert candidate["agent"] == baseline["agent"]
    assert candidate["training"] == baseline["training"]
    assert candidate["entry_supervision"]["target_r"] == 2.0
    assert candidate["entry_supervision"]["stop_r"] == 1.0
