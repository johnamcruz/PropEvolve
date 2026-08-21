from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

import propevolve.training as training_module
from propevolve.balance_aware_regime_selectivity import (
    ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
    BalanceAwareRegimeSelectivity,
    EXPANSION_CHANNELS,
    PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
    REGIME_TEACHER_CHANNELS,
    SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
)
from propevolve.cache import build_embedding_cache
from propevolve.decision import Action, PositionSide
from propevolve.environment import ChallengeSpec, ChallengeStartState, MarketSeries
from propevolve.entry_supervision import EntryTargetMetadata
from propevolve.observation import TradeManagementObservationSpec
from propevolve.replay import BalancedSequenceReplay, Episode, Transition
from propevolve.recovery import RecoveryValueStore
from propevolve.training import (
    HistoricalCandidateRunner,
    RecoveryCurriculumSettings,
    RecoveryStressResult,
    TrainingResult,
    TrainingProgress,
    _assert_recovery_entry_balance,
    _assert_recovery_regime_selectivity,
    _entry_action_balance,
    _entry_supervision_frozen_contract,
    _load_healthy_pass_replay_artifact,
    _load_recovery_success_replay_artifact,
    _regime_selectivity_agent_settings,
    _regime_selectivity_replay_settings,
    _recovery_curriculum_from_config,
    _selection_evaluation_gates,
    _plain_contract_value,
    _paired_a_plus_transition_evidence,
    assert_temporal_role,
    evaluate_agent,
    evaluate_recovery_stress,
    prop_safety_objective,
    train_agent,
)
from tests.recipe_fixtures import paired_aplus_recipe


def _recovery_curriculum_settings() -> RecoveryCurriculumSettings:
    return RecoveryCurriculumSettings(
        schedule_seed=37,
        recovery_value_loss_weight=0.25,
        recovery_value_temperature=1.0,
        recovery_value_store_capacity=200,
        target_every_episodes=1,
        supervision_start_pnls=(-2_500.0, -2_000.0, -1_500.0, -1_000.0, -500.0),
        retain_nonnegative_entry_policy=True,
        start_state=ChallengeStartState(
            realized_pnl=-2_000.0,
            equity_pnl=-2_000.0,
            peak_equity_pnl=0.0,
            mll_floor_pnl=-3_000.0,
            passmark_locked=False,
            position_side=PositionSide.FLAT,
            position_size=0,
            session_pnl=-2_000.0,
            trading_days_elapsed=1,
            recovery_success_pnl=0.0,
        ),
    )


def test_json_recovery_curriculum_projects_complete_frozen_start_contract() -> None:
    settings, stress_episodes = _recovery_curriculum_from_config({
        "schedule_seed": 37,
        "stress_evaluation_episodes": 200,
        "recovery_success_pnl": 0.0,
        "action_value_supervision": {
            "loss_weight": 0.25,
            "temperature": 1.0,
            "store_capacity": 200,
            "target_every_episodes": 1,
            "start_pnls": [-2_500, -2_000, -1_500, -1_000, -500],
            "retain_nonnegative_entry_policy": True,
        },
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
    })

    assert settings == _recovery_curriculum_settings()
    assert stress_episodes == 200


def test_json_recovery_curriculum_projects_authenticated_healthy_pass_replay() -> None:
    settings, _ = _recovery_curriculum_from_config({
        "schedule_seed": 37,
        "stress_evaluation_episodes": 200,
        "recovery_success_pnl": 0.0,
        "action_value_supervision": {
            "loss_weight": 0.25,
            "temperature": 1.0,
            "store_capacity": 200,
            "target_every_episodes": 1,
            "start_pnls": [-2_500, -2_000, -1_500, -1_000, -500],
            "retain_nonnegative_entry_policy": True,
            "healthy_pass_replay": {
                "path": "runs/v21-healthy-passes.pt",
                "sha256": "c" * 64,
                "update_period": 8,
                "max_examples": 8,
            },
        },
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
    })

    assert settings is not None
    assert settings.healthy_pass_replay_path == "runs/v21-healthy-passes.pt"
    assert settings.healthy_pass_replay_sha256 == "c" * 64
    assert settings.healthy_pass_replay_update_period == 8
    assert settings.healthy_pass_replay_max_examples == 8


def test_json_recovery_curriculum_rejects_missing_or_drifted_fields() -> None:
    with pytest.raises(ValueError, match="fields are invalid"):
        _recovery_curriculum_from_config({
            "schedule_seed": 37,
        })
    with pytest.raises(ValueError, match="contract drifted"):
        _recovery_curriculum_from_config({
            "schedule_seed": 37,
            "recovery_success_pnl": 0.0,
            "action_value_supervision": {
                "loss_weight": 0.25,
                "temperature": 1.0,
                "store_capacity": 200,
                "target_every_episodes": 1,
                "start_pnls": [
                    -2_500, -2_100, -2_000, -1_500, -1_000, -500
                ],
                "retain_nonnegative_entry_policy": True,
            },
            "start_state": {
                "realized_pnl": -2_100.0,
                "equity_pnl": -2_100.0,
                "peak_equity_pnl": 0.0,
                "mll_floor_pnl": -3_000.0,
                "passmark_locked": False,
                "position_side": 0,
                "position_size": 0,
                "session_pnl": -2_100.0,
                "trading_days_elapsed": 1,
            },
            "stress_evaluation_episodes": 200,
        })


@pytest.mark.parametrize(
    "start_pnls",
    (
        (-2_500.0, -2_000.0, 0.0),
        (-3_000.0, -2_000.0, -500.0),
        (-2_000.0, -2_000.0, -500.0),
        (-2_500.0, -1_500.0, -500.0),
    ),
)
def test_json_recovery_curriculum_rejects_invalid_negative_state_coverage(
    start_pnls: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="start PnLs are invalid"):
        _recovery_curriculum_from_config({
            "schedule_seed": 37,
            "stress_evaluation_episodes": 200,
            "recovery_success_pnl": 0.0,
            "action_value_supervision": {
                "loss_weight": 0.25,
                "temperature": 1.0,
                "store_capacity": 200,
                "target_every_episodes": 1,
                "start_pnls": start_pnls,
                "retain_nonnegative_entry_policy": True,
            },
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
        })


def test_stage2a_recipe_projects_only_declared_selectivity_settings() -> None:
    assert _regime_selectivity_agent_settings({
        "loss_weight": 0.3,
        "expansion_long_center": 0.10249102659218842,
        "expansion_short_center": 0.10399580328775007,
        "probability_epsilon": 1e-6,
        "headroom_pressure": 1.0,
        "dominant_chop_pressure": 2.0,
        "q_temperature": 1.0,
        "semantics": "static_state_v1",
        "persistent_chop_negative_emphasis": 0.0,
        "side_balance": {
            "schema": "equal_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
    }) == {
        "regime_selectivity_loss_weight": 0.3,
        "regime_selectivity_expansion_centers": (
            0.10249102659218842,
            0.10399580328775007,
        ),
        "regime_selectivity_probability_epsilon": 1e-6,
        "regime_selectivity_headroom_pressure": 1.0,
        "regime_selectivity_dominant_chop_pressure": 2.0,
        "regime_selectivity_q_temperature": 1.0,
        "regime_selectivity_semantics": "static_state_v1",
        "regime_selectivity_persistent_chop_negative_emphasis": 0.0,
        "regime_selectivity_side_balance": "equal_long_short_v1",
    }
    assert _regime_selectivity_replay_settings({
        "side_balance": {
            "schema": "equal_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
    }) == {"entry_opportunity_side_balance": "equal_long_short_v1"}


@pytest.mark.parametrize(
    ("side", "target", "economic_win", "economic_good", "expected_win"),
    (
        ("long", Action.ENTER_LONG_1, True, True, True),
        ("short", Action.WAIT, False, False, False),
    ),
)
def test_paired_a_plus_transition_binds_exact_economic_winner_and_failure(
    side: str,
    target: Action,
    economic_win: bool,
    economic_good: bool,
    expected_win: bool,
) -> None:
    context = np.array(
        [0.9, 0.8, 0.1, 0.1, 0.1, 0.7, 0.2], np.float32
    )
    metadata = EntryTargetMetadata(
        side=side,
        event_anchor_rows=(10,),
        candidate_decision_offset=0,
        fill_offset=1,
        continuation=economic_good,
        economic_win=economic_win,
        economic_good=economic_good,
        available=True,
        censored=False,
        unavailable_reason=None,
    )

    actual_context, actual_side, actual_win = _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=(*EXPANSION_CHANNELS, *REGIME_TEACHER_CHANNELS),
        entry_action_target=target,
        metadata=metadata,
    )

    np.testing.assert_array_equal(actual_context, context)
    assert actual_side == (
        Action.ENTER_LONG_1 if side == "long" else Action.ENTER_SHORT_1
    )
    assert actual_win is expected_win


def test_paired_a_plus_transition_does_not_call_nonwinning_setup_a_failure() -> None:
    context = np.array(
        [0.9, 0.8, 0.1, 0.1, 0.1, 0.7, 0.2], np.float32
    )
    metadata = EntryTargetMetadata(
        side="long",
        event_anchor_rows=(10,),
        candidate_decision_offset=0,
        fill_offset=1,
        continuation=False,
        economic_win=True,
        economic_good=False,
        available=True,
        censored=False,
        unavailable_reason=None,
    )

    assert _paired_a_plus_transition_evidence(
        teacher_target=context,
        teacher_channels=(*EXPANSION_CHANNELS, *REGIME_TEACHER_CHANNELS),
        entry_action_target=Action.WAIT,
        metadata=metadata,
    ) == (None, None, None)


def test_stage2a_recipe_projects_chop_specific_wait_margins() -> None:
    settings = _regime_selectivity_agent_settings({
        "loss_weight": 0.3,
        "expansion_long_center": 0.10,
        "expansion_short_center": 0.10,
        "probability_epsilon": 1e-6,
        "headroom_pressure": 1.0,
        "dominant_chop_pressure": 2.0,
        "q_temperature": 1.0,
        "semantics": "side_conditioned_expansion_regime_confluence_v4",
        "persistent_chop_negative_emphasis": 2.0,
        "side_balance": {
            "schema": "equal_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.35,
    })

    assert settings["regime_selectivity_chop_wait_margin"] == 0.25
    assert settings["regime_selectivity_failed_confluence_margin"] == 0.35


def test_stage2a_recipe_projects_configured_paired_a_plus_margin() -> None:
    settings = _regime_selectivity_agent_settings({
        "loss_weight": 0.3,
        "expansion_long_center": 0.10,
        "expansion_short_center": 0.10,
        "probability_epsilon": 1e-6,
        "headroom_pressure": 1.0,
        "dominant_chop_pressure": 2.0,
        "q_temperature": 1.0,
        "semantics": PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        "persistent_chop_negative_emphasis": 2.0,
        "side_balance": {
            "schema": "equal_long_short_v1",
            "action_order": ["ENTER_LONG_1", "ENTER_SHORT_1"],
        },
        "chop_wait_margin": 0.25,
        "failed_confluence_margin": 0.25,
        "paired_a_plus_margin": 0.4,
    })

    assert settings["regime_selectivity_paired_a_plus_margin"] == 0.4


def test_paired_a_plus_diagnostics_preserve_side_and_regime_evidence() -> None:
    prefix = "regime_selectivity_paired_a_plus_"
    diagnostics = training_module._paired_a_plus_diagnostic({
        prefix + "loss": [0.4, 0.2],
        prefix + "active_groups": [2.0, 1.0],
        prefix + "pair_count": [3.0],
        prefix + "pair_mass": [2.0],
        prefix + "good_advantage_sum": [1.0],
        prefix + "bad_advantage_sum": [-0.5],
        prefix + "long_chop_end_transition_pair_count": [1.0],
        prefix + "long_chop_end_transition_pair_mass": [0.75],
        prefix + "long_chop_end_transition_loss_sum": [0.3],
        prefix + "long_chop_end_transition_good_advantage_sum": [0.45],
        prefix + "long_chop_end_transition_bad_advantage_sum": [-0.15],
        prefix + "short_expansion_trend_pair_count": [2.0],
        prefix + "short_expansion_trend_pair_mass": [1.25],
        prefix + "short_expansion_trend_loss_sum": [0.5],
        prefix + "short_expansion_trend_good_advantage_sum": [0.55],
        prefix + "short_expansion_trend_bad_advantage_sum": [-0.35],
    })

    assert diagnostics["loss_mean"] == pytest.approx(0.3)
    assert diagnostics["good_advantage_mean"] == pytest.approx(0.5)
    assert diagnostics["bad_advantage_mean"] == pytest.approx(-0.25)
    assert diagnostics["sides"]["long"]["pair_mass"] == pytest.approx(0.75)
    assert diagnostics["sides"]["short"]["pair_mass"] == pytest.approx(1.25)


def test_paired_recurrent_diagnostics_expose_population_correction_by_side(
) -> None:
    prefix = "regime_selectivity_paired_a_plus_"
    diagnostics = training_module._paired_a_plus_diagnostic({
        prefix + "pair_mass": [2.0],
        prefix + "long_pair_count": [1.0],
        prefix + "long_pair_mass": [1.0],
        prefix + "long_loss_sum": [0.7],
        prefix + "long_good_advantage_sum": [0.2],
        prefix + "long_bad_advantage_sum": [-0.1],
        prefix + "long_winner_population_weight_sum": [0.4],
        prefix + "long_failure_population_weight_sum": [1.6],
        prefix + "short_pair_count": [1.0],
        prefix + "short_pair_mass": [1.0],
        prefix + "short_loss_sum": [0.8],
        prefix + "short_good_advantage_sum": [0.1],
        prefix + "short_bad_advantage_sum": [-0.2],
        prefix + "short_winner_population_weight_sum": [0.35],
        prefix + "short_failure_population_weight_sum": [1.65],
    })

    assert diagnostics["sides"]["long"] == pytest.approx({
        "pair_count": 1.0,
        "pair_mass": 1.0,
        "loss_sum": 0.7,
        "good_advantage_sum": 0.2,
        "good_advantage_mean": 0.2,
        "bad_advantage_sum": -0.1,
        "bad_advantage_mean": -0.1,
        "winner_population_weight_mean": 0.4,
        "failure_population_weight_mean": 1.6,
    })
    assert diagnostics["sides"]["short"]["winner_population_weight_mean"] == (
        pytest.approx(0.35)
    )
    assert diagnostics["sides"]["short"]["failure_population_weight_mean"] == (
        pytest.approx(1.65)
    )


@pytest.mark.parametrize(
    "semantics",
    (
        SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
        ALL_DOMINANT_CHOP_MARGIN_SEMANTICS,
    ),
)
def test_chop_margin_candidate_rejects_any_teacher_free_dominant_chop_entry(
    semantics: str,
) -> None:
    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=semantics,
        chop_wait_margin_active=True,
    )
    dominant_entry_gate = next(
        gate
        for gate in gates
        if gate.metric
        == "final_regime_probe_dominant_chop_greedy_entry_rows"
    )

    assert dominant_entry_gate.passes({
        "final_regime_probe_dominant_chop_greedy_entry_rows": 0.0,
    })
    assert not dominant_entry_gate.passes({
        "final_regime_probe_dominant_chop_greedy_entry_rows": 1.0,
    })


def test_paired_a_plus_gate_requires_decoupled_exact_action_supervision_active(
) -> None:
    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        entry_action_supervision_active=True,
    )
    entry_scale_gate = next(
        gate
        for gate in gates
        if gate.metric == "latest_entry_action_weight_scale"
    )

    assert entry_scale_gate.passes({"latest_entry_action_weight_scale": 1.0})
    assert not entry_scale_gate.passes({"latest_entry_action_weight_scale": 0.0})


def test_final_regime_probe_uses_frozen_selectivity_identity() -> None:
    assert training_module._regime_selectivity_probe_settings({
        "semantics": "persistent_chop_association_v2",
        "expansion_long_center": 0.10249102659218842,
        "expansion_short_center": 0.10399580328775007,
    }) == {
        "regime_selectivity_semantics": "persistent_chop_association_v2",
        "regime_selectivity_expansion_centers": (
            0.10249102659218842,
            0.10399580328775007,
        ),
    }


def test_runner_balance_seam_passes_authenticated_weights_and_archives_receipt() -> None:
    class Targets:
        manifest = {
            "identity_sha256": "a" * 64,
            "action_target_counts": {
                "WAIT": 4,
                "ENTER_LONG_1": 1,
                "ENTER_SHORT_1": 1,
            },
        }

        @staticmethod
        def balance_receipt():
            return {
                "schema": "propevolve_entry_action_balance_v1",
                "method": "inverse_frequency_v1",
                "source_manifest_identity_sha256": "a" * 64,
                "action_order": (
                    "WAIT",
                    "ENTER_LONG_1",
                    "ENTER_SHORT_1",
                ),
                "target_counts": Targets.manifest["action_target_counts"],
                "class_weights": {
                    "WAIT": 0.5,
                    "ENTER_LONG_1": 2.0,
                    "ENTER_SHORT_1": 2.0,
                },
                "identity_sha256": "b" * 64,
            }

    specification = {
        "action_class_balance": {
            "schema": "inverse_frequency_v1",
            "action_order": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
        }
    }
    weights, receipt = _entry_action_balance(Targets(), specification)
    contract = _entry_supervision_frozen_contract(Targets(), receipt)

    assert weights == (0.5, 2.0, 2.0)
    assert receipt is not None
    assert contract == {
        "training_only": True,
        "manifest": Targets.manifest,
        "balance_receipt": _plain_contract_value(receipt),
    }


def test_runner_balance_seam_keeps_the_v8_negative_control_unweighted() -> None:
    weights, receipt = _entry_action_balance(
        object(), {"action_class_balance": None}
    )

    assert weights == (1.0, 1.0, 1.0)
    assert receipt is None


def test_recovery_rejects_entry_balance_drift() -> None:
    class Agent:
        entry_action_class_weights = (0.5, 2.0, 2.0)
        entry_action_loss_reduction = "equal_present_class_mean_v1"

    _assert_recovery_entry_balance(
        Agent(), {
            "entry_action_class_weights": (0.5, 2.0, 2.0),
            "entry_action_loss_reduction": "equal_present_class_mean_v1",
        }
    )
    with pytest.raises(ValueError, match="recovery entry balance drifted"):
        _assert_recovery_entry_balance(
            Agent(), {
                "entry_action_class_weights": (1.0, 1.0, 1.0),
                "entry_action_loss_reduction": "equal_present_class_mean_v1",
            }
        )


def test_recovery_rejects_entry_action_loss_reduction_drift() -> None:
    class Agent:
        entry_action_class_weights = (0.5, 2.0, 2.0)
        entry_action_loss_reduction = "population_weighted_mean_v1"

    with pytest.raises(ValueError, match="recovery entry balance drifted"):
        _assert_recovery_entry_balance(Agent(), {
            "entry_action_class_weights": (0.5, 2.0, 2.0),
            "entry_action_loss_reduction": "equal_present_class_mean_v1",
        })


def test_recovery_rejects_entry_action_margin_drift() -> None:
    class Agent:
        entry_action_class_weights = (0.5, 2.0, 2.0)
        entry_action_loss_reduction = "equal_present_class_mean_v1"
        entry_action_margin = 0.25

    with pytest.raises(ValueError, match="recovery entry balance drifted"):
        _assert_recovery_entry_balance(Agent(), {
            "entry_action_class_weights": (0.5, 2.0, 2.0),
            "entry_action_loss_reduction": "equal_present_class_mean_v1",
            "entry_action_margin": 0.5,
        })


def test_recovery_rejects_regime_learning_identity_drift() -> None:
    class Agent:
        regime_selectivity_semantics = "static_state_v1"
        regime_selectivity_persistent_chop_negative_emphasis = 0.0
        regime_selectivity_side_balance = "equal_long_short_v1"

    expected = {
        "regime_selectivity_semantics": "static_state_v1",
        "regime_selectivity_persistent_chop_negative_emphasis": 0.0,
        "regime_selectivity_side_balance": "equal_long_short_v1",
    }
    _assert_recovery_regime_selectivity(Agent(), expected)

    with pytest.raises(ValueError, match="recovery Regime learning identity drifted"):
        _assert_recovery_regime_selectivity(
            Agent(),
            {**expected, "regime_selectivity_side_balance": "none"},
        )

    class MarginAgent(Agent):
        regime_selectivity_chop_wait_margin = 0.25
        regime_selectivity_failed_confluence_margin = 0.25

    margin_expected = {
        **expected,
        "regime_selectivity_chop_wait_margin": 0.25,
        "regime_selectivity_failed_confluence_margin": 0.25,
    }
    _assert_recovery_regime_selectivity(MarginAgent(), margin_expected)
    with pytest.raises(ValueError, match="recovery Regime learning identity drifted"):
        _assert_recovery_regime_selectivity(
            MarginAgent(),
            {**margin_expected, "regime_selectivity_failed_confluence_margin": 0.5},
        )

    class PairedAgent(MarginAgent):
        regime_selectivity_semantics = PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS
        regime_selectivity_paired_a_plus_margin = 0.25

    paired_expected = {
        **margin_expected,
        "regime_selectivity_semantics": PAIRED_A_PLUS_CONTRASTIVE_SEMANTICS,
        "regime_selectivity_paired_a_plus_margin": 0.25,
    }
    _assert_recovery_regime_selectivity(PairedAgent(), paired_expected)
    with pytest.raises(ValueError, match="recovery Regime learning identity drifted"):
        _assert_recovery_regime_selectivity(
            PairedAgent(),
            {**paired_expected, "regime_selectivity_paired_a_plus_margin": 0.5},
        )


def test_immutable_entry_manifest_becomes_archive_safe_plain_data() -> None:
    from types import MappingProxyType

    value = MappingProxyType({
        "contract": MappingProxyType({"offsets": (1, 2, 3, 4, 5)}),
        "count": np.int64(7),
    })

    plain = _plain_contract_value(value)

    assert plain == {"contract": {"offsets": [1, 2, 3, 4, 5]}, "count": 7}
    json.dumps(plain)


def test_incomplete_selection_cannot_pass_on_earlier_successes() -> None:
    # This represents one early pass followed by five universal-WAIT episodes.
    # Its partial pass-minus-blow result is positive, but evaluation is incomplete.
    metrics = {
        "pass_minus_blow": 1.0 / 6.0,
        "short_circuited": 1.0,
    }

    assert not all(
        gate.passes(metrics) for gate in _selection_evaluation_gates()
    )
    assert all(
        gate.passes({"pass_minus_blow": 0.1, "short_circuited": 0.0})
        for gate in _selection_evaluation_gates()
    )


def test_stage2a_selection_rejects_a_teacher_free_one_side_policy() -> None:
    gates = _selection_evaluation_gates(require_both_entry_sides=True)
    common = {"pass_minus_blow": 0.1, "short_circuited": 0.0}

    assert all(gate.passes({
        **common,
        "long_entry_count": 12.0,
        "short_entry_count": 9.0,
    }) for gate in gates)
    assert not all(gate.passes({
        **common,
        "long_entry_count": 12.0,
        "short_entry_count": 0.0,
    }) for gate in gates)


def test_training_diagnostic_summary_aggregates_side_recall_from_exact_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training-diagnostics.jsonl"
    source.write_text("\n".join(json.dumps({
        "ticker": "NQ",
        "outcome": "timeout",
        "updates": 1,
        "sampled_entry_action_target_counts": {
            "WAIT": 0,
            "ENTER_LONG_1": long_rows,
            "ENTER_SHORT_1": short_rows,
        },
        "sampled_entry_action_prediction_counts": {
            "WAIT": 0,
            "ENTER_LONG_1": long_predictions,
            "ENTER_SHORT_1": short_predictions,
        },
        "sampled_entry_action_correct_counts": {
            "WAIT": 0,
            "ENTER_LONG_1": long_correct,
            "ENTER_SHORT_1": short_correct,
        },
    }) for (
        long_rows,
        short_rows,
        long_predictions,
        short_predictions,
        long_correct,
        short_correct,
    ) in ((8, 2, 4, 2, 4, 1), (2, 8, 2, 4, 1, 4))) + "\n")
    destination = tmp_path / "summary.json"

    training_module._write_training_diagnostic_summary(source, destination)

    overall = json.loads(destination.read_text())["overall"]
    assert overall["sampled_entry_action_target_counts"] == {
        "WAIT": 0,
        "ENTER_LONG_1": 10,
        "ENTER_SHORT_1": 10,
    }
    assert overall["sampled_entry_action_recall"] == {
        "WAIT": 0.0,
        "ENTER_LONG_1": 0.5,
        "ENTER_SHORT_1": 0.5,
    }


def test_training_summary_separates_guidance_and_autonomy_regime_learning(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training-diagnostics.jsonl"
    guided = {
        "ticker": "NQ",
        "outcome": "timeout",
        "terminal_pnl": -300.0,
        "updates": 1,
        "guidance_phase": "guidance",
        "teacher_guidance_eligible_decisions": 10,
        "teacher_guidance_visible_decisions": 6,
        "regime_teacher_channels": {
            "chop_no_trend_probability": {
                "rows": 2,
                "target_probability_sum": 1.6,
                "model_probability_sum": 1.0,
                "absolute_error_sum": 0.6,
                "squared_error_sum": 0.18,
            },
        },
        "entry_action_balance": {
            "wait": {
                "rows": 4,
                "configured_weight": 0.25,
                "weighted_mass": 1.0,
                "unweighted_ce_sum": 4.0,
                "weighted_ce_sum": 1.0,
            },
            "long": {
                "rows": 2,
                "configured_weight": 2.0,
                "weighted_mass": 4.0,
                "unweighted_ce_sum": 2.0,
                "weighted_ce_sum": 4.0,
            },
            "short": {
                "rows": 1,
                "configured_weight": 4.0,
                "weighted_mass": 4.0,
                "unweighted_ce_sum": 1.0,
                "weighted_ce_sum": 4.0,
            },
        },
        "regime_entry_conflict": {
            "long": {
                "rows": 2,
                "target_wait_probability_sum": 1.4,
                "target_declared_side_probability_sum": 0.6,
                "model_wait_probability_sum": 1.0,
                "soft_wait_disagreement_rows": 2,
            },
            "short": {
                "rows": 1,
                "target_wait_probability_sum": 0.2,
                "target_declared_side_probability_sum": 0.8,
                "model_wait_probability_sum": 0.3,
                "soft_wait_disagreement_rows": 0,
            },
        },
    }
    autonomy = {
        "ticker": "NQ",
        "outcome": "pass",
        "terminal_pnl": 6_000.0,
        "updates": 1,
        "guidance_phase": "autonomy",
        "teacher_guidance_eligible_decisions": 0,
        "teacher_guidance_visible_decisions": 0,
    }
    source.write_text(json.dumps(guided) + "\n" + json.dumps(autonomy) + "\n")
    destination = tmp_path / "summary.json"

    training_module._write_training_diagnostic_summary(source, destination)
    summary = json.loads(destination.read_text())

    assert set(summary["by_guidance_phase"]) == {"guidance", "autonomy"}
    assert summary["by_guidance_phase"]["guidance"][
        "teacher_guidance_visible_fraction"
    ] == pytest.approx(0.6)
    assert summary["by_guidance_phase"]["autonomy"][
        "regime_teacher_channels"
    ] == {}
    chop = summary["overall"]["regime_teacher_channels"][
        "chop_no_trend_probability"
    ]
    assert chop["target_probability_mean"] == pytest.approx(0.8)
    assert chop["model_probability_mean"] == pytest.approx(0.5)
    assert chop["mean_absolute_error"] == pytest.approx(0.3)
    assert chop["root_mean_squared_error"] == pytest.approx(0.3)
    balance = summary["overall"]["entry_action_balance"]
    assert balance["wait"]["weighted_mass_fraction"] == pytest.approx(1 / 9)
    assert balance["long"]["weighted_mass_fraction"] == pytest.approx(4 / 9)
    assert balance["short"]["weighted_mass_fraction"] == pytest.approx(4 / 9)
    conflict = summary["overall"]["regime_entry_conflict"]
    assert conflict["long"]["target_wait_probability_mean"] == pytest.approx(0.7)
    assert conflict["long"]["soft_wait_disagreement_rate"] == 1.0
    assert conflict["short"]["soft_wait_disagreement_rate"] == 0.0


def test_persistent_regime_gate_requires_negative_only_coverage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "training-diagnostics.jsonl"
    source.write_text(json.dumps({
        "ticker": "NQ",
        "outcome": "timeout",
        "updates": 1,
        "persistent_regime_selectivity": {
            "exact_wait": {
                "rows": 10.0,
                "weight_sum": 14.0,
                "model_wait_probability_sum": 7.0,
            },
            "persistent_dead_chop": {
                "rows": 4.0,
                "weight_sum": 7.0,
                "model_wait_probability_sum": 3.2,
            },
            "transition_ready": {
                "rows": 2.0,
                "weight_sum": 2.5,
                "model_wait_probability_sum": 0.4,
            },
            "transition_positive_long": {
                "rows": 2.0,
                "declared_side_probability_sum": 1.4,
            },
            "transition_positive_short": {
                "rows": 3.0,
                "declared_side_probability_sum": 2.1,
            },
        },
    }) + "\n")
    destination = tmp_path / "summary.json"

    training_module._write_training_diagnostic_summary(source, destination)
    persistent = json.loads(destination.read_text())["overall"][
        "persistent_regime_selectivity"
    ]
    optimizer_metrics = (
        training_module._persistent_regime_selectivity_evaluation_metrics(
            persistent
        )
    )
    metrics = {
        "short_circuited": 0.0,
        "latest_teacher_weight_scale": 0.0,
        "latest_entry_action_weight_scale": 0.0,
        **optimizer_metrics,
        "sampled_entry_action_long_rows": 10.0,
        "sampled_entry_action_short_rows": 10.0,
        "sampled_entry_action_long_recall": 0.5,
        "sampled_entry_action_short_recall": 0.5,
        "regime_selectivity_positive_long_rows": 10.0,
        "regime_selectivity_positive_short_rows": 10.0,
        "regime_selectivity_positive_long_declared_side_probability_sum": 5.0,
        "regime_selectivity_positive_short_declared_side_probability_sum": 5.0,
        "regime_entry_conflict_long_rows": 10.0,
        "regime_entry_conflict_short_rows": 10.0,
        "regime_entry_conflict_long_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_short_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_long_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_short_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_long_soft_wait_disagreement_rows": 0.0,
        "regime_entry_conflict_short_soft_wait_disagreement_rows": 0.0,
        "final_regime_probe_wait_rows": 32.0,
        "final_regime_probe_long_rows": 32.0,
        "final_regime_probe_short_rows": 32.0,
        "final_regime_probe_long_recall": 0.5,
        "final_regime_probe_short_recall": 0.5,
        "final_regime_probe_wait_recall": 0.75,
        "final_regime_probe_persistent_dead_wait_mass": 12.0,
        "final_regime_probe_transition_ready_wait_mass": 8.0,
        "final_regime_probe_transition_positive_long_mass": 8.0,
        "final_regime_probe_transition_positive_short_mass": 8.0,
        "final_regime_probe_dead_wait_minus_transition_ready_wait": 0.6,
        "final_regime_probe_transition_positive_long_response": 0.2,
        "final_regime_probe_transition_positive_short_response": 0.2,
    }

    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics="persistent_chop_negative_weight_v1",
    )
    assert metrics[
        "regime_selectivity_dead_wait_minus_transition_ready_wait_model_wait"
    ] == pytest.approx(0.6)
    assert all(gate.passes(metrics) for gate in gates)

    # The completed data audit found these score contrasts near chance.  They
    # remain diagnostic evidence; this repair selects only on authenticated
    # negative-only coverage and teacher-free action recall.
    metrics["final_regime_probe_dead_wait_minus_transition_ready_wait"] = 0.0
    metrics["final_regime_probe_transition_positive_long_response"] = 0.0
    metrics["final_regime_probe_transition_positive_short_response"] = 0.0
    assert all(gate.passes(metrics) for gate in gates)

    metrics["latest_entry_action_weight_scale"] = 0.01
    assert not all(gate.passes(metrics) for gate in gates)
    metrics["latest_entry_action_weight_scale"] = 0.0
    metrics["regime_selectivity_exact_wait_weight_mean"] = 1.0
    assert not all(gate.passes(metrics) for gate in gates)
    metrics["regime_selectivity_exact_wait_weight_mean"] = 1.4
    metrics["regime_selectivity_exact_wait_rows"] = 0.0
    assert not all(gate.passes(metrics) for gate in gates)


@pytest.mark.parametrize(
    "metric",
    (
        "final_regime_probe_dead_wait_minus_transition_positive_wait",
        "final_regime_probe_transition_positive_long_response",
        "final_regime_probe_transition_positive_short_response",
    ),
)
@pytest.mark.parametrize("association", (-0.01, 0.0))
@pytest.mark.parametrize(
    "semantics",
    ("persistent_chop_association_v2", "expansion_regime_confluence_v3"),
)
def test_v6_persistent_regime_gate_requires_positive_final_association(
    metric: str,
    association: float,
    semantics: str,
) -> None:
    metrics = {
        "short_circuited": 0.0,
        "latest_teacher_weight_scale": 0.0,
        "latest_entry_action_weight_scale": 0.0,
        "sampled_entry_action_long_rows": 10.0,
        "sampled_entry_action_short_rows": 10.0,
        "sampled_entry_action_long_recall": 0.5,
        "sampled_entry_action_short_recall": 0.5,
        "regime_selectivity_positive_long_rows": 10.0,
        "regime_selectivity_positive_short_rows": 10.0,
        "regime_selectivity_positive_long_declared_side_probability_sum": 5.0,
        "regime_selectivity_positive_short_declared_side_probability_sum": 5.0,
        "regime_selectivity_exact_wait_rows": 10.0,
        "regime_selectivity_exact_wait_weight_mean": 1.2,
        "regime_selectivity_persistent_dead_chop_weight_sum": 10.0,
        "regime_selectivity_transition_ready_weight_sum": 1.0,
        "regime_selectivity_transition_positive_long_rows": 1.0,
        "regime_selectivity_transition_positive_short_rows": 1.0,
        "regime_selectivity_transition_positive_long_declared_side_probability_sum": 1.0,
        "regime_selectivity_transition_positive_short_declared_side_probability_sum": 1.0,
        "regime_selectivity_failed_setup_confluence_rows": 1.0,
        "regime_entry_conflict_long_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_short_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_long_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_short_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_long_rows": 10.0,
        "regime_entry_conflict_short_rows": 10.0,
        "regime_entry_conflict_long_soft_wait_disagreement_rows": 0.0,
        "regime_entry_conflict_short_soft_wait_disagreement_rows": 0.0,
        "final_regime_probe_wait_rows": 32.0,
        "final_regime_probe_long_rows": 32.0,
        "final_regime_probe_short_rows": 32.0,
        "final_regime_probe_long_recall": 0.5,
        "final_regime_probe_short_recall": 0.5,
        "final_regime_probe_wait_recall": 0.75,
        "final_regime_probe_persistent_dead_wait_mass": 12.0,
        "final_regime_probe_transition_ready_wait_mass": 8.0,
        "final_regime_probe_transition_positive_long_mass": 8.0,
        "final_regime_probe_transition_positive_short_mass": 8.0,
        "final_regime_probe_failed_setup_confluence_mass": 4.0,
        "final_regime_probe_dead_wait_minus_transition_ready_wait": -0.2,
        "final_regime_probe_dead_wait_minus_transition_positive_wait": 0.2,
        "final_regime_probe_transition_positive_long_response": 0.2,
        "final_regime_probe_transition_positive_short_response": 0.2,
    }

    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=semantics,
    )

    assert all(gate.passes(metrics) for gate in gates)
    metrics[metric] = association
    assert not all(gate.passes(metrics) for gate in gates)


def test_persistent_regime_gate_rejects_any_positive_entry_soft_wait_veto() -> None:
    """Economic Long/Short truth must never be softened back toward WAIT."""
    metrics = {
        "short_circuited": 0.0,
        "latest_teacher_weight_scale": 0.0,
        "latest_entry_action_weight_scale": 0.0,
        "sampled_entry_action_long_rows": 10.0,
        "sampled_entry_action_short_rows": 10.0,
        "sampled_entry_action_long_recall": 0.5,
        "sampled_entry_action_short_recall": 0.5,
        "regime_selectivity_positive_long_rows": 10.0,
        "regime_selectivity_positive_short_rows": 10.0,
        "regime_selectivity_positive_long_declared_side_probability_sum": 5.0,
        "regime_selectivity_positive_short_declared_side_probability_sum": 5.0,
        "regime_selectivity_exact_wait_rows": 10.0,
        "regime_selectivity_exact_wait_weight_mean": 1.2,
        "regime_selectivity_persistent_dead_chop_weight_sum": 10.0,
        "regime_selectivity_transition_ready_weight_sum": 1.0,
        "regime_selectivity_transition_positive_long_rows": 1.0,
        "regime_selectivity_transition_positive_short_rows": 1.0,
        "regime_selectivity_transition_positive_long_declared_side_probability_sum": 1.0,
        "regime_selectivity_transition_positive_short_declared_side_probability_sum": 1.0,
        "regime_entry_conflict_long_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_short_target_wait_probability_mean": 0.0,
        "regime_entry_conflict_long_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_short_target_declared_side_probability_mean": 1.0,
        "regime_entry_conflict_long_rows": 10.0,
        "regime_entry_conflict_short_rows": 10.0,
        "regime_entry_conflict_long_soft_wait_disagreement_rows": 0.0,
        "regime_entry_conflict_short_soft_wait_disagreement_rows": 0.0,
        "final_regime_probe_wait_rows": 32.0,
        "final_regime_probe_long_rows": 32.0,
        "final_regime_probe_short_rows": 32.0,
        "final_regime_probe_long_recall": 0.5,
        "final_regime_probe_short_recall": 0.5,
        "final_regime_probe_wait_recall": 0.75,
        "final_regime_probe_persistent_dead_wait_mass": 12.0,
        "final_regime_probe_transition_ready_wait_mass": 8.0,
        "final_regime_probe_transition_positive_long_mass": 8.0,
        "final_regime_probe_transition_positive_short_mass": 8.0,
        "final_regime_probe_dead_wait_minus_transition_ready_wait": 0.2,
        "final_regime_probe_transition_positive_long_response": 0.2,
        "final_regime_probe_transition_positive_short_response": 0.2,
    }
    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics="persistent_chop_negative_weight_v1",
    )

    assert all(gate.passes(metrics) for gate in gates)
    metrics["regime_entry_conflict_short_target_wait_probability_mean"] = 0.01
    metrics["regime_entry_conflict_short_soft_wait_disagreement_rows"] = 1.0
    assert not all(gate.passes(metrics) for gate in gates)


def test_equal_present_class_reduction_gate_requires_all_three_equal_loss_masses() -> None:
    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=False,
        entry_action_loss_reduction="equal_present_class_mean_v1",
        entry_action_supervision_active=True,
    )
    metrics = {
        "short_circuited": 0.0,
        "entry_balance_wait_rows": 100.0,
        "entry_balance_long_rows": 100.0,
        "entry_balance_short_rows": 100.0,
        "entry_balance_wait_weighted_mass_fraction": 1.0 / 3.0,
        "entry_balance_long_weighted_mass_fraction": 1.0 / 3.0,
        "entry_balance_short_weighted_mass_fraction": 1.0 / 3.0,
    }

    assert all(gate.passes(metrics) for gate in gates)

    metrics["entry_balance_wait_rows"] = 0.0
    assert not all(gate.passes(metrics) for gate in gates)
    metrics["entry_balance_wait_rows"] = 100.0

    metrics["entry_balance_wait_weighted_mass_fraction"] = 0.11
    assert not all(gate.passes(metrics) for gate in gates)


class Agent:
    def __init__(self) -> None:
        self.updates = 0
        self.retention_calls = 0

    def retain_policy(self) -> None:
        self.retention_calls += 1

    def select_action(
        self,
        observation,
        *,
        hidden,
        valid_actions,
        epsilon,
        return_action_values=False,
    ):
        return Action.WAIT, None, np.zeros(len(Action), np.float32)

    def train_batch(
        self,
        sequences,
        *,
        teacher_weight_scale=1.0,
        entry_action_weight_scale=1.0,
    ):
        self.updates += 1
        self.teacher_weight_scales = getattr(self, "teacher_weight_scales", [])
        self.teacher_weight_scales.append(teacher_weight_scale)
        self.entry_action_weight_scales = getattr(
            self, "entry_action_weight_scales", []
        )
        self.entry_action_weight_scales.append(entry_action_weight_scale)
        self.last_train_metrics = {
            "rl_loss": 0.4,
            "teacher_loss": 0.1,
            "entry_search_loss": 0.2,
            "gradient_norm": 1.25,
            "sampled_management_row_fraction": 0.75,
            "sampled_hold_reward": 0.03,
            "sampled_close_reward": -0.02,
            "sampled_hold_n_step_return": 0.12,
            "sampled_close_n_step_return": -0.08,
            "sampled_hold_td_loss": 2.1,
            "sampled_close_td_loss": 2.4,
            "management_hold_minus_close_q": 0.15,
            "sampled_management_close_fraction": 0.2,
            "entry_action_target_wait_rows": 4.0,
            "entry_action_target_long_rows": 1.0,
            "entry_action_target_short_rows": 1.0,
            "entry_action_prediction_wait_rows": 3.0,
            "entry_action_prediction_long_rows": 2.0,
            "entry_action_prediction_short_rows": 1.0,
            "entry_action_correct_wait_rows": 3.0,
            "entry_action_correct_long_rows": 1.0,
            "entry_action_correct_short_rows": 1.0,
        }
        return 0.5


class Environment:
    def __init__(self) -> None:
        self.index = 0

    def reset(self):
        self.index = 0
        return np.array([0.0], np.float32), {"valid_actions": (Action.WAIT,)}

    def step(self, action):
        self.index += 1
        terminated = self.index == 4
        info = {
            "valid_actions": () if terminated else (Action.WAIT,),
            "outcome": "pass" if terminated else None,
            "ticker": "NQ",
            "primary_side": "flat",
            "trade_count": 2 if terminated else 0,
            "win_count": 1 if terminated else 0,
            "winning_r_sum": 2.5 if terminated else 0.0,
            "equity_pnl": 6_000.0 if terminated else 0.0,
            "largest_realized_trade": (
                {
                    "side": "long",
                    "realized_r": 3.0,
                    "mfe_r": 3.5,
                    "mae_r": 0.25,
                    "hold_bars": 20,
                    "ratchet_activated": True,
                    "exit_reason": "ratchet_stop",
                }
                if terminated else None
            ),
            "largest_mfe_trade": (
                {
                    "side": "long",
                    "realized_r": 3.0,
                    "mfe_r": 3.5,
                    "mae_r": 0.25,
                    "hold_bars": 20,
                    "ratchet_activated": True,
                    "exit_reason": "ratchet_stop",
                }
                if terminated else None
            ),
        }
        return np.array([self.index], np.float32), 0.25, terminated, False, info


class MultiMarketEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()
        self.episode_tickers = []
        self.ticker = ""

    def reset(self, *, options=None):
        self.index = 0
        self.ticker = options["ticker"]
        self.episode_tickers.append(self.ticker)
        return np.array([0.0], np.float32), {
            "ticker": self.ticker,
            "valid_actions": (Action.WAIT,),
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["ticker"] = self.ticker
        return observation, reward, terminated, truncated, info


def test_historical_candidate_flow_materializes_the_challenge_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReachedEnvironment(RuntimeError):
        pass

    captured = []

    class CapturingEnvironment:
        def __init__(
            self,
            markets,
            *,
            tick_values,
            round_trip_fees,
            spec,
            observation_spec,
            seed,
            episode_coverage=None,
        ):
            captured.append((spec, observation_spec, episode_coverage))
            raise ReachedEnvironment

    monkeypatch.setattr(
        training_module.AssetContract,
        "load",
        classmethod(lambda cls, path: object()),
    )
    monkeypatch.setattr(
        training_module,
        "load_markets",
        lambda **kwargs: {"NQ": object()},
    )
    monkeypatch.setattr(training_module, "assert_temporal_role", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        training_module,
        "HistoricalChallengeEnv",
        CapturingEnvironment,
    )
    config = {
        "_root": ".",
        "assets": "config/local-assets.json",
        "cache_root": "cache",
        "tickers": ("NQ",),
        "deployment_tickers": ("NQ",),
        "timeframe_minutes": 3,
        "temporal": {
            "train_start": "2021-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01",
        },
        "challenge": {
            "profit_target": 6000.0,
            "max_loss": 3000.0,
            "episode_days": 30,
            "bars_per_day": 480,
            "max_position_size": 1,
            "minimum_mll_headroom": 500.0,
            "trailing_mll_lock": True,
            "terminal_pass_reward": 250.0,
            "terminal_blow_reward": -1500.0,
            "terminal_timeout_reward": -2.0,
            "terminal_pass_speed_reward_per_day": 20.0,
            "reward_scale": 1000.0,
            "mll_proximity_penalty_coefficient": 0.0,
            "lead_giveback_penalty_coefficient": 0.0,
            "large_win_threshold_r": 2.0,
            "large_win_bonus_coefficient": 0.0,
        },
        "training": {"seed": 7},
        "point_values": {"NQ": 20.0},
        "round_trip_fees": {"NQ": 3.84},
    }

    with pytest.raises(ReachedEnvironment):
        HistoricalCandidateRunner().run(
            config,
            parent_candidate_ids=(),
            hypothesis="flow regression",
        )

    assert len(captured) == 1
    assert isinstance(captured[0][0], ChallengeSpec)
    assert captured[0][1] == TradeManagementObservationSpec()
    assert captured[0][2] is None


def test_historical_candidate_runs_the_complete_real_training_flow(
    tmp_path: Path,
) -> None:
    class TinyEncoder:
        checkpoint = tmp_path / "checkpoint"

        def encode(self, windows: np.ndarray) -> np.ndarray:
            return windows.mean(axis=2)

    data = tmp_path / "data"
    data.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"tiny-mask")
    adapter = checkpoint / "adapter_config.json"
    adapter.write_text("{}\n")
    source = data / "NQ_3min.csv"
    times = pd.to_datetime([
        "2024-12-31T23:39:00Z",
        "2024-12-31T23:42:00Z",
        "2024-12-31T23:45:00Z",
        "2024-12-31T23:48:00Z",
        "2025-01-01T00:00:00Z",
        "2025-01-01T00:03:00Z",
        "2025-01-01T00:06:00Z",
        "2025-01-01T00:09:00Z",
        "2025-01-01T00:12:00Z",
    ])
    close = np.arange(100, 109, dtype=float)
    pd.DataFrame({
        "datetime": times,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.arange(10, 19, dtype=float),
    }).to_csv(source, index=False)
    cache_root = tmp_path / "cache"
    build_embedding_cache(
        source=source,
        destination=cache_root / "NQ",
        ticker="NQ",
        encoder=TinyEncoder(),
        checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        research_end_exclusive="2026-01-01",
        context_length=2,
        stride=1,
        chunk_windows=4,
        timeframe_minutes=3,
    )
    assets_path = tmp_path / "assets.json"
    assets_path.write_text(json.dumps({
        "schema": "propevolve_local_assets_v1",
        "market_data": str(data),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "adapter_config_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "embedding_cache": None,
    }))
    config_path = tmp_path / "experiment.json"
    config_path.write_text("{}")
    config = {
        "_root": str(tmp_path),
        "_path": str(config_path),
        "assets": str(assets_path),
        "cache_root": str(cache_root),
        "output": str(tmp_path / "run"),
        "tickers": ("NQ",),
        "deployment_tickers": ("NQ",),
        "training_only_tickers": (),
        "timeframe_minutes": 3,
        "temporal": {
            "train_start": "2024-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01",
        },
        "challenge": {
            "profit_target": 6000.0,
            "max_loss": 3000.0,
            "episode_days": 1,
            "bars_per_day": 2,
            "max_position_size": 1,
            "minimum_mll_headroom": 500.0,
            "trailing_mll_lock": True,
            "terminal_pass_reward": 250.0,
            "terminal_blow_reward": -1500.0,
            "terminal_timeout_reward": -2.0,
            "terminal_pass_speed_reward_per_day": 20.0,
            "reward_scale": 1000.0,
            "per_trade_risk_dollars": 300.0,
            "ratchet_activation_r": 2.0,
            "ratchet_giveback_r": 0.5,
        },
        "recovery_curriculum": None,
        "point_values": {"NQ": 20.0},
        "round_trip_fees": {"NQ": 3.84},
        "agent": {
            "hidden_dim": 8,
            "atoms": 11,
            "value_min": -3.0,
            "value_max": 3.0,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 10.0,
            "target_sync_updates": 2,
            "entry_action_loss_reduction": "equal_present_class_mean_v1",
            "device": "cpu",
        },
        "runtime": {
            "mixed_precision": "off",
            "compile_model": False,
            "compile_backend": "inductor",
            "compile_mode": "default",
            "mps_prefer_metal": False,
            "mps_fast_math": False,
        },
        "training": {
            "episodes": 2,
            "budget_mode": "episodes",
            "episode_coverage": {
                "schema": "full_data_episode_coverage_v1",
                "episode_budget": 2,
            },
            "validation_episodes": 1,
            "replay_capacity_episodes": 2,
            "replay_capacity_transitions": 8,
            "sequence_length": 1,
            "terminal_sequence_fraction": 0.5,
            "warmup_episodes": 1,
            "updates_per_episode": 1,
            "batch_sequences": 1,
            "recurrent_horizon": 2,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "seed": 7,
            "checkpoint_every_episodes": 1,
            "prefetch_batches": 0,
        },
    }

    candidate, evaluation = HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=(),
        hypothesis="complete flow regression",
    )

    assert candidate.model_path.is_file()
    from propevolve.agent import RecurrentC51Agent

    archived_agent, archived_manifest = RecurrentC51Agent.load(
        candidate.model_path,
        device="cpu",
    )
    archived_agent.assert_teacher_free()
    archived_payload = torch.load(
        candidate.model_path,
        map_location="cpu",
        weights_only=False,
    )
    assert archived_agent.retention_anchor is None
    assert archived_payload["retention_anchor"] is None
    assert not any(
        key.startswith("teacher_output.")
        for network in ("online", "target")
        for key in archived_payload[network]
    )
    assert "replay_state" not in archived_payload
    assert "teacher_targets" not in archived_payload
    assert "replay_state" not in archived_manifest
    assert "teacher_targets" not in archived_manifest
    assert evaluation.path.is_file()
    contract = json.loads((candidate.path / "contract.json").read_text())
    assert (
        contract["entry_action_loss_reduction"]
        == "equal_present_class_mean_v1"
    )
    assert contract["recovery_curriculum"] == config["recovery_curriculum"]
    coverage_path = tmp_path / "run" / "episode-coverage-receipt.json"
    assert coverage_path.is_file()
    coverage = json.loads(coverage_path.read_text())
    assert coverage["complete"] is True
    assert coverage["episodes_consumed"] == 2
    assert coverage["markets"]["NQ"]["coverage_fraction"] == 1.0
    assert contract["episode_coverage"]["receipt"] == coverage
    assert contract["episode_coverage"]["file_sha256"] == hashlib.sha256(
        coverage_path.read_bytes()
    ).hexdigest()
    assert contract["training_resume_identity"]
    assert set(contract["runtime_source_modules_sha256"]) >= {
        "training.py",
        "agent.py",
        "config.py",
        "decision.py",
        "replay.py",
        "environment.py",
        "observation.py",
        "evolution.py",
        "teachers/composition.py",
        "teachers/expansion.py",
        "teachers/regime.py",
    }
    assert all(
        len(value) == 64
        for value in contract["runtime_source_modules_sha256"].values()
    )
    recovery = torch.load(
        tmp_path / "run" / "training-recovery.pt",
        map_location="cpu",
        weights_only=False,
    )
    replay_state = recovery["manifest"]["replay_state"]
    assert recovery["manifest"]["replay_restored"] is True
    assert len(replay_state["episodes"]) == 2
    assert replay_state["contract"]["sequence_length"] == 1
    diagnostic_path = tmp_path / "run" / "training-diagnostics.jsonl"
    assert diagnostic_path.is_file()
    diagnostics = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert diagnostics
    assert diagnostics[-1]["schema"] == "propevolve_episode_diagnostic_v1"
    assert diagnostics[-1]["n_step_return"] == 1
    assert diagnostics[-1]["recurrent_burn_in"] == 0
    assert diagnostics[-1]["mean_sampled_recurrent_reset_fraction"] is not None
    assert diagnostics[-1]["mean_sampled_burn_in_reset_coverage"] is not None
    assert diagnostics[-1]["mean_policy_retention_loss"] == 0.0
    validation_diagnostic_path = tmp_path / "run" / "validation-diagnostics.jsonl"
    assert validation_diagnostic_path.is_file()
    validation_diagnostics = [
        json.loads(line)
        for line in validation_diagnostic_path.read_text().splitlines()
    ]
    assert len(validation_diagnostics) == 1
    assert validation_diagnostics[0]["schema"] == (
        "propevolve_validation_episode_diagnostic_v1"
    )
    assert contract["validation_diagnostics"] == {
        "schema": "propevolve_validation_episode_diagnostic_v1",
        "path": str(validation_diagnostic_path),
        "file_sha256": hashlib.sha256(
            validation_diagnostic_path.read_bytes()
        ).hexdigest(),
    }
    summary_path = tmp_path / "run" / "training-diagnostic-summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["schema"] == "propevolve_training_diagnostic_summary_v1"
    assert summary["source_sha256"] == hashlib.sha256(
        diagnostic_path.read_bytes()
    ).hexdigest()
    assert summary["overall"]["episodes"] == len(diagnostics)
    assert summary["by_ticker"]["NQ"]["episodes"] == len(diagnostics)
    assert summary["by_outcome"]["timeout"]["episodes"] == len(diagnostics)
    assert set(summary["overall"]) >= {
        "mean_gradient_norm",
        "sampled_management_row_fraction",
        "sampled_hold_reward",
        "sampled_close_reward",
        "sampled_hold_n_step_return",
        "sampled_close_n_step_return",
        "sampled_hold_td_loss",
        "sampled_close_td_loss",
        "management_hold_minus_close_q",
        "sampled_management_close_fraction",
        "sampled_recurrent_reset_fraction",
        "sampled_burn_in_reset_coverage",
        "sampled_recurrent_reset_pattern_count",
        "policy_retention_loss",
    }
    assert evaluation.candidate_id == candidate.candidate_id
    assert evaluation.status in {"PASS", "FAIL", "REVISE"}
    assert set(evaluation.metrics) >= {
        "training.pass_rate",
        "training.episode_coverage_complete",
        "training.minimum_market_episode_coverage",
        "selection.pass_rate",
        "selection.blow_rate",
        "selection.pass_minus_blow",
        "selection.timeout_mean_trade_count",
        "selection.timeout_trade_win_rate",
        "selection.timeout_average_win_r",
        "selection.timeout_mean_terminal_pnl",
        "selection.average_mfe_r",
        "selection.expectancy_r",
        "selection.near_blow_timeout_rate",
        "selection.average_mae_r",
        "selection.mfe_capture_ratio",
        "selection.gave_it_all_back_rate",
        "selection.two_r_mfe_capture_ratio",
    }

    # A killed validation leaves valid forensic evidence, but validation itself
    # is not resumable: restart it from episode one without appending duplicates.
    partial_payload = {
        "schema": "propevolve_validation_episode_diagnostic_v1",
        "episode": 1,
        "ticker": "NQ",
        "outcome": "timeout",
        "partial": True,
    }
    partial_bytes = (
        json.dumps(partial_payload, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    validation_diagnostic_path.write_bytes(partial_bytes)
    partial_sha256 = hashlib.sha256(partial_bytes).hexdigest()

    resumed_candidate, _ = HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=(),
        hypothesis="complete flow regression",
    )

    preserved_partial = (
        tmp_path
        / "run"
        / f"validation-diagnostics.partial-{partial_sha256}.jsonl"
    )
    assert preserved_partial.read_bytes() == partial_bytes
    fresh_rows = [
        json.loads(line)
        for line in validation_diagnostic_path.read_text().splitlines()
    ]
    assert [row["episode"] for row in fresh_rows] == [1]
    assert all("partial" not in row for row in fresh_rows)
    resumed_contract = json.loads(
        (resumed_candidate.path / "contract.json").read_text()
    )
    assert resumed_contract["validation_diagnostics"]["file_sha256"] == (
        hashlib.sha256(validation_diagnostic_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("stop_kind", "health_probe_milestone", "minimum_passes"),
    (
        ("health", 1, 0),
        ("outcome", 3, 1),
        (None, 3, 0),
    ),
)
def test_runner_finalizes_short_circuit_when_balanced_probe_rows_are_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_kind: str | None,
    health_probe_milestone: int,
    minimum_passes: int,
) -> None:
    class Assets:
        checkpoint_sha256 = "a" * 64

    class Targets:
        channels: tuple[str, ...] = ()

        @staticmethod
        def target(ticker: str, row: int) -> np.ndarray:
            del ticker, row
            return np.zeros(len(Targets.channels), dtype=np.float32)

    class TinyEnvironment:
        observation_dim = 1

        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def reset(self, *, options=None):
            ticker = "NQ" if options is None else options.get("ticker", "NQ")
            return np.zeros(1, dtype=np.float32), {
                "ticker": ticker,
                "start": 0,
                "end": 1,
                "valid_actions": (Action.WAIT,),
            }

        def step(self, action):
            assert action == Action.WAIT
            return np.ones(1, dtype=np.float32), 0.0, True, False, {
                "ticker": "NQ",
                "outcome": "timeout",
                "primary_side": "flat",
                "equity_pnl": 0.0,
                "valid_actions": (),
            }

        @staticmethod
        def rng_state() -> dict[str, object]:
            return {"state": "tiny"}

    class TinyAgent:
        recurrent_burn_in = 0
        n_step_return = 1

        def __init__(self, observation_dim: int, **settings) -> None:
            assert observation_dim == 1
            self.entry_action_loss_reduction = settings[
                "entry_action_loss_reduction"
            ]
            self.entry_action_class_weights = (1.0, 1.0, 1.0)
            self.regime_selectivity_loss_weight = settings[
                "regime_selectivity_loss_weight"
            ]

        def select_action(self, observation, **kwargs):
            del observation, kwargs
            return Action.WAIT, None, None

        def train_batch(self, sequences, **kwargs) -> float:
            del sequences, kwargs
            self.last_train_metrics = {
                "rl_loss": 0.5,
                "gradient_norm": 1.0,
                **{
                    f"entry_balance_{action}_{field}": value
                    for action in ("wait", "long", "short")
                    for field, value in {
                        "rows": 1.0,
                        "weighted_mass": 1.0,
                        "unweighted_ce_sum": 1.0,
                        "weighted_ce_sum": 1.0,
                    }.items()
                },
            }
            return 0.5

        @staticmethod
        def save(path: Path, *, manifest) -> None:
            del manifest
            path.write_bytes(b"tiny-health-stop-policy")

        @staticmethod
        def discard_retention_anchor() -> None:
            return None

        @staticmethod
        def discard_teacher() -> None:
            return None

    recipe_path = paired_aplus_recipe(100)
    config = json.loads(recipe_path.read_text())
    Targets.channels = tuple(
        channel
        for teacher in config["teachers"]
        for channel in teacher["channels"]
    )
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_text("{}\n")
    config.update({
        "_root": str(tmp_path),
        "_path": str(experiment_path),
        "assets": "assets.json",
        "cache_root": "cache",
        "output": "run",
        "tickers": ["NQ"],
        "deployment_tickers": ["NQ"],
        "training_only_tickers": [],
        "recovery_curriculum": None,
        "entry_supervision": None,
    })
    config["temporal"] = {
        "train_start": "2021-01-01",
        "train_end": "2025-01-01",
        "validation_start": "2025-01-01",
        "validation_end": "2026-01-01",
        "sealed_start": "2026-01-01",
    }
    config["challenge"].update({"episode_days": 1, "bars_per_day": 1})
    config["agent"].update({"device": "cpu", "hidden_dim": 8, "atoms": 11})
    config["training"].update({
        "episodes": 2,
        "budget_mode": "episodes",
        "validation_episodes": 1,
        "replay_capacity_episodes": 4,
        "replay_capacity_transitions": 8,
        "sequence_length": 1,
        "warmup_episodes": 1,
        "updates_per_episode": 1,
        "batch_sequences": 1,
        "recurrent_horizon": 1,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "management_epsilon_start": 0.0,
        "management_epsilon_end": 0.0,
            "checkpoint_every_episodes": 0,
            "prefetch_batches": 0,
            "regime_wait_sequence_update_period": 0,
            "episode_coverage": None,
        "short_circuit": {
            "minimum_passes": minimum_passes,
            "maximum_blow_rate": 1.0,
            "minimum_completed_episodes": 1,
            "policy_health": {
                "schema": "propevolve_training_policy_health_v1",
                "minimum_completed_episodes": health_probe_milestone,
                "probe_interval_episodes": health_probe_milestone,
                "minimum_probe_recall": {
                    "WAIT": 0.35,
                    "ENTER_LONG_1": 0.3,
                    "ENTER_SHORT_1": 0.3,
                },
                "entry_mass_fraction": {"minimum": 0.3, "maximum": 0.36},
                "require_zero_positive_entry_soft_wait_veto": True,
                "economic_futility": {
                    "minimum_completed_episodes": 2,
                    "maximum_near_blow_timeout_rate": 1.0,
                    "maximum_mean_terminal_pnl": -1_500.0,
                    "maximum_expectancy_r": -0.15,
                    "minimum_failed_conditions": 2,
                },
            },
        },
    })
    for teacher in config["teachers"]:
        teacher["cache_root"] = "teacher-cache"
    for relative in ("cache/NQ", "teacher-cache/NQ"):
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text("{}\n")

    import propevolve.agent as agent_module
    import propevolve.teachers as teachers_module
    import propevolve.teachers.expansion as expansion_module

    monkeypatch.setattr(agent_module, "RecurrentC51Agent", TinyAgent)
    monkeypatch.setattr(
        training_module.AssetContract,
        "load",
        classmethod(lambda cls, path: Assets()),
    )
    monkeypatch.setattr(
        training_module,
        "load_markets",
        lambda **kwargs: {"NQ": object()},
    )
    monkeypatch.setattr(
        training_module,
        "assert_temporal_role",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(training_module, "HistoricalChallengeEnv", TinyEnvironment)
    monkeypatch.setattr(
        teachers_module,
        "load_teacher_targets",
        lambda *args, **kwargs: Targets(),
    )
    monkeypatch.setattr(
        expansion_module,
        "verify_expansion_entry_center_receipt",
        lambda *args, **kwargs: None,
    )

    if stop_kind is None:
        with pytest.raises(
            ValueError,
            match="final Regime probe lacks exact balanced authentic rows",
        ):
            HistoricalCandidateRunner().run(
                config,
                parent_candidate_ids=(),
                hypothesis="insufficient non-health probe finalization",
            )
        return

    candidate, evaluation = HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=(),
        hypothesis="insufficient health probe finalization",
    )

    health_path = tmp_path / "run" / "training-policy-health.jsonl"
    health = json.loads(health_path.read_text().splitlines()[-1])
    contract = json.loads((candidate.path / "contract.json").read_text())
    if stop_kind == "health":
        assert health["stop"] is True
        assert health["probe_error"] == (
            "final Regime probe lacks exact balanced authentic rows"
        )
    else:
        assert health["stop"] is False
        assert health["probe_error"] is None
    assert evaluation.status == "FAIL"
    assert evaluation.metrics["training.short_circuited"] == 1.0
    assert [stage["name"] for stage in evaluation.stages] == ["training"]
    assert (tmp_path / "run" / "training-recovery.pt").is_file()
    assert (tmp_path / "run" / "training-diagnostic-summary.json").is_file()
    assert contract["training_policy_health"]["file_sha256"] == hashlib.sha256(
        health_path.read_bytes()
    ).hexdigest()
    assert contract["final_regime_probe"] is None
    assert not any("final_regime_probe" in key for key in evaluation.metrics)
    assert not (tmp_path / "run" / "final-regime-probe.json").exists()


def test_training_collects_episodes_then_updates_from_balanced_replay(capsys) -> None:
    agent = Agent()
    replay = BalancedSequenceReplay(capacity_episodes=10, sequence_length=2, seed=1)
    diagnostics = []

    result = train_agent(
        agent,
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=replay,
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=1,
        prefetch_batches=1,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert result.passes == 2
    assert result.environment_steps == 8
    assert result.blows == result.timeouts == 0
    assert len(replay) == 2
    assert agent.updates == 2
    assert diagnostics[-1]["sampled_entry_action_target_counts"] == {
        "WAIT": 4,
        "ENTER_LONG_1": 1,
        "ENTER_SHORT_1": 1,
    }
    assert diagnostics[-1]["sampled_entry_action_recall"] == {
        "WAIT": 0.75,
        "ENTER_LONG_1": 1.0,
        "ENTER_SHORT_1": 1.0,
    }
    assert diagnostics[-1]["sampled_entry_action_precision"] == {
        "WAIT": 1.0,
        "ENTER_LONG_1": 0.5,
        "ENTER_SHORT_1": 1.0,
    }
    assert result.mean_loss == 0.5
    assert result.trade_win_rate == 0.5
    assert result.average_win_r == 2.5
    assert len(diagnostics) == 2
    assert diagnostics[-1]["schema"] == "propevolve_episode_diagnostic_v1"
    assert diagnostics[-1]["episode"] == 2
    assert diagnostics[-1]["outcome"] == "pass"
    assert diagnostics[-1]["expectancy_r"] == 0.0
    assert diagnostics[-1]["avg_mae_r"] == 0.0
    assert diagnostics[-1]["mfe_capture_ratio"] == 0.0
    assert diagnostics[-1]["gave_it_all_back_rate"] == 0.0
    assert diagnostics[-1]["entry_epsilon"] == pytest.approx(0.135)
    assert diagnostics[-1]["management_epsilon"] == pytest.approx(0.135)
    assert diagnostics[-1]["teacher_weight_scale"] == 1.0
    assert diagnostics[-1]["teacher_guidance_dropout_probability"] == 0.0
    assert diagnostics[-1]["updates"] == 1
    assert diagnostics[-1]["mean_training_loss"] == 0.5
    assert diagnostics[-1]["mean_gradient_norm"] == 1.25
    assert diagnostics[-1]["mean_sampled_management_row_fraction"] == 0.75
    assert diagnostics[-1]["mean_sampled_hold_reward"] == 0.03
    assert diagnostics[-1]["mean_sampled_close_reward"] == -0.02
    assert diagnostics[-1]["mean_sampled_hold_n_step_return"] == 0.12
    assert diagnostics[-1]["mean_sampled_close_n_step_return"] == -0.08
    assert diagnostics[-1]["mean_sampled_hold_td_loss"] == 2.1
    assert diagnostics[-1]["mean_sampled_close_td_loss"] == 2.4
    assert diagnostics[-1]["mean_management_hold_minus_close_q"] == 0.15
    assert diagnostics[-1]["mean_sampled_management_close_fraction"] == 0.2
    assert diagnostics[-1]["cumulative_pass_rate"] == 1.0
    assert diagnostics[-1]["cumulative_blow_rate"] == 0.0
    assert diagnostics[-1]["cumulative_average_balance"] == 6_000.0
    assert diagnostics[-1]["action_counts"]["WAIT"] == 4
    assert diagnostics[-1]["shadow_h50_complete_trades"] == 0
    assert diagnostics[-1]["largest_realized_trade"]["realized_r"] == 3.0
    assert diagnostics[-1]["largest_mfe_trade"]["mfe_r"] == 3.5
    assert (
        "winR=+0.000R balance=+6000.00 avg_balance=+6000.00 steps=4/8"
        in capsys.readouterr().out
    )


def test_v2_association_telemetry_survives_episode_summary_and_evaluation(
    tmp_path: Path,
) -> None:
    class AssociationAgent(Agent):
        def train_batch(self, sequences, **kwargs) -> float:
            loss = super().train_batch(sequences, **kwargs)
            self.last_train_metrics.update({
                "regime_selectivity_association_loss": 0.4,
                "regime_selectivity_association_active": 1.0,
                "regime_selectivity_association_skipped": 0.0,
                "regime_selectivity_association_dead_wait_rows": 2.0,
                "regime_selectivity_association_dead_wait_"
                "model_wait_probability_sum": 1.6,
                "regime_selectivity_association_transition_positive_long_rows": 1.0,
                "regime_selectivity_association_transition_positive_long_"
                "model_wait_probability_sum": 0.2,
                "regime_selectivity_association_transition_positive_short_rows": 3.0,
                "regime_selectivity_association_transition_positive_short_"
                "model_wait_probability_sum": 0.9,
                "regime_selectivity_dead_wait_minus_"
                "transition_positive_model_wait": 0.55,
            })
            return loss

    diagnostics: list[dict[str, object]] = []
    train_agent(
        AssociationAgent(),
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            sequence_length=2,
            seed=61,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=61,
        episode_diagnostic_callback=diagnostics.append,
    )

    episode_association = diagnostics[-1]["persistent_regime_selectivity"][
        "association"
    ]
    assert diagnostics[-1][
        "mean_regime_selectivity_dead_wait_minus_transition_positive_model_wait"
    ] == pytest.approx(0.55)
    assert episode_association == pytest.approx({
        "loss_sum": 0.4,
        "loss_mean": 0.4,
        "update_count": 1.0,
        "active_updates": 1.0,
        "skipped_updates": 0.0,
        "dead_wait_rows": 2.0,
        "dead_wait_model_wait_probability_sum": 1.6,
        "dead_wait_model_wait_probability_mean": 0.8,
        "transition_positive_long_rows": 1.0,
        "transition_positive_long_model_wait_probability_sum": 0.2,
        "transition_positive_long_model_wait_probability_mean": 0.2,
        "transition_positive_short_rows": 3.0,
        "transition_positive_short_model_wait_probability_sum": 0.9,
        "transition_positive_short_model_wait_probability_mean": 0.3,
        "dead_wait_minus_transition_positive_model_wait": 0.55,
    })
    source = tmp_path / "training-diagnostics.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in diagnostics))
    destination = tmp_path / "training-diagnostic-summary.json"

    training_module._write_training_diagnostic_summary(source, destination)

    overall = json.loads(destination.read_text())["overall"]
    summary_association = overall["persistent_regime_selectivity"][
        "association"
    ]
    assert summary_association["loss_sum"] == pytest.approx(0.8)
    assert summary_association["active_updates"] == pytest.approx(2.0)
    assert summary_association["dead_wait_rows"] == pytest.approx(4.0)
    assert summary_association[
        "transition_positive_long_model_wait_probability_sum"
    ] == pytest.approx(0.4)
    assert summary_association[
        "transition_positive_short_model_wait_probability_sum"
    ] == pytest.approx(1.8)
    evaluation = training_module._persistent_regime_selectivity_evaluation_metrics(
        overall["persistent_regime_selectivity"]
    )
    assert evaluation["regime_selectivity_association_loss"] == pytest.approx(0.4)
    assert evaluation[
        "regime_selectivity_association_active_updates"
    ] == pytest.approx(2.0)
    assert evaluation[
        "regime_selectivity_association_dead_wait_rows"
    ] == pytest.approx(4.0)
    assert evaluation[
        "regime_selectivity_dead_wait_minus_transition_positive_model_wait"
    ] == pytest.approx(0.55)


def test_v4_side_conditioned_gradient_evidence_survives_training_evaluation(
    tmp_path: Path,
) -> None:
    class SideConditionedAgent(Agent):
        def train_batch(self, sequences, **kwargs) -> float:
            loss = super().train_batch(sequences, **kwargs)
            self.last_train_metrics.update({
                "regime_selectivity_side_conditioned_loss": 0.25,
                "regime_selectivity_side_conditioned_active_sides": 2.0,
                "regime_selectivity_failed_long_confluence_rows": 3.0,
                "regime_selectivity_failed_long_confluence_"
                "model_wait_probability_sum": 2.4,
                "regime_selectivity_failed_short_confluence_rows": 2.0,
                "regime_selectivity_failed_short_confluence_"
                "model_wait_probability_sum": 1.4,
            })
            return loss

    diagnostics: list[dict[str, object]] = []
    train_agent(
        SideConditionedAgent(),
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            sequence_length=2,
            seed=67,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=67,
        episode_diagnostic_callback=diagnostics.append,
    )

    episode = diagnostics[-1]["persistent_regime_selectivity"]
    assert episode["failed_long_confluence"] == pytest.approx({
        "rows": 3.0,
        "weight_sum": 0.0,
        "weight_mean": 0.0,
        "model_wait_probability_sum": 2.4,
        "model_wait_probability_mean": 0.8,
    })
    assert episode["failed_short_confluence"] == pytest.approx({
        "rows": 2.0,
        "weight_sum": 0.0,
        "weight_mean": 0.0,
        "model_wait_probability_sum": 1.4,
        "model_wait_probability_mean": 0.7,
    })
    assert episode["side_conditioned"] == pytest.approx({
        "loss_sum": 0.25,
        "loss_mean": 0.25,
        "update_count": 1.0,
        "active_sides_sum": 2.0,
        "both_sides_active_updates": 1.0,
    })

    source = tmp_path / "training-diagnostics.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in diagnostics))
    destination = tmp_path / "training-diagnostic-summary.json"
    training_module._write_training_diagnostic_summary(source, destination)
    overall = json.loads(destination.read_text())["overall"]
    evaluation = training_module._persistent_regime_selectivity_evaluation_metrics(
        overall["persistent_regime_selectivity"]
    )

    assert evaluation["regime_selectivity_failed_long_confluence_rows"] == 6.0
    assert evaluation["regime_selectivity_failed_short_confluence_rows"] == 4.0
    assert evaluation[
        "regime_selectivity_side_conditioned_both_sides_active_updates"
    ] == 2.0
    gates = training_module._training_evaluation_gates(
        regime_selectivity_active=True,
        regime_selectivity_semantics=(
            "side_conditioned_expansion_regime_confluence_v4"
        ),
    )
    both_sides_gate = next(
        gate
        for gate in gates
        if gate.metric
        == "regime_selectivity_side_conditioned_both_sides_active_updates"
    )
    assert both_sides_gate.passes(evaluation)
    assert not both_sides_gate.passes({
        **evaluation,
        "regime_selectivity_side_conditioned_both_sides_active_updates": 0.0,
    })


def test_episode_budget_prints_the_episode_progress_counter_once(capsys) -> None:
    class BoundedEnvironment(Environment):
        def reset(self):
            observation, info = super().reset()
            return observation, {**info, "start": 0, "end": 4, "ticker": "NQ"}

    result = train_agent(
        Agent(),
        BoundedEnvironment(),
        episodes=3,
        minimum_environment_steps=1,
        budget_mode="episodes",
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            sequence_length=2,
            seed=41,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=41,
    )

    assert result.episodes == 3
    assert result.environment_steps == 12
    train_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[train] ")
    ]
    assert len(train_lines) == 3
    assert all(" episodes=" not in line for line in train_lines)
    assert train_lines[-1].startswith("[train] episode=3/3 ")


def test_episode_budget_drives_learning_schedules_by_episode_position() -> None:
    class RecordingAgent(Agent):
        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            self.epsilons = getattr(self, "epsilons", [])
            self.epsilons.append(epsilon)
            return super().select_action(
                observation,
                hidden=hidden,
                valid_actions=valid_actions,
                epsilon=epsilon,
                return_action_values=return_action_values,
            )

    class TwoStepEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": "NQ",
                "start": 10,
                "end": 12,
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 2
            return np.array([self.index], np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.WAIT,),
                "fill_index": 10 + self.index,
                "outcome": "timeout" if terminated else None,
                "ticker": "NQ",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    agent = RecordingAgent()
    diagnostics = []
    train_agent(
        agent,
        TwoStepEnvironment(),
        episodes=2,
        minimum_environment_steps=1,
        budget_mode="episodes",
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            sequence_length=1,
            seed=43,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=1.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=43,
        teacher_loss_end_scale=0.0,
        teacher_guidance_dropout_start=0.0,
        teacher_guidance_dropout_end=1.0,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert agent.epsilons == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert agent.teacher_weight_scales == pytest.approx([0.5, 0.0])
    assert agent.entry_action_weight_scales == [1.0, 1.0]
    assert [row["teacher_schedule_progress"] for row in diagnostics] == [0.5, 1.0]
    assert [
        row["teacher_guidance_dropout_probability"] for row in diagnostics
    ] == pytest.approx([0.25, 0.75])


def test_recovery_training_starts_every_episode_at_frozen_deficit_and_builds_targets() -> None:
    class RecoveryAgent(Agent):
        recurrent_burn_in = 64
        n_step_return = 8

    class OrdinaryEnvironment:
        def __init__(self) -> None:
            self.recovery_flags: list[bool] = []
            self.starting_realized_pnls: list[float] = []
            self.ticker = "NQ"

        def reset(self, *, options=None):
            options = options or {}
            recovery = "challenge_start_state" in options
            self.recovery_flags.append(recovery)
            self.starting_realized_pnls.append(
                float(options["challenge_start_state"].realized_pnl)
                if recovery
                else 0.0
            )
            self.ticker = str(options.get("ticker", "NQ"))
            return np.zeros(1, np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": self.ticker,
                "start": 0,
                "mll_headroom_fraction": 1.0,
                "realized_pnl": (
                    float(options["challenge_start_state"].realized_pnl)
                    if recovery
                    else 0.0
                ),
            }

        def step(self, action):
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": self.ticker,
                "fill_index": 1,
                "outcome": "pass",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 6_000.0,
                "recovery_wait_decisions": 0,
            }

    class SidecarEnvironment:
        def __init__(self) -> None:
            self.resets = 0
            self.starting_realized_pnls: list[float] = []

        def reset(self, *, options=None):
            self.starting_realized_pnls.append(
                float(options["challenge_start_state"].realized_pnl)
            )
            self.resets += 1
            return np.zeros(1, np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "ticker": options["ticker"],
                "start": options["start"],
                "realized_pnl": float(
                    options["challenge_start_state"].realized_pnl
                ),
            }

        def step(self, action):
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "timeout",
                "realized_pnl": -2_600.0,
                "equity_pnl": -2_600.0,
                "recovery_status": "not_recovered",
            }

    class FrozenPolicy:
        def select_action(self, observation, *, hidden, valid_actions, epsilon,
                          return_action_values=False):
            return Action.WAIT, None, None

    settings = _recovery_curriculum_settings()
    environment = OrdinaryEnvironment()
    sidecar = SidecarEnvironment()
    store = RecoveryValueStore(capacity=10, seed=37)
    replay = BalancedSequenceReplay(
        capacity_episodes=10,
        sequence_length=96,
        recurrent_burn_in=64,
        n_step_return=8,
        seed=11,
    )
    diagnostics: list[dict[str, object]] = []

    result = train_agent(
        RecoveryAgent(),
        environment,
        episodes=4,
        minimum_environment_steps=4,
        replay=replay,
        warmup_episodes=99,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=96,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=("NQ",),
        ticker_seed=5,
        recovery_curriculum=settings,
        recovery_value_policy=FrozenPolicy(),
        recovery_value_environment=sidecar,
        recovery_value_store=store,
        recovery_value_source_identity_sha256="1" * 64,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert environment.recovery_flags == [True] * 4
    assert environment.starting_realized_pnls == [-2_000.0] * 4
    assert sidecar.resets == 12
    assert sidecar.starting_realized_pnls == [
        -1_500.0,
        -1_500.0,
        -1_500.0,
        -1_000.0,
        -1_000.0,
        -1_000.0,
        -500.0,
        -500.0,
        -500.0,
        -2_500.0,
        -2_500.0,
        -2_500.0,
    ]
    assert len(store) == 4
    assert result.passes == 4
    assert replay.transition_count == 4
    assert len(replay) == 4
    assert Counter(item["episode_kind"] for item in diagnostics) == {
        "recovery": 4,
    }


def test_recovery_training_marks_only_negative_pnl_decisions_as_recovery() -> None:
    class RecoveryAgent(Agent):
        recurrent_burn_in = 64
        n_step_return = 8

    class TwoStateEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, options=None):
            return np.zeros(1, np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": "NQ",
                "start": 0,
                "realized_pnl": -2_000.0,
                "mll_headroom_fraction": 1.0 / 3.0,
            }

        def step(self, action):
            self.steps += 1
            terminated = self.steps == 2
            realized_pnl = 0.0 if self.steps == 1 else 6_000.0
            return np.full(1, self.steps, np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.WAIT,),
                "ticker": "NQ",
                "fill_index": self.steps,
                "outcome": "pass" if terminated else None,
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "realized_pnl": realized_pnl,
                "equity_pnl": realized_pnl,
                "mll_headroom_fraction": 1.0,
                "recovery_wait_decisions": 1,
            }

    class SidecarEnvironment:
        def reset(self, *, options=None):
            pnl = float(options["challenge_start_state"].realized_pnl)
            return np.zeros(1, np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "ticker": "NQ",
                "start": 0,
                "realized_pnl": pnl,
            }

        def step(self, action):
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "timeout",
                "realized_pnl": -1_900.0,
                "equity_pnl": -1_900.0,
                "recovery_status": "not_recovered",
            }

    class FrozenPolicy:
        def select_action(self, observation, *, hidden, valid_actions, epsilon,
                          return_action_values=False):
            return Action.WAIT, None, None

    class CapturingReplay(BalancedSequenceReplay):
        recovery_flags: list[bool]

        def add(self, episode):
            self.recovery_flags = [
                transition.recovery_active
                for transition in episode.transitions
            ]
            super().add(episode)

    replay = CapturingReplay(
        capacity_episodes=2,
        sequence_length=96,
        recurrent_burn_in=64,
        n_step_return=8,
        seed=67,
    )
    train_agent(
        RecoveryAgent(),
        TwoStateEnvironment(),
        episodes=1,
        minimum_environment_steps=2,
        replay=replay,
        warmup_episodes=99,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=96,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=("NQ",),
        ticker_seed=5,
        recovery_curriculum=_recovery_curriculum_settings(),
        recovery_value_policy=FrozenPolicy(),
        recovery_value_environment=SidecarEnvironment(),
        recovery_value_store=RecoveryValueStore(capacity=10, seed=37),
        recovery_value_source_identity_sha256="2" * 64,
    )

    assert replay.recovery_flags == [True, False]


def test_recovery_and_healthy_pass_replays_are_additive_to_the_ordinary_batch(
) -> None:
    class RecoveryAgent(Agent):
        recurrent_burn_in = 1
        n_step_return = 1

        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []
            self.recovery_boundaries: list[bool] = []
            self.healthy_policy_rows: list[bool] = []

        def train_batch(self, sequences, **kwargs) -> float:
            self.batch_sizes.append(len(sequences))
            self.recovery_boundaries.append(
                len(sequences) >= 2
                and sequences[-2][1].recovery_active
                and not sequences[-2][2].recovery_active
            )
            self.healthy_policy_rows.append(
                len(sequences) == 3
                and not sequences[-1][1].recovery_active
                and sequences[-1][1].valid_actions
                == (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            )
            self.last_train_metrics = {}
            return 0.5

    class PassingRecoveryEnvironment:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, options=None):
            self.steps = 0
            return np.zeros(1, np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": "NQ",
                "start": 0,
                "realized_pnl": -2_000.0,
            }

        def step(self, action):
            self.steps += 1
            terminated = self.steps == 2
            realized_pnl = 0.0 if self.steps == 1 else 6_000.0
            return np.full(1, self.steps, np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.WAIT,),
                "ticker": "NQ",
                "fill_index": self.steps,
                "outcome": "pass" if terminated else None,
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "realized_pnl": realized_pnl,
                "equity_pnl": realized_pnl,
            }

    class SidecarEnvironment:
        def reset(self, *, options=None):
            return np.zeros(1, np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "ticker": "NQ",
                "start": 0,
                "realized_pnl": float(
                    options["challenge_start_state"].realized_pnl
                ),
            }

        def step(self, action):
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "timeout",
                "realized_pnl": -1_900.0,
                "equity_pnl": -1_900.0,
                "recovery_status": "not_recovered",
            }

    class FrozenPolicy:
        def select_action(self, observation, *, hidden, valid_actions, epsilon,
                          return_action_values=False):
            return Action.WAIT, None, None

    settings = RecoveryCurriculumSettings(**{
        **_recovery_curriculum_settings().__dict__,
        "recovery_success_replay_update_period": 2,
        "recovery_success_replay_path": "runs/recovery-passes.pt",
        "recovery_success_replay_sha256": "a" * 64,
        "healthy_pass_replay_update_period": 2,
        "healthy_pass_replay_path": "runs/v21-healthy-passes.pt",
        "healthy_pass_replay_sha256": "b" * 64,
    })
    pass_replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=4,
        recurrent_burn_in=1,
        n_step_return=1,
        seed=73,
    )
    pass_replay.add(Episode(
        episode_id="prior-v22-pass",
        ticker="NQ",
        outcome="pass",
        primary_side="long",
        ended_at_ns=1,
        transitions=tuple(
            Transition(
                observation=np.array([index], np.float32),
                action=Action.WAIT,
                reward=0.0,
                next_observation=np.array([index + 1], np.float32),
                terminated=index == 3,
                valid_actions=(Action.WAIT,),
                next_valid_actions=() if index == 3 else (Action.WAIT,),
                recovery_active=index < 2,
            )
            for index in range(4)
        ),
    ))
    healthy_replay = BalancedSequenceReplay(
        capacity_episodes=2,
        sequence_length=4,
        recurrent_burn_in=1,
        n_step_return=1,
        seed=75,
    )
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    healthy_replay.add(Episode(
        episode_id="v21-healthy-pass",
        ticker="NQ",
        outcome="pass",
        primary_side="short",
        ended_at_ns=2,
        transitions=tuple(
            Transition(
                observation=np.array([index], np.float32),
                action=Action.WAIT,
                reward=0.0,
                next_observation=np.array([index + 1], np.float32),
                terminated=index == 3,
                valid_actions=flat_actions,
                next_valid_actions=() if index == 3 else flat_actions,
                entry_action_target=Action.WAIT,
                recovery_active=False,
            )
            for index in range(4)
        ),
    ))
    agent = RecoveryAgent()
    train_agent(
        agent,
        PassingRecoveryEnvironment(),
        episodes=2,
        minimum_environment_steps=4,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            sequence_length=4,
            recurrent_burn_in=1,
            n_step_return=1,
            seed=71,
        ),
        warmup_episodes=1,
        updates_per_episode=2,
        batch_sequences=1,
        recurrent_horizon=4,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=("NQ",),
        ticker_seed=5,
        recovery_curriculum=settings,
        recovery_value_policy=FrozenPolicy(),
        recovery_value_environment=SidecarEnvironment(),
        recovery_value_store=RecoveryValueStore(capacity=10, seed=37),
        recovery_value_source_identity_sha256="3" * 64,
        recovery_success_replay=pass_replay,
        healthy_pass_replay=healthy_replay,
    )

    assert agent.batch_sizes == [1, 3, 1, 3]
    assert agent.recovery_boundaries == [False, True, False, True]
    assert agent.healthy_policy_rows == [False, True, False, True]


def test_recovery_pass_replay_artifact_is_authenticated_and_recurrent(
    tmp_path: Path,
) -> None:
    source = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=4,
        recurrent_burn_in=1,
        n_step_return=1,
        seed=79,
    )
    source.add(Episode(
        episode_id="saved-v22-pass",
        ticker="NQ",
        outcome="pass",
        primary_side="short",
        ended_at_ns=2,
        transitions=tuple(
            Transition(
                observation=np.array([index], np.float32),
                action=Action.WAIT,
                reward=0.0,
                next_observation=np.array([index + 1], np.float32),
                terminated=index == 3,
                valid_actions=(Action.WAIT,),
                next_valid_actions=() if index == 3 else (Action.WAIT,),
                recovery_active=index < 2,
            )
            for index in range(4)
        ),
    ))
    artifact = tmp_path / "v22-recovery-passes.pt"
    torch.save({
        "schema": "propevolve_recovery_success_replay_v1",
        "source_checkpoints": [{
            "causal_identity_sha256": "b" * 64,
            "resume_identity": "frozen-v22-run",
        }],
        "replay_state": source.state_dict(),
    }, artifact)
    expected_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    restored = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=4,
        recurrent_burn_in=1,
        n_step_return=1,
        seed=83,
    )

    _load_recovery_success_replay_artifact(
        artifact,
        expected_sha256=expected_sha256,
        replay=restored,
    )

    sequence = restored.sample_successful_recovery_sequences(1)[0]
    assert sequence[1].recovery_active is True
    assert sequence[2].recovery_active is False

    with pytest.raises(ValueError, match="identity drifted"):
        _load_recovery_success_replay_artifact(
            artifact,
            expected_sha256="0" * 64,
            replay=restored,
        )


def test_healthy_pass_replay_artifact_is_authenticated_and_anchored(
    tmp_path: Path,
) -> None:
    source = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=4,
        recurrent_burn_in=1,
        n_step_return=1,
        seed=89,
    )
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    source.add(Episode(
        episode_id="saved-v21-pass",
        ticker="NQ",
        outcome="pass",
        primary_side="long",
        ended_at_ns=3,
        transitions=tuple(
            Transition(
                observation=np.array([index], np.float32),
                action=Action.WAIT,
                reward=0.0,
                next_observation=np.array([index + 1], np.float32),
                terminated=index == 3,
                valid_actions=flat_actions,
                next_valid_actions=() if index == 3 else flat_actions,
                entry_action_target=Action.WAIT,
                recovery_active=False,
            )
            for index in range(4)
        ),
    ))
    artifact = tmp_path / "v21-healthy-passes.pt"
    torch.save({
        "schema": "propevolve_healthy_pass_replay_v1",
        "source_checkpoints": [{
            "causal_identity_sha256": "d" * 64,
            "resume_identity": "frozen-v21-run",
        }],
        "replay_state": source.state_dict(),
    }, artifact)
    expected_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    restored = BalancedSequenceReplay(
        capacity_episodes=4,
        sequence_length=4,
        recurrent_burn_in=1,
        n_step_return=1,
        seed=97,
    )

    _load_healthy_pass_replay_artifact(
        artifact,
        expected_sha256=expected_sha256,
        replay=restored,
    )

    anchor = restored.sample_healthy_pass_sequences(1)[0][1]
    assert anchor.recovery_active is False
    assert anchor.valid_actions == flat_actions

def test_teacher_curriculum_is_gradual_and_deterministic() -> None:
    agent = Agent()
    diagnostics = []
    observed_visibility = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed_visibility.extend(
                transition.teacher_imitation_visible
                for transition in episode.transitions
            )
            super().add(episode)

    train_agent(
        agent,
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=CapturingReplay(capacity_episodes=10, sequence_length=2, seed=1),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=19,
        teacher_lookup=lambda ticker, index: np.ones(4, dtype=np.float32),
        teacher_loss_end_scale=0.2,
        teacher_guidance_dropout_start=0.0,
        teacher_guidance_dropout_end=1.0,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert observed_visibility == [
        True, True, True, False, True, False, True, False
    ]
    assert diagnostics[0]["teacher_weight_scale"] == pytest.approx(0.6)
    assert diagnostics[1]["teacher_weight_scale"] == pytest.approx(0.2)
    assert diagnostics[0]["teacher_guidance_dropout_probability"] == 0.375
    assert diagnostics[1]["teacher_guidance_dropout_probability"] == 0.875
    assert agent.teacher_weight_scales == [pytest.approx(0.6), pytest.approx(0.2)]


def test_teacher_curriculum_has_a_declared_final_autonomy_tail() -> None:
    agent = Agent()
    diagnostics = []
    observed_visibility = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed_visibility.extend(
                transition.teacher_imitation_visible
                for transition in episode.transitions
            )
            super().add(episode)

    train_agent(
        agent,
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=CapturingReplay(capacity_episodes=10, sequence_length=2, seed=1),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=19,
        teacher_lookup=lambda ticker, index: np.ones(4, dtype=np.float32),
        teacher_loss_end_scale=0.0,
        teacher_guidance_dropout_start=0.0,
        teacher_guidance_dropout_end=1.0,
        teacher_autonomy_start_fraction=0.5,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert observed_visibility == [
        True, False, True, False, False, False, False, False
    ]
    assert diagnostics[1]["teacher_weight_scale"] == 0.0
    assert diagnostics[1]["teacher_guidance_dropout_probability"] == 1.0
    assert diagnostics[1]["teacher_schedule_progress"] == 1.0
    assert agent.teacher_weight_scales == [0.0, 0.0]


def test_teacher_autonomy_boundary_is_exact_inside_a_crossing_episode() -> None:
    class LongEpisodeEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": "NQ",
                "start": 0,
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 10
            return np.array([self.index], np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.WAIT,),
                "ticker": "NQ",
                "fill_index": self.index,
                "outcome": "timeout" if terminated else None,
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    observed_visibility = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed_visibility.extend(
                transition.teacher_imitation_visible
                for transition in episode.transitions
            )
            super().add(episode)

    train_agent(
        Agent(),
        LongEpisodeEnvironment(),
        episodes=1,
        minimum_environment_steps=10,
        replay=CapturingReplay(capacity_episodes=2, sequence_length=2, seed=3),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=4,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        teacher_lookup=lambda ticker, index: np.ones(4, dtype=np.float32),
        teacher_loss_end_scale=0.0,
        teacher_guidance_dropout_start=0.0,
        teacher_guidance_dropout_end=1.0,
        teacher_autonomy_start_fraction=0.8,
    )

    assert observed_visibility[8:] == [False, False]


def test_teacher_dropout_does_not_remove_exact_action_or_confluence_supervision() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    class LongFlatEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.array([0.0], np.float32), {
                "valid_actions": flat_actions,
                "ticker": "NQ",
                "start": 0,
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 10
            return np.array([self.index], np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else flat_actions,
                "ticker": "NQ",
                "fill_index": self.index,
                "outcome": "timeout" if terminated else None,
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    observed: list[tuple[np.ndarray | None, Action | None, bool]] = []
    diagnostics = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed.extend(
                (
                    transition.teacher_target,
                    transition.entry_action_target,
                    transition.teacher_imitation_visible,
                )
                for transition in episode.transitions
            )
            super().add(episode)

    agent = Agent()
    train_agent(
        agent,
        LongFlatEnvironment(),
        episodes=1,
        minimum_environment_steps=10,
        replay=CapturingReplay(capacity_episodes=2, sequence_length=2, seed=3),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=4,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        teacher_lookup=lambda ticker, index: np.ones(4, dtype=np.float32),
        teacher_channels=("a", "b", "c", "d"),
        entry_action_lookup=lambda ticker, index: Action.ENTER_LONG_1,
        teacher_loss_end_scale=0.0,
        teacher_guidance_dropout_start=0.0,
        teacher_guidance_dropout_end=1.0,
        teacher_autonomy_start_fraction=0.8,
        entry_supervision_autonomy_start_fraction=0.95,
        episode_diagnostic_callback=diagnostics.append,
    )

    # Dropout applies only to optional teacher imitation. Exact action and
    # confluence labels remain available to the training loss on every row.
    assert any(
        semantic is not None and action == Action.ENTER_LONG_1 and visible
        for semantic, action, visible in observed[:8]
    )
    assert all(
        semantic is not None and action == Action.ENTER_LONG_1
        for semantic, action, _ in observed
    )
    assert observed[-2][2:] == (False,)
    assert observed[-1][2:] == (False,)
    assert agent.teacher_weight_scales[0] == 0.0
    assert agent.entry_action_weight_scales[0] == 1.0
    assert diagnostics[0]["teacher_weight_scale"] == 0.0
    assert diagnostics[0]["entry_action_weight_scale"] == 1.0
    assert diagnostics[0]["teacher_schedule_progress"] == 1.0
    assert diagnostics[0]["entry_action_schedule_progress"] == 1.0
    assert diagnostics[0]["entry_action_target_counts"]["ENTER_LONG_1"] > 0
    assert diagnostics[0]["entry_action_target_counts"]["WAIT"] == 0


def test_exact_entry_supervision_remains_active_after_imitation_autonomy() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    class OneStepFlatEnvironment:
        def reset(self):
            return np.array([0.0], np.float32), {
                "valid_actions": flat_actions,
                "ticker": "NQ",
                "start": 0,
            }

        def step(self, action):
            assert action == Action.WAIT
            return np.array([1.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": "NQ",
                "fill_index": 1,
                "outcome": "timeout",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    class ReplayInspectingAgent(Agent):
        def train_batch(
            self,
            sequences,
            *,
            teacher_weight_scale=1.0,
            entry_action_weight_scale=1.0,
        ):
            self.replayed_entry_rows = getattr(self, "replayed_entry_rows", [])
            self.replayed_entry_rows.append(sum(
                transition.entry_action_target is not None
                for sequence in sequences
                for transition in sequence
            ))
            return super().train_batch(
                sequences,
                teacher_weight_scale=teacher_weight_scale,
                entry_action_weight_scale=entry_action_weight_scale,
            )

    agent = ReplayInspectingAgent()
    diagnostics: list[dict[str, object]] = []
    train_agent(
        agent,
        OneStepFlatEnvironment(),
        episodes=10,
        minimum_environment_steps=10,
        replay=BalancedSequenceReplay(
            capacity_episodes=20,
            sequence_length=1,
            seed=11,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        teacher_lookup=lambda ticker, index: np.ones(4, dtype=np.float32),
        entry_action_lookup=lambda ticker, index: Action.ENTER_SHORT_1,
        teacher_loss_end_scale=0.0,
        teacher_guidance_dropout_start=0.0,
        teacher_guidance_dropout_end=0.0,
        teacher_autonomy_start_fraction=0.8,
        entry_supervision_autonomy_start_fraction=0.95,
        episode_diagnostic_callback=diagnostics.append,
    )

    # The generic imitation teacher reaches autonomy while exact Entry labels
    # remain active through the entire training run.
    assert diagnostics[8]["teacher_weight_scale"] == 0.0
    assert diagnostics[8]["entry_action_weight_scale"] == 1.0
    assert agent.teacher_weight_scales[8] == 0.0
    assert agent.entry_action_weight_scales[8] == 1.0
    assert agent.replayed_entry_rows[8] > 0
    assert diagnostics[9]["teacher_weight_scale"] == 0.0
    assert diagnostics[9]["entry_action_weight_scale"] == 1.0
    assert agent.teacher_weight_scales[9] == 0.0
    assert agent.entry_action_weight_scales[9] == 1.0
    summary = training_module._diagnostic_aggregate(diagnostics)
    assert summary["latest_teacher_weight_scale"] == 0.0
    assert summary["latest_entry_action_weight_scale"] == 1.0
    assert summary["latest_teacher_schedule_progress"] == 1.0
    assert summary["latest_entry_action_schedule_progress"] == 1.0
    consolidation_summary = training_module._diagnostic_aggregate(
        diagnostics[8:]
    )
    assert consolidation_summary["mean_teacher_weight_scale"] == 0.0
    assert consolidation_summary["mean_entry_action_weight_scale"] == 1.0


@pytest.mark.parametrize("invalid", (True, "0.95", 0.79, 1.01))
def test_entry_supervision_runtime_schedule_fails_closed(invalid: object) -> None:
    with pytest.raises(
        ValueError,
        match="entry supervision autonomy start fraction",
    ):
        train_agent(
            Agent(),
            Environment(),
            episodes=1,
            minimum_environment_steps=1,
            replay=BalancedSequenceReplay(
                capacity_episodes=2,
                sequence_length=1,
                seed=1,
            ),
            warmup_episodes=99,
            updates_per_episode=1,
            batch_sequences=1,
            recurrent_horizon=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            episode_tickers=None,
            ticker_seed=1,
            teacher_autonomy_start_fraction=0.8,
            entry_supervision_autonomy_start_fraction=invalid,  # type: ignore[arg-type]
        )


def test_entry_action_lookup_is_not_used_after_exploration_enters() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    class EnterThenManageEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.array([0.0], np.float32), {
                "valid_actions": flat_actions,
                "ticker": "NQ",
                "start": 0,
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 3
            return np.array([self.index], np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.HOLD, Action.CLOSE),
                "ticker": "NQ",
                "fill_index": self.index,
                "outcome": "timeout" if terminated else None,
                "primary_side": "long",
                "trade_count": 1,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    class EnteringAgent(Agent):
        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            action = (
                Action.ENTER_LONG_1
                if Action.ENTER_LONG_1 in valid_actions
                else Action.HOLD
            )
            values = (
                np.zeros(len(Action), np.float32)
                if return_action_values
                else None
            )
            return action, None, values

    looked_up = []

    def lookup(ticker, row):
        looked_up.append(row)
        return Action.ENTER_LONG_1

    train_agent(
        EnteringAgent(),
        EnterThenManageEnvironment(),
        episodes=1,
        minimum_environment_steps=3,
        replay=BalancedSequenceReplay(capacity_episodes=2, sequence_length=1, seed=3),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=3,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        entry_action_lookup=lookup,
    )

    assert looked_up == [0]


def test_regime_selectivity_replay_uses_decision_time_not_post_action_headroom() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    class OneDecisionEnvironment:
        def reset(self):
            return np.array([0.0], np.float32), {
                "valid_actions": flat_actions,
                "ticker": "NQ",
                "start": 0,
                "mll_headroom_fraction": 0.10,
            }

        def step(self, action):
            return np.array([1.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": "NQ",
                "fill_index": 1,
                "outcome": "timeout",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
                # This is the state after the selected action and must not label
                # the preceding decision.
                "mll_headroom_fraction": 0.90,
            }

    class SelectivityAgent(Agent):
        regime_selectivity_loss_weight = 1.0

    captured = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            captured.extend(episode.transitions)
            super().add(episode)

    train_agent(
        SelectivityAgent(),
        OneDecisionEnvironment(),
        episodes=1,
        minimum_environment_steps=1,
        replay=CapturingReplay(
            capacity_episodes=2,
            sequence_length=1,
            seed=3,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        teacher_lookup=lambda ticker, index: np.full(22, 0.1, np.float32),
        teacher_channels=tuple(f"teacher_{index}" for index in range(22)),
        entry_action_lookup=lambda ticker, index: Action.WAIT,
    )

    assert len(captured) == 1
    assert captured[0].regime_selectivity_headroom_fraction == pytest.approx(0.10)


def test_training_marks_hard_exact_wait_for_post_burn_in_replay() -> None:
    flat_actions = (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    class OneDecisionEnvironment:
        def reset(self):
            return np.array([0.0], np.float32), {
                "valid_actions": flat_actions,
                "ticker": "NQ",
                "start": 0,
                "mll_headroom_fraction": 1.0,
            }

        def step(self, action):
            return np.array([1.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": "NQ",
                "fill_index": 1,
                "outcome": "timeout",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
                "mll_headroom_fraction": 1.0,
            }

    class SelectivityAgent(Agent):
        regime_selectivity_loss_weight = 1.0
        regime_selectivity = BalanceAwareRegimeSelectivity(
            channel_names=(
                "long_attempt_probability",
                "long_clean_retained_given_attempt_probability",
                "short_attempt_probability",
                "short_clean_retained_given_attempt_probability",
                *REGIME_TEACHER_CHANNELS,
            ),
            expansion_centers=(0.10, 0.10),
            semantics=SIDE_CONDITIONED_EXPANSION_REGIME_CONFLUENCE_SEMANTICS,
            persistent_chop_negative_emphasis=2.0,
        )

    captured = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            captured.extend(episode.transitions)
            super().add(episode)

    teacher = np.array(
        [0.2, 0.2, 0.2, 0.2, 0.95, 0.03, 0.02], np.float32
    )
    train_agent(
        SelectivityAgent(),
        OneDecisionEnvironment(),
        episodes=1,
        minimum_environment_steps=1,
        replay=CapturingReplay(
            capacity_episodes=2,
            sequence_length=1,
            terminal_sequence_fraction=1.0,
            regime_wait_sequence_update_period=1,
            seed=3,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        teacher_lookup=lambda ticker, index: teacher,
        teacher_channels=tuple(SelectivityAgent.regime_selectivity.channel_names),
        entry_action_lookup=lambda ticker, index: Action.WAIT,
        teacher_loss_end_scale=0.0,
        teacher_guidance_dropout_start=1.0,
        teacher_guidance_dropout_end=1.0,
        teacher_autonomy_start_fraction=0.8,
    )

    assert len(captured) == 1
    assert captured[0].entry_action_target is Action.WAIT
    assert captured[0].teacher_imitation_visible is False
    assert captured[0].regime_wait_priority > 0.9


def test_teacher_diagnostics_preserve_named_source_channels() -> None:
    class EnteringAgent(Agent):
        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            return Action.ENTER_LONG_1, None, np.zeros(len(Action), np.float32)

    diagnostics = []
    train_agent(
        EnteringAgent(),
        Environment(),
        episodes=1,
        minimum_environment_steps=4,
        replay=BalancedSequenceReplay(capacity_episodes=2, sequence_length=2, seed=3),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=3,
        teacher_lookup=lambda ticker, index: np.asarray(
            [0.2, 0.7, 0.1, 0.6, 0.8, 0.05], dtype=np.float32
        ),
        teacher_channels=(
            "long_attempt_probability",
            "long_clean_retained_given_attempt_probability",
            "short_attempt_probability",
            "short_clean_retained_given_attempt_probability",
            "expansion_trend_probability",
            "chop_no_trend_probability",
        ),
        episode_diagnostic_callback=diagnostics.append,
    )

    assert diagnostics[0]["selected_teacher_channel_means"] == pytest.approx({
        "long_attempt_probability": 0.2,
        "long_clean_retained_given_attempt_probability": 0.7,
        "short_attempt_probability": 0.1,
        "short_clean_retained_given_attempt_probability": 0.6,
        "expansion_trend_probability": 0.8,
        "chop_no_trend_probability": 0.05,
    })


def test_training_joins_only_visible_regime_entry_context_to_trade_economics() -> None:
    class TradingAgent(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.actions = iter((
                Action.ENTER_LONG_1,
                Action.CLOSE,
                Action.ENTER_SHORT_1,
                Action.CLOSE,
            ))

        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            values = (
                np.zeros(len(Action), np.float32)
                if return_action_values else None
            )
            return next(self.actions), None, values

    class TradeReceiptEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.zeros(1, np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "ticker": "NQ",
                "start": 10,
                "mll_headroom_fraction": 0.20,
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 4
            valid = (
                (Action.HOLD, Action.CLOSE)
                if self.index in {1, 3}
                else (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            )
            return np.ones(1, np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else valid,
                "ticker": "NQ",
                "fill_index": 10 + self.index,
                "outcome": "timeout" if terminated else None,
                "primary_side": "short",
                "trade_count": 2,
                "win_count": 1,
                "winning_r_sum": 2.0,
                "equity_pnl": 300.0,
                "mll_headroom_fraction": 0.80,
            }

        def closed_trade_receipts(self):
            return (
                {
                    "trade_index": 0,
                    "ticker": "NQ",
                    "side": "long",
                    "source_decision_index": 10,
                    "entry_mll_headroom": 600.0,
                    "pnl": -300.0,
                    "realized_r": -1.0,
                    "mfe_r": 0.25,
                    "mae_r": 1.0,
                    "exit_reason": "initial_stop",
                },
                {
                    "trade_index": 1,
                    "ticker": "NQ",
                    "side": "short",
                    "source_decision_index": 12,
                    "entry_mll_headroom": 2_400.0,
                    "pnl": 600.0,
                    "realized_r": 2.0,
                    "mfe_r": 2.5,
                    "mae_r": 0.25,
                    "exit_reason": "ratchet_stop",
                },
            )

    channels = (
        "long_attempt_probability",
        "long_clean_retained_given_attempt_probability",
        "short_attempt_probability",
        "short_clean_retained_given_attempt_probability",
        "chop_no_trend_probability",
        "chop_end_transition_probability",
        "expansion_trend_probability",
    )
    visible = {
        10: np.asarray((0.9, 0.8, 0.1, 0.1, 0.9, 0.05, 0.05), np.float32),
        # Deliberately no row 12: the second trade must be explicitly unattributed.
    }
    lookups: list[int] = []

    def lookup(ticker: str, index: int):
        lookups.append(index)
        return visible.get(index)

    diagnostics: list[dict[str, object]] = []
    train_agent(
        TradingAgent(),
        TradeReceiptEnvironment(),
        episodes=1,
        minimum_environment_steps=4,
        replay=BalancedSequenceReplay(
            capacity_episodes=2, sequence_length=1, seed=3
        ),
        warmup_episodes=99,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=8,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=23,
        teacher_lookup=lookup,
        teacher_channels=channels,
        episode_diagnostic_callback=diagnostics.append,
    )

    # Observability joins the targets already returned by the normal lookup;
    # it never performs a second or post-episode teacher query.
    assert lookups == [10, 11, 12, 13]
    economics = diagnostics[0]["regime_trade_economics"]
    assert economics["total_trades"] == 2
    assert economics["attributed_trades"] == 1
    assert economics["unattributed_trades"] == 1
    assert economics["attribution_coverage"] == 0.5
    assert economics["unattributed_by_side"] == {"long": 0, "short": 1}
    assert economics["groups"] == [{
        "side": "long",
        "static_regime": "dominant_chop",
        "headroom_stratum": "low_headroom_le_0_25",
        "episode_outcome": "timeout",
        "trades": 1,
        "wins": 0,
        "win_rate": 0.0,
        "realized_r_sum": -1.0,
        "realized_r_mean": -1.0,
        "mfe_r_sum": 0.25,
        "mfe_r_mean": 0.25,
        "mae_r_sum": 1.0,
        "mae_r_mean": 1.0,
        "initial_stop_count": 1,
        "regime_channel_probability_sums": {
            "chop_no_trend_probability": pytest.approx(0.9),
            "chop_end_transition_probability": pytest.approx(0.05),
            "expansion_trend_probability": pytest.approx(0.05),
        },
        "regime_channel_probability_means": {
            "chop_no_trend_probability": pytest.approx(0.9),
            "chop_end_transition_probability": pytest.approx(0.05),
            "expansion_trend_probability": pytest.approx(0.05),
        },
    }]
    aggregate = training_module._diagnostic_aggregate(diagnostics)[
        "regime_trade_economics"
    ]
    assert aggregate == economics


def test_training_short_circuits_without_passes_at_declared_step_boundary() -> None:
    checkpoints = []

    class TimeoutEnvironment(Environment):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if terminated:
                info["outcome"] = "timeout"
                info["equity_pnl"] = -500.0
            return observation, reward, terminated, truncated, info

    result = train_agent(
        Agent(),
        TimeoutEnvironment(),
        episodes=3,
        minimum_environment_steps=12,
        replay=BalancedSequenceReplay(capacity_episodes=4, sequence_length=2, seed=5),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        checkpoint_every_episodes=3,
        checkpoint_callback=checkpoints.append,
        short_circuit_minimum_environment_steps=4,
        short_circuit_minimum_passes=1,
        short_circuit_maximum_blow_rate=0.1,
    )

    assert result.environment_steps == 4
    assert result.short_circuited is True
    assert result.short_circuit_reason == "passes 0 < 1"
    assert checkpoints[-1].short_circuit_reason == "passes 0 < 1"


def test_training_short_circuits_only_when_blow_rate_exceeds_ceiling() -> None:
    class PassThenBlowEnvironment(Environment):
        def __init__(self) -> None:
            super().__init__()
            self.episode = 0

        def reset(self):
            self.episode += 1
            return super().reset()

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if terminated and self.episode == 2:
                info["outcome"] = "blow"
                info["equity_pnl"] = -3_000.0
            return observation, reward, terminated, truncated, info

    result = train_agent(
        Agent(),
        PassThenBlowEnvironment(),
        episodes=3,
        minimum_environment_steps=12,
        replay=BalancedSequenceReplay(capacity_episodes=4, sequence_length=2, seed=7),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=7,
        short_circuit_minimum_environment_steps=8,
        short_circuit_minimum_passes=1,
        short_circuit_maximum_blow_rate=0.1,
    )

    assert result.passes == 1
    assert result.blows == 1
    assert result.short_circuited is True
    assert result.short_circuit_reason == "blow rate 0.500000 > 0.100000"


def test_episode_budget_activates_safety_short_circuit_at_completed_episode_boundary(
) -> None:
    class BlowEnvironment:
        def reset(self):
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": "NQ",
                "start": 0,
                "end": 1,
            }

        def step(self, action):
            return np.array([1.0], np.float32), -1.0, True, False, {
                "valid_actions": (),
                "fill_index": 1,
                "outcome": "blow",
                "ticker": "NQ",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": -3_000.0,
            }

    result = train_agent(
        Agent(),
        BlowEnvironment(),
        episodes=5,
        minimum_environment_steps=1,
        budget_mode="episodes",
        replay=BalancedSequenceReplay(
            capacity_episodes=8,
            sequence_length=1,
            seed=47,
        ),
        warmup_episodes=99,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=47,
        short_circuit_minimum_episodes=3,
        short_circuit_minimum_passes=1,
        short_circuit_maximum_blow_rate=0.1,
    )

    assert result.episodes == 3
    assert result.blows == 3
    assert result.short_circuit_reason == (
        "passes 0 < 1; blow rate 1.000000 > 0.100000"
    )


def test_training_health_callback_short_circuits_and_checkpoints_episode_boundary(
) -> None:
    class BoundedEnvironment(Environment):
        def reset(self):
            observation, info = super().reset()
            return observation, {**info, "ticker": "NQ", "start": 0, "end": 4}

    observed = []
    checkpoints = []

    def health(progress, diagnostic):
        observed.append((progress.completed_episodes, diagnostic["outcome"]))
        return "policy health: entry collapse" if progress.completed_episodes == 2 else None

    result = train_agent(
        Agent(),
        BoundedEnvironment(),
        episodes=5,
        minimum_environment_steps=1,
        budget_mode="episodes",
        replay=BalancedSequenceReplay(
            capacity_episodes=8,
            sequence_length=2,
            seed=53,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=53,
        checkpoint_every_episodes=0,
        checkpoint_callback=checkpoints.append,
        training_health_callback=health,
    )

    assert observed == [(1, "pass"), (2, "pass")]
    assert result.episodes == 2
    assert result.short_circuit_reason == "policy health: entry collapse"
    assert checkpoints[-1].completed_episodes == 2
    assert checkpoints[-1].short_circuit_reason == "policy health: entry collapse"


def test_training_waits_for_the_evidence_boundary_before_collapse_detection() -> None:
    class CollapseEnvironment:
        def __init__(self) -> None:
            self.episode = -1

        def reset(self):
            self.episode += 1
            return np.array([0.0], np.float32), {
                "ticker": "NQ",
                "valid_actions": (Action.WAIT,),
            }

        def step(self, action):
            passed = self.episode == 0
            return np.array([1.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": "NQ",
                "primary_side": "long",
                "outcome": "pass" if passed else "timeout",
                "trade_count": 10,
                "win_count": 4,
                "winning_r_sum": 2.0,
                "equity_pnl": 6_000.0 if passed else -1_000.0,
                "avg_hold_bars": 1.5,
                "voluntary_close_count": 9,
            }

    result = train_agent(
        Agent(),
        CollapseEnvironment(),
        episodes=10,
        minimum_environment_steps=10,
        replay=BalancedSequenceReplay(
            capacity_episodes=10,
            sequence_length=1,
            seed=1,
        ),
        warmup_episodes=10,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=1,
        epsilon_start=0.1,
        epsilon_end=0.01,
        episode_tickers=None,
        ticker_seed=1,
        short_circuit_minimum_environment_steps=10,
        short_circuit_minimum_passes=1,
        short_circuit_maximum_blow_rate=0.1,
        collapse_window_episodes=2,
        collapse_minimum_prior_passes=1,
        collapse_maximum_recent_passes=0,
        collapse_maximum_average_hold_bars=4.0,
        collapse_minimum_voluntary_close_rate=0.8,
    )

    assert result.episodes == 10
    assert result.short_circuited is True
    assert result.short_circuit_reason == (
        "policy collapse: prior passes 1; recent passes 0/2; "
        "recent average hold 1.500000 <= 4.000000; "
        "recent voluntary-close rate 0.900000 >= 0.800000"
    )


def test_episode_budget_activates_rapid_close_collapse_at_episode_boundary() -> None:
    class CollapseEnvironment:
        def __init__(self) -> None:
            self.episode = 0

        def reset(self):
            self.episode += 1
            return np.array([0.0], np.float32), {
                "ticker": "NQ",
                "start": 0,
                "end": 1,
                "valid_actions": (Action.WAIT,),
            }

        def step(self, action):
            passed = self.episode == 1
            return np.array([1.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "fill_index": 1,
                "ticker": "NQ",
                "primary_side": "long",
                "outcome": "pass" if passed else "timeout",
                "trade_count": 10,
                "win_count": 4,
                "winning_r_sum": 2.0,
                "equity_pnl": 6_000.0 if passed else -1_000.0,
                "avg_hold_bars": 1.5,
                "voluntary_close_count": 9,
            }

    result = train_agent(
        Agent(),
        CollapseEnvironment(),
        episodes=8,
        minimum_environment_steps=1,
        budget_mode="episodes",
        replay=BalancedSequenceReplay(
            capacity_episodes=8,
            sequence_length=1,
            seed=61,
        ),
        warmup_episodes=99,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        episode_tickers=None,
        ticker_seed=61,
        short_circuit_minimum_episodes=4,
        collapse_window_episodes=2,
        collapse_minimum_prior_passes=1,
        collapse_maximum_recent_passes=0,
        collapse_maximum_average_hold_bars=4.0,
        collapse_minimum_voluntary_close_rate=0.8,
    )

    assert result.episodes == 4
    assert result.short_circuit_reason == (
        "policy collapse: prior passes 1; recent passes 0/2; "
        "recent average hold 1.500000 <= 4.000000; "
        "recent voluntary-close rate 0.900000 >= 0.800000"
    )


def test_training_preserves_a_pass_policy_before_any_following_updates() -> None:
    agent = Agent()
    retained_at_updates = []

    train_agent(
        agent,
        Environment(),
        episodes=1,
        minimum_environment_steps=4,
        replay=BalancedSequenceReplay(
            capacity_episodes=2,
            sequence_length=2,
            seed=2,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.1,
        epsilon_end=0.01,
        episode_tickers=None,
        ticker_seed=2,
        retention_checkpoint_callback=lambda evidence: retained_at_updates.append(
            (agent.updates, evidence)
        ),
    )

    assert retained_at_updates == [(0, {
        "episode": 1,
        "ticker": "NQ",
        "outcome": "pass",
        "terminal_pnl": 6_000.0,
    })]
    assert agent.retention_calls == 1
    assert agent.updates == 1


def test_retained_pass_checkpoints_are_immutable_per_episode(tmp_path: Path) -> None:
    class SavingAgent:
        def save(self, path, *, manifest):
            Path(path).write_text(json.dumps(manifest, sort_keys=True))

    alias = tmp_path / "retained-pass-policy.pt"
    for episode, ticker in ((3, "SI"), (9, "ZB")):
        training_module._save_retained_policy(
            SavingAgent(),
            alias,
            resume_identity="recipe-1",
            evidence={
                "episode": episode,
                "ticker": ticker,
                "outcome": "pass",
                "terminal_pnl": 6_000.0,
            },
        )

    retained = sorted((tmp_path / "retained-pass-policies").glob("*.pt"))
    assert [path.name for path in retained] == [
        "episode-000003-SI.pt",
        "episode-000009-ZB.pt",
    ]
    assert json.loads(alias.read_text())["retention_evidence"]["episode"] == 9


def test_recovery_reconciles_retained_passes_to_durable_episode(
    tmp_path: Path,
) -> None:
    class SavingAgent:
        def save(self, path, *, manifest):
            Path(path).write_text(json.dumps(manifest, sort_keys=True))

    alias = tmp_path / "retained-pass-policy.pt"
    agent = SavingAgent()
    for episode, ticker in ((3, "SI"), (7, "CL")):
        training_module._save_retained_policy(
            agent,
            alias,
            resume_identity="recipe-1",
            evidence={
                "episode": episode,
                "ticker": ticker,
                "outcome": "pass",
                "terminal_pnl": 6_000.0,
            },
        )

    training_module._reconcile_retained_pass_policies(
        alias,
        resume_identity="recipe-1",
        completed_episodes=5,
        manifest_loader=lambda path: json.loads(path.read_text()),
    )

    retained = sorted((tmp_path / "retained-pass-policies").glob("*.pt"))
    assert [path.name for path in retained] == ["episode-000003-SI.pt"]
    assert json.loads(alias.read_text())["retention_evidence"]["episode"] == 3
    partial = sorted(
        (tmp_path / "retained-pass-policies" / "partial").glob("*.pt")
    )
    assert len(partial) == 2
    assert any("episode-000007-CL" in path.name for path in partial)
    assert any("latest-alias" in path.name for path in partial)

    # Episode seven is replayed after recovery and can create fresh evidence.
    training_module._save_retained_policy(
        agent,
        alias,
        resume_identity="recipe-1",
        evidence={
            "episode": 7,
            "ticker": "CL",
            "outcome": "pass",
            "terminal_pnl": 6_000.0,
        },
    )
    assert json.loads(alias.read_text())["retention_evidence"]["episode"] == 7


def test_historical_candidate_recovery_reconciles_the_same_replayed_pass_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second crash in one checkpoint interval remains exactly resumable."""

    class DurableRecoveryReached(RuntimeError):
        pass

    class LoadedAgent:
        entry_action_class_weights = (1.0, 1.0, 1.0)

    class LoadingAgent:
        @classmethod
        def load(cls, path, *, device):
            del device
            path = Path(path)
            if path.name == "training-recovery.pt":
                return LoadedAgent(), {
                    "resume_identity": "recipe-1",
                    "progress": {"completed_episodes": 5},
                    "environment_rng_state": {},
                    "replay_state": {},
                    "replay_restored": True,
                    "policy_health_probe_corpus": None,
                }
            return LoadedAgent(), json.loads(path.read_text())

    class RecoveringEnvironment:
        observation_dim = 1

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def restore_rng_state(self, state):
            del state
            raise DurableRecoveryReached

    output = tmp_path / "run"
    output.mkdir()
    (output / "training-recovery.pt").write_bytes(b"durable recovery")
    (output / "training-diagnostics.jsonl").write_text("".join(
        json.dumps({"episode": episode}) + "\n" for episode in range(1, 6)
    ))
    archive = output / "retained-pass-policies"
    partial = archive / "partial"
    partial.mkdir(parents=True)
    replayed = archive / "episode-000007-CL.pt"
    alias = output / "retained-pass-policy.pt"
    manifest = {
        "resume_identity": "recipe-1",
        "retention_evidence": {
            "episode": 7,
            "ticker": "CL",
            "outcome": "pass",
            "terminal_pnl": 6_000.0,
        },
    }
    replayed.write_text(json.dumps(manifest, sort_keys=True))
    alias.write_bytes(replayed.read_bytes())
    replayed_sha256 = hashlib.sha256(replayed.read_bytes()).hexdigest()
    for label in ("episode-000007-CL", "latest-alias"):
        (partial / (
            f"{label}.after-episode-000005.{replayed_sha256}.pt"
        )).write_bytes(replayed.read_bytes())

    import propevolve.agent as agent_module

    monkeypatch.setattr(agent_module, "RecurrentC51Agent", LoadingAgent)
    monkeypatch.setattr(
        training_module.AssetContract,
        "load",
        classmethod(lambda cls, path: object()),
    )
    monkeypatch.setattr(
        training_module,
        "load_markets",
        lambda **kwargs: {"NQ": object()},
    )
    monkeypatch.setattr(training_module, "assert_temporal_role", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        training_module,
        "HistoricalChallengeEnv",
        RecoveringEnvironment,
    )
    monkeypatch.setattr(
        training_module,
        "_training_resume_identity",
        lambda *args, **kwargs: "recipe-1",
    )
    config = {
        "_root": str(tmp_path),
        "assets": "assets.json",
        "cache_root": "cache",
        "output": str(output),
        "tickers": ("NQ",),
        "deployment_tickers": ("NQ",),
        "training_only_tickers": (),
        "timeframe_minutes": 3,
        "temporal": {
            "train_start": "2021-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01",
        },
        "challenge": {
            "profit_target": 6_000.0,
            "max_loss": 3_000.0,
            "episode_days": 1,
            "bars_per_day": 2,
            "max_position_size": 1,
            "minimum_mll_headroom": 500.0,
            "trailing_mll_lock": True,
            "terminal_pass_reward": 250.0,
            "terminal_blow_reward": -1_500.0,
            "terminal_timeout_reward": -2.0,
            "terminal_pass_speed_reward_per_day": 20.0,
            "reward_scale": 1_000.0,
        },
        "point_values": {"NQ": 20.0},
        "round_trip_fees": {"NQ": 3.84},
        "agent": {"device": "cpu"},
        "training": {"seed": 7},
    }

    with pytest.raises(DurableRecoveryReached):
        HistoricalCandidateRunner().run(
            config,
            parent_candidate_ids=(),
            hypothesis="twice-crashed retained pass recovery",
        )

    assert not replayed.exists()
    assert not alias.exists()


class _ResumeEvidenceBoundaryReached(RuntimeError):
    pass


def _run_to_resume_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    diagnostic_episodes: tuple[int, ...] | None,
    policy_health_episodes: tuple[int, ...] | None,
    policy_health_enabled: bool,
    diagnostic_partial_tail: str | None = None,
) -> Path:
    class LoadedAgent:
        pass

    class LoadingAgent:
        @classmethod
        def load(cls, path, *, device):
            del path, device
            return LoadedAgent(), {
                "resume_identity": "recipe-1",
                "progress": {"completed_episodes": 5},
                "environment_rng_state": {},
                "replay_state": {},
                "replay_restored": True,
                "policy_health_probe_corpus": None,
            }

    class RecoveringEnvironment:
        observation_dim = 1

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def restore_rng_state(self, state):
            del state
            raise _ResumeEvidenceBoundaryReached

    class Challenge:
        max_loss = 3_000.0

    output = tmp_path / "run"
    output.mkdir()
    (output / "training-recovery.pt").write_bytes(b"durable recovery")
    streams = (
        ("training-diagnostics.jsonl", "episode", diagnostic_episodes),
        (
            "training-policy-health.jsonl",
            "completed_episodes",
            policy_health_episodes,
        ),
    )
    for name, field, episodes in streams:
        if episodes is not None:
            (output / name).write_text("".join(
                json.dumps({field: episode}) + "\n" for episode in episodes
            ))
    if diagnostic_partial_tail is not None:
        with (output / "training-diagnostics.jsonl").open("a") as stream:
            stream.write(diagnostic_partial_tail)

    import propevolve.agent as agent_module

    monkeypatch.setattr(agent_module, "RecurrentC51Agent", LoadingAgent)
    monkeypatch.setattr(
        training_module.AssetContract,
        "load",
        classmethod(lambda cls, path: object()),
    )
    monkeypatch.setattr(training_module, "ChallengeSpec", lambda **kwargs: Challenge())
    monkeypatch.setattr(
        training_module, "load_markets", lambda **kwargs: {"NQ": object()}
    )
    monkeypatch.setattr(
        training_module, "assert_temporal_role", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        training_module, "HistoricalChallengeEnv", RecoveringEnvironment
    )
    monkeypatch.setattr(
        training_module,
        "_training_resume_identity",
        lambda *args, **kwargs: "recipe-1",
    )
    training = {"seed": 7}
    if policy_health_enabled:
        training["short_circuit"] = {"policy_health": {}}
    config = {
        "_root": str(tmp_path),
        "assets": "assets.json",
        "cache_root": "cache",
        "output": str(output),
        "tickers": ("NQ",),
        "deployment_tickers": ("NQ",),
        "training_only_tickers": (),
        "timeframe_minutes": 3,
        "temporal": {
            "train_start": "2021-01-01",
            "train_end": "2025-01-01",
            "validation_start": "2025-01-01",
            "validation_end": "2026-01-01",
            "sealed_start": "2026-01-01",
        },
        "challenge": {},
        "point_values": {"NQ": 20.0},
        "round_trip_fees": {"NQ": 3.84},
        "agent": {"device": "cpu"},
        "training": training,
    }
    HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=(),
        hypothesis="durable resume evidence reconciliation",
    )
    return output


@pytest.mark.parametrize(
    ("stream", "episodes"),
    (
        ("diagnostics", None),
        ("diagnostics", (1, 2, 3, 4)),
        ("diagnostics", (1, 2, 2, 3, 4, 5)),
        ("diagnostics", (1, 2, 4, 5)),
        ("policy_health", None),
        ("policy_health", (1, 2, 3, 4)),
        ("policy_health", (1, 2, 2, 3, 4, 5)),
        ("policy_health", (1, 2, 4, 5)),
    ),
)
def test_historical_candidate_resume_rejects_incomplete_episode_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream: str,
    episodes: tuple[int, ...] | None,
) -> None:
    diagnostics = (1, 2, 3, 4, 5)
    health = None
    if stream == "diagnostics":
        diagnostics = episodes
    else:
        health = episodes
    with pytest.raises(ValueError, match="episode evidence"):
        _run_to_resume_evidence_boundary(
            tmp_path,
            monkeypatch,
            diagnostic_episodes=diagnostics,
            policy_health_episodes=health,
            policy_health_enabled=stream == "policy_health",
        )


@pytest.mark.parametrize("policy_health_enabled", (False, True))
def test_historical_candidate_resume_truncates_only_future_episode_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_health_enabled: bool,
) -> None:
    health = (1, 2, 3, 4, 5, 6, 7) if policy_health_enabled else None
    with pytest.raises(_ResumeEvidenceBoundaryReached):
        _run_to_resume_evidence_boundary(
            tmp_path,
            monkeypatch,
            diagnostic_episodes=(1, 2, 3, 4, 5, 6, 7),
            policy_health_episodes=health,
            policy_health_enabled=policy_health_enabled,
        )
    output = tmp_path / "run"
    assert [json.loads(line)["episode"] for line in (
        output / "training-diagnostics.jsonl"
    ).read_text().splitlines()] == [1, 2, 3, 4, 5]
    health_path = output / "training-policy-health.jsonl"
    if policy_health_enabled:
        assert [json.loads(line)["completed_episodes"] for line in (
            health_path.read_text().splitlines()
        )] == [1, 2, 3, 4, 5]
    else:
        assert not health_path.exists()


def test_historical_candidate_resume_drops_partial_final_future_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(_ResumeEvidenceBoundaryReached):
        _run_to_resume_evidence_boundary(
            tmp_path,
            monkeypatch,
            diagnostic_episodes=(1, 2, 3, 4, 5),
            policy_health_episodes=None,
            policy_health_enabled=False,
            diagnostic_partial_tail='{"episode":',
        )
    rows = (tmp_path / "run" / "training-diagnostics.jsonl").read_text()
    assert [json.loads(line)["episode"] for line in rows.splitlines()] == [
        1, 2, 3, 4, 5,
    ]
    assert rows.endswith("\n")


def test_historical_candidate_resume_rejects_partial_durable_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="episode evidence stream is malformed"):
        _run_to_resume_evidence_boundary(
            tmp_path,
            monkeypatch,
            diagnostic_episodes=(1, 2, 3, 4),
            policy_health_episodes=None,
            policy_health_enabled=False,
            diagnostic_partial_tail='{"episode":',
        )


def test_recovery_rejects_conflicting_replayed_pass_partial_evidence(
    tmp_path: Path,
) -> None:
    class SavingAgent:
        def save(self, path, *, manifest):
            Path(path).write_text(json.dumps(manifest, sort_keys=True))

    alias = tmp_path / "retained-pass-policy.pt"
    training_module._save_retained_policy(
        SavingAgent(),
        alias,
        resume_identity="recipe-1",
        evidence={
            "episode": 7,
            "ticker": "CL",
            "outcome": "pass",
            "terminal_pnl": 6_000.0,
        },
    )
    replayed = tmp_path / "retained-pass-policies" / "episode-000007-CL.pt"
    replayed_sha256 = hashlib.sha256(replayed.read_bytes()).hexdigest()
    partial = replayed.parent / "partial"
    partial.mkdir()
    (partial / (
        "episode-000007-CL.after-episode-000005."
        f"{replayed_sha256}.pt"
    )).write_bytes(b"conflicting forensic evidence")

    with pytest.raises(ValueError, match="partial evidence drifted"):
        training_module._reconcile_retained_pass_policies(
            alias,
            resume_identity="recipe-1",
            completed_episodes=5,
            manifest_loader=lambda path: json.loads(path.read_text()),
        )

    assert replayed.is_file()
    assert alias.is_file()


def test_recovery_reconciles_health_probe_to_durable_checkpoint(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "training-policy-health-probe.pkl"

    def write_probe(episode: int) -> str:
        with probe.open("wb") as stream:
            pickle.dump({
                "schema": "propevolve_training_policy_health_probe_corpus_v1",
                "resume_identity": "recipe-1",
                "completed_episodes": episode,
                "samples": ("fixed-row",),
            }, stream)
        return hashlib.sha256(probe.read_bytes()).hexdigest()

    abandoned_sha256 = write_probe(45)
    training_module._reconcile_policy_health_probe_corpus(
        probe,
        resume_identity="recipe-1",
        completed_episodes=40,
        checkpoint_contract=None,
    )

    assert not probe.exists()
    partial = tmp_path / (
        "training-policy-health-probe.partial-episode-000045-"
        f"{abandoned_sha256}.pkl"
    )
    assert partial.is_file()

    durable_sha256 = write_probe(45)
    training_module._reconcile_policy_health_probe_corpus(
        probe,
        resume_identity="recipe-1",
        completed_episodes=45,
        checkpoint_contract={
            "completed_episodes": 45,
            "file_sha256": durable_sha256,
        },
    )
    assert probe.is_file()

    with pytest.raises(ValueError, match="checkpoint identity drifted"):
        training_module._reconcile_policy_health_probe_corpus(
            probe,
            resume_identity="recipe-1",
            completed_episodes=45,
            checkpoint_contract={
                "completed_episodes": 45,
                "file_sha256": "0" * 64,
            },
        )


def test_recovery_checkpoint_authenticates_fixed_health_probe(
    tmp_path: Path,
) -> None:
    class SavingAgent:
        manifest = None

        def save(self, path, *, manifest):
            self.manifest = manifest
            Path(path).write_bytes(b"checkpoint")

    probe = tmp_path / "training-policy-health-probe.pkl"
    with probe.open("wb") as stream:
        pickle.dump({
            "schema": "propevolve_training_policy_health_probe_corpus_v1",
            "resume_identity": "recipe-1",
            "completed_episodes": 45,
            "samples": ("fixed-row",),
        }, stream)
    agent = SavingAgent()

    training_module._save_training_recovery(
        agent,
        tmp_path / "training-recovery.pt",
        resume_identity="recipe-1",
        progress=TrainingProgress(completed_episodes=45),
        environment_rng_state={},
        replay_state={},
        policy_health_probe_path=probe,
    )

    assert agent.manifest["policy_health_probe_corpus"] == {
        "completed_episodes": 45,
        "file_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
    }


def test_training_uses_lower_exploration_for_position_management() -> None:
    class RecordingAgent(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.epsilons = []

        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            self.epsilons.append((valid_actions, epsilon))
            return valid_actions[0], None, None

    class PositionEnvironment:
        def __init__(self) -> None:
            self.index = 0

        def reset(self):
            self.index = 0
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT, Action.ENTER_LONG_1),
            }

        def step(self, action):
            self.index += 1
            terminated = self.index == 2
            return np.array([self.index], np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.HOLD, Action.CLOSE),
                "outcome": "timeout" if terminated else None,
                "ticker": "NQ",
                "primary_side": "long",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    agent = RecordingAgent()
    train_agent(
        agent,
        PositionEnvironment(),
        episodes=1,
        minimum_environment_steps=2,
        replay=BalancedSequenceReplay(capacity_episodes=2, sequence_length=1, seed=3),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        management_epsilon_start=0.05,
        management_epsilon_end=0.01,
        episode_tickers=None,
        ticker_seed=3,
    )

    assert agent.epsilons == [
        ((Action.WAIT, Action.ENTER_LONG_1), 0.25),
        ((Action.HOLD, Action.CLOSE), 0.05),
    ]


def test_training_resumes_from_an_episode_boundary() -> None:
    checkpoints: list[TrainingProgress] = []
    first = train_agent(
        Agent(),
        Environment(),
        episodes=1,
        minimum_environment_steps=4,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            capacity_transitions=16,
            sequence_length=2,
            seed=5,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        checkpoint_every_episodes=1,
        checkpoint_callback=checkpoints.append,
    )
    assert first.environment_steps == 4

    resumed = train_agent(
        Agent(),
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            capacity_transitions=16,
            sequence_length=2,
            seed=5,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        resume=checkpoints[-1],
    )

    assert resumed.episodes == 2
    assert resumed.environment_steps == 8
    assert resumed.passes == 2


def test_episode_budget_resume_does_not_repeat_completed_episodes() -> None:
    class SimulatedCrash(RuntimeError):
        pass

    class BoundedEnvironment(Environment):
        def __init__(self) -> None:
            super().__init__()
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            observation, info = super().reset()
            return observation, {**info, "ticker": "NQ", "start": 0, "end": 4}

    checkpoints: list[TrainingProgress] = []
    first_environment = BoundedEnvironment()

    def interrupt_after_second_episode(progress):
        checkpoints.append(progress)
        if progress.completed_episodes == 2:
            raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        train_agent(
            Agent(),
            first_environment,
            episodes=4,
            minimum_environment_steps=1,
            budget_mode="episodes",
            replay=BalancedSequenceReplay(
                capacity_episodes=8,
                capacity_transitions=32,
                sequence_length=2,
                seed=59,
            ),
            warmup_episodes=1,
            updates_per_episode=1,
            batch_sequences=1,
            recurrent_horizon=2,
            epsilon_start=0.25,
            epsilon_end=0.02,
            episode_tickers=None,
            ticker_seed=59,
            checkpoint_every_episodes=1,
            checkpoint_callback=interrupt_after_second_episode,
        )

    resumed_environment = BoundedEnvironment()
    resumed = train_agent(
        Agent(),
        resumed_environment,
        episodes=4,
        minimum_environment_steps=1,
        budget_mode="episodes",
        replay=BalancedSequenceReplay(
            capacity_episodes=8,
            capacity_transitions=32,
            sequence_length=2,
            seed=59,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=59,
        resume=checkpoints[-1],
    )

    assert first_environment.reset_count == 2
    assert resumed.episodes == 4
    assert resumed.environment_steps == 16
    assert resumed_environment.reset_count == 2


def test_training_never_clears_a_resumed_terminal_collapse() -> None:
    terminal = TrainingProgress(
        completed_episodes=3,
        environment_steps=3,
        passes=1,
        timeouts=2,
        terminal_pnl_count=3,
        reward_count=3,
        short_circuit_reason="policy collapse",
    )

    result = train_agent(
        Agent(),
        Environment(),
        episodes=10,
        minimum_environment_steps=10,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            sequence_length=1,
            seed=5,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        resume=terminal,
    )

    assert result.episodes == 3
    assert result.short_circuited is True
    assert result.short_circuit_reason == "policy collapse"


def test_training_checkpoints_the_final_episode_outside_periodic_interval() -> None:
    checkpoints: list[TrainingProgress] = []

    result = train_agent(
        Agent(),
        Environment(),
        episodes=2,
        minimum_environment_steps=8,
        replay=BalancedSequenceReplay(
            capacity_episodes=4,
            capacity_transitions=16,
            sequence_length=2,
            seed=5,
        ),
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=None,
        ticker_seed=5,
        checkpoint_every_episodes=5,
        checkpoint_callback=checkpoints.append,
    )

    assert result.environment_steps == 8
    assert [checkpoint.completed_episodes for checkpoint in checkpoints] == [2]


def test_prop_safety_objective_hard_ranks_any_blow_below_zero_blow() -> None:
    common = dict(
        episodes=100,
        environment_steps=1000,
        passes=50,
        timeouts=50,
        trade_count=100,
        win_count=40,
        winning_r_sum=80.0,
        worst_pnl=-2_000.0,
        mean_terminal_pnl=2_000.0,
        mean_reward=0.0,
        mean_loss=1.0,
    )
    safe = TrainingResult(blows=0, **common)
    unsafe = TrainingResult(blows=1, **{**common, "passes": 99, "timeouts": 0})

    assert prop_safety_objective(
        unsafe, max_loss=3_000.0, profit_target=6_000.0
    ) < -1.0
    assert prop_safety_objective(
        safe, max_loss=3_000.0, profit_target=6_000.0
    ) >= 0.0


def test_prop_safety_objective_penalizes_near_blow_timeouts() -> None:
    common = dict(
        episodes=100,
        environment_steps=1000,
        passes=10,
        blows=0,
        timeouts=90,
        trade_count=100,
        win_count=40,
        winning_r_sum=80.0,
        worst_pnl=-2_500.0,
        mean_terminal_pnl=-500.0,
        mean_reward=0.0,
        mean_loss=1.0,
    )
    safe = TrainingResult(near_blow_timeout_count=0, **common)
    near_blow = TrainingResult(near_blow_timeout_count=45, **common)

    assert prop_safety_objective(
        safe, max_loss=3_000.0, profit_target=6_000.0
    ) > prop_safety_objective(
        near_blow, max_loss=3_000.0, profit_target=6_000.0
    )


def test_evaluation_never_updates_agent() -> None:
    agent = Agent()
    result = evaluate_agent(agent, Environment(), episodes=2, recurrent_horizon=2)
    assert result.passes == 2
    assert agent.updates == 0


def test_teacher_free_evaluation_performs_zero_teacher_lookups() -> None:
    class TeacherLookupTripwireEnvironment(Environment):
        def teacher_lookup(self, ticker: str, decision_index: int):
            raise AssertionError("teacher-free evaluation accessed a teacher")

    result = evaluate_agent(
        Agent(),
        TeacherLookupTripwireEnvironment(),
        episodes=2,
        recurrent_horizon=2,
    )

    assert result.passes == 2


def test_greedy_evaluation_preserves_serialized_agent_state(
    tmp_path: Path,
) -> None:
    from propevolve.agent import RecurrentC51Agent

    agent = RecurrentC51Agent(
        1,
        hidden_dim=8,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=1e-4,
        weight_decay=1e-5,
        gradient_clip=10.0,
        target_sync_updates=250,
        device="cpu",
        seed=71,
    )
    before_path = agent.save(tmp_path / "before.pt", manifest={})

    result = evaluate_agent(
        agent,
        Environment(),
        episodes=2,
        recurrent_horizon=2,
    )
    after_path = agent.save(tmp_path / "after.pt", manifest={})
    before = torch.load(before_path, map_location="cpu", weights_only=False)
    after = torch.load(after_path, map_location="cpu", weights_only=False)

    assert result.passes == 2
    assert all(
        torch.equal(before[network][key], after[network][key])
        for network in ("online", "target")
        for key in before[network]
    )
    assert before["optimizer"] == after["optimizer"]
    assert before["updates"] == after["updates"]
    assert before["rng_state"] == after["rng_state"]


def test_validation_rejects_a_policy_that_still_contains_training_teachers() -> None:
    from propevolve.agent import RecurrentC51Agent

    agent = RecurrentC51Agent(
        1,
        hidden_dim=8,
        atoms=11,
        value_min=-3.0,
        value_max=3.0,
        gamma=0.997,
        learning_rate=1e-4,
        weight_decay=1e-5,
        gradient_clip=10.0,
        target_sync_updates=250,
        device="cpu",
        seed=73,
        teacher_channels=1,
        teacher_channel_names=("training_only_example",),
        teacher_loss_weight=0.1,
    )

    with pytest.raises(
        ValueError,
        match="validation policy still contains training-only teacher state",
    ):
        evaluate_agent(agent, Environment(), episodes=1, recurrent_horizon=2)

    agent.discard_teacher()
    result = evaluate_agent(agent, Environment(), episodes=1, recurrent_horizon=2)
    assert result.passes == 1


def test_teacher_free_evaluation_reports_both_entry_sides() -> None:
    class DirectionalAgent(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.actions = iter((Action.ENTER_LONG_1, Action.ENTER_SHORT_1))

        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            return next(self.actions), None, np.zeros(len(Action), np.float32)

    class FlatEpisodeEnvironment:
        def reset(self):
            return np.zeros(1, np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            }

        def step(self, action):
            return np.zeros(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "timeout",
                "ticker": "NQ",
                "trade_count": 1,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    result = evaluate_agent(
        DirectionalAgent(),
        FlatEpisodeEnvironment(),
        episodes=2,
        recurrent_horizon=2,
    )

    assert result.long_entry_count == 1
    assert result.short_entry_count == 1
    assert result.greedy_entry_count == 2


def test_teacher_free_validation_emits_one_bounded_regime_diagnostic_per_episode() -> None:
    class DiagnosticAgent(Agent):
        def __init__(self) -> None:
            super().__init__()
            self.rows = iter((
                (Action.WAIT, (1.0, 0.5, 0.25)),
                (Action.ENTER_LONG_1, (0.0, 2.0, -1.0)),
                (Action.ENTER_SHORT_1, (0.0, -1.0, 3.0)),
            ))

        def select_action(
            self,
            observation,
            *,
            hidden,
            valid_actions,
            epsilon,
            return_action_values=False,
        ):
            action, flat_values = next(self.rows)
            values = np.full(len(Action), -10.0, dtype=np.float32)
            values[:3] = flat_values
            return action, None, values if return_action_values else None

    class DiagnosticEnvironment:
        def __init__(self) -> None:
            self.step_index = 0

        def reset(self):
            self.step_index = 0
            return np.zeros(1, np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "ticker": "NQ",
                "mll_headroom_fraction": 0.10,
            }

        def step(self, action):
            self.step_index += 1
            terminated = self.step_index == 3
            headroom = (0.20, 0.80, 0.80)[self.step_index - 1]
            return np.ones(1, np.float32), 0.5, terminated, False, {
                "valid_actions": () if terminated else (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                ),
                "ticker": "NQ",
                "outcome": "timeout" if terminated else None,
                "equity_pnl": -2_400.0,
                "mll_headroom_fraction": headroom,
                "trade_count": 2,
                "win_count": 1,
                "winning_r_sum": 2.5,
                "expectancy_r": 0.5,
                "avg_mfe_r": 2.0,
                "avg_mae_r": 0.75,
            }

        def closed_trade_receipts(self):
            return (
                {
                    "trade_index": 0,
                    "ticker": "NQ",
                    "side": "long",
                    "source_decision_index": 0,
                    "entry_mll_headroom": 250.0,
                    "realized_r": -1.0,
                    "mfe_r": 0.25,
                    "mae_r": 1.0,
                    "hold_bars": 2,
                    "exit_reason": "initial_stop",
                },
                {
                    "trade_index": 1,
                    "ticker": "NQ",
                    "side": "short",
                    "source_decision_index": 2,
                    "entry_mll_headroom": 600.0,
                    "realized_r": 2.0,
                    "mfe_r": 2.5,
                    "mae_r": 0.5,
                    "hold_bars": 5,
                    "exit_reason": "ratchet_stop",
                },
            )

    diagnostics: list[dict[str, object]] = []
    evaluate_agent(
        DiagnosticAgent(),
        DiagnosticEnvironment(),
        episodes=1,
        recurrent_horizon=8,
        near_blow_loss_threshold=2_250.0,
        greedy_diagnostic_interval_steps=1,
        episode_diagnostic_callback=diagnostics.append,
    )

    assert len(diagnostics) == 1
    row = diagnostics[0]
    assert row["schema"] == "propevolve_validation_episode_diagnostic_v1"
    assert row["episode"] == 1
    assert row["ticker"] == "NQ"
    assert row["outcome"] == "timeout"
    assert row["terminal_pnl"] == -2_400.0
    assert row["near_blow_timeout"] is True
    assert row["trade_count"] == 2
    assert row["win_rate"] == 0.5
    assert row["average_win_r"] == 2.5
    assert row["average_mfe_r"] == 2.0
    assert row["average_mae_r"] == 0.75
    assert row["flat_greedy_action_counts"] == {
        "WAIT": 1,
        "ENTER_LONG_1": 1,
        "ENTER_SHORT_1": 1,
    }
    assert row["flat_entry_rate"] == pytest.approx(2.0 / 3.0)
    assert row["entry_counts"] == {
        "ENTER_LONG_1": 1,
        "ENTER_SHORT_1": 1,
    }
    assert row["headroom"] == {
        "le_0_25": {"flat_decisions": 2, "entries": 1},
        "between_0_25_and_0_75": {"flat_decisions": 0, "entries": 0},
        "ge_0_75": {"flat_decisions": 1, "entries": 1},
        "unavailable": {"flat_decisions": 0, "entries": 0},
    }
    assert row["closed_trade_economics"] == {
        "reported_trade_count": 2,
        "receipt_trade_count": 2,
        "unattributed_trade_count": 0,
        "receipt_coverage": 1.0,
        "groups": [
            {
                "side": "long",
                "episode_outcome": "timeout",
                "entry_headroom_stratum": "critical_le_300",
                "trades": 1,
                "wins": 0,
                "win_rate": 0.0,
                "realized_r_sum": -1.0,
                "realized_r_mean": -1.0,
                "mfe_r_sum": 0.25,
                "mfe_r_mean": 0.25,
                "mae_r_sum": 1.0,
                "mae_r_mean": 1.0,
                "hold_bars_sum": 2,
                "hold_bars_mean": 2.0,
                "exit_reason_counts": {"initial_stop": 1},
            },
            {
                "side": "short",
                "episode_outcome": "timeout",
                "entry_headroom_stratum": "safe_ge_500",
                "trades": 1,
                "wins": 1,
                "win_rate": 1.0,
                "realized_r_sum": 2.0,
                "realized_r_mean": 2.0,
                "mfe_r_sum": 2.5,
                "mfe_r_mean": 2.5,
                "mae_r_sum": 0.5,
                "mae_r_mean": 0.5,
                "hold_bars_sum": 5,
                "hold_bars_mean": 5.0,
                "exit_reason_counts": {"ratchet_stop": 1},
            },
        ],
    }
    assert row["sampled_q_margins"] == pytest.approx({
        "rows": 3,
        "best_entry_minus_wait_mean": 1.5,
        "long_minus_wait_mean": 1.0 / 6.0,
        "short_minus_wait_mean": 5.0 / 12.0,
        "best_entry_minus_wait_min": -0.5,
        "best_entry_minus_wait_max": 3.0,
    })


def test_teacher_free_recovery_stress_keeps_public_outcomes_and_status_separate() -> None:
    class StressEnvironment:
        def __init__(self) -> None:
            self.episode = -1

        def reset(self, *, options=None):
            self.episode += 1
            assert options["challenge_start_state"] == (
                _recovery_curriculum_settings().start_state
            )
            return np.zeros(1, np.float32), {
                "valid_actions": (Action.WAIT,),
            }

        def step(self, action):
            outcome = (
                "pass",
                "timeout",
                "timeout",
                "blow",
            )[self.episode]
            recovered = self.episode == 0
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": outcome,
                "equity_pnl": (6_000.0, -2_650.0, -2_700.0, -3_000.0)[
                    self.episode
                ],
                "recovery_status": (
                    "recovered" if recovered else "not_recovered"
                ),
                "recovery_wait_decisions": int(self.episode == 2),
                "trade_count": (2, 1, 0, 1)[self.episode],
            }

    agent = Agent()
    result = evaluate_recovery_stress(
        agent,
        StressEnvironment(),
        episodes=4,
        recurrent_horizon=96,
        settings=_recovery_curriculum_settings(),
    )

    assert isinstance(result, RecoveryStressResult)
    assert result.recovered == 1
    assert result.not_recovered == 3
    assert result.passes == 1
    assert result.timeouts == 2
    assert result.blows == 1
    assert result.recovery_success_rate == 0.25
    assert result.blow_rate == 0.25
    assert result.entries_used == 4
    assert agent.updates == 0


def test_recovery_stress_integrity_allows_baseline_evidence_but_economic_gate_rejects_blow() -> None:
    baseline_metrics = {
        "outcome_accounted": 1.0,
        "blow_rate": 0.25,
    }

    assert all(
        gate.passes(baseline_metrics)
        for gate in training_module._recovery_stress_integrity_gates()
    )
    economic_gate = training_module.EvaluationGate("blow_rate", "==", 0.0)
    assert economic_gate.passes(baseline_metrics) is False


def test_recovery_stress_rejects_synthetic_recovery_outcomes() -> None:
    class InvalidEnvironment:
        def reset(self, *, options=None):
            return np.zeros(1, np.float32), {"valid_actions": (Action.WAIT,)}

        def step(self, action):
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "survived_not_recovered",
                "equity_pnl": -2_500.0,
                "recovery_status": "not_recovered",
                "trade_count": 1,
            }

    with pytest.raises(ValueError, match="unknown recovery stress outcome"):
        evaluate_recovery_stress(
            Agent(),
            InvalidEnvironment(),
            episodes=1,
            recurrent_horizon=96,
            settings=_recovery_curriculum_settings(),
        )


def test_evaluation_reports_pass_and_timeout_economics_separately(capsys) -> None:
    class OutcomeEnvironment:
        def __init__(self) -> None:
            self.episode = -1

        def reset(self):
            self.episode += 1
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,)
            }

        def step(self, action):
            outcome = ("pass", "timeout")[self.episode]
            info = {
                "valid_actions": (),
                "ticker": ("NQ", "SI")[self.episode],
                "outcome": outcome,
                "trade_count": (4, 10)[self.episode],
                "win_count": (2, 3)[self.episode],
                "winning_r_sum": (6.0, 3.0)[self.episode],
                "equity_pnl": (6_000.0, 1_500.0)[self.episode],
            }
            return np.array([1.0], np.float32), 1.0, True, False, info

    result = evaluate_agent(
        Agent(), OutcomeEnvironment(), episodes=2, recurrent_horizon=2
    )

    assert result.outcome("pass").mean_trade_count == 4.0
    assert result.outcome("pass").trade_win_rate == 0.5
    assert result.outcome("pass").average_win_r == 3.0
    assert result.outcome("timeout").mean_trade_count == 10.0
    assert result.outcome("timeout").trade_win_rate == 0.3
    assert result.outcome("timeout").average_win_r == 1.0
    assert result.outcome("timeout").mean_terminal_pnl == 1_500.0
    output = capsys.readouterr().out
    assert (
        "[validation] episode=1/2 ticker=NQ outcome=pass "
        "reward=+1.0000 trades=4 WR=50.0% winR=+3.000R pnl=+6000.00 "
        "cumulative_pass=1 cumulative_blow=0 cumulative_timeout=0"
    ) in output
    assert (
        "[validation] episode=2/2 ticker=SI outcome=timeout "
        "reward=+1.0000 trades=10 WR=30.0% winR=+1.000R pnl=+1500.00 "
        "cumulative_pass=1 cumulative_blow=0 cumulative_timeout=1"
    ) in output
    assert (
        "[validation] COMPLETE episodes=2 pass=1 blow=0 timeout=1 "
        "near_blow_timeout=0 (0.0%) WR=35.7% winR=+1.800R "
        "mean_pnl=+3750.00"
    ) in output


def test_evaluation_counts_timeouts_near_the_loss_limit(capsys) -> None:
    class NearBlowEnvironment:
        def __init__(self) -> None:
            self.episode = -1

        def reset(self):
            self.episode += 1
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,)
            }

        def step(self, action):
            pnl = (-2_500.0, -1_000.0)[self.episode]
            return np.array([1.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": "NQ",
                "outcome": "timeout",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": pnl,
            }

    result = evaluate_agent(
        Agent(),
        NearBlowEnvironment(),
        episodes=2,
        recurrent_horizon=2,
        near_blow_loss_threshold=2_250.0,
    )

    assert result.near_blow_timeout_count == 1
    assert result.near_blow_timeout_rate == 0.5
    assert "near_blow_timeout=1 (50.0%)" in capsys.readouterr().out


def test_evaluation_short_circuits_after_first_blow_when_zero_blow_is_required(
    capsys,
) -> None:
    class BlowThenPassEnvironment:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,),
            }

        def step(self, action):
            outcome = "blow" if self.reset_count == 1 else "pass"
            pnl = -3_000.0 if outcome == "blow" else 6_000.0
            return np.array([0.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": outcome,
                "ticker": "NQ",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": pnl,
            }

    environment = BlowThenPassEnvironment()

    result = evaluate_agent(
        Agent(),
        environment,
        episodes=200,
        recurrent_horizon=2,
        stop_on_first_blow=True,
    )

    assert result.episodes == 1
    assert result.blows == 1
    assert result.passes == result.timeouts == 0
    assert environment.reset_count == 1
    output = capsys.readouterr().out
    assert "SHORT_CIRCUIT reason=zero_blow_gate" in output
    assert "COMPLETE episodes=1/200" in output


def test_evaluation_short_circuits_a_universal_wait_policy(capsys) -> None:
    class ZeroTradeEnvironment:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1
            return np.array([0.0], np.float32), {
                "valid_actions": (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            }

        def step(self, action):
            assert action == Action.WAIT
            return np.array([0.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "timeout",
                "ticker": "NQ",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": 0.0,
            }

    environment = ZeroTradeEnvironment()
    result = evaluate_agent(
        Agent(),
        environment,
        episodes=200,
        recurrent_horizon=2,
        no_trade_patience_episodes=5,
    )

    assert result.episodes == 5
    assert result.short_circuited is True
    assert result.short_circuit_reason == (
        "universal_wait: 5 consecutive zero-trade episodes"
    )
    assert result.trade_count == 0
    assert environment.reset_count == 5
    output = capsys.readouterr().out
    assert output.count("SHORT_CIRCUIT reason=universal_wait") == 1
    assert "COMPLETE episodes=5/200" in output


def test_validation_no_trade_patience_resets_after_a_traded_episode() -> None:
    class SparseTradeEnvironment:
        def __init__(self) -> None:
            self.episode = -1

        def reset(self):
            self.episode += 1
            return np.array([0.0], np.float32), {
                "valid_actions": (Action.WAIT,)
            }

        def step(self, action):
            traded = self.episode in {2, 5}
            return np.array([0.0], np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": "timeout",
                "ticker": "NQ",
                "trade_count": int(traded),
                "win_count": int(traded),
                "winning_r_sum": float(traded),
                "equity_pnl": 0.0,
            }

    result = evaluate_agent(
        Agent(),
        SparseTradeEnvironment(),
        episodes=6,
        recurrent_horizon=2,
        no_trade_patience_episodes=3,
    )

    assert result.episodes == 6
    assert result.short_circuited is False
    assert result.trade_count == 2


def test_one_shared_agent_trains_on_balanced_single_market_episodes() -> None:
    tickers = ("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN")
    environment = MultiMarketEnvironment()
    agent = Agent()
    replay = BalancedSequenceReplay(capacity_episodes=30, sequence_length=2, seed=7)

    result = train_agent(
        agent,
        environment,
        episodes=18,
        minimum_environment_steps=72,
        replay=replay,
        warmup_episodes=1,
        updates_per_episode=1,
        batch_sequences=1,
        recurrent_horizon=2,
        epsilon_start=0.25,
        epsilon_end=0.02,
        episode_tickers=tickers,
        ticker_seed=7,
    )

    assert Counter(environment.episode_tickers) == Counter({ticker: 2 for ticker in tickers})
    assert result.episodes == 18
    assert agent.updates == 18


def test_temporal_preflight_rejects_any_sealed_holdout_timestamp() -> None:
    timestamps = np.array([
        "2025-12-31T23:57:00", "2026-01-01T00:00:00"
    ], dtype="datetime64[ns]")
    close = np.array([100.0, 101.0], np.float32)
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=close,
        high=close,
        low=close,
        close=close,
        embeddings=np.zeros((2, 4), np.float32),
    )

    with pytest.raises(ValueError, match="temporal contract"):
        assert_temporal_role(
            {"NQ": market},
            role="selection",
            start="2025-01-01",
            end="2026-01-01",
            sealed_start="2026-01-01",
        )
