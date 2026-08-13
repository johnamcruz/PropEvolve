from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import propevolve.training as training_module
from propevolve.cache import build_embedding_cache
from propevolve.decision import Action, PositionSide, RecoveryEntryPermit
from propevolve.environment import ChallengeSpec, ChallengeStartState, MarketSeries
from propevolve.observation import TradeManagementObservationSpec
from propevolve.replay import BalancedSequenceReplay
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
    _regime_selectivity_agent_settings,
    _regime_selectivity_replay_settings,
    _recovery_curriculum_from_config,
    _selection_evaluation_gates,
    _plain_contract_value,
    assert_temporal_role,
    evaluate_agent,
    evaluate_recovery_stress,
    prop_safety_objective,
    train_agent,
)


def _recovery_curriculum_settings(*, fraction: float = 0.5) -> RecoveryCurriculumSettings:
    return RecoveryCurriculumSettings(
        episode_fraction=fraction,
        schedule_seed=37,
        start_state=ChallengeStartState(
            realized_pnl=-2_700.0,
            equity_pnl=-2_700.0,
            peak_equity_pnl=0.0,
            mll_floor_pnl=-3_000.0,
            passmark_locked=False,
            position_side=PositionSide.FLAT,
            position_size=0,
            session_pnl=-2_700.0,
            trading_days_elapsed=1,
            recovery_entry_permit=RecoveryEntryPermit(
                remaining_entries=1,
                exception_headroom=300.0,
                success_pnl=-2_500.0,
            ),
        ),
    )


def test_json_recovery_curriculum_projects_complete_frozen_start_contract() -> None:
    settings, stress_episodes = _recovery_curriculum_from_config({
        "episode_fraction": 0.25,
        "schedule_seed": 37,
        "stress_evaluation_episodes": 200,
        "start_state": {
            "realized_pnl": -2_700.0,
            "equity_pnl": -2_700.0,
            "peak_equity_pnl": 0.0,
            "mll_floor_pnl": -3_000.0,
            "passmark_locked": False,
            "position_side": 0,
            "position_size": 0,
            "session_pnl": -2_700.0,
            "trading_days_elapsed": 1,
        },
        "entry_permit": {
            "remaining_entries": 1,
            "exception_headroom": 300.0,
            "success_pnl": -2_500.0,
        },
    })

    assert settings == _recovery_curriculum_settings(fraction=0.25)
    assert stress_episodes == 200


def test_json_recovery_curriculum_rejects_missing_or_drifted_fields() -> None:
    with pytest.raises(ValueError, match="fields are invalid"):
        _recovery_curriculum_from_config({
            "episode_fraction": 0.25,
            "schedule_seed": 37,
        })
    with pytest.raises(ValueError, match="contract drifted"):
        _recovery_curriculum_from_config({
            "episode_fraction": 0.25,
            "schedule_seed": 37,
            "start_state": {
                "realized_pnl": -2_600.0,
                "equity_pnl": -2_600.0,
                "peak_equity_pnl": 0.0,
                "mll_floor_pnl": -3_000.0,
                "passmark_locked": False,
                "position_side": 0,
                "position_size": 0,
                "session_pnl": -2_600.0,
                "trading_days_elapsed": 1,
            },
            "entry_permit": {
                "remaining_entries": 1,
                "exception_headroom": 300.0,
                "success_pnl": -2_500.0,
            },
            "stress_evaluation_episodes": 200,
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
            "structure_chop_probability": {
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
        "structure_chop_probability"
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


def test_persistent_regime_gate_requires_learned_wait_separation(
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
        **optimizer_metrics,
        "sampled_entry_action_long_rows": 10.0,
        "sampled_entry_action_short_rows": 10.0,
        "sampled_entry_action_long_recall": 0.5,
        "sampled_entry_action_short_recall": 0.5,
        "regime_selectivity_positive_long_rows": 10.0,
        "regime_selectivity_positive_short_rows": 10.0,
        "regime_selectivity_positive_long_declared_side_probability_sum": 5.0,
        "regime_selectivity_positive_short_declared_side_probability_sum": 5.0,
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

    metrics["final_regime_probe_dead_wait_minus_transition_ready_wait"] = 0.0
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
        ):
            captured.append((spec, observation_spec))
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
        "recovery_curriculum": {
            "episode_fraction": 0.0,
            "schedule_seed": 37,
            "stress_evaluation_episodes": 0,
            "start_state": {
                "realized_pnl": -2_700.0,
                "equity_pnl": -2_700.0,
                "peak_equity_pnl": 0.0,
                "mll_floor_pnl": -3_000.0,
                "passmark_locked": False,
                "position_side": 0,
                "position_size": 0,
                "session_pnl": -2_700.0,
                "trading_days_elapsed": 1,
            },
            "entry_permit": {
                "remaining_entries": 1,
                "exception_headroom": 300.0,
                "success_pnl": -2_500.0,
            },
        },
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
            "minimum_environment_steps": 2,
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
    assert evaluation.path.is_file()
    contract = json.loads((candidate.path / "contract.json").read_text())
    assert (
        contract["entry_action_loss_reduction"]
        == "equal_present_class_mean_v1"
    )
    assert contract["recovery_curriculum"] == config["recovery_curriculum"]
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


def test_training_deterministically_mixes_complete_recovery_starts_and_keeps_short_traces() -> None:
    class RecoveryAgent(Agent):
        recurrent_burn_in = 64
        n_step_return = 8

    class MixedEnvironment:
        def __init__(self) -> None:
            self.recovery_flags: list[bool] = []
            self.ticker = "NQ"

        def reset(self, *, options=None):
            options = options or {}
            recovery = "challenge_start_state" in options
            self.recovery_flags.append(recovery)
            if recovery:
                assert options["challenge_start_state"] == (
                    _recovery_curriculum_settings().start_state
                )
            self.ticker = str(options.get("ticker", "NQ"))
            return np.zeros(1, np.float32), {
                "valid_actions": (Action.WAIT,),
                "ticker": self.ticker,
                "start": 0,
                "mll_headroom_fraction": 0.1 if recovery else 1.0,
            }

        def step(self, action):
            recovery = self.recovery_flags[-1]
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": self.ticker,
                "fill_index": 1,
                "outcome": "pass",
                "primary_side": "long" if recovery else "flat",
                "trade_count": int(recovery),
                "win_count": int(recovery),
                "winning_r_sum": float(recovery),
                "equity_pnl": -2_500.0 if recovery else 6_000.0,
                "recovery_entry_used": recovery,
                "recovery_trade_closed": recovery,
                "recovery_success": recovery,
                "recovery_wait_decisions": 0,
                "recovery_entry_permit_remaining": 0,
            }

    settings = _recovery_curriculum_settings()
    environment = MixedEnvironment()
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
        episode_diagnostic_callback=diagnostics.append,
    )

    assert environment.recovery_flags.count(True) == 2
    assert result.passes == 4
    assert result.recovery_episodes == 2
    assert result.recovery_successes == 2
    assert result.recovery_success_rate == 1.0
    assert replay.transition_count == 4
    assert len(replay) == 4
    assert Counter(item["episode_kind"] for item in diagnostics) == {
        "ordinary": 2,
        "recovery": 2,
    }
    assert diagnostics[-1]["cumulative_recovery_episodes"] == 2


def test_mixed_recovery_schedule_resumes_by_episode_index_exactly() -> None:
    class ScheduleEnvironment:
        def __init__(self) -> None:
            self.flags: list[bool] = []

        def reset(self, *, options=None):
            recovery = bool(options and "challenge_start_state" in options)
            self.flags.append(recovery)
            return np.zeros(1, np.float32), {"valid_actions": (Action.WAIT,)}

        def step(self, action):
            recovery = self.flags[-1]
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "ticker": "NQ",
                "outcome": "wait_timeout" if recovery else "timeout",
                "primary_side": "flat",
                "trade_count": 0,
                "win_count": 0,
                "winning_r_sum": 0.0,
                "equity_pnl": -2_700.0 if recovery else 0.0,
                "recovery_wait_decisions": int(recovery),
                "recovery_entry_permit_remaining": int(recovery),
            }

    def run(environment, minimum_steps, *, resume=None, checkpoints=None):
        return train_agent(
            Agent(),
            environment,
            episodes=4,
            minimum_environment_steps=minimum_steps,
            replay=BalancedSequenceReplay(
                capacity_episodes=8, sequence_length=1, seed=7
            ),
            warmup_episodes=99,
            updates_per_episode=1,
            batch_sequences=1,
            recurrent_horizon=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            episode_tickers=None,
            ticker_seed=7,
            recovery_curriculum=_recovery_curriculum_settings(),
            resume=resume,
            checkpoint_every_episodes=1 if checkpoints is not None else 0,
            checkpoint_callback=(checkpoints.append if checkpoints is not None else None),
        )

    full_environment = ScheduleEnvironment()
    run(full_environment, 4)
    partial_environment = ScheduleEnvironment()
    checkpoints: list[TrainingProgress] = []
    run(partial_environment, 2, checkpoints=checkpoints)
    resumed_environment = ScheduleEnvironment()
    run(resumed_environment, 4, resume=checkpoints[-1])

    assert partial_environment.flags + resumed_environment.flags == (
        full_environment.flags
    )

def test_teacher_curriculum_is_gradual_and_deterministic() -> None:
    agent = Agent()
    diagnostics = []
    observed_targets = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed_targets.extend(
                transition.teacher_target for transition in episode.transitions
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

    assert [target is not None for target in observed_targets] == [
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
    observed_targets = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed_targets.extend(
                transition.teacher_target for transition in episode.transitions
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

    assert [target is not None for target in observed_targets] == [
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

    observed = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed.extend(
                transition.teacher_target for transition in episode.transitions
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

    assert observed[8:] == [None, None]


def test_entry_action_supervision_is_separate_and_shares_autonomy_boundary() -> None:
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

    observed: list[tuple[np.ndarray | None, Action | None]] = []
    diagnostics = []

    class CapturingReplay(BalancedSequenceReplay):
        def add(self, episode):
            observed.extend(
                (transition.teacher_target, transition.entry_action_target)
                for transition in episode.transitions
            )
            super().add(episode)

    train_agent(
        Agent(),
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
        episode_diagnostic_callback=diagnostics.append,
    )

    # The soft semantic teacher and sparse economic action target remain
    # independent fields, but share one deterministic visibility curriculum.
    assert any(
        semantic is not None and action == Action.ENTER_LONG_1
        for semantic, action in observed[:8]
    )
    assert all(
        (semantic is None) == (action is None)
        for semantic, action in observed
    )
    assert observed[8:] == [(None, None), (None, None)]
    assert diagnostics[0]["entry_action_target_counts"]["ENTER_LONG_1"] > 0
    assert diagnostics[0]["entry_action_target_counts"]["WAIT"] == 0


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
            "structure_trend_probability",
            "structure_chop_probability",
        ),
        episode_diagnostic_callback=diagnostics.append,
    )

    assert diagnostics[0]["selected_teacher_channel_means"] == pytest.approx({
        "long_attempt_probability": 0.2,
        "long_clean_retained_given_attempt_probability": 0.7,
        "short_attempt_probability": 0.1,
        "short_clean_retained_given_attempt_probability": 0.6,
        "structure_trend_probability": 0.8,
        "structure_chop_probability": 0.05,
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
        "structure_chop_probability",
        "structure_neutral_probability",
        "structure_trend_probability",
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
            "structure_chop_probability": pytest.approx(0.9),
            "structure_neutral_probability": pytest.approx(0.05),
            "structure_trend_probability": pytest.approx(0.05),
        },
        "regime_channel_probability_means": {
            "structure_chop_probability": pytest.approx(0.9),
            "structure_neutral_probability": pytest.approx(0.05),
            "structure_trend_probability": pytest.approx(0.05),
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


def test_teacher_free_recovery_stress_reports_distinct_outcomes_and_one_entry() -> None:
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
                "survived_not_recovered",
                "wait_timeout",
                "blow",
            )[self.episode]
            entered = outcome != "wait_timeout"
            return np.ones(1, np.float32), 0.0, True, False, {
                "valid_actions": (),
                "outcome": outcome,
                "equity_pnl": {
                    "pass": 6_000.0,
                    "survived_not_recovered": -2_650.0,
                    "wait_timeout": -2_700.0,
                    "blow": -3_000.0,
                }[outcome],
                "recovery_entry_used": entered,
                "recovery_trade_closed": entered,
                "recovery_success": outcome == "pass",
                "recovery_wait_decisions": int(not entered),
            }

    agent = Agent()
    result = evaluate_recovery_stress(
        agent,
        StressEnvironment(),
        episodes=4,
        recurrent_horizon=96,
        settings=_recovery_curriculum_settings(fraction=1.0),
    )

    assert isinstance(result, RecoveryStressResult)
    assert result.recovery_successes == 1
    assert result.survived_not_recovered == 1
    assert result.wait_timeouts == 1
    assert result.blows == 1
    assert result.recovery_success_rate == 0.25
    assert result.blow_rate == 0.25
    assert result.entries_used == 3
    assert result.one_entry_violations == 0
    assert agent.updates == 0


def test_recovery_stress_integrity_allows_baseline_evidence_but_economic_gate_rejects_blow() -> None:
    baseline_metrics = {
        "one_entry_violations": 0.0,
        "blow_rate": 0.25,
    }

    assert all(
        gate.passes(baseline_metrics)
        for gate in training_module._recovery_stress_integrity_gates()
    )
    economic_gate = training_module.EvaluationGate("blow_rate", "==", 0.0)
    assert economic_gate.passes(baseline_metrics) is False


def test_recovery_stress_fails_closed_on_a_second_entry() -> None:
    class InvalidEnvironment:
        def __init__(self) -> None:
            self.step_index = 0

        def reset(self, *, options=None):
            return np.zeros(1, np.float32), {"valid_actions": (Action.WAIT,)}

        def step(self, action):
            self.step_index += 1
            terminated = self.step_index == 2
            return np.ones(1, np.float32), 0.0, terminated, False, {
                "valid_actions": () if terminated else (Action.WAIT,),
                "outcome": "recovery_success" if terminated else None,
                "equity_pnl": -2_500.0,
                "recovery_entry_used": True,
            }

    with pytest.raises(ValueError, match="one-entry contract"):
        evaluate_recovery_stress(
            Agent(),
            InvalidEnvironment(),
            episodes=1,
            recurrent_horizon=96,
            settings=_recovery_curriculum_settings(fraction=1.0),
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
