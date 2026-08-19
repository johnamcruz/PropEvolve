import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from propevolve.training_health import (
    TrainingHealthDetector,
    TrainingHealthMonitor,
    TrainingHealthSnapshot,
    TrainingPolicyHealthSpec,
)
from propevolve.config import load_experiment_config
from tests.recipe_fixtures import paired_aplus_recipe


def _spec(
    *,
    require_positive_persistent_regime_association: bool | None = None,
) -> TrainingPolicyHealthSpec:
    association = (
        {}
        if require_positive_persistent_regime_association is None
        else {
            "require_positive_persistent_regime_association": (
                require_positive_persistent_regime_association
            )
        }
    )
    return TrainingPolicyHealthSpec(
        minimum_completed_episodes=45,
        probe_interval_episodes=45,
        minimum_wait_recall=0.35,
        minimum_long_recall=0.30,
        minimum_short_recall=0.30,
        minimum_entry_mass_fraction=0.30,
        maximum_entry_mass_fraction=0.36,
        require_zero_positive_entry_soft_wait_veto=True,
        economic_futility_minimum_completed_episodes=45,
        economic_futility_maximum_near_blow_timeout_rate=0.75,
        economic_futility_maximum_mean_terminal_pnl=-1_500.0,
        economic_futility_maximum_expectancy_r=-0.15,
        economic_futility_minimum_failed_conditions=2,
        **association,
    )


def _snapshot(**overrides: object) -> TrainingHealthSnapshot:
    values: dict[str, object] = {
        "completed_episodes": 45,
        "passes": 5,
        "blows": 0,
        "timeouts": 40,
        "near_blow_timeouts": 10,
        "mean_terminal_pnl": -100.0,
        "expectancy_r": 0.01,
        "optimizer_updates": 32,
        "entry_action_weight_scale": 1.0,
        "teacher_weight_scale": 1.0,
        "entry_mass_fractions": {
            "WAIT": 1.0 / 3.0,
            "ENTER_LONG_1": 1.0 / 3.0,
            "ENTER_SHORT_1": 1.0 / 3.0,
        },
        "positive_entry_soft_wait_disagreement_rows": {
            "ENTER_LONG_1": 0,
            "ENTER_SHORT_1": 0,
        },
        "probe_metrics": {
            "final_regime_probe_wait_rows": 32.0,
            "final_regime_probe_long_rows": 32.0,
            "final_regime_probe_short_rows": 32.0,
            "final_regime_probe_wait_recall": 0.60,
            "final_regime_probe_long_recall": 0.50,
            "final_regime_probe_short_recall": 0.50,
            "final_regime_probe_dead_wait_minus_transition_ready_wait": -0.05,
            "final_regime_probe_dead_wait_minus_transition_positive_wait": 0.05,
            "final_regime_probe_transition_positive_long_response": 0.05,
            "final_regime_probe_transition_positive_short_response": 0.05,
        },
    }
    values.update(overrides)
    return TrainingHealthSnapshot(**values)


def test_policy_health_accepts_noisy_balanced_learning() -> None:
    verdict = TrainingHealthDetector(_spec()).evaluate(_snapshot())

    assert verdict.stop is False
    assert verdict.reasons == ()
    assert verdict.evidence["probe_due"] is True


def test_policy_health_defers_transfer_failure_until_teacher_autonomy() -> None:
    failed_transfer = dict(_snapshot().probe_metrics or {})
    failed_transfer["final_regime_probe_short_recall"] = 0.1875
    failed_transfer["final_regime_probe_transition_positive_short_response"] = (
        -0.073294
    )
    detector = TrainingHealthDetector(
        _spec(require_positive_persistent_regime_association=True)
    )

    guided = detector.evaluate(_snapshot(
        teacher_weight_scale=0.4,
        probe_metrics=failed_transfer,
    ))
    autonomous = detector.evaluate(_snapshot(
        teacher_weight_scale=0.0,
        probe_metrics=failed_transfer,
    ))

    assert guided.stop is False
    assert guided.reasons == ()
    assert guided.evidence["probe_due"] is True
    assert guided.evidence["probe_enforced"] is False
    assert autonomous.stop is True
    assert autonomous.evidence["probe_enforced"] is True
    assert any("Short recall" in reason for reason in autonomous.reasons)


def test_stage2_v7_defers_the_v6_wait_failure_until_episode_100() -> None:
    config = load_experiment_config(
        paired_aplus_recipe(100)
    )
    spec = TrainingPolicyHealthSpec.from_config(
        config["training"]["short_circuit"]["policy_health"]
    )
    receipts: list[dict[str, object]] = []
    v6_failure = dict(_snapshot().probe_metrics or {})
    v6_failure["final_regime_probe_wait_recall"] = 0.15625
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(spec),
        probe=lambda completed: v6_failure,
        receipt_callback=receipts.append,
    )
    diagnostic = {
        "updates": 32,
        "entry_action_weight_scale": 0.5,
        "teacher_weight_scale": 0.5,
        "entry_action_balance": {
            name: {"weighted_mass_fraction": 1.0 / 3.0}
            for name in ("wait", "long", "short")
        },
        "regime_entry_conflict": {
            side: {"soft_wait_disagreement_rows": 0}
            for side in ("long", "short")
        },
    }

    progress_45 = SimpleNamespace(
        completed_episodes=45,
        passes=8,
        blows=1,
        timeouts=36,
        near_blow_timeout_count=16,
        terminal_pnl_sum=-17_807.28,
        terminal_pnl_count=45,
        trade_r_sum=-59.36,
        trade_count=7_377,
    )
    assert monitor(progress_45, diagnostic) is None
    assert receipts[-1]["evidence"]["probe_due"] is False

    progress_100 = SimpleNamespace(
        completed_episodes=100,
        passes=20,
        blows=2,
        timeouts=78,
        near_blow_timeout_count=30,
        terminal_pnl_sum=-20_000.0,
        terminal_pnl_count=100,
        trade_r_sum=-20.0,
        trade_count=10_000,
    )
    diagnostic["teacher_weight_scale"] = 0.0
    assert monitor(progress_100, diagnostic) == (
        "teacher-free policy-health WAIT recall 0.156250 < 0.350000"
    )
    assert receipts[-1]["evidence"]["probe_due"] is True


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    (
        ("final_regime_probe_wait_recall", 0.0, "WAIT recall"),
        ("final_regime_probe_long_recall", 0.0, "Long recall"),
        ("final_regime_probe_short_recall", 0.0, "Short recall"),
    ),
)
def test_policy_health_stops_direction_or_wait_collapse(
    metric: str,
    value: float,
    expected: str,
) -> None:
    probe = dict(_snapshot().probe_metrics or {})
    probe[metric] = value

    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(teacher_weight_scale=0.0, probe_metrics=probe)
    )

    assert verdict.stop is True
    assert any(expected in reason for reason in verdict.reasons)


@pytest.mark.parametrize(
    "metric",
    (
        "final_regime_probe_dead_wait_minus_transition_positive_wait",
        "final_regime_probe_transition_positive_long_response",
        "final_regime_probe_transition_positive_short_response",
    ),
)
@pytest.mark.parametrize("association", (-0.013164, 0.0))
def test_v6_policy_health_stops_nonpositive_persistent_regime_association(
    metric: str,
    association: float,
) -> None:
    probe = dict(_snapshot().probe_metrics or {})
    probe[metric] = association

    verdict = TrainingHealthDetector(_spec(
        require_positive_persistent_regime_association=True,
    )).evaluate(_snapshot(teacher_weight_scale=0.0, probe_metrics=probe))

    assert verdict.stop is True
    assert len(verdict.reasons) == 1
    assert f"{association:.6f} <= 0.000000" in verdict.reasons[0]


def test_v6_policy_health_does_not_gate_transition_ready_wait_contrast() -> None:
    verdict = TrainingHealthDetector(_spec(
        require_positive_persistent_regime_association=True,
    )).evaluate(_snapshot())

    assert verdict.stop is False
    assert verdict.reasons == ()


def test_legacy_policy_health_does_not_retroactively_gate_association() -> None:
    probe = dict(_snapshot().probe_metrics or {})
    probe["final_regime_probe_dead_wait_minus_transition_ready_wait"] = -0.05
    probe["final_regime_probe_dead_wait_minus_transition_positive_wait"] = -0.05
    probe["final_regime_probe_transition_positive_long_response"] = -0.05
    probe["final_regime_probe_transition_positive_short_response"] = -0.05

    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(probe_metrics=probe)
    )

    assert verdict.stop is False
    assert verdict.reasons == ()


def test_policy_health_stops_missing_or_skewed_optimizer_class_mass() -> None:
    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(
            entry_mass_fractions={
                "WAIT": 0.10,
                "ENTER_LONG_1": 0.45,
                "ENTER_SHORT_1": 0.45,
            }
        )
    )

    assert verdict.stop is True
    assert verdict.reasons[0] == (
        "entry optimizer WAIT mass fraction 0.100000 outside [0.300000, 0.360000]"
    )
    assert len(verdict.reasons) == 3


def test_policy_health_stops_any_positive_entry_regime_wait_veto() -> None:
    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(
            positive_entry_soft_wait_disagreement_rows={
                "ENTER_LONG_1": 0,
                "ENTER_SHORT_1": 1,
            }
        )
    )

    assert verdict.stop is True
    assert verdict.reasons == (
        "persistent Regime objective applied a soft-WAIT veto to 1 Short entry rows",
    )


def test_policy_health_requires_two_economic_futility_signals() -> None:
    one_signal = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(mean_terminal_pnl=-1_600.0)
    )
    two_signals = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(
            mean_terminal_pnl=-1_600.0,
            expectancy_r=-0.20,
        )
    )

    assert one_signal.stop is False
    assert two_signals.stop is True
    assert any("economic futility" in reason for reason in two_signals.reasons)


def test_policy_health_does_not_require_probe_before_milestone() -> None:
    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(completed_episodes=44, probe_metrics=None)
    )

    assert verdict.stop is False
    assert verdict.evidence["probe_due"] is False


def test_policy_health_checks_active_entry_objective_before_probe_milestone() -> None:
    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(
            completed_episodes=8,
            entry_mass_fractions={
                "WAIT": 0.10,
                "ENTER_LONG_1": 0.45,
                "ENTER_SHORT_1": 0.45,
            },
            probe_metrics=None,
        )
    )

    assert verdict.stop is True
    assert verdict.evidence["probe_due"] is False
    assert any("WAIT mass fraction" in reason for reason in verdict.reasons)


def test_policy_health_ignores_entry_mass_after_full_autonomy() -> None:
    class Progress:
        completed_episodes = 46
        passes = 5
        blows = 0
        timeouts = 41
        near_blow_timeout_count = 10
        terminal_pnl_sum = -4_600.0
        terminal_pnl_count = 46
        trade_r_sum = 2.0
        trade_count = 200

    receipts = []
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: {},
        receipt_callback=receipts.append,
    )
    reason = monitor(Progress(), {
        "updates": 32,
        "entry_action_weight_scale": 0.0,
        "teacher_weight_scale": 0.0,
        "entry_action_balance": {},
        "regime_entry_conflict": {},
    })

    assert reason is None
    assert receipts[0]["entry_objective_active"] is False


def test_policy_health_checks_entry_mass_but_not_regime_veto_in_consolidation() -> None:
    class Progress:
        completed_episodes = 161
        passes = 20
        blows = 0
        timeouts = 141
        near_blow_timeout_count = 20
        terminal_pnl_sum = 0.0
        terminal_pnl_count = 161
        trade_r_sum = 5.0
        trade_count = 500

    receipts = []
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: {},
        receipt_callback=receipts.append,
    )
    reason = monitor(Progress(), {
        "updates": 32,
        "entry_action_weight_scale": 0.05,
        "teacher_weight_scale": 0.0,
        "entry_action_balance": {
            "wait": {"weighted_mass_fraction": 1 / 3},
            "long": {"weighted_mass_fraction": 1 / 3},
            "short": {"weighted_mass_fraction": 1 / 3},
        },
        "regime_entry_conflict": {},
    })

    assert reason is None
    assert receipts[0]["entry_objective_active"] is True
    assert receipts[0]["regime_objective_active"] is False


def test_policy_health_fails_closed_on_missing_or_nonfinite_milestone_evidence() -> None:
    missing = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(probe_metrics=None)
    )
    nonfinite = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(expectancy_r=math.nan)
    )

    assert missing.stop is True
    assert missing.reasons == ("teacher-free policy-health probe is missing",)
    assert nonfinite.stop is True
    assert any("non-finite" in reason for reason in nonfinite.reasons)


def test_policy_health_requires_exact_balanced_probe_rows() -> None:
    probe = dict(_snapshot().probe_metrics or {})
    probe["final_regime_probe_short_rows"] = 31.0

    verdict = TrainingHealthDetector(_spec()).evaluate(
        _snapshot(probe_metrics=probe)
    )

    assert verdict.stop is True
    assert any("32 authentic Short rows" in reason for reason in verdict.reasons)


def test_policy_health_monitor_runs_fixed_probe_at_milestone_and_receipts_it() -> None:
    class Progress:
        completed_episodes = 45
        passes = 5
        blows = 0
        timeouts = 40
        near_blow_timeout_count = 10
        terminal_pnl_sum = -4_500.0
        terminal_pnl_count = 45
        trade_r_sum = 2.0
        trade_count = 200

    calls = []
    receipts = []
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: calls.append(completed) or dict(
            _snapshot().probe_metrics or {}
        ),
        receipt_callback=receipts.append,
    )
    diagnostic = {
        "updates": 32,
        "entry_action_weight_scale": 1.0,
        "teacher_weight_scale": 1.0,
        "mean_training_loss": 0.5,
        "entry_action_balance": {
            "wait": {"weighted_mass_fraction": 1 / 3},
            "long": {"weighted_mass_fraction": 1 / 3},
            "short": {"weighted_mass_fraction": 1 / 3},
        },
        "regime_entry_conflict": {
            "long": {"soft_wait_disagreement_rows": 0},
            "short": {"soft_wait_disagreement_rows": 0},
        },
    }

    reason = monitor(Progress(), diagnostic)

    assert reason is None
    assert calls == [45]
    assert len(receipts) == 1
    assert receipts[0]["schema"] == "propevolve_training_policy_health_receipt_v1"
    assert receipts[0]["completed_episodes"] == 45
    assert receipts[0]["probe_metrics"]["final_regime_probe_short_recall"] == 0.5
    assert len(receipts[0]["identity_sha256"]) == 64


def test_policy_health_monitor_uses_cumulative_entry_mass_not_one_noisy_episode() -> None:
    class Progress:
        completed_episodes = 18
        passes = 0
        blows = 0
        timeouts = 2
        near_blow_timeout_count = 0
        terminal_pnl_sum = 0.0
        terminal_pnl_count = 2
        trade_r_sum = 0.0
        trade_count = 0

    receipts = []
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: {},
        receipt_callback=receipts.append,
        initial_entry_weighted_masses={
            "WAIT": 32.0,
            "ENTER_LONG_1": 32.0,
            "ENTER_SHORT_1": 32.0,
        },
        minimum_entry_mass_completed_episodes=18,
    )
    diagnostic = {
        "updates": 32,
        "entry_action_weight_scale": 1.0,
        "teacher_weight_scale": 1.0,
        "entry_action_balance": {
            "wait": {"weighted_mass": 8.666656, "weighted_mass_fraction": 0.270833},
            "long": {"weighted_mass": 11.666672, "weighted_mass_fraction": 0.364583},
            "short": {"weighted_mass": 11.666672, "weighted_mass_fraction": 0.364583},
        },
        "regime_entry_conflict": {
            side: {"soft_wait_disagreement_rows": 0}
            for side in ("long", "short")
        },
    }

    assert monitor(Progress(), diagnostic) is None
    assert receipts[-1]["entry_mass_fractions"] == pytest.approx({
        "WAIT": 40.666656 / 128.0,
        "ENTER_LONG_1": 43.666672 / 128.0,
        "ENTER_SHORT_1": 43.666672 / 128.0,
    })


def test_policy_health_monitor_waits_for_stable_entry_mass_evidence() -> None:
    class Progress:
        completed_episodes = 1
        passes = 0
        blows = 0
        timeouts = 1
        near_blow_timeout_count = 0
        terminal_pnl_sum = 0.0
        terminal_pnl_count = 1
        trade_r_sum = 0.0
        trade_count = 0

    receipts = []
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: {},
        receipt_callback=receipts.append,
        minimum_entry_mass_completed_episodes=18,
    )
    diagnostic = {
        "updates": 32,
        "entry_action_weight_scale": 1.0,
        "teacher_weight_scale": 1.0,
        "entry_action_balance": {
            "wait": {"weighted_mass": 3.2, "weighted_mass_fraction": 0.1},
            "long": {"weighted_mass": 14.4, "weighted_mass_fraction": 0.45},
            "short": {"weighted_mass": 14.4, "weighted_mass_fraction": 0.45},
        },
        "regime_entry_conflict": {
            side: {"soft_wait_disagreement_rows": 0}
            for side in ("long", "short")
        },
    }

    assert monitor(Progress(), diagnostic) is None
    assert receipts[-1]["entry_mass_evidence_ready"] is False


def test_policy_health_monitor_still_rejects_cumulative_entry_mass_collapse() -> None:
    class Progress:
        completed_episodes = 1
        passes = 0
        blows = 0
        timeouts = 1
        near_blow_timeout_count = 0
        terminal_pnl_sum = 0.0
        terminal_pnl_count = 1
        trade_r_sum = 0.0
        trade_count = 0

    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: {},
        receipt_callback=lambda payload: None,
    )
    diagnostic = {
        "updates": 32,
        "entry_action_weight_scale": 1.0,
        "teacher_weight_scale": 1.0,
        "entry_action_balance": {
            "wait": {"weighted_mass": 3.2, "weighted_mass_fraction": 0.1},
            "long": {"weighted_mass": 14.4, "weighted_mass_fraction": 0.45},
            "short": {"weighted_mass": 14.4, "weighted_mass_fraction": 0.45},
        },
        "regime_entry_conflict": {
            side: {"soft_wait_disagreement_rows": 0}
            for side in ("long", "short")
        },
    }

    assert monitor(Progress(), diagnostic) == (
        "entry optimizer WAIT mass fraction 0.100000 outside [0.300000, 0.360000]; "
        "entry optimizer ENTER_LONG_1 mass fraction 0.450000 outside [0.300000, 0.360000]; "
        "entry optimizer ENTER_SHORT_1 mass fraction 0.450000 outside [0.300000, 0.360000]"
    )


def test_policy_health_monitor_fails_closed_and_receipts_unavailable_probe() -> None:
    class Progress:
        completed_episodes = 45
        passes = 5
        blows = 0
        timeouts = 40
        near_blow_timeout_count = 10
        terminal_pnl_sum = -4_500.0
        terminal_pnl_count = 45
        trade_r_sum = 2.0
        trade_count = 200

    receipts = []
    monitor = TrainingHealthMonitor(
        TrainingHealthDetector(_spec()),
        probe=lambda completed: (_ for _ in ()).throw(
            ValueError("missing Short rows")
        ),
        receipt_callback=receipts.append,
    )

    reason = monitor(Progress(), {
        "updates": 0,
        "entry_action_weight_scale": 0.0,
        "teacher_weight_scale": 0.0,
        "mean_training_loss": None,
        "entry_action_balance": {},
        "regime_entry_conflict": {},
    })

    assert reason == "teacher-free policy-health probe is missing"
    assert receipts[0]["probe_error"] == "missing Short rows"


@pytest.mark.parametrize(
    "field",
    (
        "minimum_completed_episodes",
        "probe_interval_episodes",
        "economic_futility_minimum_completed_episodes",
        "economic_futility_minimum_failed_conditions",
    ),
)
def test_policy_health_rejects_fractional_count_fields(field: str) -> None:
    values = dict(_spec().__dict__)
    values[field] = 1.5

    with pytest.raises(ValueError, match="counts must be positive integers"):
        TrainingPolicyHealthSpec(**values)
