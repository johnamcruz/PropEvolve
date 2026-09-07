import pytest

from propevolve.reasoning_policy.supervised_trainer import (
    action_boundary_metrics,
    EarlyStopTraining,
    PostUpdateValidation,
    ValidationLossGuard,
    balanced_validation_order,
    validate_balanced_optimizer_windows,
    validate_early_stopping_coverage,
)


def test_validation_guard_stops_after_patience_and_preserves_best_iteration():
    snapshots = []
    guard = ValidationLossGuard(
        {"enabled": True, "patience_evaluations": 2, "min_delta": 0.1,
         "restore_best": True, "monitor": "val_loss", "mode": "min"},
        on_improvement=lambda report: snapshots.append(dict(report)),
    )

    guard.on_val_loss_report({"iteration": 0, "val_loss": 4.0})
    guard.on_val_loss_report({"iteration": 16, "val_loss": 2.0})
    guard.on_val_loss_report({"iteration": 32, "val_loss": 1.95})
    with pytest.raises(EarlyStopTraining):
        guard.on_val_loss_report({"iteration": 48, "val_loss": 2.4})

    assert [item["iteration"] for item in snapshots] == [0, 16]
    assert guard.summary() == {
        "best_iteration": 16,
        "best_validation_loss": 2.0,
        "best_metric": 2.0,
        "monitor": "val_loss",
        "mode": "min",
        "best_report": {"iteration": 16, "val_loss": 2.0},
        "evaluations": 4,
        "stopped_early": True,
        "stop_iteration": 48,
        "history": [
            {"iteration": 0, "validation_loss": 4.0, "checkpoint_selected": True},
            {"iteration": 16, "validation_loss": 2.0, "checkpoint_selected": True},
            {"iteration": 32, "validation_loss": 1.95, "checkpoint_selected": False},
            {"iteration": 48, "validation_loss": 2.4, "checkpoint_selected": False},
        ],
    }


def test_validation_guard_does_not_stop_or_snapshot_when_disabled():
    snapshots = []
    guard = ValidationLossGuard(
        {"enabled": False, "patience_evaluations": 1, "min_delta": 0.0,
         "restore_best": False, "monitor": "val_loss", "mode": "min"},
        on_improvement=lambda report: snapshots.append(report),
    )
    guard.on_val_loss_report({"iteration": 0, "val_loss": 1.0})
    guard.on_val_loss_report({"iteration": 5, "val_loss": 2.0})
    assert snapshots == []
    assert guard.summary()["stopped_early"] is False


@pytest.mark.parametrize("settings", [
    {},
    {"enabled": True, "patience_evaluations": 0, "min_delta": 0.0,
     "restore_best": True},
    {"enabled": True, "patience_evaluations": 1, "min_delta": -0.1,
     "restore_best": True},
])
def test_validation_guard_rejects_invalid_settings(settings):
    with pytest.raises(ValueError, match="early stopping"):
        ValidationLossGuard(settings, on_improvement=lambda report: None)


def test_validation_order_is_fixed_balanced_and_uses_each_row_once():
    rows = ([{"target_name": "WAIT", "id": index} for index in range(5)]
            + [{"target_name": "ENTER_LONG_1", "id": 10 + index} for index in range(3)]
            + [{"target_name": "ENTER_SHORT_1", "id": 20 + index} for index in range(4)])
    order = balanced_validation_order(rows, rng=__import__("numpy").random.default_rng(9))
    labels = [rows[index]["target_name"] for index in order[:9]]
    assert set(order) == set(range(len(rows)))
    assert len(order) == len(set(order)) == len(rows)
    assert labels.count("WAIT") == labels.count("ENTER_LONG_1") == labels.count("ENTER_SHORT_1") == 3


def test_early_stopping_requires_complete_matching_validation_evidence():
    config = {"batch_size": 1, "val_batches": 2,
              "early_stopping": {"enabled": True},
              "action_supervision": {"enabled": True}}
    train = [{"target_name": name} for name in ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1")]
    valid = list(train)
    with pytest.raises(ValueError, match="complete validation"):
        validate_early_stopping_coverage(config, {"train": train, "valid": valid})
    config["val_batches"] = 3
    validate_early_stopping_coverage(config, {"train": train, "valid": valid})
    with pytest.raises(ValueError, match="action classes"):
        validate_early_stopping_coverage(config, {"train": train, "valid": valid[:-1]})


def test_validation_runs_only_after_reported_optimizer_updates():
    losses = iter([4.0, 3.0, 5.0])
    snapshots = []
    guard = ValidationLossGuard(
        {"enabled": True, "patience_evaluations": 1, "min_delta": 0.0,
         "restore_best": True, "monitor": "val_loss", "mode": "min"},
        on_improvement=lambda report: snapshots.append(report["iteration"]),
    )
    callback = PostUpdateValidation(
        guard, every=2, total_iterations=6,
        evaluate_loss=lambda: next(losses), progress=lambda message: None,
    )
    callback.evaluate(0)
    callback.on_train_loss_report({"iteration": 1})
    callback.on_train_loss_report({"iteration": 2})
    with pytest.raises(EarlyStopTraining):
        callback.on_train_loss_report({"iteration": 4})
    assert snapshots == [0, 2]
    assert [row["iteration"] for row in guard.summary()["history"]] == [0, 2, 4]


def test_post_update_validation_exposes_pruner_metrics_at_same_boundary():
    recorded = []
    guard = ValidationLossGuard(
        {"enabled": True, "patience_evaluations": 3, "min_delta": 0.0,
         "restore_best": True, "monitor": "worst_action_advantage", "mode": "max"},
        on_improvement=lambda report: None,
    )
    callback = PostUpdateValidation(
        guard, every=2, total_iterations=4,
        evaluate_loss=lambda: {"val_loss": 1.0, "worst_action_advantage": 0.3,
                               "macro_accuracy": 1.0, "per_action": {}},
        progress=lambda message: None, record_validation=recorded.append,
    )

    callback.evaluate(0)
    callback.on_train_loss_report({"iteration": 2})

    assert [row["iteration"] for row in recorded] == [0, 2]
    assert all(row["worst_action_advantage"] == 0.3 for row in recorded)


def test_balanced_action_optimizer_window_contains_every_action_equally():
    rows = ([{"target_name": "WAIT"}] * 5
            + [{"target_name": "ENTER_LONG_1"}] * 4
            + [{"target_name": "ENTER_SHORT_1"}] * 3)
    config = {"batch_sampling": "balanced_actions", "grad_accumulation_steps": 8,
              "batch_size": 1, "action_supervision": {"enabled": True}}
    with pytest.raises(ValueError, match="optimizer window"):
        validate_balanced_optimizer_windows(config, rows)
    config["grad_accumulation_steps"] = 3
    validate_balanced_optimizer_windows(config, rows)


def test_action_checkpoint_metric_cannot_hide_long_or_short_collapse():
    rows = [
        {"target_name": "WAIT", "action_targets": {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]}},
        {"target_name": "ENTER_LONG_1", "action_targets": {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]}},
        {"target_name": "ENTER_SHORT_1", "action_targets": {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]}},
    ]
    collapsed = action_boundary_metrics(rows, [
        [1.0, 0.0, 0.0],
        [1.0, 0.9, 0.0],
        [1.0, 0.0, 0.9],
    ], margin=0.25)
    mastered = action_boundary_metrics(rows, [
        [0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.0, 0.4],
    ], margin=0.25)
    assert collapsed["macro_accuracy"] == pytest.approx(1 / 3)
    assert collapsed["worst_action_advantage"] == pytest.approx(-0.1)
    assert mastered["macro_accuracy"] == 1.0
    assert mastered["worst_action_advantage"] == pytest.approx(0.4)
    assert mastered["worst_action_boundary_loss"] < collapsed["worst_action_boundary_loss"]


def test_action_boundary_loss_does_not_average_away_bad_rows():
    rows = [
        {"target_name": "WAIT", "action_targets": {"names": ["WAIT", "ENTER_LONG_1"]}},
        {"target_name": "WAIT", "action_targets": {"names": ["WAIT", "ENTER_LONG_1"]}},
    ]
    canceling = action_boundary_metrics(rows, [[1.0, 0.0], [0.0, 1.0]], margin=0.25)
    indifferent = action_boundary_metrics(rows, [[0.0, 0.0], [0.0, 0.0]], margin=0.25)
    assert canceling["per_action"]["WAIT"]["mean_target_advantage"] == pytest.approx(0.0)
    assert indifferent["per_action"]["WAIT"]["mean_target_advantage"] == pytest.approx(0.0)
    assert canceling["worst_action_boundary_loss"] > indifferent["worst_action_boundary_loss"]


def test_action_guard_selects_worst_class_margin_not_lower_average_loss():
    snapshots = []
    guard = ValidationLossGuard(
        {"enabled": True, "patience_evaluations": 1, "min_delta": 0.0,
         "restore_best": True, "monitor": "worst_action_advantage", "mode": "max"},
        on_improvement=lambda report: snapshots.append(report["iteration"]),
    )
    guard.on_val_loss_report({"iteration": 0, "val_loss": 3.0,
                              "worst_action_advantage": -2.0})
    guard.on_val_loss_report({"iteration": 8, "val_loss": 2.0,
                              "worst_action_advantage": -0.5})
    with pytest.raises(EarlyStopTraining):
        guard.on_val_loss_report({"iteration": 16, "val_loss": 1.0,
                                  "worst_action_advantage": -0.8})
    assert snapshots == [0, 8]
    assert guard.summary()["best_metric"] == pytest.approx(-0.5)
