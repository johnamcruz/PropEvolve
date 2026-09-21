"""Checkpoint selection must not reward a collapsed constant classifier.

The expansion-flow run of 2026-09-20 shipped its worst checkpoint. Its monitor,
``worst_task_advantage``, is a margin statistic: it rises whenever logits shrink
toward zero, so a policy that answers the same thing for every row climbed it
monotonically (-1.5588 -> -0.5586) while balanced accuracy sat at chance and
validation loss got steadily worse. These tests pin the properties that make a
monitor safe for that failure, using the shape the trainer really consumes.
"""
from __future__ import annotations

import pytest

from propevolve.reasoning_policy.mlx_sft import read_sft_config
from propevolve.reasoning_policy.supervised_trainer import (
    EarlyStopTraining,
    ValidationLossGuard,
    hierarchical_boundary_metrics,
)

ENTRY_NAMES = ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
MANAGE_NAMES = ["HOLD", "CLOSE"]


def _entry_row(target):
    """A flat-decision row whose economic values make ``target`` the right answer."""
    values = {"WAIT": [1.0, 0.0, 0.0],
              "ENTER_LONG_1": [0.0, 1.0, -1.0],
              "ENTER_SHORT_1": [0.0, -1.0, 1.0]}[target]
    return {"target_name": target, "action_targets": {"names": ENTRY_NAMES, "values": values}}


def _manage_row(target):
    values = {"HOLD": [1.0, 0.0], "CLOSE": [0.0, 1.0]}[target]
    return {"target_name": target, "action_targets": {"names": MANAGE_NAMES, "values": values}}


def _collapsed(rows, *, confidence):
    """Scores for a policy that always answers ENTER/LONG/HOLD, at a given confidence.

    Shrinking ``confidence`` models exactly what the real run did between evals:
    the same constant decision, held less strongly.
    """
    scores = []
    for row in rows:
        names = row["action_targets"]["names"]
        if names == ENTRY_NAMES:
            scores.append([-confidence, confidence, -confidence])
        else:
            scores.append([confidence, -confidence])
    return scores


@pytest.fixture
def balanced_rows():
    """Equal evidence on both sides of every binary decision, as balanced sampling gives."""
    return ([_entry_row("WAIT")] * 20 + [_entry_row("ENTER_LONG_1")] * 20
            + [_entry_row("ENTER_SHORT_1")] * 20
            + [_manage_row("HOLD")] * 20 + [_manage_row("CLOSE")] * 20)


def test_worst_task_advantage_rewards_a_collapsed_policy(balanced_rows):
    """Document the defect: the old monitor improves as a constant policy decays."""
    confident = hierarchical_boundary_metrics(balanced_rows, _collapsed(balanced_rows, confidence=2.0))
    decayed = hierarchical_boundary_metrics(balanced_rows, _collapsed(balanced_rows, confidence=0.5))
    assert decayed["worst_task_advantage"] > confident["worst_task_advantage"]


def test_balanced_accuracy_is_chance_for_a_collapsed_policy(balanced_rows):
    """The fix: a constant classifier scores exactly chance, at any confidence."""
    for confidence in (2.0, 0.5, 0.01):
        metrics = hierarchical_boundary_metrics(balanced_rows, _collapsed(balanced_rows, confidence=confidence))
        assert metrics["task_macro_accuracy"] == pytest.approx(0.5, abs=1e-9)


def test_balanced_accuracy_does_not_move_when_the_constant_flips(balanced_rows):
    """Rotating the bias (always-LONG -> always-SHORT) is not progress."""
    long_biased = hierarchical_boundary_metrics(balanced_rows, _collapsed(balanced_rows, confidence=1.0))
    flipped = []
    for row in balanced_rows:
        names = row["action_targets"]["names"]
        flipped.append([-1.0, -1.0, 1.0] if names == ENTRY_NAMES else [-1.0, 1.0])
    short_biased = hierarchical_boundary_metrics(balanced_rows, flipped)
    assert long_biased["task_macro_accuracy"] == pytest.approx(short_biased["task_macro_accuracy"])


def test_worst_task_accuracy_is_reported_and_zero_when_collapsed(balanced_rows):
    """The min-over-tasks accuracy diagnostic exists and names the collapse."""
    metrics = hierarchical_boundary_metrics(balanced_rows, _collapsed(balanced_rows, confidence=1.0))
    assert metrics["worst_task_accuracy"] == pytest.approx(0.0)


def test_a_conditional_policy_beats_chance(balanced_rows):
    """A policy that actually reads the row scores above chance, so the monitor can rise."""
    scores = []
    for row in balanced_rows:
        names = row["action_targets"]["names"]
        index = names.index(row["target_name"])
        scores.append([1.0 if i == index else -1.0 for i in range(len(names))])
    metrics = hierarchical_boundary_metrics(balanced_rows, scores)
    assert metrics["task_macro_accuracy"] > 0.9
    assert metrics["worst_task_accuracy"] > 0.9


def test_balanced_action_sft_admits_the_accuracy_monitor(tmp_path):
    """The config guard must accept balanced accuracy for balanced-action SFT."""
    config = read_sft_config("config/reasoning/expansion_flow_sft.json", root=".")
    assert config["early_stopping"]["monitor"] == "task_macro_accuracy"
    assert config["early_stopping"]["mode"] == "max"
    assert config["early_stopping"]["min_delta"] > 0


def test_guard_stops_a_run_that_never_leaves_chance():
    """A flat-at-chance trajectory must exhaust patience instead of shipping a checkpoint."""
    settings = {"enabled": True, "patience_evaluations": 3, "min_delta": 0.005,
                "restore_best": True, "monitor": "task_macro_accuracy", "mode": "max"}
    guard = ValidationLossGuard(settings, on_improvement=lambda report: None)
    trajectory = [(0, 2.3095, 0.4998), (10, 2.3022, 0.4988), (20, 2.3624, 0.5029),
                  (30, 2.4745, 0.4933), (40, 2.5783, 0.5001), (50, 2.5976, 0.4996)]
    with pytest.raises(EarlyStopTraining):
        for iteration, loss, macro in trajectory:
            guard.on_val_loss_report({"val_loss": loss, "iteration": iteration,
                                      "task_macro_accuracy": macro})
    summary = guard.summary()
    # Never leaving chance means the untrained baseline is kept, not the decayed tail.
    assert summary["best_iteration"] == 0
    assert summary["best_report"]["val_loss"] == pytest.approx(2.3095)


def test_guard_keeps_training_while_accuracy_really_improves():
    """A genuinely learning run must not be cut short by the new min_delta."""
    settings = {"enabled": True, "patience_evaluations": 3, "min_delta": 0.005,
                "restore_best": True, "monitor": "task_macro_accuracy", "mode": "max"}
    guard = ValidationLossGuard(settings, on_improvement=lambda report: None)
    for iteration, macro in enumerate([0.50, 0.53, 0.56, 0.60, 0.65]):
        guard.on_val_loss_report({"val_loss": 2.0, "iteration": iteration * 10,
                                  "task_macro_accuracy": macro})
    assert not guard.stopped_early
    assert guard.summary()["best_iteration"] == 40
