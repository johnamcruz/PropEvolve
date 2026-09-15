"""Decision evidence measures the independent assessments, not joint argmax."""
import pytest


def test_wrong_direction_does_not_turn_a_correct_entry_into_an_entry_error():
    from propevolve.reasoning_policy.staged_metrics import boundary_metrics
    row = {"target_name": "ENTER_LONG_1", "action_targets": {
        "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
        "probabilities": [0., 1., 0.], "values": [0., 2., -1.]}}
    metrics = boundary_metrics([row], [[.2, -4., 99.]], margin=.25)
    assert metrics["per_task"]["entry.ENTER"]["accuracy"] == 1.
    assert metrics["per_task"]["entry.ENTER"]["mean_target_advantage"] == pytest.approx(.2)
    assert metrics["per_task"]["direction.LONG"]["accuracy"] == 0.
    assert metrics["per_action"]["ENTER_LONG_1"]["accuracy"] == 0.
    assert set(metrics["per_task"]) == {"entry.ENTER", "direction.LONG"}


def test_frozen_boundary_evidence_uses_independent_assessments_not_action_scores():
    from propevolve.reasoning_policy.staged_metrics import assessment_advantages
    row = {"target": "ENTER_LONG_1", "score_type": "log_probability",
        "assessment": {"entry": .2, "direction": -.1, "management": 99.},
        "scores": {"WAIT": -10., "ENTER_LONG_1": -20., "ENTER_SHORT_1": 0.}}
    assert assessment_advantages(row) == {"entry.ENTER": .2, "direction.LONG": -.1}
