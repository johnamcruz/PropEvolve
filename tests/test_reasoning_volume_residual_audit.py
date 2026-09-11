from __future__ import annotations

import numpy as np


def test_volume_residual_audit_requires_both_raw_and_matched_separation():
    from propevolve.reasoning_policy.volume_residual_audit import assess_rows

    rows = []
    # Identical E/R/T controls per ticker make the matched comparison exact.
    for action, correct, volume in (
        ("ENTER_SHORT_1", True, [0.1, 0.1, 0.9, 0.9]),
        ("ENTER_SHORT_1", False, [0.1, 0.1, 0.2, 0.2]),
        ("WAIT", True, [0.1, 0.1, 0.1, 0.1]),
        ("WAIT", False, [0.8, 0.8, 0.1, 0.1]),
    ):
        rows.append({
            "ticker": "NQ", "target": action, "correct": correct,
            "controls": np.full(11, 0.5), "volume": np.asarray(volume),
        })

    result = assess_rows(
        rows,
        actions={"ENTER_SHORT_1": "short_edge", "WAIT": "quietness"},
        minimum_auc=0.75,
        minimum_matched_rate=0.75,
    )

    assert result["status"] == "PASS"
    assert result["actions"]["ENTER_SHORT_1"]["auc_correct"] == 1.0
    assert result["actions"]["WAIT"]["matched_correct_higher_rate"] == 1.0


def test_one_unseparated_required_action_fails_volume_promotion():
    from propevolve.reasoning_policy.volume_residual_audit import assess_rows

    rows = []
    for action in ("ENTER_SHORT_1", "WAIT"):
        for correct in (False, True):
            rows.append({
                "ticker": "NQ", "target": action, "correct": correct,
                "controls": np.full(11, 0.5),
                "volume": np.full(4, 0.5),
            })

    result = assess_rows(
        rows,
        actions={"ENTER_SHORT_1": "short_edge", "WAIT": "quietness"},
        minimum_auc=0.55,
        minimum_matched_rate=0.55,
    )

    assert result["status"] == "REJECTED"
    assert set(result["failed_actions"]) == {"ENTER_SHORT_1", "WAIT"}
