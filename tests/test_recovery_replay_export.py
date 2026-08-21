from __future__ import annotations

import numpy as np

from scripts.export_recovery_success_replay import (
    _is_successful_recovery_pass,
    _with_recovery_state,
)


def test_legacy_v22_pass_derives_recovery_boundary_from_causal_pnl_channel() -> None:
    observations = np.zeros((5, 20), np.float32)
    observations[:, 2] = np.array([-1.0 / 3.0, -0.1, 0.0, 0.5, 1.0])
    episode = {
        "outcome": "pass",
        "actions": np.zeros(4, np.int8),
        "observations": observations,
    }

    restored = _with_recovery_state(episode)

    assert restored["recovery_active"].tolist() == [True, True, False, False]
    assert _is_successful_recovery_pass(restored) is True


def test_failure_or_pass_without_breakeven_is_not_recovery_competence() -> None:
    assert _is_successful_recovery_pass({
        "outcome": "blow",
        "recovery_active": np.array([True, False]),
    }) is False
    assert _is_successful_recovery_pass({
        "outcome": "pass",
        "recovery_active": np.array([True, True]),
    }) is False
