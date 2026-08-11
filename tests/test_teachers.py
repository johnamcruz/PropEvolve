from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propevolve.teachers import CombinedTeacherTargets, TeacherTargetSource


@dataclass(frozen=True)
class _Targets:
    value: np.ndarray | None

    def target(self, ticker: str, row: int) -> np.ndarray | None:
        assert ticker == "NQ"
        assert row == 7
        return self.value


def test_combined_teacher_targets_preserve_declared_channel_order() -> None:
    targets = CombinedTeacherTargets((
        TeacherTargetSource("expansion", ("long", "short"), _Targets(
            np.array([0.8, 0.2], np.float32)
        ), loss_weight=0.2, entry_search_loss_weight=0.3),
        TeacherTargetSource("regime", ("trend", "chop"), _Targets(
            np.array([0.7, 0.3], np.float32)
        ), loss_weight=0.1, entry_search_loss_weight=0.0),
    ))

    np.testing.assert_array_equal(
        targets.target("NQ", 7),
        np.array([0.8, 0.2, 0.7, 0.3], np.float32),
    )
    assert targets.channels == ("long", "short", "trend", "chop")
    assert targets.channel_loss_weights == (0.1, 0.1, 0.05, 0.05)
    assert targets.entry_search_loss_weight == 0.3


def test_combined_teacher_target_is_unavailable_if_one_teacher_is_unavailable() -> None:
    targets = CombinedTeacherTargets((
        TeacherTargetSource("expansion", ("a",), _Targets(
            np.array([0.5], np.float32)
        ), loss_weight=0.2, entry_search_loss_weight=0.0),
        TeacherTargetSource("regime", ("b",), _Targets(None),
                            loss_weight=0.1, entry_search_loss_weight=0.0),
    ))

    assert targets.target("NQ", 7) is None
