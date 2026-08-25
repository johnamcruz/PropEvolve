from __future__ import annotations

import pytest

from propevolve.decision import Action
from propevolve.entry_timing_audit import (
    EntryTimingCandidate,
    audit_entry_timing_episode,
    classify_entry_timing_event,
)
from propevolve.entry_supervision import EntryTargetMetadata


def _candidates(*economic_good: bool) -> tuple[EntryTimingCandidate, ...]:
    return tuple(
        EntryTimingCandidate(
            decision_index=index,
            economic_good=good,
        )
        for index, good in enumerate(economic_good, start=10)
    )


@pytest.mark.parametrize(
    ("economic_good", "actions", "expected"),
    (
        (
            (False, True, True, False),
            {10: Action.ENTER_LONG_1},
            "entered_too_early",
        ),
        (
            (False, True, True, False),
            {10: Action.WAIT, 11: Action.ENTER_LONG_1},
            "entered_at_first_valid_bar",
        ),
        (
            (False, True, True, False),
            {10: Action.WAIT, 11: Action.WAIT, 12: Action.ENTER_LONG_1},
            "entered_late",
        ),
        (
            (False, True, True, False),
            {10: Action.WAIT, 11: Action.WAIT, 12: Action.WAIT, 13: Action.WAIT},
            "missed_valid_window",
        ),
        (
            (False, True, False, False),
            {10: Action.WAIT, 11: Action.WAIT, 12: Action.ENTER_LONG_1},
            "entered_after_invalidation",
        ),
        (
            (False, False, False, False),
            {10: Action.WAIT, 11: Action.WAIT, 12: Action.WAIT, 13: Action.WAIT},
            "correctly_waited",
        ),
    ),
)
def test_classifies_complete_long_entry_timing_boundary(
    economic_good: tuple[bool, ...],
    actions: dict[int, Action],
    expected: str,
) -> None:
    result = classify_entry_timing_event(
        side=Action.ENTER_LONG_1,
        candidates=_candidates(*economic_good),
        flat_actions=actions,
    )

    assert result == expected


def test_short_timing_uses_the_same_side_symmetric_contract() -> None:
    result = classify_entry_timing_event(
        side=Action.ENTER_SHORT_1,
        candidates=_candidates(False, True, True),
        flat_actions={10: Action.WAIT, 11: Action.ENTER_SHORT_1},
    )

    assert result == "entered_at_first_valid_bar"


def test_opposite_side_entry_is_not_misreported_as_correct_wait() -> None:
    result = classify_entry_timing_event(
        side=Action.ENTER_LONG_1,
        candidates=_candidates(False, False, False),
        flat_actions={10: Action.ENTER_SHORT_1},
    )

    assert result is None


def test_partial_event_is_not_a_timing_result() -> None:
    result = classify_entry_timing_event(
        side=Action.ENTER_LONG_1,
        candidates=_candidates(False, True, True),
        flat_actions={10: Action.WAIT},
    )

    assert result is None


def _metadata(
    *,
    side: str,
    anchor: int,
    offset: int,
    economic_good: bool,
    candidate_count: int,
) -> EntryTargetMetadata:
    return EntryTargetMetadata(
        side=side,
        event_anchor_rows=(anchor,),
        candidate_decision_offset=offset,
        fill_offset=offset + 1,
        continuation=economic_good,
        economic_win=economic_good,
        economic_good=economic_good,
        available=True,
        censored=False,
        unavailable_reason=None,
        candidate_count=candidate_count,
    )


def test_episode_audit_reports_all_six_outcomes_by_side() -> None:
    definitions = (
        ("long", 10, (False, True, True), (Action.ENTER_LONG_1,), "entered_too_early"),
        ("long", 20, (False, True, True), (Action.WAIT, Action.ENTER_LONG_1), "entered_at_first_valid_bar"),
        ("long", 30, (False, True, True), (Action.WAIT, Action.WAIT, Action.ENTER_LONG_1), "entered_late"),
        ("short", 40, (False, True, True), (Action.WAIT, Action.WAIT, Action.WAIT), "missed_valid_window"),
        ("short", 50, (False, True, False), (Action.WAIT, Action.WAIT, Action.ENTER_SHORT_1), "entered_after_invalidation"),
        ("short", 60, (False, False, False), (Action.WAIT, Action.WAIT, Action.WAIT), "correctly_waited"),
    )
    metadata: dict[int, EntryTargetMetadata] = {}
    flat_actions: dict[int, Action] = {}
    for side, anchor, good, actions, _ in definitions:
        for offset, is_good in enumerate(good):
            row = anchor + offset
            metadata[row] = _metadata(
                side=side,
                anchor=anchor,
                offset=offset,
                economic_good=is_good,
                candidate_count=len(good),
            )
            if offset < len(actions):
                flat_actions[row] = actions[offset]

    report = audit_entry_timing_episode(
        flat_actions=flat_actions,
        metadata_by_decision=metadata,
    )

    assert report["schema"] == "propevolve_entry_timing_audit_v1"
    assert report["classified_events"] == 6
    assert report["unclassified_events"] == 0
    assert report["counts"] == {expected: 1 for *_, expected in definitions}
    assert report["by_side"]["long"] == {
        "entered_too_early": 1,
        "entered_at_first_valid_bar": 1,
        "entered_late": 1,
        "missed_valid_window": 0,
        "entered_after_invalidation": 0,
        "correctly_waited": 0,
    }
    assert report["by_side"]["short"] == {
        "entered_too_early": 0,
        "entered_at_first_valid_bar": 0,
        "entered_late": 0,
        "missed_valid_window": 1,
        "entered_after_invalidation": 1,
        "correctly_waited": 1,
    }
