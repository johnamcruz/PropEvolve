"""Training-only audit of Stage 1 Expansion entry timing behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .decision import Action
from .entry_supervision import EntryTargetMetadata


EntryTimingOutcome = Literal[
    "entered_too_early",
    "entered_at_first_valid_bar",
    "entered_late",
    "missed_valid_window",
    "entered_after_invalidation",
    "correctly_waited",
]

ENTRY_TIMING_OUTCOMES: tuple[EntryTimingOutcome, ...] = (
    "entered_too_early",
    "entered_at_first_valid_bar",
    "entered_late",
    "missed_valid_window",
    "entered_after_invalidation",
    "correctly_waited",
)


@dataclass(frozen=True, slots=True)
class EntryTimingCandidate:
    """One causal decision row and its training-only economic outcome."""

    decision_index: int
    economic_good: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
            or type(self.economic_good) is not bool
        ):
            raise ValueError("entry timing candidate is invalid")


def classify_entry_timing_event(
    *,
    side: Action,
    candidates: Sequence[EntryTimingCandidate],
    flat_actions: Mapping[int, Action],
) -> EntryTimingOutcome | None:
    """Classify one complete Stage 1 event without changing its targets.

    Entry outcomes require the complete observed flat prefix through the first
    same-side entry. WAIT outcomes require all candidate decisions. Missing
    decisions and opposite-side entries remain unclassified rather than being
    misreported as correct timing.
    """

    try:
        side = Action(side)
    except (TypeError, ValueError) as error:
        raise ValueError("entry timing side is invalid") from error
    if side not in {Action.ENTER_LONG_1, Action.ENTER_SHORT_1}:
        raise ValueError("entry timing side must be Long or Short")
    normalized = tuple(candidates)
    if (
        not normalized
        or tuple(item.decision_index for item in normalized)
        != tuple(sorted({item.decision_index for item in normalized}))
    ):
        raise ValueError("entry timing candidates must be ordered and unique")
    candidate_rows = {item.decision_index for item in normalized}
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index not in candidate_rows
        for index in flat_actions
    ):
        raise ValueError("entry timing action row is outside the event")
    try:
        actions = {index: Action(action) for index, action in flat_actions.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("entry timing action is invalid") from error
    flat_action_set = {
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    }
    if any(action not in flat_action_set for action in actions.values()):
        raise ValueError("entry timing audit requires flat actions")

    first_good_position = next(
        (index for index, item in enumerate(normalized) if item.economic_good),
        None,
    )
    first_side_entry_position = next(
        (
            index
            for index, item in enumerate(normalized)
            if actions.get(item.decision_index) == side
        ),
        None,
    )
    if first_side_entry_position is not None:
        prefix = normalized[: first_side_entry_position + 1]
        if any(item.decision_index not in actions for item in prefix):
            return None
        if any(
            actions[item.decision_index]
            not in {Action.WAIT, side}
            for item in prefix
        ):
            return None
        if first_good_position is None:
            return "entered_after_invalidation"
        if first_side_entry_position < first_good_position:
            return "entered_too_early"
        if first_side_entry_position == first_good_position:
            return "entered_at_first_valid_bar"
        if normalized[first_side_entry_position].economic_good:
            return "entered_late"
        return "entered_after_invalidation"

    if any(item.decision_index not in actions for item in normalized):
        return None
    if any(action != Action.WAIT for action in actions.values()):
        return None
    if first_good_position is not None:
        return "missed_valid_window"
    return "correctly_waited"


def audit_entry_timing_episode(
    *,
    flat_actions: Mapping[int, Action],
    metadata_by_decision: Mapping[int, EntryTargetMetadata],
) -> dict[str, object]:
    """Aggregate complete resolved Stage 1 events for one scored episode."""

    grouped: dict[tuple[str, int], list[tuple[int, EntryTargetMetadata]]] = {}
    for decision_index, metadata in metadata_by_decision.items():
        if (
            isinstance(decision_index, bool)
            or not isinstance(decision_index, int)
            or metadata.side not in {"long", "short"}
            or len(metadata.event_anchor_rows) != 1
            or metadata.candidate_decision_offset is None
            or type(metadata.economic_good) is not bool
            or metadata.candidate_count is None
        ):
            continue
        grouped.setdefault(
            (metadata.side, metadata.event_anchor_rows[0]), []
        ).append((decision_index, metadata))

    counts = {outcome: 0 for outcome in ENTRY_TIMING_OUTCOMES}
    by_side = {
        side: {outcome: 0 for outcome in ENTRY_TIMING_OUTCOMES}
        for side in ("long", "short")
    }
    classified = 0
    unclassified = 0
    for (side_name, _), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item[1].candidate_decision_offset)
        expected_counts = {item.candidate_count for _, item in rows}
        offsets = tuple(item.candidate_decision_offset for _, item in rows)
        if (
            len(expected_counts) != 1
            or None in expected_counts
            or len(rows) != next(iter(expected_counts))
            or offsets != tuple(range(len(rows)))
        ):
            unclassified += 1
            continue
        candidates = tuple(
            EntryTimingCandidate(
                decision_index=decision_index,
                economic_good=bool(metadata.economic_good),
            )
            for decision_index, metadata in rows
        )
        event_rows = {item.decision_index for item in candidates}
        outcome = classify_entry_timing_event(
            side=(
                Action.ENTER_LONG_1
                if side_name == "long"
                else Action.ENTER_SHORT_1
            ),
            candidates=candidates,
            flat_actions={
                index: action
                for index, action in flat_actions.items()
                if index in event_rows
            },
        )
        if outcome is None:
            unclassified += 1
            continue
        classified += 1
        counts[outcome] += 1
        by_side[side_name][outcome] += 1
    return {
        "schema": "propevolve_entry_timing_audit_v1",
        "classified_events": classified,
        "unclassified_events": unclassified,
        "counts": counts,
        "by_side": by_side,
    }


__all__ = [
    "ENTRY_TIMING_OUTCOMES",
    "EntryTimingCandidate",
    "EntryTimingOutcome",
    "audit_entry_timing_episode",
    "classify_entry_timing_event",
]
