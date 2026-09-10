"""Hierarchical binary trade decisions derived from one economic action state."""

from __future__ import annotations

import json
import math


ENTRY = "entry"
DIRECTION = "direction"
MANAGEMENT = "management"


def _task_messages(record: dict, task: str, names: list[str]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if (not isinstance(messages, list) or len(messages) != 3
            or any(not isinstance(message, dict) for message in messages)):
        raise ValueError("hierarchical supervision requires one SFT conversation")
    try:
        prompt = json.loads(messages[-2]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("hierarchical supervision requires a structured causal prompt") from error
    prompt.pop("legal_actions", None)
    prompt.update(decision_task=task, choices=names)
    instructions = {
        ENTRY: (
            "Decide whether a completed-bar setup has sufficient economic edge to ENTER or "
            "should WAIT. Direction is decided separately. Return only one listed choice."
        ),
        DIRECTION: (
            "For an economically valid entry setup, choose LONG or SHORT from completed-bar "
            "evidence. Return only one listed choice."
        ),
        MANAGEMENT: (
            "For the open trade, choose HOLD while continuation remains economically favorable "
            "or CLOSE when it deteriorates. Return only one listed choice."
        ),
    }
    return [
        {"role": "system", "content": instructions[task] + " Future prices are unknown."},
        {"role": "user", "content": json.dumps(
            prompt, separators=(",", ":"), allow_nan=False)},
    ]


def _normalized(values):
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("hierarchical soft targets require positive finite mass")
    return [float(value / total) for value in values]


def _row(record, *, task, names, probabilities, values, target):
    if (target not in names or len(names) != 2 or len(set(names)) != 2
            or len(probabilities) != 2 or len(values) != 2
            or not all(math.isfinite(float(value)) for value in (*probabilities, *values))):
        raise ValueError("invalid hierarchical decision task")
    return {
        "decision_task": task,
        "target_name": target,
        "names": names,
        "probabilities": _normalized(probabilities),
        "values": [float(value) for value in values],
        "messages": _task_messages(record, task, names),
    }


def hierarchical_task_records(record: dict) -> list[dict]:
    """Split one audited full-action label into state-appropriate binary tasks.

    Flat states first learn ENTER versus WAIT.  Direction is supervised only
    when ENTER has strictly greater economic value and one side is uniquely
    better. Positioned states learn HOLD versus CLOSE and never compete with
    entry actions.
    """
    targets = record.get("targets", {})
    names = targets.get("action_order")
    probabilities = targets.get("action_probabilities")
    outcomes = targets.get("outcomes")
    if (not isinstance(names, list) or not isinstance(probabilities, list)
            or len(names) != len(probabilities) or not isinstance(outcomes, dict)
            or set(outcomes) != set(names)):
        raise ValueError("hierarchical supervision requires aligned full-action targets")
    try:
        values = {name: float(outcomes[name]["reward_to_go"]) for name in names}
        masses = {name: float(probabilities[index]) for index, name in enumerate(names)}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("hierarchical supervision requires economic action values") from error
    if not all(math.isfinite(value) and value >= 0 for value in masses.values()):
        raise ValueError("hierarchical action probabilities are invalid")

    flat = {"WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"}
    positioned = {"HOLD", "CLOSE"}
    if set(names) == positioned:
        target = "HOLD" if values["HOLD"] >= values["CLOSE"] else "CLOSE"
        return [_row(record, task=MANAGEMENT, names=["HOLD", "CLOSE"],
                     probabilities=[masses["HOLD"], masses["CLOSE"]],
                     values=[values["HOLD"], values["CLOSE"]], target=target)]
    if set(names) != flat:
        raise ValueError("hierarchical supervision received an unsupported legal-action state")

    long_value, short_value = values["ENTER_LONG_1"], values["ENTER_SHORT_1"]
    enter_value = max(long_value, short_value)
    directional = long_value != short_value
    enter = enter_value > values["WAIT"] and directional
    result = [_row(
        record, task=ENTRY, names=["WAIT", "ENTER"],
        probabilities=[masses["WAIT"], masses["ENTER_LONG_1"] + masses["ENTER_SHORT_1"]],
        values=[values["WAIT"], enter_value], target="ENTER" if enter else "WAIT",
    )]
    if enter:
        direction_probabilities = [masses["ENTER_LONG_1"], masses["ENTER_SHORT_1"]]
        result.append(_row(
            record, task=DIRECTION, names=["LONG", "SHORT"],
            probabilities=direction_probabilities, values=[long_value, short_value],
            target="LONG" if long_value > short_value else "SHORT",
        ))
    return result

