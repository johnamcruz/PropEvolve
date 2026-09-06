"""Direct supervised examples; future targets never enter the model prompt."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..decision import Action
from .context import ContextWindow
from .labels import ActionLabels
from .integrity import file_digest


def context_messages(context: ContextWindow, legal_actions) -> list[dict[str, str]]:
    """One shared, compact serializer for SFT and inference.

Only low-dimensional named specialist/account signals belong here; do not
serialize thousands of FFM latent coordinates as decimal tokens.
    """
    if not context.available.any():
        raise ValueError("context has no completed observations")
    values = context.values[context.available]
    if not np.isfinite(values).all():
        raise ValueError("context contains nonfinite data")
    actions = tuple(Action(a).name for a in legal_actions)
    if not actions:
        raise ValueError("no legal action")
    return [
        {"role": "system", "content": (
            "Choose one legal trading action from completed-bar evidence and account state. "
            "The objective is to pass the challenge without breaching its effective trailing "
            "MLL. Return only the action name. Future prices are unknown."
        )},
        {"role": "user", "content": json.dumps({
            "fields": context.fields, "history_oldest_first": values.tolist(),
            "legal_actions": actions,
        }, separators=(",", ":"), allow_nan=False)},
    ]


def supervised_record(
    context: ContextWindow, labels: ActionLabels, *, source_id: str,
    continuation_id: str, target_temperature: float,
) -> dict:
    """Retain all alternatives plus an action completion for MLX-LM SFT.

Reward softmax values are preference targets, NOT calibrated pass probabilities.
Exact value ties choose WAIT/HOLD when legal, never an arbitrary direction.
    """
    if (not source_id or not continuation_id or not np.isfinite(target_temperature)
            or target_temperature <= 0):
        raise ValueError("source identity, continuation and temperature are required")
    actions = tuple(sorted(labels.outcomes, key=int))
    values = np.asarray([labels.outcomes[a].reward_to_go for a in actions], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid action values")
    probabilities = np.exp((values - values.max()) / target_temperature)
    probabilities /= probabilities.sum()
    best = [a for a, value in zip(actions, values) if value == values.max()]
    chosen = next((a for a in (Action.WAIT, Action.HOLD) if a in best), best[0])
    return {
        "schema": "propevolve_reasoning_supervision_v1", "source_id": source_id,
        "continuation_id": continuation_id,
        "completed_at_ns": context.timestamps[-1],
        "label_end_ns": max(value.outcome_end_ns for value in labels.outcomes.values()),
        "messages": context_messages(context, actions) + [
            {"role": "assistant", "content": chosen.name},
        ],
        "targets": {
            "action_order": [a.name for a in actions],
            "action_probabilities": probabilities.tolist(),
            "outcomes": {a.name: asdict(labels.outcomes[a]) for a in actions},
        },
    }


def market_supervised_record(context: ContextWindow, *, opportunity: tuple[bool, bool],
                             source_id: str, label_end_ns: int, economic_contract: dict) -> dict:
    """An optional market-understanding SFT phase before action SFT.

Two observed binary outcomes are targets, not certain ex-ante probabilities.
Both may be false or true; never infer one side by negating the other.
    """
    if len(opportunity) != 2 or any(type(value) is not bool for value in opportunity):
        raise ValueError("market supervision requires two uncensored economic labels")
    if not source_id or label_end_ns <= context.timestamps[-1]:
        raise ValueError("invalid market label identity or horizon")
    messages = context_messages(context, (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1))
    payload = json.loads(messages[1]["content"])
    payload.pop("legal_actions")
    payload["economic_contract"] = economic_contract
    messages[1]["content"] = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    messages[0]["content"] = (
        "From completed-bar evidence, estimate whether Long and Short each reach "
        "the configured net profit barrier before the adverse barrier within the "
        "declared horizon. Return the two labeled outcomes as JSON. Future prices are unknown."
    )
    completion = {"long_target_before_stop": opportunity[0], "short_target_before_stop": opportunity[1]}
    return {
        "schema": "propevolve_reasoning_supervision_v1", "source_id": source_id,
        "completed_at_ns": context.timestamps[-1], "label_end_ns": label_end_ns,
        "messages": messages + [{"role": "assistant", "content": json.dumps(completion)}],
        "targets": completion,
    }


def write_supervised_dataset(records, output: str | Path, *, splits: dict, lineage: dict,
                             sealed_start_ns: int):
    """Publish a bounded dataset with disjoint chronological label reserves.

All timestamps are completed-bar UTC nanoseconds. The caller provides audited
lineage; this writer checks boundaries, not the truth of a provenance assertion.
Incomplete/overlapping rows fail instead of silently becoming WAIT examples.
    """
    import os
    import shutil
    import tempfile

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"dataset already exists: {output}")
    if set(splits) != {"train", "valid"}:
        raise ValueError("dataset requires train and valid chronological roles")
    bounds = {key: tuple(int(x) for x in value) for key, value in splits.items()}
    if (any(len(x) != 2 or x[0] >= x[1] for x in bounds.values())
            or bounds["train"][1] > bounds["valid"][0]):
        raise ValueError("invalid or overlapping temporal roles")
    if type(sealed_start_ns) is not int or any(upper > sealed_start_ns for _, upper in bounds.values()):
        raise ValueError("development data crosses sealed final-validation boundary")
    if not lineage or not all(lineage.get(key) for key in (
        "source_identity", "specialist_identities", "economic_contract", "split_audit",
    )):
        raise ValueError("audited source, specialist, economic and split lineage required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".reasoning-dataset-", dir=output.parent))
    counts = {key: 0 for key in bounds}
    seen = set()
    try:
        with (temporary / "train.jsonl").open("x") as train, (temporary / "valid.jsonl").open("x") as valid:
            handles = {"train": train, "valid": valid}
            for record in records:
                start, end = int(record["completed_at_ns"]), int(record["label_end_ns"])
                if end <= start:
                    raise ValueError("economic label must end after causal decision")
                roles = [key for key, (lower, upper) in bounds.items() if lower <= start < end < upper]
                if len(roles) != 1:
                    raise ValueError("label crosses temporal role or is outside declared data")
                identity = (record["source_id"], start)
                if identity in seen:
                    raise ValueError("duplicate supervised state")
                seen.add(identity)
                handles[roles[0]].write(json.dumps(record, allow_nan=False) + "\n")
                counts[roles[0]] += 1
        if not all(counts.values()):
            raise ValueError("both train and valid require examples")
        manifest = {
            "schema": "propevolve_reasoning_dataset_v1", "splits": bounds,
            "counts": counts, "lineage": lineage,
            "sealed_start_ns": sealed_start_ns,
            "files": {key: file_digest(temporary / f"{key}.jsonl") for key in bounds},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
