"""Direct supervised examples; future targets never enter the model prompt."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..decision import Action
from .context import ContextWindow
from .labels import ActionLabels
from .integrity import file_digest


_TEACHER_PREFIXES = ("expansion.", "trend.", "regime.", "volume.")
_FUTURE_PROMPT_PREFIXES = ("future_", "label_", "outcome_")


def _contains_future_target(value):
    if isinstance(value, dict):
        return any((str(key).lower().startswith(_FUTURE_PROMPT_PREFIXES)
                    or str(key).lower().endswith("_target_before_stop")
                    or _contains_future_target(item)) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_future_target(item) for item in value)
    return False


def context_messages(context: ContextWindow, legal_actions) -> list[dict[str, str]]:
    """One shared, compact serializer for SFT and inference.

Only low-dimensional named specialist/account signals belong here; do not
serialize thousands of FFM latent coordinates as decimal tokens.
    """
    if not context.available.any():
        raise ValueError("context has no completed observations")
    values = context.values[context.available]
    text_steps = len(values) if context.text_steps is None else context.text_steps
    values = values[-text_steps:]
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


def embedding_payload(context):
    """Continuous arrays travel outside language-model messages/targets."""
    if context.embeddings is None:
        return {}
    return {"market_embeddings": context.embeddings.tolist(), "market_available": context.available.tolist()}


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
        **embedding_payload(context),
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
                             source_id: str, label_end_ns: int, economic_contract: dict,
                             excursions=None, target_grid=None) -> dict:
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
    if target_grid is not None:
        if (not target_grid or any(len(value) != 2 or
                any(type(item) is not bool for item in value) for value in target_grid.values())):
            raise ValueError("market target grid requires uncensored Long/Short labels")
        completion["target_before_stop_by_r"] = {
            str(key): {"long": value[0], "short": value[1]}
            for key, value in target_grid.items()
        }
        messages[0]["content"] = (
            "From completed-bar evidence, estimate whether Long and Short each reach every "
            "configured net profit barrier before the adverse barrier within the declared "
            "horizon. Return the labeled outcomes as JSON. Future prices are unknown."
        )
    if excursions is not None:
        completion["future_excursions"] = excursions
        messages[0]["content"] += (
            " Also estimate full-horizon gross MFE/MAE and terminal net R for both sides. "
            "Excursion extrema are not stop-managed trade returns."
        )
    return {
        "schema": "propevolve_reasoning_supervision_v1", "source_id": source_id,
        "completed_at_ns": context.timestamps[-1], "label_end_ns": label_end_ns,
        "messages": messages + [{"role": "assistant", "content": json.dumps(completion, allow_nan=False)}],
        "targets": completion,
        **embedding_payload(context),
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


def audit_supervised_dataset(path: str | Path, *, specialist_score_mode: str) -> dict:
    """Inspect and publish a hash-bound causal audit for one frozen dataset.

    This is intentionally stricter than the writer.  The writer checks temporal
    placement while streaming; this independent pass reopens every serialized
    row and verifies the exact artifact consumed by SFT.
    """
    import os
    import tempfile

    root = Path(path)
    audit_path = root / "audit.json"
    if audit_path.exists():
        raise FileExistsError(f"dataset audit already exists: {audit_path}")
    if specialist_score_mode not in {"out_of_fold", "post_fit"}:
        raise ValueError("unknown specialist score mode")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "propevolve_reasoning_dataset_v1":
        raise ValueError("unsupported reasoning dataset schema")
    if set(manifest.get("splits", {})) != {"train", "valid"}:
        raise ValueError("dataset audit requires train and valid roles")
    sealed = manifest.get("sealed_start_ns")
    if type(sealed) is not int:
        raise ValueError("dataset sealed boundary is invalid")

    counts = {"train": 0, "valid": 0}
    actions = {}
    teacher_free = 0
    specialist_target_records = 0
    seen = set()
    for role in ("train", "valid"):
        filename = root / f"{role}.jsonl"
        if file_digest(filename) != manifest["files"][role]:
            raise ValueError(f"{role} dataset differs from manifest")
        lower, upper = manifest["splits"][role]
        if not (type(lower) is type(upper) is int and lower < upper <= sealed):
            raise ValueError("invalid or unsealed dataset role")
        with filename.open() as stream:
            for line in stream:
                record = json.loads(line)
                start, end = record.get("completed_at_ns"), record.get("label_end_ns")
                if (type(start) is not int or type(end) is not int
                        or not lower <= start < end < upper):
                    raise ValueError("record crosses its chronological role")
                identity = (record.get("source_id"), start)
                if not identity[0] or identity in seen:
                    raise ValueError("missing or duplicate supervised state")
                seen.add(identity)
                messages = record.get("messages")
                if (not isinstance(messages, list)
                        or [item.get("role") for item in messages] != ["system", "user", "assistant"]):
                    raise ValueError("unexpected supervised conversation schema")
                prompt = json.loads(messages[1]["content"])
                if _contains_future_target(prompt):
                    raise ValueError("future target leaked into causal prompt")
                fields = prompt.get("fields")
                history = np.asarray(prompt.get("history_oldest_first"), dtype=np.float64)
                if (not isinstance(fields, list) or not fields or history.ndim != 2
                        or history.shape[1] != len(fields) or not np.isfinite(history).all()):
                    raise ValueError("invalid causal prompt history")
                embeddings = record.get("market_embeddings")
                available = record.get("market_available")
                if embeddings is not None or available is not None:
                    values = np.asarray(embeddings, dtype=np.float64)
                    mask = np.asarray(available)
                    if values.ndim != 2 or not np.isfinite(values).all():
                        raise ValueError("nonfinite embedding window")
                    if (mask.shape != (values.shape[0],) or mask.dtype != np.bool_
                            or not mask.any()):
                        raise ValueError("invalid embedding availability mask")
                    if any(field.startswith(_TEACHER_PREFIXES) for field in fields):
                        raise ValueError("specialist field leaked into teacher-free prompt")
                    teacher_free += 1
                targets = record.get("targets", {})
                specialist_targets = targets.get("specialist_targets")
                if specialist_targets is not None:
                    if (not isinstance(specialist_targets, dict) or not specialist_targets
                            or any(not isinstance(key, str) or not key.startswith(_TEACHER_PREFIXES)
                                   for key in specialist_targets)
                            or not np.isfinite(list(specialist_targets.values())).all()
                            or any(not 0.0 <= float(value) <= 1.0
                                   for value in specialist_targets.values())):
                        raise ValueError("invalid specialist training target")
                    specialist_target_records += 1
                if "action_order" in targets:
                    from .supervision import action_targets
                    try:
                        action_targets(record)
                    except ValueError as error:
                        if "prompt legal actions" in str(error):
                            raise ValueError("legal action targets differ from prompt") from error
                        raise
                    action = messages[2]["content"]
                    if action not in targets["action_order"]:
                        raise ValueError("assistant action is not a legal target")
                    actions[action] = actions.get(action, 0) + 1
                elif not {"long_target_before_stop", "short_target_before_stop"}.issubset(targets):
                    raise ValueError("record has neither action nor market supervision")
                counts[role] += 1
        if counts[role] != manifest["counts"][role]:
            raise ValueError(f"{role} record count differs from manifest")
    if not actions and teacher_free == 0:
        raise ValueError("dataset contains no auditable supervision")

    audit = {
        "schema": "propevolve_reasoning_dataset_audit_v1",
        "status": "PASS",
        "manifest_sha256": file_digest(manifest_path),
        "specialist_score_mode": specialist_score_mode,
        "sealed_touched": False,
        "counts": counts,
        "actions": dict(sorted(actions.items())),
        "teacher_free_prompt_records": teacher_free,
        "specialist_target_records": specialist_target_records,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".audit-", suffix=".json", dir=root)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
        os.rename(temporary, audit_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return audit
