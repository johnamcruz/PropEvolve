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
                             sealed_start_ns: int, embedding_storage="json",
                             embedding_source_cache_root=None):
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
    if embedding_storage not in {
            "json", "float32_sidecar_v1", "source_embedding_reference_v1"}:
        raise ValueError("unknown embedding storage")
    if ((embedding_storage == "source_embedding_reference_v1")
            != (embedding_source_cache_root is not None)):
        raise ValueError("source embedding storage requires exactly one cache root")
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
    reference_cache = None
    reference_ticker = None
    reference_sources = {}
    reference_shape = None
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            train = stack.enter_context((temporary / "train.jsonl").open("x"))
            valid = stack.enter_context((temporary / "valid.jsonl").open("x"))
            handles = {"train": train, "valid": valid}
            embedding_handles = available_handles = None
            embedding_shapes = {key: None for key in bounds}
            if embedding_storage == "float32_sidecar_v1":
                embedding_handles = {role: stack.enter_context(
                    (temporary / f"{role}.embeddings.f32").open("xb")) for role in bounds}
                available_handles = {role: stack.enter_context(
                    (temporary / f"{role}.available.u8").open("xb")) for role in bounds}
            for record in records:
                start, end = int(record["completed_at_ns"]), int(record["label_end_ns"])
                if end <= start:
                    raise ValueError("economic label must end after causal decision")
                roles = [key for key, (lower, upper) in bounds.items() if lower <= start < end < upper]
                if len(roles) != 1:
                    raise ValueError("label crosses temporal role or is outside declared data")
                identity = (record.get("ticker", record["source_id"]), start)
                if identity in seen:
                    raise ValueError("duplicate supervised state")
                seen.add(identity)
                if embedding_handles is not None:
                    record = dict(record)
                    embeddings = np.asarray(record.pop("market_embeddings", None), np.float32)
                    available = np.asarray(record.pop("market_available", None), bool)
                    if (embeddings.ndim != 2 or available.shape != (embeddings.shape[0],)
                            or not available.any() or not np.isfinite(embeddings).all()):
                        raise ValueError("compact dataset requires a finite embedding window")
                    shape = tuple(int(value) for value in embeddings.shape)
                    if embedding_shapes[roles[0]] not in {None, shape}:
                        raise ValueError("embedding windows must have one shape per role")
                    embedding_shapes[roles[0]] = shape
                    record["market_embedding_index"] = counts[roles[0]]
                    embedding_handles[roles[0]].write(embeddings.tobytes(order="C"))
                    available_handles[roles[0]].write(available.astype(np.uint8).tobytes(order="C"))
                elif embedding_storage == "source_embedding_reference_v1":
                    from ..cache import EmbeddingCache
                    record = dict(record)
                    embeddings = np.asarray(record.pop("market_embeddings", None), np.float32)
                    available = np.asarray(record.pop("market_available", None), bool)
                    ticker = record.get("ticker")
                    if not isinstance(ticker, str) or not ticker:
                        raise ValueError("source embedding reference requires a ticker")
                    if reference_ticker != ticker:
                        reference_cache = EmbeddingCache.load(
                            Path(embedding_source_cache_root) / ticker)
                        reference_ticker = ticker
                        if reference_cache.manifest.get("ticker") != ticker:
                            raise ValueError("embedding cache ticker differs from supervised record")
                        descriptor = {
                            "manifest_sha256": file_digest(
                                reference_cache.root / "manifest.json"),
                            "rows": len(reference_cache.embeddings),
                        }
                        if ticker in reference_sources and reference_sources[ticker] != descriptor:
                            raise ValueError("embedding source changed during dataset publication")
                        reference_sources[ticker] = descriptor
                    cache = reference_cache
                    if (embeddings.ndim != 2 or available.shape != (embeddings.shape[0],)
                            or not available.any() or not np.isfinite(embeddings).all()):
                        raise ValueError("indexed dataset requires a finite embedding window")
                    shape = tuple(int(value) for value in embeddings.shape)
                    if reference_shape not in {None, shape}:
                        raise ValueError("indexed embedding windows must share one shape")
                    reference_shape = shape
                    expected_mask = np.arange(shape[0]) >= shape[0] - int(available.sum())
                    if not np.array_equal(available, expected_mask):
                        raise ValueError("embedding availability must be one causal suffix")
                    timestamp = np.datetime64(start, "ns")
                    row = int(np.searchsorted(cache.timestamps, timestamp))
                    count = int(available.sum())
                    first = row - count + 1
                    if (row >= len(cache.timestamps) or cache.timestamps[row] != timestamp
                            or first < 0 or not np.array_equal(
                                embeddings[-count:], np.asarray(cache.embeddings[first:row + 1], np.float32))):
                        raise ValueError("embedding window differs from authenticated source cache")
                    record["market_embedding_reference"] = {
                        "ticker": ticker, "row": row, "available_count": count,
                    }
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
        if embedding_storage == "float32_sidecar_v1":
            manifest["embedding_storage"] = {
                "kind": embedding_storage,
                "roles": {role: {
                    "shape": [counts[role], *embedding_shapes[role]],
                    "embeddings_file": f"{role}.embeddings.f32",
                    "embeddings_sha256": file_digest(temporary / f"{role}.embeddings.f32"),
                    "available_file": f"{role}.available.u8",
                    "available_sha256": file_digest(temporary / f"{role}.available.u8"),
                } for role in bounds},
            }
        elif embedding_storage == "source_embedding_reference_v1":
            manifest["embedding_storage"] = {
                "kind": embedding_storage,
                "cache_root": str(Path(embedding_source_cache_root)),
                "context_steps": reference_shape[0],
                "embedding_dim": reference_shape[1],
                "sources": dict(sorted(reference_sources.items())),
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
    actions_by_role = {"train": {}, "valid": {}}
    tickers_by_role = {"train": {}, "valid": {}}
    teacher_free = 0
    specialist_target_records = 0
    seen = set()
    storage = manifest.get("embedding_storage")
    reference_caches = {}
    if storage is not None and storage.get("kind") == "source_embedding_reference_v1":
        from ..cache import EmbeddingCache
        cache_root = Path(storage.get("cache_root", ""))
        if (type(storage.get("context_steps")) is not int or storage["context_steps"] < 1
                or type(storage.get("embedding_dim")) is not int or storage["embedding_dim"] < 1
                or not isinstance(storage.get("sources"), dict) or not storage["sources"]):
            raise ValueError("invalid embedding source reference storage")
        for ticker, descriptor in storage["sources"].items():
            cache = EmbeddingCache.load(cache_root / ticker)
            if (cache.manifest.get("ticker") != ticker
                    or file_digest(cache.root / "manifest.json") != descriptor.get("manifest_sha256")
                    or cache.embeddings.shape != (descriptor.get("rows"), storage["embedding_dim"])):
                raise ValueError("embedding source differs from dataset manifest")
            reference_caches[ticker] = cache
    for role in ("train", "valid"):
        filename = root / f"{role}.jsonl"
        if file_digest(filename) != manifest["files"][role]:
            raise ValueError(f"{role} dataset differs from manifest")
        lower, upper = manifest["splits"][role]
        if not (type(lower) is type(upper) is int and lower < upper <= sealed):
            raise ValueError("invalid or unsealed dataset role")
        side_embeddings = side_available = None
        if storage is not None:
            if storage.get("kind") not in {
                    "float32_sidecar_v1", "source_embedding_reference_v1"}:
                raise ValueError("unknown embedding storage")
            if storage.get("kind") == "float32_sidecar_v1":
                descriptor = storage["roles"][role]
                shape = tuple(descriptor["shape"])
                if (shape[0] != manifest["counts"][role] or len(shape) != 3
                        or min(shape) < 1):
                    raise ValueError("invalid embedding sidecar shape")
                embedding_file = root / descriptor["embeddings_file"]
                available_file = root / descriptor["available_file"]
                if (file_digest(embedding_file) != descriptor["embeddings_sha256"]
                        or file_digest(available_file) != descriptor["available_sha256"]):
                    raise ValueError("embedding sidecar differs from manifest")
                side_embeddings = np.memmap(embedding_file, dtype=np.float32, mode="r", shape=shape)
                side_available = np.memmap(
                    available_file, dtype=np.uint8, mode="r", shape=(shape[0], shape[1]))
        with filename.open() as stream:
            for line in stream:
                record = json.loads(line)
                start, end = record.get("completed_at_ns"), record.get("label_end_ns")
                if (type(start) is not int or type(end) is not int
                        or not lower <= start < end < upper):
                    raise ValueError("record crosses its chronological role")
                identity = (record.get("ticker", record.get("source_id")), start)
                if not identity[0] or identity in seen:
                    raise ValueError("missing or duplicate supervised state")
                seen.add(identity)
                ticker = record.get("ticker")
                if ticker is not None:
                    if not isinstance(ticker, str) or not ticker:
                        raise ValueError("invalid supervised ticker")
                    tickers_by_role[role][ticker] = tickers_by_role[role].get(ticker, 0) + 1
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
                if side_embeddings is not None:
                    index = record.get("market_embedding_index")
                    if type(index) is not int or index != counts[role]:
                        raise ValueError("invalid embedding sidecar index")
                    embeddings = side_embeddings[index]
                    available = side_available[index].astype(bool)
                elif reference_caches:
                    reference = record.get("market_embedding_reference")
                    if (not isinstance(reference, dict)
                            or set(reference) != {"ticker", "row", "available_count"}
                            or reference["ticker"] != ticker
                            or type(reference["row"]) is not int
                            or type(reference["available_count"]) is not int):
                        raise ValueError("invalid embedding source reference")
                    cache = reference_caches.get(ticker)
                    row, count = reference["row"], reference["available_count"]
                    first = row - count + 1
                    if (cache is None or first < 0 or row >= len(cache.timestamps)
                            or not 1 <= count <= storage["context_steps"]
                            or int(cache.timestamps[row].astype("datetime64[ns]").astype(np.int64)) != start):
                        raise ValueError("embedding source reference differs from causal row")
                    embeddings = cache.embeddings[first:row + 1]
                    available = np.ones(count, dtype=bool)
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
                    role_actions = actions_by_role[role]
                    role_actions[action] = role_actions.get(action, 0) + 1
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
        "actions_by_role": {
            role: dict(sorted(values.items())) for role, values in actions_by_role.items()
        },
        "tickers_by_role": {
            role: dict(sorted(values.items())) for role, values in tickers_by_role.items()
        },
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
