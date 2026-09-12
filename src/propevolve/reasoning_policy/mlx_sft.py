"""Audited, prompt-masked QLoRA through MLX-LM's native trainer.

Importing this module loads neither MLX nor a model. Preparing/launching are
explicit operations, never side effects of importing the application package.
"""

import argparse
import json
from pathlib import Path
import math
import numpy as np
from dataclasses import dataclass
from .integrity import file_digest
from .model_config import validate_model_settings, model_defaults, read_recipe, resolve_model_resources
from .tokenization import encode_completion


def read_sft_config(path: str | Path, *, root=None) -> dict:
    payload = resolve_model_resources({**model_defaults(), **read_recipe(path)}, root=root)
    required = {
        "model", "data", "adapter_path", "train", "fine_tune_type", "mask_prompt",
        "num_layers", "batch_size", "iters", "learning_rate", "max_seq_length",
        "grad_checkpoint", "grad_accumulation_steps", "lora_parameters",
    }
    if not required.issubset(payload):
        raise ValueError(f"missing SFT settings: {sorted(required - set(payload))}")
    validate_model_settings(payload)
    if payload.get("epochs") is not None and (
            type(payload["epochs"]) is not int or payload["epochs"] < 1):
        raise ValueError("epochs must be a positive integer or null")
    for key in ("include_partial_batch", "save_training_state"):
        if type(payload[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if payload["epochs"] is not None and not payload["include_partial_batch"]:
        raise ValueError("epoch training requires complete partial-batch coverage")
    if payload["include_partial_batch"] and payload["batch_sampling"] != "random":
        raise ValueError("partial batches require non-oversampled random coverage")
    if payload["resume_training_state"] is not None and (
            not isinstance(payload["resume_training_state"], str)
            or not payload["resume_training_state"].strip()):
        raise ValueError("resume_training_state must be a path or null")
    stage_role = payload.get("stage_role")
    if stage_role is not None and (not isinstance(stage_role, str) or not stage_role.strip()):
        raise ValueError("SFT stage role must be a nonempty string or null")
    distillation_targets = payload.get("distillation_targets")
    if distillation_targets is not None and (
            not isinstance(distillation_targets, list) or not distillation_targets
            or len(distillation_targets) != len(set(distillation_targets))
            or any(not isinstance(name, str) or not name.strip()
                   for name in distillation_targets)):
        raise ValueError("distillation targets must be unique nonempty names or null")
    parent_requirements = payload.get("resume_adapter_requirements")
    if parent_requirements is not None and (
            not isinstance(parent_requirements, dict) or not parent_requirements):
        raise ValueError("resume adapter requirements must be a nonempty object or null")
    supervision = payload["action_supervision"]
    from .coverage_sampling import validate_coverage
    validate_coverage(payload["coverage_sampling"])
    from .coverage_sampling import validate_prepared_sampling
    validate_prepared_sampling(payload["prepared_sampling"])
    from .targeted_subset import validate_targeted_sampling
    validate_targeted_sampling(payload["targeted_sampling"])
    if payload["coverage_sampling"] is not None and (
            payload["batch_sampling"] != "random" or not payload["include_partial_batch"]
            or supervision["enabled"]):
        raise ValueError("coverage sampling requires partial-batch market training")
    if payload["targeted_sampling"] is not None and (
            payload["coverage_sampling"] is not None
            or payload["prepared_sampling"] is not None
            or payload["batch_sampling"] != "balanced_actions"
            or payload["include_partial_batch"]
            or not supervision["enabled"]):
        raise ValueError(
            "targeted sampling requires full-view balanced action training")
    from .market_distillation import validate_market_distillation
    distillation = payload["market_distillation"]
    validate_market_distillation(distillation)
    if distillation is not None and (
            supervision["enabled"] or payload["input_mode"] != "embeddings"
            or payload["stage_role"] != "market_distillation"
            or payload["market_loss_chunk_size"] is not None):
        raise ValueError("direct market distillation requires embedding market SFT only")
    corrective_distillation = payload["error_selected_distillation"]
    if corrective_distillation is not None:
        if (not isinstance(corrective_distillation, dict)
                or set(corrective_distillation) != {"loss_weight", "settings"}
                or isinstance(corrective_distillation["loss_weight"], bool)
                or not isinstance(corrective_distillation["loss_weight"], (int, float))
                or not math.isfinite(float(corrective_distillation["loss_weight"]))
                or corrective_distillation["loss_weight"] <= 0):
            raise ValueError("invalid error-selected distillation configuration")
        validate_market_distillation(corrective_distillation["settings"])
        groups = {channel["name"].split(".", 1)[0]
                  for channel in corrective_distillation["settings"]["channels"]}
        if (not supervision["enabled"] or payload["input_mode"] != "embeddings"
                or payload.get("stage_role") != "trade_mastery"
                or payload.get("distillation_targets") is None
                or set(payload["distillation_targets"]) != groups):
            raise ValueError(
                "error-selected distillation requires trade-mastery action rows "
                "and exactly declared teacher groups")
    retention = payload["mastered_anchor_retention"]
    from .targeted_subset import validate_mastered_anchor_retention
    validate_mastered_anchor_retention(retention)
    if retention is not None:
        if payload["targeted_sampling"] is None or not supervision["enabled"]:
            raise ValueError(
                "mastered anchor retention requires targeted action supervision")
    chunk_size = payload["market_loss_chunk_size"]
    if chunk_size is not None and (
            type(chunk_size) is not int or chunk_size < 1
            or supervision["enabled"] or payload["input_mode"] != "embeddings"):
        raise ValueError("chunked loss requires a positive chunk size and embedding market SFT")
    if payload.get("decision_objective") not in {"full_action", "hierarchical_binary"}:
        raise ValueError("unknown reasoning decision objective")
    if (type(supervision["enabled"]) is not bool or any(
            isinstance(supervision[key], bool) or not math.isfinite(supervision[key]) or supervision[key] < 0
            for key in ("soft_target_weight", "ranking_weight", "margin"))):
        raise ValueError("invalid action supervision settings")
    if supervision["enabled"] and supervision["soft_target_weight"] + supervision["ranking_weight"] <= 0:
        raise ValueError("enabled action supervision requires learning weight")
    schedule = payload.get("lr_schedule")
    if schedule is not None and (not isinstance(schedule, dict)
            or set(schedule) != {"kind", "end", "decay_updates"}
            or schedule["kind"] != "cosine_decay"
            or isinstance(schedule["end"], bool)
            or not math.isfinite(float(schedule["end"]))
            or not 0 <= schedule["end"] <= payload["learning_rate"]
            or type(schedule["decay_updates"]) is not int
            or schedule["decay_updates"] < 1):
        raise ValueError("invalid learning-rate schedule")
    if payload.get("batch_sampling", "random") not in {"random", "balanced_actions"}:
        raise ValueError("unknown SFT batch sampling strategy")
    components = payload.get("trainable_components")
    if (not isinstance(components, list) or not components
            or len(components) != len(set(components))
            or set(components) - {"lora", "projector"}):
        raise ValueError("invalid trainable components")
    if "projector" in components and payload["input_mode"] != "embeddings":
        raise ValueError("projector trainable component requires embedding inputs")
    from .supervised_trainer import ValidationLossGuard
    guard = ValidationLossGuard(payload.get("early_stopping"), on_improvement=lambda report: None)
    if not supervision["enabled"] and payload.get("batch_sampling") == "balanced_actions":
        raise ValueError("balanced action sampling requires action supervision")
    valid_action_monitors = {
        ("worst_action_boundary_loss", "min"),
        ("worst_action_advantage", "max"),
        ("worst_task_boundary_loss", "min"),
        ("worst_task_advantage", "max"),
    }
    if (supervision["enabled"] and payload.get("batch_sampling") == "balanced_actions"
            and (guard.monitor, guard.mode) not in valid_action_monitors):
        raise ValueError("balanced action SFT must select checkpoints by a worst-action boundary")
    if payload["decision_objective"] == "hierarchical_binary" and not supervision["enabled"]:
        raise ValueError("hierarchical decision objective requires action supervision")
    if payload["adapter_path"] is None:
        raise ValueError("SFT requires an adapter output path")
    metrics_path = payload.get("validation_metrics_path")
    if metrics_path is not None and (not isinstance(metrics_path, str)
                                     or not metrics_path.strip()):
        raise ValueError("validation_metrics_path must be a nonempty path or null")
    log_names = [payload.get(key) for key in (
        "training_log_filename", "training_events_filename")]
    log_prefix = payload.get("training_log_prefix")
    if (any(not isinstance(value, str) or not value.strip()
            or Path(value).name != value for value in log_names)
            or len(set(log_names)) != len(log_names)
            or not isinstance(log_prefix, str) or not log_prefix.strip()):
        raise ValueError("invalid training log settings")
    if (payload["fine_tune_type"] != "lora" or payload["train"] is not True
            or payload["mask_prompt"] is not True or payload.get("trust_remote_code") is not False):
        raise ValueError("challenger requires prompt-masked LoRA and no remote code")
    for name in ("num_layers", "batch_size", "validation_batch_size", "iters",
                 "max_seq_length", "grad_accumulation_steps"):
        if type(payload[name]) is not int or payload[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if payload["iters"] % payload["grad_accumulation_steps"]:
        raise ValueError("SFT iterations must complete gradient accumulation groups")
    if payload["steps_per_eval"] % payload["steps_per_report"]:
        raise ValueError("SFT evaluation cadence must align with reported optimizer steps")
    if payload["save_training_state"] and (
            payload["save_every"] % payload["steps_per_report"]
            or payload["steps_per_report"] % payload["grad_accumulation_steps"]):
        raise ValueError("training checkpoints must align with reported optimizer updates")
    lora = payload["lora_parameters"]
    if set(lora) != {"rank", "scale", "dropout"}:
        raise ValueError("LoRA settings require exactly rank, scale and dropout")
    rank = lora["rank"]
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if (isinstance(lora["scale"], bool) or not math.isfinite(float(lora["scale"]))
            or lora["scale"] <= 0 or isinstance(lora["dropout"], bool)
            or not math.isfinite(float(lora["dropout"])) or not 0 <= lora["dropout"] < 1):
        raise ValueError("LoRA scale/dropout settings are invalid")
    if (isinstance(payload["learning_rate"], bool)
            or not math.isfinite(float(payload["learning_rate"])) or payload["learning_rate"] <= 0):
        raise ValueError("learning_rate must be finite and positive")
    component_rates = payload.get("component_learning_rates")
    if component_rates is not None:
        if (not isinstance(component_rates, dict)
                or set(component_rates) != {"lora", "projector"}
                or set(components) != {"lora", "projector"}
                or schedule is not None
                or any(isinstance(value, bool) or not math.isfinite(float(value))
                       or value <= 0 for value in component_rates.values())):
            raise ValueError(
                "component learning rates require positive LoRA/projector rates, "
                "both trainable components, and no shared schedule")
    requirements = payload.get("dataset_requirements")
    if requirements is not None:
        minimums = (requirements.get("minimum_rows_per_action", {})
                    if isinstance(requirements, dict) else {})
        valid_minimums = (
            isinstance(minimums, dict)
            and set(minimums) == {"train", "valid"}
            and all(
                (type(value) is int and value >= 1)
                or (
                    isinstance(value, dict) and bool(value)
                    and all(
                        isinstance(action, str) and bool(action)
                        and type(count) is int and count >= 1
                        for action, count in value.items()
                    )
                )
                for value in minimums.values()
            )
        )
        if (not isinstance(requirements, dict)
                or set(requirements) != {
                    "minimum_rows_per_action", "expected_splits", "sealed_start_ns"}
                or not valid_minimums
                or set(requirements["expected_splits"]) != {"train", "valid"}
                or any(not isinstance(bounds, list) or len(bounds) != 2
                       or any(type(value) is not int for value in bounds)
                       or bounds[0] >= bounds[1]
                       for bounds in requirements["expected_splits"].values())
                or type(requirements["sealed_start_ns"]) is not int):
            raise ValueError("invalid SFT dataset requirements")
    return payload


def verify_dataset(path: str | Path, *, requirements=None,
                   required_target_groups=None) -> dict:
    """Reject unreviewed or changed data before loading any large model."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    manifest_hash = file_digest(root / "manifest.json")
    if (manifest.get("schema") != "propevolve_reasoning_dataset_v1"
            or audit.get("status") != "PASS"
            or audit.get("manifest_sha256") != manifest_hash
            or audit.get("specialist_score_mode") not in {"out_of_fold", "post_fit"}
            or audit.get("sealed_touched") is not False):
        raise ValueError("dataset needs a matching passed causal/specialist audit")
    sealed_start = manifest.get("sealed_start_ns")
    if (type(sealed_start) is not int or any(
            bounds[1] > sealed_start for bounds in manifest["splits"].values())):
        raise ValueError("dataset crosses sealed final-validation boundary")
    for role in ("train", "valid"):
        filename = root / f"{role}.jsonl"
        digest = file_digest(filename)
        if digest != manifest["files"][role]:
            raise ValueError(f"{role} dataset changed after audit")
    if required_target_groups is not None:
        if (not isinstance(required_target_groups, list) or not required_target_groups
                or len(required_target_groups) != len(set(required_target_groups))
                or any(not isinstance(group, str) or not group.strip()
                       for group in required_target_groups)):
            raise ValueError("required target groups must be unique nonempty names")
        required = set(required_target_groups)
        for role in ("train", "valid"):
            with (root / f"{role}.jsonl").open() as stream:
                for row_number, line in enumerate(stream, 1):
                    targets = json.loads(line).get("targets", {}).get(
                        "specialist_targets", {})
                    present = {name.split(".", 1)[0] for name in targets}
                    missing = required - present
                    if missing:
                        raise ValueError(
                            f"{role} row {row_number} lacks required target groups: "
                            f"{sorted(missing)}")
    if requirements is not None:
        if (manifest.get("splits") != requirements["expected_splits"]
                or manifest.get("sealed_start_ns") != requirements["sealed_start_ns"]):
            raise ValueError("dataset temporal roles differ from SFT requirements")
        action_counts = audit.get("actions_by_role")
        for role, minimum in requirements["minimum_rows_per_action"].items():
            counts = None if not isinstance(action_counts, dict) else action_counts.get(role)
            required = ({action: minimum for action in (
                "WAIT", "ENTER_LONG_1", "ENTER_SHORT_1")}
                if type(minimum) is int else minimum)
            if (not isinstance(counts, dict)
                    or any(type(counts.get(action)) is not int
                           or counts[action] < threshold
                           for action, threshold in required.items())):
                raise ValueError(f"{role} dataset lacks minimum rows per action")
    storage = manifest.get("embedding_storage")
    if storage is not None:
        kind = storage.get("kind")
        if kind not in {"float32_sidecar_v1", "source_embedding_reference_v1"}:
            raise ValueError("unknown embedding storage")
        if kind == "float32_sidecar_v1":
            for descriptor in storage["roles"].values():
                for file_key, digest_key in (("embeddings_file", "embeddings_sha256"),
                                             ("available_file", "available_sha256")):
                    if file_digest(root / descriptor[file_key]) != descriptor[digest_key]:
                        raise ValueError("embedding sidecar changed after audit")
        else:
            cache_root = Path(storage["cache_root"])
            for ticker, descriptor in storage["sources"].items():
                ticker_root = cache_root / ticker
                if (file_digest(ticker_root / "manifest.json") != descriptor["manifest_sha256"]
                        or not (ticker_root / "embeddings.npy").is_file()
                        or not (ticker_root / "timestamps.npy").is_file()):
                    raise ValueError("embedding source changed after audit")
    return manifest


def view_contract(config: dict) -> dict:
    """Return only immutable inputs that affect prepared token/context rows."""
    keys = (
        "model", "data", "input_mode", "projector", "max_seq_length",
        "chat_template_kwargs", "action_verbalizers", "action_supervision",
        "trust_remote_code", "dataset_requirements", "distillation_targets",
    )
    contract = {key: config.get(key) for key in keys}
    if config.get("market_distillation") is not None:
        contract["market_distillation"] = config["market_distillation"]
    if config.get("error_selected_distillation") is not None:
        contract["error_selected_distillation"] = config["error_selected_distillation"]
    if config.get("coverage_sampling") is not None:
        contract["coverage_sampling"] = config["coverage_sampling"]
    if config.get("prepared_sampling") is not None:
        contract["prepared_sampling"] = config["prepared_sampling"]
    # Loss weights and margin affect optimization, never prepared alternatives.
    supervision = contract["action_supervision"]
    contract["action_supervision"] = {
        "enabled": supervision["enabled"],
    }
    return contract


def prepare_mlx_view(config_path: str | Path, output: str | Path, *, tokenizer, root=None) -> Path:
    """Make encoded native datasets without target/prompt leakage/truncation.

The supplied tokenizer is the model's actual tokenizer. This is a deliberate
system boundary for tests; the production caller loads it with MLX-LM.
    """
    import os
    import shutil
    import tempfile

    config = read_sft_config(config_path, root=root)
    source = Path(config["data"])
    manifest = verify_dataset(
        source, requirements=config.get("dataset_requirements"),
        required_target_groups=config.get("distillation_targets"))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"MLX view already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mlx-sft-view-", dir=output.parent))
    try:
        prepared_counts = {}
        for role in ("train", "valid"):
            count = 0
            selected = None
            if config.get("prepared_sampling") is not None:
                from .coverage_sampling import select_prepared_rows
                selected = select_prepared_rows(source / f"{role}.jsonl",
                    config["prepared_sampling"], role=role)
            with (source / f"{role}.jsonl").open() as raw, (temporary / f"{role}.jsonl").open("x") as target:
                for source_index, line in enumerate(raw):
                    if selected is not None and source_index not in selected:
                        continue
                    record = json.loads(line)
                    messages = record["messages"]
                    if [item["role"] for item in messages] != ["system", "user", "assistant"]:
                        raise ValueError("unexpected SFT conversation schema")
                    completion = messages[-1]["content"]
                    verbalizer = config["action_verbalizers"].get(completion, completion)
                    if config["input_mode"] == "embeddings":
                        from .projector import projector_prefix_tokens
                        reserved = projector_prefix_tokens(config["projector"])
                    else:
                        reserved = 0
                    if config.get("market_distillation") is not None:
                        from .market_distillation import encode_market_targets
                        encoded = encode_market_targets(record, config["market_distillation"],
                            tokenizer, max_seq_length=config["max_seq_length"] - reserved,
                            chat_template_kwargs=config["chat_template_kwargs"])
                    else:
                        tokens, offset = encode_completion(tokenizer, messages[:-1], verbalizer,
                            max_seq_length=config["max_seq_length"] - reserved,
                            chat_template_kwargs=config["chat_template_kwargs"])
                        encoded = {"tokens": tokens, "offset": offset}
                    if config.get("error_selected_distillation") is not None:
                        from .market_distillation import encode_market_targets
                        encoded["error_selected_distillation"] = encode_market_targets(
                            record, config["error_selected_distillation"]["settings"],
                            tokenizer, max_seq_length=config["max_seq_length"] - reserved,
                            chat_template_kwargs=config["chat_template_kwargs"])
                    if config["action_supervision"]["enabled"]:
                        from .supervision import action_targets
                        alternatives = action_targets(record)
                        encoded["action_targets"] = alternatives
                        encoded["alternatives"] = [encode_completion(
                            tokenizer, messages[:-1], config["action_verbalizers"][name],
                            max_seq_length=config["max_seq_length"] - reserved,
                            chat_template_kwargs=config["chat_template_kwargs"])
                            for name in alternatives["names"]]
                        completion_lengths = {len(tokens) - offset for tokens, offset in encoded["alternatives"]}
                        if completion_lengths != {2}:
                            raise ValueError("each action verbalizer must be exactly one tokenizer token")
                        encoded["target_name"] = completion
                    if config["input_mode"] == "embeddings":
                        import numpy as np
                        projector = config["projector"]
                        if "market_embedding_reference" in record:
                            reference = record["market_embedding_reference"]
                            storage = manifest.get("embedding_storage", {})
                            if (storage.get("kind") != "source_embedding_reference_v1"
                                    or set(reference) != {"ticker", "row", "available_count"}
                                    or reference["ticker"] not in storage["sources"]
                                    or type(reference["row"]) is not int
                                    or type(reference["available_count"]) is not int
                                    or not 0 <= reference["row"] < storage["sources"][reference["ticker"]]["rows"]
                                    or not 1 <= reference["available_count"] <= projector["context_steps"]
                                    or storage["context_steps"] != projector["context_steps"]
                                    or storage["embedding_dim"] != projector["embedding_dim"]):
                                raise ValueError("supervised record has an invalid embedding source reference")
                            encoded["market_embedding_reference"] = reference
                        elif "market_embedding_index" in record:
                            index = record["market_embedding_index"]
                            storage = manifest.get("embedding_storage", {}).get("roles", {}).get(role)
                            if (type(index) is not int or index != source_index or storage is None
                                    or storage["shape"][1:] != [projector["context_steps"],
                                                               projector["embedding_dim"]]):
                                raise ValueError("supervised record lacks matching compact embedding window")
                            encoded["market_embedding_index"] = index
                        else:
                            embeddings = np.asarray(record.get("market_embeddings"), np.float32)
                            available = np.asarray(record.get("market_available"), bool)
                            if (embeddings.shape != (projector["context_steps"], projector["embedding_dim"])
                                    or available.shape != (projector["context_steps"],)
                                    or not available.any() or not np.isfinite(embeddings).all()):
                                raise ValueError("supervised record lacks matching causal embedding window")
                            encoded.update(market_embeddings=embeddings.tolist(),
                                           market_available=available.tolist())
                        prompt = json.loads(messages[-2]["content"])
                        if any(not field.startswith(("account.", "trade.", "challenge.")) for field in prompt["fields"]):
                            raise ValueError("teacher fields leaked into teacher-free SFT prompt")
                        state_fields = config["projector"].get("state_fields", [])
                        if state_fields:
                            fields = prompt.get("fields")
                            history = np.asarray(prompt.get("history_oldest_first"), dtype=np.float32)
                            if (not isinstance(fields, list) or len(set(fields)) != len(fields)
                                    or history.ndim != 2 or history.shape[1] != len(fields)
                                    or not len(history) or not np.isfinite(history).all()
                                    or any(field not in fields for field in state_fields)):
                                raise ValueError("prepared causal state differs from projector contract")
                            encoded["causal_state"] = [
                                float(history[-1, fields.index(field)]) for field in state_fields
                            ]
                    if config.get("coverage_sampling") is not None:
                        from .coverage_sampling import coverage_metadata
                        encoded["coverage"] = coverage_metadata(record, config["coverage_sampling"])
                    target.write(json.dumps(encoded) + "\n")
                    count += 1
            expected_count = manifest["counts"][role] if selected is None else len(selected)
            if count != expected_count or count < config["batch_size"]:
                raise ValueError(f"{role} sample count mismatch or incomplete batch")
            prepared_counts[role] = count
        effective = {**config, "data": str(output.resolve())}
        (temporary / "sft.json").write_text(json.dumps(effective, indent=2))
        (temporary / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
        (temporary / "view_manifest.json").write_text(json.dumps({
            "config": config, "view_contract": view_contract(config),
            "prepared_counts": prepared_counts,
            "source_manifest": manifest,
            "files": {role: file_digest(temporary / f"{role}.jsonl") for role in ("train", "valid")},
        }, indent=2))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "sft.json"


def verify_mlx_view(config_path, output, *, root=None):
    """Reuse preparation only if its source, recipe and rendered files match."""
    config = read_sft_config(config_path, root=root)
    manifest = verify_dataset(
        config["data"], requirements=config.get("dataset_requirements"),
        required_target_groups=config.get("distillation_targets"))
    output = Path(output)
    receipt = json.loads((output / "view_manifest.json").read_text())
    recorded_contract = receipt.get("view_contract")
    if recorded_contract is None:
        recorded_contract = view_contract(receipt["config"])
    if recorded_contract != view_contract(config):
        raise ValueError("prepared view contract changed")
    if receipt["source_manifest"] != manifest:
        raise ValueError("prepared view source changed")
    for role in ("train", "valid"):
        if file_digest(output / f"{role}.jsonl") != receipt["files"][role]:
            raise ValueError("prepared view changed")
    return output / "sft.json"


@dataclass(frozen=True)
class EncodedExample:
    tokens: tuple[int, ...]
    offset: int

    def __len__(self):
        return len(self.tokens)


class EncodedDataset:
    """MLX-LM dataset interface without applying a second chat template."""
    def __init__(self, path):
        self.rows = []
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                tokens, offset = tuple(row["tokens"]), row["offset"]
                if (type(offset) is not int or not 0 < offset < len(tokens)
                        or any(type(token) is not int or token < 0 for token in tokens)):
                    raise ValueError("invalid encoded supervision")
                self.rows.append(EncodedExample(tokens, offset))

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def process(self, row):
        return list(row.tokens), row.offset


class IndexedJsonRows:
    """Random-access JSONL with only byte offsets resident in Python memory."""

    def __init__(self, path):
        from array import array
        self.path = Path(path)
        self.offsets = array("Q")
        with self.path.open("rb") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(position)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        offset = self.offsets[index]
        with self.path.open("rb") as stream:
            stream.seek(offset)
            return json.loads(stream.readline())

    def __iter__(self):
        with self.path.open("rb") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


class PreparedDataset:
    """Lightweight encoded rows with mmap-backed market context."""

    def __init__(self, view, role):
        import numpy as np
        view = Path(view)
        self.rows = IndexedJsonRows(view / f"{role}.jsonl")
        receipt = json.loads((view / "view_manifest.json").read_text())
        source = Path(receipt["config"]["data"])
        storage = receipt["source_manifest"].get("embedding_storage", {})
        descriptor = storage.get("roles", {}).get(role)
        self.embeddings = self.available = None
        self.source_embeddings = {}
        if descriptor is not None:
            shape = tuple(descriptor["shape"])
            self.embeddings = np.memmap(
                source / descriptor["embeddings_file"], dtype=np.float32, mode="r", shape=shape)
            self.available = np.memmap(
                source / descriptor["available_file"], dtype=np.uint8, mode="r",
                shape=(shape[0], shape[1]))
        if storage.get("kind") == "source_embedding_reference_v1":
            cache_root = Path(storage["cache_root"])
            for ticker, source_descriptor in storage["sources"].items():
                self.source_embeddings[ticker] = np.load(
                    cache_root / ticker / "embeddings.npy", mmap_mode="r")
                if self.source_embeddings[ticker].shape != (
                        source_descriptor["rows"], storage["embedding_dim"]):
                    raise ValueError("embedding source shape differs from dataset manifest")
            self.context_steps = storage["context_steps"]

    def __len__(self):
        return len(self.rows)

    def sampling_rows(self):
        """Return lightweight targets/references without materializing embeddings."""
        return self.rows

    def __getitem__(self, index):
        row = self.rows[index]
        if "market_embedding_reference" in row:
            reference = row["market_embedding_reference"]
            source = self.source_embeddings.get(reference["ticker"])
            if source is None:
                raise ValueError("prepared row references a missing embedding source")
            count = reference["available_count"]
            end = reference["row"] + 1
            start = end - count
            if start < 0 or count > self.context_steps:
                raise ValueError("prepared embedding reference crosses its source boundary")
            result = dict(row)
            result.pop("market_embedding_reference")
            embeddings = np.zeros((self.context_steps, source.shape[1]), np.float32)
            available = np.zeros(self.context_steps, bool)
            embeddings[-count:] = source[start:end]
            available[-count:] = True
            result["market_embeddings"] = embeddings
            result["market_available"] = available
            return result
        if "market_embedding_index" not in row:
            return row
        if self.embeddings is None:
            raise ValueError("prepared row references missing embedding sidecar")
        result = dict(row)
        sidecar_index = result.pop("market_embedding_index")
        result["market_embeddings"] = self.embeddings[sidecar_index]
        result["market_available"] = self.available[sidecar_index].astype(bool)
        return result


def train_prepared(config_path, view, *, root=None):
    """Use the shared guarded MLX trainer with already verified tokens."""
    verify_mlx_view(config_path, view, root=root)
    config = read_sft_config(config_path, root=root)
    config["data"] = str(Path(view).resolve())
    from .supervised_trainer import train_supervised
    return train_supervised(config, view)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--root")
    parser.add_argument("--train", action="store_true", help="Explicitly launch native MLX-LM QLoRA")
    args = parser.parse_args(argv)
    config = read_sft_config(args.config, root=args.root)
    verify_dataset(
        config["data"], requirements=config.get("dataset_requirements"),
        required_target_groups=config.get("distillation_targets"))
    if args.train and Path(config["adapter_path"]).exists():
        raise FileExistsError("adapter output exists; choose a new path to preserve checkpoints")
    if Path(args.view).exists():
        effective = verify_mlx_view(args.config, args.view, root=args.root)
    else:
        # Optional runtime is imported only after the cheap integrity checks.
        from mlx_lm import load
        model, tokenizer = load(config["model"], tokenizer_config={"trust_remote_code": False})
        if not any("Quantized" in type(module).__name__ for _, module in model.named_modules()):
            raise ValueError("initial challenger requires a quantized base for QLoRA")
        effective = prepare_mlx_view(args.config, args.view, tokenizer=tokenizer, root=args.root)
        del model
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()
    if args.train:
        train_prepared(args.config, args.view, root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
