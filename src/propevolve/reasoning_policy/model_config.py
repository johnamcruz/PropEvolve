"""Shared model selection for SFT recipes and finite-action inference.

Model identities are explicit repository IDs or local paths, never filename
allowlists. Relative paths retain the existing repository-working-directory
semantics. Importing configuration does not import MLX or download weights.
"""

import json
from pathlib import Path


def model_defaults():
    # Repository config discovery, not a model/run identity or numeric setting.
    return json.loads((Path(__file__).resolve().parents[3] /
                       "config/reasoning/defaults.json").read_text())


def template_options(options=None):
    result = dict(model_defaults()["chat_template_kwargs"])
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError("chat_template_kwargs must be an object")
        result.update(options)
    if {"tokenize", "add_generation_prompt", "messages", "conversation"} & result.keys():
        raise ValueError("chat template options cannot override the completion protocol")
    return result


def validate_model_settings(payload):
    for key in ("model", "adapter_path", "max_seq_length"):
        if key not in payload:
            raise ValueError(f"missing model setting: {key}")
    if not isinstance(payload["model"], str) or not payload["model"].strip():
        raise ValueError("model must be an explicit nonempty ID or local path")
    adapter = payload["adapter_path"]
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        raise ValueError("adapter_path must be a nonempty path or null for the base model")
    if type(payload["max_seq_length"]) is not int or payload["max_seq_length"] < 1:
        raise ValueError("max_seq_length must be positive")
    template_options(payload.get("chat_template_kwargs"))
    return payload


def read_model_settings(path):
    payload = {**model_defaults(), **json.loads(Path(path).read_text())}
    return validate_model_settings(payload)


def verify_adapter_base(model, adapter_path):
    if adapter_path is None:
        return
    # Native MLX-LM records the base in this artifact-format filename.
    # This is not a run-specific configuration path.
    metadata = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    recorded = metadata.get("model")
    same = recorded == model
    if isinstance(recorded, str) and Path(recorded).is_dir() and Path(model).is_dir():
        same = Path(recorded).resolve() == Path(model).resolve()
    if not same:
        raise ValueError("adapter base model differs from configured model; select a matching adapter")
