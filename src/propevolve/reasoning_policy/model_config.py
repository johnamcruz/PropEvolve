"""Shared model selection for SFT recipes and finite-action inference.

Model identities are explicit repository IDs or local paths, never filename
allowlists. Relative paths retain the existing repository-working-directory
semantics. Importing configuration does not import MLX or download weights.
"""

import json
from pathlib import Path

from ..decision import Action


def model_defaults():
    # Repository config discovery, not a model/run identity or numeric setting.
    source = Path(__file__).resolve().parents[3] / "config/reasoning/defaults.json"
    if not source.is_file():
        source = Path(__file__).with_name("defaults.json")
    return json.loads(source.read_text())


def template_options(options=None):
    result = dict(model_defaults()["chat_template_kwargs"])
    if options is not None:
        if not isinstance(options, dict):
            raise ValueError("chat_template_kwargs must be an object")
        result.update(options)
    if {"tokenize", "add_generation_prompt", "messages", "conversation"} & result.keys():
        raise ValueError("chat template options cannot override the completion protocol")
    return result


def action_verbalizers(settings=None):
    """Return the config-owned one-token policy vocabulary by canonical action."""
    values = model_defaults()["action_verbalizers"] if settings is None else settings
    expected = {action.name for action in Action}
    if (not isinstance(values, dict) or set(values) != expected
            or any(not isinstance(value, str) or not value.strip() for value in values.values())):
        raise ValueError("action verbalizers must define every action")
    if len(set(values.values())) != len(values):
        raise ValueError("action verbalizers must be unique")
    return dict(values)


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
    action_verbalizers(payload.get("action_verbalizers"))
    if payload.get("input_mode", "specialists") not in {"specialists", "embeddings"}:
        raise ValueError("invalid reasoning input mode")
    if payload.get("input_mode") == "embeddings":
        from .projector import validate_projector
        validate_projector(payload.get("projector"))
    return payload


def resolve_model_resources(payload, *, root=None):
    """An explicit workspace wins; preserve legacy CWD semantics when omitted.

Hub IDs are not paths. Local relative models must use model_source='local'.
    """
    payload = dict(payload)
    if root is None:
        root = payload.get("workspace_root")
    if root is not None:
        root = Path(root).resolve()
        for key in ("adapter_path", "data", "resume_adapter_file"):
            if payload.get(key) is not None:
                payload[key] = str((root / payload[key]).resolve())
        if payload.get("model_source") == "local":
            payload["model"] = str((root / payload["model"]).resolve())
        payload["workspace_root"] = str(root)
    return payload


def read_model_settings(path, *, root=None):
    payload = {**model_defaults(), **read_recipe(path)}
    return validate_model_settings(resolve_model_resources(payload, root=root))


def read_recipe(path, _parents=()):
    """JSON inheritance is relative to the declaring file, never its name.

Resource paths inside the recipe retain the documented workspace semantics.
Inherited mappings merge recursively; explicit null replaces a prior value.
    """
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("cyclic reasoning configuration inheritance")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("reasoning configuration must be a JSON object")
    parent = payload.pop("inherits", None)
    base = {} if parent is None else read_recipe(path.parent / parent, (*_parents, path))
    def merge(left, right):
        output = dict(left)
        for key, value in right.items():
            output[key] = merge(output[key], value) if isinstance(output.get(key), dict) and isinstance(value, dict) else value
        return output
    return merge(base, payload)


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
