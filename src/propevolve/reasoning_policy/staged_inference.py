"""Configured staged reasoning policy using an injected model backend."""
import json
from pathlib import Path

import numpy as np

from ..decision import Action
from .backend import load_mlx_backend
from .integrity import file_digest
from .model_config import read_recipe, verify_adapter_base, validate_model_settings, resolve_model_resources
from .staged_policy import staged_forward, select_legal_action
from .staged_queries import prepare_staged_queries
from .projector import projector_prefix_tokens


def seal_staged_weights(destination, settings):
    """Seal the executable adapter/projector pair, excluding transient snapshots."""
    destination = Path(destination)
    files = {name: file_digest(destination / name)
             for name in ("adapters.safetensors", "projector.safetensors")
             if (destination / name).is_file()}
    if "projector.safetensors" not in files:
        raise ValueError("staged policy requires saved market projector weights")
    metadata = {**settings, "architecture": "staged_reasoning_v1", "weight_files": files}
    (destination / "adapter_config.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))


class StagedReasoningPolicy:
    requires_specialists = False

    def __init__(self, backend, tokenizer, settings):
        self.backend, self.tokenizer, self.settings = backend, tokenizer, settings
        if settings["selection"] != "hierarchical_greedy":
            raise ValueError("unsupported staged action selection")
        if settings["projector"].get("state_fields"):
            raise ValueError("market interpretation projector cannot consume trade/account state")

    @classmethod
    def from_config(cls, path, *, root=None, backend_factory=load_mlx_backend):
        path = Path(path) if root is None else Path(root) / path
        settings = resolve_model_resources(read_recipe(path), root=root)
        return cls.from_settings(settings, backend_factory=backend_factory)

    @classmethod
    def from_settings(cls, settings, *, backend_factory=load_mlx_backend):
        """Load an explicit resolved contract, including a resumed RL artifact."""
        validate_model_settings(settings)
        if "staged_policy" not in settings:
            raise ValueError("staged interpretation and assessment configuration is required")
        adapter = settings["adapter_path"]
        if adapter is not None:
            metadata = json.loads((Path(adapter) / "adapter_config.json").read_text())
            if metadata.get("architecture") != "staged_reasoning_v1":
                raise ValueError("adapter is not a staged reasoning policy")
            verify_adapter_base(settings["model"], adapter)
            for key in ("projector", "staged_policy", "selection", "chat_template_kwargs"):
                if metadata.get(key) != settings[key]:
                    raise ValueError(f"staged adapter contract differs at {key}")
            for filename, digest in metadata["weight_files"].items():
                if file_digest(Path(adapter) / filename) != digest:
                    raise ValueError("staged adapter weight identity mismatch")
        backend, tokenizer = backend_factory(settings)
        return cls(backend, tokenizer, settings)

    def prepare(self, context, legal_actions):
        import mlx.core as mx
        if context.embeddings is None or not context.available.any():
            raise ValueError("staged inference requires available causal embeddings")
        fields = self.settings["staged_policy"]["state_fields"]
        if set(context.fields) != set(fields):
            raise ValueError("staged causal state schema differs from context")
        latest = context.values[context.available][-1]
        state = {name: float(latest[context.fields.index(name)]) for name in fields}
        queries = prepare_staged_queries(state, self.settings["staged_policy"], self.tokenizer,
            max_seq_length=self.settings["max_seq_length"] - projector_prefix_tokens(self.settings["projector"]),
            chat_template_kwargs=self.settings["chat_template_kwargs"])
        return {"embeddings": mx.array(context.embeddings[None]),
            "available": mx.array(context.available[None]),
            "market_query": {k: mx.array(v) for k, v in queries["market_query"].items()},
            "assessment_query": {k: mx.array(v) for k, v in queries["assessment_query"].items()},
            "legal_actions": [tuple(Action(a) for a in legal_actions)]}

    def assess(self, context, legal_actions):
        import mlx.core as mx
        batch = self.prepare(context, legal_actions)
        result = staged_forward(self.backend, **batch)
        mx.eval(result)
        interpretation = mx.sigmoid(result["interpretation_scores"])[0].tolist()
        assessment = result["assessment_scores"][0].tolist()
        log_probs = result["log_probs"][0].tolist()
        if not np.isfinite(interpretation + assessment + log_probs).all():
            raise ValueError("nonfinite staged reasoning prediction")
        actions = batch["legal_actions"][0]
        action = select_legal_action(assessment, actions)
        return {"action": action,
            "interpretation": dict(zip([c["name"] for c in self.settings["staged_policy"]["market"]["channels"]], interpretation)),
            "assessment": dict(zip(("entry", "direction", "management"), assessment)),
            "log_probs": dict(zip([a.name for a in actions], log_probs)),
            "probabilities": dict(zip([a.name for a in actions], np.exp(log_probs).tolist()))}

    def decide(self, context, legal_actions):
        result = self.assess(context, legal_actions)
        return result["action"], result["log_probs"]

    def save(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        self.backend.save_weights(destination)
        seal_staged_weights(destination, self.settings)
