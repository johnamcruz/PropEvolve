"""Optional continuous FFM tokens. MLX is imported only on explicit model use."""
import inspect
from pathlib import Path
import numpy as np


def validate_projector(config):
    if not isinstance(config, dict) or any(type(config.get(key)) is not int or config[key] < 1
            for key in ("embedding_dim", "context_steps", "market_tokens")):
        raise ValueError("projector requires positive dimensions from the authenticated embedding contract")
    if config["market_tokens"] > config["context_steps"]:
        raise ValueError("market token count exceeds context window")
    if config.get("temporal_encoding") not in {"pooled_levels", "latest_plus_deltas"}:
        raise ValueError("projector requires a configured temporal encoding")
    fields = config.get("state_fields", [])
    scales = config.get("state_scales", [])
    if (not isinstance(fields, list) or not isinstance(scales, list)
            or len(fields) != len(scales) or len(set(fields)) != len(fields)
            or any(not isinstance(field, str) or not field for field in fields)
            or any(isinstance(scale, bool) or not isinstance(scale, (int, float))
                   or not np.isfinite(scale) or scale <= 0
                   for scale in scales)):
        raise ValueError("projector causal state contract is invalid")


def projector_prefix_tokens(config):
    """Number of continuous prefix tokens reserved by one projector contract."""
    validate_projector(config)
    return config["market_tokens"] + bool(config.get("state_fields"))


def state_extension_of(parent, child):
    """Whether a child adds only causal state to an existing market projector."""
    if not isinstance(parent, dict) or not isinstance(child, dict):
        return False
    parent_base = {key: value for key, value in parent.items()
                   if key not in {"state_fields", "state_scales"}}
    child_base = {key: value for key, value in child.items()
                  if key not in {"state_fields", "state_scales"}}
    return (parent_base == child_base and not parent.get("state_fields")
            and bool(child.get("state_fields")))


def pooling_weights(available, market_tokens):
    """Ordered temporal bins; unavailable prefix bars contribute zero mass."""
    available = np.asarray(available, dtype=bool)
    if available.ndim != 2 or not available.any(axis=1).all():
        raise ValueError("projector requires available completed history")
    membership = np.zeros((market_tokens, available.shape[1]), np.float32)
    for index, positions in enumerate(np.array_split(np.arange(available.shape[1]), market_tokens)):
        membership[index, positions] = 1
    weights = membership[None, :, :] * available[:, None, :]
    return weights / np.maximum(weights.sum(axis=-1, keepdims=True), 1)


def temporal_features(pooled, encoding, *, xp):
    """Expose current state plus completed-history changes without future data."""
    if encoding == "pooled_levels":
        return pooled
    if encoding != "latest_plus_deltas" or pooled.ndim != 3 or pooled.shape[1] < 2:
        raise ValueError("invalid temporal projector encoding")
    return xp.concatenate([pooled[:, -1:], pooled[:, 1:] - pooled[:, :-1]], axis=1)


def attach_projector(model, config):
    import mlx.core as mx
    import mlx.nn as nn
    validate_projector(config)
    if "input_embeddings" not in inspect.signature(model.__call__).parameters:
        raise ValueError("backbone does not support continuous input_embeddings")
    if not hasattr(getattr(model, "model", None), "embed_tokens"):
        raise ValueError("backbone does not expose token embedding interface")
    width = model.model.embed_tokens(mx.array([[0]])).shape[-1]
    class MarketProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(config["embedding_dim"], width, bias=False)
            self.state_projection = (nn.Linear(len(config.get("state_fields", [])), width,
                                               bias=False)
                                     if config.get("state_fields") else None)
            self.context_steps = config["context_steps"]
            self.market_tokens = config["market_tokens"]
        def __call__(self, embeddings, available, causal_state=None):
            if embeddings.shape[1:] != (config["context_steps"], config["embedding_dim"]):
                raise ValueError("projector embedding contract mismatch")
            membership = np.zeros((config["market_tokens"], config["context_steps"]), np.float32)
            for index, positions in enumerate(np.array_split(np.arange(config["context_steps"]), config["market_tokens"])):
                membership[index, positions] = 1
            weights = mx.array(membership)[None, :, :] * available[:, None, :]
            weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1)
            clean = mx.where(available[:, :, None], embeddings, 0.)
            pooled = weights @ clean
            features = temporal_features(pooled, config["temporal_encoding"], xp=mx)
            market = self.projection(features)
            if self.state_projection is None:
                if causal_state is not None and causal_state.shape[-1] != 0:
                    raise ValueError("projector received undeclared causal state")
                return market
            expected = len(config["state_fields"])
            if causal_state is None or causal_state.shape != (embeddings.shape[0], expected):
                raise ValueError("projector causal state shape mismatch")
            scales = mx.array(config["state_scales"], dtype=causal_state.dtype)
            state = self.state_projection(causal_state / scales)
            return mx.concatenate([market, state[:, None, :]], axis=1)
    model.market_projector = MarketProjector()


def market_logits(model, tokens, embeddings, available, causal_state=None):
    import mlx.core as mx
    if not hasattr(model, "market_projector"):
        raise ValueError("teacher-free policy is missing its trained projector")
    prefix = model.market_projector(embeddings, available, causal_state)
    if prefix.shape[0] == 1 and tokens.shape[0] != 1:
        prefix = mx.broadcast_to(prefix, (tokens.shape[0], *prefix.shape[1:]))
    joined = mx.concatenate([prefix, model.model.embed_tokens(tokens)], axis=1)
    return model(tokens, input_embeddings=joined)[:, prefix.shape[1]:, :]


def export_policy_weights(model, destination):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.parameters()))
    projector = {name: value for name, value in weights.items() if name.startswith("market_projector.")}
    base = {name: value for name, value in weights.items()
            if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}}
    if base:
        mx.save_safetensors(str(Path(destination) / "adapters.safetensors"), base)
    if projector:
        mx.save_safetensors(str(Path(destination) / "projector.safetensors"), projector)


def restore_projector(model, directory, *, allow_state_extension=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    path = Path(directory) / "projector.safetensors"
    weights = mx.load(str(path))
    expected = {key: value for key, value in tree_flatten(model.parameters()) if key.startswith("market_projector.")}
    missing = set(expected) - set(weights)
    permitted = ({key for key in expected if key.startswith(
                  "market_projector.state_projection.")}
                 if allow_state_extension else set())
    if (set(weights) - set(expected) or missing - permitted
            or any(weights[key].shape != expected[key].shape for key in weights)):
        raise ValueError("saved projector does not match configured model/embedding dimensions")
    model.load_weights(list(weights.items()), strict=False)
