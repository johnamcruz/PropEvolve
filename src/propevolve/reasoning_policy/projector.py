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
            self.context_steps = config["context_steps"]
            self.market_tokens = config["market_tokens"]
        def __call__(self, embeddings, available):
            if embeddings.shape[1:] != (config["context_steps"], config["embedding_dim"]):
                raise ValueError("projector embedding contract mismatch")
            membership = np.zeros((config["market_tokens"], config["context_steps"]), np.float32)
            for index, positions in enumerate(np.array_split(np.arange(config["context_steps"]), config["market_tokens"])):
                membership[index, positions] = 1
            weights = mx.array(membership)[None, :, :] * available[:, None, :]
            weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1)
            clean = mx.where(available[:, :, None], embeddings, 0.)
            return self.projection(weights @ clean)
    model.market_projector = MarketProjector()


def market_logits(model, tokens, embeddings, available):
    import mlx.core as mx
    if not hasattr(model, "market_projector"):
        raise ValueError("teacher-free policy is missing its trained projector")
    prefix = model.market_projector(embeddings, available)
    if prefix.shape[0] == 1 and tokens.shape[0] != 1:
        prefix = mx.broadcast_to(prefix, (tokens.shape[0], *prefix.shape[1:]))
    joined = mx.concatenate([prefix, model.model.embed_tokens(tokens)], axis=1)
    return model(tokens, input_embeddings=joined)[:, prefix.shape[1]:, :]


def export_policy_weights(model, destination):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    weights = dict(tree_flatten(model.trainable_parameters()))
    projector = {name: value for name, value in weights.items() if name.startswith("market_projector.")}
    base = {name: value for name, value in weights.items() if name not in projector}
    mx.save_safetensors(str(Path(destination) / "adapters.safetensors"), base)
    if projector:
        mx.save_safetensors(str(Path(destination) / "projector.safetensors"), projector)


def restore_projector(model, directory):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    path = Path(directory) / "projector.safetensors"
    weights = mx.load(str(path))
    expected = {key: value for key, value in tree_flatten(model.parameters()) if key.startswith("market_projector.")}
    if set(weights) != set(expected) or any(weights[key].shape != expected[key].shape for key in weights):
        raise ValueError("saved projector does not match configured model/embedding dimensions")
    model.load_weights(list(weights.items()), strict=False)
