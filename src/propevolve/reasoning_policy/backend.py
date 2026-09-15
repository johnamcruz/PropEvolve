"""Injected reasoning-model operations; no model family or identifier selection."""
from typing import Any, Protocol


class ReasoningBackend(Protocol):
    """Differentiable model operations required by the shared staged policy."""

    def embed_tokens(self, tokens: Any) -> Any: ...
    def hidden_states(self, tokens: Any, embeddings: Any) -> Any: ...
    def output_logits(self, hidden: Any) -> Any: ...
    def market_prefix(self, embeddings: Any, available: Any, state: Any) -> Any: ...
    def save_weights(self, destination: Any) -> None: ...


class MLXReasoningBackend:
    """Adapt a compatible MLX-LM model loaded by configuration.

    Model architecture details live here, not in training or trade decisions.
    An incompatible model needs another adapter, not a parallel trading pipeline.
    """

    def __init__(self, model):
        self.model = model

    def embed_tokens(self, tokens):
        return self.model.model.embed_tokens(tokens)

    def hidden_states(self, tokens, embeddings):
        return self.model.model(tokens, input_embeddings=embeddings)

    def output_logits(self, hidden):
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)

    def market_prefix(self, embeddings, available, state):
        return self.model.market_projector(embeddings, available, state)

    def save_weights(self, destination):
        from .projector import export_policy_weights
        export_policy_weights(self.model, destination)


def load_mlx_backend(settings):
    """Load an explicit configured model; native MLX-LM selects its family."""
    from pathlib import Path
    from mlx_lm import load
    from .projector import attach_projector, restore_projector
    adapter = settings["adapter_path"]
    lora = (adapter if adapter is not None
            and (Path(adapter) / "adapters.safetensors").is_file() else None)
    model, tokenizer = load(settings["model"], adapter_path=lora)
    attach_projector(model, settings["projector"])
    if adapter is not None:
        restore_projector(model, adapter)
    model.eval()
    return MLXReasoningBackend(model), tokenizer
