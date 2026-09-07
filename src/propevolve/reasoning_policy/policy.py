"""Finite-action MLX-LM inference using the same prompt as supervised training."""

import numpy as np

from ..decision import Action
from .dataset import context_messages
from .model_config import read_model_settings, validate_model_settings, verify_adapter_base, template_options
from .tokenization import encode_completion


class MLXActionPolicy:
    """Score legal action completions; never execute unvalidated generated text.

This first policy is action-only, using a reasoning-capable backbone. It does
not claim that free-form generated chain-of-thought has been trained or tested.
Scores are sequence log likelihoods, not C51 Q values or pass probabilities.
    """

    requires_specialists = True

    def __init__(self, model, tokenizer, *, max_seq_length: int, chat_template_kwargs=None,
                 input_mode="specialists", projector=None):
        if type(max_seq_length) is not int or max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.input_mode = input_mode
        self.projector_config = projector
        self.requires_specialists = input_mode == "specialists"
        self.chat_template_kwargs = template_options(chat_template_kwargs)
        self.model.eval()

    @classmethod
    def load(cls, model_path, *, adapter_path, max_seq_length, chat_template_kwargs=None,
             input_mode="specialists", projector=None):
        validate_model_settings({"model": str(model_path), "adapter_path":
                                 None if adapter_path is None else str(adapter_path),
                                 "max_seq_length": max_seq_length, "input_mode": input_mode, "projector": projector})
        verify_adapter_base(str(model_path), adapter_path)
        if adapter_path is not None:
            import json
            from pathlib import Path
            metadata = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            if (metadata.get("input_mode", "specialists") != input_mode
                    or metadata.get("projector") != projector):
                raise ValueError("adapter input/projector contract differs from configured policy")
        from mlx_lm import load
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        if input_mode == "embeddings":
            from .projector import attach_projector, restore_projector
            attach_projector(model, projector)
            if adapter_path is not None:
                restore_projector(model, adapter_path)
        return cls(model, tokenizer, max_seq_length=max_seq_length,
                   chat_template_kwargs=chat_template_kwargs, input_mode=input_mode, projector=projector)

    @classmethod
    def from_config(cls, path, *, root=None):
        """Load from any JSON filename, including the existing SFT recipe.

        The caller still uses decide(context, legal_actions), regardless of
        the selected MLX-LM backbone. A null adapter selects the base model.
        """
        settings = read_model_settings(path, root=root)
        return cls.load(settings["model"], adapter_path=settings["adapter_path"],
                        max_seq_length=settings["max_seq_length"],
                        chat_template_kwargs=settings["chat_template_kwargs"],
                        input_mode=settings["input_mode"], projector=settings["projector"])

    def decide(self, context, legal_actions):
        actions = tuple(sorted({Action(action) for action in legal_actions}, key=int))
        if not actions:
            raise ValueError("cannot decide without legal actions")
        from .dataset import embedding_payload
        scores = self.completion_scores(context_messages(context, actions), [a.name for a in actions],
            market_context=embedding_payload(context) or None)
        selected = actions[int(np.argmax(list(scores.values()))) ]
        return selected, scores

    def completion_scores(self, messages, completions, *, market_context=None):
        """Public frozen-batch diagnostic for actual supervised answer likelihood."""
        import mlx.core as mx
        completions = tuple(completions)
        tokens = self.tokenize_completions(messages, completions, market_context=market_context)
        scores = sequence_scores(self.model, tokens)
        mx.eval(scores)
        values = np.asarray(scores.tolist())
        if not np.isfinite(values).all():
            raise ValueError("nonfinite action scores")
        return dict(zip(completions, values.tolist()))

    def tokenize_completions(self, messages, completions, *, market_context=None):
        """Shared token boundary for inference and differentiable RL updates."""
        completions = tuple(completions)
        if not completions or len(set(completions)) != len(completions):
            raise ValueError("completions must be nonempty and unique")
        if (market_context is not None) != (self.input_mode == "embeddings"):
            raise ValueError("policy input mode and causal embeddings disagree")
        reserved = 0 if self.projector_config is None else self.projector_config["market_tokens"]
        encoded = tuple(encode_completion(self.tokenizer, messages, completion,
            max_seq_length=self.max_seq_length - reserved, chat_template_kwargs=self.chat_template_kwargs)
            for completion in completions)
        if market_context is None:
            return encoded
        embeddings = np.asarray(market_context["market_embeddings"], np.float32)
        available = np.asarray(market_context["market_available"], bool)
        expected = (self.projector_config["context_steps"], self.projector_config["embedding_dim"])
        if (embeddings.shape != expected or available.shape != expected[:1] or not available.any()
                or not np.isfinite(embeddings).all()):
            raise ValueError("invalid causal market embedding window")
        return tuple((*item, embeddings, available) for item in encoded)


def sequence_scores(model, tokenized):
    """Differentiable action log likelihoods; no detach before RL gradients."""
    import mlx.core as mx
    scores = []
    for item in tokenized:
        full, prefix_length = item[:2]
        inputs = mx.array([full[:-1]])
        if len(item) == 4:
            from .projector import market_logits
            logits = market_logits(model, inputs, mx.array(item[2][None]), mx.array(item[3][None]))
        else:
            logits = model(inputs)
        logits = logits[:, prefix_length - 1:, :].astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(full[prefix_length:])[None, :, None]
        scores.append(mx.take_along_axis(log_probs, targets, axis=-1).sum())
    return mx.stack(scores)
