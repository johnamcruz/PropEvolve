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

    def __init__(self, model, tokenizer, *, max_seq_length: int, chat_template_kwargs=None):
        if type(max_seq_length) is not int or max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chat_template_kwargs = template_options(chat_template_kwargs)
        self.model.eval()

    @classmethod
    def load(cls, model_path, *, adapter_path, max_seq_length, chat_template_kwargs=None):
        validate_model_settings({"model": str(model_path), "adapter_path":
                                 None if adapter_path is None else str(adapter_path),
                                 "max_seq_length": max_seq_length})
        verify_adapter_base(str(model_path), adapter_path)
        from mlx_lm import load
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        return cls(model, tokenizer, max_seq_length=max_seq_length,
                   chat_template_kwargs=chat_template_kwargs)

    @classmethod
    def from_config(cls, path):
        """Load from any JSON filename, including the existing SFT recipe.

        The caller still uses decide(context, legal_actions), regardless of
        the selected MLX-LM backbone. A null adapter selects the base model.
        """
        settings = read_model_settings(path)
        return cls.load(settings["model"], adapter_path=settings["adapter_path"],
                        max_seq_length=settings["max_seq_length"],
                        chat_template_kwargs=settings["chat_template_kwargs"])

    def decide(self, context, legal_actions):
        actions = tuple(sorted({Action(action) for action in legal_actions}, key=int))
        if not actions:
            raise ValueError("cannot decide without legal actions")
        scores = self.completion_scores(context_messages(context, actions), [a.name for a in actions])
        selected = actions[int(np.argmax(list(scores.values()))) ]
        return selected, scores

    def completion_scores(self, messages, completions):
        """Public frozen-batch diagnostic for actual supervised answer likelihood."""
        import mlx.core as mx
        completions = tuple(completions)
        tokens = self.tokenize_completions(messages, completions)
        scores = sequence_scores(self.model, tokens)
        mx.eval(scores)
        values = np.asarray(scores.tolist())
        if not np.isfinite(values).all():
            raise ValueError("nonfinite action scores")
        return dict(zip(completions, values.tolist()))

    def tokenize_completions(self, messages, completions):
        """Shared token boundary for inference and differentiable RL updates."""
        completions = tuple(completions)
        if not completions or len(set(completions)) != len(completions):
            raise ValueError("completions must be nonempty and unique")
        return tuple(encode_completion(self.tokenizer, messages, completion,
            max_seq_length=self.max_seq_length, chat_template_kwargs=self.chat_template_kwargs)
            for completion in completions)


def sequence_scores(model, tokenized):
    """Differentiable action log likelihoods; no detach before RL gradients."""
    import mlx.core as mx
    scores = []
    for full, prefix_length in tokenized:
        logits = model(mx.array([full[:-1]]))[:, prefix_length - 1:, :].astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(full[prefix_length:])[None, :, None]
        scores.append(mx.take_along_axis(log_probs, targets, axis=-1).sum())
    return mx.stack(scores)
