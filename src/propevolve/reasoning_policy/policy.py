"""Finite-action MLX-LM inference using the same prompt as supervised training."""

import numpy as np

from ..decision import Action
from .dataset import context_messages


class MLXActionPolicy:
    """Score legal action completions; never execute unvalidated generated text.

This first policy is action-only, using a reasoning-capable backbone. It does
not claim that free-form generated chain-of-thought has been trained or tested.
Scores are sequence log likelihoods, not C51 Q values or pass probabilities.
    """

    requires_specialists = True

    def __init__(self, model, tokenizer, *, max_seq_length: int):
        if type(max_seq_length) is not int or max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model.eval()

    @classmethod
    def load(cls, model_path, *, adapter_path, max_seq_length):
        from mlx_lm import load
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        return cls(model, tokenizer, max_seq_length=max_seq_length)

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
        if not completions or len(set(completions)) != len(completions):
            raise ValueError("completions must be nonempty and unique")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        prefix = self.tokenizer.encode(prompt)
        if not prefix:
            raise ValueError("empty tokenized prompt")
        scores = []
        for completion in completions:
            full = self.tokenizer.encode(prompt + completion + self.tokenizer.eos_token)
            if full[:len(prefix)] != prefix:
                raise ValueError("tokenizer changes the prompt/action boundary")
            if len(full) > self.max_seq_length:
                raise ValueError("inference token budget exceeded; refusing truncation")
            inputs = mx.array([full[:-1]])
            logits = self.model(inputs)[:, len(prefix) - 1:, :].astype(mx.float32)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            targets = mx.array(full[len(prefix):])[None, :, None]
            score = mx.take_along_axis(log_probs, targets, axis=-1).sum()
            mx.eval(score)
            scores.append(float(score.item()))
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite action scores")
        return dict(zip(completions, scores))
