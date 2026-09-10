"""Shared deterministic decision interface; learner algorithms remain separate.

Adapters own model state, not simulator state. Call reset at every episode.
Resource paths are relative to the explicitly supplied workspace root.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .decision import Action


@dataclass(frozen=True)
class PolicyInput:
    observation: np.ndarray
    legal_actions: tuple[Action, ...]
    context: Any = None


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    scores: dict[str, float]
    score_type: str


class TradingPolicy(ABC):
    requires_context = True
    @property
    @abstractmethod
    def requires_specialists(self) -> bool:
        """Whether this adapter requires the causal specialist context."""

    @abstractmethod
    def reset(self) -> None:
        """Discard recurrent state at an episode boundary."""

    @abstractmethod
    def decide(self, inputs: PolicyInput) -> PolicyDecision:
        """Return one legal action; scores are not interchangeable across kinds."""


def _decision(action, scores, kind, legal):
    if not legal or action not in legal:
        raise ValueError("policy requested an illegal action")
    if set(scores) != {item.name for item in legal} or not all(
            np.isfinite(value) for value in scores.values()):
        raise ValueError("policy scores must cover exactly the legal actions and be finite")
    return PolicyDecision(action, scores, kind)


class ReasoningPolicy(TradingPolicy):
    @property
    def requires_specialists(self):
        return self.policy.requires_specialists

    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        # The caller supplies an episode-local rolling causal context.
        pass

    def decide(self, inputs):
        if inputs.context is None:
            raise ValueError("reasoning policy requires causal context")
        action, scores = self.policy.decide(inputs.context, inputs.legal_actions)
        return _decision(action, scores, "log_likelihood", inputs.legal_actions)


def load_policy(path, *, root):
    """Select an adapter through JSON, without importing the unused backend."""
    from .reasoning_policy.model_config import read_recipe
    config = read_recipe(path)
    root = Path(root)
    kind = config.get("kind")
    if kind == "reasoning":
        from .reasoning_policy.policy import MLXActionPolicy
        return ReasoningPolicy(MLXActionPolicy.from_config(root / config["model_config"], root=root))
    raise ValueError(f"unknown policy kind: {kind!r}")
