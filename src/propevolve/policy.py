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


class R2D2Policy(TradingPolicy):
    requires_specialists = False
    requires_context = False

    def __init__(self, agent, *, recurrent_horizon):
        if type(recurrent_horizon) is not int or recurrent_horizon < 1:
            raise ValueError("recurrent_horizon must be positive")
        self.agent = agent
        self.recurrent_horizon = recurrent_horizon
        self.reset()

    def reset(self):
        self._hidden = None
        self._steps = 0

    def decide(self, inputs):
        if not inputs.legal_actions:
            raise ValueError("policy requires legal actions")
        if self._steps % self.recurrent_horizon == 0:
            self._hidden = None
        action, self._hidden, values = self.agent.select_action(inputs.observation,
            hidden=self._hidden, valid_actions=inputs.legal_actions, epsilon=0,
            return_action_values=True)
        self._steps += 1
        return _decision(action, {item.name: float(values[int(item)])
            for item in inputs.legal_actions}, "q_value", inputs.legal_actions)


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
    if kind == "r2d2":
        from .agent import RecurrentC51Agent
        agent, _ = RecurrentC51Agent.load(root / config["checkpoint"],
            device=config["device"], learner_backend_override=config["learner_backend"])
        return R2D2Policy(agent, recurrent_horizon=config["recurrent_horizon"])
    if kind == "reasoning":
        from .reasoning_policy.policy import MLXActionPolicy
        return ReasoningPolicy(MLXActionPolicy.from_config(root / config["model_config"], root=root))
    raise ValueError(f"unknown policy kind: {kind!r}")
