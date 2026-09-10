"""JSON-selected policies preserve the production decision contract.

Execution deferred until the active Volume training is finished.
"""
import json

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.policy import PolicyInput, TradingPolicy, load_policy
def test_unknown_policy_kind_fails_before_loading_a_model(tmp_path):
    recipe = tmp_path / "anything.json"
    recipe.write_text('{"kind": "typo"}')
    with pytest.raises(ValueError, match="policy kind"):
        load_policy(recipe, root=tmp_path)


def test_reasoning_adapter_preserves_legal_completion_scores():
    from propevolve.policy import ReasoningPolicy
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    # External inference-runtime stand-in; simulator and adapter are real.
    class Runtime:
        requires_specialists = True
        def decide(self, context, legal_actions):
            scores = {action.name: -float(int(action) + 1) for action in legal_actions}
            return max(legal_actions, key=lambda action: scores[action.name]), scores
    context = RollingContext(ContextConfig(2, ("account.realized_pnl_norm",)))
    context.append(1, {"account.realized_pnl_norm": 0.0})
    policy = ReasoningPolicy(Runtime())
    assert isinstance(policy, TradingPolicy)
    decision = policy.decide(PolicyInput(np.ones(4), (Action.WAIT,), context.snapshot()))
    assert decision.action == Action.WAIT
    assert decision.score_type == "log_likelihood"
    assert policy.requires_specialists is True
    with pytest.raises(ValueError, match="causal context"):
        policy.decide(PolicyInput(np.ones(4), (Action.WAIT,)))
