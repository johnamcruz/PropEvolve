"""JSON-selected policies preserve the production decision contract.

Execution deferred until the active Volume training is finished.
"""
import json

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.policy import PolicyInput, TradingPolicy, load_policy
from test_agent import _agent


def test_json_r2d2_checkpoint_preserves_actions_values_and_episode_reset(tmp_path):
    agent = _agent(4)
    agent.save(tmp_path / "checkpoint.pt", manifest={})
    recipe = tmp_path / "arbitrary-name.json"
    recipe.write_text(json.dumps({"kind": "r2d2", "checkpoint": "checkpoint.pt",
        "device": "cpu", "learner_backend": "pytorch", "recurrent_horizon": 2}))
    policy = load_policy(recipe, root=tmp_path)
    assert isinstance(policy, TradingPolicy)
    legal = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    observation = np.ones(4, dtype=np.float32)
    hidden = None
    for step in range(5):
        if step % 2 == 0:
            hidden = None
        action, hidden, values = agent.select_action(observation, hidden=hidden,
            valid_actions=legal, epsilon=0, return_action_values=True)
        actual = policy.decide(PolicyInput(observation, legal))
        assert actual.action == action
        assert actual.score_type == "q_value"
        for item in legal:
            assert actual.scores[item.name] == pytest.approx(float(values[int(item)]))
    policy.reset()
    first = policy.decide(PolicyInput(observation, legal))
    policy.reset()
    assert policy.decide(PolicyInput(observation, legal)) == first
    assert policy.requires_specialists is False


def test_unknown_policy_kind_fails_before_loading_a_model(tmp_path):
    recipe = tmp_path / "anything.json"
    recipe.write_text('{"kind": "typo"}')
    with pytest.raises(ValueError, match="policy kind"):
        load_policy(recipe, root=tmp_path)


def test_r2d2_uses_same_environment_without_any_specialist_inputs():
    from propevolve.policy import R2D2Policy
    from propevolve.reasoning_policy.evaluation import evaluate_policy
    from test_reasoning_challenger_e2e import environment
    env = environment()
    observation, _ = env.reset(options={"ticker": "NQ", "start": 0})
    policy = R2D2Policy(_agent(len(observation)), recurrent_horizon=2)
    events = []
    report = evaluate_policy(policy, env,
        episodes=[{"ticker": "NQ", "start": 0}] * 2,
        context_config=None, sources=None, max_steps=8, on_decision=events.append)
    assert report["teacher_free"] is True
    assert all(not row["specialist_inputs_used"] for row in report["episodes"])
    assert report["episodes"][0]["realized_pnl"] == report["episodes"][1]["realized_pnl"]
    assert all(event["score_type"] == "q_value" for event in events)
    assert all(event["requested_action"] in event["legal_actions"] for event in events)


def test_reasoning_adapter_preserves_legal_completion_scores():
    from propevolve.policy import ReasoningPolicy
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    # External inference-runtime stand-in; simulator and adapter are real.
    class Runtime:
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
