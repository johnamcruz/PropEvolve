"""Same causal embedding window in collection and teacher-free evaluation."""
import numpy as np
import pytest
from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.inputs import observe_context
from propevolve.reasoning_policy.dataset import context_messages
import json
from test_reasoning_challenger_e2e import environment


def test_embedding_context_uses_no_teacher_and_never_serializes_latents_as_text():
    env = environment()
    obs, info = env.reset(options={"ticker": "NQ", "start": 0})
    history = RollingContext(ContextConfig(3, ("account.realized_pnl_norm",), input_mode="embeddings"))
    class ForbiddenSources:
        def __iter__(self):
            raise AssertionError("teacher lookup in teacher-free observation")
    observe_context(history, env, obs, ticker="NQ", row=0, sources=ForbiddenSources())
    window = history.snapshot()
    np.testing.assert_array_equal(window.embeddings[-1], [1., 1.])
    np.testing.assert_array_equal(window.available, [False, False, True])
    assert "market_embeddings" not in str(context_messages(window, info["valid_actions"]))
    env.markets["NQ"].embeddings[1:] = 999
    np.testing.assert_array_equal(window.embeddings[-1], [1., 1.])
    history.reset()
    assert not history.snapshot().available.any()


def test_embedding_context_keeps_full_market_window_but_only_current_account_text():
    env = environment()
    obs, info = env.reset(options={"ticker": "NQ", "start": 0})
    history = RollingContext(ContextConfig(
        3, ("account.realized_pnl_norm",), input_mode="embeddings", text_steps=1))
    for row in range(3):
        observe_context(history, env, obs, ticker="NQ", row=row, sources=())
    window = history.snapshot()
    prompt = json.loads(context_messages(window, info["valid_actions"])[1]["content"])
    assert window.embeddings.shape[0] == 3
    assert len(prompt["history_oldest_first"]) == 1


def test_teacher_fields_are_rejected_in_embedding_mode():
    with pytest.raises(ValueError, match="teacher"):
        ContextConfig(3, ("trend.long_probability",), input_mode="embeddings")


def test_teacher_free_reasoning_adapter_trades_in_shared_simulator_without_lookups():
    from propevolve.policy import ReasoningPolicy
    from propevolve.decision import Action
    from propevolve.reasoning_policy.evaluation import evaluate_policy
    class ExternalRuntime:
        requires_specialists = False
        def decide(self, context, legal_actions):
            assert context.embeddings is not None
            action = Action.HOLD if Action.HOLD in legal_actions else Action.ENTER_LONG_1
            return action, {item.name: float(item == action) for item in legal_actions}
    class ForbiddenSources:
        def __iter__(self):
            raise AssertionError("teacher used by teacher-free evaluation")
    result = evaluate_policy(ReasoningPolicy(ExternalRuntime()), environment(),
        episodes=[{"ticker": "NQ", "start": 0}], sources=ForbiddenSources(), max_steps=8,
        context_config=ContextConfig(3, ("account.realized_pnl_norm",), input_mode="embeddings"))
    assert result["pass_rate"] == 1.
    assert result["teacher_free"] is True
