"""Configured local model -> causal context -> inspectable staged decision."""
import json
import numpy as np
import pytest


def test_configured_model_reload_preserves_teacher_free_staged_decision(tmp_path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from test_reasoning_local_qlora_e2e import tiny_quantized_qwen
    from test_staged_queries import settings
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    from propevolve.reasoning_policy.staged_inference import StagedReasoningPolicy
    from propevolve.decision import Action

    model = tiny_quantized_qwen(tmp_path / "model", tied=True)
    recipe = {"model": str(model), "adapter_path": None,
        "max_seq_length": 2048, "chat_template_kwargs": {"enable_thinking": False},
        "projector": {"embedding_dim": 2, "context_steps": 4, "market_tokens": 2,
                      "temporal_encoding": "latest_plus_deltas"},
        "staged_policy": settings(), "selection": "hierarchical_greedy"}
    recipe["staged_policy"]["state_fields"] = ["trade.current_r"]
    config = tmp_path / "policy.json"
    config.write_text(json.dumps(recipe))
    history = RollingContext(ContextConfig(4, ("trade.current_r",), input_mode="embeddings"))
    history.append(100, {"trade.current_r": 1.}, embedding=np.ones(2))
    legal = (Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)
    first = StagedReasoningPolicy.from_config(config)
    before = first.assess(history.snapshot(), legal)
    artifact = tmp_path / "saved"
    first.save(artifact)
    recipe["adapter_path"] = str(artifact)
    config.write_text(json.dumps(recipe))
    second = StagedReasoningPolicy.from_config(config)
    after = second.assess(history.snapshot(), legal)
    assert before == after
    assert set(before["interpretation"]) == {"expansion.long", "expansion.short"}
    assert set(before["assessment"]) == {"entry", "direction", "management"}
    assert before["action"] in legal
    assert sum(before["probabilities"].values()) == pytest.approx(1., abs=1e-6)
    assert first.requires_specialists is False
    # Production JSON loading must preserve the very same staged computation.
    from propevolve.policy import PolicyInput, load_policy
    wrapper = tmp_path / "trading-policy.json"
    wrapper.write_text(json.dumps({"kind": "reasoning", "model_config": str(config)}))
    production = load_policy(wrapper, root=tmp_path)
    decision = production.decide(PolicyInput(np.zeros(1), legal, history.snapshot()))
    assert decision.action == before["action"]
    assert decision.scores == before["log_probs"]
    assert decision.score_type == "log_probability"
    assert decision.interpretation == before["interpretation"]
    assert decision.assessment == before["assessment"]
    from test_reasoning_challenger_e2e import environment
    from propevolve.reasoning_policy.evaluation import evaluate_policy
    env = environment()
    class ForbiddenTeachers:
        def __iter__(self):
            raise AssertionError("teacher lookup during staged inference")
    traces = []
    report = evaluate_policy(production, env,
        episodes=[{"ticker": "NQ", "start": 0}],
        context_config=history.config, sources=ForbiddenTeachers(), max_steps=8,
        on_decision=traces.append)
    assert report["teacher_free"] is True
    assert traces
    for trace in traces:
        assert set(trace["market_interpretation"]) == {"expansion.long", "expansion.short"}
        assert set(trace["trade_assessment"]) == {"entry", "direction", "management"}
        assert sum(trace["action_probabilities"].values()) == pytest.approx(1., abs=1e-6)
        assert trace["requested_action"] in trace["legal_actions"]
