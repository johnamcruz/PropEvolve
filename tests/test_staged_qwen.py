"""Native Qwen execution at the staged teacher-free boundaries.

Small randomly initialized Qwen tests mechanics, not market competence.
"""
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")


def test_qwen_assessment_consumes_predicted_interpretation_with_gradients():
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from propevolve.reasoning_policy.staged_policy import assess_interpretation
    from propevolve.reasoning_policy.backend import MLXReasoningBackend

    mx.random.seed(11)
    model = Model(ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=1,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000., head_dim=16, tie_word_embeddings=True))
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    positions = mx.array([[2, 4]])
    task_positions = mx.array([[5, 6, 7]])
    interpretation_ids = mx.array([[9, 10]])
    task_ids = mx.array([[[11, 12], [13, 14], [15, 16]]])

    def assess(scores):
        return assess_interpretation(MLXReasoningBackend(model), tokens, scores, positions,
            interpretation_ids, task_positions, task_ids)

    first = assess(mx.array([[2., -2.]]))
    changed = assess(mx.array([[-2., 2.]]))
    gradients = mx.grad(lambda scores: assess(scores).sum())(mx.zeros((1, 2)))
    mx.eval(first, changed, gradients)
    assert first.shape == (1, 3)
    assert float(mx.max(mx.abs(first - changed))) > 1e-5
    assert bool(mx.all(mx.isfinite(gradients)))
    assert float(mx.sum(mx.abs(gradients))) > 1e-6


def test_market_predictions_need_no_teacher_answers_and_match_training_scores():
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from propevolve.reasoning_policy.projector import attach_projector
    from propevolve.reasoning_policy.market_distillation import (
        predict_market_scores, market_outputs,
    )

    mx.random.seed(11)
    model = Model(ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=1,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000., head_dim=16, tie_word_embeddings=True))
    attach_projector(model, {"embedding_dim": 4, "context_steps": 4,
        "market_tokens": 2, "temporal_encoding": "latest_plus_deltas"})
    model.eval()
    tokens = mx.array([[[1, 2, 3, 4, 5]]])
    embeddings = mx.ones((1, 4, 4))
    available = mx.ones((1, 4), dtype=mx.bool_)
    state = mx.zeros((1, 0))
    positions, label_ids = mx.array([[2, 3]]), mx.array([[6, 7]])
    predicted = predict_market_scores(model, tokens, embeddings, available, state,
        positions, label_ids)
    _, _, supervised = market_outputs(model, tokens, embeddings, available, state,
        positions, mx.array([[.9, .1]]), mx.ones((1, 2)), label_ids)
    _, _, changed_targets = market_outputs(model, tokens, embeddings, available, state,
        positions, mx.array([[.1, .9]]), mx.ones((1, 2)), label_ids)
    mx.eval(predicted, supervised, changed_targets)
    assert predicted.shape == (1, 2)
    assert predicted.tolist() == supervised.tolist() == changed_targets.tolist()


@pytest.mark.parametrize("family,tied", [("qwen3", True), ("llama", False)])
def test_model_family_swap_preserves_staged_differentiable_interface(family, tied):
    from importlib import import_module
    import mlx.nn as nn
    module = import_module(f"mlx_lm.models.{family}")
    from propevolve.decision import Action
    from propevolve.reasoning_policy.projector import attach_projector
    from propevolve.reasoning_policy.staged_policy import staged_forward
    from propevolve.reasoning_policy.backend import MLXReasoningBackend
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from propevolve.reasoning_policy.staged_learning import trade_objective

    mx.random.seed(11)
    model = module.Model(module.ModelArgs(model_type=family, hidden_size=64, num_hidden_layers=1,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000., head_dim=16, tie_word_embeddings=tied))
    model.freeze()
    linear_to_lora_layers(model, 1, {"rank": 2, "scale": 4., "dropout": 0.})
    attach_projector(model, {"embedding_dim": 4, "context_steps": 4,
        "market_tokens": 2, "temporal_encoding": "latest_plus_deltas"})
    model.eval()
    market_query = {"tokens": mx.array([[[1, 2, 3, 4, 5]]]),
        "positions": mx.array([[2, 3]]), "label_ids": mx.array([[9, 10]])}
    assessment_query = {"tokens": mx.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "interpretation_positions": mx.array([[2, 4]]),
        "interpretation_label_ids": mx.array([[9, 10]]),
        "task_positions": mx.array([[5, 6, 7]]),
        "task_label_ids": mx.array([[[11, 12], [13, 14], [15, 16]]])}
    embeddings, available = mx.ones((1, 4, 4)), mx.ones((1, 4), dtype=mx.bool_)
    legal = [(Action.WAIT, Action.ENTER_LONG_1, Action.ENTER_SHORT_1)]

    def forward(m):
        return staged_forward(MLXReasoningBackend(m), embeddings, available,
                              market_query, assessment_query, legal)

    result = forward(model)
    loss, grad = nn.value_and_grad(model, lambda m: -forward(m)["log_probs"][0][1])(model)
    mx.eval(result, loss, grad)
    assert result["interpretation_scores"].shape == (1, 2)
    assert result["assessment_scores"].shape == (1, 3)
    assert float(mx.exp(result["log_probs"][0]).sum()) == pytest.approx(1., abs=1e-6)
    assert float(mx.abs(grad["market_projector"]["projection"]["weight"]).sum()) > 1e-8

    # Actual parameter-efficient update through both learned stages. This is
    # one-example acquisition mechanics, not evidence of market generalization.
    import mlx.optimizers as optim
    targets = mx.array([[[0., 1.], [0., 1.], [.5, .5]]])
    values = mx.array([[[0., 1.], [0., 1.], [0., 0.]]])
    weights = mx.array([[1., 1., 0.]])
    settings = {"soft_target_weight": 1., "ranking_weight": 1., "margin": .25}
    objective = lambda m: trade_objective(forward(m)["assessment_scores"],
        targets, values, weights, settings, xp=mx)
    before, grads = nn.value_and_grad(model, objective)(model)
    optimizer = optim.Adam(learning_rate=1e-3)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    after = objective(model)
    mx.eval(before, after)
    assert float(after) < float(before)

    # The native trainer callback must use the same target-free staged path.
    from propevolve.reasoning_policy.supervised_trainer import batch_loss
    training_batch = {
        "inputs": {"embeddings": embeddings, "available": available,
            "market_query": market_query, "assessment_query": assessment_query,
            "legal_actions": legal},
        "targets": {"probabilities": targets, "values": values,
            "boundary_weights": weights, "teacher_probabilities": mx.array([[.9, .1]]),
            "teacher_weights": mx.ones((1, 2))}}
    training_config = {"architecture": "staged_reasoning_v1",
        "action_supervision": settings, "interpretation_loss_weight": .5}
    production_loss = lambda m: batch_loss(m, training_batch, config=training_config)[0]
    loss_before, grads = nn.value_and_grad(model, production_loss)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    loss_after = production_loss(model)
    mx.eval(loss_before, loss_after)
    assert float(loss_after) < float(loss_before)
    assert float(mx.abs(grads["market_projector"]["projection"]["weight"]).sum()) > 1e-8

    from propevolve.reasoning_policy.supervised_trainer import evaluate_action_validation
    row = {"staged_queries": {"channel_names": ["expansion.long", "expansion.short"],
        "market_query": {k: v.tolist() for k, v in market_query.items()},
        "assessment_query": {k: v.tolist() for k, v in assessment_query.items()}},
        "market_embeddings": embeddings[0].tolist(), "market_available": available[0].tolist(),
        "target_name": "ENTER_LONG_1", "action_targets": {
            "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
            "probabilities": [0., 1., 0.], "values": [0., 2., -1.]}}
    traces = []
    report = evaluate_action_validation(model, [row],
        {**training_config, "validation_batch_size": 2, "max_seq_length": 128},
        on_scored=lambda index, trace: traces.append((index, trace)))
    expected = forward(model)
    mx.eval(expected)
    assert traces[0][0] == 0
    assert traces[0][1]["assessment"] == pytest.approx(expected["assessment_scores"][0].tolist())
    assert report["decision_boundary_semantics"] == "staged_independent_binary_v1"

    with pytest.raises(ValueError, match="query fields"):
        staged_forward(MLXReasoningBackend(model), embeddings, available,
            {**market_query, "teacher_targets": mx.ones((1, 2))}, assessment_query, legal)
