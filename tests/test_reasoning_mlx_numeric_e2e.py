"""Tiny actual MLX CPU operations; not a substitute for real-backbone acceptance."""
import json
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from propevolve.reasoning_policy.projector import (
    attach_projector, export_policy_weights, restore_projector, temporal_features,
)
from propevolve.reasoning_policy.supervised_trainer import (
    batch_loss, build_optimizer, configure_trainable_components, pack_examples,
)
from propevolve.reasoning_policy.policy import sequence_scores


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(7)
    yield
    mx.set_default_device(previous)


def tiny_backbone(vocabulary=16, *, state=False):
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocabulary, 8)
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
            self.output = nn.Linear(8, vocabulary)
            self.forward_calls = 0
        def __call__(self, inputs, input_embeddings=None):
            self.forward_calls += 1
            x = self.model.embed_tokens(inputs) if input_embeddings is None else input_embeddings
            return self.output(mx.cumsum(x, axis=1))
    model = Backbone()
    model.freeze()
    attach_projector(model, {"embedding_dim": 2, "context_steps": 3, "market_tokens": 2,
                             "temporal_encoding": "pooled_levels",
                             **({"state_fields": ["trade.current_r", "trade.hold_bars"],
                                 "state_scales": [4.0, 150.0]} if state else {})})
    return model


def example():
    return {"tokens": [1, 2, 3, 4], "offset": 2,
        "alternatives": [([1, 2, 3, 4], 2), ([1, 2, 5, 4], 2), ([1, 2, 6, 4], 2)],
        "action_targets": {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
                           "probabilities": [.1, .8, .1], "values": [0., 10., -10.]},
        "market_embeddings": [[0., 0.], [1., 2.], [3., 4.]], "market_available": [False, True, True]}


def test_latest_state_plus_causal_deltas_exposes_lifecycle_without_future_rows():
    pooled = np.asarray([[[1., 10.], [3., 14.], [8., 12.]]], np.float32)
    actual = temporal_features(pooled, "latest_plus_deltas", xp=np)
    np.testing.assert_array_equal(actual, [[[8., 12.], [2., 4.], [5., -2.]]])
    # Altering the latest completed bin changes only the latest-state token and
    # the final transition; there is no lookahead token.
    changed = pooled.copy()
    changed[:, -1] = [10., 15.]
    revised = temporal_features(changed, "latest_plus_deltas", xp=np)
    np.testing.assert_array_equal(revised[:, 1], actual[:, 1])
    np.testing.assert_array_equal(revised[:, 0], [[10., 15.]])
    np.testing.assert_array_equal(revised[:, -1], [[7., 1.]])


def test_real_mlx_projector_gradient_update_and_save_reload_preserve_scores(tmp_path):
    from mlx.utils import tree_flatten
    import mlx.optimizers as optim
    model = tiny_backbone()
    row = example()
    packed = tuple(mx.array(x) for x in pack_examples([row], max_seq_length=8))
    config = {"input_mode": "embeddings", "action_supervision":
        {"enabled": True, "soft_target_weight": 1., "ranking_weight": 1., "margin": .25}}
    def loss(m, *batch):
        return batch_loss(m, *batch, config=config)[0]
    old, gradients = nn.value_and_grad(model, loss)(model, *packed)
    flat = dict(tree_flatten(gradients))
    assert set(flat) == {"market_projector.projection.weight"}
    assert bool(mx.any(flat["market_projector.projection.weight"] != 0).item())
    optim.SGD(learning_rate=1e-5).update(model, gradients)
    assert float(loss(model, *packed).item()) < float(old.item())
    tokens = tuple((*item, np.asarray(row["market_embeddings"], np.float32),
                   np.asarray(row["market_available"], bool)) for item in row["alternatives"])
    expected = sequence_scores(model, tokens)
    model.market_projector.freeze()
    export_policy_weights(model, tmp_path)
    assert (tmp_path / "projector.safetensors").is_file()
    restored = tiny_backbone()
    # Preserve the same frozen external backbone; only projector is reloaded.
    restored.model = model.model
    restored.output = model.output
    restore_projector(restored, tmp_path)
    np.testing.assert_allclose(np.asarray(sequence_scores(restored, tokens)), np.asarray(expected), atol=1e-6)


def test_real_mlx_causal_state_projector_changes_scores_and_survives_reload(tmp_path):
    model = tiny_backbone(128, state=True)
    embeddings = np.asarray(example()["market_embeddings"], np.float32)
    available = np.asarray(example()["market_available"], bool)
    tokens = ([1, 2, 3, 4], 2, np.asarray([2.0, 75.0], np.float32),
              embeddings, available)
    baseline = ([1, 2, 3, 4], 2, np.asarray([0.0, 0.0], np.float32),
                embeddings, available)
    expected = sequence_scores(model, [tokens, baseline])
    assert float(expected[0].item()) != float(expected[1].item())

    model.market_projector.freeze()
    export_policy_weights(model, tmp_path)
    restored = tiny_backbone(128, state=True)
    restored.model = model.model
    restored.output = model.output
    restore_projector(restored, tmp_path)
    actual = sequence_scores(restored, [tokens, baseline])
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)


def test_existing_market_projector_can_warm_start_causal_state_extension(tmp_path):
    from mlx.utils import tree_flatten

    parent = tiny_backbone(128)
    export_policy_weights(parent, tmp_path)
    child = tiny_backbone(128, state=True)
    before = dict(tree_flatten(child.parameters()))[
        "market_projector.state_projection.weight"]
    before = np.asarray(before)

    with pytest.raises(ValueError, match="saved projector does not match"):
        restore_projector(child, tmp_path)
    restore_projector(child, tmp_path, allow_state_extension=True)

    parent_weights = dict(tree_flatten(parent.parameters()))
    child_weights = dict(tree_flatten(child.parameters()))
    np.testing.assert_allclose(
        np.asarray(child_weights["market_projector.projection.weight"]),
        np.asarray(parent_weights["market_projector.projection.weight"]),
        atol=0.,
    )
    np.testing.assert_allclose(
        np.asarray(child_weights["market_projector.state_projection.weight"]),
        before,
        atol=0.,
    )


def test_mlx_batch_vectorizes_rows_without_changing_loss():
    model = tiny_backbone()
    first = example()
    second = {**example(),
        "market_embeddings": [[0., 0.], [2., 1.], [4., 3.]],
        "action_targets": {
            "names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
            "probabilities": [.7, .2, .1], "values": [5., 1., -4.],
        },
    }
    config = {"input_mode": "embeddings", "action_supervision":
        {"enabled": True, "soft_target_weight": 1., "ranking_weight": 1., "margin": .25}}
    packed = tuple(mx.array(value) for value in pack_examples(
        [first, second], max_seq_length=8))
    batched = batch_loss(model, *packed, config=config)[0]
    mx.eval(batched)
    assert model.forward_calls == 1

    individual = []
    for row in (first, second):
        one = tuple(mx.array(value) for value in pack_examples([row], max_seq_length=8))
        loss = batch_loss(model, *one, config=config)[0]
        mx.eval(loss)
        individual.append(float(loss.item()))
    assert float(batched.item()) == pytest.approx(float(np.mean(individual)), abs=1e-6)


def test_market_sft_batch_preserves_each_row_loss_and_projector_gradient():
    from mlx.utils import tree_flatten
    model = tiny_backbone()
    first = {k: v for k, v in example().items()
             if k not in {"alternatives", "action_targets"}}
    second = {**first, "tokens": [1, 2, 5, 6, 4], "offset": 3,
              "market_embeddings": [[0., 0.], [2., 1.], [4., 3.]]}
    config = {"input_mode": "embeddings", "action_supervision": {"enabled": False}}
    def evaluate(rows):
        tensors = tuple(mx.array(x) for x in pack_examples(rows, max_seq_length=8))
        fn = nn.value_and_grad(model, lambda m, *b: batch_loss(m, *b, config=config)[0])
        loss, grads = fn(model, *tensors)
        return float(loss.item()), {k: np.array(v) for k, v in tree_flatten(grads)}
    left, lg = evaluate([first])
    right, rg = evaluate([second])
    together, bg = evaluate([first, second])
    assert together == pytest.approx((left + right) / 2, abs=1e-6)
    assert any(np.any(value != 0) for value in bg.values())
    for key in bg:
        np.testing.assert_allclose(bg[key], (lg[key] + rg[key]) / 2,
                                   atol=1e-6, rtol=1e-6)


def test_real_mlx_hierarchical_update_learns_entry_and_direction_together():
    import mlx.optimizers as optim

    model = tiny_backbone()
    row = example()
    packed = tuple(mx.array(value) for value in pack_examples([row], max_seq_length=8))
    config = {
        "input_mode": "embeddings", "decision_objective": "hierarchical_binary",
        "action_supervision": {
            "enabled": True, "soft_target_weight": 1.,
            "ranking_weight": 2., "margin": .25,
        },
    }

    def loss(m, *batch):
        return batch_loss(m, *batch, config=config)[0]

    before, gradients = nn.value_and_grad(model, loss)(model, *packed)
    optim.SGD(learning_rate=1e-5).update(model, gradients)
    after = loss(model, *packed)
    mx.eval(before, after)
    assert float(after.item()) < float(before.item())


def test_real_mlx_hierarchical_management_learns_hold_and_close_from_causal_state():
    import mlx.optimizers as optim

    model = tiny_backbone(128, state=True)
    base = example()
    def management(target, state):
        hold = target == "HOLD"
        return {
            **base,
            "alternatives": [([1, 2, 7, 4], 2), ([1, 2, 8, 4], 2)],
            "action_targets": {
                "names": ["HOLD", "CLOSE"],
                "probabilities": [.9, .1] if hold else [.1, .9],
                "values": [2., 0.] if hold else [0., 2.],
            },
            "causal_state": state,
        }
    rows = [management("HOLD", [-2., 0.]), management("CLOSE", [2., 150.])]
    packed = tuple(mx.array(value) for value in pack_examples(rows, max_seq_length=8))
    config = {
        "input_mode": "embeddings", "decision_objective": "hierarchical_binary",
        "action_supervision": {
            "enabled": True, "soft_target_weight": 1.,
            "ranking_weight": 2., "margin": .25,
        },
    }
    def loss(m, *batch):
        return batch_loss(m, *batch, config=config)[0]
    optimizer = optim.Adam(learning_rate=1e-2)
    value_and_grad = nn.value_and_grad(model, loss)
    before = float(loss(model, *packed).item())
    for _ in range(50):
        _, gradients = value_and_grad(model, *packed)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
    after = float(loss(model, *packed).item())
    assert after < before

    from propevolve.reasoning_policy.supervised_trainer import _batch_outputs
    _, _, scores = _batch_outputs(model, *packed, config=config)
    mx.eval(scores)
    margins = np.asarray(scores)[:, 0] - np.asarray(scores)[:, 1]
    assert margins[0] > 0
    assert margins[1] < 0


def test_component_selection_can_train_projector_without_lora_and_preserve_both():
    from mlx.utils import tree_flatten

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.ones((2, 2))
            self.lora_b = mx.ones((2, 2))

    model = tiny_backbone()
    model.adapter = Adapter()
    configure_trainable_components(model, ["projector"])
    assert set(dict(tree_flatten(model.trainable_parameters()))) == {
        "market_projector.projection.weight"
    }
    configure_trainable_components(model, ["lora", "projector"])
    assert set(dict(tree_flatten(model.trainable_parameters()))) == {
        "adapter.lora_a", "adapter.lora_b", "market_projector.projection.weight"
    }


def test_component_optimizer_routes_distinct_learning_rates_by_public_parameter_name():
    from mlx.utils import tree_flatten

    class TwoComponents(nn.Module):
        def __init__(self):
            super().__init__()
            self.market_projector = nn.Linear(1, 1, bias=False)
            self.adapter = nn.Linear(1, 1, bias=False)
            self.adapter.lora_a = mx.ones((1, 1))
            self.adapter.lora_b = mx.ones((1, 1))

    model = TwoComponents()
    model.freeze()
    model.market_projector.unfreeze()
    model.adapter.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    before = dict(tree_flatten(model.trainable_parameters()))
    optimizer = build_optimizer({
        "optimizer": "adam",
        "optimizer_config": {"adam": {}},
        "learning_rate": 3e-6,
        "lr_schedule": None,
        "component_learning_rates": {"lora": 1e-6, "projector": 1e-5},
    })
    gradients = {"market_projector": {"weight": mx.ones((1, 1))},
                 "adapter": {"lora_a": mx.ones((1, 1)),
                             "lora_b": mx.ones((1, 1))}}
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    after = dict(tree_flatten(model.trainable_parameters()))
    projector_delta = float(mx.abs(
        before["market_projector.weight"] - after["market_projector.weight"]).item())
    lora_delta = float(mx.abs(before["adapter.lora_a"] - after["adapter.lora_a"]).item())
    assert projector_delta == pytest.approx(10 * lora_delta, rel=1e-2)


def test_masked_history_cannot_affect_projected_scores():
    model = tiny_backbone()
    embeddings = np.array(example()["market_embeddings"], np.float32)
    available = np.array(example()["market_available"], bool)
    expected = sequence_scores(model, [([1, 2, 3, 4], 2, embeddings, available)])
    embeddings[0] = 10000
    actual = sequence_scores(model, [([1, 2, 3, 4], 2, embeddings, available)])
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)


@pytest.mark.parametrize("names", [("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"), ("HOLD", "CLOSE"), ("WAIT",)])
def test_actual_policy_selects_only_legal_actions_and_preserves_score_parity(names):
    from propevolve.decision import Action
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    from propevolve.reasoning_policy.dataset import context_messages, embedding_payload
    from propevolve.reasoning_policy.policy import MLXActionPolicy
    from test_reasoning_token_parity_e2e import LiteralTokenizer
    history = RollingContext(ContextConfig(3, ("account.realized_pnl_norm",), input_mode="embeddings"))
    history.append(1, {"account.realized_pnl_norm": -.5}, embedding=np.array([1., 2.]))
    policy = MLXActionPolicy(tiny_backbone(128), LiteralTokenizer(), max_seq_length=2048,
        input_mode="embeddings", projector={"embedding_dim": 2, "context_steps": 3, "market_tokens": 2,
                                             "temporal_encoding": "pooled_levels"})
    actions = tuple(Action[name] for name in names)
    context = history.snapshot()
    chosen, scores = policy.decide(context, actions)
    assert chosen in actions
    assert set(scores) == set(names)
    assert scores[chosen.name] == max(scores.values())
    encoded = policy.tokenize_completions(context_messages(context, actions), names,
        market_context=embedding_payload(context))
    np.testing.assert_allclose(list(scores.values()), np.asarray(sequence_scores(policy.model, encoded)), atol=1e-6)
    with pytest.raises(ValueError, match="without legal actions"):
        policy.decide(context, [])


def test_policy_rejects_nonfinite_scores_instead_of_executing_an_action():
    from propevolve.reasoning_policy.policy import MLXActionPolicy
    from test_reasoning_token_parity_e2e import LiteralTokenizer
    model = tiny_backbone(128)
    model.output.weight = mx.full(model.output.weight.shape, float("nan"))
    policy = MLXActionPolicy(model, LiteralTokenizer(), max_seq_length=256)
    with pytest.raises(ValueError, match="nonfinite action scores"):
        policy.completion_scores([{"role": "user", "content": "state"}], ["WAIT", "ENTER_LONG_1"])
