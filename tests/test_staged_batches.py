"""Prepared causal queries and economic targets enter the native batch iterator."""
import numpy as np
import pytest


def test_native_batches_preserve_independent_targets_and_partial_batches():
    pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.supervised_trainer import tensor_batches
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    from test_staged_queries import settings, Tokenizer
    queries = prepare_staged_queries({"trade.unrealized_r": 0.}, settings(), Tokenizer(),
        max_seq_length=1024, chat_template_kwargs={})
    rows = []
    for target in (
        {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
         "probabilities": [0., 1., 0.], "values": [0., 2., -1.]},
        {"names": ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"],
         "probabilities": [1., 0., 0.], "values": [0., -1., -1.]},
        {"names": ["HOLD", "CLOSE"], "probabilities": [0., 1.], "values": [-1., 1.]},
    ):
        rows.append({"staged_queries": queries, "action_targets": target,
            "market_embeddings": np.ones((4, 2)).tolist(),
            "market_available": [False, False, True, True],
            "teacher_probabilities": [.8, .1], "teacher_weights": [1., 1.]})
    batches = list(tensor_batches(rows, 2, 1024, include_partial=True))
    assert len(batches) == 2
    first, last = batches[0][0], batches[1][0]
    assert first["targets"]["boundary_weights"].tolist() == [[1., 1., 0.], [1., 0., 0.]]
    assert first["targets"]["probabilities"][0].tolist()[:2] == [[0., 1.], [0., 1.]]
    assert last["targets"]["boundary_weights"].tolist() == [[0., 0., 1.]]
    assert last["targets"]["probabilities"][0, 2].tolist() == [1., 0.]
    assert "teacher_probabilities" not in first["inputs"]
    assert first["inputs"]["available"].tolist() == [[False, False, True, True]] * 2
