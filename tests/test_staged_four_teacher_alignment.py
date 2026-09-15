"""Named four-teacher targets must survive the same causal staged batch path."""
import copy
import json

import numpy as np
import pytest

from test_staged_queries import settings, Tokenizer


def test_all_teacher_channels_align_by_name_without_changing_embedding_inputs():
    mx = pytest.importorskip("mlx.core")
    from propevolve.teachers.expansion import CHANNELS as expansion
    from propevolve.teachers.trend import CHANNELS as trend
    from propevolve.teachers.regime import CHANNELS as regime
    from propevolve.teachers.volume import CHANNELS as volume
    from propevolve.reasoning_policy.staged_preparation import encode_staged_record
    from propevolve.reasoning_policy.staged_batches import pack_staged_examples
    from propevolve.reasoning_policy.market_distillation import probability_loss

    names = [f"{teacher}.{channel}" for teacher, channels in
             (("expansion", expansion), ("trend", trend), ("regime", regime), ("volume", volume))
             for channel in channels]
    # Distinct values expose row/channel swaps. Independent Long/Short targets
    # are not normalized against one another; only Regime is a class simplex.
    expected = [.81, .72, .63, .54, .85, .76, .67, .58, .1, .3, .6, .91, .82, .73, .64]
    query = settings()
    query["market"]["channels"] = [dict(name=n, query=f"Context {i}?", weight=1.)
                                    for i, n in enumerate(names)]
    config = dict(staged_policy=query, max_seq_length=4096, chat_template_kwargs={},
                  projector=dict(market_tokens=2, embedding_dim=2, context_steps=20,
                                 temporal_encoding="latest_plus_deltas"))
    legal = ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
    record = dict(messages=[dict(role="system", content="Trade"),
        dict(role="user", content=json.dumps(dict(fields=["trade.unrealized_r"],
            history_oldest_first=[[0.]], legal_actions=legal))),
        dict(role="assistant", content="ENTER_LONG_1")],
        targets=dict(action_order=legal, action_probabilities=[0., 1., 0.],
            outcomes={n: dict(reward_to_go=v) for n, v in zip(legal, [0., 2., -1.])},
            specialist_targets=dict(reversed(list(zip(names, expected))))),
        market_embeddings=np.arange(40).reshape(20, 2).tolist(), market_available=[True] * 20)

    def pack(source):
        row = encode_staged_record(source, config, Tokenizer())
        row.update(market_embeddings=source["market_embeddings"], market_available=source["market_available"])
        return pack_staged_examples([row], max_seq_length=4096)

    batch = pack(record)
    assert batch["targets"]["teacher_probabilities"][0].tolist() == pytest.approx(expected)
    np.testing.assert_array_equal(batch["inputs"]["embeddings"][0], record["market_embeddings"])
    changed = copy.deepcopy(record)
    changed["targets"]["specialist_targets"] = dict(zip(names, [1. - p for p in expected]))
    other = pack(changed)
    for kind in ("market_query", "assessment_query"):
        for key in batch["inputs"][kind]:
            np.testing.assert_array_equal(batch["inputs"][kind][key], other["inputs"][kind][key])
    np.testing.assert_array_equal(batch["inputs"]["embeddings"], other["inputs"]["embeddings"])
    _, grad = mx.value_and_grad(lambda scores: probability_loss(scores,
        mx.array(batch["targets"]["teacher_probabilities"]),
        mx.array(batch["targets"]["teacher_weights"])))(mx.zeros((1, 15)))
    mx.eval(grad)
    assert (np.asarray(grad)[0] < 0).tolist() == [p > .5 for p in expected]

    incomplete = copy.deepcopy(record)
    del incomplete["targets"]["specialist_targets"][names[-1]]
    with pytest.raises(ValueError, match="teacher channels"):
        pack(incomplete)
