"""Frozen staged assessments -> selected corrections and mastered anchors."""
import numpy as np
import pytest


@pytest.mark.parametrize("target,scores,expected", [
    ("ENTER_LONG_1", [2., -2., 0.], [True, False, False]),
    ("ENTER_LONG_1", [-2., 2., 0.], [False, True, False]),
    ("ENTER_LONG_1", [2., 2., 0.], [True, True, False]),
    ("ENTER_LONG_1", [-2., -2., 0.], [False, False, False]),
    ("ENTER_SHORT_1", [2., 2., 0.], [True, False, False]),
    ("ENTER_SHORT_1", [-2., -2., 0.], [False, True, False]),
    ("ENTER_SHORT_1", [2., -2., 0.], [True, True, False]),
    ("ENTER_SHORT_1", [-2., 2., 0.], [False, False, False]),
    ("WAIT", [-2., 2., 0.], [True, False, False]),
    ("WAIT", [2., -2., 0.], [False, False, False]),
    ("HOLD", [2., 2., 2.], [False, False, True]),
    ("HOLD", [2., 2., -2.], [False, False, False]),
    ("CLOSE", [2., 2., -2.], [False, False, True]),
    ("CLOSE", [2., 2., 2.], [False, False, False]),
])
def test_staged_sampler_retains_only_correct_applicable_boundaries(target, scores, expected):
    from propevolve.reasoning_policy.targeted_subset import TargetedSampler
    from propevolve.reasoning_policy.staged_policy import legal_action_log_probs
    from propevolve.decision import Action
    from test_reasoning_targeted_subset import settings
    names = (["HOLD", "CLOSE"] if target in {"HOLD", "CLOSE"} else
             ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"])
    values = ([2., 0.] if target == "HOLD" else [0., 2.] if target == "CLOSE" else
              [0., -1., -1.] if target == "WAIT" else
              [0., -1., 2.] if target == "ENTER_SHORT_1" else [0., 2., -1.])
    log_probs = legal_action_log_probs(np.array(scores), [Action[n] for n in names], xp=np)
    evidence = {"index": 0, "ticker": "NQ", "target": target, "completed_at_ns": 100,
        "target_advantage": -1., "scores": dict(zip(names, log_probs.tolist())),
        "score_type": "log_probability", "assessment": dict(zip(
            ("entry", "direction", "management"), scores))}
    sampler = TargetedSampler([evidence], settings(), train_bounds=(100, 200), expected_rows=1)
    row = {"staged_queries": {}, "target_name": target, "action_targets": {
        "names": names, "values": values,
        "probabilities": [float(n == target) for n in names]}}
    selected = sampler.training_row(0, row, retain_mastery=True)
    retained = selected["mastered_anchor_retention"]
    assert list(retained["boundaries"].values()) == expected
    assert retained["assessment"] == scores
    from propevolve.reasoning_policy.staged_queries import prepare_staged_queries
    from propevolve.reasoning_policy.supervised_trainer import pack_examples
    from test_staged_queries import settings as query_settings, Tokenizer
    selected.update(staged_queries=prepare_staged_queries({"trade.unrealized_r": 0.},
        query_settings(), Tokenizer(), max_seq_length=1024, chat_template_kwargs={}),
        market_embeddings=[[1., 1.]] * 4, market_available=[True] * 4,
        teacher_probabilities=[.9, .1], teacher_weights=[1., 1.])
    targets = pack_examples([selected], max_seq_length=1024)[0]["targets"]
    assert targets["parent_assessment"].tolist() == [scores]
    assert targets["retention_weights"].tolist() == [expected]


def test_corrective_loss_preserves_entry_while_correcting_direction():
    mx = pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.staged_learning import corrective_trade_objective
    targets = {"probabilities": mx.array([[[0., 1.], [0., 1.], [.5, .5]]]),
        "values": mx.array([[[0., 2.], [-1., 2.], [0., 0.]]]),
        "boundary_weights": mx.array([[1., 1., 0.]]),
        "parent_assessment": mx.array([[2., -2., 9.]]),
        "retention_weights": mx.array([[True, False, False]])}
    config = {"action_supervision": {"soft_target_weight": 1., "ranking_weight": 0., "margin": .25},
        "mastered_anchor_retention": {"loss_weight": 1., "temperature": 1., "supervision_weight": 0.}}
    objective = lambda scores: corrective_trade_objective(scores, targets, config)
    _, before = mx.value_and_grad(objective)(mx.array([[2., -2., 9.]]))
    _, damaged = mx.value_and_grad(objective)(mx.array([[-2., -2., 9.]]))
    mx.eval(before, damaged)
    assert float(before[0, 0]) == pytest.approx(0., abs=1e-6)
    assert float(before[0, 1]) < 0
    assert float(before[0, 2]) == 0.
    assert float(damaged[0, 0]) < 0
    assert float(damaged[0, 1]) == pytest.approx(float(before[0, 1]))
    assert float(damaged[0, 2]) == 0.
