"""Learning objectives use independent applicable decision boundaries."""
import numpy as np
import pytest


def test_wait_supervision_cannot_directly_train_direction_or_management():
    mx = pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.staged_learning import trade_objective
    probabilities = mx.array([[[1., 0.], [.5, .5], [.5, .5]]])
    values = mx.array([[[1., 0.], [0., 0.], [0., 0.]]])
    settings = {"soft_target_weight": 1., "ranking_weight": 0., "margin": .25}
    loss, grad = mx.value_and_grad(lambda scores: trade_objective(scores,
        probabilities, values, mx.array([[1., 0., 0.]]), settings, xp=mx))(mx.zeros((1, 3)))
    mx.eval(loss, grad)
    assert float(loss) == pytest.approx(np.log(2.), abs=1e-6)
    assert grad.tolist()[0] == pytest.approx([.5, 0., 0.], abs=1e-6)
