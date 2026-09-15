"""Learning objectives use independent applicable decision boundaries."""
import numpy as np
import pytest


def test_balanced_entry_update_is_unchanged_by_microbatch_partition():
    """Two ENTER and two WAIT examples must not acquire a WAIT bias by batching."""
    from propevolve.reasoning_policy.staged_learning import trade_objective
    probabilities = np.array([
        [[0., 1.], [0., 1.], [.5, .5]],
        [[0., 1.], [1., 0.], [.5, .5]],
        [[1., 0.], [.5, .5], [.5, .5]],
        [[1., 0.], [.5, .5], [.5, .5]],
    ])
    values = probabilities.copy()
    weights = np.array([[1., 1., 0.], [1., 1., 0.],
                        [1., 0., 0.], [1., 0., 0.]])
    settings = {"soft_target_weight": 1., "ranking_weight": 0., "margin": .1}

    def loss(entry_bias, groups):
        scores = np.zeros((4, 3))
        scores[:, 0] = entry_bias
        return np.mean([trade_objective(scores[g], probabilities[g], values[g],
                                       weights[g], settings, xp=np) for g in groups])

    full, split = [slice(0, 4)], [slice(0, 2), slice(2, 4)]
    epsilon = 1e-5
    derivative = lambda groups: (loss(epsilon, groups) - loss(-epsilon, groups)) / (2 * epsilon)
    assert derivative(full) == pytest.approx(0., abs=1e-8)
    assert derivative(split) == pytest.approx(0., abs=1e-8)
    assert loss(.3, full) == pytest.approx(loss(.3, split), abs=1e-10)


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
