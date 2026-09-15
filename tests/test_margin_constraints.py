import numpy as np
import pytest

from propevolve.reasoning_policy.decisive_learning import constraint_multipliers


def test_constraints_keep_retention_and_acquisition_separate():
    # Orthogonal constraints require independent corrections, not an average.
    result = constraint_multipliers(np.eye(2), np.array([2., 4.]), tolerance=1e-8, maximum_cycles=100)
    np.testing.assert_allclose(result, [2., 4.])
    result = constraint_multipliers(np.eye(2), np.array([-2., 4.]), tolerance=1e-8, maximum_cycles=100)
    np.testing.assert_allclose(result, [0., 4.])


def test_inconsistent_constraints_do_not_claim_a_solution():
    with pytest.raises(ValueError, match='converge'):
        constraint_multipliers(np.array([[1., -1.], [-1., 1.]]),
                               np.array([1., 1.]), tolerance=1e-8, maximum_cycles=20)


def test_correlated_constraints_allow_an_already_satisfied_boundary_to_be_slack():
    # G=[[1,0],[1,1]], native step=(-2,-3), require G*step >= 0.
    # Nearest feasible step is (.5,-.5), obtained with multipliers (0,2.5).
    result = constraint_multipliers(np.array([[1., 1.], [1., 2.]]),
        np.array([2., 5.]), tolerance=1e-8, maximum_cycles=100)
    np.testing.assert_allclose(result, [0., 2.5], atol=1e-8)


def test_differentiable_margins_follow_hierarchical_decisions():
    from propevolve.reasoning_policy.decisive_learning import differentiable_boundary_margin
    names = ['WAIT', 'ENTER_LONG_1', 'ENTER_SHORT_1']
    scores = np.array([1., 4., 2.])
    for boundary, expected in [('ENTER', 3.), ('WAIT', -3.), ('LONG', 2.), ('SHORT', -2.)]:
        assert differentiable_boundary_margin(scores, names, boundary, xp=np) == expected
    assert differentiable_boundary_margin(np.array([5., 2.]), ['CLOSE', 'HOLD'], 'CLOSE', xp=np) == 3.
    assert differentiable_boundary_margin(np.array([5., 2.]), ['CLOSE', 'HOLD'], 'HOLD', xp=np) == -3.


def test_gram_keeps_small_differences_between_similar_gradients():
    from propevolve.reasoning_policy.decisive_learning import precise_gram
    rows = np.array([[10000., 1.], [10000., -1.]], dtype=np.float32)
    np.testing.assert_array_equal(precise_gram(rows, block_size=1),
        [[100000001., 99999999.], [99999999., 100000001.]])


def test_partial_diagnostic_progress_never_allows_forgetting_or_stagnation():
    from propevolve.reasoning_policy.decisive_learning import retained_margin_progress
    assert retained_margin_progress(-.86, -.56, forgotten=0, minimum_gain=.01)
    assert not retained_margin_progress(-.86, -.56, forgotten=1, minimum_gain=.01)
    assert not retained_margin_progress(-.86, -.86, forgotten=0, minimum_gain=.01)
    assert not retained_margin_progress(-.86, float('nan'), forgotten=0, minimum_gain=.01)
