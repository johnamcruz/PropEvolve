import numpy as np
import pytest

from propevolve.reasoning_policy.decisive_learning import fractional_snapshot


def test_saved_update_fraction_preserves_sources_and_exact_endpoints():
    before = {'layer.lora_a': np.array([2., 4.]),
              'market_projector.weight': np.array([10.])}
    after = {'layer.lora_a': np.array([6., 0.]),
             'market_projector.weight': np.array([18.])}
    result = fractional_snapshot(before, after, fraction=.25)
    np.testing.assert_array_equal(result['layer.lora_a'], [3., 3.])
    np.testing.assert_array_equal(result['market_projector.weight'], [12.])
    assert fractional_snapshot(before, after, fraction=0)['layer.lora_a'] is before['layer.lora_a']
    assert fractional_snapshot(before, after, fraction=1)['layer.lora_a'] is after['layer.lora_a']
    np.testing.assert_array_equal(before['layer.lora_a'], [2., 4.])
    for invalid in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            fractional_snapshot(before, after, fraction=invalid)


def test_projector_retention_does_not_alter_lora_learning_step():
    mx = pytest.importorskip('mlx.core')
    from propevolve.reasoning_policy.decisive_learning import project_retention_displacement
    before = {'layer.lora_a': mx.array([0., 0.]),
              'market_projector.weight': mx.array([0., 0.])}
    proposed = {'layer.lora_a': mx.array([3., 2.]),
                'market_projector.weight': mx.array([2., 1.])}
    gradient = {'layer.lora_a': mx.array([1., 0.]),
                'market_projector.weight': mx.array([1., 0.])}
    result, receipt = project_retention_displacement(before, proposed, gradient, component='projector')
    assert result['market_projector.weight'].tolist() == [0., 1.]
    assert result['layer.lora_a'].tolist() == [3., 2.]
    assert receipt['dot_after'] == 0
