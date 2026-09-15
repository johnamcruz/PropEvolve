import numpy as np
import pytest

from propevolve.reasoning_policy.supervision import hierarchical_action_objective


@pytest.mark.parametrize('values,task,mask,expected', [
    ([0, 2, -1], 0, [1, 1, 0], [0.34657359, 0.34657359, 0]),
    ([0, -1, -1], 0, [1, 1, 0], [0.69314718, 0, 0]),
    ([0, 2, -1], 0, [0, 1, 0], [0, 0.69314718, 0]),
    ([2, 0], 1, [0, 0, 1], [0, 0, 0.69314718]),
])
def test_named_objectives_preserve_applicable_hierarchical_mass(values, task, mask, expected):
    settings = dict(margin=0.1, soft_target_weight=1., ranking_weight=0.)
    probabilities = np.zeros(len(values))
    probabilities[np.argmax(values)] = 1.
    args = (np.zeros(len(values)), probabilities, values, settings)
    terms = hierarchical_action_objective(*args, task_code=task, xp=np,
        correction_boundaries=mask, return_terms=True)
    assert terms == pytest.approx(expected)
    assert sum(terms) == pytest.approx(hierarchical_action_objective(
        *args, task_code=task, xp=np, correction_boundaries=mask))


def test_gradient_report_separates_parameter_groups_and_marks_zero_as_undefined():
    from propevolve.reasoning_policy.gradient_audit import gradient_geometry
    gradients = {
        'entry': {'a.lora_a': np.array([3., 4.]), 'market_projector.w': np.array([2.])},
        'direction': {'a.lora_a': np.array([-3., -4.]), 'market_projector.w': np.array([2.])},
        'retention': {'a.lora_a': np.zeros(2), 'market_projector.w': np.zeros(1)},
    }
    result = gradient_geometry(gradients,
        {'a.lora_a': np.array([-0.3, -0.4]), 'market_projector.w': np.array([-0.2])})
    assert result['lora']['norms']['entry'] == 5.
    assert result['lora']['cosines']['entry|direction'] == pytest.approx(-1.)
    assert result['projector']['cosines']['entry|direction'] == pytest.approx(1.)
    assert result['lora']['cosines']['entry|retention'] is None
    assert result['lora']['gradient_dot_update']['direction'] == pytest.approx(2.5)
