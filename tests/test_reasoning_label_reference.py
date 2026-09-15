import pytest

from propevolve.reasoning_policy.label_reference import barrier_result


@pytest.mark.parametrize("side,high,low,expected", [
    (1, [102.125], [99.5], True),
    (-1, [100.5], [97.875], True),
    (1, [103.], [98.], False),
    (1, [101.], [99.5], False),
])
def test_independent_barrier_reference(side, high, low, expected):
    assert barrier_result(100., high, low, side=side, risk=1., point_value=1.,
                          fee=.125, target=2., stop=1.) is expected


@pytest.mark.parametrize('action,sign', [('ENTER_LONG_1', 1), ('ENTER_SHORT_1', -1)])
def test_management_entry_reference_without_flat_label(action, sign):
    from propevolve.reasoning_policy.label_reference import management_entry
    row = {'completed_at_ns': 30, 'targets': {'position_entry': {
        'action': action, 'completed_at_ns': 20, 'execution': 'bar_open'}}}
    assert management_entry(row, None, {10: 0, 20: 1, 30: 2}) == (1, sign)
    row['targets']['position_entry']['completed_at_ns'] = 40
    with pytest.raises(ValueError, match='entry lineage'):
        management_entry(row, None, {10: 0, 20: 1, 30: 2, 40: 3})


def test_management_entry_reference_rejects_wait_parent_and_disagreement():
    from propevolve.reasoning_policy.label_reference import management_entry
    parent = {'completed_at_ns': 10, 'messages': [{'content': 'WAIT'}]}
    row = {'completed_at_ns': 30, 'targets': {}}
    with pytest.raises(ValueError, match='entry lineage'):
        management_entry(row, parent, {10: 0, 20: 1, 30: 2})
    parent['messages'][-1]['content'] = 'ENTER_LONG_1'
    assert management_entry(row, parent, {10: 0, 20: 1, 30: 2}) == (1, 1)
    row['targets']['position_entry'] = {'action': 'ENTER_SHORT_1',
                                      'completed_at_ns': 20, 'execution': 'bar_open'}
    with pytest.raises(ValueError, match='entry lineage'):
        management_entry(row, parent, {10: 0, 20: 1, 30: 2})
