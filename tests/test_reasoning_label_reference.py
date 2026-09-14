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
