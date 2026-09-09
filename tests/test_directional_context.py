"""Public JSON -> completed-bar directional context contract."""
import json

import numpy as np
import pytest

from propevolve.directional_context import DirectionalContext, DirectionalContextSpec


def settings(**overrides):
    return DirectionalContextSpec.from_mapping(json.loads(json.dumps(dict(
        enabled=True, horizons=[1, 2], volatility_window=2,
        volatility_floor=0.01,
    ) | overrides)))


def test_worked_log_return_example_and_warmup():
    # Log prices 0,1,0,2: final returns -1,+2 have population std 1.5.
    result = DirectionalContext(settings()).transform(np.exp([0., 1., 0., 2.]))
    np.testing.assert_array_equal(result.available[:2], False)
    assert np.isnan(result.values[:2]).all()
    np.testing.assert_allclose(result.values[-1], [1.3333333333333333,
                                                  0.4714045207910317])
    np.testing.assert_array_equal(result.available[-1], [True, True])


def test_incremental_batch_prefix_reset_and_symmetry():
    closes = np.array([100., 102., 99., 103., 101., 107., 104.])
    spec = settings(horizons=[1, 2, 4])
    batch = DirectionalContext(spec).transform(closes)
    stream = DirectionalContext(spec)
    for row, close in enumerate(closes):
        point = stream.update(row, close)
        prefix = DirectionalContext(spec).transform(closes[:row+1])
        np.testing.assert_allclose(point.values, batch.values[row], equal_nan=True)
        np.testing.assert_allclose(prefix.values[-1], point.values, equal_nan=True)
        np.testing.assert_array_equal(point.available, batch.available[row])
    mirrored = DirectionalContext(spec).transform(1 / closes)
    scaled = DirectionalContext(spec).transform(closes * 1000)
    np.testing.assert_allclose(mirrored.values, -batch.values, atol=1e-10)
    np.testing.assert_allclose(scaled.values, batch.values, atol=1e-10)
    stream.reset()
    assert not stream.update(0, closes[0]).available.any()


def test_disabled_flat_prices_and_partial_horizon_availability():
    disabled = DirectionalContext(settings(enabled=False)).transform([100., 101.])
    assert disabled.values.shape == disabled.available.shape == (2, 0)
    flat = DirectionalContext(settings(horizons=[1, 4])).transform([100.] * 6)
    np.testing.assert_array_equal(flat.available[2], [True, False])
    np.testing.assert_array_equal(flat.values[-1], [0., 0.])


@pytest.mark.parametrize('change', [
    {'enabled': 1}, {'horizons': []}, {'horizons': [0]},
    {'horizons': [1, 1]}, {'horizons': [True]}, {'horizons': [1.5]},
    {'volatility_window': 1}, {'volatility_window': True},
    {'volatility_floor': 0}, {'volatility_floor': float('nan')},
    {'volatility_floor': float('inf')}, {'volatility_floor': True},
])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        settings(**change)


def test_invalid_bar_never_advances_incremental_state():
    stream = DirectionalContext(settings())
    stream.update(0, 100.)
    for index, price in [(0, 101.), (2, 101.), (True, 101.), (1, 0.),
                         (1, -1.), (1, np.nan), (1, np.inf)]:
        with pytest.raises(ValueError):
            stream.update(index, price)
    stream.update(1, 101.)
    result = stream.update(2, 99.)
    expected = DirectionalContext(settings()).transform([100., 101., 99.])
    np.testing.assert_allclose(result.values, expected.values[-1])
    for prices in [[100., np.nan], [[100., 101.]], [100., -1.]]:
        with pytest.raises(ValueError):
            stream.transform(prices)


def test_json_filename_independent_horizons_and_future_mutation(tmp_path):
    path = tmp_path / 'arbitrary-name.json'
    path.write_text(json.dumps(dict(enabled=True, horizons=[2, 1, 5],
        volatility_window=3, volatility_floor=0.001)))
    spec = DirectionalContextSpec.from_mapping(json.loads(path.read_text()))
    assert spec.channels == ('momentum_2_bars', 'momentum_1_bars', 'momentum_5_bars')
    prices = np.array([100., 102., 99., 104., 105., 101., 109., 98.])
    original = DirectionalContext(spec).transform(prices)
    prices[5:] = [1., 5000., 10.]
    changed = DirectionalContext(spec).transform(prices)
    np.testing.assert_allclose(original.values[:5], changed.values[:5], equal_nan=True)
    np.testing.assert_array_equal(original.available[:5], changed.available[:5])
    np.testing.assert_array_equal(original.available[3], [True, True, False])
    assert DirectionalContext(spec).transform([]).values.shape == (0, 3)


def test_batch_does_not_change_live_state_or_mutate_input():
    stream = DirectionalContext(settings())
    stream.update(0, 100.)
    prices = np.array([100., 101., 102.])
    saved = prices.copy()
    stream.transform(prices)
    np.testing.assert_array_equal(prices, saved)
    stream.update(1, 101.)
    np.testing.assert_allclose(stream.update(2, 102.).values,
                             DirectionalContext(settings()).transform(prices).values[-1])


@pytest.mark.parametrize('payload', [{}, {'enabled': True},
    {'enabled': True, 'horizons': [1], 'volatility_window': 3,
     'volatility_floor': 0.01, 'typo': 5}])
def test_missing_and_unknown_json_keys_fail_explicitly(payload):
    with pytest.raises(ValueError):
        DirectionalContextSpec.from_mapping(payload)
