"""Future excursions are supervision, never decision-time prompt values."""
import json
import numpy as np
import pytest

from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.dataset import market_supervised_record
from propevolve.reasoning_policy.labels import label_future_excursions
from test_reasoning_challenger_e2e import environment


def test_market_labels_measure_both_sides_and_keep_future_out_of_prompt():
    market = environment().markets["NQ"]
    market.open[:] = 100
    market.high[:] = 105
    market.low[:] = 98
    market.close[:] = 103
    context = RollingContext(ContextConfig(2, ("expansion.strength",)))
    context.append(1, {"expansion.strength": 0.5})
    labels = label_future_excursions(market, decision=0, role_end=8,
        horizon=2, risk_dollars=100, point_value=20, round_trip_fee=4)
    assert labels["long"]["mfe_r_gross"] == 1.0
    assert labels["long"]["mae_r_gross"] == 0.4
    assert labels["long"]["terminal_r_net"] == pytest.approx(0.56)
    assert labels["short"]["mfe_r_gross"] == 0.4
    assert labels["short"]["mae_r_gross"] == 1.0
    assert labels["short"]["terminal_r_net"] == pytest.approx(-0.64)
    record = market_supervised_record(context.snapshot(), opportunity=(False, False),
        source_id="fixture", label_end_ns=3, economic_contract={"horizon": 2}, excursions=labels)
    before = record["messages"][:-1]
    assert "mfe_r_gross" not in json.dumps(before)
    market.high[1:] = 110
    changed = label_future_excursions(market, decision=0, role_end=8,
        horizon=2, risk_dollars=100, point_value=20, round_trip_fee=4)
    after = market_supervised_record(context.snapshot(), opportunity=(True, False),
        source_id="fixture", label_end_ns=3, economic_contract={"horizon": 2}, excursions=changed)
    assert after["messages"][:-1] == before
    assert after["targets"] != record["targets"]
    assert label_future_excursions(market, decision=6, role_end=8,
        horizon=2, risk_dollars=100, point_value=20, round_trip_fee=4) is None
