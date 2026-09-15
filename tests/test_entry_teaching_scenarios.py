"""Economic scenarios through serialized labels and production binary teaching."""
import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.reasoning_policy.context import ContextConfig, RollingContext
from propevolve.reasoning_policy.dataset import supervised_record
from propevolve.reasoning_policy.labels import label_market_actions, classify_market_action_rows, label_entry_opportunity
from propevolve.reasoning_policy.supervision import action_targets
from propevolve.reasoning_policy.staged_batches import binary_targets
from test_reasoning_challenger_e2e import environment


def scenario_record(sign, scenario, temperature):
    env = environment()
    market = env.markets["NQ"]
    market.open[:] = market.close[:] = 1000.
    market.high[:], market.low[:] = 1000.1, 999.9
    if scenario == "stop_then_target":
        if sign > 0:
            market.low[1], market.high[2] = 985.2, 1070.
        else:
            market.high[1], market.low[2] = 1014.8, 930.
    elif scenario == "winner":
        if sign > 0:
            market.high[2] = 1065.
        else:
            market.low[2] = 935.
    elif scenario == "target_then_stop":
        if sign > 0:
            market.high[1], market.low[2] = 1030.2, 985.2
        else:
            market.low[1], market.high[2] = 969.8, 1014.8
    elif scenario == "same_bar_collision":
        market.high[1], market.low[1] = 1040., 960.
    elif scenario != "wait":
        raise AssertionError("unknown fixture")
    options = dict(decision=0, role_end=8, observation=[0.], risk_dollars=300.,
        point_value=20., round_trip_fee=4., minimum_mll_headroom=3000.,
        horizon=3, target_rs=(2., 3., 4.), stop_r=1.,
        utilities={"winner": 2., "failure": -1., "wait": 0.,
                   "missed_opportunity": -.25, "conflict_margin": .25})
    labels = label_market_actions(market, **options)
    context = RollingContext(ContextConfig(2, ("trade.current_r",), input_mode="embeddings"))
    context.append(int(market.timestamps[0].astype("datetime64[ns]").astype(np.int64)),
        {"trade.current_r": 0.}, embedding=np.ones(2))
    return market, labels, supervised_record(context.snapshot(), labels,
        source_id="scenario", continuation_id="barrier-reference", target_temperature=temperature)


@pytest.mark.parametrize("sign", [1, -1])
def test_exact_one_r_stop_cannot_become_later_two_r_winner(sign):
    market, labels, record = scenario_record(sign, "stop_then_target", .5)
    side = Action.ENTER_LONG_1 if sign > 0 else Action.ENTER_SHORT_1
    assert labels.outcomes[side].outcome == "stop_before_target"
    assert labels.outcomes[side].terminal_pnl == pytest.approx(-300.)
    assert record["messages"][-1]["content"] == "WAIT"
    census = classify_market_action_rows(market, role_end=8, risk_dollars=300.,
        point_value=20., round_trip_fee=4., horizon=3, target_rs=(2., 3., 4.), stop_r=1., chunk_size=2)
    assert census[0] == int(Action.WAIT)


@pytest.mark.parametrize("temperature", [.5, 1., 2., 10.])
def test_wait_soft_teaching_stays_wait_at_every_temperature(temperature):
    _, _, record = scenario_record(1, "wait", temperature)
    probabilities, values, weights = binary_targets(action_targets(record))
    assert probabilities[0, 0] > probabilities[0, 1]
    assert values[0, 0] > values[0, 1]
    assert weights.tolist() == [1., 0., 0.]


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("scenario", ["winner", "target_then_stop", "stop_then_target", "same_bar_collision", "wait"])
@pytest.mark.parametrize("temperature", [.5, 2.])
def test_economic_record_to_mlx_update_teaches_only_correct_boundaries(sign, scenario, temperature):
    mx = pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.staged_learning import trade_objective
    market, labels, record = scenario_record(sign, scenario, temperature)
    qualified = scenario in {"winner", "target_then_stop"}
    side = Action.ENTER_LONG_1 if sign > 0 else Action.ENTER_SHORT_1
    assert record["messages"][-1]["content"] == (side.name if qualified else "WAIT")
    census = classify_market_action_rows(market, role_end=8, risk_dollars=300.,
        point_value=20., round_trip_fee=4., horizon=3, target_rs=(2., 3., 4.), stop_r=1., chunk_size=2)
    assert census[0] == int(side if qualified else Action.WAIT)
    if scenario == "target_then_stop":
        assert labels.outcomes[side].outcome == "target_2r_before_stop"
        assert labels.outcomes[side].terminal_pnl == pytest.approx(600.)
    p, v, w = binary_targets(action_targets(record))
    assert w.tolist() == ([1., 1., 0.] if qualified else [1., 0., 0.])
    loss, grad = mx.value_and_grad(lambda scores: trade_objective(scores,
        mx.array(p[None]), mx.array(v[None]), mx.array(w[None]),
        {"soft_target_weight": 1., "ranking_weight": 1., "margin": .25}, xp=mx))(mx.zeros((1, 3)))
    mx.eval(loss, grad)
    # Gradient descent subtracts the gradient: negative entry means ENTER
    # increases; positive means WAIT increases. No positioned loss is active.
    assert (grad[0, 0].item() < 0) == qualified
    assert grad[0, 2].item() == 0.
    if qualified:
        assert grad[0, 1].item() * sign < 0
    else:
        assert grad[0, 1].item() == 0.


@pytest.mark.parametrize("field", ["open", "high", "low"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_price_cannot_be_silently_taught_as_wait(field, value):
    market, _, _ = scenario_record(1, "wait", .5)
    getattr(market, field)[1] = value
    common = dict(role_end=8, risk_dollars=300., point_value=20.,
                  round_trip_fee=4., horizon=3, stop_r=1.)
    with pytest.raises(ValueError, match="finite"):
        label_entry_opportunity(market, decision=0, target_r=2., **common)
    with pytest.raises(ValueError, match="finite"):
        classify_market_action_rows(market, target_rs=(2., 3., 4.), **common)
