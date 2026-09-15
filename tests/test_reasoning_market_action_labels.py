from propevolve.decision import Action
from propevolve.reasoning_policy.labels import classify_market_action_rows, label_market_actions
from test_reasoning_challenger_e2e import environment
import numpy as np
import pytest


@pytest.mark.parametrize("side,sign", [(Action.ENTER_LONG_1, 1), (Action.ENTER_SHORT_1, -1)])
def test_entry_labels_distinguish_stop_from_profitable_target_miss(side, sign):
    env = environment()
    market = env.markets["NQ"]
    path = 1000 + sign * np.array([0., 0., 5., 10., 0., 0., 0., 0.])
    market.open[:] = market.close[:] = path
    market.high[:], market.low[:] = path + .1, path - .1
    kwargs = dict(decision=0, role_end=8, observation=[0.], risk_dollars=300.,
        point_value=20., round_trip_fee=4., minimum_mll_headroom=3000.,
        horizon=3, target_rs=(2., 3., 4.), stop_r=1.,
        utilities={"winner": 2., "failure": -1., "wait": 0.,
                   "missed_opportunity": -.25, "conflict_margin": .25})
    missed = label_market_actions(market, **kwargs)
    assert missed.outcomes[side].outcome == "below_target_profit"
    assert missed.outcomes[side].terminal_pnl == pytest.approx(196.)
    assert missed.outcomes[Action.WAIT].reward_to_go > missed.outcomes[side].reward_to_go
    # Same terminal profit, but this path first crossed the initial stop.
    if sign > 0:
        market.low[2] = 980.
    else:
        market.high[2] = 1020.
    stopped = label_market_actions(market, **kwargs)
    assert stopped.outcomes[side].outcome == "stop_before_target"
    assert stopped.outcomes[side].terminal_pnl == pytest.approx(-300.)
    assert stopped.entry_evidence[side.name]["full_horizon_terminal_r_net"] == pytest.approx(196 / 300)
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    from propevolve.reasoning_policy.dataset import supervised_record
    context = RollingContext(ContextConfig(2, ("trade.current_r",), input_mode="embeddings"))
    context.append(int(market.timestamps[0].astype("datetime64[ns]").astype(np.int64)),
        {"trade.current_r": 0.}, embedding=np.ones(2))
    record = supervised_record(context.snapshot(), stopped, source_id="fixture",
        continuation_id="barrier-reference", target_temperature=1.)
    assert record["targets"]["entry_evidence"][side.name]["barrier_outcome"] == "stop_before_target"
    assert "entry_evidence" not in record["messages"][-2]["content"]
    assert "mfe_r_gross" not in record["messages"][-2]["content"]


def _labels(kind):
    env = environment(kind)
    observation, _ = env.reset(options={"ticker": "NQ", "start": 0})
    return label_market_actions(
        env.markets["NQ"], decision=0, role_end=len(env.markets["NQ"].close),
        observation=observation, risk_dollars=300.0, point_value=20.0,
        round_trip_fee=0.0, minimum_mll_headroom=3000.0,
        horizon=2, target_rs=(2.0, 3.0, 4.0), stop_r=1.0,
        utilities={"winner": 2.0, "failure": -1.0, "wait": 0.0,
                   "missed_opportunity": -0.25, "conflict_margin": 0.25},
    )


def test_rising_market_teaches_long_above_wait_above_short():
    labels = _labels(1)
    values = {action: outcome.reward_to_go for action, outcome in labels.outcomes.items()}
    assert values[Action.ENTER_LONG_1] > values[Action.WAIT] > values[Action.ENTER_SHORT_1]
    assert values[Action.ENTER_LONG_1] >= 4.0


def test_falling_market_teaches_short_above_wait_above_long():
    labels = _labels(-1)
    values = {action: outcome.reward_to_go for action, outcome in labels.outcomes.items()}
    assert values[Action.ENTER_SHORT_1] > values[Action.WAIT] > values[Action.ENTER_LONG_1]


def test_no_economic_winner_teaches_wait_above_both_entries():
    labels = _labels(0)
    values = {action: outcome.reward_to_go for action, outcome in labels.outcomes.items()}
    assert values[Action.WAIT] > values[Action.ENTER_LONG_1]
    assert values[Action.WAIT] > values[Action.ENTER_SHORT_1]


def test_target_grid_assigns_more_credit_to_larger_trend_capture():
    two_r = _labels(1).outcomes[Action.ENTER_LONG_1].reward_to_go
    assert two_r == 4.0  # fixture reaches the configured 4R target before -1R


def test_vectorized_full_history_classes_match_scalar_economic_labels():
    utilities = {"winner": 2.0, "failure": -1.0, "wait": 0.0,
                 "missed_opportunity": -0.25, "conflict_margin": 0.25}
    for kind in (-1, 0, 1):
        env = environment(kind)
        market = env.markets["NQ"]
        actual = classify_market_action_rows(
            market, role_end=len(market.close), risk_dollars=300.0,
            point_value=20.0, round_trip_fee=0.0, horizon=2,
            target_rs=(2.0, 3.0, 4.0), stop_r=1.0, chunk_size=2,
        )
        for decision in range(len(market.close) - 2):
            labels = label_market_actions(
                market, decision=decision, role_end=len(market.close), observation=[0.0],
                risk_dollars=300.0, point_value=20.0, round_trip_fee=0.0,
                minimum_mll_headroom=3000.0, horizon=2,
                target_rs=(2.0, 3.0, 4.0), stop_r=1.0, utilities=utilities,
            )
            expected = max(labels.outcomes, key=lambda action: labels.outcomes[action].reward_to_go)
            assert actual[decision] == int(expected)
        assert (actual[-2:] == -1).all()
