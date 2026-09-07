from propevolve.decision import Action
from propevolve.reasoning_policy.labels import label_market_actions
from test_reasoning_challenger_e2e import environment


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
