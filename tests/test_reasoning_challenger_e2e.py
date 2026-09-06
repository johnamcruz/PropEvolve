"""Public challenger boundaries exercised with the real challenge simulator."""

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.environment import ChallengeSpec, HistoricalChallengeEnv, MarketSeries
from propevolve.reasoning_policy.labels import label_actions


def environment(direction=1):
    prices = 1000 + direction * np.arange(8) * 60.0
    market = MarketSeries(
        ticker="NQ", timestamps=np.datetime64("2024-01-02T15:00")
        + np.arange(8) * np.timedelta64(3, "m"),
        open=prices, high=prices + 1, low=prices - 1, close=prices,
        embeddings=np.ones((8, 2), dtype=np.float32),
    )
    return HistoricalChallengeEnv(
        {"NQ": market}, tick_values={"NQ": 20.0}, round_trip_fees={"NQ": 4.0},
        spec=ChallengeSpec(
            profit_target=6000, max_loss=3000, episode_days=1,
            bars_per_day=8, max_position_size=1, minimum_mll_headroom=300,
            trailing_mll_lock=True, terminal_pass_reward=250,
            terminal_blow_reward=-1500, terminal_timeout_reward=-2,
            terminal_pass_speed_reward_per_day=0, reward_scale=1000,
        ), seed=7,
    )


def passive_factory():
    return lambda observation, info: (
        Action.HOLD if Action.HOLD in info["valid_actions"] else Action.WAIT
    )


@pytest.mark.parametrize("direction,winner,loser", [
    (1, Action.ENTER_LONG_1, Action.ENTER_SHORT_1),
    (-1, Action.ENTER_SHORT_1, Action.ENTER_LONG_1),
])
def test_same_state_labels_preserve_pass_blow_timeout_and_do_not_mutate_source(
    direction, winner, loser,
):
    env = environment(direction)
    observation, _ = env.reset(options={"ticker": "NQ", "start": 0})
    result = label_actions(
        env, reset_options={"ticker": "NQ", "start": 0}, prefix=(),
        continuation_factory=passive_factory, max_steps=8,
    )
    np.testing.assert_array_equal(result.observation, observation)
    assert result.outcomes[winner].outcome == "pass"
    assert result.outcomes[loser].outcome == "blow"
    assert result.outcomes[Action.WAIT].outcome == "timeout"
    assert result.outcomes[winner].terminal_pnl >= 6000
    assert result.outcomes[loser].terminal_pnl <= -3000
    assert result.outcomes[Action.WAIT].terminal_pnl == 0
    # Label generation must not advance or reset the caller's active episode.
    _, _, _, _, info = env.step(winner)
    assert info["decision_index"] == 0


def test_positioned_labels_cover_hold_and_close_instead_of_new_entries():
    result = label_actions(
        environment(), reset_options={"ticker": "NQ", "start": 0},
        prefix=(Action.ENTER_LONG_1,), continuation_factory=passive_factory,
        max_steps=8,
    )
    assert set(result.outcomes) == {Action.HOLD, Action.CLOSE}
    assert result.outcomes[Action.HOLD].outcome == "pass"
    assert result.outcomes[Action.CLOSE].outcome == "timeout"
    assert result.outcomes[Action.CLOSE].terminal_pnl == 1196


def test_incomplete_rollout_is_rejected_not_labeled_as_timeout():
    with pytest.raises(ValueError, match="incomplete"):
        label_actions(
            environment(), reset_options={"ticker": "NQ", "start": 0}, prefix=(),
            continuation_factory=passive_factory, max_steps=1,
        )


def test_json_config_builds_causal_rolling_history_with_all_named_inputs(tmp_path):
    import json
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext

    path = tmp_path / "arbitrary-name.json"
    path.write_text(json.dumps({"context_steps": 2, "fields": [
        "expansion_long", "expansion_short", "trend_long", "trend_short",
        "regime_chop", "balance", "mll_headroom",
    ]}))
    spec = ContextConfig.load(path)
    history = RollingContext(spec)
    first = dict(zip(spec.fields, [0.8, 0.2, 0.7, 0.1, 0.1, -1500, 1500]))
    history.append(10, first)
    frozen = history.snapshot()
    assert frozen.available.tolist() == [False, True]
    history.append(20, {**first, "balance": -1000})
    history.append(30, {**first, "balance": 100})
    result = history.snapshot()
    assert result.timestamps == (20, 30)
    assert result.values[:, spec.fields.index("balance")].tolist() == [-1000, 100]
    assert frozen.values[-1, spec.fields.index("balance")] == -1500
    assert result.fields == spec.fields
    with pytest.raises(ValueError, match="increasing"):
        history.append(20, first)
    with pytest.raises(ValueError, match="fields"):
        history.append(40, {**first, "future_outcome": 1})
    with pytest.raises(ValueError, match="finite"):
        history.append(40, {**first, "balance": float("nan")})
    history.reset()
    assert not history.snapshot().available.any()


@pytest.mark.parametrize("direction,expected", [(1, (True, False)), (-1, (False, True))])
def test_direct_opportunity_labels_use_next_open_two_r_and_split_reserve(direction, expected):
    from propevolve.reasoning_policy.labels import label_entry_opportunity
    market = environment(direction).markets["NQ"]
    label = label_entry_opportunity(
        market, decision=0, role_end=8, horizon=3, risk_dollars=300,
        point_value=20, round_trip_fee=4, target_r=2, stop_r=1,
    )
    assert label == expected
    assert label_entry_opportunity(
        market, decision=6, role_end=8, horizon=3, risk_dollars=300,
        point_value=20, round_trip_fee=4, target_r=2, stop_r=1,
    ) is None


def test_supervised_record_keeps_future_targets_out_of_prompt():
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    from propevolve.reasoning_policy.dataset import supervised_record
    context = RollingContext(ContextConfig(2, ("expansion", "trend", "regime", "balance")))
    context.append(1, dict(expansion=0.8, trend=0.7, regime=0.2, balance=0))
    labels = label_actions(
        environment(), reset_options={"ticker": "NQ", "start": 0}, prefix=(),
        continuation_factory=passive_factory, max_steps=8,
    )
    result = supervised_record(
        context.snapshot(), labels, source_id="fixture", continuation_id="passive",
        target_temperature=1.0,
    )
    assert "terminal_pnl" not in result["messages"][1]["content"]
    assert result["messages"][2]["content"] == "ENTER_LONG_1"
    assert sum(result["targets"]["action_probabilities"]) == pytest.approx(1)
    assert result["targets"]["outcomes"]["ENTER_SHORT_1"]["outcome"] == "blow"


def test_future_mutation_changes_labels_but_not_the_causal_anchor():
    from dataclasses import replace

    up = environment()
    down = environment(-1)
    kwargs = dict(reset_options={"ticker": "NQ", "start": 0}, prefix=(),
                  continuation_factory=passive_factory, max_steps=8)
    first, second = label_actions(up, **kwargs), label_actions(down, **kwargs)
    np.testing.assert_array_equal(first.observation, second.observation)
    assert first.outcomes[Action.ENTER_LONG_1].outcome == "pass"
    assert second.outcomes[Action.ENTER_LONG_1].outcome == "blow"


def test_both_barriers_same_bar_is_not_a_winner_and_costs_are_in_the_target():
    from dataclasses import replace
    from propevolve.reasoning_policy.labels import label_entry_opportunity

    market = environment().markets["NQ"]
    # Entry at 1060. +2R after $4 fee needs 30.2 points, not 30.
    high = market.high.copy()
    low = market.low.copy()
    high[1], low[1] = 1090, 1059
    fixture = replace(market, high=high, low=low)
    contract = dict(decision=0, role_end=8, horizon=1, risk_dollars=300,
                    point_value=20, target_r=2, stop_r=1)
    assert label_entry_opportunity(fixture, round_trip_fee=0, **contract)[0] is True
    assert label_entry_opportunity(fixture, round_trip_fee=4, **contract)[0] is False
    high[1], low[1] = 1100, 1040
    fixture = replace(market, high=high, low=low)
    assert label_entry_opportunity(fixture, round_trip_fee=4, **contract) == (False, False)


def test_equal_action_value_does_not_invent_a_direction():
    from dataclasses import replace
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    from propevolve.reasoning_policy.dataset import supervised_record
    from propevolve.reasoning_policy.labels import ActionLabels

    labels = label_actions(environment(), reset_options={"ticker": "NQ", "start": 0},
                          prefix=(), continuation_factory=passive_factory, max_steps=8)
    tied = ActionLabels(labels.observation, {
        action: replace(outcome, reward_to_go=0.0) for action, outcome in labels.outcomes.items()
    })
    history = RollingContext(ContextConfig(1, ("balance",)))
    history.append(1, {"balance": 0})
    result = supervised_record(history.snapshot(), tied, source_id="tie", continuation_id="fixed",
                               target_temperature=1)
    assert result["messages"][-1]["content"] == "WAIT"


def test_source_episode_keeps_its_existing_position_after_labeling():
    env = environment()
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)
    label_actions(env, reset_options={"ticker": "NQ", "start": 0},
                  prefix=(Action.ENTER_LONG_1,), continuation_factory=passive_factory, max_steps=8)
    assert env.valid_actions() == (Action.HOLD, Action.CLOSE)
    _, _, _, _, info = env.step(Action.CLOSE)
    assert info["decision_index"] == 1
    assert info["realized_pnl"] == 1196
