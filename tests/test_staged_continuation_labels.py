"""Management teaching values are executable returns, not hindsight peaks."""
from dataclasses import replace

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.environment import HistoricalChallengeEnv
from test_reasoning_challenger_e2e import environment, passive_factory


def assert_management_teaching(result, expected):
    """Real simulator receipts through serialization and native learning loss."""
    mx = pytest.importorskip("mlx.core")
    from propevolve.reasoning_policy.context import ContextConfig, RollingContext
    from propevolve.reasoning_policy.dataset import supervised_record
    from propevolve.reasoning_policy.supervision import action_targets
    from propevolve.reasoning_policy.staged_batches import binary_targets
    from propevolve.reasoning_policy.staged_learning import trade_objective
    context = RollingContext(ContextConfig(2, ("trade.current_r",), input_mode="embeddings"))
    context.append(1, {"trade.current_r": 0.}, embedding=np.ones(2))
    record = supervised_record(context.snapshot(), result, source_id="management-test",
                               continuation_id="simulator", target_temperature=.5)
    assert record["messages"][-1]["content"] == expected.name
    p, v, w = binary_targets(action_targets(record))
    assert w.tolist() == [0., 0., 1.]
    assert p.sum(axis=1).tolist() == pytest.approx([1., 1., 1.])
    _, gradient = mx.value_and_grad(lambda scores: trade_objective(scores,
        mx.array(p[None]), mx.array(v[None]), mx.array(w[None]),
        {"soft_target_weight": 1., "ranking_weight": 1., "margin": .25}, xp=mx))(mx.zeros((1, 3)))
    mx.eval(gradient)
    assert gradient[0, :2].tolist() == [0., 0.]
    assert (gradient[0, 2].item() < 0) == (expected == Action.HOLD)


def test_collector_rejects_unknown_management_recipe_instead_of_using_oracle_labels():
    from propevolve.reasoning_policy.collector import collect_examples
    with pytest.raises(ValueError, match="management label mode"):
        list(collect_examples(None, reset_options={}, context_config=None, sources=(),
            behavior_factory=passive_factory, continuation_factory=passive_factory,
            source_id="fixture", continuation_id="fixture", maximum_examples=1,
            sample_stride=1, rollout_max_steps=8, target_temperature=.5,
            opportunity_contract={"management_label_mode": "typo"}))


@pytest.mark.parametrize("side,sign", [(Action.ENTER_LONG_1, 1), (Action.ENTER_SHORT_1, -1)])
def test_trailing_continuation_keeps_more_than_four_r_and_records_excursions(side, sign):
    from propevolve.reasoning_policy.labels import label_position_continuation
    env = environment()
    market = env.markets["NQ"]
    path = 1000 + sign * np.array([0., 0., 30., 60., 90., 85., 75., 70.])
    market.open[:] = market.close[:] = path
    market.high[:], market.low[:] = path + .1, path - .1
    env = HistoricalChallengeEnv(env.markets, tick_values=env.tick_values,
        round_trip_fees=env.round_trip_fees,
        spec=replace(env.spec, per_trade_risk_dollars=300,
            ratchet_activation_r=2., ratchet_giveback_r=.75, ratchet_lock_floor_r=2.), seed=7)
    result = label_position_continuation(env,
        reset_options={"ticker": "NQ", "start": 0}, prefix=(side,),
        continuation_factory=passive_factory, max_steps=6, minimum_improvement_r=.1)
    assert result.outcomes[Action.CLOSE].terminal_pnl == pytest.approx(596.)
    assert result.outcomes[Action.HOLD].terminal_pnl == pytest.approx(1496.)
    assert result.outcomes[Action.HOLD].outcome == "ratchet_stop"
    assert result.outcomes[Action.HOLD].reward_to_go > 4.
    evidence = result.management_evidence["HOLD"]
    assert evidence["mfe_r"] > 6.
    assert evidence["mae_r"] >= 0.
    assert evidence["exit_reason"] == "ratchet_stop"
    assert evidence["net_r"] == pytest.approx(1496. / 300.)
    assert_management_teaching(result, Action.HOLD)


@pytest.mark.parametrize("side,sign", [(Action.ENTER_LONG_1, 1), (Action.ENTER_SHORT_1, -1)])
def test_management_label_matches_simulator_exit_not_best_future_open(side, sign):
    from propevolve.reasoning_policy.labels import label_position_continuation
    env = environment()
    market = env.markets["NQ"]
    path = 1000 + sign * np.array([0., 0., 5., 30., 1., 0., 0., 0.])
    market.open[:] = path
    market.close[:] = path
    market.high[:] = path + .1
    market.low[:] = path - .1
    env = HistoricalChallengeEnv(env.markets, tick_values=env.tick_values,
        round_trip_fees=env.round_trip_fees,
        spec=replace(env.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    env.reset(options={"ticker": "NQ", "start": 0})
    result = label_position_continuation(env,
        reset_options={"ticker": "NQ", "start": 0}, prefix=(side,),
        continuation_factory=passive_factory, max_steps=3,
        minimum_improvement_r=.1)
    # CLOSE fills at index 2 (+5 points); HOLD's fixed horizon closes at
    # index 4 (+1 point). Index 3's +30-point open is not clairvoyantly captured.
    assert result.outcomes[Action.CLOSE].terminal_pnl == pytest.approx(96.)
    assert result.outcomes[Action.HOLD].terminal_pnl == pytest.approx(16.)
    assert result.outcomes[Action.HOLD].reward_to_go == pytest.approx(16 / 300 - .1)
    assert result.outcomes[Action.CLOSE].reward_to_go > result.outcomes[Action.HOLD].reward_to_go
    assert result.outcomes[Action.HOLD].outcome_end_ns == int(market.timestamps[4].astype("datetime64[ns]").astype(np.int64))
    assert env.closed_trade_receipts() == ()
    assert_management_teaching(result, Action.CLOSE)
    from propevolve.reasoning_policy.collector import collect_examples
    from propevolve.reasoning_policy.context import ContextConfig
    records = list(collect_examples(env,
        reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("trade.current_r", "trade.mfe_r_so_far",
            "trade.mae_r_so_far", "trade.giveback_r", "trade.hold_bars"), input_mode="embeddings"),
        sources=(), behavior_factory=passive_factory, continuation_factory=passive_factory,
        source_id="fixture", continuation_id="passive-fixed-horizon-v1",
        maximum_examples=1, sample_stride=1, rollout_max_steps=8, target_temperature=.5,
        action_label_mode="trade_mastery_grid", initial_entry_action=side,
        management_only=True, management_sampling="all_states", collect_market_targets=False,
        opportunity_contract={"horizon": 3, "target_rs": [2., 3., 4.], "stop_r": 1.,
            "position_minimum_improvement_r": .1,
            "management_label_mode": "simulator_continuation",
            "utilities": {"winner": 2., "failure": -1., "wait": 0.,
                          "missed_opportunity": -.25, "conflict_margin": .25}}))
    assert records[0]["action"]["messages"][-1]["content"] == "CLOSE"
    assert records[0]["action"]["targets"]["outcomes"]["HOLD"]["terminal_pnl"] == pytest.approx(16.)
    record = records[0]["action"]
    assert record["targets"]["management_evidence"]["HOLD"]["net_r"] == pytest.approx(16 / 300)
    import json
    prompt = json.loads(record["messages"][-2]["content"])
    state = dict(zip(prompt["fields"], prompt["history_oldest_first"][-1]))
    assert state["trade.mfe_r_so_far"] == pytest.approx(.1 / 14.8)
    assert state["trade.mae_r_so_far"] == pytest.approx(.1 / 14.8)
    assert "management_evidence" not in record["messages"][-2]["content"]
    assert "exit_reason" not in record["messages"][-2]["content"]
    # A gap through the initial stop must realize the worse opening fill, not
    # clamp the loss to -1R or report the earlier favorable peak as capture.
    gap = 1000 - sign * 20.
    market.open[3] = market.close[3] = gap
    market.high[3], market.low[3] = gap + .1, gap - .1
    stopped = label_position_continuation(env,
        reset_options={"ticker": "NQ", "start": 0}, prefix=(side,),
        continuation_factory=passive_factory, max_steps=3, minimum_improvement_r=.1)
    assert stopped.outcomes[Action.HOLD].outcome == "initial_stop"
    assert stopped.outcomes[Action.HOLD].terminal_pnl == pytest.approx(-404.)
    assert_management_teaching(stopped, Action.CLOSE)
