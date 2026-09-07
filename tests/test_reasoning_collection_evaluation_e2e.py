"""Exercise production collection and evaluation, not isolated label helpers."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from propevolve.decision import Action
from propevolve.environment import HistoricalChallengeEnv
from propevolve.reasoning_policy.collector import collect_examples
from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.evaluation import evaluate_policy
from test_reasoning_challenger_e2e import environment, passive_factory


def sources():
    # External specialist contract: fixed causal scores, not future outcomes.
    return tuple(SimpleNamespace(kind=kind, channels=("probability",),
                                targets=SimpleNamespace(target=lambda ticker, row: np.array([0.5])))
                 for kind in ("expansion", "trend", "regime"))


def test_collection_handles_reset_partial_context_and_preserves_future_isolation():
    original = environment()
    env = HistoricalChallengeEnv(original.markets, tick_values=original.tick_values,
        round_trip_fees=original.round_trip_fees,
        spec=replace(original.spec, per_trade_risk_dollars=300), seed=7)
    examples = list(collect_examples(
        env, reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(20, ("account.realized_pnl_norm", "trade.mae_r_so_far")),
        sources=sources(), behavior_factory=passive_factory, continuation_factory=passive_factory,
        source_id="fixture", continuation_id="passive", maximum_examples=1, sample_stride=1,
        rollout_max_steps=8, target_temperature=1.0,
        opportunity_contract={"horizon": 2, "target_r": 2.0, "stop_r": 1.0},
    ))
    assert len(examples) == 1
    record = examples[0]["action"]
    assert record["completed_at_ns"] == int(env.markets["NQ"].timestamps[0].astype("datetime64[ns]").astype(np.int64))
    assert record["label_end_ns"] > record["completed_at_ns"]
    assert "outcomes" not in record["messages"][1]["content"]


def test_policy_evaluation_can_pass_in_unchanged_simulator_from_reset():
    class LongAndHold:
        def decide(self, context, legal_actions):
            assert context.available.any()
            action = Action.HOLD if Action.HOLD in legal_actions else Action.ENTER_LONG_1
            return action, {}
    result = evaluate_policy(LongAndHold(), environment(),
        episodes=[{"ticker": "NQ", "start": 0}],
        context_config=ContextConfig(20, ("account.realized_pnl_norm",)),
        sources=sources(), max_steps=8)
    assert result["pass_rate"] == 1.0
    assert result["blow_rate"] == 0.0
    assert result["mean_terminal_pnl"] >= 6000
    assert result["teacher_free"] is False
