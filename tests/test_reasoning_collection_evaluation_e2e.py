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
        spec=replace(original.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
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


def test_action_collection_in_embedding_mode_never_reads_specialists():
    class ForbiddenTargets:
        def target(self, ticker, row):
            raise AssertionError("action collection requested a training-only specialist")

    forbidden = tuple(SimpleNamespace(kind=kind, channels=("probability",), targets=ForbiddenTargets())
                      for kind in ("expansion", "trend", "regime"))
    base = environment()
    original = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees,
        spec=replace(base.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    config = ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings")
    examples = list(collect_examples(
        original, reset_options={"ticker": "NQ", "start": 0},
        context_config=config, sources=forbidden,
        behavior_factory=passive_factory, continuation_factory=passive_factory,
        source_id="fixture", continuation_id="passive", maximum_examples=1,
        sample_stride=1, rollout_max_steps=8, target_temperature=1.0,
        opportunity_contract={"horizon": 2, "target_r": 2.0, "stop_r": 1.0},
        collect_market_targets=False,
    ))
    assert examples[0]["action"]["market_embeddings"]
    assert examples[0]["market"] is None


def test_action_corpus_joins_existing_specialists_and_multi_r_economics_as_targets_only():
    base = environment(1)
    env = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees,
        spec=replace(base.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    record = next(collect_examples(
        env, reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings"),
        sources=sources(), behavior_factory=passive_factory,
        continuation_factory=passive_factory, source_id="fixture",
        continuation_id="market-barrier-grid", maximum_examples=1, sample_stride=1,
        rollout_max_steps=8, target_temperature=1.0,
        opportunity_contract={
            "horizon": 2, "target_rs": [2.0, 3.0, 4.0], "stop_r": 1.0,
            "utilities": {"winner": 2.0, "failure": -1.0, "wait": 0.0,
                          "missed_opportunity": -0.25, "conflict_margin": 0.25},
        }, action_label_mode="market_barrier_grid", collect_market_targets=False,
        augment_action_targets=True,
    ))["action"]
    assert record["ticker"] == "NQ"
    assert record["targets"]["target_before_stop_by_r"]["4"] == {
        "long": True, "short": False,
    }
    assert record["targets"]["future_excursions"]["long"]["mfe_r_gross"] >= 4.0
    assert set(record["targets"]["specialist_targets"]) == {
        "expansion.probability", "trend.probability", "regime.probability",
    }
    prompt = record["messages"][1]["content"]
    assert "specialist_targets" not in prompt
    assert "future_excursions" not in prompt


def test_market_collection_does_not_compute_expensive_action_counterfactuals():
    class ForbiddenContinuation:
        def __call__(self):
            raise AssertionError("market-only collection requested action continuation")

    base = environment()
    env = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees,
        spec=replace(base.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    examples = list(collect_examples(
        env, reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings"),
        sources=sources(), behavior_factory=passive_factory,
        continuation_factory=ForbiddenContinuation(), source_id="fixture",
        continuation_id="unused", maximum_examples=1, sample_stride=1,
        rollout_max_steps=8, target_temperature=1.0,
        opportunity_contract={"horizon": 2, "target_r": 2.0, "stop_r": 1.0},
        collect_action_targets=False, collect_market_targets=True,
    ))
    assert examples[0]["action"] is None
    assert examples[0]["market"]["targets"]["specialist_targets"]


def test_corpus_skips_unavailable_teacher_rows_without_calling_them_wait():
    class InitiallyUnavailable:
        def target(self, ticker, row):
            return None if row == 0 else np.array([0.5])

    delayed = tuple(SimpleNamespace(kind=kind, channels=("probability",),
                                    targets=InitiallyUnavailable())
                    for kind in ("expansion", "trend", "regime"))
    base = environment()
    env = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees,
        spec=replace(base.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    record = next(collect_examples(
        env, reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings"),
        sources=delayed, behavior_factory=passive_factory,
        continuation_factory=passive_factory, source_id="fixture", continuation_id="unused",
        maximum_examples=1, sample_stride=1, rollout_max_steps=8, target_temperature=1.0,
        opportunity_contract={"horizon": 2, "target_r": 2.0, "stop_r": 1.0},
        collect_action_targets=False, collect_market_targets=True,
    ))["market"]
    assert record["completed_at_ns"] == int(
        env.markets["NQ"].timestamps[1].astype("datetime64[ns]").astype(np.int64))


def test_market_collection_teaches_multi_r_capture_and_excursions():
    base = environment(1)
    env = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees,
        spec=replace(base.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    record = next(collect_examples(
        env, reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings"),
        sources=sources(), behavior_factory=passive_factory,
        continuation_factory=passive_factory, source_id="fixture", continuation_id="unused",
        maximum_examples=1, sample_stride=1, rollout_max_steps=8, target_temperature=1.0,
        opportunity_contract={
            "horizon": 2, "target_rs": [2.0, 3.0, 4.0], "stop_r": 1.0,
            "utilities": {"winner": 2.0, "failure": -1.0, "wait": 0.0,
                          "missed_opportunity": -0.25, "conflict_margin": 0.25},
        }, collect_action_targets=False, collect_market_targets=True,
    ))["market"]
    assert record["targets"]["target_before_stop_by_r"] == {
        "2": {"long": True, "short": False},
        "3": {"long": True, "short": False},
        "4": {"long": True, "short": False},
    }
    assert record["targets"]["future_excursions"]["long"]["mfe_r_gross"] >= 4.0


def test_scratch_action_collection_uses_market_economics_without_a_continuation():
    class ForbiddenContinuation:
        def __call__(self):
            raise AssertionError("scratch action labels requested a continuation policy")

    base = environment(1)
    env = HistoricalChallengeEnv(base.markets, tick_values=base.tick_values,
        round_trip_fees=base.round_trip_fees,
        spec=replace(base.spec, per_trade_risk_dollars=300,
                     ratchet_activation_r=10, ratchet_giveback_r=1), seed=7)
    examples = list(collect_examples(
        env, reset_options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(2, ("account.realized_pnl_norm",), input_mode="embeddings"),
        sources=(), behavior_factory=passive_factory,
        continuation_factory=ForbiddenContinuation(), source_id="fixture",
        continuation_id="market-barrier-grid", maximum_examples=1, sample_stride=1,
        rollout_max_steps=8, target_temperature=1.0, collection_warmup_steps=1,
        action_label_mode="market_barrier_grid",
        opportunity_contract={
            "horizon": 2, "target_rs": [2.0, 3.0, 4.0], "stop_r": 1.0,
            "utilities": {"winner": 2.0, "failure": -1.0, "wait": 0.0,
                          "missed_opportunity": -0.25, "conflict_margin": 0.25},
        }, collect_market_targets=False,
    ))
    record = examples[0]["action"]
    assert record["market_available"] == [True, True]
    values = {name: outcome["reward_to_go"]
              for name, outcome in record["targets"]["outcomes"].items()}
    assert values["ENTER_LONG_1"] > values["WAIT"] > values["ENTER_SHORT_1"]


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
