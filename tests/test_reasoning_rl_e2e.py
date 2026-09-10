"""Real simulator rollouts plus the production economic policy-gradient loss."""

import json

import numpy as np
import pytest

from propevolve.reasoning_policy.context import ContextConfig
from propevolve.reasoning_policy.rl import (
    CHALLENGE_MASTERY_FIELDS,
    clipped_action_loss,
    require_challenge_mastery_context,
    rollout,
    training_rows,
)
from test_reasoning_challenger_e2e import environment
from test_reasoning_collection_evaluation_e2e import sources


class ScriptedRuntime:
    """External-model stand-in; never claim this proves neural learning."""
    def __init__(self, entry):
        self.entry = entry

    def completion_scores(self, messages, completions):
        preferred = "HOLD" if "HOLD" in completions else self.entry
        return {name: 0.0 if name == preferred else -1000.0 for name in completions}


class RecordingRuntime(ScriptedRuntime):
    def __init__(self, entry):
        super().__init__(entry)
        self.requests = []

    def completion_scores(self, messages, completions, **kwargs):
        self.requests.append((messages, tuple(completions), kwargs))
        return super().completion_scores(messages, completions)


def test_challenge_mastery_context_rejects_trade_only_sft_fields():
    context = ContextConfig(
        20,
        ("trade.open", "trade.position_side", "trade.current_r"),
        input_mode="embeddings",
        text_steps=1,
    )
    with pytest.raises(ValueError, match="challenge-mastery context is missing"):
        require_challenge_mastery_context(context)


def test_rl_receives_prop_state_costs_time_and_legal_actions_end_to_end():
    runtime = RecordingRuntime("WAIT")
    context = ContextConfig(
        20, CHALLENGE_MASTERY_FIELDS, input_mode="embeddings", text_steps=1,
    )
    require_challenge_mastery_context(context)

    decisions, terminal = rollout(
        runtime,
        environment(),
        options={"ticker": "NQ", "start": 0},
        context_config=context,
        sources=sources(),
        rng=np.random.default_rng(7),
        max_steps=8,
    )

    assert terminal["outcome"] == "timeout"
    assert len(runtime.requests) == len(decisions)
    prompt = json.loads(runtime.requests[0][0][1]["content"])
    assert tuple(prompt["fields"]) == CHALLENGE_MASTERY_FIELDS
    latest = dict(zip(prompt["fields"], prompt["history_oldest_first"][-1]))
    assert latest["challenge.profit_target_dollars"] == 6_000.0
    assert latest["challenge.max_loss_dollars"] == 3_000.0
    assert latest["challenge.target_remaining_dollars"] == 6_000.0
    assert latest["challenge.headroom_dollars"] == 3_000.0
    assert latest["account.challenge_remaining"] == 1.0
    assert latest["account.point_value_norm"] == pytest.approx(20.0 / 3_000.0)
    assert latest["account.round_trip_fee_norm"] == pytest.approx(4.0 / 3_000.0)
    assert prompt["legal_actions"] == ["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
    assert "market_context" in runtime.requests[0][2]


def test_complete_pass_and_blow_generate_opposite_learning_pressure():
    trajectories = []
    for entry, expected in (("ENTER_LONG_1", "pass"), ("ENTER_SHORT_1", "blow")):
        decisions, terminal = rollout(ScriptedRuntime(entry), environment(),
            options={"ticker": "NQ", "start": 0},
            context_config=ContextConfig(20, ("account.realized_pnl_norm",)),
            sources=sources(), rng=np.random.default_rng(7), max_steps=8)
        assert terminal["outcome"] == expected
        assert all(row.actions[row.selected] in row.actions for row in decisions)
        trajectories.append(decisions)
    rows = training_rows(trajectories, advantage_scale=1.0)
    winner, failure = rows[0], rows[len(trajectories[0])]
    assert winner[1] > 0
    assert failure[1] < 0
    # Independent finite-difference reference: increasing chosen action's logit
    # must lower winner loss and raise failure loss, with others held fixed.
    old = np.log(np.array([1 / 3, 1 / 3, 1 / 3]))
    def loss(logit, advantage):
        logits = np.array([logit, 0.0, 0.0])
        new = logits - np.log(np.exp(logits).sum())
        return clipped_action_loss(new, old, 0, advantage, clip_epsilon=0.2,
                                   kl_weight=0.0, entropy_weight=0.0, xp=np)
    assert loss(0.001, winner[1]) < loss(0.0, winner[1])
    assert loss(0.001, failure[1]) > loss(0.0, failure[1])


def test_wait_rollout_reaches_real_timeout_without_fabricating_pass():
    decisions, terminal = rollout(ScriptedRuntime("WAIT"), environment(),
        options={"ticker": "NQ", "start": 0},
        context_config=ContextConfig(20, ("account.realized_pnl_norm",)),
        sources=sources(), rng=np.random.default_rng(7), max_steps=8)
    assert terminal["outcome"] == "timeout"
    assert terminal["realized_pnl"] == 0
    assert {row.actions[row.selected] for row in decisions} == {"WAIT"}


def test_resource_limit_is_not_a_training_timeout_label():
    with pytest.raises(ValueError, match="incomplete"):
        rollout(ScriptedRuntime("WAIT"), environment(), options={"ticker": "NQ", "start": 0},
            context_config=ContextConfig(20, ("account.realized_pnl_norm",)), sources=sources(),
            rng=np.random.default_rng(7), max_steps=1)


def test_clipped_update_cannot_keep_rewarding_already_oversized_ratio():
    old = np.log(np.array([0.25, 0.75]))
    new = np.log(np.array([0.50, 0.50]))
    assert clipped_action_loss(new, old, 0, 1.0, clip_epsilon=0.2,
                               kl_weight=0, entropy_weight=0, xp=np) == pytest.approx(-1.2)
