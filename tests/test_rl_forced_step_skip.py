"""RL must not spend a model call on a bar where only one outcome is reachable.

A challenge episode is 30 days x 480 bars = 14,400 steps, and ``rollout`` asked the
policy on every one of them. With ``gate_entries`` on, a bar where the policy is flat
and the rule has not triggered can only end in WAIT -- the gate declines any entry -- so
the call cannot change the episode. Triggered bars are ~1.3% of all bars, so skipping
forced ones is the difference between roughly an hour per episode and a few minutes.

This is a semi-MDP reformulation: decisions are recorded only where a choice exists, and
reward earned on forced bars accrues to the decision that led into them, so the
undiscounted return that ``training_rows`` consumes is unchanged. That conservation is
the property worth pinning, because silently dropping per-bar penalties (the MLL
proximity term accrues every bar) would bias the advantage without failing anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propevolve.environment import (
    Action, ChallengeSpec, HistoricalChallengeEnv, MarketSeries)
from propevolve.setup_signals import CHANNEL_NAMES, SetupSignalSpec

WIDTH = len(CHANNEL_NAMES)
TRIGGER = CHANNEL_NAMES.index("setup_trigger")
SIDE = CHANNEL_NAMES.index("setup_side")
AVAILABLE = CHANNEL_NAMES.index("setup_available")


def _channels(n, *, trigger_every=None, side=1.0):
    rows = np.zeros((n, WIDTH), dtype=np.float32)
    rows[:, AVAILABLE] = 1.0
    if trigger_every:
        rows[::trigger_every, TRIGGER] = 1.0
        rows[::trigger_every, SIDE] = side
    return rows


def _market(n=240, channels=None):
    stamps = pd.date_range("2024-03-05 14:33", periods=n, freq="1h").to_numpy("datetime64[ns]")
    close = np.full(n, 100.0, dtype=np.float32)
    return MarketSeries(
        ticker="NQ", timestamps=stamps, open=close.copy(), high=close + 1,
        low=close - 1, close=close, embeddings=np.zeros((n, 4), dtype=np.float32),
        setup_channels=channels)


def _spec():
    return ChallengeSpec(
        profit_target=6_000.0, max_loss=3_000.0, episode_days=2, bars_per_day=10,
        max_position_size=1, minimum_mll_headroom=500.0, trailing_mll_lock=True,
        terminal_pass_reward=250.0, terminal_blow_reward=-1_500.0,
        terminal_timeout_reward=-2.0, terminal_pass_speed_reward_per_day=20.0,
        reward_scale=1_000.0, per_trade_risk_dollars=500.0,
        ratchet_activation_r=2.0, ratchet_giveback_r=0.5, ratchet_lock_floor_r=2.0)


def _env(channels, *, gate):
    return HistoricalChallengeEnv(
        {"NQ": _market(channels=channels)}, tick_values={"NQ": 20.0}, spec=_spec(),
        round_trip_fees={"NQ": 3.84}, seed=1,
        setup_signals=SetupSignalSpec(state="expansion_flow_v1", gate_entries=gate))


def test_a_forced_bar_is_recognised_when_flat_and_untriggered():
    from propevolve.reasoning_policy.rl import decision_is_forced
    env = _env(_channels(240, trigger_every=None), gate=True)
    env.reset()
    assert decision_is_forced(env) is True


def test_a_triggered_bar_is_not_forced():
    from propevolve.reasoning_policy.rl import decision_is_forced
    env = _env(_channels(240, trigger_every=1), gate=True)
    env.reset()
    assert decision_is_forced(env) is False


def test_an_open_position_is_never_forced():
    """Management is always the policy's call, trigger or not."""
    from propevolve.reasoning_policy.rl import decision_is_forced
    env = _env(_channels(240, trigger_every=1), gate=True)
    env.reset()
    env.step(Action.ENTER_LONG_1)
    assert env._position is not None
    assert decision_is_forced(env) is False


def test_nothing_is_forced_when_the_gate_is_off():
    """Ungated, the policy owns direction, so every flat bar is a real choice."""
    from propevolve.reasoning_policy.rl import decision_is_forced
    env = _env(_channels(240, trigger_every=None), gate=False)
    env.reset()
    assert decision_is_forced(env) is False


def test_forced_skipping_is_off_without_the_gate():
    """The optimisation is only sound because the gate makes the outcome predetermined."""
    from propevolve.reasoning_policy.rl import decision_is_forced
    env = _env(_channels(240, trigger_every=4), gate=False)
    env.reset()
    assert decision_is_forced(env) is False
