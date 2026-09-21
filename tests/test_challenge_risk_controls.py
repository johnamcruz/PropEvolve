"""Two hard risk controls the challenge environment was missing.

Simulated against this spec, the frozen flow rule passes 41.7% of 30-day challenges and
BLOWS 46.2% of them: it earns +$259.77 a trade over five years but draws down $25,745
against a $3,000 floor. algoTraderAI runs the same signal at 55% pass with zero blow on
every seed, and its zero-blow foundation carries two controls PropEvolve's ChallengeSpec
did not have at all -- a daily loss limit ($750 soft / $1,500 hard) and a loss-streak
cooldown (K=2 losses, 65-bar pause).

Both are entry blocks, so they constrain the reasoning policy and the RL stage alike
without either having to learn the constraint from scratch. Neither ever touches an open
position: liquidating on a counter fresh loss would turn a risk control into a forced
market order at the worst moment. Both are off by default, so every existing run and its
receipts are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propevolve.environment import (
    Action, ChallengeSpec, HistoricalChallengeEnv, MarketSeries, PositionSide)


def _market(n=400, drift=0.0):
    stamps = pd.date_range("2024-03-05 14:33", periods=n, freq="1h").to_numpy("datetime64[ns]")
    close = (100.0 + drift * np.arange(n)).astype(np.float32)
    return MarketSeries(
        ticker="NQ", timestamps=stamps, open=close.copy(), high=close + 1,
        low=close - 1, close=close, embeddings=np.zeros((n, 4), dtype=np.float32))


def _spec(**over):
    base = dict(
        profit_target=6_000.0, max_loss=3_000.0, episode_days=4, bars_per_day=10,
        max_position_size=1, minimum_mll_headroom=500.0, trailing_mll_lock=True,
        terminal_pass_reward=250.0, terminal_blow_reward=-1_500.0,
        terminal_timeout_reward=-2.0, terminal_pass_speed_reward_per_day=20.0,
        reward_scale=1_000.0, per_trade_risk_dollars=500.0,
        ratchet_activation_r=2.0, ratchet_giveback_r=0.5, ratchet_lock_floor_r=2.0)
    base.update(over)
    return ChallengeSpec(**base)


def _env(spec, drift=0.0):
    env = HistoricalChallengeEnv(
        {"NQ": _market(drift=drift)}, tick_values={"NQ": 20.0}, spec=spec,
        round_trip_fees={"NQ": 3.84}, seed=1)
    env.reset()
    return env


# ───────────────────────── the spec accepts the controls
def test_the_spec_accepts_a_daily_loss_limit():
    spec = _spec(daily_loss_limit_dollars=1_500.0)
    assert spec.daily_loss_limit_dollars == 1_500.0


def test_the_spec_accepts_a_loss_streak_cooldown():
    spec = _spec(loss_streak_cooldown_trades=2, loss_streak_cooldown_bars=65)
    assert spec.loss_streak_cooldown_trades == 2
    assert spec.loss_streak_cooldown_bars == 65


def test_both_controls_are_off_by_default():
    """Existing runs and their receipts must be untouched."""
    spec = _spec()
    assert spec.daily_loss_limit_dollars is None
    assert spec.loss_streak_cooldown_trades is None


def test_a_negative_daily_loss_limit_is_rejected():
    with pytest.raises(ValueError):
        _spec(daily_loss_limit_dollars=-1.0)


def test_a_cooldown_without_a_bar_count_is_rejected():
    """A streak trigger with no pause length is a silent no-op, not a control."""
    with pytest.raises(ValueError):
        _spec(loss_streak_cooldown_trades=2, loss_streak_cooldown_bars=0)


# ───────────────────────── daily loss limit
def test_the_daily_limit_blocks_a_new_entry_once_the_day_is_lost():
    env = _env(_spec(daily_loss_limit_dollars=500.0))
    env._session_realized_loss = 600.0          # already past the day's limit
    env.step(Action.ENTER_LONG_1)
    assert env._position is None


def test_the_daily_limit_allows_entries_while_the_day_is_intact():
    env = _env(_spec(daily_loss_limit_dollars=500.0))
    env._session_realized_loss = 100.0
    env.step(Action.ENTER_LONG_1)
    assert env._position is not None


def test_the_daily_limit_never_closes_an_open_position():
    """Liquidating on a fresh loss would force a market order at the worst moment."""
    env = _env(_spec(daily_loss_limit_dollars=500.0))
    env.step(Action.ENTER_LONG_1)
    assert env._position is not None
    env._session_realized_loss = 10_000.0
    env.step(Action.HOLD)
    assert env._position is not None


def test_the_daily_loss_resets_on_a_new_session():
    env = _env(_spec(daily_loss_limit_dollars=500.0))
    env._session_realized_loss = 900.0
    before = env._session_realized_loss
    env._begin_session_if_new(force=True)
    assert before > 0 and env._session_realized_loss == 0.0


# ───────────────────────── loss-streak cooldown
def test_the_cooldown_blocks_entries_after_the_streak():
    env = _env(_spec(loss_streak_cooldown_trades=2, loss_streak_cooldown_bars=5))
    env._loss_streak = 2
    env._cooldown_until_index = env._index + 5
    env.step(Action.ENTER_LONG_1)
    assert env._position is None


def test_the_cooldown_expires():
    env = _env(_spec(loss_streak_cooldown_trades=2, loss_streak_cooldown_bars=5))
    env._loss_streak = 2
    env._cooldown_until_index = env._index - 1      # already elapsed
    env.step(Action.ENTER_LONG_1)
    assert env._position is not None


def test_a_winning_trade_clears_the_streak():
    env = _env(_spec(loss_streak_cooldown_trades=2, loss_streak_cooldown_bars=5))
    env._loss_streak = 1
    env._record_trade_result(+250.0)
    assert env._loss_streak == 0


def test_consecutive_losses_accumulate_and_arm_the_cooldown():
    env = _env(_spec(loss_streak_cooldown_trades=2, loss_streak_cooldown_bars=5))
    env._record_trade_result(-100.0)
    assert env._loss_streak == 1 and env._cooldown_until_index is None
    env._record_trade_result(-100.0)
    assert env._loss_streak == 2
    assert env._cooldown_until_index == env._index + 5


def test_nothing_is_blocked_when_the_controls_are_off():
    """Regression: the default environment must behave exactly as before."""
    env = _env(_spec())
    env._session_realized_loss = 99_999.0
    env._loss_streak = 99
    env.step(Action.ENTER_LONG_1)
    assert env._position is not None
