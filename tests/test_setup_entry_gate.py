"""With ``gate_entries`` on, the rule supplies the side and the policy supplies the rest.

The expansion-flow v2 run measured direction at 0.4905 balanced accuracy over 4000
iterations: the reasoning policy cannot produce a side. The flow rule can, and its side
(``sign(persistence_45)``) is the profitable part of the setup. Gating hands the rule's
side to any entry the policy asks for, which removes direction from the policy's job and
leaves selection and management measurable on their own.

Overriding rather than refusing is deliberate. Refusing an entry whose side disagrees
would charge the policy for a direction mistake it was never going to win, and so would
conflate direction with selection -- the exact confound the gate exists to remove.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propevolve.environment import (
    Action, ChallengeSpec, HistoricalChallengeEnv, MarketSeries, PositionSide)
from propevolve.setup_signals import CHANNEL_NAMES, SetupSignalSpec

WIDTH = len(CHANNEL_NAMES)
TRIGGER = CHANNEL_NAMES.index("setup_trigger")
SIDE = CHANNEL_NAMES.index("setup_side")
AVAILABLE = CHANNEL_NAMES.index("setup_available")


def _channels(n, *, side, trigger=1.0, available=1.0):
    """Every bar carries the same rule verdict, so the gate is exercised wherever we land."""
    rows = np.zeros((n, WIDTH), dtype=np.float32)
    rows[:, AVAILABLE] = available
    rows[:, TRIGGER] = trigger
    rows[:, SIDE] = side
    return rows


def _market(n=40, channels=None, ticker="NQ"):
    stamps = pd.date_range("2024-03-05 14:33", periods=n, freq="1h").to_numpy("datetime64[ns]")
    close = np.full(n, 100.0, dtype=np.float32)
    return MarketSeries(
        ticker=ticker, timestamps=stamps,
        open=close.copy(), high=close + 1, low=close - 1, close=close,
        embeddings=np.zeros((n, 4), dtype=np.float32), setup_channels=channels)


def _spec():
    return ChallengeSpec(
        profit_target=6_000.0, max_loss=3_000.0, episode_days=2, bars_per_day=10,
        max_position_size=1, minimum_mll_headroom=500.0, trailing_mll_lock=True,
        terminal_pass_reward=250.0, terminal_blow_reward=-1_500.0,
        terminal_timeout_reward=-2.0, terminal_pass_speed_reward_per_day=20.0,
        reward_scale=1_000.0, per_trade_risk_dollars=500.0,
        ratchet_activation_r=2.0, ratchet_giveback_r=0.5, ratchet_lock_floor_r=2.0)


def _env(channels, *, gate):
    signals = SetupSignalSpec(state="expansion_flow_v1", gate_entries=gate)
    env = HistoricalChallengeEnv(
        {"NQ": _market(channels=channels)}, tick_values={"NQ": 20.0}, spec=_spec(),
        round_trip_fees={"NQ": 3.84}, seed=1, setup_signals=signals)
    env.reset()
    return env


def _side_after(env, action):
    env.step(action)
    return None if env._position is None else env._position.side


def test_the_gate_overrides_a_long_request_with_the_rules_short_side():
    env = _env(_channels(40, side=-1.0), gate=True)
    assert _side_after(env, Action.ENTER_LONG_1) is PositionSide.SHORT


def test_the_gate_overrides_a_short_request_with_the_rules_long_side():
    env = _env(_channels(40, side=1.0), gate=True)
    assert _side_after(env, Action.ENTER_SHORT_1) is PositionSide.LONG


def test_the_gate_agrees_silently_when_the_policy_already_matches():
    env = _env(_channels(40, side=1.0), gate=True)
    assert _side_after(env, Action.ENTER_LONG_1) is PositionSide.LONG


def test_an_untriggered_bar_opens_nothing_even_though_the_policy_asked():
    """No trigger means the rule has no side to lend, so there is no gated entry to make."""
    env = _env(_channels(40, side=1.0, trigger=0.0), gate=True)
    assert _side_after(env, Action.ENTER_LONG_1) is None


def test_an_unavailable_bar_opens_nothing():
    env = _env(_channels(40, side=1.0, available=0.0), gate=True)
    assert _side_after(env, Action.ENTER_LONG_1) is None


def test_a_flat_rule_side_opens_nothing():
    env = _env(_channels(40, side=0.0), gate=True)
    assert _side_after(env, Action.ENTER_LONG_1) is None


def test_without_the_gate_the_policys_own_side_is_honoured():
    """Regression: the ungated configuration must be unchanged by this feature."""
    env = _env(_channels(40, side=-1.0), gate=False)
    assert _side_after(env, Action.ENTER_LONG_1) is PositionSide.LONG


def test_the_gate_does_not_touch_an_open_position():
    """Gating is an entry rule. Management stays entirely the policy's decision."""
    env = _env(_channels(40, side=1.0), gate=True)
    env.step(Action.ENTER_LONG_1)
    assert env._position is not None
    env.step(Action.HOLD)
    assert env._position is not None and env._position.side is PositionSide.LONG


def test_waiting_is_never_turned_into_an_entry_by_the_gate():
    """The rule lends a side to an entry the policy chose; it never chooses to enter."""
    env = _env(_channels(40, side=1.0), gate=True)
    assert _side_after(env, Action.WAIT) is None
