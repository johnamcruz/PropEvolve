"""Anchors taken from the Expansion + order-flow rule.

What these guard: that the policy is taught the SAME events algoTraderAI's PPO trades,
with the side the flow implies, that it is shown declined bars as Wait so it cannot learn
to always enter, and that anchors never sit where an episode could not run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from propevolve.decision import Action
from propevolve.environment import ChallengeSpec, HistoricalChallengeEnv, MarketSeries
from propevolve.reasoning_policy.setup_episodes import (
    setup_anchor_census,
    setup_episode_specs,
)
from propevolve.setup_signals import CHANNEL_NAMES

WIDTH = len(CHANNEL_NAMES)
_TRIGGER = CHANNEL_NAMES.index("setup_trigger")
_SIDE = CHANNEL_NAMES.index("setup_side")
_AVAILABLE = CHANNEL_NAMES.index("setup_available")
_ARMED = CHANNEL_NAMES.index("expansion_armed")
N = 200


def _channels(longs=(20, 40), shorts=(60,), armed=range(15, 90), available=True):
    channels = np.zeros((N, WIDTH), dtype=np.float32)
    channels[:, _AVAILABLE] = 1.0 if available else 0.0
    for row in armed:
        channels[row, _ARMED] = 1.0
    for row in longs:
        channels[row, _TRIGGER] = 1.0
        channels[row, _SIDE] = 1.0
        channels[row, _ARMED] = 1.0
    for row in shorts:
        channels[row, _TRIGGER] = 1.0
        channels[row, _SIDE] = -1.0
        channels[row, _ARMED] = 1.0
    return channels


def _market(channels):
    stamps = pd.date_range("2024-03-05 14:33", periods=N, freq="1h").to_numpy("datetime64[ns]")
    close = np.full(N, 100.0, dtype=np.float32)
    return MarketSeries(ticker="NQ", timestamps=stamps, open=close.copy(),
                        high=close + 1, low=close - 1, close=close,
                        embeddings=np.zeros((N, 4), dtype=np.float32),
                        setup_channels=channels)


def _env(channels):
    spec = ChallengeSpec(
        profit_target=6_000.0, max_loss=3_000.0, episode_days=2, bars_per_day=4,
        max_position_size=1, minimum_mll_headroom=500.0, trailing_mll_lock=True,
        terminal_pass_reward=250.0, terminal_blow_reward=-1_500.0,
        terminal_timeout_reward=-2.0, terminal_pass_speed_reward_per_day=20.0,
        reward_scale=1_000.0, per_trade_risk_dollars=500.0,
        ratchet_activation_r=2.0, ratchet_giveback_r=0.5, ratchet_lock_floor_r=2.0)
    return HistoricalChallengeEnv({"NQ": _market(channels)}, tick_values={"NQ": 20.0},
                                  spec=spec, round_trip_fees={"NQ": 3.84}, seed=1)


def _config(**sampling):
    base = {"per_action": 10, "seed": 7}
    base.update(sampling)
    return {"tickers": {"train": ["NQ"]}, "setup_action_sampling": {"train": base}}


def test_every_trigger_becomes_an_entry_anchor_on_the_flow_side():
    env = _env(_channels())
    specs = setup_episode_specs(_config(), env, "train")
    by_row = {s["start"]: s["expected_action"] for s in specs}
    assert by_row[20] == int(Action.ENTER_LONG_1)
    assert by_row[40] == int(Action.ENTER_LONG_1)
    assert by_row[60] == int(Action.ENTER_SHORT_1)


def test_declined_bars_are_taught_as_wait_so_the_policy_can_refuse():
    specs = setup_episode_specs(_config(), _env(_channels()), "train")
    actions = [s["expected_action"] for s in specs]
    assert int(Action.WAIT) in actions
    assert actions.count(int(Action.WAIT)) > 0


def test_the_default_wait_pool_is_the_hard_negatives_inside_armed_windows():
    """Bars where Expansion armed but the flow declined are the informative refusals;
    an unarmed bar teaches almost nothing."""
    channels = _channels(armed=range(15, 30))
    specs = setup_episode_specs(_config(), _env(channels), "train")
    waits = [s["start"] for s in specs if s["expected_action"] == int(Action.WAIT)]
    assert waits and all(channels[row, _ARMED] > 0.0 for row in waits)


def test_the_any_wait_pool_widens_beyond_armed_windows():
    channels = _channels(armed=range(15, 30))
    narrow = setup_episode_specs(_config(wait_pool="armed"), _env(channels), "train")
    wide = setup_episode_specs(_config(wait_pool="any", per_action=50),
                               _env(channels), "train")
    count = lambda specs: sum(1 for s in specs if s["expected_action"] == int(Action.WAIT))
    assert count(wide) > count(narrow)


def test_unavailable_bars_are_never_anchored():
    """No Expansion score or no flow means no setup; anchoring there teaches noise."""
    with pytest.raises(ValueError, match="no anchors"):
        setup_episode_specs(_config(per_action=50), _env(_channels(available=False)), "train")


def test_anchors_leave_room_for_a_full_episode():
    late = _channels(longs=(20, N - 3), shorts=())
    specs = setup_episode_specs(_config(), _env(late), "train")
    starts = [s["start"] for s in specs]
    assert N - 3 not in starts and 20 in starts


def test_per_action_caps_each_class_and_is_reproducible():
    channels = _channels(longs=tuple(range(20, 50)), shorts=(60,))
    env = _env(channels)
    first = setup_episode_specs(_config(per_action=5), env, "train")
    again = setup_episode_specs(_config(per_action=5), env, "train")
    longs = [s for s in first if s["expected_action"] == int(Action.ENTER_LONG_1)]
    assert len(longs) == 5 and first == again


def test_a_market_without_channels_is_rejected():
    stamps = pd.date_range("2024-03-05 14:33", periods=N, freq="1h").to_numpy("datetime64[ns]")
    close = np.full(N, 100.0, dtype=np.float32)
    plain = MarketSeries(ticker="NQ", timestamps=stamps, open=close.copy(), high=close + 1,
                         low=close - 1, close=close,
                         embeddings=np.zeros((N, 4), dtype=np.float32))
    spec = _env(_channels()).spec
    env = HistoricalChallengeEnv({"NQ": plain}, tick_values={"NQ": 20.0}, spec=spec,
                                 round_trip_fees={"NQ": 3.84}, seed=1)
    with pytest.raises(ValueError, match="no setup channels"):
        setup_episode_specs(_config(), env, "train")


def test_malformed_sampling_is_rejected():
    env = _env(_channels())
    with pytest.raises(ValueError, match="per_action and seed"):
        setup_episode_specs({"tickers": {"train": ["NQ"]},
                             "setup_action_sampling": {"train": {"seed": 1}}}, env, "train")
    with pytest.raises(ValueError, match="unsupported"):
        setup_episode_specs(_config(nonsense=1), env, "train")
    with pytest.raises(ValueError, match="wait_pool"):
        setup_episode_specs(_config(wait_pool="everything"), env, "train")


def test_the_census_reports_what_the_anchors_were_drawn_from():
    census = setup_anchor_census(_env(_channels()), "NQ")
    assert census["triggers"] == 3
    assert census["long_triggers"] == 2 and census["short_triggers"] == 1
    assert census["bars"] == N and census["available"] == N
