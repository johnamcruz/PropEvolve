from __future__ import annotations

import numpy as np
import pytest

from propevolve.decision import Action
from propevolve.environment import (
    ChallengeSpec,
    HistoricalChallengeEnv,
    MarketSeries,
    PropChallengeAccount,
)
from propevolve.observation import AccountState, ObservationAssembler


def _spec(**overrides) -> ChallengeSpec:
    settings = {
        "profit_target": 6_000.0,
        "max_loss": 3_000.0,
        "episode_days": 1,
        "bars_per_day": 6,
        "max_position_size": 1,
        "minimum_mll_headroom": 250.0,
        "trailing_mll_lock": True,
        "terminal_pass_reward": 250.0,
        "terminal_blow_reward": -1_500.0,
        "terminal_timeout_reward": -2.0,
        "terminal_pass_speed_reward_per_day": 20.0,
        "reward_scale": 1_000.0,
    }
    settings.update(overrides)
    return ChallengeSpec(**settings)


def _market(*, low_at_one: float = 100.5) -> MarketSeries:
    values = np.arange(8, dtype=np.float32)
    return MarketSeries(
        ticker="NQ",
        timestamps=np.datetime64("2024-01-01T00:00") + np.arange(8).astype("timedelta64[m]"),
        open=np.array([100, 101, 102, 103, 104, 105, 106, 107], np.float32),
        high=np.array([101, 103, 104, 105, 106, 107, 108, 109], np.float32),
        low=np.array([99, low_at_one, 101, 102, 103, 104, 105, 106], np.float32),
        close=np.array([100, 102, 103, 104, 105, 106, 107, 108], np.float32),
        embeddings=np.stack((values, values + 10), axis=1),
    )


def test_authenticated_cache_defers_embedding_finite_check_to_observation() -> None:
    values = np.arange(8, dtype=np.float32)
    embeddings = np.stack((values, values + 10), axis=1)
    embeddings[0, 0] = np.nan
    market = MarketSeries(
        ticker="NQ",
        timestamps=(
            np.datetime64("2024-01-01T00:00")
            + np.arange(8).astype("timedelta64[m]")
        ),
        open=np.arange(100, 108, dtype=np.float32),
        high=np.arange(101, 109, dtype=np.float32),
        low=np.arange(99, 107, dtype=np.float32),
        close=np.arange(100, 108, dtype=np.float32),
        embeddings=embeddings,
        embeddings_authenticated=True,
    )

    assembler = ObservationAssembler(2, max_loss=3_000, profit_target=6_000)
    with pytest.raises(ValueError, match="embedding must be finite"):
        assembler.assemble(market.embeddings[0], AccountState())


def test_combine_accumulates_pnl_across_trades_and_sessions_until_pass() -> None:
    account = PropChallengeAccount(
        max_loss=3_000.0,
        profit_target=6_000.0,
        trailing_mll_lock=True,
    )

    account.realize(2_000.0)
    account.close_session()
    assert account.cumulative_pnl == 2_000.0
    assert account.outcome() is None

    account.realize(2_000.0)
    account.close_session()
    assert account.cumulative_pnl == 4_000.0
    assert account.outcome() is None

    account.realize(2_000.0)
    assert account.cumulative_pnl == 6_000.0
    assert account.outcome() == "pass"


def test_action_is_filled_on_next_bar_and_can_pass_challenge() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(profit_target=15),
        seed=1,
    )
    observation, info = env.reset(options={"ticker": "NQ", "start": 0})

    _, reward, terminated, truncated, info = env.step(Action.ENTER_LONG_1)

    assert observation.shape == (14,)
    assert terminated and not truncated
    assert info["outcome"] == "pass"
    assert info["fill_price"] == 101.0
    assert info["equity_pnl"] == 20.0
    assert info["trade_count"] == 1
    assert info["win_rate"] == 1.0
    assert info["avg_win_r"] == 20.0 / 3_000.0
    assert reward > 0.25


def test_intrabar_adverse_excursion_enforces_mll_before_close_recovery() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market(low_at_one=80.0)},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(max_loss=300),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert terminated
    assert info["outcome"] == "blow"
    assert info["worst_equity_pnl"] <= -300


def test_round_trip_fee_is_included_before_declaring_a_pass() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 10.0},
        spec=_spec(profit_target=15),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert not terminated
    assert info["equity_pnl"] == 10.0


def test_timeout_occurs_after_exact_cme_trading_session_count() -> None:
    timestamps = np.asarray(
        [
            "2024-01-02T23:00", "2024-01-03T12:00",
            "2024-01-03T23:00", "2024-01-04T12:00",
            "2024-01-04T23:00", "2024-01-05T12:00",
        ],
        dtype="datetime64[m]",
    )
    prices = np.full(len(timestamps), 100.0, dtype=np.float32)
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        embeddings=np.zeros((len(timestamps), 2), dtype=np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        # Deliberately wrong bar estimate: episode duration must be based on
        # exchange sessions rather than episode_days * bars_per_day.
        spec=_spec(episode_days=2, bars_per_day=99),
        seed=1,
    )
    _, reset_info = env.reset(options={"ticker": "NQ", "start": 0})

    assert reset_info["end"] == 3
    for expected_terminated in (False, False, True):
        _, _, terminated, _, info = env.step(Action.WAIT)
        assert terminated is expected_terminated

    assert info["outcome"] == "timeout"
    assert info["trading_days_elapsed"] == 2


def test_mechanical_stop_limits_trade_to_one_r_including_fees() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(4) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.full(4, 100.0, np.float32),
        high=np.full(4, 101.0, np.float32),
        low=np.array([99.0, 90.0, 99.0, 99.0], np.float32),
        close=np.full(4, 100.0, np.float32),
        embeddings=np.zeros((4, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 4.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert not terminated
    assert info["exit_reason"] == "initial_stop"
    assert info["realized_pnl"] == pytest.approx(-200.0)
    assert info["trade_count"] == 1
    assert info["initial_stop_count"] == 1
    assert info["ratchet_stop_count"] == 0
    assert info["voluntary_close_count"] == 0
    assert info["avg_loss_r"] == pytest.approx(-1.0)
    assert info["expectancy_r"] == pytest.approx(-1.0)


def test_ratchet_activates_at_two_r_on_the_following_bar() -> None:
    timestamps = np.datetime64("2024-01-02T23:00") + np.arange(5) * np.timedelta64(3, "m")
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 116.0, 116.0, 116.0], np.float32),
        high=np.array([101.0, 120.0, 117.0, 117.0, 117.0], np.float32),
        # The activation bar trades below the future +1.5R stop. It must not
        # use that newly observed high to exit retroactively on the same bar.
        low=np.array([99.0, 95.0, 114.0, 115.0, 115.0], np.float32),
        close=np.array([100.0, 110.0, 116.0, 116.0, 116.0], np.float32),
        embeddings=np.zeros((5, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, activation = env.step(Action.ENTER_LONG_1)
    assert not terminated
    assert activation["ratchet_active"] is True
    assert activation["protective_stop_price"] == 115.0
    assert activation["trade_count"] == 0

    _, _, terminated, _, stopped = env.step(Action.HOLD)
    assert not terminated
    assert stopped["exit_reason"] == "ratchet_stop"
    assert stopped["realized_pnl"] == 300.0
    assert stopped["avg_win_r"] == 1.5
    assert stopped["winning_r_sum"] == 1.5
    assert stopped["ratchet_stop_count"] == 1
    assert stopped["ratchet_activation_rate"] == 1.0
    assert stopped["activated_avg_realized_r"] == 1.5


def test_voluntary_close_reports_entry_quality_and_winner_retention() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(5) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 105.0, 105.0, 105.0], np.float32),
        high=np.array([101.0, 108.0, 106.0, 106.0, 106.0], np.float32),
        low=np.array([99.0, 99.0, 104.0, 104.0, 104.0], np.float32),
        close=np.array([100.0, 105.0, 105.0, 105.0, 105.0], np.float32),
        embeddings=np.zeros((5, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, _ = env.step(Action.ENTER_LONG_1)
    assert not terminated
    _, _, terminated, _, closed = env.step(Action.CLOSE)

    assert not terminated
    assert closed["exit_reason"] == "voluntary_close"
    assert closed["voluntary_close_count"] == 1
    assert closed["ratchet_stop_count"] == 0
    assert closed["avg_mfe_r"] == pytest.approx(0.8)
    assert closed["avg_hold_bars"] == 1.0
    assert closed["expectancy_r"] == pytest.approx(0.5)


def test_closed_entry_reports_counterfactual_multi_horizon_expansion() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(60) * np.timedelta64(3, "m")
    )
    open_ = np.full(60, 100.0, np.float32)
    high = np.full(60, 105.0, np.float32)
    low = np.full(60, 95.0, np.float32)
    close = np.full(60, 100.0, np.float32)
    high[3] = 120.0
    high[7] = 130.0
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=open_,
        high=high,
        low=low,
        close=close,
        embeddings=np.zeros((60, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            bars_per_day=59,
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)
    _, _, _, _, closed = env.step(Action.CLOSE)

    assert closed["shadow_h5_complete_trades"] == 1
    assert closed["shadow_h5_avg_mfe_r"] == 2.0
    assert closed["shadow_h5_avg_mae_r"] == 0.5
    assert closed["shadow_h5_2r_before_1r_rate"] == 1.0
    assert closed["shadow_h5_3r_before_1r_rate"] == 0.0
    assert closed["shadow_h10_3r_before_1r_rate"] == 1.0
    assert closed["shadow_h50_complete_trades"] == 1


def test_ratchet_lock_floor_protects_two_r_at_activation() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(4) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 116.0, 116.0], np.float32),
        high=np.array([101.0, 120.0, 117.0, 117.0], np.float32),
        low=np.array([99.0, 95.0, 114.0, 115.0], np.float32),
        close=np.array([100.0, 110.0, 116.0, 116.0], np.float32),
        embeddings=np.zeros((4, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
            ratchet_lock_floor_r=2.0,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, activation = env.step(Action.ENTER_LONG_1)

    assert not terminated
    assert activation["ratchet_active"] is True
    assert activation["protective_stop_price"] == 120.0


def test_short_ratchet_uses_an_independent_downside_path() -> None:
    timestamps = np.datetime64("2024-01-02T23:00") + np.arange(5) * np.timedelta64(3, "m")
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 84.0, 84.0, 84.0], np.float32),
        high=np.array([101.0, 105.0, 86.0, 85.0, 85.0], np.float32),
        low=np.array([99.0, 80.0, 83.0, 83.0, 83.0], np.float32),
        close=np.array([100.0, 90.0, 84.0, 84.0, 84.0], np.float32),
        embeddings=np.zeros((5, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, _, _, activation = env.step(Action.ENTER_SHORT_1)
    assert activation["ratchet_active"] is True
    assert activation["protective_stop_price"] == 85.0

    _, _, _, _, stopped = env.step(Action.HOLD)
    assert stopped["exit_reason"] == "ratchet_stop"
    assert stopped["realized_pnl"] == 300.0
    assert stopped["avg_win_r"] == 1.5


def test_short_ratchet_lock_floor_protects_two_r_at_activation() -> None:
    timestamps = np.datetime64("2024-01-02T23:00") + np.arange(4) * np.timedelta64(3, "m")
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 84.0, 84.0], np.float32),
        high=np.array([101.0, 105.0, 86.0, 85.0], np.float32),
        low=np.array([99.0, 80.0, 83.0, 83.0], np.float32),
        close=np.array([100.0, 90.0, 84.0, 84.0], np.float32),
        embeddings=np.zeros((4, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
            ratchet_lock_floor_r=2.0,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, activation = env.step(Action.ENTER_SHORT_1)

    assert not terminated
    assert activation["ratchet_active"] is True
    assert activation["protective_stop_price"] == 80.0


def test_pass_speed_reward_uses_completed_cme_sessions_not_row_count() -> None:
    timestamps = np.asarray(
        [
            "2024-01-02T23:00", "2024-01-03T12:00",
            "2024-01-03T23:00", "2024-01-04T12:00",
        ],
        dtype="datetime64[m]",
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 101.0, 102.0, 103.0], np.float32),
        high=np.array([100.0, 103.0, 104.0, 105.0], np.float32),
        low=np.array([100.0, 101.0, 102.0, 103.0], np.float32),
        close=np.array([100.0, 102.0, 103.0, 104.0], np.float32),
        embeddings=np.zeros((4, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(episode_days=2, bars_per_day=99, profit_target=15.0),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, reward, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert terminated
    assert info["outcome"] == "pass"
    assert info["trading_days_elapsed"] == 1
    assert reward == pytest.approx(20.0 / 3_000.0 + (250.0 + 20.0) / 1_000.0)


def test_reward_shaping_penalizes_mll_proximity_only_while_exposed() -> None:
    timestamps = np.datetime64("2024-01-02T23:00") + np.arange(5) * np.timedelta64(3, "m")
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 90.0, 90.0, 90.0], np.float32),
        high=np.array([100.0, 100.0, 90.0, 90.0, 90.0], np.float32),
        low=np.array([100.0, 90.0, 90.0, 90.0, 90.0], np.float32),
        close=np.array([100.0, 90.0, 90.0, 90.0, 90.0], np.float32),
        embeddings=np.zeros((5, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            max_loss=300.0,
            mll_proximity_penalty_coefficient=0.09,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)

    _, reward, terminated, _, info = env.step(Action.HOLD)

    assert not terminated
    assert info["mll_proximity_penalty"] == pytest.approx(0.04)
    assert reward == pytest.approx(-0.04)


def test_reward_shaping_rewards_only_realized_wins_above_threshold_r() -> None:
    timestamps = np.datetime64("2024-01-02T23:00") + np.arange(5) * np.timedelta64(3, "m")
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 120.0, 120.0, 120.0], np.float32),
        high=np.array([100.0, 120.0, 120.0, 120.0, 120.0], np.float32),
        low=np.array([100.0, 100.0, 120.0, 120.0, 120.0], np.float32),
        close=np.array([100.0, 120.0, 120.0, 120.0, 120.0], np.float32),
        embeddings=np.zeros((5, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=3.0,
            ratchet_giveback_r=0.5,
            large_win_threshold_r=1.5,
            large_win_bonus_coefficient=0.1,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)

    _, reward, _, _, info = env.step(Action.CLOSE)

    assert info["realized_win_r"] == pytest.approx(2.0)
    assert info["large_win_bonus"] == pytest.approx(0.05)
    assert reward == pytest.approx(0.05)


def test_reward_shaping_penalizes_giveback_more_near_the_pass_target() -> None:
    timestamps = np.datetime64("2024-01-02T23:00") + np.arange(6) * np.timedelta64(3, "m")
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 110.0, 110.0, 100.0, 100.0], np.float32),
        high=np.array([100.0, 110.0, 110.0, 110.0, 100.0, 100.0], np.float32),
        low=np.array([100.0, 100.0, 110.0, 100.0, 100.0, 100.0], np.float32),
        close=np.array([100.0, 110.0, 110.0, 100.0, 100.0, 100.0], np.float32),
        embeddings=np.zeros((6, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            profit_target=1_000.0,
            max_loss=300.0,
            lead_giveback_penalty_coefficient=0.3,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)
    env.step(Action.CLOSE)

    _, _, _, _, info = env.step(Action.ENTER_LONG_1)

    # $200 realized progress is 20% of target; the new position gives back
    # $200 from peak equity, or two-thirds of MLL.
    assert info["lead_giveback_penalty"] == pytest.approx(
        0.3 * 0.2**2 * (200.0 / 300.0)
    )
