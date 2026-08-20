from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from propevolve.decision import Action, PositionSide
from propevolve.environment import (
    ChallengeStartState,
    ChallengeSpec,
    HistoricalChallengeEnv,
    MarketSeries,
    PropChallengeAccount,
    RecoveryEntryPermit,
)
from propevolve.observation import (
    AccountState,
    ObservationAssembler,
    TradeManagementObservationSpec,
)


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


def _recovery_start_state() -> ChallengeStartState:
    return ChallengeStartState(
        realized_pnl=-2_700.0,
        equity_pnl=-2_700.0,
        peak_equity_pnl=0.0,
        mll_floor_pnl=-3_000.0,
        passmark_locked=False,
        position_side=PositionSide.FLAT,
        position_size=0,
        session_pnl=-2_700.0,
        trading_days_elapsed=1,
        recovery_success_pnl=0.0,
        recovery_entry_permit=RecoveryEntryPermit(
            remaining_entries=1,
            exception_headroom=300.0,
            ordinary_entry_resume_pnl=-2_500.0,
        ),
    )


def _recovery_market(
    *,
    opens: tuple[float, ...],
    highs: tuple[float, ...] | None = None,
    lows: tuple[float, ...] | None = None,
    closes: tuple[float, ...] | None = None,
) -> MarketSeries:
    open_values = np.asarray(opens, dtype=np.float32)
    close_values = np.asarray(closes or opens, dtype=np.float32)
    high_values = np.asarray(
        highs or tuple(max(open_, close) for open_, close in zip(opens, close_values)),
        dtype=np.float32,
    )
    low_values = np.asarray(
        lows or tuple(min(open_, close) for open_, close in zip(opens, close_values)),
        dtype=np.float32,
    )
    length = len(open_values)
    return MarketSeries(
        ticker="NQ",
        timestamps=(
            np.datetime64("2024-01-02T23:00")
            + np.arange(length) * np.timedelta64(3, "m")
        ),
        open=open_values,
        high=high_values,
        low=low_values,
        close=close_values,
        embeddings=np.zeros((length, 2), np.float32),
    )


def test_recovery_start_allows_one_entry_and_wait_does_not_consume_it() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )

    _, reset_info = env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    assert reset_info["valid_actions"] == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert reset_info["realized_pnl"] == -2_700.0
    assert reset_info["equity_pnl"] == -2_700.0
    assert reset_info["mll_headroom"] == 300.0
    assert reset_info["mll_headroom_fraction"] == 0.1
    assert reset_info["recovery_entry_permit_remaining"] == 1
    assert "recovery_status" not in reset_info

    _, _, terminated, _, wait_info = env.step(Action.WAIT)

    assert not terminated
    assert wait_info["recovery_entry_permit_remaining"] == 1
    assert wait_info["mll_headroom"] == 300.0
    assert wait_info["mll_headroom_fraction"] == 0.1
    assert wait_info["recovery_wait_decisions"] == 1
    assert wait_info["valid_actions"] == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )


def test_first_recovery_trade_restores_ordinary_entries_without_ending_recovery() -> None:
    market = _recovery_market(opens=(100.0, 100.0, 110.0, 110.0, 110.0, 110.0))
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    _, _, terminated, _, entered = env.step(Action.ENTER_LONG_1)
    _, recovered_reward, terminated, _, recovered = env.step(Action.CLOSE)

    assert not terminated
    assert entered["recovery_entry_permit_remaining"] == 0
    assert entered["recovery_entry_used"] is True
    assert recovered["realized_pnl"] == -2_500.0
    assert recovered["mll_headroom"] == 500.0
    assert recovered["mll_headroom_fraction"] == pytest.approx(1.0 / 6.0)
    assert recovered["recovery_success"] is False
    assert "recovery_status" not in recovered
    assert recovered["ordinary_entry_eligible"] is True
    assert recovered["outcome"] is None
    assert recovered["valid_actions"] == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert recovered_reward == pytest.approx(200.0 / 3_000.0)


def test_first_short_recovery_trade_mirrors_long_recovery_eligibility() -> None:
    market = _recovery_market(opens=(100.0, 100.0, 90.0, 90.0, 90.0, 90.0))
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    env.step(Action.ENTER_SHORT_1)
    _, reward, terminated, _, recovered = env.step(Action.CLOSE)

    assert not terminated
    assert recovered["realized_pnl"] == -2_500.0
    assert recovered["recovery_success"] is False
    assert "recovery_status" not in recovered
    assert recovered["ordinary_entry_eligible"] is True
    assert recovered["valid_actions"] == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )
    assert reward == pytest.approx(200.0 / 3_000.0)


def test_closed_trade_receipts_expose_entry_risk_and_economic_outcome() -> None:
    """Episode diagnostics can join decision-time Regime evidence to trades."""
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    env.step(Action.ENTER_LONG_1)
    env.step(Action.CLOSE)

    assert env.closed_trade_receipts() == ({
        "trade_index": 0,
        "ticker": "NQ",
        "side": "long",
        "source_decision_index": 0,
        "entry_index": 1,
        "exit_index": 2,
        "entry_timestamp": "2024-01-01T00:01",
        "exit_timestamp": "2024-01-01T00:02",
        "entry_realized_pnl": 0.0,
        "entry_mll_floor_pnl": -3_000.0,
        "entry_mll_headroom": 3_000.0,
        "pnl": 20.0,
        "realized_r": pytest.approx(1.0 / 15.0),
        "mfe_r": pytest.approx(2.0 / 15.0),
        "mae_r": pytest.approx(0.5 / 15.0),
        "ratchet_activated": False,
        "exit_reason": "voluntary_close",
        "hold_bars": 1,
    },)


def test_huge_first_recovery_winner_is_a_pass_and_records_recovery_success() -> None:
    market = _recovery_market(
        opens=(100.0, 100.0, 600.0, 600.0, 600.0, 600.0),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    _, _, entered_terminated, _, _ = env.step(Action.ENTER_LONG_1)
    _, reward, terminated, _, recovered = env.step(Action.CLOSE)

    assert entered_terminated is False
    assert terminated is True
    assert recovered["realized_pnl"] == 7_300.0
    assert recovered["recovery_trade_closed"] is True
    assert recovered["recovery_success"] is True
    assert recovered["recovery_status"] == "recovered"
    assert recovered["outcome"] == "pass"
    assert reward == pytest.approx(10_000.0 / 3_000.0 + 250.0 / 1_000.0)


def test_consumed_recovery_permit_blocks_second_exception_but_episode_continues() -> None:
    market = _recovery_market(opens=(100.0,) * 6)
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    env.step(Action.ENTER_LONG_1)
    _, recovery_reward, terminated, _, closed = env.step(Action.CLOSE)

    assert not terminated
    assert closed["realized_pnl"] == -2_700.0
    assert closed["recovery_success"] is False
    assert "recovery_status" not in closed
    assert closed["ordinary_entry_eligible"] is False
    assert closed["outcome"] is None
    assert closed["valid_actions"] == (Action.WAIT,)
    assert recovery_reward == 0.0
    with pytest.raises(ValueError, match="invalid"):
        env.step(Action.ENTER_LONG_1)


def test_recovery_continues_across_ordinary_trades_until_breakeven() -> None:
    market = _recovery_market(
        opens=(100.0, 100.0, 110.0, 110.0, 190.0, 190.0, 235.0, 235.0),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    expected_realized = (-2_500.0, -900.0, 0.0)
    for trade, realized in enumerate(expected_realized):
        _, _, entered_done, _, _ = env.step(Action.ENTER_LONG_1)
        _, _, closed_done, _, info = env.step(Action.CLOSE)
        assert not entered_done
        assert info["realized_pnl"] == realized
        assert not closed_done

    assert info["recovery_success"] is True
    assert info["recovery_status"] == "recovered"
    assert info["outcome"] is None
    assert info["valid_actions"] == (
        Action.WAIT,
        Action.ENTER_LONG_1,
        Action.ENTER_SHORT_1,
    )

    _, _, terminated, _, terminal = env.step(Action.WAIT)

    assert terminated
    assert terminal["outcome"] == "timeout"
    assert terminal["recovery_status"] == "recovered"
    assert terminal["recovery_success"] is True


def test_recovery_episode_can_wait_until_normal_challenge_timeout() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _recovery_market(opens=(100.0,) * 8)},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    for decision in range(1, 8):
        _, reward, terminated, _, info = env.step(Action.WAIT)
        assert info["recovery_wait_decisions"] == decision
        assert terminated is (decision == 7)

    assert info["outcome"] == "timeout"
    assert info["recovery_status"] == "not_recovered"
    assert info["recovery_success"] is False
    assert info["recovery_entry_permit_remaining"] == 1
    assert info["valid_actions"] == ()
    assert reward == pytest.approx(-2.0 / 1_000.0)


def test_recovery_stop_is_fee_inclusive_and_blows_at_the_mll_floor() -> None:
    market = _recovery_market(
        opens=(100.0,) * 6,
        highs=(100.0,) * 6,
        lows=(100.0, 85.2, 100.0, 100.0, 100.0, 100.0),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 4.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    _, _, terminated, _, stopped = env.step(Action.ENTER_LONG_1)

    assert terminated
    assert stopped["outcome"] == "blow"
    assert stopped["exit_reason"] == "initial_stop"
    assert stopped["realized_pnl"] == pytest.approx(-3_000.0)
    assert stopped["fees_paid"] == 4.0
    assert stopped["recovery_entry_permit_remaining"] == 0
    assert stopped["recovery_success"] is False
    assert stopped["recovery_status"] == "not_recovered"


def test_recovery_stop_allows_realistic_gap_through_beyond_max_loss() -> None:
    market = _recovery_market(
        opens=(100.0, 100.0, 80.0, 80.0, 80.0, 80.0),
        highs=(100.0, 100.0, 80.0, 80.0, 80.0, 80.0),
        lows=(100.0, 100.0, 80.0, 80.0, 80.0, 80.0),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        round_trip_fees={"NQ": 4.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={
        "ticker": "NQ",
        "start": 0,
        "challenge_start_state": _recovery_start_state(),
    })

    _, _, terminated, _, _ = env.step(Action.ENTER_LONG_1)
    assert not terminated
    _, _, terminated, _, gapped = env.step(Action.HOLD)

    assert terminated
    assert gapped["outcome"] == "blow"
    assert gapped["realized_pnl"] < -3_000.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"equity_pnl": -2_699.0}, "flat recovery equity"),
        ({"peak_equity_pnl": -2_800.0}, "peak equity"),
        ({"passmark_locked": True}, "passmark lock"),
        (
            {"position_side": PositionSide.LONG, "position_size": 1},
            "start flat",
        ),
        ({"trading_days_elapsed": 0}, "positive integer"),
    ],
)
def test_recovery_start_state_rejects_incomplete_account_state(
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_recovery_start_state(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mll_floor_pnl": -3_100.0}, "MLL floor"),
        ({"peak_equity_pnl": 100.0}, "peak equity must be zero"),
        (
            {
                "recovery_entry_permit": RecoveryEntryPermit(
                    remaining_entries=1,
                    exception_headroom=300.0,
                    ordinary_entry_resume_pnl=-2_400.0,
                )
            },
            "ordinary-entry PnL must restore exactly",
        ),
        (
            {
                "realized_pnl": -2_600.0,
                "equity_pnl": -2_600.0,
                "recovery_entry_permit": RecoveryEntryPermit(
                    remaining_entries=1,
                    exception_headroom=400.0,
                    ordinary_entry_resume_pnl=-2_500.0,
                ),
            },
            "headroom must equal per-trade risk",
        ),
    ],
)
def test_environment_rejects_recovery_state_inconsistent_with_challenge(
    changes: dict,
    message: str,
) -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _recovery_market(opens=(100.0,) * 6)},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            minimum_mll_headroom=500.0,
            per_trade_risk_dollars=300.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )

    with pytest.raises(ValueError, match=message):
        env.reset(options={
            "ticker": "NQ",
            "start": 0,
            "challenge_start_state": replace(_recovery_start_state(), **changes),
        })


def test_reset_rejects_unvalidated_recovery_state_mapping() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _recovery_market(opens=(100.0,) * 6)},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(),
        seed=1,
    )

    with pytest.raises(TypeError, match="ChallengeStartState"):
        env.reset(options={
            "ticker": "NQ",
            "start": 0,
            "challenge_start_state": {"realized_pnl": -2_700.0},
        })


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
    assert "recovery_status" not in info

    _, reward, terminated, truncated, info = env.step(Action.ENTER_LONG_1)

    assert observation.shape == (14,)
    assert terminated and not truncated
    assert info["outcome"] == "pass"
    assert "recovery_status" not in info
    assert info["fill_price"] == 101.0
    assert info["equity_pnl"] == 20.0
    assert info["trade_count"] == 1
    assert info["win_rate"] == 1.0
    assert info["avg_win_r"] == 20.0 / 3_000.0
    assert reward > 0.25


def test_environment_management_observation_uses_only_current_trade_state() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=_spec(
            profit_target=6_000,
            per_trade_risk_dollars=20,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
            ratchet_lock_floor_r=0.0,
        ),
        observation_spec=TradeManagementObservationSpec.entry_risk_v1(
            r_scale=10.0,
            hold_horizon_bars=120,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    observation, _, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert not terminated
    assert info["decision_index"] == 0
    assert info["fill_index"] == 1
    np.testing.assert_allclose(
        observation[-6:],
        np.asarray([0.1, 0.2, 0.1, 0.0, 1.0, 0.15], np.float32),
    )


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


def test_reported_mll_headroom_fraction_is_bounded_after_large_profit() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.01},
        spec=_spec(profit_target=15.0),
        seed=1,
    )
    _, reset_info = env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert reset_info["mll_headroom_fraction"] == 1.0
    assert terminated
    assert info["outcome"] == "pass"
    assert info["mll_headroom"] > env.spec.max_loss
    assert info["mll_headroom_fraction"] == 1.0


def test_flat_wait_does_not_pay_mll_proximity_penalty_after_drawdown() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(6) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 90.0, 90.0, 90.0, 90.0], np.float32),
        high=np.array([101.0, 101.0, 91.0, 91.0, 91.0, 91.0], np.float32),
        low=np.array([99.0, 90.0, 89.0, 89.0, 89.0, 89.0], np.float32),
        close=np.array([100.0, 90.0, 90.0, 90.0, 90.0, 90.0], np.float32),
        embeddings=np.zeros((6, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            max_loss=300.0,
            minimum_mll_headroom=0.0,
            mll_proximity_penalty_coefficient=0.1,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)

    _, close_reward, terminated, _, close_info = env.step(Action.CLOSE)
    _, wait_reward, _, _, wait_info = env.step(Action.WAIT)

    expected_penalty = 0.1 * (1.0 - 100.0 / 300.0) ** 2
    assert not terminated
    assert close_info["equity_pnl"] == -200.0
    assert close_info["mll_proximity_penalty"] == pytest.approx(expected_penalty)
    assert wait_info["mll_proximity_penalty"] == 0.0
    assert close_reward == pytest.approx(-expected_penalty)
    assert wait_reward == 0.0


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
    assert closed["avg_mae_r"] == pytest.approx(0.1)
    assert closed["avg_hold_bars"] == 1.0
    assert closed["expectancy_r"] == pytest.approx(0.5)
    expected_context = {
        "side": "long",
        "entry_index": 1,
        "exit_index": 2,
        "entry_timestamp": "2024-01-02T23:03",
        "exit_timestamp": "2024-01-02T23:06",
        "hold_bars": 1,
        "realized_r": pytest.approx(0.5),
        "mfe_r": pytest.approx(0.8),
        "mae_r": pytest.approx(0.1),
        "ratchet_activated": False,
        "exit_reason": "voluntary_close",
    }
    assert closed["largest_realized_trade"] == expected_context
    assert closed["largest_mfe_trade"] == expected_context


def test_trade_retention_reports_profit_capture_and_round_trips() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(8) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0], np.float32),
        high=np.array([100.0, 120.0, 106.0, 106.0, 106.0, 106.0, 106.0, 106.0], np.float32),
        low=np.array([100.0, 95.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0], np.float32),
        close=np.array([100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0], np.float32),
        embeddings=np.zeros((8, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=3.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)
    _, _, _, _, closed = env.step(Action.CLOSE)

    # The trade touched +2R, experienced 0.5R MAE, and banked only +0.5R.
    assert closed["retention_eligible_count"] == 1
    assert closed["avg_mfe_r"] == pytest.approx(2.0)
    assert closed["avg_mae_r"] == pytest.approx(0.5)
    assert closed["mfe_capture_ratio"] == pytest.approx(0.25)
    assert closed["mfe_realized_gap_r"] == pytest.approx(1.5)
    assert closed["gave_it_all_back_rate"] == 0.0
    assert closed["two_r_eligible_count"] == 1
    assert closed["two_r_mfe_capture_ratio"] == pytest.approx(0.25)


def test_trade_retention_flags_a_real_winner_round_trip() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(8) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0], np.float32),
        high=np.array([100.0, 120.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0], np.float32),
        low=np.array([100.0, 95.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0], np.float32),
        close=np.array([100.0, 105.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0], np.float32),
        embeddings=np.zeros((8, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=3.0,
            ratchet_giveback_r=0.5,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})
    env.step(Action.ENTER_LONG_1)
    _, _, _, _, closed = env.step(Action.CLOSE)

    assert closed["mfe_capture_ratio"] == pytest.approx(-0.25)
    assert closed["gave_it_all_back_rate"] == 1.0
    assert closed["two_r_gave_it_all_back_rate"] == 1.0


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


def test_ratchet_leaves_upside_uncapped_through_a_twelve_r_trend() -> None:
    timestamps = (
        np.datetime64("2024-01-02T23:00")
        + np.arange(6) * np.timedelta64(3, "m")
    )
    market = MarketSeries(
        ticker="NQ",
        timestamps=timestamps,
        open=np.array([100.0, 100.0, 130.0, 180.0, 220.0, 215.0], np.float32),
        high=np.array([101.0, 130.0, 180.0, 220.0, 225.0, 216.0], np.float32),
        low=np.array([99.0, 99.0, 129.0, 179.0, 219.0, 214.0], np.float32),
        close=np.array([100.0, 130.0, 180.0, 220.0, 224.0, 215.0], np.float32),
        embeddings=np.zeros((6, 2), np.float32),
    )
    env = HistoricalChallengeEnv(
        {"NQ": market},
        tick_values={"NQ": 20.0},
        round_trip_fees={"NQ": 0.0},
        spec=_spec(
            profit_target=100_000.0,
            per_trade_risk_dollars=200.0,
            ratchet_activation_r=2.0,
            ratchet_giveback_r=0.5,
            ratchet_lock_floor_r=2.0,
        ),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, at_three_r = env.step(Action.ENTER_LONG_1)
    assert not terminated
    assert at_three_r["ratchet_active"] is True
    assert at_three_r["protective_stop_price"] == 125.0

    _, _, terminated, _, at_eight_r = env.step(Action.HOLD)
    assert not terminated
    assert at_eight_r["trade_count"] == 0
    assert at_eight_r["protective_stop_price"] == 175.0

    _, _, terminated, _, at_twelve_r = env.step(Action.HOLD)
    assert not terminated
    assert at_twelve_r["trade_count"] == 0
    assert at_twelve_r["protective_stop_price"] == 215.0

    _, _, terminated, _, above_twelve_r = env.step(Action.HOLD)
    assert not terminated
    assert above_twelve_r["trade_count"] == 0
    assert above_twelve_r["protective_stop_price"] == 220.0

    _, _, terminated, _, stopped = env.step(Action.HOLD)
    assert terminated
    assert stopped["exit_reason"] == "ratchet_stop"
    assert stopped["avg_win_r"] == pytest.approx(11.5)


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
