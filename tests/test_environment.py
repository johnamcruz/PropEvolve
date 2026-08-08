from __future__ import annotations

import numpy as np

from propevolve.decision import Action
from propevolve.environment import ChallengeSpec, HistoricalChallengeEnv, MarketSeries


class Ledger:
    def __init__(self) -> None:
        self.cumulative_pnl = 0.0

    def close_trade(self, ticker: str, gross_pnl: float, size: int) -> float:
        self.cumulative_pnl += gross_pnl
        return gross_pnl


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


def test_action_is_filled_on_next_bar_and_can_pass_challenge() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market()},
        ledger_factory=Ledger,
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=ChallengeSpec(profit_target=15, max_loss=3_000, episode_days=1, bars_per_day=6),
        seed=1,
    )
    observation, info = env.reset(options={"ticker": "NQ", "start": 0})

    _, reward, terminated, truncated, info = env.step(Action.ENTER_LONG_1)

    assert observation.shape == (14,)
    assert terminated and not truncated
    assert info["outcome"] == "pass"
    assert info["fill_price"] == 101.0
    assert info["equity_pnl"] == 20.0
    assert reward > 1.0


def test_intrabar_adverse_excursion_enforces_mll_before_close_recovery() -> None:
    env = HistoricalChallengeEnv(
        {"NQ": _market(low_at_one=80.0)},
        ledger_factory=Ledger,
        round_trip_fees={"NQ": 0.0},
        tick_values={"NQ": 20.0},
        spec=ChallengeSpec(profit_target=6_000, max_loss=300, episode_days=1, bars_per_day=6),
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
        spec=ChallengeSpec(profit_target=15, max_loss=3_000, episode_days=1, bars_per_day=6),
        seed=1,
    )
    env.reset(options={"ticker": "NQ", "start": 0})

    _, _, terminated, _, info = env.step(Action.ENTER_LONG_1)

    assert not terminated
    assert info["equity_pnl"] == 10.0
