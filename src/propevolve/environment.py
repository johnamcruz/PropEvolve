"""Thirty-day Monte Carlo prop challenge driven by frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .decision import Action, ActionMasker, PositionSide
from .observation import AccountState, ObservationAssembler


@dataclass(frozen=True)
class ChallengeSpec:
    profit_target: float = 6_000.0
    max_loss: float = 3_000.0
    episode_days: int = 30
    bars_per_day: int = 130
    max_position_size: int = 2
    terminal_pass_reward: float = 1.0
    terminal_blow_reward: float = -1.0
    terminal_timeout_reward: float = -0.10

    def __post_init__(self) -> None:
        if min(
            self.profit_target,
            self.max_loss,
            self.episode_days,
            self.bars_per_day,
            self.max_position_size,
        ) <= 0:
            raise ValueError("challenge economics and durations must be positive")

    @property
    def episode_bars(self) -> int:
        return self.episode_days * self.bars_per_day


@dataclass(frozen=True)
class MarketSeries:
    ticker: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        lengths = {
            len(self.timestamps), len(self.open), len(self.high), len(self.low),
            len(self.close), len(self.embeddings),
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
            raise ValueError("market arrays must have one common length of at least two")
        prices = np.column_stack((self.open, self.high, self.low, self.close))
        if self.embeddings.ndim != 2 or not np.isfinite(prices).all():
            raise ValueError("market prices and embeddings must be finite matrices")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("market prices and embeddings must be finite matrices")
        if (
            (self.high < np.maximum(self.open, self.close)).any()
            or (self.low > np.minimum(self.open, self.close)).any()
            or (self.high < self.low).any()
        ):
            raise ValueError("market OHLC relationships are invalid")
        if not np.all(self.timestamps[1:] > self.timestamps[:-1]):
            raise ValueError("market timestamps must be strictly increasing")


class TradeLedger(Protocol):
    cumulative_pnl: float

    def close_trade(self, ticker: str, gross_pnl: float, size: int) -> float: ...


class PropFirmLedger:
    """Cumulative prop-challenge balance and round-trip fee ledger."""

    ROUND_TRIP_FEES = {
        "NQ": 3.78,
        "ES": 3.78,
        "GC": 4.32,
        "RTY": 3.78,
        "YM": 3.78,
        "CL": 4.02,
        "SI": 4.32,
        "ZB": 2.76,
        "ZN": 2.58,
    }

    def __init__(self, round_trip_fees: dict[str, float] | None = None) -> None:
        self.cumulative_pnl = 0.0
        self.round_trip_fees = dict(round_trip_fees or self.ROUND_TRIP_FEES)

    def close_trade(self, ticker: str, gross_pnl: float, size: int) -> float:
        if ticker not in self.round_trip_fees:
            raise ValueError(f"missing round-trip fee for {ticker}")
        net = float(gross_pnl) - self.round_trip_fees[ticker] * int(size)
        self.cumulative_pnl += net
        return net


@dataclass
class _Position:
    side: PositionSide
    size: int
    average_entry: float


class HistoricalChallengeEnv:
    """Sample challenge windows and execute decisions at the next bar open."""

    def __init__(
        self,
        markets: dict[str, MarketSeries],
        *,
        tick_values: dict[str, float],
        spec: ChallengeSpec | None = None,
        ledger_factory: Callable[[], TradeLedger] | None = None,
        round_trip_fees: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        if not markets:
            raise ValueError("at least one market is required")
        embedding_dims = {market.embeddings.shape[1] for market in markets.values()}
        if len(embedding_dims) != 1:
            raise ValueError("all markets must use one frozen embedding width")
        missing = set(markets) - set(tick_values)
        if missing:
            raise ValueError(f"missing tick values for {sorted(missing)}")
        self.markets = dict(markets)
        self.tick_values = {key: float(value) for key, value in tick_values.items()}
        self.spec = spec or ChallengeSpec()
        self.round_trip_fees = dict(round_trip_fees or PropFirmLedger.ROUND_TRIP_FEES)
        missing_fees = set(markets) - set(self.round_trip_fees)
        if missing_fees and ledger_factory is None:
            raise ValueError(f"missing round-trip fees for {sorted(missing_fees)}")
        self._ledger_factory = (
            ledger_factory
            if ledger_factory is not None
            else lambda: PropFirmLedger(self.round_trip_fees)
        )
        self._rng = np.random.default_rng(seed)
        self._assembler = ObservationAssembler(
            next(iter(embedding_dims)),
            max_loss=self.spec.max_loss,
            profit_target=self.spec.profit_target,
        )
        self._masker = ActionMasker(
            max_position_size=self.spec.max_position_size,
            max_loss=self.spec.max_loss,
        )
        self._market: MarketSeries | None = None
        self._ticker = ""
        self._index = 0
        self._start = 0
        self._end = 0
        self._ledger: TradeLedger | None = None
        self._position: _Position | None = None
        self._peak_equity = 0.0
        self._terminated = True
        self._primary_side = "flat"

    def reset(self, *, options: dict | None = None) -> tuple[np.ndarray, dict]:
        options = options or {}
        ticker = str(options.get("ticker") or self._rng.choice(tuple(self.markets)))
        market = self.markets[ticker]
        maximum_start = len(market.close) - 2
        desired = min(self.spec.episode_bars, len(market.close) - 1)
        maximum_start = len(market.close) - desired - 1
        start = int(options.get("start", self._rng.integers(0, max(1, maximum_start + 1))))
        if start < 0 or start > maximum_start:
            raise ValueError("episode start cannot fit the challenge window")
        self._market = market
        self._ticker = ticker
        self._index = start
        self._start = start
        self._end = start + desired
        self._ledger = self._ledger_factory()
        self._position = None
        self._peak_equity = 0.0
        self._terminated = False
        self._primary_side = "flat"
        return self._observation(), {
            "ticker": ticker,
            "start": start,
            "end": self._end,
            "valid_actions": self.valid_actions(),
        }

    def valid_actions(self) -> tuple[Action, ...]:
        return self._masker.valid_actions(self._account_state())

    def step(self, action: Action | int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._terminated or self._market is None or self._ledger is None:
            raise RuntimeError("reset the challenge before stepping")
        action = Action(int(action))
        if action not in self.valid_actions():
            raise ValueError(f"action {action.name} is invalid in the current state")
        previous_equity = self._equity(self._market.close[self._index])
        next_index = self._index + 1
        fill = float(self._market.open[next_index])
        info: dict = {"decision_index": self._index, "fill_index": next_index}
        self._apply_action(action, fill, info)

        worst_price = self._adverse_price(next_index)
        worst_equity = self._equity(worst_price)
        info["worst_equity_pnl"] = worst_equity
        self._index = next_index
        outcome = None
        if worst_equity <= -self.spec.max_loss:
            self._liquidate(worst_price)
            outcome = "blow"
        else:
            current_equity = self._equity(float(self._market.close[self._index]))
            if current_equity >= self.spec.profit_target:
                self._liquidate(float(self._market.close[self._index]))
                outcome = "pass"
            elif self._index >= self._end:
                self._liquidate(float(self._market.close[self._index]))
                outcome = "timeout"

        equity = self._equity(float(self._market.close[self._index]))
        self._peak_equity = max(self._peak_equity, equity)
        reward = (equity - previous_equity) / self.spec.max_loss
        if outcome == "pass":
            reward += self.spec.terminal_pass_reward
        elif outcome == "blow":
            reward += self.spec.terminal_blow_reward
        elif outcome == "timeout":
            reward += self.spec.terminal_timeout_reward
        self._terminated = outcome is not None
        info.update({
            "outcome": outcome,
            "equity_pnl": equity,
            "ticker": self._ticker,
            "primary_side": self._primary_side,
            "timestamp": str(self._market.timestamps[self._index]),
            "valid_actions": () if self._terminated else self.valid_actions(),
        })
        return self._observation(), float(reward), self._terminated, False, info

    def _apply_action(self, action: Action, fill: float, info: dict) -> None:
        position = self._position
        if position is None:
            entries = {
                Action.ENTER_LONG_1: (PositionSide.LONG, 1),
                Action.ENTER_LONG_2: (PositionSide.LONG, 2),
                Action.ENTER_SHORT_1: (PositionSide.SHORT, 1),
                Action.ENTER_SHORT_2: (PositionSide.SHORT, 2),
            }
            if action in entries:
                side, size = entries[action]
                self._position = _Position(side, size, fill)
                self._primary_side = "long" if side == PositionSide.LONG else "short"
                info.update({"fill_price": fill, "fill_side": self._primary_side, "fill_size": size})
            return
        if action == Action.CLOSE:
            self._liquidate(fill)
            info.update({"fill_price": fill, "fill_side": "close", "fill_size": position.size})
        elif action == Action.REDUCE_1:
            self._close_size(fill, 1)
            info.update({"fill_price": fill, "fill_side": "reduce", "fill_size": 1})
        elif action == Action.ADD_1:
            new_size = position.size + 1
            position.average_entry = (
                position.average_entry * position.size + fill
            ) / new_size
            position.size = new_size
            info.update({"fill_price": fill, "fill_side": "add", "fill_size": 1})

    def _close_size(self, price: float, size: int) -> None:
        if self._position is None or self._ledger is None:
            return
        size = min(size, self._position.size)
        points = (price - self._position.average_entry) * int(self._position.side)
        gross = points * self.tick_values[self._ticker] * size
        self._ledger.close_trade(self._ticker, gross, size)
        self._position.size -= size
        if self._position.size == 0:
            self._position = None

    def _liquidate(self, price: float) -> None:
        if self._position is not None:
            self._close_size(price, self._position.size)

    def _adverse_price(self, index: int) -> float:
        assert self._market is not None
        if self._position is None:
            return float(self._market.close[index])
        return float(
            self._market.low[index]
            if self._position.side == PositionSide.LONG
            else self._market.high[index]
        )

    def _equity(self, price: float) -> float:
        realized = 0.0 if self._ledger is None else float(self._ledger.cumulative_pnl)
        if self._position is None:
            return realized
        points = (price - self._position.average_entry) * int(self._position.side)
        close_fee = self.round_trip_fees.get(self._ticker, 0.0) * self._position.size
        return (
            realized
            + points * self.tick_values[self._ticker] * self._position.size
            - close_fee
        )

    def _account_state(self) -> AccountState:
        assert self._market is not None
        realized = 0.0 if self._ledger is None else float(self._ledger.cumulative_pnl)
        equity = self._equity(float(self._market.close[self._index]))
        side = PositionSide.FLAT if self._position is None else self._position.side
        size = 0 if self._position is None else self._position.size
        elapsed = self._index - self._start
        bars_left = max(0, self._end - self._index)
        return AccountState(
            realized_pnl=realized,
            equity_pnl=equity,
            peak_equity_pnl=self._peak_equity,
            position_side=side,
            position_size=size,
            max_position_size=self.spec.max_position_size,
            unrealized_pnl=equity - realized,
            session_remaining=(self.spec.bars_per_day - (elapsed % self.spec.bars_per_day))
            / self.spec.bars_per_day,
            challenge_remaining=bars_left / max(1, self.spec.episode_bars),
            point_value=self.tick_values[self._ticker],
            round_trip_fee=self.round_trip_fees.get(self._ticker, 0.0),
        )

    def _observation(self) -> np.ndarray:
        assert self._market is not None
        return self._assembler.assemble(
            self._market.embeddings[self._index], self._account_state()
        )
