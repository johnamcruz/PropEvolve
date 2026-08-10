"""Thirty-day Monte Carlo prop challenge driven by frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .decision import Action, ActionMasker, PositionSide
from .observation import AccountState, ObservationAssembler


@dataclass(frozen=True)
class ChallengeSpec:
    profit_target: float
    max_loss: float
    episode_days: int
    bars_per_day: int
    max_position_size: int
    minimum_mll_headroom: float
    trailing_mll_lock: bool
    terminal_pass_reward: float
    terminal_blow_reward: float
    terminal_timeout_reward: float
    terminal_pass_speed_reward_per_day: float
    reward_scale: float
    per_trade_risk_dollars: float | None = None
    ratchet_activation_r: float | None = None
    ratchet_giveback_r: float | None = None
    ratchet_lock_floor_r: float = 0.0
    mll_proximity_penalty_coefficient: float = 0.0
    lead_giveback_penalty_coefficient: float = 0.0
    large_win_threshold_r: float = 2.0
    large_win_bonus_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.profit_target,
            self.max_loss,
            self.episode_days,
            self.bars_per_day,
            self.max_position_size,
            self.reward_scale,
        ) <= 0:
            raise ValueError("challenge economics and durations must be positive")
        if self.max_position_size != 1:
            raise ValueError("PropEvolve v1 supports exactly one contract")
        if self.terminal_pass_reward <= 0 or self.terminal_blow_reward >= 0:
            raise ValueError("terminal pass and blow rewards must have opposite signs")
        if min(
            self.mll_proximity_penalty_coefficient,
            self.lead_giveback_penalty_coefficient,
            self.large_win_threshold_r,
            self.large_win_bonus_coefficient,
            self.ratchet_lock_floor_r,
        ) < 0:
            raise ValueError("reward-shaping settings must be nonnegative")
        ratchet = (
            self.per_trade_risk_dollars,
            self.ratchet_activation_r,
            self.ratchet_giveback_r,
        )
        if any(value is not None for value in ratchet):
            if any(value is None for value in ratchet):
                raise ValueError("trade risk and ratchet settings must be declared together")
            if (
                float(self.per_trade_risk_dollars) <= 0
                or float(self.ratchet_giveback_r) <= 0
                or float(self.ratchet_activation_r) <= float(self.ratchet_giveback_r)
                or float(self.ratchet_lock_floor_r)
                > float(self.ratchet_activation_r)
            ):
                raise ValueError("trade risk and ratchet settings are invalid")
        elif self.ratchet_lock_floor_r > 0:
            raise ValueError("ratchet lock floor requires declared trade risk")
        if self.large_win_bonus_coefficient > 0 and self.per_trade_risk_dollars is None:
            raise ValueError("large-win reward requires declared per-trade risk")

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
        if np.isnat(self.timestamps).any():
            raise ValueError("market timestamps must be finite")


_CENTRAL = ZoneInfo("America/Chicago")


def _cme_session_keys(timestamps: np.ndarray) -> np.ndarray:
    """Precompute DST-aware CME session keys; the session rolls at 5 PM CT."""
    nanoseconds = np.asarray(timestamps).astype("datetime64[ns]").astype(np.int64)
    keys = np.empty(len(nanoseconds), dtype=np.int32)
    for index, value in enumerate(nanoseconds):
        local = datetime.fromtimestamp(value / 1_000_000_000, timezone.utc).astimezone(
            _CENTRAL
        )
        session_day = local.date() if local.hour >= 17 else local.date() - timedelta(days=1)
        keys[index] = session_day.toordinal()
    return keys


class PropChallengeAccount:
    """Cumulative challenge balance with EOD trailing and passmark-lock MLL."""

    def __init__(
        self,
        *,
        max_loss: float,
        profit_target: float,
        trailing_mll_lock: bool,
    ) -> None:
        if max_loss <= 0 or profit_target <= 0:
            raise ValueError("challenge economics must be positive")
        self.max_loss = float(max_loss)
        self.profit_target = float(profit_target)
        self.trailing_mll_lock = bool(trailing_mll_lock)
        self.realized_pnl = 0.0
        self.session_pnl = 0.0
        self.mll_floor_pnl = -self.max_loss
        self.passmark_locked = False

    @property
    def cumulative_pnl(self) -> float:
        return self.realized_pnl

    def realize(self, net_pnl: float) -> float:
        value = float(net_pnl)
        self.realized_pnl += value
        self.session_pnl += value
        if (
            self.trailing_mll_lock
            and not self.passmark_locked
            and self.realized_pnl >= self.max_loss
        ):
            self.passmark_locked = True
            self.mll_floor_pnl = 0.0
        return value

    def close_session(self) -> None:
        if self.trailing_mll_lock and not self.passmark_locked:
            self.mll_floor_pnl = min(
                max(self.mll_floor_pnl, self.realized_pnl - self.max_loss),
                0.0,
            )
        self.session_pnl = 0.0

    def mll_headroom(self, equity_pnl: float | None = None) -> float:
        equity = self.realized_pnl if equity_pnl is None else float(equity_pnl)
        return max(0.0, equity - self.mll_floor_pnl)

    def outcome(self, equity_pnl: float | None = None) -> str | None:
        equity = self.realized_pnl if equity_pnl is None else float(equity_pnl)
        if equity <= self.mll_floor_pnl:
            return "blow"
        if equity >= self.profit_target:
            return "pass"
        return None


@dataclass
class _Position:
    side: PositionSide
    size: int
    average_entry: float
    initial_risk_points: float | None = None
    protective_stop: float | None = None
    peak_favorable_r: float = 0.0
    ratchet_active: bool = False


class HistoricalChallengeEnv:
    """Sample challenge windows and execute decisions at the next bar open."""

    def __init__(
        self,
        markets: dict[str, MarketSeries],
        *,
        tick_values: dict[str, float],
        spec: ChallengeSpec,
        round_trip_fees: dict[str, float],
        seed: int,
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
        self.spec = spec
        self.round_trip_fees = dict(round_trip_fees)
        missing_fees = set(markets) - set(self.round_trip_fees)
        if missing_fees:
            raise ValueError(f"missing round-trip fees for {sorted(missing_fees)}")
        if self.spec.per_trade_risk_dollars is not None and any(
            fee >= self.spec.per_trade_risk_dollars
            for fee in self.round_trip_fees.values()
        ):
            raise ValueError("round-trip fee must be smaller than per-trade risk")
        self._session_keys = {
            ticker: _cme_session_keys(market.timestamps)
            for ticker, market in self.markets.items()
        }
        self._rng = np.random.default_rng(seed)
        self._assembler = ObservationAssembler(
            next(iter(embedding_dims)),
            max_loss=self.spec.max_loss,
            profit_target=self.spec.profit_target,
        )
        self._masker = ActionMasker(
            max_position_size=self.spec.max_position_size,
            max_loss=self.spec.max_loss,
            minimum_mll_headroom=self.spec.minimum_mll_headroom,
        )
        self._market: MarketSeries | None = None
        self._ticker = ""
        self._index = 0
        self._start = 0
        self._end = 0
        self._account: PropChallengeAccount | None = None
        self._position: _Position | None = None
        self._peak_equity = 0.0
        self._terminated = True
        self._primary_side = "flat"
        self._closed_trade_pnls: list[float] = []
        self._trading_days_elapsed = 0

    def rng_state(self) -> dict:
        """Return the episode-sampling RNG state for exact boundary recovery."""
        return copy.deepcopy(self._rng.bit_generator.state)

    def restore_rng_state(self, state: dict) -> None:
        """Restore episode sampling only while no episode is active."""
        if not self._terminated:
            raise RuntimeError("environment RNG can only be restored between episodes")
        self._rng.bit_generator.state = copy.deepcopy(state)

    def reset(self, *, options: dict | None = None) -> tuple[np.ndarray, dict]:
        options = options or {}
        ticker = str(options.get("ticker") or self._rng.choice(tuple(self.markets)))
        market = self.markets[ticker]
        session_keys = self._session_keys[ticker]
        unique_sessions = np.unique(session_keys)
        if len(unique_sessions) < self.spec.episode_days:
            raise ValueError(
                f"market {ticker} cannot fit {self.spec.episode_days} trading days"
            )
        last_start_session = unique_sessions[-self.spec.episode_days]
        maximum_start = min(
            len(market.close) - 2,
            int(np.searchsorted(session_keys, last_start_session, side="right") - 1),
        )
        start = int(options.get("start", self._rng.integers(0, max(1, maximum_start + 1))))
        if start < 0 or start > maximum_start:
            raise ValueError("episode start cannot fit the challenge window")
        start_session_index = int(np.searchsorted(unique_sessions, session_keys[start]))
        final_session = unique_sessions[
            start_session_index + self.spec.episode_days - 1
        ]
        end = int(np.searchsorted(session_keys, final_session, side="right") - 1)
        if end <= start:
            raise ValueError("episode start does not leave a causal decision step")
        self._market = market
        self._ticker = ticker
        self._index = start
        self._start = start
        self._end = end
        self._account = PropChallengeAccount(
            max_loss=self.spec.max_loss,
            profit_target=self.spec.profit_target,
            trailing_mll_lock=self.spec.trailing_mll_lock,
        )
        self._position = None
        self._peak_equity = 0.0
        self._terminated = False
        self._primary_side = "flat"
        self._closed_trade_pnls = []
        self._trading_days_elapsed = 1
        return self._observation(), {
            "ticker": ticker,
            "start": start,
            "end": self._end,
            "valid_actions": self.valid_actions(),
        }

    def valid_actions(self) -> tuple[Action, ...]:
        return self._masker.valid_actions(self._account_state())

    def step(self, action: Action | int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._terminated or self._market is None or self._account is None:
            raise RuntimeError("reset the challenge before stepping")
        action = Action(int(action))
        if action not in self.valid_actions():
            raise ValueError(f"action {action.name} is invalid in the current state")
        previous_equity = self._equity(self._market.close[self._index])
        closed_trade_count = len(self._closed_trade_pnls)
        next_index = self._index + 1
        fill = float(self._market.open[next_index])
        info: dict = {
            "decision_index": self._index,
            "fill_index": next_index,
            "fees_paid": 0.0,
        }
        if self._session_keys[self._ticker][next_index] != self._session_keys[self._ticker][self._index]:
            self._account.close_session()
            self._trading_days_elapsed += 1
            info["session_boundary"] = True
        self._apply_action(action, fill, info)

        self._apply_protective_stop(next_index, info)

        worst_price = self._adverse_price(next_index)
        worst_equity = self._equity(worst_price)
        info["worst_equity_pnl"] = worst_equity
        self._index = next_index
        outcome = None
        if self._account.outcome(worst_equity) == "blow":
            self._liquidate(worst_price, info)
            outcome = "blow"
        else:
            current_equity = self._equity(float(self._market.close[self._index]))
            if self._account.outcome(current_equity) == "pass":
                self._liquidate(float(self._market.close[self._index]), info)
                outcome = "pass"
            elif self._index >= self._end:
                self._liquidate(float(self._market.close[self._index]), info)
                outcome = "timeout"
        if outcome is None and self._position is not None:
            self._update_ratchet(next_index, info)

        equity = self._equity(float(self._market.close[self._index]))
        reward = (equity - previous_equity) / self.spec.max_loss
        shaping_reward, shaping_info = self._reward_shaping(
            equity,
            closed_trade_count=closed_trade_count,
        )
        reward += shaping_reward
        self._peak_equity = max(self._peak_equity, equity)
        if outcome == "pass":
            # Match the proven environment's continuous faster-pass shaping;
            # session keys govern the hard 30-day termination contract.
            days_saved = max(
                0.0,
                self.spec.episode_days - self._trading_days_elapsed,
            )
            reward += (
                self.spec.terminal_pass_reward
                + self.spec.terminal_pass_speed_reward_per_day * days_saved
            ) / self.spec.reward_scale
        elif outcome == "blow":
            reward += self.spec.terminal_blow_reward / self.spec.reward_scale
        elif outcome == "timeout":
            reward += self.spec.terminal_timeout_reward / self.spec.reward_scale
        self._terminated = outcome is not None
        info.update({
            "outcome": outcome,
            "equity_pnl": equity,
            "ticker": self._ticker,
            "primary_side": self._primary_side,
            "realized_pnl": self._account.realized_pnl,
            "mll_floor_pnl": self._account.mll_floor_pnl,
            "mll_headroom": self._account.mll_headroom(equity),
            "passmark_locked": self._account.passmark_locked,
            "timestamp": str(self._market.timestamps[self._index]),
            "trading_days_elapsed": self._trading_days_elapsed,
            "valid_actions": () if self._terminated else self.valid_actions(),
            **shaping_info,
            **self._trade_statistics(),
        })
        return self._observation(), float(reward), self._terminated, False, info

    def _reward_shaping(
        self,
        equity: float,
        *,
        closed_trade_count: int,
    ) -> tuple[float, dict[str, float]]:
        proximity_penalty = 0.0
        lead_giveback_penalty = 0.0
        if self._position is not None and self._account is not None:
            cushion_fraction = min(
                1.0,
                self._account.mll_headroom(equity) / self.spec.max_loss,
            )
            proximity_penalty = (
                self.spec.mll_proximity_penalty_coefficient
                * (1.0 - cushion_fraction) ** 2
            )
            progress = min(
                1.0,
                max(0.0, self._account.realized_pnl / self.spec.profit_target),
            )
            drawdown_fraction = max(0.0, self._peak_equity - equity) / self.spec.max_loss
            lead_giveback_penalty = (
                self.spec.lead_giveback_penalty_coefficient
                * progress**2
                * drawdown_fraction
            )

        realized_win_r = 0.0
        large_win_bonus = 0.0
        if len(self._closed_trade_pnls) > closed_trade_count:
            closed_pnl = self._closed_trade_pnls[-1]
            risk = self.spec.per_trade_risk_dollars or self.spec.max_loss
            realized_win_r = max(0.0, closed_pnl / risk)
            large_win_bonus = self.spec.large_win_bonus_coefficient * max(
                0.0,
                realized_win_r - self.spec.large_win_threshold_r,
            )

        return (
            large_win_bonus - proximity_penalty - lead_giveback_penalty,
            {
                "mll_proximity_penalty": proximity_penalty,
                "lead_giveback_penalty": lead_giveback_penalty,
                "realized_win_r": realized_win_r,
                "large_win_bonus": large_win_bonus,
            },
        )

    def _apply_action(self, action: Action, fill: float, info: dict) -> None:
        position = self._position
        if position is None:
            entries = {
                Action.ENTER_LONG_1: (PositionSide.LONG, 1),
                Action.ENTER_SHORT_1: (PositionSide.SHORT, 1),
            }
            if action in entries:
                side, size = entries[action]
                initial_risk_points = None
                protective_stop = None
                if self.spec.per_trade_risk_dollars is not None:
                    fee = self.round_trip_fees[self._ticker] * size
                    initial_risk_points = (
                        self.spec.per_trade_risk_dollars - fee
                    ) / (self.tick_values[self._ticker] * size)
                    protective_stop = (
                        fill - initial_risk_points
                        if side == PositionSide.LONG
                        else fill + initial_risk_points
                    )
                self._position = _Position(
                    side,
                    size,
                    fill,
                    initial_risk_points=initial_risk_points,
                    protective_stop=protective_stop,
                )
                self._primary_side = "long" if side == PositionSide.LONG else "short"
                info.update({"fill_price": fill, "fill_side": self._primary_side, "fill_size": size})
            return
        if action == Action.CLOSE:
            closing_size = position.size
            self._liquidate(fill, info)
            info.update({
                "fill_price": fill,
                "fill_side": "close",
                "fill_size": closing_size,
            })

    def _apply_protective_stop(self, index: int, info: dict) -> None:
        position = self._position
        if position is None or position.protective_stop is None:
            return
        assert self._market is not None
        stop = position.protective_stop
        opening = float(self._market.open[index])
        if position.side == PositionSide.LONG:
            if opening <= stop:
                fill = opening
            elif float(self._market.low[index]) <= stop:
                fill = stop
            else:
                return
        else:
            if opening >= stop:
                fill = opening
            elif float(self._market.high[index]) >= stop:
                fill = stop
            else:
                return
        info["exit_reason"] = (
            "ratchet_stop" if position.ratchet_active else "initial_stop"
        )
        info["protective_stop_price"] = stop
        self._liquidate(fill, info)

    def _update_ratchet(self, index: int, info: dict) -> None:
        position = self._position
        if position is None or position.initial_risk_points is None:
            return
        assert self._market is not None
        favorable = (
            float(self._market.high[index]) - position.average_entry
            if position.side == PositionSide.LONG
            else position.average_entry - float(self._market.low[index])
        )
        position.peak_favorable_r = max(
            position.peak_favorable_r,
            favorable / position.initial_risk_points,
        )
        activation = float(self.spec.ratchet_activation_r)
        giveback = float(self.spec.ratchet_giveback_r)
        if position.peak_favorable_r >= activation:
            protected_r = max(
                position.peak_favorable_r - giveback,
                float(self.spec.ratchet_lock_floor_r),
            )
            candidate = (
                position.average_entry + protected_r * position.initial_risk_points
                if position.side == PositionSide.LONG
                else position.average_entry - protected_r * position.initial_risk_points
            )
            if position.side == PositionSide.LONG:
                position.protective_stop = max(position.protective_stop, candidate)
            else:
                position.protective_stop = min(position.protective_stop, candidate)
            position.ratchet_active = True
        info.update({
            "peak_favorable_r": position.peak_favorable_r,
            "ratchet_active": position.ratchet_active,
            "protective_stop_price": position.protective_stop,
        })

    def _close_size(self, price: float, size: int) -> float:
        if self._position is None or self._account is None:
            return 0.0
        size = min(size, self._position.size)
        points = (price - self._position.average_entry) * int(self._position.side)
        gross = points * self.tick_values[self._ticker] * size
        fee = self.round_trip_fees[self._ticker] * size
        net_pnl = gross - fee
        closes_trade = size == self._position.size
        self._account.realize(net_pnl)
        self._position.size -= size
        if self._position.size == 0:
            self._position = None
        if closes_trade:
            self._closed_trade_pnls.append(float(net_pnl))
        return fee

    def _trade_statistics(self) -> dict[str, float | int]:
        trades = len(self._closed_trade_pnls)
        winners = [value for value in self._closed_trade_pnls if value > 0.0]
        risk_denominator = self.spec.per_trade_risk_dollars or self.spec.max_loss
        return {
            "trade_count": trades,
            "win_count": len(winners),
            "win_rate": len(winners) / trades if trades else 0.0,
            # Ratchet recipes use true per-trade R; legacy recipes retain the
            # challenge-level denominator used by their historical receipts.
            "avg_win_r": (
                float(np.mean(winners))
                / risk_denominator
                if winners else 0.0
            ),
            "winning_r_sum": float(np.sum(winners)) / risk_denominator,
        }

    def _liquidate(self, price: float, info: dict) -> None:
        if self._position is not None:
            info["fees_paid"] += self._close_size(price, self._position.size)

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
        realized = 0.0 if self._account is None else self._account.realized_pnl
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
        realized = 0.0 if self._account is None else self._account.realized_pnl
        equity = self._equity(float(self._market.close[self._index]))
        side = PositionSide.FLAT if self._position is None else self._position.side
        size = 0 if self._position is None else self._position.size
        elapsed = self._index - self._start
        days_left = max(
            0, self.spec.episode_days - self._trading_days_elapsed + 1
        )
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
            challenge_remaining=days_left / max(1, self.spec.episode_days),
            point_value=self.tick_values[self._ticker],
            round_trip_fee=self.round_trip_fees.get(self._ticker, 0.0),
            mll_headroom=(
                self.spec.max_loss + equity
                if self._account is None
                else self._account.mll_headroom(equity)
            ),
        )

    def _observation(self) -> np.ndarray:
        assert self._market is not None
        return self._assembler.assemble(
            self._market.embeddings[self._index], self._account_state()
        )
