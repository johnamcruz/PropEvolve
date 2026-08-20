"""Thirty-day Monte Carlo prop challenge driven by frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from datetime import datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo

import numpy as np

from .decision import Action, ActionMasker, PositionSide
from .episode_coverage import (
    DeterministicEpisodeCoverage,
    FullDataEpisodeCoverageSpec,
)
from .observation import (
    AccountState,
    ObservationAssembler,
    TradeManagementObservationSpec,
)


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
class ChallengeStartState:
    """Validated flat account state for a bounded recovery episode."""

    realized_pnl: float
    equity_pnl: float
    peak_equity_pnl: float
    mll_floor_pnl: float
    passmark_locked: bool
    position_side: PositionSide
    position_size: int
    session_pnl: float
    trading_days_elapsed: int
    recovery_success_pnl: float

    def __post_init__(self) -> None:
        values = (
            self.realized_pnl,
            self.equity_pnl,
            self.peak_equity_pnl,
            self.mll_floor_pnl,
            self.session_pnl,
            self.recovery_success_pnl,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("challenge start account values must be finite")
        if PositionSide(self.position_side) != PositionSide.FLAT or self.position_size != 0:
            raise ValueError("recovery challenge must start flat")
        if self.equity_pnl != self.realized_pnl:
            raise ValueError("flat recovery equity must equal realized PnL")
        if self.peak_equity_pnl < self.equity_pnl:
            raise ValueError("challenge peak equity cannot trail current equity")
        if self.mll_floor_pnl >= self.equity_pnl:
            raise ValueError("recovery challenge must start above the MLL floor")
        if self.passmark_locked:
            raise ValueError("recovery challenge cannot start after passmark lock")
        if (
            isinstance(self.trading_days_elapsed, bool)
            or not isinstance(self.trading_days_elapsed, int)
            or self.trading_days_elapsed < 1
        ):
            raise ValueError("trading_days_elapsed must be a positive integer")


@dataclass(frozen=True)
class MarketSeries:
    ticker: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    embeddings: np.ndarray
    embeddings_authenticated: bool = False

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
        if (
            not self.embeddings_authenticated
            and not np.isfinite(self.embeddings).all()
        ):
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
    source_decision_index: int = 0
    entry_index: int = 0
    entry_realized_pnl: float = 0.0
    entry_mll_floor_pnl: float = 0.0
    entry_mll_headroom: float = 0.0


@dataclass(frozen=True)
class _ClosedTrade:
    pnl: float
    realized_r: float
    peak_favorable_r: float
    peak_adverse_r: float
    side: str
    source_decision_index: int
    entry_index: int
    exit_index: int
    entry_timestamp: str
    exit_timestamp: str
    ratchet_activated: bool
    exit_reason: str
    hold_bars: int
    entry_realized_pnl: float
    entry_mll_floor_pnl: float
    entry_mll_headroom: float
    shadow_outcomes: tuple["_ShadowOutcome", ...]


@dataclass(frozen=True)
class _ShadowOutcome:
    horizon: int
    complete: bool
    mfe_r: float
    mae_r: float
    hit_2r_before_1r: bool
    hit_3r_before_1r: bool


class HistoricalChallengeEnv:
    """Sample challenge windows and execute decisions at the next bar open."""

    def __init__(
        self,
        markets: dict[str, MarketSeries],
        *,
        tick_values: dict[str, float],
        spec: ChallengeSpec,
        round_trip_fees: dict[str, float],
        observation_spec: TradeManagementObservationSpec | None = None,
        seed: int,
        episode_coverage: FullDataEpisodeCoverageSpec | None = None,
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
        self._episode_coverage = (
            DeterministicEpisodeCoverage(
                {
                    ticker: market.timestamps
                    for ticker, market in self.markets.items()
                },
                self._session_keys,
                episode_days=self.spec.episode_days,
                spec=episode_coverage,
            )
            if episode_coverage is not None
            else None
        )
        self._rng = np.random.default_rng(seed)
        self._assembler = ObservationAssembler(
            next(iter(embedding_dims)),
            max_loss=self.spec.max_loss,
            profit_target=self.spec.profit_target,
            trade_management=observation_spec,
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
        self._recovery_entry_open = False
        self._recovery_success_pnl: float | None = None
        self._recovery_status: str | None = None
        self._recovery_wait_decisions = 0

    @property
    def observation_dim(self) -> int:
        return self._assembler.output_dim

    def rng_state(self) -> dict:
        """Return the episode-sampling RNG state for exact boundary recovery."""
        rng = copy.deepcopy(self._rng.bit_generator.state)
        if self._episode_coverage is None:
            return rng
        return {
            "schema": "propevolve_coverage_environment_state_v1",
            "episode_sampling_rng": rng,
            "episode_coverage": self._episode_coverage.state_dict(),
        }

    def restore_rng_state(self, state: dict) -> None:
        """Restore episode sampling only while no episode is active."""
        if not self._terminated:
            raise RuntimeError("environment RNG can only be restored between episodes")
        if self._episode_coverage is None:
            if state.get("schema") == "propevolve_coverage_environment_state_v1":
                raise ValueError("episode coverage recovery mode drifted")
            self._rng.bit_generator.state = copy.deepcopy(state)
            return
        if state.get("schema") != "propevolve_coverage_environment_state_v1":
            raise ValueError("episode coverage recovery state is missing")
        self._episode_coverage.load_state_dict(state["episode_coverage"])
        self._rng.bit_generator.state = copy.deepcopy(
            state["episode_sampling_rng"]
        )

    def episode_coverage_receipt(
        self,
        *,
        require_complete: bool = False,
    ) -> dict[str, object]:
        """Return an authenticated receipt for actually visited decision rows."""
        if self._episode_coverage is None:
            raise RuntimeError("deterministic episode coverage is not configured")
        return self._episode_coverage.receipt(require_complete=require_complete)

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
        explicit_start = options.get("start")
        if self._episode_coverage is not None:
            start = self._episode_coverage.begin_episode(
                ticker,
                explicit_start=explicit_start,
            )
        else:
            start = int(
                explicit_start
                if explicit_start is not None
                else self._rng.integers(0, max(1, maximum_start + 1))
            )
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
        self._closed_trades: list[_ClosedTrade] = []
        self._trading_days_elapsed = 1
        self._recovery_entry_open = False
        self._recovery_success_pnl = None
        self._recovery_status = None
        self._recovery_wait_decisions = 0
        start_state = options.get("challenge_start_state")
        if start_state is not None:
            if not isinstance(start_state, ChallengeStartState):
                raise TypeError("challenge_start_state must be a ChallengeStartState")
            self._validate_challenge_start_state(start_state)
            self._account.realized_pnl = start_state.realized_pnl
            self._account.session_pnl = start_state.session_pnl
            self._account.mll_floor_pnl = start_state.mll_floor_pnl
            self._account.passmark_locked = start_state.passmark_locked
            self._peak_equity = start_state.peak_equity_pnl
            self._trading_days_elapsed = start_state.trading_days_elapsed
            self._recovery_success_pnl = start_state.recovery_success_pnl
        equity = self._equity(float(self._market.close[self._index]))
        headroom = self._account.mll_headroom(equity)
        return self._observation(), {
            "ticker": ticker,
            "start": start,
            "end": self._end,
            "realized_pnl": self._account.realized_pnl,
            "equity_pnl": equity,
            "mll_floor_pnl": self._account.mll_floor_pnl,
            "mll_headroom": headroom,
            "mll_headroom_fraction": min(
                1.0, max(0.0, headroom / self.spec.max_loss)
            ),
            "passmark_locked": self._account.passmark_locked,
            "valid_actions": self.valid_actions(),
            "recovery_wait_decisions": self._recovery_wait_decisions,
        }

    def valid_actions(self) -> tuple[Action, ...]:
        return self._masker.valid_actions(
            self._account_state(),
            recovery_active=self._recovery_success_pnl is not None,
        )

    def _validate_challenge_start_state(self, state: ChallengeStartState) -> None:
        if not math.isclose(state.mll_floor_pnl, -self.spec.max_loss):
            raise ValueError("recovery start MLL floor must equal negative max_loss")
        if not math.isclose(state.peak_equity_pnl, 0.0):
            raise ValueError("recovery start peak equity must be zero")
        if self.spec.per_trade_risk_dollars is None:
            raise ValueError("recovery start requires declared per-trade risk")
        if not math.isclose(state.recovery_success_pnl, 0.0):
            raise ValueError("recovery success PnL must be breakeven")
        if state.trading_days_elapsed > self.spec.episode_days:
            raise ValueError("recovery start exceeds the challenge duration")

    def step(self, action: Action | int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._terminated or self._market is None or self._account is None:
            raise RuntimeError("reset the challenge before stepping")
        action = Action(int(action))
        if action not in self.valid_actions():
            raise ValueError(f"action {action.name} is invalid in the current state")
        if self._episode_coverage is not None:
            self._episode_coverage.record_decision(self._ticker, self._index)
        previous_equity = self._equity(self._market.close[self._index])
        closed_trade_count = len(self._closed_trade_pnls)
        next_index = self._index + 1
        fill = float(self._market.open[next_index])
        info: dict = {
            "decision_index": self._index,
            "fill_index": next_index,
            "fees_paid": 0.0,
            "recovery_entry_used": False,
            "recovery_success": False,
            "recovery_trade_closed": False,
        }
        if (
            action == Action.WAIT
            and self._recovery_success_pnl is not None
        ):
            self._recovery_wait_decisions += 1
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
            info.setdefault("exit_reason", "challenge_blow")
            self._liquidate(worst_price, info)
            outcome = "blow"
        else:
            current_equity = self._equity(float(self._market.close[self._index]))
            if self._account.outcome(current_equity) == "pass":
                info.setdefault("exit_reason", "challenge_pass")
                self._liquidate(float(self._market.close[self._index]), info)
                if self._recovery_success_pnl is not None:
                    self._recovery_status = "recovered"
                    self._recovery_success_pnl = None
                outcome = "pass"
            elif (
                self._recovery_success_pnl is not None
                and current_equity >= self._recovery_success_pnl
            ):
                info.setdefault("exit_reason", "recovery_breakeven")
                self._liquidate(float(self._market.close[self._index]), info)
                current_equity = self._equity(float(self._market.close[self._index]))
                if self._account.realized_pnl >= self._recovery_success_pnl:
                    self._recovery_status = "recovered"
                    self._recovery_success_pnl = None
            if outcome is None and self._index >= self._end:
                info.setdefault("exit_reason", "episode_timeout")
                self._liquidate(float(self._market.close[self._index]), info)
                outcome = "timeout"
        if (
            outcome is not None
            and self._recovery_status is None
            and self._recovery_success_pnl is not None
        ):
            self._recovery_status = "not_recovered"
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
        mll_headroom = self._account.mll_headroom(equity)
        info.update({
            "outcome": outcome,
            "equity_pnl": equity,
            "ticker": self._ticker,
            "primary_side": self._primary_side,
            "realized_pnl": self._account.realized_pnl,
            "mll_floor_pnl": self._account.mll_floor_pnl,
            "mll_headroom": mll_headroom,
            "mll_headroom_fraction": min(
                1.0, max(0.0, mll_headroom / self.spec.max_loss)
            ),
            "ordinary_entry_eligible": (
                self._position is None
                and mll_headroom >= self.spec.minimum_mll_headroom
            ),
            "passmark_locked": self._account.passmark_locked,
            "timestamp": str(self._market.timestamps[self._index]),
            "trading_days_elapsed": self._trading_days_elapsed,
            "valid_actions": () if self._terminated else self.valid_actions(),
            "recovery_success": self._recovery_status == "recovered",
            "recovery_wait_decisions": self._recovery_wait_decisions,
            **shaping_info,
            **self._trade_statistics(),
        })
        if self._recovery_status in {"recovered", "not_recovered"}:
            info["recovery_status"] = self._recovery_status
        return self._observation(), float(reward), self._terminated, False, info

    def _reward_shaping(
        self,
        equity: float,
        *,
        closed_trade_count: int,
    ) -> tuple[float, dict[str, float]]:
        proximity_penalty = 0.0
        lead_giveback_penalty = 0.0
        if self._account is not None:
            cushion_fraction = min(
                1.0,
                max(
                    0.0,
                    self._account.mll_headroom(equity) / self.spec.max_loss,
                ),
            )
            proximity_penalty = (
                self.spec.mll_proximity_penalty_coefficient
                * (1.0 - cushion_fraction) ** 2
            )

        if self._position is not None and self._account is not None:
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
                assert self._account is not None
                if self._recovery_success_pnl is not None:
                    self._recovery_entry_open = True
                    info["recovery_entry_used"] = True
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
                    source_decision_index=int(info["decision_index"]),
                    entry_index=self._index + 1,
                    entry_realized_pnl=float(self._account.realized_pnl),
                    entry_mll_floor_pnl=float(self._account.mll_floor_pnl),
                    entry_mll_headroom=float(self._account.mll_headroom()),
                )
                self._primary_side = "long" if side == PositionSide.LONG else "short"
                info.update({"fill_price": fill, "fill_side": self._primary_side, "fill_size": size})
            return
        if action == Action.CLOSE:
            closing_size = position.size
            info["exit_reason"] = "voluntary_close"
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

    @staticmethod
    def _trade_context(trade: _ClosedTrade) -> dict[str, object]:
        return {
            "side": trade.side,
            "entry_index": trade.entry_index,
            "exit_index": trade.exit_index,
            "entry_timestamp": trade.entry_timestamp,
            "exit_timestamp": trade.exit_timestamp,
            "hold_bars": trade.hold_bars,
            "realized_r": trade.realized_r,
            "mfe_r": trade.peak_favorable_r,
            "mae_r": trade.peak_adverse_r,
            "ratchet_activated": trade.ratchet_activated,
            "exit_reason": trade.exit_reason,
        }

    def closed_trade_receipts(self) -> tuple[dict[str, object], ...]:
        """Return detached episode-local trade economics for diagnostics.

        The receipt contains only causal entry state plus realized execution
        outcomes. Training and validation callers may join the source decision
        index to separately authenticated Regime evidence without exposing
        future outcomes to the policy.
        """
        ticker = "" if self._ticker is None else self._ticker
        return tuple({
            "trade_index": index,
            "ticker": ticker,
            **self._trade_context(trade),
            "source_decision_index": trade.source_decision_index,
            "entry_realized_pnl": trade.entry_realized_pnl,
            "entry_mll_floor_pnl": trade.entry_mll_floor_pnl,
            "entry_mll_headroom": trade.entry_mll_headroom,
            "pnl": trade.pnl,
        } for index, trade in enumerate(self._closed_trades))

    def _trade_statistics(self) -> dict[str, object]:
        trades = len(self._closed_trade_pnls)
        winners = [value for value in self._closed_trade_pnls if value > 0.0]
        risk_denominator = self.spec.per_trade_risk_dollars or self.spec.max_loss
        realized_rs = [trade.realized_r for trade in self._closed_trades]
        losing_rs = [value for value in realized_rs if value < 0.0]
        retention_trades = [
            trade for trade in self._closed_trades if trade.peak_favorable_r >= 1.0
        ]
        two_r_trades = [
            trade for trade in self._closed_trades if trade.peak_favorable_r >= 2.0
        ]
        activated = [trade for trade in self._closed_trades if trade.ratchet_activated]
        exit_counts = {
            reason: sum(trade.exit_reason == reason for trade in self._closed_trades)
            for reason in ("voluntary_close", "initial_stop", "ratchet_stop")
        }
        terminal_liquidations = sum(
            trade.exit_reason in {"challenge_pass", "challenge_blow", "episode_timeout"}
            for trade in self._closed_trades
        )
        largest_realized = (
            max(self._closed_trades, key=lambda trade: trade.realized_r)
            if self._closed_trades else None
        )
        largest_mfe = (
            max(self._closed_trades, key=lambda trade: trade.peak_favorable_r)
            if self._closed_trades else None
        )
        statistics: dict[str, object] = {
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
            "avg_loss_r": float(np.mean(losing_rs)) if losing_rs else 0.0,
            "expectancy_r": float(np.mean(realized_rs)) if realized_rs else 0.0,
            "avg_mfe_r": (
                float(np.mean([trade.peak_favorable_r for trade in self._closed_trades]))
                if self._closed_trades else 0.0
            ),
            "avg_mae_r": (
                float(np.mean([trade.peak_adverse_r for trade in self._closed_trades]))
                if self._closed_trades else 0.0
            ),
            "retention_eligible_count": len(retention_trades),
            "mfe_capture_ratio": (
                float(np.mean([
                    trade.realized_r / trade.peak_favorable_r
                    for trade in retention_trades
                ]))
                if retention_trades else 0.0
            ),
            "mfe_realized_gap_r": (
                float(np.mean([
                    trade.peak_favorable_r - trade.realized_r
                    for trade in retention_trades
                ]))
                if retention_trades else 0.0
            ),
            "gave_it_all_back_rate": (
                sum(trade.realized_r <= 0.0 for trade in retention_trades)
                / len(retention_trades)
                if retention_trades else 0.0
            ),
            "two_r_eligible_count": len(two_r_trades),
            "two_r_mfe_capture_ratio": (
                float(np.mean([
                    trade.realized_r / trade.peak_favorable_r
                    for trade in two_r_trades
                ]))
                if two_r_trades else 0.0
            ),
            "two_r_gave_it_all_back_rate": (
                sum(trade.realized_r <= 0.0 for trade in two_r_trades)
                / len(two_r_trades)
                if two_r_trades else 0.0
            ),
            "ratchet_activation_rate": len(activated) / trades if trades else 0.0,
            "activated_avg_realized_r": (
                float(np.mean([trade.realized_r for trade in activated]))
                if activated else 0.0
            ),
            "avg_hold_bars": (
                float(np.mean([trade.hold_bars for trade in self._closed_trades]))
                if self._closed_trades else 0.0
            ),
            "voluntary_close_count": exit_counts["voluntary_close"],
            "initial_stop_count": exit_counts["initial_stop"],
            "ratchet_stop_count": exit_counts["ratchet_stop"],
            "terminal_liquidation_count": terminal_liquidations,
            "largest_realized_trade": (
                self._trade_context(largest_realized)
                if largest_realized is not None else None
            ),
            "largest_mfe_trade": (
                self._trade_context(largest_mfe)
                if largest_mfe is not None else None
            ),
        }
        for horizon in (5, 10, 20, 50):
            outcomes = [
                outcome
                for trade in self._closed_trades
                for outcome in trade.shadow_outcomes
                if outcome.horizon == horizon and outcome.complete
            ]
            prefix = f"shadow_h{horizon}"
            statistics[f"{prefix}_complete_trades"] = len(outcomes)
            statistics[f"{prefix}_avg_mfe_r"] = (
                float(np.mean([outcome.mfe_r for outcome in outcomes]))
                if outcomes else 0.0
            )
            statistics[f"{prefix}_avg_mae_r"] = (
                float(np.mean([outcome.mae_r for outcome in outcomes]))
                if outcomes else 0.0
            )
            statistics[f"{prefix}_2r_before_1r_rate"] = (
                sum(outcome.hit_2r_before_1r for outcome in outcomes) / len(outcomes)
                if outcomes else 0.0
            )
            statistics[f"{prefix}_3r_before_1r_rate"] = (
                sum(outcome.hit_3r_before_1r for outcome in outcomes) / len(outcomes)
                if outcomes else 0.0
            )
        return statistics

    def _liquidate(self, price: float, info: dict) -> None:
        if self._position is not None:
            position = self._position
            exit_index = int(info.get("fill_index", self._index))
            exit_reason = str(info.get("exit_reason", "unknown"))
            peak_favorable_r, peak_adverse_r = self._closed_trade_excursions(
                position,
                exit_index=exit_index,
                exit_price=price,
                exit_reason=exit_reason,
            )
            info["fees_paid"] += self._close_size(price, self._position.size)
            if self._recovery_entry_open:
                info["recovery_trade_closed"] = True
                self._recovery_entry_open = False
            pnl = self._closed_trade_pnls[-1]
            risk = self.spec.per_trade_risk_dollars or self.spec.max_loss
            self._closed_trades.append(_ClosedTrade(
                pnl=pnl,
                realized_r=pnl / risk,
                peak_favorable_r=peak_favorable_r,
                peak_adverse_r=peak_adverse_r,
                side=position.side.name.removeprefix("ENTER_").lower(),
                source_decision_index=position.source_decision_index,
                entry_index=position.entry_index,
                exit_index=exit_index,
                entry_timestamp=np.datetime_as_string(
                    self._market.timestamps[position.entry_index], unit="m"
                ),
                exit_timestamp=np.datetime_as_string(
                    self._market.timestamps[exit_index], unit="m"
                ),
                ratchet_activated=position.ratchet_active,
                exit_reason=exit_reason,
                hold_bars=max(0, exit_index - position.entry_index),
                entry_realized_pnl=position.entry_realized_pnl,
                entry_mll_floor_pnl=position.entry_mll_floor_pnl,
                entry_mll_headroom=position.entry_mll_headroom,
                shadow_outcomes=self._shadow_entry_outcomes(position),
            ))

    def _closed_trade_excursions(
        self,
        position: _Position,
        *,
        exit_index: int,
        exit_price: float,
        exit_reason: str,
    ) -> tuple[float, float]:
        """Return causal MFE/MAE actually available before the recorded exit.

        A close/pass/timeout at bar close observes that complete bar. Stops,
        blows, and next-open voluntary closes conservatively observe only prior
        completed bars plus their executable exit price.
        """
        if position.initial_risk_points is None:
            return 0.0, 0.0
        assert self._market is not None
        include_exit_bar = exit_reason in {"challenge_pass", "episode_timeout"}
        stop = min(
            len(self._market.close),
            exit_index + int(include_exit_bar),
        )
        highs = self._market.high[position.entry_index:stop]
        lows = self._market.low[position.entry_index:stop]
        entry = position.average_entry
        favorable_prices = [float(exit_price)]
        adverse_prices = [float(exit_price)]
        if len(highs):
            favorable_prices.append(
                float(np.max(highs)) if position.side == PositionSide.LONG
                else float(np.min(lows))
            )
            adverse_prices.append(
                float(np.min(lows)) if position.side == PositionSide.LONG
                else float(np.max(highs))
            )
        if position.side == PositionSide.LONG:
            favorable = max(favorable_prices) - entry
            adverse = entry - min(adverse_prices)
        else:
            favorable = entry - min(favorable_prices)
            adverse = max(adverse_prices) - entry
        risk = position.initial_risk_points
        return max(0.0, favorable / risk), max(0.0, adverse / risk)

    def _shadow_entry_outcomes(
        self, position: _Position
    ) -> tuple[_ShadowOutcome, ...]:
        if position.initial_risk_points is None:
            return ()
        assert self._market is not None
        risk_points = position.initial_risk_points
        outcomes = []
        for horizon in (5, 10, 20, 50):
            stop = position.entry_index + horizon
            complete = stop <= self._end + 1
            bounded_stop = min(stop, self._end + 1)
            if bounded_stop <= position.entry_index:
                continue
            highs = self._market.high[position.entry_index:bounded_stop]
            lows = self._market.low[position.entry_index:bounded_stop]
            if position.side == PositionSide.LONG:
                favorable = (highs - position.average_entry) / risk_points
                adverse = (position.average_entry - lows) / risk_points
            else:
                favorable = (position.average_entry - lows) / risk_points
                adverse = (highs - position.average_entry) / risk_points

            def target_before_stop(target_r: float) -> bool:
                for favorable_r, adverse_r in zip(favorable, adverse):
                    # Conservative OHLC ambiguity: an adverse and favorable
                    # touch on the same bar is counted as stop-first.
                    if adverse_r >= 1.0:
                        return False
                    if favorable_r >= target_r:
                        return True
                return False

            outcomes.append(_ShadowOutcome(
                horizon=horizon,
                complete=complete,
                mfe_r=max(0.0, float(np.max(favorable))),
                mae_r=max(0.0, float(np.max(adverse))),
                hit_2r_before_1r=target_before_stop(2.0),
                hit_3r_before_1r=target_before_stop(3.0),
            ))
        return tuple(outcomes)

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
        current_r = 0.0
        peak_favorable_r = 0.0
        giveback_r = 0.0
        hold_bars = 0
        ratchet_active = False
        protected_r = 0.0
        if self._position is not None:
            position = self._position
            if position.initial_risk_points is not None:
                current_points = (
                    float(self._market.close[self._index])
                    - position.average_entry
                ) * int(position.side)
                current_r = current_points / position.initial_risk_points
                peak_favorable_r = position.peak_favorable_r
                giveback_r = max(0.0, peak_favorable_r - current_r)
                assert position.protective_stop is not None
                protected_r = (
                    (position.protective_stop - position.average_entry)
                    * int(position.side)
                    / position.initial_risk_points
                )
            hold_bars = max(0, self._index - position.entry_index)
            ratchet_active = position.ratchet_active
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
            current_r=current_r,
            peak_favorable_r=peak_favorable_r,
            giveback_r=giveback_r,
            hold_bars=hold_bars,
            ratchet_active=ratchet_active,
            protected_r=protected_r,
        )

    def _observation(self) -> np.ndarray:
        assert self._market is not None
        return self._assembler.assemble(
            self._market.embeddings[self._index], self._account_state()
        )
