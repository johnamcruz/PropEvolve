"""Teacher-free economic gate for the one-time sealed 2026 confirmation."""

from __future__ import annotations

from datetime import date
import math
from typing import Mapping, Sequence


_OUTCOMES = {"pass", "blow", "timeout"}


def _finite_number(row: Mapping, name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sealed episode {name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"sealed episode {name} must be finite")
    return value


def _summary(rows: Sequence[Mapping]) -> dict[str, float]:
    episodes = len(rows)
    trade_count = sum(int(row["trade_count"]) for row in rows)
    win_count = sum(int(row["win_count"]) for row in rows)
    winning_r_sum = sum(float(row["winning_r_sum"]) for row in rows)
    trade_r_sum = sum(float(row["trade_r_sum"]) for row in rows)
    long_count = sum(int(row["long_trade_count"]) for row in rows)
    short_count = sum(int(row["short_trade_count"]) for row in rows)
    two_r_count = sum(int(row["two_r_eligible_count"]) for row in rows)
    two_r_capture = sum(float(row["two_r_capture_sum"]) for row in rows)
    directional_count = long_count + short_count
    return {
        "episodes": float(episodes),
        "pass_rate": sum(row["outcome"] == "pass" for row in rows) / episodes,
        "blow_rate": sum(row["outcome"] == "blow" for row in rows) / episodes,
        "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / episodes,
        "mean_terminal_pnl": sum(float(row["terminal_pnl"]) for row in rows)
        / episodes,
        "trade_win_rate": win_count / trade_count if trade_count else 0.0,
        "average_win_r": winning_r_sum / win_count if win_count else 0.0,
        "expectancy_r": trade_r_sum / trade_count if trade_count else 0.0,
        "long_trade_fraction": (
            long_count / directional_count if directional_count else 0.0
        ),
        "two_r_mfe_capture_ratio": (
            two_r_capture / two_r_count if two_r_count else 0.0
        ),
        "near_blow_rate": sum(bool(row["near_blow"]) for row in rows) / episodes,
    }


def evaluate_sealed_confirmation(
    contract: Mapping,
    episodes: Sequence[Mapping],
) -> dict:
    """Evaluate immutable, non-overlapping 30-session episode evidence.

    This function is deliberately not part of training or campaign reasoning.
    Once these rows are inspected, the named period is no longer sealed.
    """

    tickers = tuple(str(value) for value in contract["tickers"])
    if (
        contract.get("teacher_free") is not True
        or contract.get("window_mode") != "non_overlapping"
        or not tickers
        or len(set(tickers)) != len(tickers)
    ):
        raise ValueError("sealed confirmation contract must be teacher-free")
    start_boundary = date.fromisoformat(str(contract["start"]))
    end_boundary = date.fromisoformat(str(contract["end"]))
    episode_sessions = int(contract["episode_sessions"])
    minimum_per_ticker = int(contract["minimum_episodes_per_ticker"])
    if start_boundary >= end_boundary or min(episode_sessions, minimum_per_ticker) < 1:
        raise ValueError("sealed confirmation temporal contract is invalid")

    normalized: list[dict] = []
    for raw in episodes:
        ticker = str(raw.get("ticker", ""))
        if ticker not in tickers:
            raise ValueError(f"sealed episode has undeclared ticker: {ticker}")
        if raw.get("teacher_accessed") is not False:
            raise ValueError("sealed confirmation must remain teacher-free")
        outcome = str(raw.get("outcome", ""))
        if outcome not in _OUTCOMES:
            raise ValueError("sealed episode outcome is invalid")
        start = date.fromisoformat(str(raw.get("start_session")))
        end = date.fromisoformat(str(raw.get("end_session")))
        if not start_boundary <= start <= end < end_boundary:
            raise ValueError("sealed episode lies outside the declared period")
        if int(raw.get("session_count", -1)) != episode_sessions:
            raise ValueError("sealed episode does not contain exactly 30 sessions")
        row = {
            **dict(raw),
            "ticker": ticker,
            "outcome": outcome,
            "start": start,
            "end": end,
        }
        for name in (
            "terminal_pnl",
            "winning_r_sum",
            "trade_r_sum",
            "two_r_capture_sum",
        ):
            row[name] = _finite_number(raw, name)
        for name in (
            "trade_count",
            "win_count",
            "long_trade_count",
            "short_trade_count",
            "two_r_eligible_count",
        ):
            value = raw.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"sealed episode {name} must be nonnegative")
            row[name] = value
        if (
            row["win_count"] > row["trade_count"]
            or row["long_trade_count"] + row["short_trade_count"]
            != row["trade_count"]
        ):
            raise ValueError("sealed episode trade diagnostics are inconsistent")
        normalized.append(row)

    per_market_rows: dict[str, list[dict]] = {ticker: [] for ticker in tickers}
    for row in normalized:
        per_market_rows[row["ticker"]].append(row)
    for ticker, rows in per_market_rows.items():
        ordered = sorted(rows, key=lambda row: (row["start"], row["end"]))
        if len(ordered) < minimum_per_ticker:
            raise ValueError(
                f"sealed confirmation has too few episodes for {ticker}"
            )
        if any(
            current["start"] <= previous["end"]
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise ValueError(f"sealed confirmation windows overlap for {ticker}")

    metrics = _summary(normalized)
    per_market = {
        ticker: _summary(rows) for ticker, rows in per_market_rows.items()
    }
    failures = []

    def require(metric: str, actual: float, operator: str, threshold: float) -> None:
        passed = {
            ">=": actual >= threshold,
            "<=": actual <= threshold,
            ">": actual > threshold,
        }[operator]
        if not passed:
            failures.append({
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "actual": actual,
            })

    minimum_pass = float(contract["minimum_pass_rate"])
    minimum_market_pass = float(contract["minimum_per_market_pass_rate"])
    maximum_blow = float(contract["maximum_blow_rate"])
    minimum_expectancy = float(contract["minimum_expectancy_r"])
    require("pass_rate", metrics["pass_rate"], ">=", minimum_pass)
    require("blow_rate", metrics["blow_rate"], "<=", maximum_blow)
    require("expectancy_r", metrics["expectancy_r"], ">", minimum_expectancy)
    for ticker, values in per_market.items():
        require(
            f"{ticker}.pass_rate",
            values["pass_rate"],
            ">=",
            minimum_market_pass,
        )
        require(
            f"{ticker}.blow_rate",
            values["blow_rate"],
            "<=",
            maximum_blow,
        )
    return {
        "schema": "propevolve_sealed_confirmation_report_v1",
        "status": "PASS" if not failures else "FAIL",
        "teacher_free": True,
        "period": [str(start_boundary), str(end_boundary)],
        "episode_sessions": episode_sessions,
        "window_mode": "non_overlapping",
        "metrics": metrics,
        "per_market": per_market,
        "failures": failures,
    }


__all__ = ["evaluate_sealed_confirmation"]
