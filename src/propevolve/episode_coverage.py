"""Deterministic, additive coverage of historical challenge decision rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import numpy as np


_RECEIPT_SCHEMA = "propevolve_episode_coverage_receipt_v1"
_STATE_SCHEMA = "propevolve_episode_coverage_state_v1"
_PLAN_SCHEMA = "propevolve_episode_coverage_plan_v1"
FULL_DATA_COVERAGE_SCHEMA = "full_data_episode_coverage_v1"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _timestamp(value: np.datetime64) -> str:
    return np.datetime_as_string(
        np.datetime64(value, "ns"),
        unit="ns",
    )


def _row_map_identity(ticker: str, timestamps: np.ndarray) -> str:
    values = (
        np.asarray(timestamps)
        .astype("datetime64[ns]")
        .astype("<i8", copy=False)
    )
    digest = hashlib.sha256()
    digest.update(ticker.encode())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FullDataEpisodeCoverageSpec:
    """Declare one balanced, deterministic full-row training budget."""

    episode_budget: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_budget, bool)
            or not isinstance(self.episode_budget, int)
            or self.episode_budget < 1
        ):
            raise ValueError("episode coverage budget must be a positive integer")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> "FullDataEpisodeCoverageSpec":
        if (
            not isinstance(config, Mapping)
            or set(config) != {"schema", "episode_budget"}
            or config.get("schema") != FULL_DATA_COVERAGE_SCHEMA
        ):
            raise ValueError("episode coverage mode contract is invalid")
        return cls(episode_budget=config["episode_budget"])


@dataclass(frozen=True, slots=True)
class _MarketLayout:
    ticker: str
    timestamps: np.ndarray
    session_keys: np.ndarray
    unique_sessions: np.ndarray
    episode_days: int
    maximum_start: int
    eligible_rows: int
    row_map_identity_sha256: str
    full_window_starts: tuple[int, ...]

    def end_for_start(self, start: int) -> int:
        start_session = int(
            np.searchsorted(self.unique_sessions, self.session_keys[start])
        )
        final_session = self.unique_sessions[
            start_session + self.episode_days - 1
        ]
        return int(
            np.searchsorted(self.session_keys, final_session, side="right") - 1
        )


def _layout(
    ticker: str,
    timestamps: np.ndarray,
    session_keys: np.ndarray,
    *,
    episode_days: int,
) -> _MarketLayout:
    timestamps = np.asarray(timestamps)
    session_keys = np.asarray(session_keys)
    if timestamps.ndim != 1 or session_keys.shape != timestamps.shape:
        raise ValueError(f"market {ticker} coverage row map is malformed")
    unique_sessions = np.unique(session_keys)
    if len(unique_sessions) < episode_days:
        raise ValueError(
            f"market {ticker} cannot fit {episode_days} coverage days"
        )
    last_start_session = unique_sessions[-episode_days]
    maximum_start = min(
        len(timestamps) - 2,
        int(
            np.searchsorted(
                session_keys,
                last_start_session,
                side="right",
            ) - 1
        ),
    )
    if maximum_start < 0:
        raise ValueError(f"market {ticker} has no eligible challenge start")
    partial = _MarketLayout(
        ticker=ticker,
        timestamps=timestamps,
        session_keys=session_keys,
        unique_sessions=unique_sessions,
        episode_days=episode_days,
        maximum_start=maximum_start,
        eligible_rows=len(timestamps) - 1,
        row_map_identity_sha256=_row_map_identity(ticker, timestamps),
        full_window_starts=(),
    )
    covered = np.zeros(partial.eligible_rows, dtype=np.bool_)
    starts: list[int] = []
    while not bool(covered.all()):
        first_uncovered = int(np.flatnonzero(~covered)[0])
        start = min(first_uncovered, partial.maximum_start)
        if starts and start <= starts[-1]:
            raise ValueError(
                f"market {ticker} challenge windows cannot cover every decision row"
            )
        end = partial.end_for_start(start)
        if end <= start:
            raise ValueError(
                f"market {ticker} coverage window has no causal decision step"
            )
        covered[start:min(end, partial.eligible_rows)] = True
        starts.append(start)
    return _MarketLayout(
        **{
            field: getattr(partial, field)
            for field in (
                "ticker",
                "timestamps",
                "session_keys",
                "unique_sessions",
                "episode_days",
                "maximum_start",
                "eligible_rows",
                "row_map_identity_sha256",
            )
        },
        full_window_starts=tuple(starts),
    )


class DeterministicEpisodeCoverage:
    """Choose starts from the first uncovered row and attest visited rows."""

    def __init__(
        self,
        timestamps: Mapping[str, np.ndarray],
        session_keys: Mapping[str, np.ndarray],
        *,
        episode_days: int,
        spec: FullDataEpisodeCoverageSpec,
    ) -> None:
        if set(timestamps) != set(session_keys) or not timestamps:
            raise ValueError("coverage markets and session row maps must match")
        self.spec = spec
        self._tickers = tuple(sorted(timestamps))
        if spec.episode_budget < len(self._tickers):
            raise ValueError(
                "episode coverage budget cannot allocate every market"
            )
        self._minimum_market_budget = spec.episode_budget // len(self._tickers)
        self._maximum_market_budget = (
            spec.episode_budget + len(self._tickers) - 1
        ) // len(self._tickers)
        self._layouts = {
            ticker: _layout(
                ticker,
                timestamps[ticker],
                session_keys[ticker],
                episode_days=episode_days,
            )
            for ticker in self._tickers
        }
        insufficient = {
            ticker: len(layout.full_window_starts)
            for ticker, layout in self._layouts.items()
            if len(layout.full_window_starts) > self._minimum_market_budget
        }
        if insufficient:
            raise ValueError(
                "episode coverage budget cannot cover every eligible row: "
                + ", ".join(
                    f"{ticker} requires {required}, has "
                    f"{self._minimum_market_budget}"
                    for ticker, required in insufficient.items()
                )
            )
        self._covered = {
            ticker: np.zeros(layout.eligible_rows, dtype=np.bool_)
            for ticker, layout in self._layouts.items()
        }
        self._starts: dict[str, list[int]] = {
            ticker: [] for ticker in self._tickers
        }
        self._episodes_consumed = 0
        plan = {
            "schema": _PLAN_SCHEMA,
            "episode_budget": spec.episode_budget,
            "episode_days": episode_days,
            "minimum_market_episode_budget": self._minimum_market_budget,
            "maximum_market_episode_budget": self._maximum_market_budget,
            "markets": {
                ticker: {
                    "eligible_decision_rows": layout.eligible_rows,
                    "maximum_start": layout.maximum_start,
                    "minimum_full_window_episodes": len(
                        layout.full_window_starts
                    ),
                    "full_window_starts": list(layout.full_window_starts),
                    "row_map_identity_sha256": (
                        layout.row_map_identity_sha256
                    ),
                }
                for ticker, layout in self._layouts.items()
            },
        }
        self.plan_identity_sha256 = _canonical_sha256(plan)

    def begin_episode(
        self,
        ticker: str,
        *,
        explicit_start: int | None = None,
    ) -> int:
        if ticker not in self._layouts:
            raise KeyError(f"coverage plan does not contain ticker {ticker!r}")
        if self._episodes_consumed >= self.spec.episode_budget:
            raise RuntimeError("episode coverage budget is exhausted")
        starts = self._starts[ticker]
        prospective_counts = [
            len(self._starts[value]) + int(value == ticker)
            for value in self._tickers
        ]
        if max(prospective_counts) - min(prospective_counts) > 1:
            raise RuntimeError("episode coverage ticker schedule is not balanced")
        if len(starts) >= self._maximum_market_budget:
            raise RuntimeError(
                f"episode coverage schedule is not balanced for {ticker}"
            )
        layout = self._layouts[ticker]
        if explicit_start is None:
            uncovered = np.flatnonzero(~self._covered[ticker])
            if len(uncovered):
                start = min(int(uncovered[0]), layout.maximum_start)
            else:
                start = layout.full_window_starts[
                    len(starts) % len(layout.full_window_starts)
                ]
        else:
            if isinstance(explicit_start, bool) or not isinstance(
                explicit_start, int
            ):
                raise TypeError("explicit coverage start must be int")
            start = explicit_start
        if start < 0 or start > layout.maximum_start:
            raise ValueError("episode start cannot fit the coverage window")
        starts.append(start)
        self._episodes_consumed += 1
        return start

    def record_decision(self, ticker: str, index: int) -> None:
        if ticker not in self._layouts:
            raise KeyError(f"coverage plan does not contain ticker {ticker!r}")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("covered decision index must be int")
        layout = self._layouts[ticker]
        if index < 0 or index >= layout.eligible_rows:
            raise IndexError("covered decision index is out of bounds")
        self._covered[ticker][index] = True

    def state_dict(self) -> dict[str, object]:
        receipt = self.receipt()
        return {
            "schema": _STATE_SCHEMA,
            "plan_identity_sha256": self.plan_identity_sha256,
            "episodes_consumed": self._episodes_consumed,
            "markets": {
                ticker: {
                    "starts": tuple(self._starts[ticker]),
                    "covered_packbits": np.packbits(
                        self._covered[ticker],
                        bitorder="little",
                    ).tobytes(),
                }
                for ticker in self._tickers
            },
            "identity_sha256": receipt["identity_sha256"],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != _STATE_SCHEMA:
            raise ValueError("unsupported episode coverage recovery schema")
        if state.get("plan_identity_sha256") != self.plan_identity_sha256:
            raise ValueError("episode coverage recovery plan drifted")
        markets = state.get("markets")
        if not isinstance(markets, Mapping) or set(markets) != set(self._tickers):
            raise ValueError("episode coverage recovery markets drifted")
        restored_starts: dict[str, list[int]] = {}
        restored_covered: dict[str, np.ndarray] = {}
        for ticker in self._tickers:
            item = markets[ticker]
            if not isinstance(item, Mapping):
                raise ValueError("episode coverage recovery market is malformed")
            raw_starts = item.get("starts")
            if not isinstance(raw_starts, (tuple, list)) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_starts
            ):
                raise ValueError("episode coverage recovery starts are malformed")
            starts = [int(value) for value in raw_starts]
            layout = self._layouts[ticker]
            if (
                len(starts) > self._maximum_market_budget
                or any(value < 0 or value > layout.maximum_start for value in starts)
            ):
                raise ValueError("episode coverage recovery starts are invalid")
            packed = item.get("covered_packbits")
            if not isinstance(packed, bytes):
                raise ValueError("episode coverage recovery bitmap is malformed")
            covered = np.unpackbits(
                np.frombuffer(packed, dtype=np.uint8),
                count=layout.eligible_rows,
                bitorder="little",
            ).astype(np.bool_)
            restored_starts[ticker] = starts
            restored_covered[ticker] = covered
        episodes_consumed = state.get("episodes_consumed")
        if (
            isinstance(episodes_consumed, bool)
            or not isinstance(episodes_consumed, int)
            or episodes_consumed != sum(map(len, restored_starts.values()))
            or not 0 <= episodes_consumed <= self.spec.episode_budget
        ):
            raise ValueError("episode coverage recovery episode count drifted")
        previous_starts = self._starts
        previous_covered = self._covered
        previous_episodes = self._episodes_consumed
        self._starts = restored_starts
        self._covered = restored_covered
        self._episodes_consumed = episodes_consumed
        if state.get("identity_sha256") != self.receipt()["identity_sha256"]:
            self._starts = previous_starts
            self._covered = previous_covered
            self._episodes_consumed = previous_episodes
            raise ValueError("episode coverage recovery identity drifted")

    def receipt(self, *, require_complete: bool = False) -> dict[str, object]:
        market_receipts: dict[str, dict[str, object]] = {}
        for ticker in self._tickers:
            layout = self._layouts[ticker]
            covered = self._covered[ticker]
            indices = np.flatnonzero(covered)
            covered_rows = int(len(indices))
            bitmap = np.packbits(covered, bitorder="little").tobytes()
            first_covered = int(indices[0]) if covered_rows else None
            last_covered = int(indices[-1]) if covered_rows else None
            payload: dict[str, object] = {
                "eligible_decision_rows": layout.eligible_rows,
                "covered_decision_rows": covered_rows,
                "coverage_fraction": covered_rows / layout.eligible_rows,
                "first_eligible_index": 0,
                "last_eligible_index": layout.eligible_rows - 1,
                "first_eligible_timestamp": _timestamp(layout.timestamps[0]),
                "last_eligible_timestamp": _timestamp(
                    layout.timestamps[layout.eligible_rows - 1]
                ),
                "first_covered_index": first_covered,
                "last_covered_index": last_covered,
                "first_covered_timestamp": (
                    _timestamp(layout.timestamps[first_covered])
                    if first_covered is not None else None
                ),
                "last_covered_timestamp": (
                    _timestamp(layout.timestamps[last_covered])
                    if last_covered is not None else None
                ),
                "episodes_consumed": len(self._starts[ticker]),
                "episode_starts": tuple(self._starts[ticker]),
                "minimum_full_window_episodes": len(
                    layout.full_window_starts
                ),
                "complete": covered_rows == layout.eligible_rows,
                "row_map_identity_sha256": layout.row_map_identity_sha256,
                "covered_row_bitmap_sha256": hashlib.sha256(bitmap).hexdigest(),
            }
            payload["identity_sha256"] = _canonical_sha256(payload)
            market_receipts[ticker] = payload
        counts = [len(self._starts[ticker]) for ticker in self._tickers]
        balanced = max(counts) - min(counts) <= 1
        complete = (
            self._episodes_consumed == self.spec.episode_budget
            and balanced
            and all(item["complete"] for item in market_receipts.values())
        )
        body: dict[str, object] = {
            "schema": _RECEIPT_SCHEMA,
            "mode": FULL_DATA_COVERAGE_SCHEMA,
            "episode_budget": self.spec.episode_budget,
            "episodes_consumed": self._episodes_consumed,
            "market_count": len(self._tickers),
            "balanced": balanced,
            "complete": complete,
            "plan_identity_sha256": self.plan_identity_sha256,
            "markets": market_receipts,
        }
        body["identity_sha256"] = _canonical_sha256(body)
        if require_complete and not complete:
            incomplete = {
                ticker: {
                    "episodes": item["episodes_consumed"],
                    "coverage_fraction": item["coverage_fraction"],
                }
                for ticker, item in market_receipts.items()
                if not item["complete"]
            }
            raise ValueError(
                "episode coverage is incomplete at the declared budget: "
                f"episodes={self._episodes_consumed}/{self.spec.episode_budget}, "
                f"balanced={balanced}, markets={incomplete}"
            )
        return body
