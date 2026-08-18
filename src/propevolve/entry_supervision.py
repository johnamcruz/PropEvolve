"""Deterministic training-only Expansion-entry supervision.

The event anchor is a *pre-launch* completed decision bar whose next open begins
the potential first launch bar.  Launch is confirmed only retrospectively by a
future price path.  Candidate fill offsets ``1 ... 5`` are the first five
execution bars of that retrospectively confirmed launch, not five bars after an
observable confirmation.  Future launch and economic outcomes are training
labels only and must never be model inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

from .decision import Action

if TYPE_CHECKING:
    from .environment import MarketSeries


CANDIDATE_COUNT = 5
ENTRY_ACTION_ORDER = (
    "WAIT",
    "ENTER_LONG_1",
    "ENTER_SHORT_1",
)

EntryAction = Literal["WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"]
EntrySide = Literal["long", "short", "ambiguous"]
EntryStatus = Literal["enter", "abstain", "ambiguous"]
UnavailableReason = Literal[
    "after_entry",
    "ambiguous_side",
    "unresolved_split_end",
]


@dataclass(frozen=True, slots=True)
class CandidateEntryTarget:
    """One decision target and its next-open execution offset."""

    decision_offset: int
    fill_offset: int
    action: EntryAction | None
    available: bool
    censored: bool
    unavailable_reason: UnavailableReason | None


@dataclass(frozen=True, slots=True)
class PostLaunchEntrySupervision:
    """The complete five-state target for one directional launch event."""

    side: EntrySide
    status: EntryStatus
    enter_fill_offset: int | None
    candidates: tuple[CandidateEntryTarget, ...]


@dataclass(frozen=True, slots=True)
class EntryTargetMetadata:
    """Auditable provenance for one post-launch candidate decision."""

    side: EntrySide
    event_anchor_rows: tuple[int, ...]
    candidate_decision_offset: int | None
    fill_offset: int | None
    continuation: bool | None
    economic_win: bool | None
    economic_good: bool | None
    available: bool
    censored: bool
    unavailable_reason: UnavailableReason | None


@dataclass(frozen=True, slots=True)
class EntryActionTargets:
    """In-memory, immutable action targets for authenticated market rows."""

    _targets: Mapping[str, np.ndarray]
    _opportunity_values: Mapping[str, np.ndarray]
    _metadata: Mapping[str, Mapping[int, EntryTargetMetadata]]
    manifest: Mapping[str, object]
    opportunity_value_manifest: Mapping[str, object]

    def target(self, ticker: str, row: int) -> Action | None:
        value = int(self._lookup_target(ticker=ticker, row=row))
        return None if value < 0 else Action(value)

    def metadata(self, ticker: str, row: int) -> EntryTargetMetadata | None:
        self._validate_row(ticker=ticker, row=row)
        return self._metadata[ticker].get(row)

    def opportunity_values(
        self,
        ticker: str,
        row: int,
    ) -> tuple[float, float, float] | None:
        """Return the training-only WAIT/Long/Short opportunity preference."""

        self._validate_row(ticker=ticker, row=row)
        values = self._opportunity_values[ticker][row]
        return (
            None
            if not np.isfinite(values).all()
            else tuple(float(value) for value in values)
        )

    def _validate_row(self, *, ticker: str, row: int) -> None:
        if ticker not in self._targets:
            raise KeyError(f"entry targets do not contain ticker {ticker!r}")
        if type(row) is not int:
            raise TypeError("entry target row must be int")
        if row < 0 or row >= len(self._targets[ticker]):
            raise IndexError("entry target row is out of bounds")

    def _lookup_target(self, *, ticker: str, row: int) -> np.int8:
        self._validate_row(ticker=ticker, row=row)
        return self._targets[ticker][row]

    def inverse_frequency_class_weights(self) -> tuple[float, float, float]:
        """Return authenticated equal-mass weights in fixed action order."""

        return inverse_frequency_entry_action_class_weights(
            self.manifest["action_target_counts"]
        )

    def balance_receipt(self) -> Mapping[str, object]:
        """Bind derived class weights to this exact target manifest."""

        counts = self.manifest["action_target_counts"]
        weights = self.inverse_frequency_class_weights()
        payload: dict[str, object] = {
            "schema": "propevolve_entry_action_balance_v1",
            "method": "inverse_frequency_v1",
            "source_manifest_identity_sha256": self.manifest["identity_sha256"],
            "action_order": ENTRY_ACTION_ORDER,
            "target_counts": {
                action: int(counts[action]) for action in ENTRY_ACTION_ORDER
            },
            "class_weights": {
                action: weights[index]
                for index, action in enumerate(ENTRY_ACTION_ORDER)
            },
        }
        payload["identity_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return _deep_freeze(payload)


def inverse_frequency_entry_action_class_weights(
    action_target_counts: Mapping[str, int],
) -> tuple[float, float, float]:
    """Equalize aggregate WAIT/Long/Short mass without arbitrary weights."""

    if not isinstance(action_target_counts, Mapping) or set(
        action_target_counts
    ) != set(ENTRY_ACTION_ORDER):
        raise ValueError("entry action target counts must match the action order")
    counts: list[int] = []
    for action in ENTRY_ACTION_ORDER:
        value = action_target_counts[action]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                "entry action target counts must be strictly positive integers"
            )
        value = int(value)
        if value <= 0:
            raise ValueError(
                "entry action target counts must be strictly positive integers"
            )
        counts.append(value)
    total = sum(counts)
    classes = len(counts)
    return tuple(total / (classes * count) for count in counts)


def build_post_launch_entry_supervision(
    *,
    long_launch: bool,
    short_launch: bool,
    long_economic_good: Sequence[bool],
    short_economic_good: Sequence[bool],
) -> PostLaunchEntrySupervision:
    """Build five next-open entry targets from side-specific economic truth."""

    if type(long_launch) is not bool or type(short_launch) is not bool:
        raise TypeError("launch flags must be bool")
    long_good = _strict_bool_targets(
        long_economic_good, name="long_economic_good"
    )
    short_good = _strict_bool_targets(
        short_economic_good, name="short_economic_good"
    )
    if not long_launch and not short_launch:
        raise ValueError("post-launch supervision requires at least one launch side")
    if long_launch and short_launch:
        return PostLaunchEntrySupervision(
            side="ambiguous",
            status="ambiguous",
            enter_fill_offset=None,
            candidates=tuple(
                CandidateEntryTarget(
                    decision_offset=index,
                    fill_offset=index + 1,
                    action=None,
                    available=False,
                    censored=False,
                    unavailable_reason="ambiguous_side",
                )
                for index in range(CANDIDATE_COUNT)
            ),
        )
    side: Literal["long", "short"] = "long" if long_launch else "short"
    economic_good = long_good if side == "long" else short_good
    if not any(economic_good):
        return PostLaunchEntrySupervision(
            side=side,
            status="abstain",
            enter_fill_offset=None,
            candidates=tuple(
                CandidateEntryTarget(
                    decision_offset=index,
                    fill_offset=index + 1,
                    action="WAIT",
                    available=True,
                    censored=False,
                    unavailable_reason=None,
                )
                for index in range(CANDIDATE_COUNT)
            ),
        )
    enter_index = economic_good.index(True)
    enter_action: EntryAction = (
        "ENTER_LONG_1" if side == "long" else "ENTER_SHORT_1"
    )
    candidates = tuple(
        CandidateEntryTarget(
            decision_offset=index,
            fill_offset=index + 1,
            action=(
                "WAIT"
                if index < enter_index
                else enter_action
                if index == enter_index
                else None
            ),
            available=index <= enter_index,
            censored=index > enter_index,
            unavailable_reason="after_entry" if index > enter_index else None,
        )
        for index in range(CANDIDATE_COUNT)
    )
    return PostLaunchEntrySupervision(
        side=side,
        status="enter",
        enter_fill_offset=enter_index + 1,
        candidates=candidates,
    )


_SPEC_KEYS = frozenset({
    "schema",
    "training_only",
    "decision_count",
    "fill_offsets",
    "execution",
    "risk_dollars",
    "launch",
    "continuation",
    "target_r",
    "stop_r",
    "horizon_bars",
    "collision",
    "loss_weight",
})
_FIXED_SPEC = {
    "schema": "post_launch_entry_v1",
    "training_only": True,
    "decision_count": 5,
    "fill_offsets": (1, 2, 3, 4, 5),
    "execution": "next_bar_open",
    "risk_dollars": 300.0,
    "launch": MappingProxyType({
        "favorable_r": 0.5,
        "adverse_r": 0.25,
        "horizon_bars": 3,
    }),
    "continuation": MappingProxyType({
        "favorable_r": 0.5,
        "adverse_r": 0.25,
        "horizon_bars": 3,
    }),
    "target_r": 2.0,
    "stop_r": 1.0,
    "horizon_bars": 150,
    "collision": "stop_first",
}


@dataclass(frozen=True, slots=True)
class _Event:
    side: Literal["long", "short"]
    anchor: int
    resolved: bool
    continuation: tuple[bool, ...] | None
    economic_win: tuple[bool, ...] | None
    economic_good: tuple[bool, ...] | None
    opportunity_good: tuple[tuple[bool, bool], ...] | None


def build_entry_action_targets(
    markets: Mapping[str, "MarketSeries"],
    spec: Mapping[str, object],
    *,
    point_values: Mapping[str, float],
    round_trip_fees: Mapping[str, float],
    training_end_exclusive: str,
) -> EntryActionTargets:
    """Build exact five-bar post-launch targets without reading or writing files."""

    if not isinstance(markets, Mapping) or not markets:
        raise ValueError("entry supervision requires at least one market")
    tickers = tuple(sorted(markets))
    if any(type(ticker) is not str or not ticker for ticker in tickers):
        raise ValueError("entry supervision market keys must be nonempty strings")
    contract = _validated_contract(spec)
    economics = _validated_market_economics(
        point_values,
        round_trip_fees,
        tickers=tickers,
        risk_dollars=float(contract["risk_dollars"]),
    )
    if type(training_end_exclusive) is not str or training_end_exclusive != "2025-01-01":
        raise ValueError("entry supervision training end contract drifted")
    boundary = np.datetime64(training_end_exclusive, "ns")
    all_targets: dict[str, np.ndarray] = {}
    all_opportunity_values: dict[str, np.ndarray] = {}
    all_metadata: dict[str, Mapping[int, EntryTargetMetadata]] = {}
    summaries: dict[str, object] = {}
    opportunity_summaries: dict[str, object] = {}

    for ticker in tickers:
        market = markets[ticker]
        if getattr(market, "ticker", None) != ticker:
            raise ValueError(f"market identity drift for {ticker}")
        timestamps = np.asarray(market.timestamps).astype("datetime64[ns]", copy=False)
        role_end = int(np.searchsorted(timestamps, boundary, side="left"))
        if role_end < 1:
            raise ValueError(f"market {ticker} has no rows before training end")
        started_at = time.monotonic()
        print(
            f"[entry-supervision] START ticker={ticker} "
            f"training_rows={role_end:,}",
            flush=True,
        )
        point_value = economics["point_values"][ticker]
        fee = economics["round_trip_fees"][ticker]
        events = _find_events(
            market,
            role_end=role_end,
            point_value=point_value,
            round_trip_fee=fee,
            contract=contract,
        )
        targets, opportunity_values, metadata, summary = _materialize_market_targets(
            rows=len(timestamps),
            role_end=role_end,
            events=events,
        )
        all_targets[ticker] = targets
        all_opportunity_values[ticker] = opportunity_values
        all_metadata[ticker] = MappingProxyType(metadata)
        target_sha256 = _target_digest(targets, metadata)
        summaries[ticker] = {
            **summary,
            "rows": len(timestamps),
            "training_rows": role_end,
            **_market_source_digests(market, stop=role_end),
            "targets_sha256": target_sha256,
        }
        opportunity_summaries[ticker] = {
            "rows": len(timestamps),
            "training_rows": role_end,
            "values_sha256": _opportunity_value_digest(opportunity_values),
            "available_rows": int(
                np.count_nonzero(np.isfinite(opportunity_values).all(axis=1))
            ),
        }
        print(
            f"[entry-supervision] COMPLETE ticker={ticker} "
            f"events={summary['events']:,} "
            f"enter_targets={summary['enter_targets']:,} "
            f"wait_targets={summary['wait_targets']:,} "
            f"elapsed_seconds={time.monotonic() - started_at:.3f} "
            f"identity={target_sha256}",
            flush=True,
        )

    action_target_counts = {
        action: sum(
            int(summaries[ticker]["action_target_counts"][action])
            for ticker in tickers
        )
        for action in ENTRY_ACTION_ORDER
    }
    manifest_payload: dict[str, object] = {
        "schema": contract["schema"],
        "contract": contract,
        "training_end_exclusive": training_end_exclusive,
        "point_values": economics["point_values"],
        "round_trip_fees": economics["round_trip_fees"],
        "label_semantics": {
            "fresh_lookback_bars": 5,
            "launch_collision": "adverse_first",
            "continuation_collision": "adverse_first",
            "economic_collision": "stop_first",
            "action_order": ENTRY_ACTION_ORDER,
            "unavailable_sentinel": -1,
        },
        "storage": {
            "target_dtype": "int8",
            "unavailable_sentinel": -1,
            "metadata": "sparse_mapping_by_supervised_row",
        },
        "tickers": tickers,
        "markets": summaries,
        "action_target_counts": action_target_counts,
    }
    manifest_payload["identity_sha256"] = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    opportunity_manifest_payload: dict[str, object] = {
        "schema": "post_launch_entry_opportunity_value_v1",
        "training_only": True,
        "source_entry_manifest_identity_sha256": manifest_payload[
            "identity_sha256"
        ],
        "training_end_exclusive": training_end_exclusive,
        "semantics": {
            "order": ENTRY_ACTION_ORDER,
            "wait": 0.0,
            "continuation_and_economic_winner": 2.0,
            "non_winner": -1.0,
        },
        "storage": {
            "dtype": "float32",
            "unavailable": "all_nan",
        },
        "tickers": tickers,
        "markets": opportunity_summaries,
    }
    opportunity_manifest_payload["identity_sha256"] = hashlib.sha256(
        json.dumps(
            opportunity_manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return EntryActionTargets(
        _targets=MappingProxyType(all_targets),
        _opportunity_values=MappingProxyType(all_opportunity_values),
        _metadata=MappingProxyType(all_metadata),
        manifest=_deep_freeze(manifest_payload),
        opportunity_value_manifest=_deep_freeze(opportunity_manifest_payload),
    )


def _validated_contract(
    spec: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(spec, Mapping):
        raise TypeError("entry supervision spec must be a mapping")
    keys = set(spec)
    label_keys = keys - {"action_class_balance"}
    if label_keys != _SPEC_KEYS or keys - _SPEC_KEYS - {"action_class_balance"}:
        raise ValueError(
            "entry supervision spec keys drifted: "
            f"missing={sorted(_SPEC_KEYS - label_keys)} "
            f"extra={sorted(keys - _SPEC_KEYS - {'action_class_balance'})}"
        )
    balance = spec.get("action_class_balance")
    if balance is not None and (
        not isinstance(balance, Mapping)
        or set(balance) != {"schema", "action_order"}
        or balance.get("schema") != "inverse_frequency_v1"
        or tuple(balance.get("action_order", ())) != ENTRY_ACTION_ORDER
    ):
        raise ValueError("entry supervision class balance contract drifted")
    for name, expected in _FIXED_SPEC.items():
        value = spec[name]
        if name == "fill_offsets":
            if isinstance(value, (str, bytes)) or tuple(value) != expected:
                raise ValueError(f"entry supervision {name} contract drifted")
        elif name in {"launch", "continuation"}:
            if not isinstance(value, Mapping) or set(value) != set(expected):
                raise ValueError(f"entry supervision {name} contract drifted")
            for field, phase_expected in expected.items():
                phase_value = value[field]
                if (
                    isinstance(phase_value, bool)
                    or not isinstance(phase_value, Real)
                    or not math.isfinite(float(phase_value))
                    or float(phase_value) != float(phase_expected)
                ):
                    raise ValueError(f"entry supervision {name} contract drifted")
        elif isinstance(expected, float):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError(f"entry supervision {name} contract drifted")
        elif type(value) is not type(expected) or value != expected:
            raise ValueError(f"entry supervision {name} contract drifted")

    loss_weight = spec["loss_weight"]
    if (
        isinstance(loss_weight, bool)
        or not isinstance(loss_weight, Real)
        or not math.isfinite(float(loss_weight))
        or float(loss_weight) <= 0.0
    ):
        raise ValueError("entry supervision loss_weight must be finite and positive")
    normalized: dict[str, object] = {
        **_FIXED_SPEC,
        "fill_offsets": tuple(_FIXED_SPEC["fill_offsets"]),
        "launch": dict(_FIXED_SPEC["launch"]),
        "continuation": dict(_FIXED_SPEC["continuation"]),
        "loss_weight": float(loss_weight),
    }
    return normalized


def _validated_market_economics(
    point_values: Mapping[str, float],
    round_trip_fees: Mapping[str, float],
    *,
    tickers: tuple[str, ...],
    risk_dollars: float,
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for field in ("point_values", "round_trip_fees"):
        values = point_values if field == "point_values" else round_trip_fees
        if not isinstance(values, Mapping) or set(values) != set(tickers):
            raise ValueError(f"entry supervision {field} must exactly match markets")
        converted: dict[str, float] = {}
        for ticker in tickers:
            value = values[ticker]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"entry supervision {field} must be finite numeric")
            converted[ticker] = float(value)
        normalized[field] = converted
    if any(value <= 0 for value in normalized["point_values"].values()):
        raise ValueError("entry supervision point values must be positive")
    if any(
        fee < 0 or fee >= risk_dollars
        for fee in normalized["round_trip_fees"].values()
    ):
        raise ValueError("entry supervision fees violate the risk contract")
    smallest_adverse_r = min(
        float(_FIXED_SPEC["launch"]["adverse_r"]),
        float(_FIXED_SPEC["continuation"]["adverse_r"]),
        float(_FIXED_SPEC["stop_r"]),
    )
    if any(
        fee >= smallest_adverse_r * risk_dollars
        for fee in normalized["round_trip_fees"].values()
    ):
        raise ValueError("entry supervision fee leaves no positive adverse distance")
    return normalized


def _find_events(
    market: "MarketSeries",
    *,
    role_end: int,
    point_value: float,
    round_trip_fee: float,
    contract: Mapping[str, object],
) -> tuple[_Event, ...]:
    open_ = np.asarray(market.open, dtype=np.float64)
    high = np.asarray(market.high, dtype=np.float64)
    low = np.asarray(market.low, dtype=np.float64)
    raw: dict[str, np.ndarray] = {}
    launch = contract["launch"]
    continuation_spec = contract["continuation"]
    launch_horizon = int(launch["horizon_bars"])
    for side in ("long", "short"):
        raw[side] = _all_targets_before_adverse(
            open_,
            high,
            low,
            role_end=role_end,
            side=side,
            risk_dollars=float(contract["risk_dollars"]),
            point_value=point_value,
            round_trip_fee=round_trip_fee,
            target_r=float(launch["favorable_r"]),
            adverse_r=float(launch["adverse_r"]),
            horizon=launch_horizon,
        )

    events: list[_Event] = []
    lookback = CANDIDATE_COUNT
    economic_horizon = int(contract["horizon_bars"])
    for side in ("long", "short"):
        fresh = [
            decision
            for decision in np.flatnonzero(raw[side])
            if not bool(raw[side][max(0, decision - lookback):decision].any())
        ]
        for raw_anchor in fresh:
            anchor = int(raw_anchor)
            # Candidate four fills at anchor+5.  A complete H-bar path then
            # ends at anchor+5+H-1, so the exclusive bound is anchor+5+H.
            resolved = anchor + CANDIDATE_COUNT + economic_horizon <= role_end
            if not resolved:
                events.append(
                    _Event(side, anchor, False, None, None, None, None)
                )
                continue
            continuation = []
            economic_win = []
            opportunity_good = []
            for offset in range(CANDIDATE_COUNT):
                decision = anchor + offset
                side_outcomes: dict[str, tuple[bool, bool]] = {}
                for candidate_side in ("long", "short"):
                    continued = _target_before_adverse(
                        open_,
                        high,
                        low,
                        decision=decision,
                        role_end=role_end,
                        side=candidate_side,
                        risk_dollars=float(contract["risk_dollars"]),
                        point_value=point_value,
                        round_trip_fee=round_trip_fee,
                        target_r=float(continuation_spec["favorable_r"]),
                        adverse_r=float(continuation_spec["adverse_r"]),
                        horizon=int(continuation_spec["horizon_bars"]),
                    )
                    won = _target_before_adverse(
                        open_,
                        high,
                        low,
                        decision=decision,
                        role_end=role_end,
                        side=candidate_side,
                        risk_dollars=float(contract["risk_dollars"]),
                        point_value=point_value,
                        round_trip_fee=round_trip_fee,
                        target_r=float(contract["target_r"]),
                        adverse_r=float(contract["stop_r"]),
                        horizon=economic_horizon,
                    )
                    side_outcomes[candidate_side] = (continued, won)
                continuation.append(side_outcomes[side][0])
                economic_win.append(side_outcomes[side][1])
                opportunity_good.append(tuple(
                    side_outcomes[candidate_side][0]
                    and side_outcomes[candidate_side][1]
                    for candidate_side in ("long", "short")
                ))
            good = tuple(
                continued and won
                for continued, won in zip(continuation, economic_win)
            )
            events.append(_Event(
                side,
                anchor,
                True,
                tuple(continuation),
                tuple(economic_win),
                good,
                tuple(opportunity_good),
            ))
    return tuple(sorted(events, key=lambda event: (event.anchor, event.side)))


def _all_targets_before_adverse(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    role_end: int,
    side: str,
    risk_dollars: float,
    point_value: float,
    round_trip_fee: float,
    target_r: float,
    adverse_r: float,
    horizon: int,
) -> np.ndarray:
    """Vectorize the fixed three-bar launch scan without an N-by-H table."""

    output = np.zeros(role_end, dtype=np.bool_)
    decisions = role_end - horizon
    if decisions <= 0:
        return output
    entries = open_[1:decisions + 1]
    favorable_points = (target_r * risk_dollars + round_trip_fee) / point_value
    adverse_points = (adverse_r * risk_dollars - round_trip_fee) / point_value
    if not adverse_points > 0.0:
        raise ValueError("entry supervision adverse distance must be positive")
    alive = np.ones(decisions, dtype=np.bool_)
    resolved = np.zeros(decisions, dtype=np.bool_)
    for offset in range(horizon):
        selected_high = high[1 + offset:decisions + 1 + offset]
        selected_low = low[1 + offset:decisions + 1 + offset]
        if side == "long":
            favorable = selected_high - entries
            adverse = entries - selected_low
        else:
            favorable = entries - selected_low
            adverse = selected_high - entries
        adverse_hit = alive & (adverse >= adverse_points)
        alive &= ~adverse_hit
        favorable_hit = alive & (favorable >= favorable_points)
        resolved |= favorable_hit
        alive &= ~favorable_hit
    output[:decisions] = resolved
    return output


def _target_before_adverse(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    decision: int,
    role_end: int,
    side: str,
    risk_dollars: float,
    point_value: float,
    round_trip_fee: float,
    target_r: float,
    adverse_r: float,
    horizon: int,
) -> bool:
    fill_index = decision + 1
    stop = fill_index + horizon
    if fill_index >= role_end or stop > role_end:
        return False
    favorable_points = (target_r * risk_dollars + round_trip_fee) / point_value
    adverse_points = (adverse_r * risk_dollars - round_trip_fee) / point_value
    if not adverse_points > 0.0:
        raise ValueError("entry supervision adverse distance must be positive")
    entry = float(open_[fill_index])
    for index in range(fill_index, stop):
        if side == "long":
            favorable = float(high[index]) - entry
            adverse = entry - float(low[index])
        else:
            favorable = entry - float(low[index])
            adverse = float(high[index]) - entry
        # OHLC cannot reveal within-bar order, so adverse always wins ties.
        if adverse >= adverse_points:
            return False
        if favorable >= favorable_points:
            return True
    return False


def _materialize_market_targets(
    *,
    rows: int,
    role_end: int,
    events: tuple[_Event, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, EntryTargetMetadata],
    dict[str, int],
]:
    targets = np.full(rows, -1, dtype=np.int8)
    opportunity_values = np.full((rows, 3), np.nan, dtype=np.float32)
    metadata: dict[int, EntryTargetMetadata] = {}
    ambiguous_ids: set[int] = set()
    ambiguous_anchors: dict[int, tuple[int, ...]] = {}
    for left_id, left in enumerate(events):
        for right_id in range(left_id + 1, len(events)):
            right = events[right_id]
            if right.anchor > left.anchor + CANDIDATE_COUNT - 1:
                break
            if left.side != right.side and right.anchor <= left.anchor + CANDIDATE_COUNT - 1:
                ambiguous_ids.update((left_id, right_id))
    for event_id in ambiguous_ids:
        event = events[event_id]
        anchors = tuple(sorted({
            other.anchor
            for other in events
            if other.side != event.side
            and other.anchor <= event.anchor + CANDIDATE_COUNT - 1
            and event.anchor <= other.anchor + CANDIDATE_COUNT - 1
        } | {event.anchor}))
        ambiguous_anchors[event_id] = anchors

    for event_id, event in enumerate(events):
        if event_id in ambiguous_ids:
            for row in range(event.anchor, min(event.anchor + CANDIDATE_COUNT, role_end)):
                existing = metadata.get(row)
                anchors = set(ambiguous_anchors[event_id])
                if existing is not None:
                    anchors.update(existing.event_anchor_rows)
                metadata[row] = EntryTargetMetadata(
                    side="ambiguous",
                    event_anchor_rows=tuple(sorted(anchors)),
                    candidate_decision_offset=None,
                    fill_offset=None,
                    continuation=None,
                    economic_win=None,
                    economic_good=None,
                    available=False,
                    censored=False,
                    unavailable_reason="ambiguous_side",
                )
            continue
        if not event.resolved:
            for offset in range(CANDIDATE_COUNT):
                row = event.anchor + offset
                if row >= role_end:
                    break
                metadata[row] = EntryTargetMetadata(
                    side=event.side,
                    event_anchor_rows=(event.anchor,),
                    candidate_decision_offset=offset,
                    fill_offset=offset + 1,
                    continuation=None,
                    economic_win=None,
                    economic_good=None,
                    available=False,
                    censored=False,
                    unavailable_reason="unresolved_split_end",
                )
            continue
        assert event.continuation is not None
        assert event.economic_win is not None
        assert event.economic_good is not None
        assert event.opportunity_good is not None
        supervision = build_post_launch_entry_supervision(
            long_launch=event.side == "long",
            short_launch=event.side == "short",
            long_economic_good=(
                event.economic_good if event.side == "long" else (False,) * 5
            ),
            short_economic_good=(
                event.economic_good if event.side == "short" else (False,) * 5
            ),
        )
        for candidate in supervision.candidates:
            row = event.anchor + candidate.decision_offset
            action = (
                Action[candidate.action]
                if candidate.action is not None
                else None
            )
            targets[row] = -1 if action is None else int(action)
            if candidate.available:
                long_good, short_good = event.opportunity_good[
                    candidate.decision_offset
                ]
                opportunity_values[row] = (
                    0.0,
                    2.0 if long_good else -1.0,
                    2.0 if short_good else -1.0,
                )
            visible_outcomes = not candidate.censored
            metadata[row] = EntryTargetMetadata(
                side=event.side,
                event_anchor_rows=(event.anchor,),
                candidate_decision_offset=candidate.decision_offset,
                fill_offset=candidate.fill_offset,
                continuation=(
                    event.continuation[candidate.decision_offset]
                    if visible_outcomes else None
                ),
                economic_win=(
                    event.economic_win[candidate.decision_offset]
                    if visible_outcomes else None
                ),
                economic_good=(
                    event.economic_good[candidate.decision_offset]
                    if visible_outcomes else None
                ),
                available=candidate.available,
                censored=candidate.censored,
                unavailable_reason=candidate.unavailable_reason,
            )
    targets.setflags(write=False)
    opportunity_values.setflags(write=False)
    return (
        targets,
        opportunity_values,
        metadata,
        {
            "events": len(events),
            "resolved_events": sum(event.resolved for event in events),
            "unresolved_events": sum(not event.resolved for event in events),
            "ambiguous_events": len(ambiguous_ids),
            "enter_targets": int(np.count_nonzero(
                (targets == int(Action.ENTER_LONG_1))
                | (targets == int(Action.ENTER_SHORT_1))
            )),
            "wait_targets": int(np.count_nonzero(targets == int(Action.WAIT))),
            "action_target_counts": {
                action.name: int(np.count_nonzero(targets == int(action)))
                for action in (
                    Action.WAIT,
                    Action.ENTER_LONG_1,
                    Action.ENTER_SHORT_1,
                )
            },
        },
    )


def _market_source_digests(
    market: "MarketSeries", *, stop: int
) -> dict[str, str]:
    component_digests: dict[str, str] = {}
    source = hashlib.sha256()
    for name in ("timestamps", "open", "high", "low", "close"):
        values = np.ascontiguousarray(np.asarray(getattr(market, name))[:stop])
        component = hashlib.sha256()
        for digest in (component, source):
            digest.update(name.encode())
            digest.update(values.dtype.str.encode())
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
        component_digests[f"{name}_sha256"] = component.hexdigest()
    return {"source_sha256": source.hexdigest(), **component_digests}


def _target_digest(
    targets: np.ndarray,
    metadata: Mapping[int, EntryTargetMetadata],
) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(targets).tobytes())
    payload = {
        str(row): {
            field: _json_value(getattr(item, field))
            for field in item.__dataclass_fields__
        }
        for row, item in sorted(metadata.items())
    }
    digest.update(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode())
    return digest.hexdigest()


def _opportunity_value_digest(opportunity_values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(opportunity_values.dtype.str.encode())
    digest.update(str(opportunity_values.shape).encode())
    digest.update(np.ascontiguousarray(opportunity_values).tobytes())
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, tuple):
        return tuple(_json_value(item) for item in value)
    return value


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _strict_bool_targets(values: Sequence[bool], *, name: str) -> tuple[bool, ...]:
    try:
        targets = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly 5 bool values") from exc
    if len(targets) != CANDIDATE_COUNT or any(type(value) is not bool for value in targets):
        raise ValueError(f"{name} must contain exactly 5 bool values")
    return targets


__all__ = [
    "CANDIDATE_COUNT",
    "ENTRY_ACTION_ORDER",
    "CandidateEntryTarget",
    "EntryActionTargets",
    "EntryAction",
    "EntrySide",
    "EntryStatus",
    "EntryTargetMetadata",
    "PostLaunchEntrySupervision",
    "UnavailableReason",
    "build_entry_action_targets",
    "build_post_launch_entry_supervision",
    "inverse_frequency_entry_action_class_weights",
]
