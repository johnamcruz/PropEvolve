"""Expansion + order-flow setup channels, as causal per-bar context for the policy.

The algoTraderAI strategy this mirrors takes its TIMING from a Chronos2 Expansion score
(a rising edge through a fitted threshold arms a short window) and its SIDE from
45-minute order-flow persistence (the share of recent one-minute bars with positive
aggressor delta, minus one half). Its PPO risk controller only chose skip/enter and
hold/close on top of that fixed rule.

Here the intent is different: the reasoning policy should LEARN the setup rather than be
handed a binary gate, so these arrive as observation channels. The policy sees the same
ingredients the rule sees and decides for itself; the RL stage still teaches the prop
constraints. A hard gate remains possible on top (``SetupSignalSpec.gate_entries``), but
it is off by default precisely so the learning question stays open.

Provenance. The channels are read straight from the frozen ffm-strategies research
bundle (``runs/nq_expansion_flow_trigger_v1/bundle/signals.parquet``), the SAME parquet
algoTraderAI's signal cache is built from — Expansion checkpoint ``a1e56e4d``, paired
3-minute + developing-15-minute cache ``63a9bd04``, encoder ``de8dc18e``. There is no
conversion step and no second copy, so the reasoning run and the PPO run cannot drift
onto different signals.

Causality. Every channel is attached to the bar whose close produced it and is consumed
on the following bar's open, matching both the research rule and this environment's
next-bar-open execution. Bars the bundle never covered, or covered without a usable
signal, are zero with the availability channel clear, so the policy can tell "no signal"
from "signal reads zero".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bundle column -> exposed channel, and the scale onto roughly [-1, 1]. No scale clips a
# real value, so nothing is destroyed. ``flow_persistence`` is naturally -0.5..0.5.
_SOURCE: tuple[tuple[str, str, float], ...] = (
    ("expansion_pooled_score", "expansion_score", 1.0),
    ("armed", "expansion_armed", 1.0),
    ("persistence_45", "flow_persistence", 2.0),
    ("trigger", "setup_trigger", 1.0),
    ("side", "setup_side", 1.0),
)
CHANNEL_NAMES: tuple[str, ...] = tuple(name for _, name, _ in _SOURCE) + ("setup_available",)
_AVAILABLE = CHANNEL_NAMES.index("setup_available")
# Timing without a side is not this setup, so availability needs BOTH of these finite.
_REQUIRED_FINITE = ("expansion_pooled_score", "persistence_45")


@dataclass(frozen=True)
class SetupSignalSpec:
    """Which setup channels reach the policy, and whether they also gate entries."""

    state: str = "off"
    gate_entries: bool = False

    def __post_init__(self) -> None:
        if self.state not in {"off", "expansion_flow_v1"}:
            raise ValueError("unsupported setup-signal state")
        if not isinstance(self.gate_entries, bool):
            raise ValueError("gate_entries must be boolean")
        if self.state == "off" and self.gate_entries:
            raise ValueError("cannot gate entries with the setup signal disabled")

    @classmethod
    def from_config(cls, config: dict | None) -> "SetupSignalSpec":
        return cls() if config is None else cls(**config)

    @property
    def output_dim(self) -> int:
        return 0 if self.state == "off" else len(CHANNEL_NAMES)


def read_setup_bundle(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the research bundle parquet into (timestamps, channels).

    Returns bar-open timestamps as ``datetime64[ns]`` and an ``(n, 6)`` float32 matrix in
    ``CHANNEL_NAMES`` order.
    """
    import pandas as pd

    path = Path(path)
    if path.is_dir():
        path = path / "signals.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"setup-signal bundle not found: {path}")
    frame = pd.read_parquet(path)
    required = {"bar_open_utc", *(column for column, _, _ in _SOURCE)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"setup-signal bundle {path} is missing columns {missing}")

    stamps = pd.DatetimeIndex(frame["bar_open_utc"])
    stamps = stamps.tz_convert(None) if stamps.tz is not None else stamps
    order = np.argsort(stamps.to_numpy())
    stamps = stamps[order].to_numpy(dtype="datetime64[ns]")
    frame = frame.iloc[order]

    values = {}
    for column, _, _ in _SOURCE:
        series = np.asarray(frame[column].to_numpy(), dtype=np.float64)
        if column == "expansion_pooled_score":
            # the bundle encodes "never scored" as a negative sentinel; treating it as
            # 0.0 would read as a confident "no expansion" instead of "unknown"
            series = np.where(series >= 0.0, series, np.nan)
        values[column] = series
    available = np.ones(len(frame), dtype=bool)
    for column in _REQUIRED_FINITE:
        available &= np.isfinite(values[column])

    channels = np.zeros((len(frame), len(CHANNEL_NAMES)), dtype=np.float32)
    for index, (column, _, scale) in enumerate(_SOURCE):
        channels[:, index] = np.where(available, np.nan_to_num(values[column]) * scale, 0.0)
    channels[:, _AVAILABLE] = available.astype(np.float32)
    if not np.isfinite(channels).all():
        raise RuntimeError("setup channels must be finite")
    return stamps, channels


def align_setup_channels(bundle_timestamps: np.ndarray, bundle_channels: np.ndarray,
                         timestamps: np.ndarray) -> np.ndarray:
    """Align bundle channels onto ``timestamps`` by EXACT bar-open match.

    Never nearest-neighbour: a bar the bundle did not cover must read unavailable rather
    than inherit a neighbour's signal.
    """
    target = np.asarray(timestamps).astype("datetime64[ns]")
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional array")
    if len(target) > 1 and not np.all(target[1:] > target[:-1]):
        raise ValueError("timestamps must be strictly increasing")
    out = np.zeros((len(target), len(CHANNEL_NAMES)), dtype=np.float32)
    if len(bundle_timestamps) == 0:
        return out
    position = np.searchsorted(bundle_timestamps, target)
    in_range = position < len(bundle_timestamps)
    matched = np.zeros(len(target), dtype=bool)
    matched[in_range] = bundle_timestamps[position[in_range]] == target[in_range]
    out[matched] = bundle_channels[position[matched]]
    return out


def load_setup_channels(path: str | Path, timestamps: np.ndarray) -> np.ndarray:
    """Read the bundle parquet and align it to ``timestamps`` in one call."""
    stamps, channels = read_setup_bundle(path)
    return align_setup_channels(stamps, channels, timestamps)


def entry_side_from_channels(row: np.ndarray) -> int:
    """The rule's side for one bar: +1 long, -1 short, 0 no setup.

    Used only when ``gate_entries`` is on; the learning configuration leaves the side to
    the policy so that whether it rediscovers the flow side stays measurable.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (len(CHANNEL_NAMES),):
        raise ValueError(f"setup row must have {len(CHANNEL_NAMES)} channels")
    if not (row[_AVAILABLE] > 0.0 and row[CHANNEL_NAMES.index("setup_trigger")] > 0.0):
        return 0
    return int(np.sign(row[CHANNEL_NAMES.index("setup_side")]))
