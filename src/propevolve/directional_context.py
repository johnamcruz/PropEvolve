"""Causal signed momentum coordinates; no labels, teacher, or action rules.

Inputs are consecutive completed closes from one correctly constructed price
series. Horizons count observed bars, not wall-clock time. Callers own session,
contract-roll and missing-bar policy. Output at t is available after close t;
it must not be used to fill a trade earlier than the next executable price.
"""
from dataclasses import dataclass
from collections import deque
from collections.abc import Mapping
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class DirectionalContextSpec:
    enabled: bool
    horizons: tuple[int, ...]
    volatility_window: int
    volatility_floor: float

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if (not isinstance(self.horizons, (tuple, list)) or not self.horizons
                or any(type(h) is not int or h < 1 for h in self.horizons)
                or len(set(self.horizons)) != len(self.horizons)):
            raise ValueError("horizons must be distinct positive integers")
        object.__setattr__(self, "horizons", tuple(self.horizons))
        if type(self.volatility_window) is not int or self.volatility_window < 2:
            raise ValueError("volatility_window must be an integer >= 2")
        if (isinstance(self.volatility_floor, bool)
                or not isinstance(self.volatility_floor, Real)
                or not np.isfinite(self.volatility_floor)
                or self.volatility_floor <= 0):
            raise ValueError("volatility_floor must be finite and positive")

    @classmethod
    def from_mapping(cls, settings: Mapping) -> "DirectionalContextSpec":
        if not isinstance(settings, Mapping) or set(settings) != {
            "enabled", "horizons", "volatility_window", "volatility_floor"
        }:
            raise ValueError("directional context requires exactly its four settings")
        return cls(**settings)

    @property
    def channels(self) -> tuple[str, ...]:
        return (tuple(f"momentum_{h}_bars" for h in self.horizons)
                if self.enabled else ())


@dataclass(frozen=True)
class DirectionalContextResult:
    values: np.ndarray
    available: np.ndarray


class DirectionalContext:
    def __init__(self, spec: DirectionalContextSpec):
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        """Start a new independent series, whose first bar index is zero."""
        self._logs = deque(maxlen=max(*self.spec.horizons,
                                     self.spec.volatility_window)+1)
        self._next_bar = 0

    def update(self, bar_index: int, close: float) -> DirectionalContextResult:
        """Consume one completed bar; return one vector in horizon order."""
        if type(bar_index) is not int or bar_index != self._next_bar:
            raise ValueError("bar_index must be the next consecutive integer")
        if (isinstance(close, bool) or not isinstance(close, Real)
                or not np.isfinite(close) or close <= 0):
            raise ValueError("close must be finite and positive")
        self._logs.append(float(np.log(close)))
        self._next_bar += 1
        logs = np.asarray(self._logs)
        spec = self.spec
        values = np.full(len(spec.channels), np.nan)
        if spec.enabled and len(logs) > spec.volatility_window:
            volatility = max(float(np.std(np.diff(
                logs[-spec.volatility_window-1:]
            ))), spec.volatility_floor)
            for col, horizon in enumerate(spec.horizons):
                if len(logs) > horizon:
                    values[col] = ((logs[-1]-logs[-horizon-1])
                                   / volatility / np.sqrt(horizon))
        return DirectionalContextResult(values, np.isfinite(values))

    def transform(self, closes) -> DirectionalContextResult:
        """Independent batch calculation; does not change incremental state.

        Uses population standard deviation of the latest volatility_window
        one-bar log returns (including the just-completed bar). No fitting or
        clipping. Warmup values are NaN with a false per-horizon mask.
        """
        prices = np.asarray(closes, dtype=np.float64)
        if prices.ndim != 1 or not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("closes must be a finite positive one-dimensional series")
        values = np.full((len(prices), len(self.spec.channels)), np.nan)
        stream = DirectionalContext(self.spec)
        for row, close in enumerate(prices):
            values[row] = stream.update(row, close).values
        return DirectionalContextResult(values, np.isfinite(values))
