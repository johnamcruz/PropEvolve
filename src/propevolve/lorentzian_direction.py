"""Dependency-free, causal directional evidence; never a trading action.

Uses the Lorentzian distance formula documented by AI Edge's Python port:
https://github.com/artificial-intelligence-edge/lorentzian-classification
Unlike its chart-oriented ANN, this helper uses matured forward labels and
nearest neighbours in bounded history. It is not Pine signal-parity code.
Caller supplies causally normalized features, separately for each market.
"""
from collections import deque
from dataclasses import dataclass
from heapq import nsmallest
import math
from typing import Literal, Sequence


@dataclass(frozen=True, slots=True)
class DirectionEvidence:
    direction: Literal['long', 'short', 'uncertain']
    vote: float
    neighbors_used: int
    # Vote is an average signed label, NOT a calibrated success probability.


class LorentzianDirection:
    """One update per completed bar; all research settings supplied by caller."""

    def __init__(self, *, neighbors: int, history_size: int,
                 label_horizon: int, minimum_move: float, minimum_vote: float,
                 label_source: str = 'close'):
        if label_source not in {'close', 'economic'}:
            raise ValueError('label_source must be close or economic')
        self.label_source = label_source
        for name, value in [('neighbors', neighbors), ('history_size', history_size),
                            ('label_horizon', label_horizon)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if neighbors > history_size:
            raise ValueError('neighbors cannot exceed history_size')
        for name, value in [('minimum_move', minimum_move), ('minimum_vote', minimum_vote)]:
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and within [0,1]')
        self.neighbors = neighbors
        self.label_horizon = label_horizon
        self.minimum_move = minimum_move
        self.minimum_vote = minimum_vote
        self._history = deque(maxlen=history_size)
        self._pending = deque()
        self._last_bar = None
        self._width = None

    @property
    def history_count(self) -> int:
        return len(self._history)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def update(self, bar_index: int, features: Sequence[float], price: float,
               *, matured_label: int | None = None) -> DirectionEvidence:
        """Label old observations using this closed price, then score this bar.

        Close-mode labels use forward close-to-close fractional movement over
        label_horizon observations. Economic mode instead requires the caller
        to supply the signed outcome of the example exactly label_horizon
        bars ago; earlier or missing labels are rejected before state changes.
        Indices must be consecutive; timestamps/session rules belong to caller.
        Invalid inputs are rejected before state changes.
        """
        if type(bar_index) is not int or bar_index < 0 or (
            self._last_bar is not None and bar_index != self._last_bar + 1
        ):
            raise ValueError('bar_index must be consecutive and nonnegative')
        values = tuple(float(value) for value in features)
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError('features must be finite and nonempty')
        if self._width is not None and len(values) != self._width:
            raise ValueError('feature dimension changed')
        if isinstance(price, bool) or not math.isfinite(price) or price <= 0:
            raise ValueError('price must be finite and positive')
        needs_label = (self.label_source == 'economic'
                       and len(self._pending) == self.label_horizon)
        if needs_label:
            if type(matured_label) is not int or matured_label not in {-1, 0, 1}:
                raise ValueError('economic label must be matured and signed')
        elif matured_label is not None:
            raise ValueError('no economic label may be supplied before maturity')
        if len(self._pending) == self.label_horizon:
            old_values, old_price = self._pending.popleft()
            movement = (price - old_price) / old_price
            label = int(movement > self.minimum_move) - int(movement < -self.minimum_move)
            if self.label_source == 'economic':
                label = matured_label
            self._history.append((old_values, label))
        self._pending.append((values, float(price)))
        self._last_bar = bar_index
        self._width = len(values)
        if len(self._history) < self.neighbors:
            return DirectionEvidence('uncertain', 0., len(self._history))
        # Deterministic history-order tie breaking; no current/unmatured rows.
        distances = (
            (sum(math.log1p(abs(a-b)) for a,b in zip(values, old)), index, label)
            for index,(old,label) in enumerate(self._history)
        )
        chosen = nsmallest(self.neighbors, distances)
        vote = sum(item[2] for item in chosen) / self.neighbors
        direction = ('long' if vote > self.minimum_vote else
                     'short' if vote < -self.minimum_vote else 'uncertain')
        return DirectionEvidence(direction, vote, self.neighbors)
