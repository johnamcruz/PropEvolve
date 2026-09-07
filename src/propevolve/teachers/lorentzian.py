"""Training-only LC evidence from existing causal Expansion vectors.

Four-channel envelope reuses the directional confluence score product:
positive vote * 1, negative vote * 1. These are strengths, not calibrated
probabilities, Trend predictions, trading actions, or economic winner labels.
"""
from dataclasses import dataclass

import numpy as np

from ..lorentzian_direction import LorentzianDirection


CHANNELS = (
    "lc_long_direction_strength", "lc_short_direction_strength",
    "lc_long_scale", "lc_short_scale",
)


@dataclass(frozen=True)
class LorentzianTargets:
    values: dict[str, np.ndarray]

    @classmethod
    def build(cls, markets, expansion, settings, *, economic_labels=None):
        values = {}
        economic = settings.get('label_source', 'close') == 'economic'
        if economic != (economic_labels is not None):
            raise ValueError('economic LC requires explicit aligned outcome labels')
        for ticker, market in markets.items():
            helper = LorentzianDirection(**settings)
            labels = None if not economic else np.asarray(economic_labels[ticker])
            if labels is not None and (
                labels.shape != (len(market.close),)
                or not np.isin(labels, [-1, 0, 1]).all()
            ):
                raise ValueError('economic LC labels must be aligned signed outcomes')
            targets = np.zeros((len(market.close), 4), dtype=np.float32)
            targets[:, 2:] = 1.0
            for row, price in enumerate(market.close):
                features = expansion.target(ticker, row)
                if features is None:
                    # Do not label across a missing-input gap, or suppress
                    # independent Expansion/Regime supervision during warmup.
                    helper = LorentzianDirection(**settings)
                    continue
                features = np.asarray(features)
                if (features.ndim != 1 or not features.size or features.size % 4
                        or not np.isfinite(features).all()):
                    raise ValueError("LC requires complete finite Expansion snapshots")
                if np.any((features < 0) | (features > 1)):
                    raise ValueError("LC Expansion inputs must be within [0,1]")
                matured = None
                if economic and helper.pending_count == helper.label_horizon:
                    matured = int(labels[row-helper.label_horizon])
                evidence = helper.update(row, features, float(price), matured_label=matured)
                if evidence.direction == "long":
                    targets[row, 0] = evidence.vote
                elif evidence.direction == "short":
                    targets[row, 1] = -evidence.vote
            targets.flags.writeable = False
            values[ticker] = targets
        return cls(values)

    def target(self, ticker, row):
        if ticker not in self.values:
            raise ValueError(f"LC has no aligned market {ticker}")
        if type(row) is not int or not 0 <= row < len(self.values[ticker]):
            raise IndexError("LC row is outside the aligned market")
        return self.values[ticker][row].copy()
