"""Causal agent observation assembled from FFM context and account state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision import PositionSide


@dataclass(frozen=True)
class AccountState:
    realized_pnl: float = 0.0
    equity_pnl: float = 0.0
    peak_equity_pnl: float = 0.0
    position_side: PositionSide = PositionSide.FLAT
    position_size: int = 0
    max_position_size: int = 2
    unrealized_pnl: float = 0.0
    session_remaining: float = 1.0
    challenge_remaining: float = 1.0
    point_value: float = 0.0
    round_trip_fee: float = 0.0


class ObservationAssembler:
    """Join an unchanged frozen embedding with coordinate-invariant state."""

    ACCOUNT_DIM = 12

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_loss: float = 3_000.0,
        profit_target: float = 6_000.0,
    ) -> None:
        if embedding_dim < 1 or max_loss <= 0 or profit_target <= 0:
            raise ValueError("observation dimensions and economics must be positive")
        self.embedding_dim = int(embedding_dim)
        self.max_loss = float(max_loss)
        self.profit_target = float(profit_target)

    @property
    def output_dim(self) -> int:
        return self.embedding_dim + self.ACCOUNT_DIM

    def assemble(self, embedding: np.ndarray, account: AccountState) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding shape {embedding.shape} != ({self.embedding_dim},)")
        if not np.isfinite(embedding).all():
            raise ValueError("embedding must be finite")
        maximum_size = max(1, int(account.max_position_size))
        account_values = np.asarray(
            [
                account.realized_pnl / self.profit_target,
                account.equity_pnl / self.profit_target,
                account.peak_equity_pnl / self.profit_target,
                (account.equity_pnl + self.max_loss) / self.max_loss,
                (account.peak_equity_pnl - account.equity_pnl) / self.max_loss,
                float(account.position_side),
                account.position_size / maximum_size,
                account.unrealized_pnl / self.max_loss,
                account.session_remaining,
                account.challenge_remaining,
                account.point_value / self.max_loss,
                account.round_trip_fee / self.max_loss,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(account_values).all():
            raise ValueError("account state must be finite")
        return np.concatenate((embedding, account_values))
