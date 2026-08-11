"""Causal agent observation assembled from FFM context and account state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision import PositionSide


@dataclass(frozen=True)
class TradeManagementObservationSpec:
    """Causal coordinates describing an already-open trade."""

    management_state: str = "off"
    include_unrealized_r: bool = False
    include_peak_favorable_r: bool = False
    include_giveback_r: bool = False
    include_hold_fraction: bool = False
    include_ratchet_active: bool = False
    include_protected_r: bool = False
    r_scale: float = 10.0
    hold_horizon_bars: int = 120

    def __post_init__(self) -> None:
        if self.management_state not in {"off", "entry_risk_v1"}:
            raise ValueError("unsupported trade-management observation state")
        flags = (
            self.include_unrealized_r,
            self.include_peak_favorable_r,
            self.include_giveback_r,
            self.include_hold_fraction,
            self.include_ratchet_active,
            self.include_protected_r,
        )
        if not all(isinstance(flag, bool) for flag in flags):
            raise ValueError("trade-management field flags must be boolean")
        if self.management_state == "off" and any(flags):
            raise ValueError("disabled trade-management state cannot expose fields")
        if self.management_state == "entry_risk_v1" and not all(flags):
            raise ValueError("entry-risk trade-management state is incomplete")
        if (
            isinstance(self.r_scale, bool)
            or not isinstance(self.r_scale, (int, float))
            or self.r_scale <= 0
            or isinstance(self.hold_horizon_bars, bool)
            or not isinstance(self.hold_horizon_bars, int)
            or self.hold_horizon_bars < 1
        ):
            raise ValueError("trade-management normalization is invalid")

    @classmethod
    def entry_risk_v1(
        cls,
        *,
        r_scale: float,
        hold_horizon_bars: int,
    ) -> "TradeManagementObservationSpec":
        return cls(
            management_state="entry_risk_v1",
            include_unrealized_r=True,
            include_peak_favorable_r=True,
            include_giveback_r=True,
            include_hold_fraction=True,
            include_ratchet_active=True,
            include_protected_r=True,
            r_scale=float(r_scale),
            hold_horizon_bars=int(hold_horizon_bars),
        )

    @classmethod
    def from_config(cls, config: dict | None) -> "TradeManagementObservationSpec":
        return cls() if config is None else cls(**config)

    @property
    def output_dim(self) -> int:
        return 0 if self.management_state == "off" else 6


@dataclass(frozen=True)
class AccountState:
    realized_pnl: float = 0.0
    equity_pnl: float = 0.0
    peak_equity_pnl: float = 0.0
    position_side: PositionSide = PositionSide.FLAT
    position_size: int = 0
    max_position_size: int = 1
    unrealized_pnl: float = 0.0
    session_remaining: float = 1.0
    challenge_remaining: float = 1.0
    point_value: float = 0.0
    round_trip_fee: float = 0.0
    mll_headroom: float | None = None
    current_r: float = 0.0
    peak_favorable_r: float = 0.0
    giveback_r: float = 0.0
    hold_bars: int = 0
    ratchet_active: bool = False
    protected_r: float = 0.0


class ObservationAssembler:
    """Join an unchanged frozen embedding with coordinate-invariant state."""

    ACCOUNT_DIM = 12

    def __init__(
        self,
        embedding_dim: int,
        *,
        max_loss: float,
        profit_target: float,
        trade_management: TradeManagementObservationSpec | None = None,
    ) -> None:
        if embedding_dim < 1 or max_loss <= 0 or profit_target <= 0:
            raise ValueError("observation dimensions and economics must be positive")
        self.embedding_dim = int(embedding_dim)
        self.max_loss = float(max_loss)
        self.profit_target = float(profit_target)
        self.trade_management = trade_management or TradeManagementObservationSpec()

    @property
    def output_dim(self) -> int:
        return (
            self.embedding_dim
            + self.ACCOUNT_DIM
            + self.trade_management.output_dim
        )

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
                (
                    account.mll_headroom
                    if account.mll_headroom is not None
                    else account.equity_pnl + self.max_loss
                ) / self.max_loss,
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
        management_values = np.empty(0, dtype=np.float32)
        if self.trade_management.management_state == "entry_risk_v1":
            r_scale = self.trade_management.r_scale
            management_values = np.asarray(
                [
                    np.clip(account.current_r / r_scale, -1.0, 1.0),
                    np.clip(account.peak_favorable_r / r_scale, 0.0, 1.0),
                    np.clip(account.giveback_r / r_scale, 0.0, 1.0),
                    np.clip(
                        account.hold_bars
                        / self.trade_management.hold_horizon_bars,
                        0.0,
                        1.0,
                    ),
                    float(account.ratchet_active),
                    np.clip(account.protected_r / r_scale, -1.0, 1.0),
                ],
                dtype=np.float32,
            )
            if not np.isfinite(management_values).all():
                raise ValueError("trade-management state must be finite")
        return np.concatenate((embedding, account_values, management_values))


__all__ = [
    "AccountState",
    "ObservationAssembler",
    "TradeManagementObservationSpec",
]
