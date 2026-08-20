"""State-dependent trading actions and deterministic risk masking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observation import AccountState


class PositionSide(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


class Action(IntEnum):
    WAIT = 0
    ENTER_LONG_1 = 1
    ENTER_SHORT_1 = 2
    HOLD = 3
    CLOSE = 4


@dataclass(frozen=True)
class RecoveryEntryPermit:
    """One-shot exception to the ordinary flat-account entry guard."""

    remaining_entries: int
    exception_headroom: float
    ordinary_entry_resume_pnl: float

    def __post_init__(self) -> None:
        if self.remaining_entries not in (0, 1):
            raise ValueError("recovery permit remaining_entries must be 0 or 1")
        if (
            not math.isfinite(self.exception_headroom)
            or self.exception_headroom <= 0
            or not math.isfinite(self.ordinary_entry_resume_pnl)
        ):
            raise ValueError("recovery permit economics must be finite and valid")
    def permits(self, mll_headroom: float) -> bool:
        return (
            self.remaining_entries == 1
            and math.isclose(mll_headroom, self.exception_headroom)
        )


class ActionMasker:
    """Expose only actions valid for the current position and hard risk state."""

    def __init__(
        self,
        *,
        max_position_size: int,
        max_loss: float,
        minimum_mll_headroom: float,
    ) -> None:
        if max_position_size < 1:
            raise ValueError("max_position_size must be positive")
        self.max_position_size = int(max_position_size)
        self.max_loss = float(max_loss)
        self.minimum_mll_headroom = float(minimum_mll_headroom)

    def valid_actions(
        self,
        account: "AccountState",
        *,
        recovery_entry_permit: RecoveryEntryPermit | None = None,
    ) -> tuple[Action, ...]:
        if account.position_side == PositionSide.FLAT:
            actions = [Action.WAIT]
            headroom = (
                account.mll_headroom
                if account.mll_headroom is not None
                else account.equity_pnl + self.max_loss
            )
            if (
                headroom >= self.minimum_mll_headroom
                or (
                    recovery_entry_permit is not None
                    and recovery_entry_permit.permits(headroom)
                )
            ):
                actions.extend((Action.ENTER_LONG_1, Action.ENTER_SHORT_1))
            return tuple(sorted(actions, key=int))

        return (Action.HOLD, Action.CLOSE)

    @staticmethod
    def boolean_mask(actions: tuple[Action, ...]) -> tuple[bool, ...]:
        allowed = set(actions)
        return tuple(action in allowed for action in Action)
