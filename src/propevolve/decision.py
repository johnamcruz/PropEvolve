"""State-dependent trading actions and deterministic risk masking."""

from __future__ import annotations

from enum import IntEnum
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
        recovery_active: bool = False,
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
                or (recovery_active and headroom > 0.0)
            ):
                actions.extend((Action.ENTER_LONG_1, Action.ENTER_SHORT_1))
            return tuple(sorted(actions, key=int))

        return (Action.HOLD, Action.CLOSE)

    @staticmethod
    def boolean_mask(actions: tuple[Action, ...]) -> tuple[bool, ...]:
        allowed = set(actions)
        return tuple(action in allowed for action in Action)
