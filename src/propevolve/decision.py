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
    ENTER_LONG_2 = 2
    ENTER_SHORT_1 = 3
    ENTER_SHORT_2 = 4
    HOLD = 5
    ADD_1 = 6
    REDUCE_1 = 7
    CLOSE = 8


class ActionMasker:
    """Expose only actions valid for the current position and hard risk state."""

    def __init__(
        self,
        *,
        max_position_size: int = 2,
        max_loss: float = 3_000.0,
        minimum_mll_headroom: float = 250.0,
    ) -> None:
        if max_position_size < 1:
            raise ValueError("max_position_size must be positive")
        self.max_position_size = int(max_position_size)
        self.max_loss = float(max_loss)
        self.minimum_mll_headroom = float(minimum_mll_headroom)

    def valid_actions(self, account: "AccountState") -> tuple[Action, ...]:
        if account.position_side == PositionSide.FLAT:
            actions = [Action.WAIT]
            headroom = account.equity_pnl + self.max_loss
            if headroom >= self.minimum_mll_headroom:
                actions.extend((Action.ENTER_LONG_1, Action.ENTER_SHORT_1))
                if min(account.max_position_size, self.max_position_size) >= 2:
                    actions.extend((Action.ENTER_LONG_2, Action.ENTER_SHORT_2))
            return tuple(sorted(actions, key=int))

        actions = [Action.HOLD]
        if account.position_size > 1:
            actions.append(Action.REDUCE_1)
        actions.append(Action.CLOSE)
        headroom = account.equity_pnl + self.max_loss
        maximum = min(account.max_position_size, self.max_position_size)
        if account.position_size < maximum and headroom >= self.minimum_mll_headroom:
            actions.append(Action.ADD_1)
        return tuple(sorted(actions, key=int))

    @staticmethod
    def boolean_mask(actions: tuple[Action, ...]) -> tuple[bool, ...]:
        allowed = set(actions)
        return tuple(action in allowed for action in Action)

