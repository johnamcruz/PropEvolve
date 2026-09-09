"""Action answer validation derived from the unchanged environment action enum."""

from ..decision import Action


def legal_completion_names(answer: str) -> tuple[str, ...]:
    flat = (Action.WAIT.name, Action.ENTER_LONG_1.name, Action.ENTER_SHORT_1.name)
    positioned = (Action.HOLD.name, Action.CLOSE.name)
    if answer in flat:
        return flat
    if answer in positioned:
        return positioned
    raise ValueError("not an action-supervision answer")
