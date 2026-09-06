"""Named, completed-bar context shared by dataset building and inference."""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ContextConfig:
    context_steps: int
    fields: tuple[str, ...]

    def __post_init__(self):
        if type(self.context_steps) is not int or self.context_steps < 1:
            raise ValueError("context_steps must be a positive integer")
        if (not self.fields or len(set(self.fields)) != len(self.fields)
                or any(not isinstance(x, str) or not x for x in self.fields)):
            raise ValueError("fields must be unique nonempty names")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text())
        if set(payload) != {"context_steps", "fields"}:
            raise ValueError("unexpected context config fields")
        return cls(payload["context_steps"], tuple(payload["fields"]))


@dataclass(frozen=True)
class ContextWindow:
    values: np.ndarray
    available: np.ndarray
    timestamps: tuple[int, ...]
    fields: tuple[str, ...]


class RollingContext:
    def __init__(self, config: ContextConfig):
        self.config = config
        self._rows = deque(maxlen=config.context_steps)

    def reset(self):
        self._rows.clear()

    def append(self, completed_at_ns: int, fields: Mapping[str, float]):
        if type(completed_at_ns) is not int:
            raise ValueError("timestamp must be integer completed-bar nanoseconds")
        if self._rows and completed_at_ns <= self._rows[-1][0]:
            raise ValueError("completed timestamps must be strictly increasing")
        if set(fields) != set(self.config.fields):
            raise ValueError("observation fields differ from configured schema")
        values = np.asarray([fields[key] for key in self.config.fields], dtype=np.float32)
        if values.shape != (len(self.config.fields),) or not np.isfinite(values).all():
            raise ValueError("observation values must be finite scalars")
        self._rows.append((completed_at_ns, values))

    def snapshot(self) -> ContextWindow:
        values = np.zeros((self.config.context_steps, len(self.config.fields)), np.float32)
        available = np.zeros(self.config.context_steps, dtype=bool)
        if self._rows:
            values[-len(self._rows):] = np.stack([row for _, row in self._rows])
            available[-len(self._rows):] = True
        values.setflags(write=False)
        available.setflags(write=False)
        return ContextWindow(values, available, tuple(t for t, _ in self._rows), self.config.fields)
