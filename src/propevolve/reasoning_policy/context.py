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
    input_mode: str = "specialists"
    text_steps: int | None = None

    def __post_init__(self):
        if self.input_mode not in {"specialists", "embeddings"}:
            raise ValueError("unknown reasoning input mode")
        if self.input_mode == "embeddings" and any(
                not name.startswith(("account.", "trade.", "challenge.")) for name in self.fields):
            raise ValueError("teacher fields cannot be inputs in embedding mode")
        if type(self.context_steps) is not int or self.context_steps < 1:
            raise ValueError("context_steps must be a positive integer")
        text_steps = self.context_steps if self.text_steps is None else self.text_steps
        if type(text_steps) is not int or not 1 <= text_steps <= self.context_steps:
            raise ValueError("text_steps must fit inside context_steps")
        object.__setattr__(self, "text_steps", text_steps)
        if (not self.fields or len(set(self.fields)) != len(self.fields)
                or any(not isinstance(x, str) or not x for x in self.fields)):
            raise ValueError("fields must be unique nonempty names")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text())
        if set(payload) - {"context_steps", "fields", "input_mode", "text_steps"}:
            raise ValueError("unexpected context config fields")
        return cls(payload["context_steps"], tuple(payload["fields"]),
                   payload.get("input_mode", "specialists"), payload.get("text_steps"))


@dataclass(frozen=True)
class ContextWindow:
    values: np.ndarray
    available: np.ndarray
    timestamps: tuple[int, ...]
    fields: tuple[str, ...]
    embeddings: np.ndarray | None = None
    text_steps: int | None = None


class RollingContext:
    def __init__(self, config: ContextConfig):
        self.config = config
        self._rows = deque(maxlen=config.context_steps)

    def reset(self):
        self._rows.clear()

    def append(self, completed_at_ns: int, fields: Mapping[str, float], *, embedding=None):
        if type(completed_at_ns) is not int:
            raise ValueError("timestamp must be integer completed-bar nanoseconds")
        if self._rows and completed_at_ns <= self._rows[-1][0]:
            raise ValueError("completed timestamps must be strictly increasing")
        if set(fields) != set(self.config.fields):
            raise ValueError("observation fields differ from configured schema")
        values = np.asarray([fields[key] for key in self.config.fields], dtype=np.float32)
        if values.shape != (len(self.config.fields),) or not np.isfinite(values).all():
            raise ValueError("observation values must be finite scalars")
        if self.config.input_mode == "embeddings":
            embedding = np.array(embedding, dtype=np.float32, copy=True)
            if (embedding.ndim != 1 or not embedding.size or not np.isfinite(embedding).all()
                    or (self._rows and embedding.shape != self._rows[-1][2].shape)):
                raise ValueError("invalid causal embedding shape or values")
        elif embedding is not None:
            raise ValueError("unexpected embedding in specialist context")
        self._rows.append((completed_at_ns, values, embedding))

    def snapshot(self) -> ContextWindow:
        values = np.zeros((self.config.context_steps, len(self.config.fields)), np.float32)
        available = np.zeros(self.config.context_steps, dtype=bool)
        if self._rows:
            values[-len(self._rows):] = np.stack([item[1] for item in self._rows])
            available[-len(self._rows):] = True
        values.setflags(write=False)
        available.setflags(write=False)
        embeddings = None
        if self._rows and self.config.input_mode == "embeddings":
            embeddings = np.zeros((self.config.context_steps, len(self._rows[-1][2])), np.float32)
            embeddings[-len(self._rows):] = np.stack([item[2] for item in self._rows])
            embeddings.setflags(write=False)
        return ContextWindow(values, available, tuple(item[0] for item in self._rows),
                             self.config.fields, embeddings, self.config.text_steps)
