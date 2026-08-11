"""Common interface for authenticated, training-only soft teachers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch


@dataclass(frozen=True)
class BaseTeacher(ABC):
    """Deep seam shared by every disposable distillation teacher.

    Subclasses own artifact-specific authentication and model construction. This
    base owns the causal trajectory contract and finite probability output.
    """

    device: torch.device
    manifest: dict
    checkpoint_sha256: str
    context_length: int
    embedding_dim: int

    kind: ClassVar[str]
    channels: ClassVar[tuple[str, ...]]

    @classmethod
    @abstractmethod
    def load(cls, manifest_path: str | Path, *, device: str) -> "BaseTeacher":
        """Authenticate and load one portable teacher artifact."""

    @abstractmethod
    def _score_trusted(
        self,
        trajectory: torch.Tensor,
        *,
        ticker: str | None = None,
    ) -> torch.Tensor:
        """Score an already validated causal batch on the resolved device."""

    def score(
        self,
        trajectory: np.ndarray,
        *,
        ticker: str | None = None,
    ) -> np.ndarray:
        values = np.asarray(trajectory)
        expected = (self.context_length, self.embedding_dim)
        if (
            values.ndim != 3
            or tuple(values.shape[1:]) != expected
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"{self.kind.title()} teacher trajectories must be finite "
                f"[rows,{expected[0]},{expected[1]}]"
            )
        with torch.inference_mode():
            result = self._score_trusted(
                torch.from_numpy(np.ascontiguousarray(values)), ticker=ticker
            )
        probabilities = result.detach().cpu().numpy()
        if (
            probabilities.shape != (len(values), len(self.channels))
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
        ):
            raise RuntimeError(f"{self.kind.title()} teacher output contract drifted")
        return probabilities


__all__ = ["BaseTeacher"]
