"""Composition seam for training-only soft teachers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class _Targets(Protocol):
    def target(self, ticker: str, row: int) -> np.ndarray | None: ...


@dataclass(frozen=True)
class TeacherTargetSource:
    kind: str
    channels: tuple[str, ...]
    targets: _Targets
    loss_weight: float
    entry_search_loss_weight: float

    def __post_init__(self) -> None:
        if (
            self.kind not in {"expansion", "regime", "trend"}
            or not self.channels
            or len(set(self.channels)) != len(self.channels)
            or self.loss_weight < 0
            or (self.loss_weight == 0 and self.kind != "trend")
            or self.entry_search_loss_weight < 0
            or (self.kind != "expansion" and self.entry_search_loss_weight != 0)
        ):
            raise ValueError("training-only teacher source is invalid")


@dataclass(frozen=True)
class CombinedTeacherTargets:
    sources: tuple[TeacherTargetSource, ...]

    def __post_init__(self) -> None:
        if (
            not self.sources
            or len({source.kind for source in self.sources}) != len(self.sources)
            or (self.entry_search_loss_weight > 0 and self.sources[0].kind != "expansion")
        ):
            raise ValueError("combined teacher ordering or identity is invalid")

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(channel for source in self.sources for channel in source.channels)

    @property
    def channel_loss_weights(self) -> tuple[float, ...]:
        return tuple(
            source.loss_weight / len(source.channels)
            for source in self.sources
            for _ in source.channels
        )

    @property
    def entry_search_loss_weight(self) -> float:
        return sum(source.entry_search_loss_weight for source in self.sources)

    def target(self, ticker: str, row: int) -> np.ndarray | None:
        values = [source.targets.target(ticker, row) for source in self.sources]
        if any(value is None for value in values):
            return None
        return np.concatenate(values).astype(np.float32, copy=False)


def agent_teacher_settings(specs: tuple[dict, ...]) -> dict[str, object]:
    """Project validated JSON teacher recipes onto agent constructor settings."""
    if not specs:
        return {}
    settings: dict[str, object] = {
        "teacher_channels": sum(len(item["channels"]) for item in specs),
        "teacher_channel_names": tuple(
            str(channel)
            for item in specs
            for channel in item["channels"]
        ),
        "teacher_loss_weight": sum(float(item["loss_weight"]) for item in specs),
        "teacher_channel_loss_weights": tuple(
            float(item["loss_weight"]) / len(item["channels"])
            for item in specs
            for _ in item["channels"]
        ),
    }
    expansion = next(
        (item for item in specs if item["kind"] == "expansion"), None
    )
    if expansion is None:
        settings["teacher_entry_search_loss_weight"] = 0.0
        return settings
    settings.update({
        "teacher_entry_search_loss_weight": float(
            expansion["entry_search_loss_weight"]
        ),
        "teacher_entry_search_objective": str(
            expansion["entry_search_objective"]
        ),
        "teacher_entry_search_centers": (
            float(expansion["entry_search_long_center"]),
            float(expansion["entry_search_short_center"]),
        ),
        "teacher_entry_search_probability_epsilon": float(
            expansion["entry_search_probability_epsilon"]
        ),
        "teacher_entry_search_teacher_temperature": float(
            expansion["entry_search_teacher_temperature"]
        ),
        "teacher_entry_search_q_temperature": float(
            expansion["entry_search_q_temperature"]
        ),
    })
    return settings


def load_teacher_targets(
    specs: tuple[dict, ...],
    *,
    root: Path,
    markets: dict[str, object],
) -> CombinedTeacherTargets:
    sources = []
    for spec in specs:
        kind = str(spec["kind"])
        if kind == "expansion":
            from .expansion import (
                ExpansionTeacherTargets,
                verify_expansion_entry_center_receipt,
            )

            if spec["entry_search_objective"] == "centered_log_odds":
                verify_expansion_entry_center_receipt(
                    spec,
                    root=root,
                    expected_tickers=tuple(markets),
                )

            targets = ExpansionTeacherTargets.load(root / spec["cache_root"], markets)
        elif kind == "regime":
            from .regime import RegimeTeacherTargets

            targets = RegimeTeacherTargets.load(root / spec["cache_root"], markets)
        elif kind == "trend":
            from .trend import TrendTeacherTargets

            targets = TrendTeacherTargets.load(root / spec["cache_root"], markets)
        else:  # pragma: no cover - configuration validation owns this boundary
            raise ValueError(f"unsupported teacher kind: {kind}")
        sources.append(TeacherTargetSource(
            kind=kind,
            channels=tuple(spec["channels"]),
            targets=targets,
            loss_weight=float(spec["loss_weight"]),
            entry_search_loss_weight=float(spec["entry_search_loss_weight"]),
        ))
    return CombinedTeacherTargets(tuple(sources))


__all__ = [
    "agent_teacher_settings",
    "CombinedTeacherTargets",
    "TeacherTargetSource",
    "load_teacher_targets",
]
