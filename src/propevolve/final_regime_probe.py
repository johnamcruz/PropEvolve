"""Deterministic no-update probe of final shared-policy Regime behavior."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

from .balance_aware_regime_selectivity import (
    BalanceAwareRegimeSelectivity,
    EXPANSION_CHANNELS,
    PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
    REGIME_STATE_CHANNELS,
    REGIME_TRANSITION_CHANNELS,
)
from .decision import Action
from .replay import (
    FinalRegimeProbeSequence,
    final_regime_probe_row_identity,
)


SCHEMA = "propevolve_final_regime_probe_v1"
SAMPLES_PER_ACTION = 32
_FLAT_ACTIONS = (
    Action.WAIT,
    Action.ENTER_LONG_1,
    Action.ENTER_SHORT_1,
)


class GreedySequencePolicy(Protocol):
    """The only policy capability visible to the final Regime probe."""

    def greedy_sequence_action_values(
        self,
        sequences: Sequence[Sequence],
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class FinalRegimeProbeReport:
    """Auditable fixed-row result from the actual final shared policy."""

    schema: str
    source_period: tuple[str, str]
    sample_identity_sha256: str
    rows: tuple[Mapping[str, object], ...]
    metrics: Mapping[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_period": list(self.source_period),
            "sample_identity_sha256": self.sample_identity_sha256,
            "rows": [dict(row) for row in self.rows],
            "metrics": dict(self.metrics),
        }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mass = float(weights.sum())
    return float(np.dot(values, weights) / mass) if mass > 0.0 else 0.0


def evaluate_final_regime_probe(
    policy: GreedySequencePolicy,
    samples: Sequence[FinalRegimeProbeSequence],
    *,
    teacher_channel_names: Sequence[str],
    q_temperature: float,
    source_period: tuple[str, str],
) -> FinalRegimeProbeReport:
    """Measure final-policy action behavior; use teachers only after scoring.

    The policy sees only the same observations, recurrent resets, and valid
    actions used by its normal greedy action path. Exact Entry labels and
    teacher probabilities classify already-produced Q values post hoc.
    """
    channels = tuple(str(value) for value in teacher_channel_names)
    required_channels = (
        *EXPANSION_CHANNELS,
        *REGIME_STATE_CHANNELS,
        *REGIME_TRANSITION_CHANNELS,
    )
    if (
        not samples
        or len(set(channels)) != len(channels)
        or any(channel not in channels for channel in required_channels)
        or isinstance(q_temperature, bool)
        or not math.isfinite(float(q_temperature))
        or float(q_temperature) <= 0.0
        or len(source_period) != 2
        or not all(isinstance(value, str) and value for value in source_period)
        or source_period[0] >= source_period[1]
    ):
        raise ValueError("final Regime probe contract is invalid")
    burn_ins = {len(sample.sequence) for sample in samples}
    if len(burn_ins) != 1:
        raise ValueError("final Regime probe sequence lengths drifted")

    anchor_positions = []
    target_actions = []
    teacher_rows = []
    row_receipts = []
    sequences = []
    for sample in samples:
        sequence = tuple(sample.sequence)
        anchor_position = int(sample.sequence_anchor_index)
        if not 0 <= anchor_position < len(sequence):
            raise ValueError("final Regime probe anchor is not authentic")
        anchor = sequence[anchor_position]
        if (
            not anchor.training_valid
            or anchor.teacher_target is None
            or anchor.entry_action_target != sample.target_action
            or not set(_FLAT_ACTIONS).issubset(anchor.valid_actions)
        ):
            raise ValueError("final Regime probe anchor is not authentic")
        teacher = np.asarray(anchor.teacher_target, dtype=np.float32).reshape(-1)
        if teacher.shape != (len(channels),) or not np.isfinite(teacher).all():
            raise ValueError("final Regime probe teacher row is invalid")
        if final_regime_probe_row_identity(
            ticker=sample.ticker,
            source_decision_index=sample.source_decision_index,
            target_action=sample.target_action,
            observation=anchor.observation,
            teacher_target=teacher,
        ) != sample.row_identity_sha256:
            raise ValueError("final Regime probe row identity drifted")
        sequences.append(sequence)
        anchor_positions.append(anchor_position)
        target_actions.append(int(sample.target_action))
        teacher_rows.append(teacher)
        row_receipts.append({
            "row_identity_sha256": sample.row_identity_sha256,
            "ticker": sample.ticker,
            "source_anchor_index": int(sample.anchor_index),
            "source_decision_index": int(sample.source_decision_index),
            "sequence_anchor_index": anchor_position,
            "target_action": sample.target_action.name,
        })

    greedy_actions, all_q_values = policy.greedy_sequence_action_values(sequences)
    expected_shape = (len(samples), len(sequences[0]))
    if (
        np.asarray(greedy_actions).shape != expected_shape
        or np.asarray(all_q_values).shape
        != (*expected_shape, len(Action))
    ):
        raise ValueError("final Regime probe policy output shape is invalid")
    row_indices = np.arange(len(samples))
    anchors = np.asarray(anchor_positions, dtype=np.int64)
    predictions = np.asarray(greedy_actions, dtype=np.int64)[row_indices, anchors]
    q_values = np.asarray(all_q_values, dtype=np.float64)[row_indices, anchors]
    flat_q = q_values[:, [int(action) for action in _FLAT_ACTIONS]]
    if not np.isfinite(flat_q).all():
        raise ValueError("final Regime probe flat-action values are invalid")
    for row_index, receipt in enumerate(row_receipts):
        receipt["greedy_action"] = Action(int(predictions[row_index])).name
        receipt["flat_action_q_values"] = {
            action.name: float(flat_q[row_index, action_index])
            for action_index, action in enumerate(_FLAT_ACTIONS)
        }
    scaled = flat_q / float(q_temperature)
    normalized = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probabilities = normalized / normalized.sum(axis=1, keepdims=True)
    targets = np.asarray(target_actions, dtype=np.int64)
    teachers = np.stack(teacher_rows)

    metrics: dict[str, float] = {}
    for action, name in zip(_FLAT_ACTIONS, ("wait", "long", "short"), strict=True):
        rows = targets == int(action)
        count = float(rows.sum())
        metrics[f"final_regime_probe_{name}_rows"] = count
        metrics[f"final_regime_probe_{name}_recall"] = (
            float((predictions[rows] == int(action)).mean()) if count else 0.0
        )

    names = {channel: channels.index(channel) for channel in required_channels}
    positive_rows = np.isin(
        targets,
        (int(Action.ENTER_LONG_1), int(Action.ENTER_SHORT_1)),
    )
    chop = teachers[:, names["structure_chop_probability"]]
    neutral = teachers[:, names["structure_neutral_probability"]]
    trend = teachers[:, names["structure_trend_probability"]]
    dominant_chop = positive_rows & (chop > np.maximum(neutral, trend))
    nonchop = positive_rows & ~dominant_chop
    wait_probability = probabilities[:, int(Action.WAIT)]
    metrics["final_regime_probe_dominant_chop_rows"] = float(
        dominant_chop.sum()
    )
    metrics["final_regime_probe_nonchop_rows"] = float(nonchop.sum())
    metrics["final_regime_probe_chop_minus_nonchop_wait"] = (
        _weighted_mean(wait_probability, dominant_chop.astype(np.float64))
        - _weighted_mean(wait_probability, nonchop.astype(np.float64))
    )

    compiler = BalanceAwareRegimeSelectivity(
        channel_names=channels,
        expansion_centers=(0.5, 0.5),
        semantics=PERSISTENT_CHOP_NEGATIVE_WEIGHT_SEMANTICS,
        persistent_chop_negative_emphasis=1.0,
    )
    evidence = compiler.exact_wait_negative_weight_evidence(
        torch.as_tensor(teachers, dtype=torch.float32),
        torch.as_tensor(targets, dtype=torch.long),
    )
    dead = evidence.persistent_dead_chop_membership.numpy().astype(np.float64)
    ready = evidence.transition_ready_membership.numpy().astype(np.float64)
    long_ready = (
        evidence.transition_positive_long_membership.numpy().astype(np.float64)
    )
    short_ready = (
        evidence.transition_positive_short_membership.numpy().astype(np.float64)
    )
    metrics["final_regime_probe_persistent_dead_wait_mass"] = float(dead.sum())
    metrics["final_regime_probe_transition_ready_wait_mass"] = float(ready.sum())
    metrics["final_regime_probe_dead_wait_minus_transition_ready_wait"] = (
        _weighted_mean(wait_probability, dead)
        - _weighted_mean(wait_probability, ready)
    )
    for side, membership, action in (
        ("long", long_ready, Action.ENTER_LONG_1),
        ("short", short_ready, Action.ENTER_SHORT_1),
    ):
        metrics[f"final_regime_probe_transition_positive_{side}_mass"] = float(
            membership.sum()
        )
        response = probabilities[:, int(action)] - wait_probability
        metrics[f"final_regime_probe_transition_positive_{side}_response"] = (
            _weighted_mean(response, membership)
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("final Regime probe produced non-finite metrics")

    identity_payload = {
        "schema": SCHEMA,
        "source_period": list(source_period),
        "teacher_channel_names": list(channels),
        "q_temperature": float(q_temperature),
        "rows": [
            {
                key: value
                for key, value in receipt.items()
                if key not in {"greedy_action", "flat_action_q_values"}
            }
            for receipt in row_receipts
        ],
    }
    sample_identity = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return FinalRegimeProbeReport(
        schema=SCHEMA,
        source_period=tuple(source_period),
        sample_identity_sha256=sample_identity,
        rows=tuple(row_receipts),
        metrics=metrics,
    )


__all__ = [
    "FinalRegimeProbeReport",
    "SAMPLES_PER_ACTION",
    "SCHEMA",
    "evaluate_final_regime_probe",
]
