"""Fail-fast policy-health evidence for long episode-budgeted training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping, Protocol


_ACTIONS = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1")
_SIDE_NAMES = {
    "ENTER_LONG_1": "Long",
    "ENTER_SHORT_1": "Short",
}
_PROBE_NAMES = {
    "WAIT": "wait",
    "ENTER_LONG_1": "long",
    "ENTER_SHORT_1": "short",
}
HIERARCHICAL_ENTRY_MASS_ABS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TrainingPolicyHealthSpec:
    """Frozen evidence thresholds; no model or environment policy leaks in."""

    minimum_completed_episodes: int
    probe_interval_episodes: int
    minimum_wait_recall: float
    minimum_long_recall: float
    minimum_short_recall: float
    minimum_entry_mass_fraction: float
    maximum_entry_mass_fraction: float
    require_zero_positive_entry_soft_wait_veto: bool
    economic_futility_minimum_completed_episodes: int
    economic_futility_maximum_near_blow_timeout_rate: float
    economic_futility_maximum_mean_terminal_pnl: float
    economic_futility_maximum_expectancy_r: float
    economic_futility_minimum_failed_conditions: int
    require_positive_persistent_regime_association: bool = False
    hierarchical_entry_mass_fractions: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        integer_values = (
            self.minimum_completed_episodes,
            self.probe_interval_episodes,
            self.economic_futility_minimum_completed_episodes,
            self.economic_futility_minimum_failed_conditions,
        )
        numeric_values = (
            self.minimum_wait_recall,
            self.minimum_long_recall,
            self.minimum_short_recall,
            self.minimum_entry_mass_fraction,
            self.maximum_entry_mass_fraction,
            self.economic_futility_maximum_near_blow_timeout_rate,
            self.economic_futility_maximum_mean_terminal_pnl,
            self.economic_futility_maximum_expectancy_r,
        )
        if any(type(value) is not int or value < 1 for value in integer_values):
            raise ValueError(
                "policy-health episode and evidence counts must be positive integers"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError("policy-health thresholds must be finite")
        if not (
            0.0 <= self.minimum_wait_recall <= 1.0
            and 0.0 <= self.minimum_long_recall <= 1.0
            and 0.0 <= self.minimum_short_recall <= 1.0
            and 0.0 <= self.minimum_entry_mass_fraction
            <= self.maximum_entry_mass_fraction
            <= 1.0
            and 0.0
            <= self.economic_futility_maximum_near_blow_timeout_rate
            <= 1.0
            and self.economic_futility_minimum_failed_conditions <= 3
        ):
            raise ValueError("policy-health threshold ranges are invalid")
        if not isinstance(self.require_zero_positive_entry_soft_wait_veto, bool):
            raise TypeError("policy-health veto contract must be boolean")
        if not isinstance(
            self.require_positive_persistent_regime_association, bool
        ):
            raise TypeError("policy-health association contract must be boolean")
        if self.hierarchical_entry_mass_fractions is not None:
            expected = {
                "timing_wait",
                "timing_enter",
                "direction_long",
                "direction_short",
            }
            if (
                not isinstance(self.hierarchical_entry_mass_fractions, Mapping)
                or set(self.hierarchical_entry_mass_fractions) != expected
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) != 0.5
                    for value in self.hierarchical_entry_mass_fractions.values()
                )
            ):
                raise ValueError(
                    "hierarchical policy-health mass contract must be exact"
                )
            object.__setattr__(
                self,
                "hierarchical_entry_mass_fractions",
                MappingProxyType(dict(self.hierarchical_entry_mass_fractions)),
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> "TrainingPolicyHealthSpec":
        """Compile the already fail-closed public JSON contract."""
        recalls = config["minimum_probe_recall"]
        hierarchical_mass = config.get("hierarchical_entry_mass_fraction")
        mass = (
            {"minimum": 0.0, "maximum": 0.0}
            if hierarchical_mass is not None
            else config["entry_mass_fraction"]
        )
        economic = config["economic_futility"]
        if not all(isinstance(value, Mapping) for value in (recalls, mass, economic)):
            raise ValueError("policy-health nested contracts are invalid")
        return cls(
            minimum_completed_episodes=config["minimum_completed_episodes"],
            probe_interval_episodes=config["probe_interval_episodes"],
            minimum_wait_recall=recalls["WAIT"],
            minimum_long_recall=recalls["ENTER_LONG_1"],
            minimum_short_recall=recalls["ENTER_SHORT_1"],
            minimum_entry_mass_fraction=mass["minimum"],
            maximum_entry_mass_fraction=mass["maximum"],
            require_zero_positive_entry_soft_wait_veto=(
                config["require_zero_positive_entry_soft_wait_veto"]
            ),
            economic_futility_minimum_completed_episodes=(
                economic["minimum_completed_episodes"]
            ),
            economic_futility_maximum_near_blow_timeout_rate=(
                economic["maximum_near_blow_timeout_rate"]
            ),
            economic_futility_maximum_mean_terminal_pnl=(
                economic["maximum_mean_terminal_pnl"]
            ),
            economic_futility_maximum_expectancy_r=(
                economic["maximum_expectancy_r"]
            ),
            economic_futility_minimum_failed_conditions=(
                economic["minimum_failed_conditions"]
            ),
            require_positive_persistent_regime_association=config.get(
                "require_positive_persistent_regime_association", False
            ),
            hierarchical_entry_mass_fractions=(
                {
                    "timing_wait": hierarchical_mass["timing"]["WAIT"],
                    "timing_enter": hierarchical_mass["timing"]["ENTER"],
                    "direction_long": hierarchical_mass[
                        "conditional_direction"
                    ]["ENTER_LONG_1"],
                    "direction_short": hierarchical_mass[
                        "conditional_direction"
                    ]["ENTER_SHORT_1"],
                }
                if isinstance(hierarchical_mass, Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class TrainingHealthSnapshot:
    """One completed-episode boundary observed by the detector."""

    completed_episodes: int
    passes: int
    blows: int
    timeouts: int
    near_blow_timeouts: int
    mean_terminal_pnl: float
    expectancy_r: float
    optimizer_updates: int
    entry_action_weight_scale: float
    teacher_weight_scale: float
    entry_mass_fractions: Mapping[str, float]
    positive_entry_soft_wait_disagreement_rows: Mapping[str, int]
    probe_metrics: Mapping[str, float] | None
    hierarchical_entry_mass_fractions: Mapping[str, float] | None = None


@dataclass(frozen=True)
class TrainingHealthVerdict:
    stop: bool
    reasons: tuple[str, ...]
    evidence: Mapping[str, object]


class TrainingProgressView(Protocol):
    completed_episodes: int
    passes: int
    blows: int
    timeouts: int
    near_blow_timeout_count: int
    terminal_pnl_sum: float
    terminal_pnl_count: int
    trade_r_sum: float
    trade_count: int


class TrainingHealthDetector:
    """Evaluate integrity, action competence, and conjunctive futility."""

    def __init__(self, spec: TrainingPolicyHealthSpec) -> None:
        if not isinstance(spec, TrainingPolicyHealthSpec):
            raise TypeError("policy-health detector requires a frozen spec")
        self.spec = spec

    def evaluate(self, snapshot: TrainingHealthSnapshot) -> TrainingHealthVerdict:
        if not isinstance(snapshot, TrainingHealthSnapshot):
            raise TypeError("policy-health detector requires a typed snapshot")
        reasons: list[str] = []
        finite_fields = {
            "mean terminal PnL": snapshot.mean_terminal_pnl,
            "expectancy R": snapshot.expectancy_r,
        }
        for name, value in finite_fields.items():
            if not math.isfinite(float(value)):
                reasons.append(f"policy-health {name} is non-finite")

        if (
            not math.isfinite(float(snapshot.entry_action_weight_scale))
            or snapshot.entry_action_weight_scale < 0.0
        ):
            reasons.append(
                "entry-action objective weight scale is missing, non-finite, or negative"
            )
        elif (
            snapshot.optimizer_updates > 0
            and snapshot.entry_action_weight_scale > 0.0
        ):
            if self.spec.hierarchical_entry_mass_fractions is None:
                self._entry_objective_reasons(snapshot, reasons)
            else:
                self._hierarchical_entry_objective_reasons(snapshot, reasons)
        if (
            not math.isfinite(float(snapshot.teacher_weight_scale))
            or snapshot.teacher_weight_scale < 0.0
        ):
            reasons.append(
                "Regime objective weight scale is missing, non-finite, or negative"
            )
        elif (
            snapshot.optimizer_updates > 0
            and snapshot.teacher_weight_scale > 0.0
            and self.spec.require_zero_positive_entry_soft_wait_veto
        ):
            self._regime_veto_reasons(snapshot, reasons)

        probe_due = (
            snapshot.completed_episodes >= self.spec.minimum_completed_episodes
            and snapshot.completed_episodes % self.spec.probe_interval_episodes == 0
        )
        if probe_due:
            self._probe_reasons(snapshot.probe_metrics, reasons)

        futility_signals: list[str] = []
        near_blow_rate = (
            snapshot.near_blow_timeouts / snapshot.timeouts
            if snapshot.timeouts
            else 0.0
        )
        if (
            snapshot.completed_episodes
            >= self.spec.economic_futility_minimum_completed_episodes
        ):
            if (
                near_blow_rate
                > self.spec.economic_futility_maximum_near_blow_timeout_rate
            ):
                futility_signals.append(
                    "near-blow timeout rate "
                    f"{near_blow_rate:.6f} > "
                    f"{self.spec.economic_futility_maximum_near_blow_timeout_rate:.6f}"
                )
            if (
                snapshot.mean_terminal_pnl
                <= self.spec.economic_futility_maximum_mean_terminal_pnl
            ):
                futility_signals.append(
                    "mean terminal PnL "
                    f"{snapshot.mean_terminal_pnl:.6f} <= "
                    f"{self.spec.economic_futility_maximum_mean_terminal_pnl:.6f}"
                )
            if (
                snapshot.expectancy_r
                <= self.spec.economic_futility_maximum_expectancy_r
            ):
                futility_signals.append(
                    f"expectancy {snapshot.expectancy_r:.6f}R <= "
                    f"{self.spec.economic_futility_maximum_expectancy_r:.6f}R"
                )
            if (
                len(futility_signals)
                >= self.spec.economic_futility_minimum_failed_conditions
            ):
                reasons.append("economic futility: " + "; ".join(futility_signals))

        evidence = {
            "schema": "propevolve_training_policy_health_v1",
            "completed_episodes": snapshot.completed_episodes,
            "probe_due": probe_due,
            "near_blow_timeout_rate": near_blow_rate,
            "economic_futility_signals": tuple(futility_signals),
        }
        return TrainingHealthVerdict(bool(reasons), tuple(reasons), evidence)

    def _entry_objective_reasons(
        self,
        snapshot: TrainingHealthSnapshot,
        reasons: list[str],
    ) -> None:
        for action in _ACTIONS:
            value = snapshot.entry_mass_fractions.get(action)
            if value is None or not math.isfinite(float(value)):
                reasons.append(f"entry optimizer {action} mass fraction is missing or non-finite")
                continue
            if not (
                self.spec.minimum_entry_mass_fraction
                <= float(value)
                <= self.spec.maximum_entry_mass_fraction
            ):
                reasons.append(
                    f"entry optimizer {action} mass fraction {float(value):.6f} "
                    f"outside [{self.spec.minimum_entry_mass_fraction:.6f}, "
                    f"{self.spec.maximum_entry_mass_fraction:.6f}]"
                )

    def _hierarchical_entry_objective_reasons(
        self,
        snapshot: TrainingHealthSnapshot,
        reasons: list[str],
    ) -> None:
        actual = snapshot.hierarchical_entry_mass_fractions
        actual = actual if isinstance(actual, Mapping) else {}
        expected = self.spec.hierarchical_entry_mass_fractions or {}
        displays = {
            "timing_wait": "timing WAIT",
            "timing_enter": "timing ENTER",
            "direction_long": "conditional direction Long",
            "direction_short": "conditional direction Short",
        }
        for cohort, display in displays.items():
            value = actual.get(cohort)
            target = expected[cohort]
            if value is None or not math.isfinite(float(value)):
                reasons.append(
                    f"entry optimizer {display} mass fraction is missing or non-finite"
                )
            elif not math.isclose(
                float(value),
                float(target),
                rel_tol=0.0,
                abs_tol=HIERARCHICAL_ENTRY_MASS_ABS_TOLERANCE,
            ):
                reasons.append(
                    f"entry optimizer {display} mass fraction "
                    f"{float(value):.6f} != {float(target):.6f}"
                )

    @staticmethod
    def _regime_veto_reasons(
        snapshot: TrainingHealthSnapshot,
        reasons: list[str],
    ) -> None:
        for action in ("ENTER_LONG_1", "ENTER_SHORT_1"):
            rows = snapshot.positive_entry_soft_wait_disagreement_rows.get(action)
            if rows is None or isinstance(rows, bool) or int(rows) < 0:
                reasons.append(
                    f"persistent Regime {action} soft-WAIT veto evidence is invalid"
                )
            elif int(rows) > 0:
                reasons.append(
                    "persistent Regime objective applied a soft-WAIT veto to "
                    f"{int(rows)} {_SIDE_NAMES[action]} entry rows"
                )

    def _probe_reasons(
        self,
        metrics: Mapping[str, float] | None,
        reasons: list[str],
    ) -> None:
        if metrics is None:
            reasons.append("teacher-free policy-health probe is missing")
            return
        thresholds = {
            "WAIT": self.spec.minimum_wait_recall,
            "ENTER_LONG_1": self.spec.minimum_long_recall,
            "ENTER_SHORT_1": self.spec.minimum_short_recall,
        }
        for action in _ACTIONS:
            probe_name = _PROBE_NAMES[action]
            display = "WAIT" if action == "WAIT" else _SIDE_NAMES[action]
            rows = metrics.get(f"final_regime_probe_{probe_name}_rows")
            recall = metrics.get(f"final_regime_probe_{probe_name}_recall")
            if rows != 32.0:
                reasons.append(
                    f"teacher-free policy-health probe lacks 32 authentic {display} rows"
                )
            if recall is None or not math.isfinite(float(recall)):
                reasons.append(
                    f"teacher-free policy-health {display} recall is missing or non-finite"
                )
            elif float(recall) < thresholds[action]:
                reasons.append(
                    f"teacher-free policy-health {display} recall "
                    f"{float(recall):.6f} < {thresholds[action]:.6f}"
                )
        if self.spec.require_positive_persistent_regime_association:
            associations = {
                "persistent-dead minus transition-positive WAIT probability": (
                    "final_regime_probe_dead_wait_minus_transition_positive_wait"
                ),
                "transition-positive Long response": (
                    "final_regime_probe_transition_positive_long_response"
                ),
                "transition-positive Short response": (
                    "final_regime_probe_transition_positive_short_response"
                ),
            }
            for display, metric in associations.items():
                value = metrics.get(metric)
                if value is None or not math.isfinite(float(value)):
                    reasons.append(
                        f"teacher-free policy-health {display} is missing or non-finite"
                    )
                elif float(value) <= 0.0:
                    reasons.append(
                        f"teacher-free policy-health {display} "
                        f"{float(value):.6f} <= 0.000000"
                    )


class TrainingHealthMonitor:
    """Adapt one completed training episode to the frozen health detector."""

    def __init__(
        self,
        detector: TrainingHealthDetector,
        *,
        probe: Callable[[int], Mapping[str, float]],
        receipt_callback: Callable[[dict[str, object]], None],
    ) -> None:
        if not isinstance(detector, TrainingHealthDetector):
            raise TypeError("training health monitor requires a detector")
        if not callable(probe) or not callable(receipt_callback):
            raise TypeError("training health monitor callbacks must be callable")
        self.detector = detector
        self.probe = probe
        self.receipt_callback = receipt_callback

    def __call__(
        self,
        progress: TrainingProgressView,
        diagnostic: Mapping[str, object],
    ) -> str | None:
        completed = int(progress.completed_episodes)
        probe_due = (
            completed >= self.detector.spec.minimum_completed_episodes
            and completed % self.detector.spec.probe_interval_episodes == 0
        )
        probe_metrics: Mapping[str, float] | None = None
        probe_error: str | None = None
        if probe_due:
            try:
                probe_metrics = dict(self.probe(completed))
            except (RuntimeError, TypeError, ValueError) as error:
                probe_error = str(error)

        entry_balance = diagnostic.get("entry_action_balance") or {}
        conflict = diagnostic.get("regime_entry_conflict") or {}
        if not isinstance(entry_balance, Mapping) or not isinstance(
            conflict, Mapping
        ):
            raise ValueError("training policy-health diagnostics are malformed")
        mass = {
            action: self._nested_number(
                entry_balance,
                action_name,
                "weighted_mass_fraction",
                default=float("nan"),
            )
            for action, action_name in (
                ("WAIT", "wait"),
                ("ENTER_LONG_1", "long"),
                ("ENTER_SHORT_1", "short"),
            )
        }
        disagreements = {
            action: int(self._nested_number(
                conflict,
                side,
                "soft_wait_disagreement_rows",
                default=-1.0,
            ))
            for action, side in (
                ("ENTER_LONG_1", "long"),
                ("ENTER_SHORT_1", "short"),
            )
        }
        hierarchical_balance = diagnostic.get(
            "hierarchical_entry_balance"
        ) or {}
        hierarchical_balance = (
            hierarchical_balance
            if isinstance(hierarchical_balance, Mapping)
            else {}
        )
        hierarchical_mass = None
        if self.detector.spec.hierarchical_entry_mass_fractions is not None:
            hierarchical_mass = {}
            for output, task, cohort in (
                ("timing_wait", "timing", "wait"),
                ("timing_enter", "timing", "enter"),
                ("direction_long", "direction", "long"),
                ("direction_short", "direction", "short"),
            ):
                task_values = hierarchical_balance.get(task)
                task_values = (
                    task_values if isinstance(task_values, Mapping) else {}
                )
                hierarchical_mass[output] = self._nested_number(
                    task_values,
                    cohort,
                    "weighted_mass_fraction",
                    default=float("nan"),
                )
        snapshot = TrainingHealthSnapshot(
            completed_episodes=completed,
            passes=int(progress.passes),
            blows=int(progress.blows),
            timeouts=int(progress.timeouts),
            near_blow_timeouts=int(progress.near_blow_timeout_count),
            mean_terminal_pnl=(
                float(progress.terminal_pnl_sum)
                / int(progress.terminal_pnl_count)
                if int(progress.terminal_pnl_count) else float("nan")
            ),
            expectancy_r=(
                float(progress.trade_r_sum) / int(progress.trade_count)
                if int(progress.trade_count) else 0.0
            ),
            optimizer_updates=int(diagnostic.get("updates", 0) or 0),
            entry_action_weight_scale=self._number(
                diagnostic,
                "entry_action_weight_scale",
                default=float("nan"),
            ),
            teacher_weight_scale=self._number(
                diagnostic,
                "teacher_weight_scale",
                default=float("nan"),
            ),
            entry_mass_fractions=mass,
            positive_entry_soft_wait_disagreement_rows=disagreements,
            probe_metrics=probe_metrics,
            hierarchical_entry_mass_fractions=hierarchical_mass,
        )
        verdict = self.detector.evaluate(snapshot)
        body: dict[str, object] = {
            "schema": "propevolve_training_policy_health_receipt_v1",
            "completed_episodes": completed,
            "stop": verdict.stop,
            "reasons": list(verdict.reasons),
            "evidence": dict(verdict.evidence),
            "entry_mass_fractions": mass,
            "hierarchical_entry_mass_fractions": hierarchical_mass,
            "entry_action_weight_scale": snapshot.entry_action_weight_scale,
            "teacher_weight_scale": snapshot.teacher_weight_scale,
            "entry_objective_active": bool(
                snapshot.optimizer_updates > 0
                and math.isfinite(snapshot.entry_action_weight_scale)
                and snapshot.entry_action_weight_scale > 0.0
            ),
            "regime_objective_active": bool(
                snapshot.optimizer_updates > 0
                and math.isfinite(snapshot.teacher_weight_scale)
                and snapshot.teacher_weight_scale > 0.0
            ),
            "positive_entry_soft_wait_disagreement_rows": disagreements,
            "probe_metrics": None if probe_metrics is None else dict(probe_metrics),
            "probe_error": probe_error,
        }
        body["identity_sha256"] = hashlib.sha256(json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()).hexdigest()
        self.receipt_callback(body)
        return "; ".join(verdict.reasons) if verdict.stop else None

    @staticmethod
    def _number(
        payload: Mapping[str, object],
        field: str,
        *,
        default: float,
    ) -> float:
        value = payload.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)

    @staticmethod
    def _nested_number(
        payload: Mapping[str, object],
        group: str,
        field: str,
        *,
        default: float,
    ) -> float:
        item = payload.get(group)
        if not isinstance(item, Mapping):
            return default
        value = item.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value)
