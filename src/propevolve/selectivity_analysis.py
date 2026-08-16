"""Read-only Expansion and Regime selectivity analysis for one attempt.

This module consumes trusted local PropEvolve run artifacts.  It never trains,
mutates a campaign, or treats the balanced final probe as natural-frequency
economic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np

from .config import load_experiment_config
from .decision import Action
from .replay import FinalRegimeProbeSequence, final_regime_probe_row_identity


SCHEMA = "propevolve_selectivity_analysis_v1"
_PROBE_SCHEMA = "propevolve_final_regime_probe_v1"
_PROBE_CORPUS_SCHEMA = "propevolve_training_policy_health_probe_corpus_v1"
_SUMMARY_SCHEMA = "propevolve_training_diagnostic_summary_v1"
_REGIME_NAMES = (
    "chop_no_trend",
    "chop_end_transition",
    "expansion_trend",
)
_ENTRY_ACTIONS = (Action.ENTER_LONG_1, Action.ENTER_SHORT_1)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _finite_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} is not a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} is not a probability")
    return result


def _nonnegative_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} is not a nonnegative count")
    return int(value)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _cohort(rows: Sequence[Mapping[str, object]]) -> dict[str, int | float]:
    opportunities = sum(bool(row["opportunity"]) for row in rows)
    predicted_entries = sum(bool(row["predicted_entry"]) for row in rows)
    correct_entries = sum(bool(row["correct_entry"]) for row in rows)
    failed_setups = sum(bool(row["failed_setup"]) for row in rows)
    rejected = sum(bool(row["failed_setup_rejected"]) for row in rows)
    return {
        "rows": len(rows),
        "opportunities": opportunities,
        "predicted_entries": predicted_entries,
        "correct_entries": correct_entries,
        "entry_precision": _ratio(correct_entries, predicted_entries),
        "opportunity_recall": _ratio(correct_entries, opportunities),
        "failed_setups": failed_setups,
        "failed_setup_rejections": rejected,
        "failed_setup_rejection_rate": _ratio(rejected, failed_setups),
    }


def _grouped(
    rows: Sequence[Mapping[str, object]], key: str
) -> dict[str, dict[str, int | float]]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: _cohort([row for row in rows if str(row[key]) == value])
        for value in values
    }


def _confluence(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int | float]]:
    keys = sorted({
        "|".join((
            str(row["regime_state"]),
            str(row["expansion_quantile"]),
            str(row["side"]),
        ))
        for row in rows
    })
    return {
        key: _cohort([
            row
            for row in rows
            if "|".join((
                str(row["regime_state"]),
                str(row["expansion_quantile"]),
                str(row["side"]),
            ))
            == key
        ])
        for key in keys
    }


def _load_probe_samples(path: Path) -> tuple[FinalRegimeProbeSequence, ...]:
    # This is intentionally limited to trusted, locally generated PropEvolve
    # attempt artifacts: pickle is not a safe interchange format.
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _PROBE_CORPUS_SCHEMA
        or not isinstance(payload.get("samples"), (tuple, list))
    ):
        raise ValueError("training policy-health probe corpus is invalid")
    samples = tuple(payload["samples"])
    if not samples or any(
        not isinstance(sample, FinalRegimeProbeSequence) for sample in samples
    ):
        raise ValueError("training policy-health probe samples are invalid")
    identities = [sample.row_identity_sha256 for sample in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("training policy-health probe identities are duplicated")
    return samples


def _joined_probe_rows(
    probe: Mapping[str, object],
    samples: Sequence[FinalRegimeProbeSequence],
) -> list[dict[str, object]]:
    if probe.get("schema") != _PROBE_SCHEMA or not isinstance(
        probe.get("rows"), list
    ):
        raise ValueError("final Regime probe is invalid")
    probe_rows = probe["rows"]
    if len(probe_rows) != len(samples):
        raise ValueError("final Regime probe row count drifted")
    by_identity = {sample.row_identity_sha256: sample for sample in samples}
    row_identities = [
        row.get("row_identity_sha256")
        for row in probe_rows
        if isinstance(row, dict)
    ]
    if len(row_identities) != len(probe_rows) or len(set(row_identities)) != len(
        row_identities
    ):
        raise ValueError("final Regime probe row identities are invalid")
    if set(row_identities) != set(by_identity):
        raise ValueError("final Regime probe rows do not match the probe corpus")

    joined: list[dict[str, object]] = []
    for probe_row in probe_rows:
        identity = str(probe_row["row_identity_sha256"])
        sample = by_identity[identity]
        anchor_index = int(sample.sequence_anchor_index)
        if not 0 <= anchor_index < len(sample.sequence):
            raise ValueError("final Regime probe anchor is invalid")
        anchor = sample.sequence[anchor_index]
        teacher = np.asarray(anchor.teacher_target, dtype=np.float32).reshape(-1)
        if teacher.shape != (7,) or not np.isfinite(teacher).all():
            raise ValueError("final Regime probe teacher row must have 7 channels")
        probabilities = [
            _finite_probability(value, name=f"teacher channel {index}")
            for index, value in enumerate(teacher)
        ]
        if not math.isclose(sum(probabilities[4:]), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("Regime teacher probabilities do not sum to one")
        if anchor.entry_action_target != sample.target_action:
            raise ValueError("final Regime probe Entry target drifted")
        if final_regime_probe_row_identity(
            ticker=sample.ticker,
            source_decision_index=sample.source_decision_index,
            target_action=sample.target_action,
            observation=anchor.observation,
            teacher_target=teacher,
        ) != identity:
            raise ValueError("final Regime probe source identity drifted")
        if (
            probe_row.get("target_action") != sample.target_action.name
            or probe_row.get("ticker") != sample.ticker
            or int(probe_row.get("source_decision_index", -1))
            != sample.source_decision_index
        ):
            raise ValueError("final Regime probe published row drifted")
        try:
            prediction = Action[str(probe_row["greedy_action"])]
        except (KeyError, TypeError) as error:
            raise ValueError("final Regime probe greedy action is invalid") from error
        if prediction not in (Action.WAIT, *_ENTRY_ACTIONS):
            raise ValueError("final Regime probe greedy action is not flat-state")

        long_score = probabilities[0] * probabilities[1]
        short_score = probabilities[2] * probabilities[3]
        side = "long" if long_score >= short_score else "short"
        target = sample.target_action
        joined.append({
            "row_identity_sha256": identity,
            "regime_state": _REGIME_NAMES[int(np.argmax(probabilities[4:]))],
            "expansion_score": max(long_score, short_score),
            "side": side,
            "opportunity": target in _ENTRY_ACTIONS,
            "predicted_entry": prediction in _ENTRY_ACTIONS,
            "correct_entry": target in _ENTRY_ACTIONS and prediction == target,
            "failed_setup": target == Action.WAIT,
            "failed_setup_rejected": (
                target == Action.WAIT and prediction == Action.WAIT
            ),
        })

    ordered = sorted(
        range(len(joined)),
        key=lambda index: (
            float(joined[index]["expansion_score"]),
            str(joined[index]["row_identity_sha256"]),
        ),
    )
    for rank, row_index in enumerate(ordered):
        joined[row_index]["expansion_quantile"] = (
            f"Q{min(4, rank * 4 // len(joined) + 1)}"
        )
    return joined


def _training_entry_supervision(overall: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for action in ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1"):
        targets = _nonnegative_count(
            dict(overall.get("sampled_entry_action_target_counts", {})).get(action, 0),
            name=f"{action} target count",
        )
        predictions = _nonnegative_count(
            dict(overall.get(
                "sampled_entry_action_prediction_counts", {}
            )).get(action, 0),
            name=f"{action} prediction count",
        )
        correct = _nonnegative_count(
            dict(overall.get("sampled_entry_action_correct_counts", {})).get(action, 0),
            name=f"{action} correct count",
        )
        if correct > min(targets, predictions):
            raise ValueError(f"{action} correct count exceeds its marginals")
        result[action] = {
            "targets": targets,
            "predictions": predictions,
            "correct": correct,
            "precision": _ratio(correct, predictions),
            "recall": _ratio(correct, targets),
        }
    return result


def _training_economics(overall: Mapping[str, object]) -> dict[str, object]:
    episodes = _nonnegative_count(overall.get("episodes", 0), name="episodes")
    passes = _nonnegative_count(overall.get("passes", 0), name="passes")
    blows = _nonnegative_count(overall.get("blows", 0), name="blows")
    timeouts = _nonnegative_count(overall.get("timeouts", 0), name="timeouts")
    near = _nonnegative_count(
        overall.get("near_blow_timeout_count", 0), name="near-blow timeouts"
    )
    if passes + blows + timeouts != episodes or near > timeouts:
        raise ValueError("training episode outcome counts are inconsistent")
    shadows: dict[str, object] = {}
    for horizon in (5, 10, 20, 50):
        value = overall.get(f"shadow_h{horizon}")
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"shadow_h{horizon} is invalid")
        complete = _nonnegative_count(
            value.get("complete_trades", 0), name=f"shadow_h{horizon} trades"
        )
        shadows[f"h{horizon}"] = {
            "complete_trades": complete,
            "hit_2r_before_1r_rate": _finite_probability(
                value.get("hit_2r_before_1r_rate", 0.0),
                name=f"shadow_h{horizon} 2R rate",
            ),
            "hit_3r_before_1r_rate": _finite_probability(
                value.get("hit_3r_before_1r_rate", 0.0),
                name=f"shadow_h{horizon} 3R rate",
            ),
            "average_mfe_r": float(value.get("average_mfe_r", 0.0)),
            "average_mae_r": float(value.get("average_mae_r", 0.0)),
        }
        if not all(
            math.isfinite(float(item))
            for item in shadows[f"h{horizon}"].values()
        ):
            raise ValueError(f"shadow_h{horizon} contains non-finite metrics")
    return {
        "episodes": episodes,
        "passes": passes,
        "pass_rate": _ratio(passes, episodes),
        "blows": blows,
        "blow_rate": _ratio(blows, episodes),
        "timeouts": timeouts,
        "near_blow_timeout_count": near,
        "near_blow_timeout_rate": _ratio(near, timeouts),
        "shadow_outcomes": shadows,
    }


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _bound_candidate_contract(attempt: Path, config_sha256: str) -> Path:
    for parent in (attempt.parent, *attempt.parents):
        candidates = parent / "archive" / "candidates"
        if not candidates.is_dir():
            continue
        matches = []
        for path in sorted(candidates.glob("*/contract.json")):
            contract = _json(path)
            if contract.get("experiment_config_sha256") == config_sha256:
                matches.append(path.resolve(strict=True))
        if len(matches) != 1:
            raise ValueError(
                "attempt must bind exactly one candidate contract to the config"
            )
        return matches[0]
    raise ValueError("attempt has no authenticated candidate contract")


def _training_regime_trade_economics(
    overall: Mapping[str, object],
) -> dict[str, object]:
    raw = overall.get("regime_trade_economics")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("groups"), list):
        raise ValueError("training Regime trade economics are invalid")
    total = _nonnegative_count(raw.get("total_trades", 0), name="total trades")
    attributed = _nonnegative_count(
        raw.get("attributed_trades", 0), name="attributed trades"
    )
    unattributed = _nonnegative_count(
        raw.get("unattributed_trades", 0), name="unattributed trades"
    )
    groups = raw["groups"]
    if attributed + unattributed != total or any(
        not isinstance(group, Mapping) for group in groups
    ):
        raise ValueError("training Regime trade economics are inconsistent")
    grouped_trades = sum(
        _nonnegative_count(group.get("trades", 0), name="group trades")
        for group in groups
    )
    if grouped_trades != attributed:
        raise ValueError("training Regime trade groups drifted")
    return json.loads(json.dumps(raw, sort_keys=True))


def analyze_selectivity_attempt(
    attempt_dir: str | Path,
    *,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Analyze one completed attempt without changing its evidence."""
    attempt = Path(attempt_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    output = Path(output_path).resolve(strict=False)
    if not attempt.is_dir() or not config_file.is_file():
        raise ValueError("selectivity analysis inputs are invalid")
    if output == config_file or output.is_relative_to(attempt):
        raise ValueError("selectivity output must not overwrite input evidence")
    config = load_experiment_config(config_file)
    config_sha256 = _sha256(config_file)
    candidate_contract = _bound_candidate_contract(attempt, config_sha256)
    if output == candidate_contract:
        raise ValueError("selectivity output must not overwrite input evidence")
    if output.exists():
        raise FileExistsError(f"selectivity output already exists: {output}")
    probe_path = attempt / "final-regime-probe.json"
    corpus_path = attempt / "training-policy-health-probe.pkl"
    summary_path = attempt / "training-diagnostic-summary.json"
    for path in (probe_path, corpus_path, summary_path):
        if not path.is_file():
            raise ValueError(f"required attempt evidence is missing: {path.name}")

    probe = _json(probe_path)
    summary = _json(summary_path)
    if summary.get("schema") != _SUMMARY_SCHEMA or not isinstance(
        summary.get("overall"), dict
    ):
        raise ValueError("training diagnostic summary is invalid")
    entry = config.get("entry_supervision")
    if not isinstance(entry, dict):
        raise ValueError("trial config lacks Entry supervision")
    entry_target = {
        "target_r": _positive_number(entry.get("target_r"), name="target_r"),
        "stop_r": _positive_number(entry.get("stop_r"), name="stop_r"),
    }
    samples = _load_probe_samples(corpus_path)
    rows = _joined_probe_rows(probe, samples)
    overall = summary["overall"]

    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PARTIAL",
        "probe_population": "balanced_final_regime_probe",
        "entry_target": entry_target,
        "overall_probe_selectivity": _cohort(rows),
        "by_regime_state": _grouped(rows, "regime_state"),
        "by_expansion_quantile": _grouped(rows, "expansion_quantile"),
        "by_side": _grouped(rows, "side"),
        "by_regime_expansion_side": _confluence(rows),
        "training_entry_supervision": _training_entry_supervision(overall),
        "training_economics": _training_economics(overall),
        "training_regime_trade_economics": (
            _training_regime_trade_economics(overall)
        ),
        "coverage": {
            "probe_rows_setup_joined": True,
            "probe_is_natural_frequency": False,
            "three_r_is_setup_joined": False,
            "three_r_note": (
                "3R outcomes are aggregate actual-trade shadow diagnostics; "
                "they are not joined to individual balanced probe setups."
            ),
        },
        "inputs": {
            "attempt_dir": str(attempt),
            "config_path": str(config_file),
            "config_sha256": config_sha256,
            "candidate_contract_path": str(candidate_contract),
            "candidate_contract_sha256": _sha256(candidate_contract),
            "final_regime_probe_sha256": _sha256(probe_path),
            "training_policy_health_probe_sha256": _sha256(corpus_path),
            "training_diagnostic_summary_sha256": _sha256(summary_path),
        },
    }
    report["identity_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Expansion and Regime confluence from one trusted local "
            "PropEvolve attempt."
        )
    )
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = analyze_selectivity_attempt(
        arguments.attempt_dir,
        config_path=arguments.config,
        output_path=arguments.output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "analyze_selectivity_attempt", "main"]
