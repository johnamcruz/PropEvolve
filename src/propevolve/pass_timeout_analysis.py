"""Fast read-only comparison of Stage 2A passes and timeouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "propevolve_pass_timeout_analysis_v2"
_ACTIONS = {
    "wait": "WAIT",
    "long": "ENTER_LONG_1",
    "short": "ENTER_SHORT_1",
}
_REGIMES = ("dominant_chop", "nonchop")
_SIDES = ("long", "short")


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _count(value: object, *, name: str) -> int:
    result = _number(value, name=name)
    if result < 0.0 or int(result) != result:
        raise ValueError(f"{name} must be a nonnegative count")
    return int(result)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _load_snapshot(path: Path) -> tuple[bytes, list[Mapping[str, object]]]:
    source = path.resolve(strict=True)
    payload = source.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("training diagnostics snapshot is incomplete")
    records: list[Mapping[str, object]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"training diagnostics line {line_number} is invalid"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"training diagnostics line {line_number} is not an object"
            )
        records.append(record)
    episodes = [
        _count(record.get("episode"), name="episode") for record in records
    ]
    if not episodes or episodes != sorted(set(episodes)):
        raise ValueError("training diagnostic episodes are not strictly ordered")
    return payload, records


def _selection(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    actions: dict[str, dict[str, int | float]] = {}
    for name, key in _ACTIONS.items():
        targets = predictions = correct = 0
        for record in records:
            targets += _count(
                dict(record.get("sampled_entry_action_target_counts", {})).get(
                    key, 0
                ),
                name=f"{key} targets",
            )
            predictions += _count(
                dict(record.get(
                    "sampled_entry_action_prediction_counts", {}
                )).get(key, 0),
                name=f"{key} predictions",
            )
            correct += _count(
                dict(record.get("sampled_entry_action_correct_counts", {})).get(
                    key, 0
                ),
                name=f"{key} correct",
            )
        if correct > min(targets, predictions):
            raise ValueError(f"{key} correct count exceeds its population")
        actions[name] = {
            "targets": targets,
            "predictions": predictions,
            "correct": correct,
            "precision": _ratio(correct, predictions),
            "recall": _ratio(correct, targets),
        }

    result: dict[str, object] = {"actions": actions}
    for regime in _REGIMES:
        predicted = {name: 0 for name in _ACTIONS}
        exact_wait_total = exact_wait_correct = 0
        for record in records:
            selectivity = record.get("regime_selectivity", {})
            if not isinstance(selectivity, dict):
                raise ValueError("Regime selectivity diagnostic is invalid")
            regime_payload = selectivity.get(regime, {})
            if not isinstance(regime_payload, dict):
                raise ValueError(f"{regime} diagnostic is invalid")
            confusion = regime_payload.get("confusion", {})
            if not isinstance(confusion, dict):
                raise ValueError(f"{regime} confusion is invalid")
            for target in _ACTIONS:
                target_values = confusion.get(target, {})
                if not isinstance(target_values, dict):
                    raise ValueError(f"{regime} {target} confusion is invalid")
                for prediction in _ACTIONS:
                    value = _count(
                        target_values.get(prediction, 0),
                        name=f"{regime} {target}/{prediction}",
                    )
                    predicted[prediction] += value
                    if target == "wait":
                        exact_wait_total += value
                        if prediction == "wait":
                            exact_wait_correct += value
        rows = sum(predicted.values())
        entries = predicted["long"] + predicted["short"]
        result[regime] = {
            "rows": rows,
            "predicted_wait": predicted["wait"],
            "predicted_long": predicted["long"],
            "predicted_short": predicted["short"],
            "entry_rate": _ratio(entries, rows),
            "wait_rate": _ratio(predicted["wait"], rows),
            "exact_wait_recall": _ratio(exact_wait_correct, exact_wait_total),
        }
    result["nonchop_minus_chop_entry_rate"] = (
        result["nonchop"]["entry_rate"]
        - result["dominant_chop"]["entry_rate"]
    )
    return result


def _economic_groups(
    records: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    groups: list[Mapping[str, object]] = []
    for record in records:
        economics = record.get("regime_trade_economics", {})
        if not isinstance(economics, dict) or not isinstance(
            economics.get("groups", []), list
        ):
            raise ValueError("Regime trade economics are invalid")
        for group in economics.get("groups", []):
            if not isinstance(group, dict):
                raise ValueError("Regime trade economic group is invalid")
            groups.append(group)
    return groups


def _economic_summary(
    groups: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    trades = wins = initial_stops = 0
    realized_r_sum = mfe_r_sum = mae_r_sum = 0.0
    for group in groups:
        group_trades = _count(group.get("trades"), name="economic trades")
        group_wins = _count(group.get("wins"), name="economic wins")
        if group_wins > group_trades:
            raise ValueError("economic wins exceed trades")
        trades += group_trades
        wins += group_wins
        initial_stops += _count(
            group.get("initial_stop_count", 0), name="initial stops"
        )
        realized_r_sum += _number(
            group.get("realized_r_sum", 0.0), name="realized R"
        )
        mfe_r_sum += _number(group.get("mfe_r_sum", 0.0), name="MFE R")
        mae_r_sum += _number(group.get("mae_r_sum", 0.0), name="MAE R")
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": _ratio(wins, trades),
        "realized_r_sum": realized_r_sum,
        "realized_r_mean": _ratio(realized_r_sum, trades),
        "mfe_r_mean": _ratio(mfe_r_sum, trades),
        "mae_r_mean": _ratio(mae_r_sum, trades),
        "initial_stop_count": initial_stops,
    }


def _outcome_summary(
    records: Sequence[Mapping[str, object]], outcome: str
) -> dict[str, object]:
    selected = [record for record in records if record.get("outcome") == outcome]
    groups = _economic_groups(selected)
    count = len(selected)
    result: dict[str, object] = {
        "episodes": count,
        "mean_terminal_pnl": _ratio(sum(
            _number(record.get("terminal_pnl"), name="terminal PnL")
            for record in selected
        ), count),
        "mean_expectancy_r": _ratio(sum(
            _number(record.get("expectancy_r"), name="expectancy R")
            for record in selected
        ), count),
        "mean_trade_count": _ratio(sum(
            _count(record.get("trade_count"), name="trade count")
            for record in selected
        ), count),
        "mean_win_rate": _ratio(sum(
            _number(record.get("win_rate"), name="win rate")
            for record in selected
        ), count),
        "mean_average_win_r": _ratio(sum(
            _number(record.get("avg_win_r"), name="average win R")
            for record in selected
        ), count),
        "mean_giveback_rate": _ratio(sum(
            _number(
                record.get("gave_it_all_back_rate"), name="giveback rate"
            )
            for record in selected
        ), count),
    }
    for regime in _REGIMES:
        result[regime] = _economic_summary([
            group for group in groups if group.get("static_regime") == regime
        ])
    result["by_side"] = {
        side: _economic_summary([
            group for group in groups if group.get("side") == side
        ])
        for side in _SIDES
    }
    result["by_regime_side"] = {
        f"{regime}|{side}": _economic_summary([
            group
            for group in groups
            if group.get("static_regime") == regime
            and group.get("side") == side
        ])
        for regime in _REGIMES
        for side in _SIDES
    }
    return result


def _pass_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    for record in records:
        if record.get("outcome") != "pass":
            continue
        groups = _economic_groups((record,))
        chop = _economic_summary([
            group
            for group in groups
            if group.get("static_regime") == "dominant_chop"
        ])
        nonchop = _economic_summary([
            group for group in groups if group.get("static_regime") == "nonchop"
        ])
        total_realized = (
            float(chop["realized_r_sum"]) + float(nonchop["realized_r_sum"])
        )
        result.append({
            "episode": _count(record.get("episode"), name="episode"),
            "ticker": str(record.get("ticker")),
            "terminal_pnl": _number(
                record.get("terminal_pnl"), name="terminal PnL"
            ),
            "trades": _count(record.get("trade_count"), name="trade count"),
            "expectancy_r": _number(
                record.get("expectancy_r"), name="expectancy R"
            ),
            "average_win_r": _number(
                record.get("avg_win_r"), name="average win R"
            ),
            "giveback_rate": _number(
                record.get("gave_it_all_back_rate"), name="giveback rate"
            ),
            "dominant_chop_trades": int(chop["trades"]),
            "nonchop_trades": int(nonchop["trades"]),
            "dominant_chop_realized_r": float(chop["realized_r_sum"]),
            "nonchop_realized_r": float(nonchop["realized_r_sum"]),
            "nonchop_realized_r_share": _ratio(
                float(nonchop["realized_r_sum"]), total_realized
            ),
        })
    return result


def _risk_summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    low_headroom_by_outcome = {outcome: 0 for outcome in ("pass", "timeout", "blow")}
    low_headroom_entry_trades = 0
    for record in records:
        groups = _economic_groups((record,))
        low_headroom_trades = sum(
            _count(group.get("trades"), name="low-headroom trades")
            for group in groups
            if group.get("headroom_stratum") == "low_headroom_le_0_25"
        )
        low_headroom_entry_trades += low_headroom_trades
        if low_headroom_trades:
            low_headroom_by_outcome[str(record.get("outcome"))] += 1
    terminal_near_blow_timeouts = sum(
        bool(record.get("near_blow_timeout", False))
        for record in records
        if record.get("outcome") == "timeout"
    )
    return {
        "terminal_near_blow_timeouts": terminal_near_blow_timeouts,
        "episodes_with_low_headroom_entries": sum(
            low_headroom_by_outcome.values()
        ),
        "passes_with_low_headroom_entries": low_headroom_by_outcome["pass"],
        "timeouts_with_low_headroom_entries": low_headroom_by_outcome["timeout"],
        "blows_with_low_headroom_entries": low_headroom_by_outcome["blow"],
        "low_headroom_entry_trades": low_headroom_entry_trades,
        "coverage_note": (
            "terminal_near_blow_timeouts counts only flagged terminal timeouts; "
            "low-headroom entries are a broader entry-time risk indicator, not "
            "a path-wise minimum-headroom measurement"
        ),
    }


def _default_output_dir(diagnostics: Path) -> Path:
    parents = diagnostics.resolve().parents
    if len(parents) < 5 or parents[3].name != "campaign-runs":
        raise ValueError(
            "--output-dir is required outside a PropEvolve campaign layout"
        )
    return parents[4] / "analysis"


def analyze_pass_timeout_diagnostics(
    diagnostics_path: str | Path,
    *,
    recent_window: int = 5,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Analyze one immutable JSONL snapshot and reuse its sealed report."""
    if isinstance(recent_window, bool) or int(recent_window) <= 0:
        raise ValueError("recent_window must be positive")
    diagnostics = Path(diagnostics_path).resolve(strict=True)
    payload, records = _load_snapshot(diagnostics)
    source_sha256 = hashlib.sha256(payload).hexdigest()
    window = min(int(recent_window), len(records))
    recent = records[-window:]
    episodes = [int(record["episode"]) for record in records]
    outcomes = [str(record.get("outcome")) for record in records]
    if any(outcome not in {"pass", "timeout", "blow"} for outcome in outcomes):
        raise ValueError("training diagnostic outcome is invalid")
    passes = outcomes.count("pass")
    timeouts = outcomes.count("timeout")
    blows = outcomes.count("blow")
    terminal_near_blow_timeouts = sum(
        bool(record.get("near_blow_timeout", False))
        for record in records
        if record.get("outcome") == "timeout"
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PARTIAL",
        "episodes": {
            "through_episode": episodes[-1],
            "count": len(records),
            "passes": passes,
            "timeouts": timeouts,
            "blows": blows,
            "terminal_near_blow_timeouts": terminal_near_blow_timeouts,
            "pass_rate": _ratio(passes, len(records)),
            "blow_rate": _ratio(blows, len(records)),
        },
        "selection": {
            "overall": _selection(records),
            "recent": {
                "episode_range": [int(recent[0]["episode"]), int(recent[-1]["episode"])],
                **_selection(recent),
            },
        },
        "outcome_comparison": {
            outcome: _outcome_summary(records, outcome)
            for outcome in ("pass", "timeout", "blow")
        },
        "risk": _risk_summary(records),
        "passes": _pass_rows(records),
        "inputs": {
            "training_diagnostics_path": str(diagnostics),
            "training_diagnostics_sha256": source_sha256,
            "recent_window": window,
        },
    }
    report["identity_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _default_output_dir(diagnostics)
    )
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (
        f"pass-timeout-v2-through-episode-{episodes[-1]:06d}-"
        f"{source_sha256[:12]}.json"
    )
    report["output_path"] = str(output)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text() != serialized:
            raise ValueError("existing pass/timeout analysis identity conflicts")
        return report
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(serialized)
        try:
            os.link(temporary, output)
        except FileExistsError:
            if output.read_text() != serialized:
                raise ValueError(
                    "concurrent pass/timeout analysis identity conflicts"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PropEvolve pass and timeout diagnostics."
    )
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--recent-window", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = analyze_pass_timeout_diagnostics(
        arguments.diagnostics,
        recent_window=arguments.recent_window,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "analyze_pass_timeout_diagnostics", "main"]
