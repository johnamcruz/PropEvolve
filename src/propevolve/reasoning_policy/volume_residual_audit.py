"""Matched audit for incremental Volume information in policy mistakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .integrity import file_digest


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0 or not np.isfinite(scores).all():
        raise ValueError("AUC requires finite scores and both classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _signal(values: np.ndarray, name: str) -> np.ndarray:
    long_edge = values[:, 0] * values[:, 1]
    short_edge = values[:, 2] * values[:, 3]
    if name == "long_edge":
        return long_edge
    if name == "short_edge":
        return short_edge
    if name == "quietness":
        return 1.0 - np.maximum(long_edge, short_edge)
    raise ValueError(f"unknown Volume residual signal: {name}")


def assess_rows(rows, *, actions: dict[str, str], minimum_auc: float,
                minimum_matched_rate: float) -> dict:
    """Compare correct and incorrect decisions after nearest E/R/T matching."""
    if (not actions or not 0.5 <= minimum_auc <= 1.0
            or not 0.5 <= minimum_matched_rate <= 1.0):
        raise ValueError("invalid Volume residual acceptance contract")
    reports, failed = {}, []
    for action, signal_name in actions.items():
        selected = [row for row in rows if row["target"] == action]
        labels = np.asarray([row["correct"] for row in selected], dtype=bool)
        controls = np.stack([np.asarray(row["controls"], np.float64)
                             for row in selected])
        volume = np.stack([np.asarray(row["volume"], np.float64)
                           for row in selected])
        if (controls.ndim != 2 or volume.shape != (len(selected), 4)
                or not np.isfinite(controls).all() or not np.isfinite(volume).all()
                or np.any(volume < 0) or np.any(volume > 1)):
            raise ValueError("invalid residual-audit rows")
        values = _signal(volume, signal_name)
        differences = []
        for ticker in sorted({row["ticker"] for row in selected}):
            indices = np.asarray([index for index, row in enumerate(selected)
                                  if row["ticker"] == ticker], dtype=np.int64)
            correct = indices[labels[indices]]
            mistakes = indices[~labels[indices]]
            if not len(correct) or not len(mistakes):
                continue
            local = controls[indices]
            scale = local.std(axis=0)
            scale[scale < 1e-6] = 1.0
            normalized = (local - local.mean(axis=0)) / scale
            positions = {int(global_index): local_index
                         for local_index, global_index in enumerate(indices)}
            correct_positions = np.asarray([positions[int(index)] for index in correct])
            for mistake in mistakes:
                delta = normalized[positions[int(mistake)]] - normalized[correct_positions]
                partner = correct[int(np.argmin(np.square(delta).sum(axis=1)))]
                differences.append(float(values[partner] - values[mistake]))
        if not differences:
            raise ValueError(f"{action} lacks within-ticker matched correctness pairs")
        report = {
            "rows": len(selected),
            "correct": int(labels.sum()),
            "mistakes": int((~labels).sum()),
            "signal": signal_name,
            "auc_correct": _rank_auc(labels, values),
            "correct_mean": float(values[labels].mean()),
            "mistake_mean": float(values[~labels].mean()),
            "matched_pairs": len(differences),
            "matched_correct_minus_mistake_mean": float(np.mean(differences)),
            "matched_correct_higher_rate": float(np.mean(np.asarray(differences) > 0)),
        }
        report["passed"] = (
            report["auc_correct"] >= minimum_auc
            and report["matched_correct_higher_rate"] >= minimum_matched_rate
        )
        reports[action] = report
        if not report["passed"]:
            failed.append(action)
    return {
        "schema": "propevolve_volume_residual_audit_v1",
        "status": "PASS" if not failed else "REJECTED",
        "minimum_auc": minimum_auc,
        "minimum_matched_rate": minimum_matched_rate,
        "failed_actions": failed,
        "actions": reports,
    }


def _load_cache(root: Path, ticker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest_path = root / ticker / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    arrays = tuple(np.load(root / ticker / f"{name}.npy", mmap_mode="r")
                   for name in ("timestamps", "probabilities", "availability"))
    timestamps, probabilities, availability = arrays
    if (probabilities.shape[0] != len(timestamps)
            or availability.shape != (len(timestamps),)
            or tuple(probabilities.shape[1:]) != (len(manifest["channels"]),)
            or not all(file_digest(root / ticker / f"{name}.npy")
                       == manifest[f"{name}_sha256"]
                       for name in ("timestamps", "probabilities", "availability"))):
        raise ValueError("specialist cache failed residual-audit authentication")
    return timestamps, probabilities, availability


def run_audit(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    required = {"schema", "scores", "teacher_cache_roots", "actions",
                "minimum_auc", "minimum_matched_rate", "output"}
    if set(config) != required or config["schema"] != "volume_residual_audit_v1":
        raise ValueError("invalid Volume residual audit configuration")
    root = config_path.parent.parent.parent
    cache_roots = {kind: root / path
                   for kind, path in config["teacher_cache_roots"].items()}
    if tuple(cache_roots) != ("expansion", "regime", "trend", "volume"):
        raise ValueError("residual audit requires ordered E/R/T controls and Volume")
    scores_path = root / config["scores"]
    raw_rows = [json.loads(line) for line in scores_path.read_text().splitlines()
                if line.strip()]
    cache = {}
    rows = []
    for source in raw_rows:
        ticker = source["ticker"]
        if ticker not in cache:
            cache[ticker] = {kind: _load_cache(path, ticker)
                             for kind, path in cache_roots.items()}
        features = {}
        for kind, (timestamps, probabilities, availability) in cache[ticker].items():
            timestamp = np.datetime64(int(source["completed_at_ns"]), "ns")
            index = int(np.searchsorted(timestamps, timestamp))
            if (index >= len(timestamps) or timestamps[index] != timestamp
                    or not availability[index]):
                raise ValueError("assessment row does not align to causal teacher cache")
            features[kind] = np.asarray(probabilities[index], np.float64)
        rows.append({
            "ticker": ticker, "target": source["target"],
            "correct": bool(source["correct"]),
            "controls": np.concatenate(tuple(features[kind]
                                             for kind in ("expansion", "regime", "trend"))),
            "volume": features["volume"],
        })
    result = assess_rows(
        rows, actions=config["actions"],
        minimum_auc=float(config["minimum_auc"]),
        minimum_matched_rate=float(config["minimum_matched_rate"]),
    )
    result.update(
        scores_sha256=file_digest(scores_path),
        config_sha256=file_digest(config_path),
        assessed_rows=len(rows),
    )
    output = root / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_audit(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
