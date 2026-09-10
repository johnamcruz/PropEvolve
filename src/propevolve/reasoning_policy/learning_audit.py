"""Read-only frozen-model scoring of real labeled SFT examples (explicit command)."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .decision_schema import legal_completion_names
from .policy import MLXActionPolicy


def score_labeled_examples(policy, records):
    """Measure every action class independently; do not hide side collapse."""
    output = []
    for record in records:
        answer = record["messages"][-1]["content"]
        choices = legal_completion_names(answer)
        try:
            supplied = json.loads(record["messages"][-2]["content"]).get("legal_actions")
        except (ValueError, AttributeError):
            supplied = None  # legacy externally prepared text-only fixtures
        if supplied is not None:
            if (not supplied or len(set(supplied)) != len(supplied)
                    or not set(supplied).issubset(choices) or answer not in supplied):
                raise ValueError("audit target conflicts with legal actions")
            choices = tuple(supplied)
        context = {key: record[key] for key in ("market_embeddings", "market_available") if key in record}
        scores = policy.completion_scores(record["messages"][:-1], choices,
            **({"market_context": context} if context else {}))
        if set(scores) != set(choices) or not all(math.isfinite(x) for x in scores.values()):
            raise ValueError("invalid frozen audit scores")
        predicted = max(scores, key=scores.get)
        output.append({
            "source_id": record["source_id"], "completed_at_ns": record["completed_at_ns"],
            "target": answer, "predicted": predicted,
            "correct": predicted == answer, "target_log_likelihood": scores[answer],
            "scores": scores,
            "target_advantage": (scores[answer] - max(value for name, value in scores.items() if name != answer)
                                 if len(scores) > 1 else None),
        })
    if not output:
        raise ValueError("learning audit needs labeled examples")
    return output


def summarize_trade_mastery(scored):
    """Aggregate action-boundary evidence without challenge reward metrics."""
    required = ("WAIT", "ENTER_LONG_1", "ENTER_SHORT_1", "HOLD", "CLOSE")
    groups = {name: [] for name in required}
    for row in scored:
        target = row.get("target")
        advantage = row.get("target_advantage")
        if target not in groups or type(row.get("correct")) is not bool:
            raise ValueError("invalid trade-mastery audit row")
        if advantage is None or not math.isfinite(advantage):
            raise ValueError("trade-mastery audit requires competing legal actions")
        groups[target].append(row)
    if any(not rows for rows in groups.values()):
        raise ValueError("trade-mastery audit requires all five legal actions")
    per_action = {
        name: {
            "count": len(rows),
            "accuracy": float(np.mean([row["correct"] for row in rows])),
            "mean_target_advantage": float(np.mean(
                [row["target_advantage"] for row in rows])),
            "minimum_target_advantage": float(min(
                row["target_advantage"] for row in rows)),
        }
        for name, rows in groups.items()
    }
    return {
        "per_action": per_action,
        "macro_accuracy": float(np.mean(
            [row["accuracy"] for row in per_action.values()])),
        "worst_action_advantage": float(min(
            row["minimum_target_advantage"] for row in per_action.values())),
    }


def assess_prepared(config_path, view, *, role, output, root=None):
    """Score an authenticated prepared role without updating model weights."""
    from .mlx_sft import read_sft_config, verify_mlx_view, PreparedDataset, IndexedJsonRows
    from .supervised_trainer import evaluate_action_validation, hierarchical_boundary_metrics, action_boundary_metrics
    from .supervision import action_targets
    from .integrity import file_digest
    if role not in {"train", "valid"}:
        raise ValueError("assessment role must be train or valid")
    config = read_sft_config(config_path, root=root)
    if config.get("prepared_sampling") is not None:
        raise ValueError("assessment requires a full prepared role for exact source alignment")
    verify_mlx_view(config_path, view, root=root)
    dataset = PreparedDataset(view, role)
    source = IndexedJsonRows(Path(config["data"]) / f"{role}.jsonl")
    if len(source) != len(dataset):
        raise ValueError("assessment source and prepared rows differ")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    policy = MLXActionPolicy.from_config(config_path, root=root)
    groups = {}
    with (destination / "scores.jsonl").open("x") as stream:
        def record_score(index, scores):
            original = source[index]
            prepared = dataset.rows[index]
            if action_targets(original) != prepared["action_targets"]:
                raise ValueError("assessment economic targets differ from source")
            names = prepared["action_targets"]["names"]
            target = original["messages"][-1]["content"]
            values = dict(zip(names, scores))
            predicted = max(values, key=values.get)
            row = {"index": index, "source_id": original["source_id"],
                "completed_at_ns": original["completed_at_ns"],
                "ticker": original.get("ticker", "unknown"), "target": target,
                "predicted": predicted, "correct": predicted == target,
                "scores": values, "target_advantage": values[target] - max(
                    value for name, value in values.items() if name != target),
                "specialist_targets": original["targets"].get("specialist_targets", {})}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            grouped = groups.setdefault(row["ticker"], ([], []))
            grouped[0].append({"action_targets": prepared["action_targets"], "target_name": target})
            grouped[1].append(scores)
            if sum(len(items[0]) for items in groups.values()) % 100 == 0:
                stream.flush()
                print(f"[assessment] scored={sum(len(items[0]) for items in groups.values())}/{len(dataset)}", flush=True)
        metrics = evaluate_action_validation(policy.model, dataset, config, on_scored=record_score)
    metric = (hierarchical_boundary_metrics if config.get("decision_objective") == "hierarchical_binary"
              else action_boundary_metrics)
    report = {"role": role, "rows": len(dataset), "weights_updated": False,
        "config_sha256": file_digest(config_path),
        "view_manifest_sha256": file_digest(Path(view) / "view_manifest.json"),
        "metrics": metrics, "per_ticker": {ticker: metric(rows, scores,
            margin=config["action_supervision"]["margin"]) for ticker, (rows, scores) in groups.items()}}
    (destination / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="JSON model settings; can reuse the SFT recipe")
    selection.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--records", required=True)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--limit", required=True, type=int)
    args = parser.parse_args(argv)
    if args.config and (args.adapter is not None or args.max_seq_length is not None):
        parser.error("use config alone or explicit model/adapter/token-budget arguments")
    if not args.config and args.max_seq_length is None:
        parser.error("--model requires --max-seq-length")
    if args.limit < 1:
        raise ValueError("audit limit must be positive")
    from itertools import islice
    with Path(args.records).open() as stream:
        records = [json.loads(line) for line in islice(stream, args.limit)]
    policy = (MLXActionPolicy.from_config(args.config) if args.config else
              MLXActionPolicy.load(args.model, adapter_path=args.adapter,
                                   max_seq_length=args.max_seq_length))
    print(json.dumps(score_labeled_examples(policy, records), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
