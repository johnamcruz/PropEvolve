"""Read-only frozen-model scoring of real labeled SFT examples (explicit command)."""

import argparse
import json
from pathlib import Path

from .decision_schema import legal_completion_names
from .policy import MLXActionPolicy


def score_labeled_examples(policy, records):
    """Measure every action class independently; do not hide side collapse."""
    output = []
    for record in records:
        answer = record["messages"][-1]["content"]
        choices = legal_completion_names(answer)
        scores = policy.completion_scores(record["messages"][:-1], choices)
        predicted = max(scores, key=scores.get)
        output.append({
            "source_id": record["source_id"], "completed_at_ns": record["completed_at_ns"],
            "target": answer, "predicted": predicted,
            "correct": predicted == answer, "target_log_likelihood": scores[answer],
            "target_advantage": scores[answer] - max(value for name, value in scores.items() if name != answer),
        })
    if not output:
        raise ValueError("learning audit needs labeled examples")
    return output


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
