"""Inference-only assessment using production prepared embeddings and scores."""
import argparse
import json
from propevolve.reasoning_policy.learning_audit import assess_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--role", choices=("train", "valid"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root")
    args = parser.parse_args()
    print(json.dumps(assess_prepared(args.config, args.view, role=args.role,
        output=args.output, root=args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
