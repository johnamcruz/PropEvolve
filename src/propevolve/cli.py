"""Command-line entrypoints for the bounded PropEvolve POC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .assets import AssetContract, link_local_assets
from .config import load_experiment_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="propevolve")
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup-assets", help="symlink data and Mask checkpoint")
    setup.add_argument("--workspace", default=".")
    setup.add_argument("--market-data", required=True)
    setup.add_argument("--checkpoint", required=True)
    validate = subparsers.add_parser("validate-config", help="validate experiment JSON")
    validate.add_argument("--config", required=True)
    cache = subparsers.add_parser("build-cache", help="build frozen Chronos2 caches")
    cache.add_argument("--config", required=True)
    cache.add_argument("--ticker", action="append")
    train = subparsers.add_parser("train", help="train and validate historical challenger")
    train.add_argument("--config", required=True)
    evolve = subparsers.add_parser(
        "evolve", help="run or resume reasoning-guided offline evolution"
    )
    evolve.add_argument("--config", required=True)
    evolve.add_argument("--run-id", required=True)
    status = subparsers.add_parser("evolve-status", help="show durable campaign state")
    status.add_argument("--config", required=True)
    status.add_argument("--run-id", required=True)
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_caches(config: dict) -> int:
    from .cache import build_embedding_cache
    from .encoder import FrozenChronos2Encoder

    root = Path(config["_path"]).parent.parent
    assets = AssetContract.load(_resolve(root, config["assets"]))
    assets.verify()
    cache_config = config["cache"]
    encoder = FrozenChronos2Encoder(
        assets.checkpoint,
        device=cache_config["device"],
        batch_series=int(cache_config["batch_series"]),
        fast_group_attention=bool(cache_config.get("fast_group_attention", False)),
    )
    cache_root = _resolve(root, config["cache_root"])
    research_end = str(config["temporal"]["sealed_start"])
    requested = tuple(config.get("_requested_tickers") or config["tickers"])
    unknown = set(requested) - set(config["tickers"])
    if unknown:
        raise ValueError(f"requested cache tickers are outside the contract: {sorted(unknown)}")
    for ticker in requested:
        destination = cache_root / ticker
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            source = Path(assets.market_data) / (
                f"{ticker}_{config['timeframe_minutes']}min.csv")
            from .cache import EmbeddingCache
            cached = EmbeddingCache.load(destination)
            if (
                manifest.get("checkpoint_sha256") == assets.checkpoint_sha256
                and manifest.get("context_length") == int(cache_config["context_length"])
                and manifest.get("stride") == int(cache_config["stride"])
                and manifest.get("source_sha256") == _file_sha256(source)
                and manifest.get("research_end_exclusive")
                == _utc_isoformat(research_end)
                and manifest.get("sealed_holdout_touched") is False
                and len(cached.embeddings) == int(manifest["rows"])
            ):
                print(f"[cache] HIT {ticker} {destination}", flush=True)
                continue
            raise ValueError(f"existing cache identity conflicts for {ticker}")
        build_embedding_cache(
            source=Path(assets.market_data)
            / f"{ticker}_{config['timeframe_minutes']}min.csv",
            destination=destination,
            ticker=ticker,
            encoder=encoder,
            checkpoint_sha256=assets.checkpoint_sha256,
            research_end_exclusive=research_end,
            context_length=int(cache_config["context_length"]),
            stride=int(cache_config["stride"]),
            chunk_windows=int(cache_config["chunk_windows"]),
            timeframe_minutes=int(config["timeframe_minutes"]),
        )
        print(f"[cache] COMPLETE {ticker} {destination}", flush=True)
    return 0


def _utc_isoformat(value: str) -> str:
    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train(config: dict) -> int:
    from .training import HistoricalCandidateRunner

    candidate, evaluation = HistoricalCandidateRunner().run(
        config,
        parent_candidate_ids=tuple(config["evolution"]["parent_candidate_ids"]),
        hypothesis=str(config["evolution"]["hypothesis"]),
    )
    print(
        f"[train] COMPLETE candidate={candidate.candidate_id} "
        f"evaluation={evaluation.evaluation_id} decision={evaluation.status}",
        flush=True,
    )
    return 0


def _state_payload(state) -> dict:
    payload = asdict(state)
    payload["phase"] = state.phase.value
    if state.last_gate is not None:
        payload["last_gate"]["decision"] = state.last_gate.decision.value
    return payload


def _evolve(config_path: str, run_id: str) -> int:
    from .orchestration import run_evolution_campaign

    state = run_evolution_campaign(config_path, run_id=run_id)
    print(json.dumps(_state_payload(state), indent=2, sort_keys=True, default=str))
    return 0 if state.phase.value == "COMPLETE" else 2


def _evolve_status(config: dict, run_id: str) -> int:
    from ml_training_loop.stores import JsonRunStore

    root = Path(config["_root"])
    state_root = root / str(config["campaign"]["state_root"])
    state = JsonRunStore(state_root).load(run_id)
    if state is None:
        print(json.dumps({"run_id": run_id, "status": "not_found"}))
        return 1
    print(json.dumps(_state_payload(state), indent=2, sort_keys=True, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "setup-assets":
        contract = link_local_assets(args.workspace, args.market_data, args.checkpoint)
        print(json.dumps(contract.__dict__, indent=2, sort_keys=True))
        return 0
    config = load_experiment_config(args.config)
    if args.command == "validate-config":
        print(f"VALID {config['_path']}")
        return 0
    if args.command == "build-cache":
        config["_requested_tickers"] = tuple(args.ticker or ())
        return _build_caches(config)
    if args.command == "train":
        return _train(config)
    if args.command == "evolve":
        return _evolve(args.config, args.run_id)
    if args.command == "evolve-status":
        return _evolve_status(config, args.run_id)
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
