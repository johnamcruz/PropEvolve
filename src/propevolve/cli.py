"""Command-line entrypoints for the bounded PropEvolve POC."""

from __future__ import annotations

import argparse
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
            context_length=int(cache_config["context_length"]),
            stride=int(cache_config["stride"]),
            chunk_windows=int(cache_config["chunk_windows"]),
            timeframe_minutes=int(config["timeframe_minutes"]),
        )
        print(f"[cache] COMPLETE {ticker} {destination}", flush=True)
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train(config: dict) -> int:
    from .agent import RecurrentC51Agent
    from .environment import ChallengeSpec, HistoricalChallengeEnv
    from .replay import BalancedSequenceReplay
    from .training import (
        evaluate_agent,
        load_markets,
        train_agent,
        write_run_report,
    )

    root = Path(config["_path"]).parent.parent
    assets = AssetContract.load(_resolve(root, config["assets"]))
    temporal = config["temporal"]
    cache_root = _resolve(root, config["cache_root"])
    train_markets = load_markets(
        asset_contract=assets,
        cache_root=cache_root,
        tickers=config["tickers"],
        timeframe_minutes=int(config["timeframe_minutes"]),
        start=temporal["train_start"],
        end=temporal["train_end"],
    )
    validation_markets = load_markets(
        asset_contract=assets,
        cache_root=cache_root,
        tickers=config["deployment_tickers"],
        timeframe_minutes=int(config["timeframe_minutes"]),
        start=temporal["validation_start"],
        end=temporal["validation_end"],
    )
    challenge = ChallengeSpec(**config["challenge"])
    seed = int(config["training"]["seed"])
    train_environment = HistoricalChallengeEnv(
        train_markets,
        tick_values=config["point_values"],
        round_trip_fees=config["round_trip_fees"],
        spec=challenge,
        seed=seed,
    )
    validation_environment = HistoricalChallengeEnv(
        validation_markets,
        tick_values={key: config["point_values"][key] for key in validation_markets},
        round_trip_fees={key: config["round_trip_fees"][key] for key in validation_markets},
        spec=challenge,
        seed=seed + 1,
    )
    observation_dim = next(iter(train_markets.values())).embeddings.shape[1] + 12
    agent_config = dict(config["agent"])
    agent = RecurrentC51Agent(observation_dim, seed=seed, **agent_config)
    training_config = config["training"]
    replay = BalancedSequenceReplay(
        capacity_episodes=int(training_config["replay_capacity_episodes"]),
        sequence_length=int(training_config["sequence_length"]),
        seed=seed,
    )
    training = train_agent(
        agent,
        train_environment,
        episodes=int(training_config["episodes"]),
        replay=replay,
        warmup_episodes=int(training_config["warmup_episodes"]),
        updates_per_episode=int(training_config["updates_per_episode"]),
        batch_sequences=int(training_config["batch_sequences"]),
        recurrent_horizon=int(training_config["recurrent_horizon"]),
    )
    validation = evaluate_agent(
        agent,
        validation_environment,
        episodes=int(training_config["validation_episodes"]),
        recurrent_horizon=int(training_config["recurrent_horizon"]),
    )
    output = _resolve(root, config["output"])
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = Path(config["_path"]).read_bytes()
    model_manifest = {
        "checkpoint_sha256": assets.checkpoint_sha256,
        "experiment_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_tickers": list(config["tickers"]),
        "deployment_tickers": list(config["deployment_tickers"]),
        "sealed_start": temporal["sealed_start"],
    }
    agent.save(output / "challenger.pt", manifest=model_manifest)
    report = write_run_report(
        output / "report.json",
        config_path=config["_path"],
        assets=assets,
        training=training,
        validation=validation,
    )
    print(f"[train] COMPLETE report={report}", flush=True)
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
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
