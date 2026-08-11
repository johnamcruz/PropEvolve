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
    setup.add_argument("--embedding-cache")
    validate = subparsers.add_parser("validate-config", help="validate experiment JSON")
    validate.add_argument("--config", required=True)
    cache = subparsers.add_parser("build-cache", help="build frozen Chronos2 caches")
    cache.add_argument("--config", required=True)
    cache.add_argument("--ticker", action="append")
    expansion_teacher = subparsers.add_parser(
        "build-expansion-teacher-cache",
        help="score verified Expansion teachers over existing Mask caches",
    )
    expansion_teacher.add_argument("--config", required=True)
    expansion_teacher.add_argument("--ticker", action="append")
    regime_teacher = subparsers.add_parser(
        "build-regime-teacher-cache",
        help="score the verified Regime teacher over existing Mask caches",
    )
    regime_teacher.add_argument("--config", required=True)
    regime_teacher.add_argument("--ticker", action="append")
    train = subparsers.add_parser("train", help="train and validate historical challenger")
    train.add_argument("--config", required=True)
    evolve = subparsers.add_parser(
        "evolve", help="run or resume reasoning-guided offline evolution"
    )
    evolve.add_argument("--config", required=True)
    evolve.add_argument("--run-id", required=True)
    evolve.add_argument(
        "--recover-reasoning",
        action="store_true",
        help="resume from the latest durable NEEDS_REASONING checkpoint",
    )
    status = subparsers.add_parser("evolve-status", help="show durable campaign state")
    status.add_argument("--config", required=True)
    status.add_argument("--run-id", required=True)
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _build_caches(config: dict) -> int:
    from .cache import (
        EmbeddingCache,
        build_embedding_cache,
        import_ffm_representation_cache,
    )

    root = Path(config["_path"]).parent.parent
    assets = AssetContract.load(_resolve(root, config["assets"]))
    assets.verify()
    cache_config = config["cache"]
    cache_format = str(cache_config["format"])
    encoder = None
    if cache_format == "native":
        from .encoder import FrozenChronos2Encoder

        encoder = FrozenChronos2Encoder(
            assets.checkpoint,
            device=cache_config["device"],
            batch_series=int(cache_config["batch_series"]),
            fast_group_attention=bool(cache_config["fast_group_attention"]),
        )
    elif assets.embedding_cache is None:
        raise ValueError("FFM cache import requires assets.embedding_cache")
    cache_root = _resolve(root, config["cache_root"])
    research_end = str(config["temporal"]["sealed_start"])
    requested = tuple(config.get("_requested_tickers") or config["tickers"])
    unknown = set(requested) - set(config["tickers"])
    if unknown:
        raise ValueError(f"requested cache tickers are outside the contract: {sorted(unknown)}")
    for ticker in requested:
        destination = cache_root / ticker
        source = Path(assets.market_data) / (
            f"{ticker}_{config['timeframe_minutes']}min.csv"
        )
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            cached = EmbeddingCache.load(destination)
            common_match = (
                manifest.get("context_length") == int(cache_config["context_length"])
                and manifest.get("stride") == int(cache_config["stride"])
                and manifest.get("source_sha256") == _file_sha256(source)
                and manifest.get("research_end_exclusive")
                == _utc_isoformat(research_end)
                and manifest.get("sealed_holdout_touched") is False
                and len(cached.embeddings) == int(manifest["rows"])
            )
            identity_match = (
                manifest.get("encoder_identity_sha256")
                == cache_config["encoder_identity_sha256"]
                if cache_format == "ffm_frozen_representation_v2"
                else manifest.get("checkpoint_sha256") == assets.checkpoint_sha256
            )
            if common_match and identity_match:
                print(f"[cache] HIT {ticker} {destination}", flush=True)
                continue
            raise ValueError(f"existing cache identity conflicts for {ticker}")
        if cache_format == "ffm_frozen_representation_v2":
            import_ffm_representation_cache(
                source=source,
                source_cache_root=assets.embedding_cache,
                destination=destination,
                ticker=ticker,
                timeframe_minutes=int(config["timeframe_minutes"]),
                context_length=int(cache_config["context_length"]),
                stride=int(cache_config["stride"]),
                research_end_exclusive=research_end,
                encoder_identity_sha256=str(cache_config["encoder_identity_sha256"]),
            )
            print(f"[cache] IMPORTED {ticker} {destination}", flush=True)
        else:
            assert encoder is not None
            build_embedding_cache(
                source=source,
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


def _build_expansion_teacher_caches(
    config_path: str,
    requested_tickers: Sequence[str],
) -> int:
    from .teachers.expansion import build_expansion_teacher_caches

    build_expansion_teacher_caches(
        config_path,
        requested_tickers=tuple(requested_tickers),
    )
    return 0


def _build_regime_teacher_caches(
    config_path: str,
    requested_tickers: Sequence[str],
) -> int:
    from .teachers.regime import build_regime_teacher_caches

    build_regime_teacher_caches(
        config_path,
        requested_tickers=tuple(requested_tickers),
    )
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


def _evolve(
    config_path: str,
    run_id: str,
    *,
    recover_reasoning: bool = False,
) -> int:
    from .orchestration import run_evolution_campaign

    state = run_evolution_campaign(
        config_path,
        run_id=run_id,
        recover_reasoning=recover_reasoning,
    )
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
        contract = link_local_assets(
            args.workspace,
            args.market_data,
            args.checkpoint,
            args.embedding_cache,
        )
        print(json.dumps(contract.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "build-expansion-teacher-cache":
        return _build_expansion_teacher_caches(
            args.config,
            tuple(args.ticker or ()),
        )
    if args.command == "build-regime-teacher-cache":
        return _build_regime_teacher_caches(
            args.config,
            tuple(args.ticker or ()),
        )
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
        return _evolve(
            args.config,
            args.run_id,
            recover_reasoning=args.recover_reasoning,
        )
    if args.command == "evolve-status":
        return _evolve_status(config, args.run_id)
    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
