"""Dense, causal, memory-mapped cache of frozen FFM embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


OHLCV = ("open", "high", "low", "close", "volume")
CACHE_SCHEMA = "propevolve_chronos2_embedding_cache_v2"


class WindowEncoder(Protocol):
    checkpoint: Path

    def encode(self, windows: np.ndarray) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EmbeddingCache:
    root: Path
    embeddings: np.ndarray
    timestamps: np.ndarray
    manifest: dict

    @classmethod
    def load(cls, root: str | Path) -> "EmbeddingCache":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != CACHE_SCHEMA:
            raise ValueError("unsupported embedding cache schema")
        embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
        timestamps = np.load(root / "timestamps.npy", mmap_mode="r")
        if embeddings.ndim != 2 or timestamps.shape != (len(embeddings),):
            raise ValueError("embedding cache arrays violate their shape contract")
        if int(manifest.get("rows", -1)) != len(embeddings):
            raise ValueError("embedding cache manifest row count drift")
        research_end = manifest.get("research_end_exclusive")
        if not research_end or manifest.get("sealed_holdout_touched") is not False:
            raise ValueError("embedding cache lacks an authenticated sealed boundary")
        boundary = np.datetime64(pd.Timestamp(research_end).tz_convert("UTC").tz_localize(None))
        if len(timestamps) and not (timestamps < boundary).all():
            raise ValueError("embedding cache crosses its sealed boundary")
        return cls(root.resolve(), embeddings, timestamps, manifest)


def load_market_series(
    source: str | Path,
    cache_root: str | Path,
    *,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
):
    """Align source OHLC bars to cached decision-close timestamps."""
    from .environment import MarketSeries

    source = Path(source).resolve(strict=True)
    cache = EmbeddingCache.load(cache_root)
    if cache.manifest.get("ticker") != ticker:
        raise ValueError("embedding cache ticker does not match requested market")
    if cache.manifest.get("source_sha256") != _sha256(source):
        raise ValueError("embedding cache source identity drift")
    research_end = np.datetime64(
        pd.Timestamp(cache.manifest["research_end_exclusive"])
        .tz_convert("UTC")
        .tz_localize(None)
    )
    if end is not None and np.datetime64(end) > research_end:
        raise ValueError("requested market slice crosses the sealed boundary")
    if start is not None and np.datetime64(start) >= research_end:
        raise ValueError("requested market slice begins in the sealed holdout")
    timeframe = int(cache.manifest["timeframe_minutes"])
    frame = pd.read_csv(source, usecols=["datetime", "open", "high", "low", "close"])
    opens = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    closes = (opens + pd.Timedelta(minutes=timeframe)).to_numpy(dtype="datetime64[ns]")
    cache_times = np.asarray(cache.timestamps)
    indices = np.searchsorted(closes, cache_times)
    if (
        (indices >= len(closes)).any()
        or not np.array_equal(closes[indices], cache_times)
    ):
        raise ValueError("embedding timestamps do not align to source bar closes")
    eligible = np.ones(len(indices), dtype=bool)
    if start is not None:
        eligible &= cache_times >= np.datetime64(start)
    if end is not None:
        eligible &= cache_times < np.datetime64(end)
    indices = indices[eligible]
    cache_rows = np.flatnonzero(eligible)
    if len(indices) < 2:
        raise ValueError("requested market slice has fewer than two cached rows")
    return MarketSeries(
        ticker=ticker,
        timestamps=closes[indices],
        open=frame["open"].to_numpy(np.float32)[indices],
        high=frame["high"].to_numpy(np.float32)[indices],
        low=frame["low"].to_numpy(np.float32)[indices],
        close=frame["close"].to_numpy(np.float32)[indices],
        embeddings=np.asarray(cache.embeddings[cache_rows], dtype=np.float32),
    )


def build_embedding_cache(
    *,
    source: str | Path,
    destination: str | Path,
    ticker: str,
    encoder: WindowEncoder,
    checkpoint_sha256: str,
    research_end_exclusive: str,
    context_length: int,
    stride: int,
    chunk_windows: int,
    timeframe_minutes: int,
) -> Path:
    """Encode causal pre-holdout windows and write a manifest-last cache."""
    if min(context_length, stride, chunk_windows, timeframe_minutes) < 1:
        raise ValueError("cache dimensions must be positive")
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    frame = pd.read_csv(source, usecols=["datetime", *OHLCV])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    if frame.empty or frame["datetime"].isna().any():
        raise ValueError(f"invalid source timestamps: {source}")
    if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
        raise ValueError(f"source timestamps must be unique and ordered: {source}")
    research_end = pd.Timestamp(research_end_exclusive)
    if research_end.tzinfo is None:
        research_end = research_end.tz_localize("UTC")
    else:
        research_end = research_end.tz_convert("UTC")
    close_availability = frame["datetime"] + pd.Timedelta(minutes=timeframe_minutes)
    frame = frame.loc[close_availability < research_end].reset_index(drop=True)
    if frame.empty:
        raise ValueError("source has no rows strictly before the sealed boundary")
    values = frame.loc[:, OHLCV].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"source OHLCV must be finite: {source}")
    o, h, low, c, volume = values.T
    if (
        (h < np.maximum(o, c)).any()
        or (low > np.minimum(o, c)).any()
        or (h < low).any()
        or (volume < 0).any()
    ):
        raise ValueError(f"source OHLCV relationships are invalid: {source}")
    if len(values) < context_length:
        raise ValueError("source is shorter than the requested context")

    windows = np.lib.stride_tricks.sliding_window_view(
        values, context_length, axis=0
    )[::stride]
    # sliding_window_view returns [N, C, T] for an [rows, C] source.
    count = len(windows)
    close_times = (
        frame["datetime"] + pd.Timedelta(minutes=timeframe_minutes)
    ).to_numpy(dtype="datetime64[ns]")[context_length - 1::stride]
    if len(close_times) != count:
        raise RuntimeError("cache timestamp alignment failed")

    destination.mkdir(parents=True, exist_ok=True)
    temp_embeddings = destination / ".embeddings.tmp.npy"
    chunks = (
        np.ascontiguousarray(windows[start:min(count, start + chunk_windows)])
        for start in range(0, count, chunk_windows)
    )
    if hasattr(encoder, "encode_chunks"):
        encoded_chunks = encoder.encode_chunks(chunks, context_length=context_length)
    else:
        encoded_chunks = (encoder.encode(chunk) for chunk in chunks)
    iterator = iter(encoded_chunks)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("embedding encoder returned no cache chunks") from exc
    output = np.lib.format.open_memmap(
        temp_embeddings,
        mode="w+",
        dtype=np.float32,
        shape=(count, first.shape[1]),
    )
    position = len(first)
    output[:position] = first
    for encoded in iterator:
        end = position + len(encoded)
        if end > count or encoded.shape[1:] != first.shape[1:]:
            raise RuntimeError("embedding chunks violate the cache shape contract")
        output[position:end] = encoded
        position = end
    if position != count:
        raise RuntimeError(f"embedding cache wrote {position} of {count} rows")
    output.flush()
    del output
    temp_embeddings.replace(destination / "embeddings.npy")
    np.save(destination / "timestamps.npy", close_times)
    manifest = {
        "schema": CACHE_SCHEMA,
        "ticker": str(ticker),
        "timeframe_minutes": timeframe_minutes,
        "context_length": context_length,
        "stride": stride,
        "rows": count,
        "embedding_dim": int(first.shape[1]),
        "source": str(source),
        "source_sha256": _sha256(source),
        "checkpoint": str(Path(encoder.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "decision_timing": "embedding includes bar close at timestamp",
        "first_decision_time": str(close_times[0]),
        "last_decision_time": str(close_times[-1]),
        "research_end_exclusive": research_end.isoformat(),
        "sealed_holdout_touched": False,
    }
    temporary_manifest = destination / ".manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(destination / "manifest.json")
    return destination
