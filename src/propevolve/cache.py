"""Dense, causal, memory-mapped cache of frozen FFM embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
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
        embeddings=_contiguous_embedding_view(cache.embeddings, cache_rows),
        embeddings_authenticated=True,
    )


def _contiguous_embedding_view(
    embeddings: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    """Preserve the memory map for chronological slices instead of copying it."""
    if len(rows) < 1:
        raise ValueError("embedding slice cannot be empty")
    if len(rows) > 1 and not np.all(np.diff(rows) == 1):
        raise ValueError("embedding selection must be one chronological range")
    return embeddings[int(rows[0]):int(rows[-1]) + 1]


def import_ffm_representation_cache(
    *,
    source: str | Path,
    source_cache_root: str | Path,
    destination: str | Path,
    ticker: str,
    timeframe_minutes: int,
    context_length: int,
    stride: int,
    research_end_exclusive: str,
    encoder_identity_sha256: str,
) -> Path:
    """Import one authenticated FFM cache using a zero-copy embedding symlink."""
    source = Path(source).resolve(strict=True)
    source_cache_root = Path(source_cache_root).resolve(strict=True)
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"refusing to replace existing cache: {destination}")
    timeframe = f"{int(timeframe_minutes)}min"
    manifests = tuple(source_cache_root.glob(f"{ticker}_{timeframe}_*.manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"expected exactly one FFM cache manifest for {ticker}@{timeframe}, "
            f"found {len(manifests)}"
        )
    source_manifest_path = manifests[0]
    source_manifest = json.loads(source_manifest_path.read_text())
    expected = {
        "schema": "pivot-frozen-representation-bar-cache-v2",
        "ticker": ticker,
        "timeframe": timeframe,
        "stride": int(stride),
        "encoder_identity_sha256": encoder_identity_sha256,
        "decision_availability": "endpoint_bar_close",
        "source_bar_timestamp_semantics": "bar_open",
        "boundary_policy": "endpoint_bar_close_strictly_before_research_end",
    }
    drift = {
        key: (source_manifest.get(key), value)
        for key, value in expected.items()
        if source_manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"FFM cache contract drift for {ticker}: {drift}")
    if source_manifest.get("encoder", {}).get("checkpoint", {}).get("stage") != "mask":
        raise ValueError("FFM cache is not from the Mask checkpoint")
    if (
        int(source_manifest.get("encoder", {}).get("input", {}).get("context_length", -1))
        != int(context_length)
    ):
        raise ValueError("FFM cache context length drift")
    if source_manifest.get("bars_sha256") != _sha256(source):
        raise ValueError("FFM cache source identity drift")

    research_end = pd.Timestamp(research_end_exclusive)
    research_end = (
        research_end.tz_localize("UTC")
        if research_end.tzinfo is None
        else research_end.tz_convert("UTC")
    )
    manifest_end = pd.Timestamp(source_manifest.get("research_end_exclusive"))
    manifest_end = (
        manifest_end.tz_localize("UTC")
        if manifest_end.tzinfo is None
        else manifest_end.tz_convert("UTC")
    )
    if manifest_end != research_end:
        raise ValueError("FFM cache sealed boundary drift")

    stem = source_manifest_path.name.removesuffix(".manifest.json")
    index_path = source_cache_root / f"{stem}.npz"
    embeddings_path = source_cache_root / str(source_manifest.get("embedding_file", ""))
    if not index_path.is_file() or not embeddings_path.is_file():
        raise ValueError("FFM cache artifact set is incomplete")
    if _sha256(index_path) != source_manifest.get("cache_sha256"):
        raise ValueError("FFM cache row-map identity drift")
    if _sha256(embeddings_path) != source_manifest.get("embeddings_sha256"):
        raise ValueError("FFM cache embedding identity drift")

    with np.load(index_path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"identity_sha256", "bar_idx", "timestamp_ns"}:
            raise ValueError("FFM cache row-map schema drift")
        identity = str(artifact["identity_sha256"].item())
        bar_indices = np.asarray(artifact["bar_idx"], dtype=np.int64)
        open_timestamp_ns = np.asarray(artifact["timestamp_ns"], dtype=np.int64)
    if identity != source_manifest.get("identity_sha256"):
        raise ValueError("FFM cache manifest and row-map identities disagree")
    rows = int(source_manifest.get("rows", -1))
    if bar_indices.shape != (rows,) or open_timestamp_ns.shape != (rows,):
        raise ValueError("FFM cache row-map shape drift")
    if rows < 2 or (np.diff(bar_indices) <= 0).any() or (np.diff(open_timestamp_ns) <= 0).any():
        raise ValueError("FFM cache row map must be nonempty, unique, and ordered")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    expected_shape = (rows, int(source_manifest.get("feature_width", -1)))
    if embeddings.shape != expected_shape or embeddings.dtype != np.float16:
        raise ValueError("FFM cache embedding shape or dtype drift")
    del embeddings

    source_times = pd.read_csv(source, usecols=["datetime"])["datetime"]
    source_times = pd.to_datetime(source_times, utc=True, errors="coerce")
    if source_times.isna().any() or int(bar_indices[-1]) >= len(source_times):
        raise ValueError("FFM cache row map exceeds its source timeline")
    selected_open_ns = source_times.iloc[bar_indices].to_numpy(
        dtype="datetime64[ns]"
    ).astype(np.int64)
    if not np.array_equal(selected_open_ns, open_timestamp_ns):
        raise ValueError("FFM cache row map does not match source timestamps")
    decision_ns = open_timestamp_ns + int(pd.Timedelta(minutes=timeframe_minutes).value)
    decision_times = decision_ns.astype("datetime64[ns]")
    boundary = np.datetime64(research_end.tz_localize(None))
    if not (decision_times < boundary).all():
        raise ValueError("FFM cache crosses the sealed boundary")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        os.symlink(embeddings_path, temporary / "embeddings.npy")
        np.save(temporary / "timestamps.npy", decision_times)
        manifest = {
            "schema": CACHE_SCHEMA,
            "ticker": ticker,
            "timeframe_minutes": int(timeframe_minutes),
            "context_length": int(context_length),
            "stride": int(stride),
            "rows": rows,
            "embedding_dim": expected_shape[1],
            "source": str(source),
            "source_sha256": source_manifest["bars_sha256"],
            "checkpoint": "imported authenticated FFM Mask cache",
            "checkpoint_sha256": source_manifest["encoder"]["checkpoint"]["sha256"],
            "encoder_identity_sha256": encoder_identity_sha256,
            "decision_timing": "embedding includes completed bar at decision close",
            "first_decision_time": str(decision_times[0]),
            "last_decision_time": str(decision_times[-1]),
            "research_end_exclusive": research_end.isoformat(),
            "sealed_holdout_touched": False,
            "imported_ffm_cache_identity_sha256": identity,
            "imported_ffm_cache_manifest": str(source_manifest_path),
            "imported_ffm_embeddings_sha256": source_manifest["embeddings_sha256"],
            "embedding_storage": "external_npy_float16_memmap_symlink_v1",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(destination)
    return destination


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
