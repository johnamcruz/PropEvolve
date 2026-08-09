from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propevolve.cache import (
    EmbeddingCache,
    build_embedding_cache,
    import_ffm_representation_cache,
    load_market_series,
)


class MeanEncoder:
    checkpoint = Path("/frozen/mask")

    def encode(self, windows: np.ndarray) -> np.ndarray:
        return windows.mean(axis=2)


def test_cache_uses_only_windows_ending_at_each_decision_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "NQ_3min.csv"
    times = pd.date_range("2024-01-01", periods=6, freq="3min", tz="UTC")
    close = np.arange(100, 106, dtype=float)
    pd.DataFrame({
        "datetime": times,
        "open": close - 0.25,
        "high": close + 0.50,
        "low": close - 0.50,
        "close": close,
        "volume": np.arange(10, 16, dtype=float),
    }).to_csv(source, index=False)

    path = build_embedding_cache(
        source=source,
        destination=tmp_path / "cache/NQ",
        ticker="NQ",
        encoder=MeanEncoder(),
        checkpoint_sha256="abc",
        context_length=3,
        stride=1,
        chunk_windows=2,
        timeframe_minutes=3,
        research_end_exclusive="2026-01-01",
    )
    cache = EmbeddingCache.load(path)

    assert cache.embeddings.shape == (4, 5)
    assert cache.timestamps.astype("datetime64[m]").tolist() == [
        np.datetime64((value + pd.Timedelta(minutes=3)).to_datetime64(), "m")
        for value in times[2:]
    ]
    assert cache.manifest["decision_timing"] == "embedding includes bar close at timestamp"
    assert cache.manifest["research_end_exclusive"] == "2026-01-01T00:00:00+00:00"
    assert cache.manifest["sealed_holdout_touched"] is False
    np.testing.assert_allclose(cache.embeddings[0, 3], close[:3].mean())

    market = load_market_series(source, path, ticker="NQ")
    assert market.ticker == "NQ"
    assert market.embeddings.shape == (4, 5)
    np.testing.assert_array_equal(market.timestamps, cache.timestamps)


def test_cache_physically_censors_sealed_rows_before_encoding(tmp_path: Path) -> None:
    boundary = "2026-01-01T00:00:00Z"
    times = pd.to_datetime([
        "2025-12-31T23:45:00Z",
        "2025-12-31T23:48:00Z",
        "2025-12-31T23:51:00Z",
        "2025-12-31T23:54:00Z",
        "2025-12-31T23:57:00Z",  # closes exactly at the sealed boundary
        "2026-01-01T00:00:00Z",
    ])

    def write_source(path: Path, sealed_close: float) -> None:
        close = np.array([100, 101, 102, 103, sealed_close, sealed_close + 1], dtype=float)
        pd.DataFrame({
            "datetime": times,
            "open": close - 0.25,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": np.arange(10, 16, dtype=float),
        }).to_csv(path, index=False)

    source_a = tmp_path / "a.csv"
    source_b = tmp_path / "b.csv"
    write_source(source_a, 104.0)
    write_source(source_b, 10_000.0)

    paths = []
    for name, source in (("a", source_a), ("b", source_b)):
        paths.append(build_embedding_cache(
            source=source,
            destination=tmp_path / name,
            ticker="NQ",
            encoder=MeanEncoder(),
            checkpoint_sha256="abc",
            context_length=3,
            stride=1,
            chunk_windows=2,
            timeframe_minutes=3,
            research_end_exclusive=boundary,
        ))

    first, second = (EmbeddingCache.load(path) for path in paths)
    np.testing.assert_array_equal(first.timestamps, second.timestamps)
    np.testing.assert_allclose(first.embeddings, second.embeddings)
    assert first.timestamps[-1] == np.datetime64("2025-12-31T23:57:00")
    assert (first.timestamps < np.datetime64("2026-01-01T00:00:00")).all()


def test_cache_rejects_manifest_without_authenticated_sealed_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    np.save(root / "embeddings.npy", np.zeros((2, 3), np.float32))
    np.save(root / "timestamps.npy", np.array([
        "2025-01-01T00:00:00", "2025-01-01T00:03:00"
    ], dtype="datetime64[ns]"))
    (root / "manifest.json").write_text(json.dumps({
        "schema": "propevolve_chronos2_embedding_cache_v1",
        "rows": 2,
    }))

    with pytest.raises(ValueError, match="unsupported embedding cache schema"):
        EmbeddingCache.load(root)


def test_imports_authenticated_ffm_cache_without_copying_embeddings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "NQ_3min.csv"
    times = pd.date_range("2024-01-01", periods=6, freq="3min", tz="UTC")
    close = np.arange(100, 106, dtype=float)
    pd.DataFrame({
        "datetime": times,
        "open": close - 0.25,
        "high": close + 0.50,
        "low": close - 0.50,
        "close": close,
        "volume": np.arange(10, 16, dtype=float),
    }).to_csv(source, index=False)

    external = tmp_path / "ffm-cache"
    external.mkdir()
    stem = "NQ_3min_frozen_representation_deadbeefdeadbeef"
    embeddings_path = external / f"{stem}.embeddings.npy"
    embeddings = np.arange(12, dtype=np.float16).reshape(3, 4)
    np.save(embeddings_path, embeddings)
    identity = "a" * 64
    index_path = external / f"{stem}.npz"
    np.savez(
        index_path,
        identity_sha256=np.asarray(identity),
        bar_idx=np.asarray([2, 3, 4], dtype=np.int64),
        timestamp_ns=times[2:5].to_numpy(dtype="datetime64[ns]").astype(np.int64),
    )
    source_sha = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    embeddings_sha = __import__("hashlib").sha256(
        embeddings_path.read_bytes()
    ).hexdigest()
    manifest_path = external / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "pivot-frozen-representation-bar-cache-v2",
        "ticker": "NQ",
        "timeframe": "3min",
        "rows": 3,
        "feature_width": 4,
        "stride": 1,
        "bars_sha256": source_sha,
        "identity_sha256": identity,
        "encoder_identity_sha256": "b" * 64,
        "embedding_file": embeddings_path.name,
        "embedding_dtype": "float16",
        "embeddings_sha256": embeddings_sha,
        "cache_sha256": __import__("hashlib").sha256(
            index_path.read_bytes()
        ).hexdigest(),
        "decision_availability": "endpoint_bar_close",
        "source_bar_timestamp_semantics": "bar_open",
        "boundary_policy": "endpoint_bar_close_strictly_before_research_end",
        "research_end_exclusive": "2026-01-01T00:00:00.000000000Z",
        "encoder": {
            "checkpoint": {"stage": "mask", "sha256": "d" * 64},
            "input": {"context_length": 3},
        },
    }))

    destination = import_ffm_representation_cache(
        source=source,
        source_cache_root=external,
        destination=tmp_path / "cache/NQ",
        ticker="NQ",
        timeframe_minutes=3,
        context_length=3,
        stride=1,
        research_end_exclusive="2026-01-01",
        encoder_identity_sha256="b" * 64,
    )
    cache = EmbeddingCache.load(destination)

    assert (destination / "embeddings.npy").is_symlink()
    assert (destination / "embeddings.npy").resolve() == embeddings_path.resolve()
    assert cache.embeddings.dtype == np.float16
    np.testing.assert_array_equal(cache.embeddings, embeddings)
    np.testing.assert_array_equal(
        cache.timestamps,
        (times[2:5] + pd.Timedelta(minutes=3)).to_numpy(dtype="datetime64[ns]"),
    )
    assert cache.manifest["imported_ffm_cache_identity_sha256"] == identity
    assert cache.manifest["sealed_holdout_touched"] is False
    market = load_market_series(source, destination, ticker="NQ")
    assert isinstance(market.embeddings, np.memmap)
    assert market.embeddings.dtype == np.float16
