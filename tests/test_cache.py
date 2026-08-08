from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from propevolve.cache import EmbeddingCache, build_embedding_cache, load_market_series


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
