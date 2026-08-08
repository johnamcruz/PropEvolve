from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    )
    cache = EmbeddingCache.load(path)

    assert cache.embeddings.shape == (4, 5)
    assert cache.timestamps.astype("datetime64[m]").tolist() == [
        np.datetime64((value + pd.Timedelta(minutes=3)).to_datetime64(), "m")
        for value in times[2:]
    ]
    assert cache.manifest["decision_timing"] == "embedding includes bar close at timestamp"
    np.testing.assert_allclose(cache.embeddings[0, 3], close[:3].mean())

    market = load_market_series(source, path, ticker="NQ")
    assert market.ticker == "NQ"
    assert market.embeddings.shape == (4, 5)
    np.testing.assert_array_equal(market.timestamps, cache.timestamps)
