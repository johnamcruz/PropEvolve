from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from propevolve.teachers.base import BaseTeacher
from propevolve.teachers.trend import (
    TrendTeacher,
    TrendTeacherCache,
    TrendTeacherTargets,
    build_trend_teacher_cache,
    load_builder_config,
)


def test_verified_trend_teacher_matches_original_golden_scores() -> None:
    teacher = TrendTeacher.load("teachers/manifest.json", device="cpu")
    trajectory = (
        ((np.arange(50 * 2560, dtype=np.int64) % 257) - 128)
        .astype(np.float32)
        .reshape(1, 50, 2560)
        / 64.0
    )

    probabilities = teacher.score(trajectory, ticker="NQ")

    assert isinstance(teacher, BaseTeacher)
    np.testing.assert_allclose(
        probabilities,
        [[
            0.05807192623615265,
            0.09233824908733368,
            0.5490055084228516,
            0.6543344259262085,
        ]],
        rtol=1e-6,
        atol=1e-6,
    )


def _embedding_cache(tmp_path: Path) -> Path:
    root = tmp_path / "embeddings/NQ"
    root.mkdir(parents=True)
    rows = 80
    values = (
        ((np.arange(rows * 2560, dtype=np.int64) % 113) - 56)
        .astype(np.float16)
        .reshape(rows, 2560)
        / np.float16(32.0)
    )
    timestamps = np.datetime64("2024-12-31T21:00") + (
        np.arange(rows) * np.timedelta64(3, "m")
    )
    np.save(root / "embeddings.npy", values)
    np.save(root / "timestamps.npy", timestamps.astype("datetime64[ns]"))
    source_manifest = tmp_path / "NQ_3min_fixture.manifest.json"
    source_manifest.write_text("{}")
    embedding_stat = os.stat(root / "embeddings.npy")
    validation = source_manifest.with_name("NQ_3min_fixture.validation.json")
    validation.write_text(json.dumps({
        "schema": "pivot-frozen-representation-validation-v1",
        "cache_identity_sha256": (
            "1087cb9b2d7bd1dd51e219dd5d792cb52368b1221f6255f662db02d900dd72ca"
        ),
        "embeddings_sha256": (
            "35e025c51631327f96a26838ed76d85448d7defde64c4d54b89b5ae19de723bc"
        ),
        "finite": True,
        "embedding_stat": {
            "device": embedding_stat.st_dev,
            "inode": embedding_stat.st_ino,
            "mtime_ns": embedding_stat.st_mtime_ns,
            "size": embedding_stat.st_size,
        },
    }))
    (root / "manifest.json").write_text(json.dumps({
        "schema": "propevolve_chronos2_embedding_cache_v2",
        "ticker": "NQ",
        "rows": rows,
        "encoder_identity_sha256": (
            "1b8b7f001b0b4e501aa47ca90a3c2fd31d0b41dbd1d896e98ce084f6ed325710"
        ),
        "imported_ffm_cache_identity_sha256": (
            "1087cb9b2d7bd1dd51e219dd5d792cb52368b1221f6255f662db02d900dd72ca"
        ),
        "imported_ffm_embeddings_sha256": (
            "35e025c51631327f96a26838ed76d85448d7defde64c4d54b89b5ae19de723bc"
        ),
        "imported_ffm_cache_manifest": str(source_manifest),
        "research_end_exclusive": "2026-01-01T00:00:00+00:00",
        "sealed_holdout_touched": False,
    }))
    return root


def test_builder_censors_selection_and_aligns_training_targets(tmp_path: Path) -> None:
    source = _embedding_cache(tmp_path)
    teacher = TrendTeacher.load("teachers/manifest.json", device="cpu")
    destination = tmp_path / "teacher/NQ"

    result = build_trend_teacher_cache(
        teacher=teacher,
        embedding_cache=source,
        destination=destination,
        ticker="NQ",
        training_end_exclusive="2025-01-01",
        expected_cache_identity_sha256=(
            "1087cb9b2d7bd1dd51e219dd5d792cb52368b1221f6255f662db02d900dd72ca"
        ),
        batch_size=4,
        synchronization_batches=2,
    )
    cache = TrendTeacherCache.load(result)
    targets = TrendTeacherTargets.load(
        tmp_path / "teacher",
        {"NQ": SimpleNamespace(timestamps=cache.timestamps)},
    )

    assert cache.probabilities.shape == (60, 4)
    assert not cache.availability[:49].any()
    assert cache.availability[49:].all()
    assert targets.target("NQ", 0) is None
    assert targets.target("NQ", 49).shape == (4,)
    assert (cache.timestamps < np.datetime64("2025-01-01")).all()


def test_builder_rejects_embedding_identity_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache identity mismatch"):
        build_trend_teacher_cache(
            teacher=TrendTeacher.load("teachers/manifest.json", device="cpu"),
            embedding_cache=_embedding_cache(tmp_path),
            destination=tmp_path / "teacher/NQ",
            ticker="NQ",
            training_end_exclusive="2025-01-01",
            expected_cache_identity_sha256="e" * 64,
            batch_size=4,
            synchronization_batches=2,
        )


def test_promoted_trend_builder_config_is_training_only() -> None:
    config = load_builder_config("config/trend_teacher_cache_v1.json")

    assert config["tickers"] == (
        "NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"
    )
    assert config["batch_sizes"] == (1024, 512, 256)
    assert config["training_end_exclusive"] == "2025-01-01"
    assert config["sealed_start"] == "2026-01-01"
