from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from propevolve.teachers.base import BaseTeacher
from propevolve.teachers.expansion import (
    ExpansionTeacher,
    ExpansionTeacherCache,
    ExpansionTeacherTargets,
    build_expansion_teacher_cache,
    load_builder_config,
)


def test_verified_expansion_teacher_matches_original_golden_scores() -> None:
    teacher = ExpansionTeacher.load(
        "teachers/manifest.json",
        device="cpu",
    )
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
        [[0.6735228608968472, 0.6628845980562301,
          0.37719524754537676, 0.5494872737809613]],
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


def test_builder_physically_censors_selection_and_is_resumable(tmp_path: Path) -> None:
    source = _embedding_cache(tmp_path)
    teacher = ExpansionTeacher.load("teachers/manifest.json", device="cpu")
    destination = tmp_path / "teacher/NQ"

    result = build_expansion_teacher_cache(
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
    before = (destination / "manifest.json").stat().st_mtime_ns
    hit = build_expansion_teacher_cache(
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
    cache = ExpansionTeacherCache.load(destination)

    assert result == hit == destination.resolve()
    assert (destination / "manifest.json").stat().st_mtime_ns == before
    assert cache.probabilities.shape == (60, 4)
    assert not cache.availability[:49].any()
    assert cache.availability[49:].all()
    assert np.isfinite(cache.probabilities[cache.availability]).all()
    assert (cache.timestamps < np.datetime64("2025-01-01")).all()


def test_builder_rejects_embedding_cache_identity_drift(tmp_path: Path) -> None:
    source = _embedding_cache(tmp_path)
    teacher = ExpansionTeacher.load("teachers/manifest.json", device="cpu")

    with pytest.raises(ValueError, match="cache identity mismatch"):
        build_expansion_teacher_cache(
            teacher=teacher,
            embedding_cache=source,
            destination=tmp_path / "teacher/NQ",
            ticker="NQ",
            training_end_exclusive="2025-01-01",
            expected_cache_identity_sha256="e" * 64,
            batch_size=4,
            synchronization_batches=2,
        )


def test_teacher_targets_require_exact_training_row_alignment(tmp_path: Path) -> None:
    source = _embedding_cache(tmp_path)
    teacher = ExpansionTeacher.load("teachers/manifest.json", device="cpu")
    destination = tmp_path / "teacher/NQ"
    build_expansion_teacher_cache(
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
    cache = ExpansionTeacherCache.load(destination)

    targets = ExpansionTeacherTargets.load(
        tmp_path / "teacher",
        {"NQ": SimpleNamespace(timestamps=cache.timestamps)},
    )

    assert targets.target("NQ", 0) is None
    assert targets.target("NQ", 49).shape == (4,)


def test_promoted_builder_config_is_bounded_to_training_data() -> None:
    config = load_builder_config("config/expansion_teacher_cache_v1.json")

    assert config["tickers"] == (
        "NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"
    )
    assert config["batch_sizes"] == (1024, 512, 256)
    assert config["training_end_exclusive"] == "2025-01-01"
    assert config["sealed_start"] == "2026-01-01"
