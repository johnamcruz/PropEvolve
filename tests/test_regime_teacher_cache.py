from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from propevolve.teachers.base import BaseTeacher
from propevolve.teachers.regime import (
    CHANNELS,
    RegimeTeacher,
    RegimeTeacherCache,
    RegimeTeacherTargets,
    build_regime_teacher_cache,
    load_builder_config,
)
from test_expansion_teacher_cache import _embedding_cache


def test_verified_regime_teacher_has_authenticated_soft_outputs() -> None:
    teacher = RegimeTeacher.load("teachers/manifest.json", device="cpu")
    trajectory = np.zeros((1, 50, 2560), dtype=np.float32)

    probabilities = teacher.score(trajectory)

    assert isinstance(teacher, BaseTeacher)
    assert probabilities.shape == (1, len(CHANNELS))
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(probabilities[:, 0:3].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(probabilities[:, 3:8].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(probabilities[:, 9:12].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(probabilities[:, 12:17].sum(axis=1), 1.0, atol=1e-6)


def test_regime_targets_require_exact_training_row_alignment(tmp_path: Path) -> None:
    root = tmp_path / "regime/NQ"
    root.mkdir(parents=True)
    timestamps = np.datetime64("2024-01-01") + np.arange(4) * np.timedelta64(3, "m")
    probabilities = np.full((4, len(CHANNELS)), 0.5, dtype=np.float32)
    availability = np.array([False, True, True, True], dtype=np.bool_)
    np.save(root / "probabilities.npy", probabilities)
    np.save(root / "availability.npy", availability)
    np.save(root / "timestamps.npy", timestamps.astype("datetime64[ns]"))

    import hashlib

    def digest(name: str) -> str:
        return hashlib.sha256((root / name).read_bytes()).hexdigest()

    (root / "manifest.json").write_text(json.dumps({
        "schema": "propevolve_regime_teacher_cache_v1",
        "channels": list(CHANNELS),
        "rows": 4,
        "training_end_exclusive": "2025-01-01",
        "selection_rows_included": False,
        "sealed_rows_included": False,
        "probabilities_sha256": digest("probabilities.npy"),
        "availability_sha256": digest("availability.npy"),
        "timestamps_sha256": digest("timestamps.npy"),
    }))

    cache = RegimeTeacherCache.load(root)
    targets = RegimeTeacherTargets.load(
        tmp_path / "regime",
        {"NQ": SimpleNamespace(timestamps=cache.timestamps)},
    )

    assert targets.target("NQ", 0) is None
    assert targets.target("NQ", 1).shape == (len(CHANNELS),)


def test_promoted_regime_builder_is_bounded_to_training_data() -> None:
    config = load_builder_config("config/regime_teacher_cache_v1.json")

    assert config["tickers"] == (
        "NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"
    )
    assert config["batch_sizes"] == (1024, 512, 256)
    assert config["training_end_exclusive"] == "2025-01-01"
    assert config["sealed_start"] == "2026-01-01"


def test_regime_builder_censors_selection_and_is_resumable(tmp_path: Path) -> None:
    source = _embedding_cache(tmp_path)
    teacher = RegimeTeacher.load("teachers/manifest.json", device="cpu")
    destination = tmp_path / "regime/NQ"

    result = build_regime_teacher_cache(
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
    hit = build_regime_teacher_cache(
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
    cache = RegimeTeacherCache.load(destination)

    assert result == hit == destination.resolve()
    assert (destination / "manifest.json").stat().st_mtime_ns == before
    assert cache.probabilities.shape == (60, len(CHANNELS))
    assert not cache.availability[:49].any()
    assert cache.availability[49:].all()
    assert (cache.timestamps < np.datetime64("2025-01-01")).all()
