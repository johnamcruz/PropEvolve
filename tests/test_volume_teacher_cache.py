from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from propevolve.teachers.base import BaseTeacher
from propevolve.teachers.volume import (
    VolumeTeacher,
    VolumeTeacherCache,
    VolumeTeacherTargets,
    build_volume_teacher_cache,
    load_builder_config,
)
from test_trend_teacher_cache import _embedding_cache


def test_verified_volume_teacher_matches_source_artifact_golden_scores() -> None:
    teacher = VolumeTeacher.load("teachers/manifest.json", device="cpu")
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
            0.0005371672892477446,
            0.04509155972007788,
            0.9548159006720348,
            0.5686046661538035,
        ]],
        rtol=1e-5,
        atol=1e-7,
    )


def test_volume_cache_is_causal_aligned_and_training_only(tmp_path: Path) -> None:
    source = _embedding_cache(tmp_path)
    teacher = VolumeTeacher.load("teachers/manifest.json", device="cpu")
    destination = tmp_path / "volume/NQ"

    result = build_volume_teacher_cache(
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
    cache = VolumeTeacherCache.load(result)
    targets = VolumeTeacherTargets.load(
        tmp_path / "volume",
        {"NQ": SimpleNamespace(timestamps=cache.timestamps)},
    )

    assert cache.probabilities.shape == (60, 4)
    assert not cache.availability[:49].any()
    assert cache.availability[49:].all()
    assert targets.target("NQ", 0) is None
    assert targets.target("NQ", 49).shape == (4,)
    assert (cache.timestamps < np.datetime64("2025-01-01")).all()


def test_promoted_volume_builder_config_is_training_only() -> None:
    config = load_builder_config("config/volume_teacher_cache_v1.json")

    assert config["tickers"] == (
        "NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"
    )
    assert config["batch_sizes"] == (1024, 512, 256)
    assert config["training_end_exclusive"] == "2025-01-01"
    assert config["sealed_start"] == "2026-01-01"
