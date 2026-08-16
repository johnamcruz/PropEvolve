from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

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
    assert CHANNELS == (
        "chop_no_trend_probability",
        "chop_end_transition_probability",
        "expansion_trend_probability",
    )
    assert probabilities.shape == (1, 3)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        probabilities[0],
        (0.48450127, 0.34060439, 0.17489433),
        rtol=0.0,
        # Supported Torch CPU kernels can differ slightly in float32 softmax.
        atol=1e-6,
    )
    assert teacher.manifest["regime"]["validation_verdict"] == "PROCEED"
    assert teacher.manifest["regime"]["best_epoch"] == 39


def test_regime_targets_require_exact_training_row_alignment(tmp_path: Path) -> None:
    root = tmp_path / "regime/NQ"
    root.mkdir(parents=True)
    timestamps = np.datetime64("2024-01-01") + np.arange(4) * np.timedelta64(3, "m")
    probabilities = np.full(
        (4, len(CHANNELS)), 1.0 / len(CHANNELS), dtype=np.float32
    )
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


@pytest.mark.parametrize(
    "invalid_row",
    (
        (-0.1, 0.6, 0.5),
        (0.5, 0.5, 0.5),
    ),
)
def test_regime_cache_rejects_non_simplex_probabilities(
    tmp_path: Path,
    invalid_row: tuple[float, float, float],
) -> None:
    root = tmp_path / "regime"
    root.mkdir()
    probabilities = np.asarray([invalid_row], dtype=np.float32)
    availability = np.ones(1, dtype=np.bool_)
    timestamps = np.asarray(["2024-01-01"], dtype="datetime64[ns]")
    np.save(root / "probabilities.npy", probabilities)
    np.save(root / "availability.npy", availability)
    np.save(root / "timestamps.npy", timestamps)

    def digest(name: str) -> str:
        return hashlib.sha256((root / name).read_bytes()).hexdigest()

    (root / "manifest.json").write_text(json.dumps({
        "schema": "propevolve_regime_teacher_cache_v1",
        "channels": list(CHANNELS),
        "rows": 1,
        "training_end_exclusive": "2025-01-01",
        "selection_rows_included": False,
        "sealed_rows_included": False,
        "probabilities_sha256": digest("probabilities.npy"),
        "availability_sha256": digest("availability.npy"),
        "timestamps_sha256": digest("timestamps.npy"),
    }))

    with pytest.raises(ValueError, match="cache arrays are invalid"):
        RegimeTeacherCache.load(root)


def test_regime_teacher_rejects_validation_for_another_checkpoint(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "teachers"
    (assets / "regime").mkdir(parents=True)
    shutil.copy2(
        "teachers/regime/regime_teacher.pt",
        assets / "regime/regime_teacher.pt",
    )
    validation_path = assets / "regime/regime.validation.json"
    validation = json.loads(
        Path("teachers/regime/regime.validation.json").read_text()
    )
    validation["candidate_identity"]["checkpoint_sha256"] = "0" * 64
    validation_path.write_text(json.dumps(validation))
    manifest = json.loads(Path("teachers/manifest.json").read_text())
    manifest["regime"]["validation_sha256"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    manifest_path = assets / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="validation evidence drifted"):
        RegimeTeacher.load(manifest_path, device="cpu")


def test_promoted_regime_builder_is_bounded_to_training_data() -> None:
    config = load_builder_config("config/regime_teacher_cache_v1.json")

    assert config["tickers"] == (
        "NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"
    )
    assert config["batch_sizes"] == (1024, 512, 256)
    assert config["training_end_exclusive"] == "2025-01-01"
    assert config["sealed_start"] == "2026-01-01"
    assert config["output_root"] == (
        "cache/expansion_anchored_regime_teacher_9market_3min_pre2025_v1"
    )


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


def test_regime_builder_reuses_validated_embeddings_after_volume_remount(
    tmp_path: Path,
) -> None:
    source = _embedding_cache(tmp_path)
    source_manifest = Path(
        json.loads((source / "manifest.json").read_text())[
            "imported_ffm_cache_manifest"
        ]
    )
    validation_path = source_manifest.with_name(
        "NQ_3min_fixture.validation.json"
    )
    validation = json.loads(validation_path.read_text())
    validation["embedding_stat"]["device"] += 1
    validation_path.write_text(json.dumps(validation))

    result = build_regime_teacher_cache(
        teacher=RegimeTeacher.load("teachers/manifest.json", device="cpu"),
        embedding_cache=source,
        destination=tmp_path / "regime/NQ",
        ticker="NQ",
        training_end_exclusive="2025-01-01",
        expected_cache_identity_sha256=(
            "1087cb9b2d7bd1dd51e219dd5d792cb52368b1221f6255f662db02d900dd72ca"
        ),
        batch_size=4,
        synchronization_batches=2,
    )

    assert result == (tmp_path / "regime/NQ").resolve()
