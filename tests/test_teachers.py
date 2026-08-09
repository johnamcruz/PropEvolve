from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from propevolve.teachers import (
    TeacherSignalCache,
    build_directional_oof_teacher_cache,
    load_teacher_bundle,
    write_teacher_signal_cache,
)


CHANNELS = (
    "pivot_long_probability",
    "pivot_short_probability",
    "expansion_long_probability",
    "expansion_short_probability",
)


def test_portable_teacher_bundle_authenticates_all_versioned_assets() -> None:
    bundle = load_teacher_bundle("models/teachers/manifest.json")

    assert bundle["inference_dependency"] is False
    assert bundle["pivot"]["checkpoint"] == "pivot/pivot.pt"
    assert bundle["expansion"]["checkpoint"] == "expansion/expansion.pt"


def test_teacher_bundle_rejects_non_oof_policy_before_loading_assets(
    tmp_path: Path,
) -> None:
    payload = json.loads(Path("models/teachers/manifest.json").read_text())
    payload["pivot"]["oof_policy"] = "in_sample"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="strict temporal OOF"):
        load_teacher_bundle(manifest)


def _cache(tmp_path: Path) -> Path:
    timestamps = np.arange(4, dtype=np.int64).astype("datetime64[ns]")
    probabilities = np.asarray([
        [0.8, 0.2, 0.7, 0.3],
        [0.1, 0.9, 0.2, 0.8],
        [0.6, 0.4, 0.5, 0.5],
        [0.4, 0.6, 0.3, 0.7],
    ], dtype=np.float32)
    availability = np.ones_like(probabilities, dtype=np.bool_)
    availability[2, :2] = False
    return write_teacher_signal_cache(
        destination=tmp_path / "teachers/NQ",
        ticker="NQ",
        timestamps=timestamps,
        probabilities=probabilities,
        availability=availability,
        channels=CHANNELS,
        source_artifact_sha256s={name: "a" * 64 for name in CHANNELS},
        research_end_exclusive="2026-01-01",
    )


def test_teacher_cache_authenticates_oof_lineage_and_exact_row_alignment(
    tmp_path: Path,
) -> None:
    root = _cache(tmp_path)
    expected = np.arange(4, dtype=np.int64).astype("datetime64[ns]")

    cache = TeacherSignalCache.load(
        root,
        ticker="NQ",
        channels=CHANNELS,
        expected_timestamps=expected,
    )

    assert cache.manifest["source_policy"] == "strict_temporal_oof"
    assert cache.manifest["inference_dependency"] is False
    assert cache.probabilities.shape == (4, 4)
    assert cache.availability.dtype == np.bool_
    assert not cache.availability[2, 0]


def test_teacher_cache_rejects_timestamp_or_artifact_drift(tmp_path: Path) -> None:
    root = _cache(tmp_path)
    shifted = np.arange(1, 5, dtype=np.int64).astype("datetime64[ns]")
    with pytest.raises(ValueError, match="timestamps do not exactly align"):
        TeacherSignalCache.load(
            root,
            ticker="NQ",
            channels=CHANNELS,
            expected_timestamps=shifted,
        )

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_policy"] = "in_sample"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="strict temporal OOF"):
        TeacherSignalCache.load(root, ticker="NQ", channels=CHANNELS)


def test_teacher_cache_physically_excludes_sealed_2026_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed boundary"):
        write_teacher_signal_cache(
            destination=tmp_path / "teachers/NQ",
            ticker="NQ",
            timestamps=np.asarray(
                ["2025-12-31T23:57", "2026-01-01T00:00"],
                dtype="datetime64[ns]",
            ),
            probabilities=np.full((2, 4), 0.5, dtype=np.float32),
            availability=np.ones((2, 4), dtype=np.bool_),
            channels=CHANNELS,
            source_artifact_sha256s={name: "a" * 64 for name in CHANNELS},
            research_end_exclusive="2026-01-01",
        )


def test_directional_oof_converter_routes_candidates_without_dense_inputs(
    tmp_path: Path,
) -> None:
    embedding_root = tmp_path / "embedding"
    embedding_root.mkdir()
    timestamps = np.asarray(
        ["2024-01-01T00:03", "2024-01-01T00:06", "2024-01-01T00:09"],
        dtype="datetime64[ns]",
    )
    np.save(embedding_root / "embeddings.npy", np.zeros((3, 2), np.float16))
    np.save(embedding_root / "timestamps.npy", timestamps)
    (embedding_root / "manifest.json").write_text(json.dumps({
        "schema": "propevolve_chronos2_embedding_cache_v2",
        "ticker": "NQ",
        "rows": 3,
        "research_end_exclusive": "2026-01-01T00:00:00+00:00",
        "sealed_holdout_touched": False,
    }))
    common = {
        "bus_identity_sha256": np.asarray("b" * 64),
        "decision_timestamps": timestamps[[0, 2]],
        "directions": np.asarray([1, -1], np.int8),
        "fold_years": np.asarray([2024, 2024], np.int16),
    }
    pivot = tmp_path / "pivot.npz"
    expansion = tmp_path / "expansion.npz"
    np.savez(pivot, **common, pivot_probabilities=np.asarray([0.8, 0.9], np.float32))
    np.savez(
        expansion,
        **{**common, "bus_identity_sha256": np.asarray("c" * 64)},
        launch_probabilities=np.asarray([0.7, 0.6], np.float32),
    )

    result = build_directional_oof_teacher_cache(
        embedding_cache_root=embedding_root,
        pivot_oof=pivot,
        expansion_oof=expansion,
        destination=tmp_path / "teachers/NQ",
        ticker="NQ",
        channels=CHANNELS,
        research_end_exclusive="2026-01-01",
    )
    cache = TeacherSignalCache.load(result, ticker="NQ", channels=CHANNELS)

    np.testing.assert_allclose(cache.probabilities[0], [0.8, 0.0, 0.7, 0.0])
    np.testing.assert_allclose(cache.probabilities[2], [0.0, 0.9, 0.0, 0.6])
    assert not cache.availability[1].any()
