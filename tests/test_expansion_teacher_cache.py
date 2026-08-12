from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from propevolve.teachers.base import BaseTeacher
from propevolve.teachers.composition import load_teacher_targets
from propevolve.teachers.expansion import (
    CHANNELS,
    ExpansionTeacher,
    ExpansionTeacherCache,
    ExpansionTeacherTargets,
    build_expansion_teacher_cache,
    load_builder_config,
    verify_expansion_entry_center_receipt,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry_center_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    tickers = ("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN")
    cache_root = tmp_path / "teacher"
    sources = []
    probabilities = np.array(
        [[0.2, 0.5, 0.4, 0.5], [0.6, 0.5, 0.2, 0.5]],
        dtype=np.float32,
    )
    availability = np.array([True, True], dtype=np.bool_)
    timestamps = np.array(
        ["2021-01-04T14:30", "2024-12-31T22:00"], dtype="datetime64[ns]"
    )
    for ticker in tickers:
        destination = cache_root / ticker
        destination.mkdir(parents=True)
        np.save(destination / "probabilities.npy", probabilities)
        np.save(destination / "availability.npy", availability)
        np.save(destination / "timestamps.npy", timestamps)
        manifest = {
            "schema": "propevolve_expansion_teacher_cache_v1",
            "channels": [
                "long_attempt_probability",
                "long_clean_retained_given_attempt_probability",
                "short_attempt_probability",
                "short_clean_retained_given_attempt_probability",
            ],
            "rows": 2,
            "training_end_exclusive": "2025-01-01",
            "selection_rows_included": False,
            "sealed_rows_included": False,
            "probabilities_sha256": _sha256(destination / "probabilities.npy"),
            "availability_sha256": _sha256(destination / "availability.npy"),
            "timestamps_sha256": _sha256(destination / "timestamps.npy"),
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        sources.append({
            "ticker": ticker,
            "manifest_sha256": _sha256(manifest_path),
            "probabilities_sha256": manifest["probabilities_sha256"],
            "availability_sha256": manifest["availability_sha256"],
            "timestamps_sha256": manifest["timestamps_sha256"],
            "rows": 2,
            "available_rows": 2,
            "first_timestamp": "2021-01-04T14:30:00.000000000",
            "last_timestamp": "2024-12-31T22:00:00.000000000",
        })
    receipt = {
        "schema": "propevolve_expansion_entry_center_receipt_v1",
        "formula": (
            "pooled_available_float64_mean("
            "attempt_probability*clean_retained_given_attempt_probability)"
        ),
        "training_start_inclusive": "2021-01-01",
        "training_end_exclusive": "2025-01-01",
        "tickers": list(tickers),
        "channels": list(manifest["channels"]),
        "pooled_available_rows": 18,
        "long_center": 0.20000000670552254,
        "short_center": 0.15000000223517418,
        "sources": sources,
    }
    receipt_path = tmp_path / "entry-centers.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    spec = {
        "cache_root": str(cache_root.relative_to(tmp_path)),
        "entry_search_long_center": receipt["long_center"],
        "entry_search_short_center": receipt["short_center"],
        "entry_search_center_receipt": str(receipt_path.relative_to(tmp_path)),
        "entry_search_center_receipt_sha256": _sha256(receipt_path),
    }
    return cache_root, receipt_path, spec


def test_entry_centers_are_verified_against_authenticated_cache_rows(
    tmp_path: Path,
) -> None:
    _, _, spec = _entry_center_fixture(tmp_path)

    centers = verify_expansion_entry_center_receipt(
        spec, root=tmp_path,
        expected_tickers=("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"),
    )

    assert centers == pytest.approx((0.20000000670552254, 0.15000000223517418))


def test_composed_teacher_load_verifies_entry_center_receipt_before_use(
    tmp_path: Path,
) -> None:
    cache_root, _, spec = _entry_center_fixture(tmp_path)
    spec.update({
        "kind": "expansion",
        "channels": CHANNELS,
        "loss_weight": 0.2,
        "entry_search_loss_weight": 0.3,
        "entry_search_objective": "centered_log_odds",
    })
    markets = {
        ticker: SimpleNamespace(
            timestamps=np.load(cache_root / ticker / "timestamps.npy")
        )
        for ticker in ("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN")
    }

    targets = load_teacher_targets((spec,), root=tmp_path, markets=markets)

    assert targets.target("NQ", 0) is not None


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("receipt_bytes", "receipt identity drifted"),
        ("formula", "receipt contract"),
        ("center", "configured centers"),
        ("manifest", "source identity drifted"),
        ("array", "cache probabilities identity drifted"),
        ("rows", "source rows drifted"),
        ("period", "source period drifted"),
    ],
)
def test_entry_center_receipt_fails_closed_on_tamper(
    tmp_path: Path, tamper: str, message: str,
) -> None:
    cache_root, receipt_path, spec = _entry_center_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    if tamper == "receipt_bytes":
        receipt_path.write_text(receipt_path.read_text() + " ")
    elif tamper == "formula":
        receipt["formula"] = "different"
    elif tamper == "center":
        spec["entry_search_long_center"] += 0.01
    elif tamper == "manifest":
        receipt["sources"][0]["manifest_sha256"] = "0" * 64
    elif tamper == "array":
        values = np.load(cache_root / "NQ/probabilities.npy")
        values[0, 0] += np.float32(0.01)
        np.save(cache_root / "NQ/probabilities.npy", values)
    elif tamper == "rows":
        receipt["sources"][0]["available_rows"] = 1
    elif tamper == "period":
        receipt["training_end_exclusive"] = "2024-01-01"
    if tamper != "receipt_bytes":
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        spec["entry_search_center_receipt_sha256"] = _sha256(receipt_path)

    with pytest.raises(ValueError, match=message):
        verify_expansion_entry_center_receipt(
            spec, root=tmp_path,
            expected_tickers=("NQ", "ES", "GC", "RTY", "YM", "CL", "SI", "ZB", "ZN"),
        )
