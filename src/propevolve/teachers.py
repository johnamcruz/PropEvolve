"""Authenticated, training-only Pivot and Expansion teacher probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TEACHER_CACHE_SCHEMA = "propevolve_temporary_teacher_cache_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _boundary(value: str) -> np.datetime64:
    parsed = pd.Timestamp(value)
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return np.datetime64(parsed.tz_localize(None), "ns")


def load_teacher_bundle(path: str | Path) -> dict:
    """Authenticate every portable checkpoint and OOF score source."""
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "propevolve_temporary_teacher_bundle_v1"
        or payload.get("inference_dependency") is not False
        or payload.get("temporary_heads_saved") is not False
    ):
        raise ValueError("temporary teacher bundle contract is invalid")
    for family in ("pivot", "expansion"):
        record = payload.get(family)
        if not isinstance(record, dict):
            raise ValueError(f"temporary teacher bundle lacks {family}")
        if record.get("oof_policy") != "strict_temporal_oof":
            raise ValueError(
                f"temporary teacher bundle {family} must use strict temporal OOF"
            )
        for key, value in record.items():
            if not key.endswith("_sha256"):
                continue
            artifact_key = key.removesuffix("_sha256")
            artifact = path.parent / str(record.get(artifact_key, ""))
            if not artifact.is_file() or _sha256(artifact) != value:
                raise ValueError(f"temporary teacher bundle {family} identity drifted")
    payload["_path"] = str(path)
    payload["_root"] = str(path.parent)
    return payload


@dataclass(frozen=True)
class TeacherSignalCache:
    """Exact decision-row soft targets that never cross the inference seam."""

    root: Path
    probabilities: np.ndarray
    availability: np.ndarray
    timestamps: np.ndarray
    manifest: dict

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        ticker: str,
        channels: Sequence[str],
        expected_timestamps: np.ndarray | None = None,
    ) -> "TeacherSignalCache":
        root = Path(root).resolve(strict=True)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("temporary teacher cache manifest is missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != TEACHER_CACHE_SCHEMA:
            raise ValueError("unsupported temporary teacher cache schema")
        if manifest.get("source_policy") != "strict_temporal_oof":
            raise ValueError("temporary teachers must use strict temporal OOF predictions")
        if manifest.get("inference_dependency") is not False:
            raise ValueError("temporary teacher cache declares an inference dependency")
        declared_channels = tuple(str(value) for value in manifest.get("channels", ()))
        if manifest.get("ticker") != ticker or declared_channels != tuple(channels):
            raise ValueError("temporary teacher ticker or channel contract drifted")
        paths = {
            name: root / name
            for name in ("probabilities.npy", "availability.npy", "timestamps.npy")
        }
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("temporary teacher cache artifact set is incomplete")
        for key, path in paths.items():
            if _sha256(path) != manifest.get(f"{key.removesuffix('.npy')}_sha256"):
                raise ValueError(f"temporary teacher cache {key} identity drifted")
        probabilities = np.load(paths["probabilities.npy"], mmap_mode="r")
        availability = np.load(paths["availability.npy"], mmap_mode="r")
        timestamps = np.load(paths["timestamps.npy"], mmap_mode="r")
        shape = (int(manifest.get("rows", -1)), len(declared_channels))
        if (
            probabilities.shape != shape
            or availability.shape != shape
            or timestamps.shape != (shape[0],)
            or probabilities.dtype != np.float32
            or availability.dtype != np.bool_
        ):
            raise ValueError("temporary teacher cache array contract drifted")
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ValueError("temporary teacher probabilities must be finite in [0, 1]")
        if len(timestamps) and (
            np.isnat(timestamps).any()
            or not np.all(timestamps[1:] > timestamps[:-1])
            or np.any(timestamps >= _boundary(manifest["research_end_exclusive"]))
        ):
            raise ValueError("temporary teacher cache crosses its sealed boundary")
        if expected_timestamps is not None and not np.array_equal(
            np.asarray(timestamps), np.asarray(expected_timestamps).astype("datetime64[ns]")
        ):
            raise ValueError("temporary teacher timestamps do not exactly align")
        return cls(root, probabilities, availability, timestamps, manifest)


def write_teacher_signal_cache(
    *,
    destination: str | Path,
    ticker: str,
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    availability: np.ndarray,
    channels: Sequence[str],
    source_artifact_sha256s: Mapping[str, str],
    research_end_exclusive: str,
) -> Path:
    """Materialize one immutable dense row map from external OOF predictions."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"refusing to replace teacher cache: {destination}")
    channels = tuple(str(value) for value in channels)
    timestamps = np.asarray(timestamps).astype("datetime64[ns]")
    probabilities = np.asarray(probabilities, dtype=np.float32)
    availability = np.asarray(availability, dtype=np.bool_)
    shape = (len(timestamps), len(channels))
    if (
        not ticker
        or not channels
        or len(set(channels)) != len(channels)
        or probabilities.shape != shape
        or availability.shape != shape
        or len(timestamps) < 2
    ):
        raise ValueError("temporary teacher arrays or channel contract are invalid")
    if set(source_artifact_sha256s) != set(channels) or any(
        _SHA256.fullmatch(str(value)) is None
        for value in source_artifact_sha256s.values()
    ):
        raise ValueError("every teacher channel requires an authenticated source artifact")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("temporary teacher probabilities must be finite in [0, 1]")
    if (
        np.isnat(timestamps).any()
        or not np.all(timestamps[1:] > timestamps[:-1])
        or np.any(timestamps >= _boundary(research_end_exclusive))
    ):
        raise ValueError("temporary teacher cache crosses its sealed boundary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        np.save(temporary / "probabilities.npy", probabilities)
        np.save(temporary / "availability.npy", availability)
        np.save(temporary / "timestamps.npy", timestamps)
        manifest = {
            "schema": TEACHER_CACHE_SCHEMA,
            "ticker": ticker,
            "channels": list(channels),
            "rows": len(timestamps),
            "source_policy": "strict_temporal_oof",
            "decision_availability": "endpoint_bar_close",
            "research_end_exclusive": str(pd.Timestamp(research_end_exclusive)),
            "sealed_holdout_touched": False,
            "inference_dependency": False,
            "source_artifact_sha256s": dict(source_artifact_sha256s),
        }
        for name in ("probabilities", "availability", "timestamps"):
            manifest[f"{name}_sha256"] = _sha256(temporary / f"{name}.npy")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(destination)
    return destination


def build_directional_oof_teacher_cache(
    *,
    embedding_cache_root: str | Path,
    pivot_oof: str | Path,
    expansion_oof: str | Path,
    destination: str | Path,
    ticker: str,
    channels: Sequence[str],
    research_end_exclusive: str,
) -> Path:
    """Densify matched directional Pivot/Expansion OOF candidate scores."""
    from .cache import EmbeddingCache

    channels = tuple(str(value) for value in channels)
    if len(channels) != 4:
        raise ValueError("directional Pivot/Expansion teachers require four channels")
    embedding_cache = EmbeddingCache.load(embedding_cache_root)
    sources = (Path(pivot_oof).resolve(strict=True), Path(expansion_oof).resolve(strict=True))

    def load(path: Path, probability_key: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "bus_identity_sha256",
                "decision_timestamps",
                "directions",
                "fold_years",
                probability_key,
            }
            if not required <= set(artifact.files):
                raise ValueError(f"OOF teacher artifact schema is incomplete: {path}")
            values = {key: np.asarray(artifact[key]).copy() for key in required}
        rows = len(values[probability_key])
        if (
            values["decision_timestamps"].shape != (rows,)
            or values["directions"].shape != (rows,)
            or values["fold_years"].shape != (rows,)
            or not np.isfinite(values[probability_key]).all()
            or np.any((values[probability_key] < 0.0) | (values[probability_key] > 1.0))
            or not np.isin(values["directions"], (-1, 1)).all()
        ):
            raise ValueError(f"OOF teacher artifact values are malformed: {path}")
        return values

    pivot = load(sources[0], "pivot_probabilities")
    expansion = load(sources[1], "launch_probabilities")
    for key in ("decision_timestamps", "directions", "fold_years"):
        if not np.array_equal(pivot[key], expansion[key]):
            raise ValueError("Pivot and Expansion OOF teacher row identities disagree")
    dense_times = np.asarray(embedding_cache.timestamps)
    sparse_times = pivot["decision_timestamps"].astype("datetime64[ns]")
    rows = np.searchsorted(dense_times, sparse_times)
    if (
        (rows >= len(dense_times)).any()
        or not np.array_equal(dense_times[rows], sparse_times)
    ):
        raise ValueError("OOF teacher decisions do not align to frozen embeddings")
    probabilities = np.zeros((len(dense_times), 4), dtype=np.float32)
    availability = np.zeros_like(probabilities, dtype=np.bool_)
    sides = np.where(pivot["directions"] == 1, 0, 1)
    for row, side, pivot_score, expansion_score in zip(
        rows,
        sides,
        pivot["pivot_probabilities"],
        expansion["launch_probabilities"],
        strict=True,
    ):
        # Multiple causal candidates can resolve at one bar. The most confident
        # same-side teacher value is deterministic and preserves availability.
        probabilities[row, side] = max(probabilities[row, side], float(pivot_score))
        probabilities[row, side + 2] = max(
            probabilities[row, side + 2], float(expansion_score)
        )
        availability[row, side] = True
        availability[row, side + 2] = True
    source_hashes = {
        channels[0]: _sha256(sources[0]),
        channels[1]: _sha256(sources[0]),
        channels[2]: _sha256(sources[1]),
        channels[3]: _sha256(sources[1]),
    }
    return write_teacher_signal_cache(
        destination=destination,
        ticker=ticker,
        timestamps=dense_times,
        probabilities=probabilities,
        availability=availability,
        channels=channels,
        source_artifact_sha256s=source_hashes,
        research_end_exclusive=research_end_exclusive,
    )
