"""Portable, authenticated Expansion-teacher scoring over frozen Mask caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .agent import resolve_device
from .cache import EmbeddingCache


TEACHER_CACHE_SCHEMA = "propevolve_expansion_teacher_cache_v1"
CHANNELS = (
    "long_attempt_probability",
    "long_clean_retained_given_attempt_probability",
    "short_attempt_probability",
    "short_clean_retained_given_attempt_probability",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    value = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _utc_boundary(value: str) -> np.datetime64:
    parsed = pd.Timestamp(value)
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return np.datetime64(parsed.tz_localize(None), "ns")


@dataclass(frozen=True)
class _ExpansionConfig:
    embedding_dim: int
    model_dim: int
    num_heads: int
    feedforward_dim: int
    num_layers: int
    dropout: float
    context_length: int
    precursor_horizons: tuple[int, ...]

    @classmethod
    def from_payload(cls, payload: dict) -> "_ExpansionConfig":
        values = dict(payload)
        values["precursor_horizons"] = tuple(values["precursor_horizons"])
        config = cls(**values)
        if (
            config.context_length != 50
            or config.precursor_horizons != (5, 10, 20, 50)
            or config.embedding_dim < 1
            or config.model_dim % config.num_heads
        ):
            raise ValueError("Expansion model configuration is unsupported")
        return config


class _FrozenInputLayerNorm(nn.LayerNorm):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            values, self.normalized_shape, weight=None, bias=None, eps=self.eps
        )
        return normalized * self.weight + self.bias


class _DirectionalExpansionExpert(nn.Module):
    def __init__(self, config: _ExpansionConfig):
        super().__init__()
        self.config = config
        self.input_norm = _FrozenInputLayerNorm(config.embedding_dim)
        self.input_projection = nn.Linear(config.embedding_dim, config.model_dim)
        self.age_embedding = nn.Embedding(config.context_length, config.model_dim)
        self.horizon_embedding = nn.Embedding(
            len(config.precursor_horizons), config.model_dim
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.view_projection = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                config.model_dim * len(config.precursor_horizons),
                config.model_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.attempted_expansion_head = nn.Linear(config.model_dim, 1)
        self.retained_clean_given_attempt_head = nn.Linear(config.model_dim, 1)
        for view_index, horizon in enumerate(config.precursor_horizons):
            positions = torch.arange(horizon, dtype=torch.long)[None, :]
            self.register_buffer(
                f"_age_index_{view_index}",
                horizon - 1 - positions,
                persistent=False,
            )
            self.register_buffer(
                f"_view_index_{view_index}",
                torch.full_like(positions, view_index),
                persistent=False,
            )
            self.register_buffer(
                f"_causal_mask_{view_index}",
                torch.triu(torch.ones((horizon, horizon), dtype=torch.bool), diagonal=1),
                persistent=False,
            )

    def _suffix(
        self,
        projected: torch.Tensor,
        horizon: int,
        view_index: int,
    ) -> torch.Tensor:
        view = projected[:, -horizon:]
        ages = getattr(self, f"_age_index_{view_index}")
        view_ids = getattr(self, f"_view_index_{view_index}")
        return view + self.age_embedding(ages) + self.horizon_embedding(view_ids)

    def forward(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(self.input_norm(trajectory))
        summaries = []
        for view_index, horizon in enumerate(self.config.precursor_horizons):
            view = self._suffix(projected, horizon, view_index)
            encoded = self.temporal_encoder(
                view,
                mask=getattr(self, f"_causal_mask_{view_index}"),
                is_causal=True,
            )
            summaries.append(
                self.view_projection(
                    torch.cat((encoded[:, -1], encoded.mean(dim=1)), dim=-1)
                )
            )
        fused = self.fusion(torch.cat(summaries, dim=-1))
        return (
            self.attempted_expansion_head(fused).squeeze(-1),
            self.retained_clean_given_attempt_head(fused).squeeze(-1),
        )


class _ExpansionModel(nn.Module):
    def __init__(self, config: _ExpansionConfig):
        super().__init__()
        self.config = config
        self.long_expert = _DirectionalExpansionExpert(config)
        self.short_expert = _DirectionalExpansionExpert(config)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        long_attempt, long_clean = self.long_expert(trajectory)
        short_attempt, short_clean = self.short_expert(trajectory)
        return torch.stack(
            (long_attempt, long_clean, short_attempt, short_clean), dim=-1
        )


@dataclass(frozen=True)
class ExpansionTeacher:
    """Verified Expansion artifact with stream-specific calibrated scoring."""

    model: _ExpansionModel
    device: torch.device
    channels: tuple[str, ...]
    stream_names: tuple[str, ...]
    calibration_scales: torch.Tensor
    calibration_biases: torch.Tensor
    manifest: dict
    checkpoint_sha256: str
    context_length: int
    embedding_dim: int

    @classmethod
    def load(cls, manifest_path: str | Path, *, device: str) -> "ExpansionTeacher":
        manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        expansion = manifest.get("expansion")
        if (
            manifest.get("schema") != "propevolve_verified_teacher_assets_v1"
            or manifest.get("status") != "verified_checkpoint_only"
            or manifest.get("pivot") is not None
            or not isinstance(expansion, dict)
        ):
            raise ValueError("verified Expansion teacher manifest is invalid")
        checkpoint = manifest_path.parent / str(expansion.get("checkpoint", ""))
        checkpoint_sha256 = _sha256(checkpoint)
        if checkpoint_sha256 != expansion.get("checkpoint_sha256"):
            raise ValueError("Expansion teacher checkpoint identity drifted")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != expansion.get("artifact_schema")
            or payload.get("training_schema") != expansion.get("training_schema")
            or not isinstance(payload.get("state_dict"), dict)
            or not isinstance(payload.get("calibrator"), dict)
        ):
            raise ValueError("Expansion teacher artifact contract drifted")
        lineage = payload.get("lineage") or {}
        if (
            lineage.get("encoder_sha256")
            != expansion.get("encoder_checkpoint_sha256")
            or lineage.get("encoder_identity_sha256")
            != expansion.get("encoder_identity_sha256")
            or lineage.get("dataset_lineage_sha256")
            != expansion.get("dataset_lineage_sha256")
            or set(lineage.get("cache_sha256s", ()))
            != set((expansion.get("cache_sha256s") or {}).values())
        ):
            raise ValueError("Expansion teacher lineage does not match its caches")
        calibration = payload["calibrator"]
        parameters = dict(calibration)
        reported_parameters_sha256 = parameters.pop("parameters_sha256", None)
        if (
            reported_parameters_sha256 != payload.get("calibrator_sha256")
            or _canonical_sha256(parameters) != reported_parameters_sha256
            or calibration.get("side_order") != ["long", "short"]
            or calibration.get("factor_order")
            != ["attempted", "clean_given_attempt"]
        ):
            raise ValueError("Expansion teacher calibration identity drifted")
        streams = calibration.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ValueError("Expansion teacher calibration lacks streams")
        names, scales, biases = [], [], []
        for stream_id, stream in enumerate(streams):
            if stream.get("stream_id") != stream_id:
                raise ValueError("Expansion teacher stream ordering drifted")
            names.append(str(stream["stream_name"]))
            stream_scales, stream_biases = [], []
            for side in ("long", "short"):
                for factor in ("attempted", "clean_given_attempt"):
                    values = stream["sides"][side][factor]
                    stream_scales.append(float(values["scale"]))
                    stream_biases.append(float(values["bias"]))
            scales.append(stream_scales)
            biases.append(stream_biases)
        expected_streams = {f"{ticker}@3min" for ticker in expansion["universe"]}
        if set(names) != expected_streams:
            raise ValueError("Expansion teacher calibrated universe drifted")
        resolved_device = resolve_device(device)
        config = _ExpansionConfig.from_payload(payload["model_config"])
        model = _ExpansionModel(config)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(resolved_device).eval()
        return cls(
            model=model,
            device=resolved_device,
            channels=CHANNELS,
            stream_names=tuple(names),
            calibration_scales=torch.as_tensor(
                scales, dtype=torch.float32, device=resolved_device
            ),
            calibration_biases=torch.as_tensor(
                biases, dtype=torch.float32, device=resolved_device
            ),
            manifest=manifest,
            checkpoint_sha256=checkpoint_sha256,
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
        )

    def _stream_id(self, ticker: str) -> int:
        name = f"{ticker}@3min"
        try:
            return self.stream_names.index(name)
        except ValueError as exc:
            raise ValueError(f"Expansion teacher does not support {ticker}@3min") from exc

    def _score_trusted(self, trajectory: torch.Tensor, *, ticker: str) -> torch.Tensor:
        logits = self.model(trajectory.to(self.device, dtype=torch.float32))
        stream_id = self._stream_id(ticker)
        return torch.sigmoid(
            logits * self.calibration_scales[stream_id]
            + self.calibration_biases[stream_id]
        )

    def score(self, trajectory: np.ndarray, *, ticker: str) -> np.ndarray:
        values = np.asarray(trajectory)
        expected = (self.context_length, self.embedding_dim)
        if (
            values.ndim != 3
            or tuple(values.shape[1:]) != expected
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Expansion teacher trajectories must be finite [rows,{expected[0]},{expected[1]}]"
            )
        with torch.inference_mode():
            result = self._score_trusted(
                torch.from_numpy(np.ascontiguousarray(values)), ticker=ticker
            )
        return result.cpu().numpy()


@dataclass(frozen=True)
class ExpansionTeacherCache:
    root: Path
    probabilities: np.ndarray
    availability: np.ndarray
    timestamps: np.ndarray
    manifest: dict

    @classmethod
    def load(cls, root: str | Path) -> "ExpansionTeacherCache":
        root = Path(root).resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != TEACHER_CACHE_SCHEMA
            or tuple(manifest.get("channels", ())) != CHANNELS
            or manifest.get("selection_rows_included") is not False
            or manifest.get("sealed_rows_included") is not False
        ):
            raise ValueError("Expansion teacher cache contract is invalid")
        paths = {
            name: root / f"{name}.npy"
            for name in ("probabilities", "availability", "timestamps")
        }
        for name, path in paths.items():
            if _sha256(path) != manifest.get(f"{name}_sha256"):
                raise ValueError(f"Expansion teacher cache {name} identity drifted")
        probabilities = np.load(paths["probabilities"], mmap_mode="r")
        availability = np.load(paths["availability"], mmap_mode="r")
        timestamps = np.load(paths["timestamps"], mmap_mode="r")
        rows = int(manifest.get("rows", -1))
        if (
            probabilities.shape != (rows, len(CHANNELS))
            or probabilities.dtype != np.float32
            or availability.shape != (rows,)
            or availability.dtype != np.bool_
            or timestamps.shape != (rows,)
            or np.isnat(timestamps).any()
            or (len(timestamps) > 1 and not np.all(timestamps[1:] > timestamps[:-1]))
            or np.any(timestamps >= _utc_boundary(manifest["training_end_exclusive"]))
            or not np.isfinite(probabilities[availability]).all()
        ):
            raise ValueError("Expansion teacher cache arrays are invalid")
        return cls(root, probabilities, availability, timestamps, manifest)


def _flush_scores(
    pending: list[torch.Tensor],
    probabilities: np.memmap,
    *,
    destination_start: int,
) -> int:
    if not pending:
        return destination_start
    values = torch.cat(pending, dim=0).cpu().numpy().astype(np.float32, copy=False)
    destination_end = destination_start + len(values)
    probabilities[destination_start:destination_end] = values
    pending.clear()
    return destination_end


def _validate_source_receipt(
    embedding_cache: Path,
    manifest: dict,
    *,
    expected_cache_identity_sha256: str,
    expected_embeddings_sha256: str,
) -> None:
    source_manifest_value = manifest.get("imported_ffm_cache_manifest")
    if (
        manifest.get("imported_ffm_embeddings_sha256")
        != expected_embeddings_sha256
        or not source_manifest_value
    ):
        raise ValueError("Expansion teacher embedding identity mismatch")
    source_manifest = Path(str(source_manifest_value))
    validation_path = source_manifest.with_name(
        source_manifest.name.removesuffix(".manifest.json") + ".validation.json"
    )
    if not validation_path.is_file():
        raise ValueError("Expansion teacher source validation receipt is missing")
    receipt = json.loads(validation_path.read_text())
    current = os.stat(embedding_cache / "embeddings.npy")
    expected_stat = receipt.get("embedding_stat") or {}
    if (
        receipt.get("schema") != "pivot-frozen-representation-validation-v1"
        or receipt.get("cache_identity_sha256") != expected_cache_identity_sha256
        or receipt.get("embeddings_sha256") != expected_embeddings_sha256
        or receipt.get("finite") is not True
        or int(expected_stat.get("device", -1)) != current.st_dev
        or int(expected_stat.get("inode", -1)) != current.st_ino
        or int(expected_stat.get("mtime_ns", -1)) != current.st_mtime_ns
        or int(expected_stat.get("size", -1)) != current.st_size
    ):
        raise ValueError("Expansion teacher source validation receipt drifted")


def _load_validated_embedding_source(
    *,
    teacher: ExpansionTeacher,
    embedding_cache: str | Path,
    ticker: str,
    expected_cache_identity_sha256: str,
) -> EmbeddingCache:
    source = EmbeddingCache.load(embedding_cache)
    if source.manifest.get("ticker") != ticker:
        raise ValueError("Expansion teacher ticker does not match embedding cache")
    expansion = teacher.manifest["expansion"]
    if (
        source.manifest.get("encoder_identity_sha256")
        != expansion["encoder_identity_sha256"]
        or source.manifest.get("imported_ffm_cache_identity_sha256")
        != expected_cache_identity_sha256
        or expansion["cache_identity_sha256s"].get(ticker)
        != expected_cache_identity_sha256
    ):
        raise ValueError("Expansion teacher cache identity mismatch")
    _validate_source_receipt(
        Path(embedding_cache),
        source.manifest,
        expected_cache_identity_sha256=expected_cache_identity_sha256,
        expected_embeddings_sha256=expansion["embeddings_sha256s"][ticker],
    )
    if source.embeddings.shape[1] != teacher.embedding_dim:
        raise ValueError("Expansion teacher embedding width mismatch")
    return source


def build_expansion_teacher_cache(
    *,
    teacher: ExpansionTeacher,
    embedding_cache: str | Path,
    destination: str | Path,
    ticker: str,
    training_end_exclusive: str,
    expected_cache_identity_sha256: str,
    batch_size: int,
    synchronization_batches: int,
    training_start_inclusive: str | None = None,
    progress_batches: int = 256,
) -> Path:
    """Score one ticker atomically; an authenticated existing result is a HIT."""
    if min(batch_size, synchronization_batches, progress_batches) < 1:
        raise ValueError("Expansion teacher batch settings must be positive")
    source = _load_validated_embedding_source(
        teacher=teacher,
        embedding_cache=embedding_cache,
        ticker=ticker,
        expected_cache_identity_sha256=expected_cache_identity_sha256,
    )
    expansion = teacher.manifest["expansion"]
    timestamps = np.asarray(source.timestamps)
    end = _utc_boundary(training_end_exclusive)
    sealed = _utc_boundary(source.manifest["research_end_exclusive"])
    if end > sealed:
        raise ValueError("Expansion teacher training boundary crosses sealed data")
    start = (
        _utc_boundary(training_start_inclusive)
        if training_start_inclusive is not None
        else timestamps[0]
    )
    first = int(np.searchsorted(timestamps, start, side="left"))
    stop = int(np.searchsorted(timestamps, end, side="left"))
    if stop <= first or stop < teacher.context_length:
        raise ValueError("Expansion teacher cache has insufficient training history")
    selected_timestamps = timestamps[first:stop].astype("datetime64[ns]", copy=True)
    rows = len(selected_timestamps)
    input_manifest_sha256 = _sha256(Path(embedding_cache) / "manifest.json")
    expected_contract = {
        "schema": TEACHER_CACHE_SCHEMA,
        "ticker": ticker,
        "channels": list(CHANNELS),
        "rows": rows,
        "context_length": teacher.context_length,
        "embedding_dim": teacher.embedding_dim,
        "training_start_inclusive": str(training_start_inclusive or timestamps[0]),
        "training_end_exclusive": str(training_end_exclusive),
        "selection_rows_included": False,
        "sealed_rows_included": False,
        "checkpoint_sha256": teacher.checkpoint_sha256,
        "encoder_identity_sha256": expansion["encoder_identity_sha256"],
        "source_cache_identity_sha256": expected_cache_identity_sha256,
        "source_manifest_sha256": input_manifest_sha256,
    }
    destination = Path(destination)
    if destination.is_dir():
        existing = ExpansionTeacherCache.load(destination)
        if any(existing.manifest.get(key) != value for key, value in expected_contract.items()):
            raise ValueError("existing Expansion teacher cache identity conflicts")
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        probabilities = np.lib.format.open_memmap(
            temporary / "probabilities.npy",
            mode="w+",
            dtype=np.float32,
            shape=(rows, len(CHANNELS)),
        )
        probabilities[:] = 0.0
        availability = np.lib.format.open_memmap(
            temporary / "availability.npy", mode="w+", dtype=np.bool_, shape=(rows,)
        )
        availability[:] = False
        np.save(temporary / "timestamps.npy", selected_timestamps)
        first_endpoint = max(first, teacher.context_length - 1)
        windows = np.lib.stride_tricks.sliding_window_view(
            source.embeddings[:stop],
            window_shape=teacher.context_length,
            axis=0,
        ).transpose(0, 2, 1)
        window_start = first_endpoint - (teacher.context_length - 1)
        window_stop = stop - (teacher.context_length - 1)
        pending: list[torch.Tensor] = []
        output_start = first_endpoint - first
        output_cursor = output_start
        total_batches = (window_stop - window_start + batch_size - 1) // batch_size
        with torch.inference_mode():
            for batch_number, batch_start in enumerate(
                range(window_start, window_stop, batch_size), start=1
            ):
                batch_stop = min(batch_start + batch_size, window_stop)
                batch = np.ascontiguousarray(windows[batch_start:batch_stop])
                pending.append(
                    teacher._score_trusted(torch.from_numpy(batch), ticker=ticker)
                )
                if (
                    len(pending) >= synchronization_batches
                    or batch_number == total_batches
                ):
                    output_cursor = _flush_scores(
                        pending, probabilities, destination_start=output_cursor
                    )
                if batch_number % progress_batches == 0:
                    print(
                        f"[expansion-teacher] {ticker} batch={batch_number}/{total_batches}",
                        flush=True,
                    )
        if output_cursor != rows:
            raise RuntimeError("Expansion teacher output row accounting drifted")
        availability[output_start:] = True
        probabilities.flush()
        availability.flush()
        del probabilities, availability
        artifact_hashes = {
            f"{name}_sha256": _sha256(temporary / f"{name}.npy")
            for name in ("probabilities", "availability", "timestamps")
        }
        manifest = {
            **expected_contract,
            **artifact_hashes,
            "available_rows": rows - output_start,
            "device": str(teacher.device),
            "batch_size": int(batch_size),
            "synchronization_batches": int(synchronization_batches),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        ExpansionTeacherCache.load(temporary)
        temporary.rename(destination)
    return destination.resolve()


def load_builder_config(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text())
    required = {
        "schema", "teacher_manifest", "embedding_cache_root", "output_root",
        "tickers", "training_start_inclusive", "training_end_exclusive",
        "sealed_start", "device", "batch_size", "fallback_batch_sizes",
        "synchronization_batches", "progress_batches",
    }
    if set(payload) != required or payload.get("schema") != "expansion_teacher_cache_build_v1":
        raise ValueError("Expansion teacher builder configuration is invalid")
    tickers = tuple(str(value) for value in payload["tickers"])
    batches = (int(payload["batch_size"]), *(int(v) for v in payload["fallback_batch_sizes"]))
    if (
        not tickers
        or len(set(tickers)) != len(tickers)
        or any(value < 1 for value in batches)
        or len(set(batches)) != len(batches)
        or not _utc_boundary(payload["training_start_inclusive"])
        < _utc_boundary(payload["training_end_exclusive"])
        < _utc_boundary(payload["sealed_start"])
    ):
        raise ValueError("Expansion teacher builder ranges are invalid")
    payload["tickers"] = tickers
    payload["batch_sizes"] = batches
    payload["_path"] = str(path)
    payload["_root"] = str(path.parent.parent)
    return payload


def build_expansion_teacher_caches(
    config_path: str | Path,
    *,
    requested_tickers: Sequence[str] = (),
) -> tuple[Path, ...]:
    config = load_builder_config(config_path)
    root = Path(config["_root"])
    tickers = tuple(requested_tickers) or config["tickers"]
    if not set(tickers) <= set(config["tickers"]):
        raise ValueError("requested Expansion teacher ticker is outside the contract")
    teacher = ExpansionTeacher.load(
        root / config["teacher_manifest"], device=config["device"]
    )
    expansion = teacher.manifest["expansion"]
    for ticker in tickers:
        _load_validated_embedding_source(
            teacher=teacher,
            embedding_cache=root / config["embedding_cache_root"] / ticker,
            ticker=ticker,
            expected_cache_identity_sha256=(
                expansion["cache_identity_sha256s"][ticker]
            ),
        )
    print(
        f"[expansion-teacher] PREFLIGHT COMPLETE tickers={len(tickers)}",
        flush=True,
    )
    results = []
    for ticker in tickers:
        destination = root / config["output_root"] / ticker
        last_error = None
        for batch_size in config["batch_sizes"]:
            try:
                result = build_expansion_teacher_cache(
                    teacher=teacher,
                    embedding_cache=root / config["embedding_cache_root"] / ticker,
                    destination=destination,
                    ticker=ticker,
                    training_start_inclusive=config["training_start_inclusive"],
                    training_end_exclusive=config["training_end_exclusive"],
                    expected_cache_identity_sha256=(
                        expansion["cache_identity_sha256s"][ticker]
                    ),
                    batch_size=batch_size,
                    synchronization_batches=int(config["synchronization_batches"]),
                    progress_batches=int(config["progress_batches"]),
                )
                print(f"[expansion-teacher] COMPLETE {ticker} {result}", flush=True)
                results.append(result)
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" not in message or batch_size == config["batch_sizes"][-1]:
                    raise
                last_error = exc
                if teacher.device.type == "mps":
                    torch.mps.empty_cache()
                print(
                    f"[expansion-teacher] RETRY {ticker} batch_size={batch_size} reason=oom",
                    flush=True,
                )
        else:  # pragma: no cover - defensive; loop either appends or raises
            raise RuntimeError("Expansion teacher batch fallbacks exhausted") from last_error
    return tuple(results)


__all__ = [
    "CHANNELS",
    "ExpansionTeacher",
    "ExpansionTeacherCache",
    "build_expansion_teacher_cache",
    "build_expansion_teacher_caches",
    "load_builder_config",
]
