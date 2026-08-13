"""Portable, authenticated Regime-teacher inference over frozen Mask caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..agent import resolve_device
from ..balance_aware_regime_selectivity import REGIME_TEACHER_CHANNELS
from ..cache import EmbeddingCache
from .base import BaseTeacher
from .expansion import _flush_scores, _validate_source_receipt


TEACHER_CACHE_SCHEMA = "propevolve_regime_teacher_cache_v1"
MODEL_SCHEMA = "frozen-mask-structure-volatility-regime-teacher-v2"
TRAINING_SCHEMA = "structure-volatility-regime-teacher-training-v2"
CONTEXT_LENGTH = 50
SUFFIX_LOOKBACKS = (5, 10, 20, 50)
CHANNELS = REGIME_TEACHER_CHANNELS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_boundary(value: str) -> np.datetime64:
    parsed = pd.Timestamp(value)
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return np.datetime64(parsed.tz_localize(None), "ns")


@dataclass(frozen=True)
class _RegimeConfig:
    embedding_dim: int
    model_dim: int
    num_heads: int
    feedforward_dim: int
    num_layers: int
    dropout: float
    context_length: int
    suffix_lookbacks: tuple[int, ...]

    @classmethod
    def from_payload(cls, payload: dict) -> "_RegimeConfig":
        values = dict(payload)
        values["suffix_lookbacks"] = tuple(values["suffix_lookbacks"])
        config = cls(**values)
        if (
            config.context_length != CONTEXT_LENGTH
            or config.suffix_lookbacks != SUFFIX_LOOKBACKS
            or config.embedding_dim < 1
            or config.model_dim < 1
            or config.num_heads < 1
            or config.model_dim % config.num_heads
        ):
            raise ValueError("Regime model configuration is unsupported")
        return config


class _RegimeModel(nn.Module):
    def __init__(self, config: _RegimeConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.embedding_dim)
        self.input_projection = nn.Linear(config.embedding_dim, config.model_dim)
        self.age_embedding = nn.Embedding(config.context_length, config.model_dim)
        self.view_embedding = nn.Embedding(len(config.suffix_lookbacks), config.model_dim)
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
            nn.Linear(config.model_dim * len(config.suffix_lookbacks), config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.state_head = nn.Linear(config.model_dim, 3)
        self.transition_head = nn.Linear(config.model_dim, 5)
        self.efficiency_head = nn.Linear(config.model_dim, 1)
        self.volatility_state_head = nn.Linear(config.model_dim, 3)
        self.volatility_transition_head = nn.Linear(config.model_dim, 5)
        self.volatility_percentile_head = nn.Linear(config.model_dim, 1)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        lengths = torch.full(
            (len(trajectory),), self.config.context_length,
            dtype=torch.long, device=trajectory.device,
        )
        projected = self.input_projection(self.input_norm(trajectory))
        summaries = []
        for view_index, lookback in enumerate(self.config.suffix_lookbacks):
            positions = torch.arange(lookback, device=trajectory.device)[None, :]
            starts = lengths - lookback
            source = starts[:, None] + positions
            view = projected.gather(
                1, source[:, :, None].expand(-1, -1, projected.shape[-1])
            )
            ages = lookback - 1 - positions
            view = view + self.age_embedding(ages) + self.view_embedding(
                torch.full_like(positions, view_index)
            )
            causal_mask = torch.triu(
                torch.ones(lookback, lookback, dtype=torch.bool, device=trajectory.device),
                diagonal=1,
            )
            encoded = self.temporal_encoder(view, mask=causal_mask, is_causal=True)
            summaries.append(
                self.view_projection(
                    torch.cat((encoded[:, -1], encoded.mean(dim=1)), dim=-1)
                )
            )
        fused = self.fusion(torch.cat(summaries, dim=-1))
        return torch.cat((
            self.state_head(fused).softmax(-1),
            self.transition_head(fused).softmax(-1),
            torch.sigmoid(self.efficiency_head(fused)),
            self.volatility_state_head(fused).softmax(-1),
            self.volatility_transition_head(fused).softmax(-1),
            torch.sigmoid(self.volatility_percentile_head(fused)),
        ), dim=-1)


@dataclass(frozen=True)
class RegimeTeacher(BaseTeacher):
    model: _RegimeModel

    kind: ClassVar[str] = "regime"
    channels: ClassVar[tuple[str, ...]] = CHANNELS

    @classmethod
    def load(cls, manifest_path: str | Path, *, device: str) -> "RegimeTeacher":
        manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        regime = manifest.get("regime")
        if (
            manifest.get("schema") != "propevolve_verified_teacher_assets_v1"
            or manifest.get("status") != "verified_checkpoint_only"
            or not isinstance(regime, dict)
        ):
            raise ValueError("verified Regime teacher manifest is invalid")
        checkpoint = manifest_path.parent / str(regime.get("checkpoint", ""))
        checkpoint_sha256 = _sha256(checkpoint)
        if checkpoint_sha256 != regime.get("checkpoint_sha256"):
            raise ValueError("Regime teacher checkpoint identity drifted")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != MODEL_SCHEMA
            or payload.get("training_schema") != TRAINING_SCHEMA
            or not isinstance(payload.get("state_dict"), dict)
            or payload.get("cache_identities") != regime.get("cache_identity_sha256s")
            or payload.get("temporal", {}).get("sealed_start") != "2026-01-01"
            or tuple(regime.get("channels", ())) != CHANNELS
        ):
            raise ValueError("Regime teacher artifact contract drifted")
        config = _RegimeConfig.from_payload(payload["model_config"])
        model = _RegimeModel(config)
        model.load_state_dict(payload["state_dict"], strict=True)
        resolved_device = resolve_device(device)
        model.to(resolved_device).eval()
        return cls(
            model=model,
            device=resolved_device,
            manifest=manifest,
            checkpoint_sha256=checkpoint_sha256,
            context_length=config.context_length,
            embedding_dim=config.embedding_dim,
        )

    def _score_trusted(
        self,
        trajectory: torch.Tensor,
        *,
        ticker: str | None = None,
    ) -> torch.Tensor:
        return self.model(trajectory.to(self.device, dtype=torch.float32))


@dataclass(frozen=True)
class RegimeTeacherCache:
    root: Path
    probabilities: np.ndarray
    availability: np.ndarray
    timestamps: np.ndarray
    manifest: dict

    @classmethod
    def load(cls, root: str | Path) -> "RegimeTeacherCache":
        root = Path(root).resolve(strict=True)
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            manifest.get("schema") != TEACHER_CACHE_SCHEMA
            or tuple(manifest.get("channels", ())) != CHANNELS
            or manifest.get("selection_rows_included") is not False
            or manifest.get("sealed_rows_included") is not False
        ):
            raise ValueError("Regime teacher cache contract is invalid")
        paths = {
            name: root / f"{name}.npy"
            for name in ("probabilities", "availability", "timestamps")
        }
        for name, path in paths.items():
            if _sha256(path) != manifest.get(f"{name}_sha256"):
                raise ValueError(f"Regime teacher cache {name} identity drifted")
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
            raise ValueError("Regime teacher cache arrays are invalid")
        return cls(root, probabilities, availability, timestamps, manifest)


@dataclass(frozen=True)
class RegimeTeacherTargets:
    probabilities: dict[str, np.ndarray]
    availability: dict[str, np.ndarray]

    @classmethod
    def load(cls, cache_root: str | Path, markets: dict[str, object]) -> "RegimeTeacherTargets":
        root = Path(cache_root)
        probabilities: dict[str, np.ndarray] = {}
        availability: dict[str, np.ndarray] = {}
        for ticker, market in markets.items():
            cache = RegimeTeacherCache.load(root / ticker)
            market_timestamps = np.asarray(getattr(market, "timestamps"))
            indices = np.searchsorted(cache.timestamps, market_timestamps)
            if (
                (indices >= len(cache.timestamps)).any()
                or not np.array_equal(cache.timestamps[indices], market_timestamps)
            ):
                raise ValueError(f"Regime teacher rows do not align to training market {ticker}")
            probabilities[ticker] = cache.probabilities[indices]
            availability[ticker] = cache.availability[indices]
        return cls(probabilities=probabilities, availability=availability)

    def target(self, ticker: str, row: int) -> np.ndarray | None:
        if ticker not in self.probabilities:
            raise ValueError(f"Regime teacher has no aligned market {ticker}")
        if row < 0 or row >= len(self.probabilities[ticker]):
            raise IndexError("Regime teacher row is outside the aligned market")
        if not bool(self.availability[ticker][row]):
            return None
        return np.asarray(self.probabilities[ticker][row], dtype=np.float32)


def _load_validated_embedding_source(
    *,
    teacher: RegimeTeacher,
    embedding_cache: str | Path,
    ticker: str,
    expected_cache_identity_sha256: str,
) -> EmbeddingCache:
    source = EmbeddingCache.load(embedding_cache)
    regime = teacher.manifest["regime"]
    if (
        source.manifest.get("ticker") != ticker
        or source.manifest.get("encoder_identity_sha256")
        != regime["encoder_identity_sha256"]
        or source.manifest.get("imported_ffm_cache_identity_sha256")
        != expected_cache_identity_sha256
        or regime["cache_identity_sha256s"].get(ticker)
        != expected_cache_identity_sha256
    ):
        raise ValueError("Regime teacher cache identity mismatch")
    _validate_source_receipt(
        Path(embedding_cache),
        source.manifest,
        expected_cache_identity_sha256=expected_cache_identity_sha256,
        expected_embeddings_sha256=regime["embeddings_sha256s"][ticker],
    )
    if source.embeddings.shape[1] != teacher.embedding_dim:
        raise ValueError("Regime teacher embedding width mismatch")
    return source


def build_regime_teacher_cache(
    *,
    teacher: RegimeTeacher,
    embedding_cache: str | Path,
    destination: str | Path,
    ticker: str,
    training_end_exclusive: str,
    expected_cache_identity_sha256: str,
    batch_size: int,
    synchronization_batches: int,
    progress_batches: int = 100,
) -> Path:
    if min(batch_size, synchronization_batches, progress_batches) < 1:
        raise ValueError("Regime teacher batch settings must be positive")
    source = _load_validated_embedding_source(
        teacher=teacher,
        embedding_cache=embedding_cache,
        ticker=ticker,
        expected_cache_identity_sha256=expected_cache_identity_sha256,
    )
    regime = teacher.manifest["regime"]
    boundary = _utc_boundary(training_end_exclusive)
    if boundary >= _utc_boundary("2026-01-01"):
        raise ValueError("Regime teacher training boundary crosses sealed data")
    first = int(np.searchsorted(source.timestamps, _utc_boundary("2021-01-01")))
    stop = int(np.searchsorted(source.timestamps, boundary))
    if stop <= first or stop < teacher.context_length:
        raise ValueError("Regime teacher cache has insufficient training history")
    selected_timestamps = source.timestamps[first:stop]
    rows = len(selected_timestamps)
    expected_contract = {
        "schema": TEACHER_CACHE_SCHEMA,
        "channels": list(CHANNELS),
        "rows": rows,
        "ticker": ticker,
        "training_start_inclusive": "2021-01-01",
        "training_end_exclusive": training_end_exclusive,
        "selection_rows_included": False,
        "sealed_rows_included": False,
        "checkpoint_sha256": teacher.checkpoint_sha256,
        "embedding_cache_identity_sha256": expected_cache_identity_sha256,
        "encoder_identity_sha256": regime["encoder_identity_sha256"],
        "context_length": teacher.context_length,
        "embedding_dim": teacher.embedding_dim,
    }
    destination = Path(destination)
    if destination.exists():
        existing = RegimeTeacherCache.load(destination)
        if any(existing.manifest.get(key) != value for key, value in expected_contract.items()):
            raise ValueError("existing Regime teacher cache identity conflicts")
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_value:
        temporary = Path(temporary_value)
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
            source.embeddings[:stop], window_shape=teacher.context_length, axis=0
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
                pending.append(teacher._score_trusted(torch.from_numpy(batch)))
                if len(pending) >= synchronization_batches or batch_number == total_batches:
                    output_cursor = _flush_scores(
                        pending, probabilities, destination_start=output_cursor
                    )
                if batch_number % progress_batches == 0:
                    print(
                        f"[regime-teacher] {ticker} batch={batch_number}/{total_batches}",
                        flush=True,
                    )
        if output_cursor != rows:
            raise RuntimeError("Regime teacher output row accounting drifted")
        availability[output_start:] = True
        probabilities.flush(); availability.flush()
        del probabilities, availability
        manifest = {
            **expected_contract,
            **{
                f"{name}_sha256": _sha256(temporary / f"{name}.npy")
                for name in ("probabilities", "availability", "timestamps")
            },
            "available_rows": rows - output_start,
            "device": str(teacher.device),
            "batch_size": int(batch_size),
            "synchronization_batches": int(synchronization_batches),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        RegimeTeacherCache.load(temporary)
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
    if set(payload) != required or payload.get("schema") != "regime_teacher_cache_build_v1":
        raise ValueError("Regime teacher builder configuration is invalid")
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
        raise ValueError("Regime teacher builder ranges are invalid")
    payload["tickers"] = tickers
    payload["batch_sizes"] = batches
    payload["_path"] = str(path)
    payload["_root"] = str(path.parent.parent)
    return payload


def build_regime_teacher_caches(
    config_path: str | Path,
    *,
    requested_tickers: Sequence[str] = (),
) -> tuple[Path, ...]:
    config = load_builder_config(config_path)
    root = Path(config["_root"])
    tickers = tuple(requested_tickers) or config["tickers"]
    if not set(tickers) <= set(config["tickers"]):
        raise ValueError("requested Regime teacher ticker is outside the contract")
    teacher = RegimeTeacher.load(root / config["teacher_manifest"], device=config["device"])
    regime = teacher.manifest["regime"]
    results = []
    for ticker in tickers:
        destination = root / config["output_root"] / ticker
        last_error = None
        for batch_size in config["batch_sizes"]:
            try:
                result = build_regime_teacher_cache(
                    teacher=teacher,
                    embedding_cache=root / config["embedding_cache_root"] / ticker,
                    destination=destination,
                    ticker=ticker,
                    training_end_exclusive=config["training_end_exclusive"],
                    expected_cache_identity_sha256=regime["cache_identity_sha256s"][ticker],
                    batch_size=batch_size,
                    synchronization_batches=int(config["synchronization_batches"]),
                    progress_batches=int(config["progress_batches"]),
                )
                print(f"[regime-teacher] COMPLETE {ticker} {result}", flush=True)
                results.append(result)
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch_size == config["batch_sizes"][-1]:
                    raise
                last_error = exc
                if teacher.device.type == "mps":
                    torch.mps.empty_cache()
        else:  # pragma: no cover
            raise RuntimeError("Regime teacher batch fallbacks exhausted") from last_error
    return tuple(results)


__all__ = [
    "CHANNELS",
    "RegimeTeacher",
    "RegimeTeacherCache",
    "RegimeTeacherTargets",
    "build_regime_teacher_cache",
    "build_regime_teacher_caches",
    "load_builder_config",
]
