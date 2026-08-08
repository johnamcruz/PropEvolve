"""Adapter to the installed FFM package's frozen Chronos2 embedder."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable, Iterator
from collections import deque

import numpy as np


EmbeddingChunks = Callable[..., Iterable[np.ndarray]]


def _load_package_embedder() -> EmbeddingChunks:
    try:
        from futures_foundation.finetune.classifiers.chronos2._embed_worker_fast import (
            embed_window_chunks,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install futures-foundation-model[foundation] in the PropEvolve "
            "environment; FFM source is not vendored by this repository"
        ) from exc
    return embed_window_chunks


class FrozenChronos2Encoder:
    """Encode finite OHLCV windows through one immutable FFM Mask adapter."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "mps",
        batch_series: int = 512,
        fast_group_attention: bool = False,
        package_embedder: EmbeddingChunks | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        weights = self.checkpoint / "adapter_model.safetensors"
        config_path = self.checkpoint / "adapter_config.json"
        if not weights.is_file() or not config_path.is_file():
            raise ValueError(f"incomplete Chronos2 checkpoint: {self.checkpoint}")
        config = json.loads(config_path.read_text())
        if (
            config.get("base_model_name_or_path") != "autogluon/chronos-2-small"
            or config.get("peft_type") != "LORA"
            or config.get("inference_mode") is not True
        ):
            raise ValueError("checkpoint is not the frozen Chronos2-small LoRA contract")
        if batch_series < 5:
            raise ValueError("batch_series must fit one five-channel OHLCV group")
        self.device = str(device)
        self.batch_series = int(batch_series)
        self.fast_group_attention = bool(fast_group_attention)
        self._embedder = package_embedder

    def encode(self, windows: np.ndarray) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != 5:
            raise ValueError("Chronos2 windows must have shape [N,5,T] in OHLCV order")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("Chronos2 windows must be nonempty and finite")
        return next(self.encode_chunks((values,), context_length=values.shape[-1]))

    def encode_chunks(
        self,
        chunks: Iterable[np.ndarray],
        *,
        context_length: int,
    ) -> Iterator[np.ndarray]:
        """Load FFM once and stream bounded chunks through one pipeline."""
        if self.fast_group_attention:
            os.environ["FFM_CHRONOS2_FAST_GROUP_ATTENTION"] = "1"
        embedder = self._embedder or _load_package_embedder()

        expected_lengths: deque[int] = deque()

        def validated():
            for windows in chunks:
                values = np.asarray(windows, dtype=np.float32)
                if values.ndim != 3 or values.shape[1] != 5:
                    raise ValueError(
                        "Chronos2 windows must have shape [N,5,T] in OHLCV order")
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError("Chronos2 windows must be nonempty and finite")
                expected_lengths.append(len(values))
                yield values

        encoded_chunks = embedder(
            validated(),
            checkpoint=self.checkpoint,
            device=self.device,
            batch=self.batch_series,
            pool="reg",
            context_length=int(context_length),
        )
        produced = False
        for encoded in encoded_chunks:
            produced = True
            encoded = np.asarray(encoded, dtype=np.float32)
            if not expected_lengths:
                raise RuntimeError("FFM embedding interface returned excess output")
            expected = expected_lengths.popleft()
            if encoded.ndim != 2 or len(encoded) != expected:
                raise RuntimeError(
                    f"FFM embedding output shape {encoded.shape} does not match "
                    f"{expected} windows")
            if not np.isfinite(encoded).all():
                raise RuntimeError("FFM embedding output is non-finite")
            yield encoded
        if not produced:
            raise RuntimeError("FFM embedding interface returned no output")
        if expected_lengths:
            raise RuntimeError("FFM embedding interface omitted output chunks")
