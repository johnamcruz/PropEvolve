from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from propevolve.encoder import FrozenChronos2Encoder


def test_encoder_delegates_to_installed_ffm_package_interface(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"autogluon/chronos-2-small",'
        '"peft_type":"LORA","inference_mode":true}'
    )
    calls = []

    def package_embedder(chunks, **kwargs):
        calls.append(kwargs)
        for chunk in chunks:
            yield chunk.mean(axis=(1, 2), keepdims=False)[:, None]

    encoder = FrozenChronos2Encoder(
        checkpoint=checkpoint,
        device="cpu",
        batch_series=10,
        fast_group_attention=False,
        package_embedder=package_embedder,
    )
    windows = np.arange(2 * 5 * 4, dtype=np.float32).reshape(2, 5, 4)

    encoded = encoder.encode(windows)

    assert encoded.shape == (2, 1)
    assert calls == [{
        "checkpoint": checkpoint,
        "device": "cpu",
        "batch": 10,
        "pool": "reg",
        "context_length": 4,
    }]


def test_encoder_rejects_wrong_channel_order_or_nonfinite_data(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"autogluon/chronos-2-small",'
        '"peft_type":"LORA","inference_mode":true}'
    )
    encoder = FrozenChronos2Encoder(
        checkpoint=checkpoint,
        device="cpu",
        batch_series=10,
        fast_group_attention=False,
        package_embedder=lambda chunks, **kwargs: chunks,
    )

    with pytest.raises(ValueError, match=r"\[N,5,T\]"):
        encoder.encode(np.zeros((2, 4, 8), np.float32))
    windows = np.zeros((2, 5, 8), np.float32)
    windows[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        encoder.encode(windows)


def test_encoder_streams_multiple_chunks_through_one_ffm_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"autogluon/chronos-2-small",'
        '"peft_type":"LORA","inference_mode":true}'
    )
    loads = []

    def package_embedder(chunks, **kwargs):
        loads.append(kwargs)
        for chunk in chunks:
            yield chunk.mean(axis=(1, 2))[:, None]

    encoder = FrozenChronos2Encoder(
        checkpoint=checkpoint,
        device="cpu",
        batch_series=10,
        fast_group_attention=False,
        package_embedder=package_embedder,
    )
    chunks = (
        np.ones((2, 5, 4), np.float32),
        np.ones((3, 5, 4), np.float32) * 2,
    )

    output = tuple(encoder.encode_chunks(chunks, context_length=4))

    assert len(loads) == 1
    assert [len(value) for value in output] == [2, 3]
